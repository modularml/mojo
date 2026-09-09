# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Smoke tests for the text-to-video benchmark task plumbing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from max.benchmark.benchmark_shared.config import (
    PIXEL_GENERATION_TASKS,
    get_pixel_gen_endpoint,
)
from max.benchmark.benchmark_shared.datasets.pixel_synthetic import (
    SyntheticPixelBenchmarkDataset,
)
from max.benchmark.benchmark_shared.datasets.types import (
    PixelGenerationImageOptions,
    PixelGenerationSampledRequest,
)
from max.benchmark.benchmark_shared.request import (
    PixelGenerationRequestFuncInput,
    _add_input_reference,
    _build_pixel_generation_payload,
    _build_sglang_video_form,
)


def test_text_to_video_is_a_pixel_generation_task() -> None:
    assert "text-to-video" in PIXEL_GENERATION_TASKS


def test_synthetic_pixel_dataset_propagates_num_frames() -> None:
    dataset = SyntheticPixelBenchmarkDataset()
    dataset.dataset_name = "synthetic-pixel"
    samples = dataset.sample_requests(
        num_requests=2,
        tokenizer=None,
        benchmark_task="text-to-video",
        image_width=480,
        image_height=480,
        image_steps=8,
        num_frames=17,
    )
    assert len(samples.requests) == 2
    for request in samples.requests:
        assert isinstance(request, PixelGenerationSampledRequest)
        # text-to-video has no input image
        assert request.input_image_paths == []
        assert request.image_options is not None
        assert request.image_options.num_frames == 17
        assert request.image_options.width == 480


def test_pixel_generation_payload_routes_video_to_video_provider_options() -> (
    None
):
    image_options = PixelGenerationImageOptions(
        width=480, height=480, steps=8, num_frames=17
    )
    func_input = PixelGenerationRequestFuncInput(
        model="some/model",
        session_id=None,
        prompt="A cat dancing",
        input_image_paths=None,
        api_url="http://localhost:8000/v1/responses",
        image_options=image_options,
    )
    payload = _build_pixel_generation_payload(func_input)
    assert "provider_options" in payload
    assert "video" in payload["provider_options"]
    assert "image" not in payload["provider_options"]
    video = payload["provider_options"]["video"]
    assert video["num_frames"] == 17
    assert video["width"] == 480
    assert video["height"] == 480
    assert video["steps"] == 8


def test_pixel_generation_payload_routes_image_when_no_num_frames() -> None:
    image_options = PixelGenerationImageOptions(
        width=1024, height=1024, steps=28
    )
    func_input = PixelGenerationRequestFuncInput(
        model="some/model",
        session_id=None,
        prompt="A cat",
        input_image_paths=None,
        api_url="http://localhost:8000/v1/responses",
        image_options=image_options,
    )
    payload = _build_pixel_generation_payload(func_input)
    assert "provider_options" in payload
    assert "image" in payload["provider_options"]
    assert "video" not in payload["provider_options"]
    image = payload["provider_options"]["image"]
    assert "num_frames" not in image
    assert image["width"] == 1024


def test_get_pixel_gen_endpoint_vllm_video() -> None:
    assert get_pixel_gen_endpoint("vllm", "text-to-video") == "/v1/videos/sync"


def test_get_pixel_gen_endpoint_vllm_image() -> None:
    assert (
        get_pixel_gen_endpoint("vllm", "text-to-image")
        == "/v1/chat/completions"
    )


def test_get_pixel_gen_endpoint_modular_video() -> None:
    assert get_pixel_gen_endpoint("modular", "text-to-video") == "/v1/responses"


# ---------------------------------------------------------------------------
# image-to-video: routes to the same video endpoints as text-to-video, and
# uploads the conditioning image as a multipart `input_reference` file on the
# vllm-omni / sglang backends.
# ---------------------------------------------------------------------------


def test_image_to_video_is_a_pixel_generation_task() -> None:
    assert "image-to-video" in PIXEL_GENERATION_TASKS


def test_get_pixel_gen_endpoint_image_to_video() -> None:
    # i2v diverts to the video endpoints just like t2v.
    assert get_pixel_gen_endpoint("vllm", "image-to-video") == "/v1/videos/sync"
    assert get_pixel_gen_endpoint("sglang", "image-to-video") == "/v1/videos"
    assert (
        get_pixel_gen_endpoint("modular", "image-to-video") == "/v1/responses"
    )


def _form_field_map(
    form: aiohttp.FormData,
) -> dict[str, tuple[Any, Any]]:
    """Map a FormData's field name to its (type_options, value).

    Reaches into aiohttp's untyped ``_fields`` internals, hence ``Any``.
    """
    return {
        opts["name"]: (opts, value) for opts, _headers, value in form._fields
    }


def test_add_input_reference_attaches_image_file(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"\x89PNG fake bytes")
    form = aiohttp.FormData()
    _add_input_reference(form, [str(image_path)])

    fields = _form_field_map(form)
    assert "input_reference" in fields
    opts, value = fields["input_reference"]
    assert opts["filename"] == "frame.png"
    assert value == b"\x89PNG fake bytes"


def test_add_input_reference_noop_without_image() -> None:
    form = aiohttp.FormData()
    _add_input_reference(form, None)
    _add_input_reference(form, [])
    assert form._fields == []


def test_add_input_reference_raises_on_missing_file() -> None:
    form = aiohttp.FormData()
    with pytest.raises(FileNotFoundError):
        _add_input_reference(form, ["/does/not/exist.png"])


def test_build_sglang_video_form_for_image_to_video(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"img")
    func_input = PixelGenerationRequestFuncInput(
        model="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        session_id=None,
        prompt="pan across the scene",
        input_image_paths=[str(image_path)],
        api_url="http://localhost:8000/v1/videos",
        image_options=PixelGenerationImageOptions(
            width=832,
            height=480,
            steps=30,
            guidance_scale=4.0,
            num_frames=81,
        ),
    )
    form = _build_sglang_video_form(func_input)
    fields = _form_field_map(form)

    # size is a single WxH form field; num_frames is top-level.
    assert fields["size"][1] == "832x480"
    assert fields["num_frames"][1] == "81"
    # sampling knobs ride along JSON-encoded under extra_body.
    extra_body = json.loads(fields["extra_body"][1])
    assert extra_body["num_inference_steps"] == 30
    assert extra_body["guidance_scale"] == 4.0
    # the conditioning image is attached as input_reference.
    assert "input_reference" in fields
