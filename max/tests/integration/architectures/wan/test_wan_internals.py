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

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest
from max.driver import Buffer, DeviceSpec
from max.dtype import DType
from max.pipelines import PipelineConfig
from max.pipelines.architectures.wan.arch import WanArchConfig
from max.pipelines.architectures.wan.components import (
    vae_wrapper as wan_vae_wrapper,
)
from max.pipelines.architectures.wan.tokenizer import WanTokenizer
from max.pipelines.architectures.wan.weight_adapters import (
    adapt_wan_fp8_weights,
    is_wan_fp8_checkpoint,
)
from max.pipelines.context import PixelContext, TokenBuffer
from max.pipelines.lib.config.model_config import MAXModelConfig
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.pixel_tokenizer import PixelGenerationTokenizer
from max.pipelines.modeling.types import RequestID
from max.pipelines.request.provider_options import (
    ImageProviderOptions,
    VideoProviderOptions,
)


@pytest.fixture(autouse=True)
def _offline_hf_construction() -> Iterator[None]:
    """Keep ``MAXModelConfig`` construction offline (CI runs
    ``HF_HUB_OFFLINE=1``): ``__init__`` eagerly builds the HuggingFace repo
    handles. Real cached repos resolve normally; uncached/placeholder repos
    get a fake path.
    """

    with (
        patch("max.pipelines.lib.config.model_config.validate_hf_repo_access"),
        patch("max.pipelines.weights.hf_utils.validate_hf_repo_access"),
        patch(
            "max.pipelines.weights.hf_utils.generate_local_model_path",
            side_effect=lambda repo_id, revision=None: f"/fake/cache/{repo_id}",
        ),
    ):
        yield


def test_cfg_batched_detection_guards_1d_i2v_tokens() -> None:
    """Regression for MODELS-1537: I2V tokens are 1-D ``[seq_len]``, so a
    naive ``shape[0] > 1`` check mistook the sequence length for a CFG batch
    of 2. That skipped negative-prompt encoding, zeroed the unconditional
    pass, and over-guided the prediction into degenerate (flat-green) video.
    """
    from max.pipelines.architectures.wan.wan_executor import (
        _tokens_are_cfg_batched,
    )

    # T2V: pre-concatenated [cond; uncond] -> 2-D [2, seq_len] is batched.
    assert _tokens_are_cfg_batched((2, 512)) is True
    # I2V / sequential: 1-D [seq_len] must NOT count as batched (the bug).
    assert _tokens_are_cfg_batched((512,)) is False
    # Sequential single-prompt 2-D [1, seq_len] is not batched either.
    assert _tokens_are_cfg_batched((1, 512)) is False


def test_wan_arch_config_initialize_uses_transformer_component() -> None:
    pipeline_config = cast(
        PipelineConfig,
        SimpleNamespace(
            models=ModelManifest(
                {
                    "transformer": MAXModelConfig(
                        model_path="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                        device_specs=[DeviceSpec.cpu()],
                    )
                }
            )
        ),
    )

    # WanArchConfig ignores the received length (metadata-only policy).
    config = WanArchConfig.initialize(
        pipeline_config=pipeline_config, max_seq_len=512
    )

    assert config.pipeline_config is pipeline_config


def test_wan_vae_decode_preserves_5d_video_shape() -> None:
    decoded_video = np.arange(1 * 4 * 2 * 2 * 3, dtype=np.uint8).reshape(
        1, 4, 2, 2, 3
    )

    wrapper = cast(Any, object.__new__(wan_vae_wrapper.VaeWrapper))
    wrapper._denorm_model = cast(
        Any, SimpleNamespace(execute=lambda *args: [object()])
    )
    wrapper._vae = cast(
        Any, SimpleNamespace(decode_5d=lambda _latents: object())
    )
    wrapper._postprocess_model = cast(
        Any, SimpleNamespace(execute=lambda _decoded_video: [decoded_video])
    )
    wrapper._vae_std_buf = cast(Any, object())
    wrapper._vae_mean_buf = cast(Any, object())

    output = wrapper.decode(
        latents=cast(Any, object()),
        num_frames=4,
        height=2,
        width=2,
    )

    assert output.dtype == np.uint8
    assert output.shape == (1, 3, 4, 2, 2)


def test_wan_vae_decode_preserves_single_frame_image_shape() -> None:
    decoded_video = np.arange(1 * 1 * 2 * 2 * 3, dtype=np.uint8).reshape(
        1, 1, 2, 2, 3
    )

    wrapper = cast(Any, object.__new__(wan_vae_wrapper.VaeWrapper))
    wrapper._denorm_model = cast(
        Any, SimpleNamespace(execute=lambda *args: [object()])
    )
    wrapper._vae = cast(
        Any, SimpleNamespace(decode_5d=lambda _latents: object())
    )
    wrapper._postprocess_model = cast(
        Any, SimpleNamespace(execute=lambda _decoded_video: [decoded_video])
    )
    wrapper._vae_std_buf = cast(Any, object())
    wrapper._vae_mean_buf = cast(Any, object())

    output = wrapper.decode(
        latents=cast(Any, object()),
        num_frames=1,
        height=2,
        width=2,
    )

    assert output.dtype == np.uint8
    assert output.shape == (1, 2, 2, 3)


def test_wan_tokenizer_uses_single_frame_video_latents_for_images(
    monkeypatch: Any,
) -> None:
    async def _mock_base_new_context(
        self: PixelGenerationTokenizer,
        request: Any,
        input_image: Any = None,
    ) -> PixelContext:
        return PixelContext(
            tokens=TokenBuffer(np.array([0], dtype=np.int64)),
            request_id=RequestID(),
            latents=np.zeros((1, 16, 104, 60), dtype=np.float32),
            height=832,
            width=480,
            num_inference_steps=4,
            num_images_per_prompt=1,
        )

    monkeypatch.setattr(
        PixelGenerationTokenizer,
        "new_context",
        _mock_base_new_context,
    )

    tokenizer = cast(Any, object.__new__(WanTokenizer))
    tokenizer._scheduler = SimpleNamespace(
        use_flow_sigmas=False,
        order=1,
        retrieve_timesteps_and_sigmas=lambda image_seq_len, num_inference_steps: (
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0], dtype=np.float32),
        ),
        build_step_coefficients=lambda: np.zeros((2, 9), dtype=np.float32),
    )
    tokenizer._vae_scale_factor = 8
    tokenizer._num_channels_latents = 16
    tokenizer._manifest_metadata = {}
    tokenizer._randn_tensor = lambda shape, seed: np.zeros(
        shape, dtype=np.float32
    )

    request = SimpleNamespace(
        body=SimpleNamespace(
            provider_options=SimpleNamespace(video=None, image=None),
            seed=42,
        )
    )

    context = asyncio.run(tokenizer.new_context(request))

    assert context.num_frames == 1
    assert context.latents.shape == (1, 16, 1, 104, 60)


async def _mock_base_new_context(
    self: PixelGenerationTokenizer,
    request: Any,
    input_image: Any = None,
) -> PixelContext:
    return PixelContext(
        tokens=TokenBuffer(np.array([0], dtype=np.int64)),
        request_id=RequestID(),
        latents=np.zeros((1, 16, 104, 60), dtype=np.float32),
        height=832,
        width=480,
        num_inference_steps=4,
        num_images_per_prompt=1,
    )


def _build_flow_shift_tokenizer(captured: list[float | None]) -> Any:
    """WanTokenizer whose scheduler records the flow_shift passed to
    retrieve_timesteps_and_sigmas."""

    def _retrieve(
        image_seq_len: int,
        num_inference_steps: int,
        flow_shift: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        captured.append(flow_shift)
        return (
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0], dtype=np.float32),
        )

    tokenizer = cast(Any, object.__new__(WanTokenizer))
    tokenizer._scheduler = SimpleNamespace(
        use_flow_sigmas=True,
        flow_shift=3.0,  # model-config default
        order=1,
        retrieve_timesteps_and_sigmas=_retrieve,
        build_step_coefficients=lambda: np.zeros((2, 9), dtype=np.float32),
    )
    tokenizer._vae_scale_factor = 8
    tokenizer._num_channels_latents = 16
    tokenizer._manifest_metadata = {}
    tokenizer._randn_tensor = lambda shape, seed: np.zeros(
        shape, dtype=np.float32
    )
    return tokenizer


class _FakeWeightData:
    """Minimal stand-in for a loaded WeightData: ``.dtype`` + ``.to_buffer()``."""

    def __init__(self, buffer: Buffer) -> None:
        self._buffer = buffer
        self.dtype = buffer.dtype

    def to_buffer(self) -> Buffer:
        return self._buffer


class _FakeEntry:
    def __init__(self, wd: _FakeWeightData) -> None:
        self._wd = wd

    def data(self) -> _FakeWeightData:
        return self._wd


class _FakeWeights:
    """Stand-in for ``max.graph.weights.Weights`` over an in-memory dict."""

    def __init__(self, tensors: dict[str, Buffer]) -> None:
        self._entries = {
            k: _FakeEntry(_FakeWeightData(v)) for k, v in tensors.items()
        }

    def items(self) -> Any:
        return self._entries.items()


def _fp8_weight(out_dim: int, in_dim: int) -> Buffer:
    """A tiny float8_e4m3fn weight (built by viewing a uint8 buffer)."""
    raw = np.arange(out_dim * in_dim, dtype=np.uint8).reshape(out_dim, in_dim)
    return Buffer.from_numpy(raw).view(DType.float8_e4m3fn, [out_dim, in_dim])


def _f16(*shape: int) -> Buffer:
    return Buffer.from_numpy(np.ones(shape, dtype=np.float16))


def _f32_scalar(value: float) -> Buffer:
    return Buffer.from_numpy(np.asarray(value, dtype=np.float32))


def _fake_fp8_linear(
    tensors: dict[str, Buffer],
    prefix: str,
    out_dim: int,
    in_dim: int,
    scale_input: float,
    scale_weight: float,
) -> None:
    tensors[f"{prefix}.weight"] = _fp8_weight(out_dim, in_dim)
    tensors[f"{prefix}.bias"] = _f16(out_dim)
    tensors[f"{prefix}.scale_input"] = _f32_scalar(scale_input)
    tensors[f"{prefix}.scale_weight"] = _f32_scalar(scale_weight)


def test_is_wan_fp8_checkpoint_detects_marker() -> None:
    with_marker = _FakeWeights(
        {"scaled_fp8": _fp8_weight(1, 1), "patch_embedding.bias": _f16(8)}
    )
    without_marker = _FakeWeights({"patch_embedding.bias": _f16(8)})

    assert is_wan_fp8_checkpoint(cast(Any, with_marker))
    assert not is_wan_fp8_checkpoint(cast(Any, without_marker))


def test_adapt_wan_fp8_weights_maps_native_names_and_scales() -> None:
    dim, text_dim, ffn_dim = 8, 6, 16
    t: dict[str, Buffer] = {}
    # Marker + patch embedding (Conv3d FCDHW) + post head.
    t["scaled_fp8"] = _fp8_weight(1, 1)
    t["patch_embedding.weight"] = Buffer.from_numpy(
        np.ones((dim, 4, 1, 2, 2), dtype=np.float16)
    )
    t["patch_embedding.bias"] = _f16(dim)
    t["head.modulation"] = _f16(1, 2, dim)
    _fake_fp8_linear(t, "head.head", 4, dim, 0.5, 2.0)
    # Top-level condition-embedder linears.
    _fake_fp8_linear(t, "time_embedding.0", dim, 4, 0.5, 2.0)
    _fake_fp8_linear(t, "time_embedding.2", dim, dim, 0.5, 2.0)
    _fake_fp8_linear(t, "time_projection.1", dim * 6, dim, 0.5, 2.0)
    _fake_fp8_linear(t, "text_embedding.0", dim, text_dim, 0.5, 2.0)
    _fake_fp8_linear(t, "text_embedding.2", dim, dim, 0.5, 2.0)
    # One block.
    for sub, (o, i) in {
        "self_attn.q": (dim, dim),
        "self_attn.k": (dim, dim),
        "self_attn.v": (dim, dim),
        "self_attn.o": (dim, dim),
        "cross_attn.q": (dim, dim),
        "cross_attn.k": (dim, text_dim),
        "cross_attn.v": (dim, text_dim),
        "cross_attn.o": (dim, dim),
        "ffn.0": (ffn_dim, dim),
        "ffn.2": (dim, ffn_dim),
    }.items():
        _fake_fp8_linear(t, f"blocks.0.{sub}", o, i, 4.0, 2.0)
    t["blocks.0.self_attn.norm_q.weight"] = _f16(dim)
    t["blocks.0.self_attn.norm_k.weight"] = _f16(dim)
    t["blocks.0.cross_attn.norm_q.weight"] = _f16(dim)
    t["blocks.0.cross_attn.norm_k.weight"] = _f16(dim)
    t["blocks.0.norm3.weight"] = _f16(dim)
    t["blocks.0.norm3.bias"] = _f16(dim)
    t["blocks.0.modulation"] = _f16(1, 6, dim)

    out = adapt_wan_fp8_weights(cast(Any, _FakeWeights(t)))

    # Marker dropped; native cross_attn k/v map to separate to_k/to_v.
    assert "scaled_fp8" not in out
    assert out["blocks.0.attn1.to_q.weight"].dtype == DType.float8_e4m3fn
    assert out["blocks.0.attn1.to_q.bias"].dtype == DType.bfloat16
    assert "blocks.0.attn2.to_k.weight" in out
    assert "blocks.0.attn2.to_v.weight" in out
    assert "blocks.0.attn2.to_kv.weight" not in out
    # Native norm3 -> MAX norm2 (the affine cross-attn LayerNorm).
    assert "blocks.0.norm2.weight" in out
    assert "blocks.0.norm2.bias" in out
    # Modulation -> scale_shift_table; head.head -> proj_out; head.modulation
    # -> post scale_shift_table.
    assert "blocks.0.scale_shift_table" in out
    assert out["proj_out.weight"].dtype == DType.float8_e4m3fn
    assert "scale_shift_table" in out

    # Per-tensor scales are scalars; input scale is inverted (1/scale_input),
    # weight scale passes through.
    ws = out["blocks.0.attn1.to_q.weight_scale"]
    is_ = out["blocks.0.attn1.to_q.input_scale"]
    assert tuple(np.asarray(ws).shape) == ()
    assert tuple(np.asarray(is_).shape) == ()
    assert np.isclose(float(np.asarray(ws)), 2.0)
    assert np.isclose(float(np.asarray(is_)), 1.0 / 4.0)


def test_wan_tokenizer_honors_request_flow_shift_override(
    monkeypatch: Any,
) -> None:
    """video_options.flow_shift is passed through to the scheduler."""
    monkeypatch.setattr(
        PixelGenerationTokenizer, "new_context", _mock_base_new_context
    )

    captured: list[float | None] = []
    tokenizer = _build_flow_shift_tokenizer(captured)

    request = SimpleNamespace(
        body=SimpleNamespace(
            provider_options=SimpleNamespace(
                video=VideoProviderOptions(flow_shift=5.5), image=None
            ),
            seed=42,
        )
    )
    asyncio.run(tokenizer.new_context(request))

    assert captured == [5.5]
    # The shared scheduler default is left untouched.
    assert tokenizer._scheduler.flow_shift == 3.0


def test_wan_tokenizer_honors_image_block_flow_shift(
    monkeypatch: Any,
) -> None:
    """flow_shift on the image block is honored when no video block is set."""
    monkeypatch.setattr(
        PixelGenerationTokenizer, "new_context", _mock_base_new_context
    )

    captured: list[float | None] = []
    tokenizer = _build_flow_shift_tokenizer(captured)

    request = SimpleNamespace(
        body=SimpleNamespace(
            provider_options=SimpleNamespace(
                video=None, image=ImageProviderOptions(flow_shift=6.0)
            ),
            seed=42,
        )
    )
    asyncio.run(tokenizer.new_context(request))

    assert captured == [6.0]


def test_wan_tokenizer_flow_shift_override_does_not_leak_across_requests(
    monkeypatch: Any,
) -> None:
    """A per-request override must not bleed into a later request that omits
    the field — the second request resolves the model-config default."""
    monkeypatch.setattr(
        PixelGenerationTokenizer, "new_context", _mock_base_new_context
    )

    captured: list[float | None] = []
    tokenizer = _build_flow_shift_tokenizer(captured)

    req_override = SimpleNamespace(
        body=SimpleNamespace(
            provider_options=SimpleNamespace(
                video=VideoProviderOptions(flow_shift=7.0), image=None
            ),
            seed=42,
        )
    )
    asyncio.run(tokenizer.new_context(req_override))

    req_no_override = SimpleNamespace(
        body=SimpleNamespace(
            provider_options=SimpleNamespace(video=None, image=None),
            seed=42,
        )
    )
    asyncio.run(tokenizer.new_context(req_no_override))

    # Request 1 passes 7.0; request 2 falls back to the config default (3.0),
    # not the 7.0 from request 1.
    assert captured == [7.0, 3.0]
