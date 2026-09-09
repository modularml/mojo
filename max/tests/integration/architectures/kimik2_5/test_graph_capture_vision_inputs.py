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
"""Regression tests for the KimiK2.5 graph-capture warmup input ABI.

The compiled Kimi language graph unconditionally declares ``2 * n_devices``
vision-merge inputs (per-device ``image_embeddings`` + ``image_token_indices``)
right after ``tokens``. Those buffers used to be populated only inside
``KimiK2_5Model.execute()``, but device graph-capture warmup and replay pack
``model_inputs.buffers`` straight from prepared inputs without calling
``execute()`` -- so a DP=8 decode worker crashed at the first capture shape
with ``ValueError: Number of inputs (99) does not match expected number
(115)`` (the deficit is exactly ``2 * data_parallel_degree``).

The vision-merge inputs are set by the pipeline's vision seam
(``VisionEncoderCache.finalize_vision_inputs``) on every prepared batch --
including the warmup path, which passes ``vision_result=None``. These
CPU-only structural tests pin the two sides of that ABI without weights or
GPUs:

* prepared-then-finalized inputs carry the per-device vision-merge buffers,
  so ``.buffers`` arity matches the language graph's ``input_types`` arity;
  and
* the Eagle spec-decode subclasses expose the same finalized fields through
  their own ``*Inputs`` packing.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.architectures.kimik2_5.batch_processor import (
    KimiK2_5BatchProcessor,
)
from max.pipelines.architectures.kimik2_5.layers.language_model import (
    KimiK2_5MoEDecoder,
)
from max.pipelines.architectures.kimik2_5.model import (
    KimiK2_5Model,
    KimiK2_5ModelInputs,
)
from max.pipelines.architectures.kimik2_5.unified_eagle_mha_pipeline_model import (
    Eagle3MHAKimiK25Model,
)
from max.pipelines.architectures.kimik2_5.unified_eagle_pipeline_model import (
    Eagle3KimiK25Model,
)
from max.pipelines.context import ImageMetadata
from max.pipelines.lib.vision_encoder_cache import VisionEncoderCache

_N_DEVICES = 4
_HIDDEN_SIZE = 64


def _bare_batch_inputs(n_devices: int) -> KimiK2_5ModelInputs:
    """Mirrors what ``KimiK2_5BatchProcessor.prepare_initial_token_inputs``
    returns: text inputs only, with the vision-merge lists left at their empty
    dataclass defaults."""
    return KimiK2_5ModelInputs(
        tokens=Buffer.zeros((1,), dtype=DType.int64),
        input_row_offsets=Buffer.zeros((2,), dtype=DType.uint32),
        host_input_row_offsets=Buffer.zeros((2,), dtype=DType.uint32),
        signal_buffers=[
            Buffer.zeros((1,), dtype=DType.uint8) for _ in range(n_devices)
        ],
        kv_cache_inputs=None,
        return_n_logits=Buffer.zeros((1,), dtype=DType.int64),
        data_parallel_splits=Buffer.zeros((n_devices + 1,), dtype=DType.int64),
        batch_context_lengths=[
            Buffer.zeros((1,), dtype=DType.int32) for _ in range(n_devices)
        ],
    )


def _fake_pipeline_model(cls: type[KimiK2_5Model], n_devices: int) -> Any:
    """A real ``cls`` instance without ``__init__``, exposing only the
    attributes the prepare/finalize path reads."""
    model = cast(Any, object.__new__(cls))
    model.devices = [CPU() for _ in range(n_devices)]

    pipeline_config = MagicMock()
    pipeline_config.model.huggingface_config.text_config.hidden_size = (
        _HIDDEN_SIZE
    )
    pipeline_config.needs_bitmask_constraints = False
    model.pipeline_config = pipeline_config

    model._eplb_stats_accumulator = None
    model.ep_comm_initializer = None
    model._eplb_log2phy_buffers = []
    model._eplb_logcnt_buffers = []

    batch_processor = MagicMock()
    batch_processor.prepare_initial_token_inputs.return_value = (
        _bare_batch_inputs(n_devices)
    )
    model._batch_processor = batch_processor
    return model


def _language_graph_input_types(n_devices: int) -> tuple[Any, ...]:
    """The language graph's declared input arity, from the real
    ``KimiK2_5MoEDecoder.input_types`` with KV inputs zeroed out (matching the
    ``kv_cache_inputs=None`` model inputs on the buffers side)."""
    decoder = cast(Any, object.__new__(KimiK2_5MoEDecoder))
    decoder.config = SimpleNamespace(
        devices=[DeviceRef("gpu", i) for i in range(n_devices)],
        hidden_size=_HIDDEN_SIZE,
        data_parallel_degree=n_devices,
        eplb_profile_enabled=False,
    )
    decoder.ep_manager = None
    kv_params = MagicMock()
    kv_params.flattened_kv_inputs.return_value = []
    return decoder.input_types(kv_params)


def test_finalized_warmup_inputs_match_language_graph_arity() -> None:
    """Prepared-then-finalized inputs must emit every buffer the language
    graph declares: graph-capture warmup and replay pack ``.buffers`` without
    going through ``execute()``, after the pipeline seam finalizes with
    ``vision_result=None`` (exactly what ``_warmup_model_inputs`` does).

    Without the finalize the buffers tuple is short by ``2 * n_devices`` (the
    99-vs-115 warmup crash at DP=8, scaled down).
    """
    model = _fake_pipeline_model(KimiK2_5Model, _N_DEVICES)
    inputs = model.prepare_initial_token_inputs([[]])
    # The model no longer owns the vision-merge inputs; prepare leaves them
    # empty for the pipeline's vision seam to set.
    assert inputs.vision_embeddings == []
    assert inputs.vision_scatter_indices == []

    encoder_cache: VisionEncoderCache[Any] = VisionEncoderCache(
        devices=model.devices
    )
    encoder_cache.finalize_vision_inputs(model, inputs, model.devices, None)
    buffers = inputs.buffers

    graph_input_types = _language_graph_input_types(_N_DEVICES)
    assert len(buffers) == len(graph_input_types)

    # The vision-merge buffers sit right after ``tokens``, matching
    # ``_build_language_graph``'s destructuring order, with the decode-step
    # empty shapes ([0, hidden] embeddings, [0] int32 scatter indices).
    embeddings = inputs.vision_embeddings
    indices = inputs.vision_scatter_indices
    assert len(embeddings) == _N_DEVICES
    assert len(indices) == _N_DEVICES
    assert all(
        packed is expected
        for packed, expected in zip(
            buffers[1 : 1 + _N_DEVICES], embeddings, strict=True
        )
    )
    assert all(
        packed is expected
        for packed, expected in zip(
            buffers[1 + _N_DEVICES : 1 + 2 * _N_DEVICES], indices, strict=True
        )
    )
    for embed in embeddings:
        assert tuple(embed.shape) == (0, _HIDDEN_SIZE)
    for index in indices:
        assert tuple(index.shape) == (0,)
        assert index.dtype == DType.int32


@pytest.mark.parametrize(
    "cls", [Eagle3KimiK25Model, Eagle3MHAKimiK25Model], ids=["mla", "mha"]
)
def test_eagle_inputs_pack_finalized_vision_merge_buffers(
    cls: type[KimiK2_5Model],
) -> None:
    """The Eagle subclasses' ``*Inputs`` must splat the seam-set vision-merge
    fields for the unified spec-decode graph, right after ``tokens``."""
    model = _fake_pipeline_model(cls, _N_DEVICES)
    inputs = model.prepare_initial_token_inputs([[]])
    encoder_cache: VisionEncoderCache[Any] = VisionEncoderCache(
        devices=model.devices
    )
    encoder_cache.finalize_vision_inputs(model, inputs, model.devices, None)
    assert len(inputs.vision_embeddings) == _N_DEVICES
    assert len(inputs.vision_scatter_indices) == _N_DEVICES

    # Fill the mandatory spec-decode tail so ``.buffers`` can be packed, then
    # check the vision-merge buffers land right after ``tokens``.
    dummy = Buffer.zeros((1,), dtype=DType.int64)
    inputs.draft_tokens = dummy
    inputs.seed = dummy
    inputs.temperature = dummy
    inputs.top_k = dummy
    inputs.max_k = dummy
    inputs.top_p = dummy
    inputs.min_top_p = dummy
    inputs.in_thinking_phase = dummy
    buffers = inputs.buffers
    assert all(
        packed is expected
        for packed, expected in zip(
            buffers[1 : 1 + _N_DEVICES],
            inputs.vision_embeddings,
            strict=True,
        )
    )
    assert all(
        packed is expected
        for packed, expected in zip(
            buffers[1 + _N_DEVICES : 1 + 2 * _N_DEVICES],
            inputs.vision_scatter_indices,
            strict=True,
        )
    )


def test_collect_uncached_image_inputs_matches_by_span() -> None:
    """Miss images are collected by span position, not object identity.

    Regression test for MXSERV-330: a selection whose miss list holds
    reconstructed ``ImageMetadata`` instances for the same logical image
    collected zero per-image entries under identity matching and tripped
    the non-empty assertion in ``encode_uncached_chunked``.
    """
    processor = cast(Any, object.__new__(KimiK2_5BatchProcessor))
    pixels = np.zeros((4, 8), dtype=np.float32)
    stored = ImageMetadata(
        start_idx=2, end_idx=4, pixel_values=pixels, image_hash=0xB
    )
    reconstructed = ImageMetadata(
        start_idx=2, end_idx=4, pixel_values=pixels, image_hash=0xB
    )
    ctx = SimpleNamespace(
        images=[stored],
        grid_thws=[(1, 2, 2)],
        position_ids=np.arange(4, dtype=np.int64),
    )

    per_image = processor._collect_uncached_image_inputs(
        [(cast(Any, ctx), [reconstructed])]
    )

    assert len(per_image) == 1
    assert per_image[0]["pixel_values"] is pixels
    np.testing.assert_array_equal(per_image[0]["grid_thw"], [1, 2, 2])
    np.testing.assert_array_equal(per_image[0]["position_ids"], [0, 1, 2, 3])
