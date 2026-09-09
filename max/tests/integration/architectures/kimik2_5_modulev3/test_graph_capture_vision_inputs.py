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
"""Regression tests for the KimiK2.5 (ModuleV3) vision-cache ownership seam.

``KimiK2_5Model`` used to construct and own its own ``VisionEncoderCache``
internally (bound into the batch processor via ``bind_vision_cache``), unlike
every other vision-capable architecture, where the pipeline owns the cache and
drives it through the ``SupportsVisionEncoding`` protocol. That split required
threading a ``vision_cache_plan`` into this one model's constructor.

The model now implements ``SupportsVisionEncoding`` (``pack_vision_inputs`` /
``vision_execute`` / ``empty_vision_embeddings``) like its non-ModuleV3
sibling, so the pipeline's vision seam (``VisionEncoderCache.
finalize_vision_inputs``) sets the base ``vision_embeddings`` /
``vision_scatter_indices`` fields directly -- no per-model plan threading.
These CPU-only structural tests pin that seam without weights or GPUs.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
from max.driver import CPU, Buffer
from max.dtype import DType
from max.pipelines.architectures.kimik2_5_modulev3.batch_processor import (
    KimiK2_5BatchProcessor,
)
from max.pipelines.architectures.kimik2_5_modulev3.model import (
    KimiK2_5Model,
    KimiK2_5ModelInputs,
)
from max.pipelines.context import ImageMetadata
from max.pipelines.lib.vision_encoder_cache import (
    SupportsVisionEncoding,
    VisionEncoderCache,
)

_N_DEVICES = 4
_HIDDEN_SIZE = 64


def _bare_batch_inputs(n_devices: int) -> KimiK2_5ModelInputs:
    """Mirrors what ``KimiK2_5BatchProcessor.prepare_initial_token_inputs``
    returns: text inputs only, with the vision-merge lists left at their empty
    dataclass defaults."""
    return KimiK2_5ModelInputs(
        tokens=Buffer.zeros((1,), dtype=DType.int64),
        input_row_offsets=Buffer.zeros((2,), dtype=DType.uint32),
        return_n_logits=Buffer.zeros((1,), dtype=DType.int64),
        kv_cache_inputs=None,
        batch_context_lengths=[
            Buffer.zeros((1,), dtype=DType.int32) for _ in range(n_devices)
        ],
    )


def _fake_pipeline_model(n_devices: int) -> Any:
    """A real ``KimiK2_5Model`` instance without ``__init__``, exposing only
    the attributes the prepare/finalize path reads."""
    model = cast(Any, object.__new__(KimiK2_5Model))
    model.devices = [CPU() for _ in range(n_devices)]
    model.model_config = SimpleNamespace(
        llm_config=SimpleNamespace(hidden_size=_HIDDEN_SIZE)
    )

    batch_processor = MagicMock()
    batch_processor.prepare_initial_token_inputs.return_value = (
        _bare_batch_inputs(n_devices)
    )
    model._batch_processor = batch_processor
    return model


def test_model_satisfies_supports_vision_encoding() -> None:
    """``KimiK2_5Model`` must implement the standard vision-encoding protocol.

    Otherwise the pipeline's ``isinstance(pipeline_model,
    SupportsVisionEncoding)`` gate never fires and the model falls back to
    re-encoding every image on every step (no cache at all).
    """
    model = _fake_pipeline_model(_N_DEVICES)
    assert isinstance(model, SupportsVisionEncoding)


def test_finalized_warmup_inputs_populate_base_vision_fields() -> None:
    """Prepared-then-finalized inputs must carry the base ``vision_embeddings``
    / ``vision_scatter_indices`` fields that ``.buffers`` reads.

    Regression test for the vision-cache-ownership refactor: before it,
    ``KimiK2_5ModelInputs.buffers`` read its own ``image_embeddings`` /
    ``image_token_indices`` fields, which the pipeline's vision seam never
    touched -- only the model's own (now-removed) internal cache wiring did.
    """
    model = _fake_pipeline_model(_N_DEVICES)
    inputs = model.prepare_initial_token_inputs([[]])
    # The model no longer owns the vision-merge inputs; prepare leaves them
    # empty for the pipeline's vision seam to set.
    assert inputs.vision_embeddings == []
    assert inputs.vision_scatter_indices == []

    encoder_cache: VisionEncoderCache[Any] = VisionEncoderCache(
        devices=model.devices
    )
    encoder_cache.finalize_vision_inputs(model, inputs, model.devices, None)

    embeddings = inputs.vision_embeddings
    indices = inputs.vision_scatter_indices
    assert len(embeddings) == _N_DEVICES
    assert len(indices) == _N_DEVICES
    for embed in embeddings:
        assert tuple(embed.shape) == (0, _HIDDEN_SIZE)
    for index in indices:
        assert tuple(index.shape) == (0,)
        assert index.dtype == DType.int32

    # buffers ABI: (tokens, return_n_logits, input_row_offsets,
    # *vision_embeddings, *vision_scatter_indices, *batch_context_lengths).
    buffers = inputs.buffers
    assert buffers[0] is inputs.tokens
    assert buffers[1] is inputs.return_n_logits
    assert buffers[2] is inputs.input_row_offsets
    assert all(
        packed is expected
        for packed, expected in zip(
            buffers[3 : 3 + _N_DEVICES], embeddings, strict=True
        )
    )
    assert all(
        packed is expected
        for packed, expected in zip(
            buffers[3 + _N_DEVICES : 3 + 2 * _N_DEVICES], indices, strict=True
        )
    )


def test_collect_uncached_image_inputs_matches_by_span() -> None:
    """Miss images are collected by span position, not object identity.

    Mirrors the non-ModuleV3 sibling's MXSERV-330 regression test: a
    selection whose miss list holds reconstructed ``ImageMetadata`` instances
    for the same logical image must still collect a per-image entry.
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
