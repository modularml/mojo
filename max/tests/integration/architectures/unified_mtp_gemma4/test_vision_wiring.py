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
"""Regression tests for CENG-684: the gemma4 MTP path must carry vision.

The unified MTP (speculative-decoding) path for Gemma 4 was originally
text-only: the arch used a text-only context/modalities and the fused graph
hardcoded empty image embeddings into the target LM, so image tokens were
ingested by the tokenizer but the vision encoder output never reached the
model -- the served model was effectively blind on image prompts.

These CPU-only structural tests guard the two wiring points that made the
model blind, so a revert to the text-only path fails fast without needing a
full GPU serve (the end-to-end "produces non-blind output" check is the
served smoke on the real checkpoint):

* the arch advertises image/video and uses the multimodal context, so the
  tokenizer injects image tokens and a vision-aware batch processor runs; and
* the fused MTP graph declares the per-device image-embedding + scatter-index
  inputs (``enable_vision``), so the vision encoder output has somewhere to
  bind and reach the target's ``merge_multimodal_embeddings``.
"""

from __future__ import annotations

import dataclasses
import inspect
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.architectures.gemma4.context import Gemma4Context
from max.pipelines.architectures.unified_mtp_gemma4 import (
    unified_mtp_gemma4_arch,
)
from max.pipelines.architectures.unified_mtp_gemma4.batch_processor import (
    UnifiedMTPGemma4BatchProcessor,
)
from max.pipelines.architectures.unified_mtp_gemma4.model import (
    UnifiedMTPGemma4Inputs,
    UnifiedMTPGemma4Model,
)
from max.pipelines.architectures.unified_mtp_gemma4.unified_mtp_gemma4 import (
    UnifiedMTPGemma4,
)
from max.pipelines.context import ImageMetadata
from max.pipelines.context.context import TokenBuffer
from max.pipelines.modeling.types import InputModality

_HIDDEN_SIZE = 128


def _fake_model_for_pack() -> UnifiedMTPGemma4Model:
    """Duck-typed ``self`` exposing only what ``pack_vision_inputs`` reads.

    ``pack_vision_inputs`` touches only ``self.config``; the rest of the model
    (weights, graphs) is irrelevant to the pixel-packing logic under test.
    """
    return cast(
        UnifiedMTPGemma4Model,
        SimpleNamespace(
            config=SimpleNamespace(
                vision_config=SimpleNamespace(pooling_kernel_size=1),
                unquantized_dtype=DType.float32,
            )
        ),
    )


def _two_image_context() -> Gemma4Context:
    """A 2-image context whose first image is already encoded (chunked prefill).

    ``pixel_position_ids`` is the full per-image list; after skipping img0 only
    img1 remains unencoded (``image_idx == 1``). The pipeline's ``select`` would
    return this context paired with its cache-miss set. Here every
    ``next_image`` is a miss.
    """
    # fmt: off
    tokens = np.array(
        [51, 52, 53, 54, 98, 98, 98, 98, 55, 56, 57, 58, 98, 98, 98, 98, 59, 60],
        dtype=np.int64,
    )
    # fmt: on

    def _pixels() -> np.ndarray:
        return np.arange(4 * 3, dtype=np.float32).reshape(4, 3)

    pos0 = np.stack([np.arange(4), np.full(4, 0)], axis=1).astype(np.int32)
    pos1 = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int32)

    ctx = Gemma4Context(
        max_length=64,
        tokens=TokenBuffer(tokens),
        images=[
            ImageMetadata(start_idx=4, end_idx=8, pixel_values=_pixels()),
            ImageMetadata(start_idx=12, end_idx=16, pixel_values=_pixels()),
        ],
        vision_token_ids=[98],
        mm_token_type_ids=np.zeros(len(tokens), dtype=np.int64),
        pixel_position_ids=[pos0, pos1],
    )
    ctx.tokens.skip_processing(8)
    assert ctx.image_idx == 1
    assert len(ctx.next_images) == 1
    return ctx


def test_unified_mtp_gemma4_arch_is_multimodal() -> None:
    """The MTP arch must be multimodal, not served text-only.

    A text-only ``context_type``/``input_modalities`` is exactly what made
    the model blind: without image/video modalities the request path never
    injects image tokens or runs a vision-aware batch processor.
    """
    assert unified_mtp_gemma4_arch.context_type is Gemma4Context
    assert InputModality.IMAGE in unified_mtp_gemma4_arch.input_modalities
    assert InputModality.VIDEO in unified_mtp_gemma4_arch.input_modalities
    assert unified_mtp_gemma4_arch.batching is UnifiedMTPGemma4BatchProcessor
    assert unified_mtp_gemma4_arch.pipeline_model is UnifiedMTPGemma4Model


def test_unified_mtp_gemma4_defaults_to_xgrammar_backend() -> None:
    """The MTP arch defaults to the xgrammar structured-output backend.

    gemma4 structured output compiles through the xgrammar StructuralTag path
    (config-driven bare keys and <|"|> string delimiters), and the speculative
    decoding fixes make that path safe under MTP, so the MTP arch tracks the
    base gemma4 default of xgrammar rather than pinning llguidance. Override
    with --structured-output-backend.
    """
    assert (
        unified_mtp_gemma4_arch.default_structured_output_backend == "xgrammar"
    )


def test_mtp_graph_declares_per_device_vision_inputs() -> None:
    """The fused MTP graph signature must carry vision inputs.

    ``input_types`` must enable the vision inputs so the per-device image
    embeddings + scatter indices immediately follow ``tokens`` and reach the
    target LM. If ``enable_vision`` regresses, the tensor right after
    ``tokens`` is the (uint32, 1-D) row-offsets input and these assertions
    fail.
    """
    n_devices = 2
    devices = [DeviceRef("gpu", i) for i in range(n_devices)]

    # Duck-typed ``self`` exposing only what ``input_types`` reads; avoids
    # constructing the full target + draft modules. Cast for the type checker
    # since we deliberately pass a stand-in to the unbound method.
    fake_self = cast(
        UnifiedMTPGemma4,
        SimpleNamespace(
            config=SimpleNamespace(
                devices=devices,
                text_config=SimpleNamespace(hidden_size=_HIDDEN_SIZE),
            ),
            enable_structured_output=False,
        ),
    )
    kv_params = MagicMock()
    kv_params.flattened_kv_inputs.return_value = []

    input_types = UnifiedMTPGemma4.input_types(fake_self, kv_params)

    # tokens, then per-device image embeddings, then per-device scatter indices.
    image_embeddings = input_types[1 : 1 + n_devices]
    image_indices = input_types[1 + n_devices : 1 + 2 * n_devices]

    for embed_type in image_embeddings:
        assert embed_type.dtype == DType.bfloat16
        assert int(embed_type.shape[-1]) == _HIDDEN_SIZE

    for index_type in image_indices:
        assert index_type.dtype == DType.int32


def test_pack_vision_inputs_aligns_pos_ids_after_partial_encode() -> None:
    """pack_vision_inputs must slice ``[image_idx:]`` so the full per-image
    ``pixel_position_ids`` realigns with ``next_images`` under chunked
    prefill (``image_idx > 0``) and selects img1's grid, not the
    already-encoded img0's.
    """
    ctx = _two_image_context()
    raw = UnifiedMTPGemma4Model.pack_vision_inputs(
        _fake_model_for_pack(), [(ctx, list(ctx.next_images))], [CPU()]
    )
    assert raw is not None
    expected = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int32)
    np.testing.assert_array_equal(
        raw.pixel_position_ids[0].to_numpy(), expected
    )


def test_pack_vision_inputs_raises_on_copied_selection() -> None:
    """A selection holding copies of the context's images raises at the
    source instead of failing later on a counts mismatch."""
    ctx = _two_image_context()
    copies = [dataclasses.replace(img) for img in ctx.next_images]
    with pytest.raises(ValueError, match=r"not present in ctx\.next_images"):
        UnifiedMTPGemma4Model.pack_vision_inputs(
            _fake_model_for_pack(), [(ctx, copies)], [CPU()]
        )


def test_pack_vision_inputs_returns_none_without_miss() -> None:
    """No cache-miss images (all hits / text-only) means nothing to pack."""
    ctx = _two_image_context()
    raw = UnifiedMTPGemma4Model.pack_vision_inputs(
        _fake_model_for_pack(), [(ctx, [])], [CPU()]
    )
    assert raw is None


def test_batch_processor_carries_no_vision_plumbing() -> None:
    """The migrated batch processor owns no cache and emits no image carrier.

    Vision is selected/encoded/assembled by the pipeline's encoder cache
    (``run_vision_encode``); the batch processor builds only text inputs. A
    regression that re-adds a model/processor-owned cache or an ``images``
    carrier to the graph inputs would fail here.
    """
    assert not hasattr(UnifiedMTPGemma4BatchProcessor, "_ve_cache")
    bind_params = inspect.signature(
        UnifiedMTPGemma4BatchProcessor.bind_model_state
    ).parameters
    assert "ve_cache" not in bind_params
    assert "config" in bind_params
    field_names = {f.name for f in dataclasses.fields(UnifiedMTPGemma4Inputs)}
    assert "images" not in field_names
    # The vision-merge inputs are the base ModelInputs fields set by the
    # pipeline's vision seam, not a model-specific carrier.
    assert "combined_embeds" not in field_names
    assert "vision_embeddings" in field_names
    assert "vision_scatter_indices" in field_names
