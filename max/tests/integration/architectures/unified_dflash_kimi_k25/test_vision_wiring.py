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
"""Regression tests for MXSERV-334: the Kimi DFlash path must carry vision.

The unified DFlash pipeline advertised image input but was text-only inside:
``load_model`` stubbed the vision model with a ``MagicMock`` and the fused
graph declared no vision inputs, so image tokens reached the target as bare
placeholder embeddings -- the served model was effectively blind on image
prompts (the same failure Gemma 4 MTP had).

These CPU-only structural tests pin the two wiring points, so a revert to the
text-only path fails fast without a full GPU serve (the end-to-end "produces
non-blind output" check is the served smoke on the real checkpoint):

* the fused graph declares the per-device image-embedding + scatter-index
  inputs, so the vision encoder output has somewhere to bind and reach
  ``merge_multimodal_embeddings``; and
* ``UnifiedDflashKimiK25Inputs`` packs the seam-set vision-merge buffers in
  that same position, so the buffer tuple matches the graph signature.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.architectures.kimik2_5.context import (
    KimiK2_5TextAndVisionContext,
)
from max.pipelines.architectures.unified_dflash_kimi_k25 import (
    unified_dflash_kimi_k25_arch,
)
from max.pipelines.architectures.unified_dflash_kimi_k25.model import (
    UnifiedDflashKimiK25Inputs,
    UnifiedDflashKimiK25Model,
)
from max.pipelines.architectures.unified_dflash_kimi_k25.unified_dflash_kimi_k25 import (
    UnifiedDflashKimiK25,
)
from max.pipelines.lib.vision_encoder_cache import VisionEncoderCache
from max.pipelines.modeling.types import InputModality

_N_DEVICES = 4
_HIDDEN_SIZE = 64


def _bare_dflash_inputs(n_devices: int) -> UnifiedDflashKimiK25Inputs:
    """What ``UnifiedDflashKimiK25BatchProcessor`` returns: text inputs only,
    with the vision-merge lists left at their empty dataclass defaults for the
    pipeline's vision seam to set."""
    return UnifiedDflashKimiK25Inputs(
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


def _fake_pipeline_model(n_devices: int) -> Any:
    """A real model instance without ``__init__``, exposing only what
    ``empty_vision_embeddings`` reads."""
    model = cast(Any, object.__new__(UnifiedDflashKimiK25Model))
    model.devices = [CPU() for _ in range(n_devices)]
    pipeline_config = MagicMock()
    pipeline_config.model.huggingface_config.text_config.hidden_size = (
        _HIDDEN_SIZE
    )
    model.pipeline_config = pipeline_config
    return model


def _graph_input_types(
    n_devices: int, *, enable_structured_output: bool = False
) -> tuple[Any, ...]:
    """The fused graph's declared input arity, from the real ``input_types``
    with KV inputs zeroed out (matching the ``kv_cache_inputs=None`` model
    inputs on the buffers side).

    ``input_types`` reads only the target's device list, DP degree, hidden size,
    EP manager and structured-output flag, so a duck-typed stand-in avoids
    building the target + draft modules.
    """
    fake_self = cast(
        UnifiedDflashKimiK25,
        SimpleNamespace(
            config=SimpleNamespace(
                target=SimpleNamespace(
                    devices=[DeviceRef("gpu", i) for i in range(n_devices)],
                    data_parallel_degree=1,
                    hidden_size=_HIDDEN_SIZE,
                )
            ),
            target=SimpleNamespace(ep_manager=None),
            enable_structured_output=enable_structured_output,
        ),
    )
    kv_params = MagicMock()
    kv_params.flattened_kv_inputs.return_value = []
    return UnifiedDflashKimiK25.input_types(fake_self, kv_params)


def _fill_spec_decode_tail(inputs: UnifiedDflashKimiK25Inputs) -> None:
    """The unconditional tail fields, as the batch processor sets them."""
    dummy = Buffer.zeros((1,), dtype=DType.int64)
    inputs.draft_tokens = dummy
    inputs.seed = dummy
    inputs.temperature = dummy
    inputs.top_k = dummy
    inputs.max_k = dummy
    inputs.top_p = dummy
    inputs.min_top_p = dummy


def test_arch_is_multimodal() -> None:
    """The DFlash arch must advertise image and use the multimodal context.

    A text-only ``context_type`` / ``input_modalities`` would stop the request
    path from injecting image tokens at all.
    """
    assert (
        unified_dflash_kimi_k25_arch.context_type
        is KimiK2_5TextAndVisionContext
    )
    assert InputModality.IMAGE in unified_dflash_kimi_k25_arch.input_modalities
    assert unified_dflash_kimi_k25_arch.pipeline_model is (
        UnifiedDflashKimiK25Model
    )


def test_graph_declares_per_device_vision_inputs() -> None:
    """The fused DFlash graph signature must carry vision inputs.

    The per-device image embeddings + scatter indices must immediately follow
    ``tokens``. If ``enable_vision`` regresses, the tensor right after
    ``tokens`` is the (uint32, 1-D) row-offsets input and these assertions
    fail.
    """
    input_types = _graph_input_types(_N_DEVICES)

    image_embeddings = input_types[1 : 1 + _N_DEVICES]
    image_indices = input_types[1 + _N_DEVICES : 1 + 2 * _N_DEVICES]

    for embed_type in image_embeddings:
        assert embed_type.dtype == DType.bfloat16
        assert int(embed_type.shape[-1]) == _HIDDEN_SIZE

    for index_type in image_indices:
        assert index_type.dtype == DType.int32


def test_inputs_pack_finalized_vision_merge_buffers() -> None:
    """The packed buffers must match the fused graph's declared inputs.

    The vision seam sets the base ``vision_embeddings`` /
    ``vision_scatter_indices`` fields on every prepared batch (with the model's
    zero-row empties on text-only / decode steps); ``.buffers`` must splat them
    right after ``tokens``, or execute fails on an input-count mismatch.
    """
    model = _fake_pipeline_model(_N_DEVICES)
    inputs = _bare_dflash_inputs(_N_DEVICES)
    assert inputs.vision_embeddings == []
    assert inputs.vision_scatter_indices == []

    encoder_cache: VisionEncoderCache[Any] = VisionEncoderCache(
        devices=model.devices
    )
    encoder_cache.finalize_vision_inputs(model, inputs, model.devices, None)

    _fill_spec_decode_tail(inputs)

    buffers = inputs.buffers
    assert len(buffers) == len(_graph_input_types(_N_DEVICES))

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


def test_inputs_pack_vision_and_bitmask_together() -> None:
    """Vision and the structured-output bitmask triple must coexist in the ABI.

    Vision leads the signature and the bitmask triple trails it, so both
    per-feature tests pass while the combination is misaligned. Pins the full
    arity of what the deployment serves: distributed + vision + bitmask.
    """
    model = _fake_pipeline_model(_N_DEVICES)
    inputs = _bare_dflash_inputs(_N_DEVICES)

    encoder_cache: VisionEncoderCache[Any] = VisionEncoderCache(
        devices=model.devices
    )
    encoder_cache.finalize_vision_inputs(model, inputs, model.devices, None)
    _fill_spec_decode_tail(inputs)

    pinned_bitmask = Buffer.zeros((1,), dtype=DType.int32)
    wait_payload = Buffer.zeros((2,), dtype=DType.int64)
    device_bitmask_scratch = Buffer.zeros((1,), dtype=DType.int32)
    inputs.structured_output = True
    inputs.pinned_bitmask = pinned_bitmask
    inputs.wait_payload = wait_payload
    inputs.device_bitmask_scratch = device_bitmask_scratch

    buffers = inputs.buffers
    assert len(buffers) == len(
        _graph_input_types(_N_DEVICES, enable_structured_output=True)
    )

    # Vision still leads, bitmask still trails.
    assert all(
        packed is expected
        for packed, expected in zip(
            buffers[1 : 1 + _N_DEVICES],
            inputs.vision_embeddings,
            strict=True,
        )
    )
    assert buffers[-3:] == (
        pinned_bitmask,
        wait_payload,
        device_bitmask_scratch,
    )
