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
"""Structured-output bitmask wiring for the unified Kimi dflash pipeline.

The Kimi dflash graph must declare the constrained-decoding bitmask
input triple (pinned_bitmask, wait_payload, device_bitmask_scratch) when
structured output is enabled, and the ``*Inputs.buffers`` ABI tail must
pack the same triple so the buffer list lines up with the compiled graph
signature. These tests fail without the bitmask wiring: the graph used
to compile without bitmask inputs, silently serving structured-output
requests unconstrained.

The dflash Llama3 graph is deliberately still unwired, so it is covered
here only by a guard that it keeps omitting the triple.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.kv_cache import MHAKVCacheParams, MultiKVCacheParams
from max.pipelines.architectures.unified_dflash_kimi_k25.model import (
    UnifiedDflashKimiK25Inputs,
)
from max.pipelines.architectures.unified_dflash_kimi_k25.unified_dflash_kimi_k25 import (
    UnifiedDflashKimiK25,
)
from max.pipelines.architectures.unified_dflash_llama3.model import (
    UnifiedDflashLlama3Inputs,
)
from max.pipelines.lib.interfaces.pipeline_model import UnifiedSpecDecodeInputs
from max.pipelines.speculative.spec_input_types import (
    SpecDecodeInputTypeSpec,
    build_spec_decode_input_types,
)


def _kv_params() -> MHAKVCacheParams:
    """Minimal single-device MHA KV cache leaf for input-type construction."""
    return MHAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=64,
        num_layers=2,
        devices=[DeviceRef.GPU()],
        n_kv_heads=8,
    )


def _is_bitmask_triple(
    types: tuple[TensorType | BufferType, ...],
) -> bool:
    """Whether the last three input types are the bitmask triple."""
    if len(types) < 3:
        return False
    pinned, wait, scratch = types[-3], types[-2], types[-1]
    return (
        isinstance(pinned, TensorType)
        and pinned.dtype == DType.int32
        and pinned.device == DeviceRef.CPU()
        and isinstance(wait, BufferType)
        and wait.dtype == DType.int64
        and wait.device == DeviceRef.CPU()
        and isinstance(scratch, BufferType)
        and scratch.dtype == DType.int32
        and scratch.device != DeviceRef.CPU()
    )


@pytest.mark.parametrize("enable_structured_output", [False, True])
def test_dflash_kimi_k25_input_types_bitmask_triple(
    enable_structured_output: bool,
) -> None:
    stub: Any = SimpleNamespace(
        config=SimpleNamespace(
            target=SimpleNamespace(
                devices=[DeviceRef.GPU()],
                data_parallel_degree=1,
                hidden_size=128,
            )
        ),
        target=SimpleNamespace(ep_manager=None),
        enable_structured_output=enable_structured_output,
    )
    kv_params = MultiKVCacheParams.from_params(
        {"target": _kv_params(), "draft": _kv_params()}
    )
    types = UnifiedDflashKimiK25.input_types(stub, kv_params)
    assert _is_bitmask_triple(types) == enable_structured_output


def _tail_sentinels() -> dict[str, object]:
    return {
        "draft_tokens": object(),
        "seed": object(),
        "temperature": object(),
        "top_k": object(),
        "max_k": object(),
        "top_p": object(),
        "min_top_p": object(),
    }


_BITMASK_SENTINELS = {
    "pinned_bitmask": object(),
    "wait_payload": object(),
    "device_bitmask_scratch": object(),
}


@pytest.mark.parametrize("structured_output", [False, True])
def test_dflash_kimi_k25_buffers_tail_packs_bitmask_triple(
    structured_output: bool,
) -> None:
    inputs = UnifiedDflashKimiK25Inputs(
        tokens=object(),  # type: ignore[arg-type]
        input_row_offsets=object(),  # type: ignore[arg-type]
        host_input_row_offsets=object(),  # type: ignore[arg-type]
        return_n_logits=object(),  # type: ignore[arg-type]
        data_parallel_splits=object(),  # type: ignore[arg-type]
        signal_buffers=[],
        batch_context_lengths=[],
        structured_output=structured_output,
        **_tail_sentinels(),  # type: ignore[arg-type]
        **_BITMASK_SENTINELS,  # type: ignore[arg-type]
    )
    buffers = inputs.buffers
    if structured_output:
        assert buffers[-3:] == (
            _BITMASK_SENTINELS["pinned_bitmask"],
            _BITMASK_SENTINELS["wait_payload"],
            _BITMASK_SENTINELS["device_bitmask_scratch"],
        )
    else:
        assert not set(_BITMASK_SENTINELS.values()) & set(buffers)


def test_dflash_llama3_buffers_omit_bitmask_triple() -> None:
    """The Llama3 dflash graph declares no bitmask inputs, so its ABI
    tail must omit the triple even when the pipeline binds one."""
    inputs = UnifiedDflashLlama3Inputs(
        tokens=object(),  # type: ignore[arg-type]
        input_row_offsets=object(),  # type: ignore[arg-type]
        return_n_logits=object(),  # type: ignore[arg-type]
        structured_output=True,
        **_tail_sentinels(),  # type: ignore[arg-type]
        **_BITMASK_SENTINELS,  # type: ignore[arg-type]
    )
    assert not set(_BITMASK_SENTINELS.values()) & set(inputs.buffers)


@pytest.mark.parametrize("include_in_thinking_phase", [False, True])
@pytest.mark.parametrize("enable_structured_output", [False, True])
def test_tail_buffers_match_input_types_lockstep(
    include_in_thinking_phase: bool,
    enable_structured_output: bool,
) -> None:
    """The ABI tail must stay in lockstep with the graph-signature tail.

    ``build_spec_decode_input_types`` declares three leading inputs
    (tokens, input_row_offsets, return_n_logits) followed by the
    flattened KV tree in the non-distributed layout; everything after
    them is the spec-decode tail that ``_spec_decode_tail_buffers``
    packs.
    """
    kv_params = _kv_params()
    types = build_spec_decode_input_types(
        SpecDecodeInputTypeSpec(
            distributed=False,
            include_in_thinking_phase=include_in_thinking_phase,
            enable_structured_output=enable_structured_output,
        ),
        devices=[DeviceRef.GPU()],
        kv_params=kv_params,
    )
    tail_types = types[3 + len(kv_params.flattened_kv_inputs()) :]

    inputs = UnifiedSpecDecodeInputs(
        in_thinking_phase=object(),  # type: ignore[arg-type]
        structured_output=enable_structured_output,
        **_tail_sentinels(),  # type: ignore[arg-type]
        **_BITMASK_SENTINELS,  # type: ignore[arg-type]
    )
    tail_buffers = inputs._spec_decode_tail_buffers(
        include_in_thinking_phase=include_in_thinking_phase,
    )
    assert len(tail_buffers) == len(tail_types)
