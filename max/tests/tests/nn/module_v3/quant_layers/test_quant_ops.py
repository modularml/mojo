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

"""Checks for the quant-format dispatch helpers in ``quant_ops``.

These cover the format predicates, weight construction/stacking, and the EP
dispatch-payload parsing — the plumbing that decides which kernel family a
weight or activation ends up on. The kernel calls themselves are covered by the
layer-level lazy-trace tests.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import TensorValue
from max.nn.comm.ep import EPConfig
from max.nn.quant_config import QuantConfig, QuantFormat
from max.pipelines.architectures.deepseekV3_modulev3.layers import quant_ops
from max.pipelines.architectures.deepseekV3_modulev3.layers.quant_tensor import (
    FP8BlockTensor,
    NVFP4Activation,
    NVFP4Tensor,
)

_HIDDEN_DIM = 256
_MOE_DIM = 128
_NUM_EXPERTS = 4
_NUM_LOCAL_EXPERTS = 2
_TOKENS = 8


# --------------------------------------------------------------------------- #
# Format predicates
# --------------------------------------------------------------------------- #


def test_predicates_separate_fp8_from_nvfp4(
    fp8_quant_config: QuantConfig, nvfp4_quant_config: QuantConfig
) -> None:
    """FP8 and NVFP4 are distinguished by format, not by block granularity.

    Both configs are block-scaled, so a ``weight_scale.is_block`` test would
    route NVFP4 weights onto the FP8 kernels.
    """
    assert nvfp4_quant_config.weight_scale.is_block

    assert quant_ops.is_fp8_block_quantized(fp8_quant_config)
    assert not quant_ops.is_fp8_block_quantized(nvfp4_quant_config)
    assert not quant_ops.is_fp8_block_quantized(None)

    assert quant_ops.is_nvfp4_quantized(nvfp4_quant_config)
    assert not quant_ops.is_nvfp4_quantized(fp8_quant_config)
    assert not quant_ops.is_nvfp4_quantized(None)


def test_only_nvfp4_needs_dispatch_scales_and_offsets(
    fp8_quant_config: QuantConfig, nvfp4_quant_config: QuantConfig
) -> None:
    """The EP dispatch and MoE index kernels take extra NVFP4-only inputs."""
    assert quant_ops.ep_requires_dispatch_scales(nvfp4_quant_config)
    assert not quant_ops.ep_requires_dispatch_scales(fp8_quant_config)
    assert not quant_ops.ep_requires_dispatch_scales(None)

    assert quant_ops.moe_requires_scales_offsets(nvfp4_quant_config)
    assert not quant_ops.moe_requires_scales_offsets(fp8_quant_config)
    assert not quant_ops.moe_requires_scales_offsets(None)


# --------------------------------------------------------------------------- #
# quantized_weight
# --------------------------------------------------------------------------- #


def test_quantized_weight_nvfp4(nvfp4_quant_config: QuantConfig) -> None:
    """An NVFP4 config yields a packed NVFP4Tensor for the logical shape."""
    with F.lazy():
        weight = quant_ops.quantized_weight(
            _MOE_DIM, _HIDDEN_DIM, nvfp4_quant_config
        )

        assert isinstance(weight, NVFP4Tensor)
        assert weight.block_size == (1, 16)
        assert list(weight.data.shape) == [_MOE_DIM, _HIDDEN_DIM // 2]
        assert list(weight.weight_scale.shape) == [_MOE_DIM, _HIDDEN_DIM // 16]


def test_quantized_weight_dispatch_by_format(
    fp8_quant_config: QuantConfig, nvfp4_quant_config: QuantConfig
) -> None:
    """Each supported format maps to its own weight wrapper."""
    with F.lazy():
        assert isinstance(
            quant_ops.quantized_weight(_MOE_DIM, _HIDDEN_DIM, None), Tensor
        )
        assert isinstance(
            quant_ops.quantized_weight(_MOE_DIM, _HIDDEN_DIM, fp8_quant_config),
            FP8BlockTensor,
        )
        assert isinstance(
            quant_ops.quantized_weight(
                _MOE_DIM, _HIDDEN_DIM, nvfp4_quant_config
            ),
            NVFP4Tensor,
        )


def test_quantized_weight_rejects_unsupported_format(
    nvfp4_quant_config: QuantConfig,
) -> None:
    """Formats this architecture has not wired up are rejected up front."""
    mxfp4 = dataclasses.replace(nvfp4_quant_config, format=QuantFormat.MXFP4)
    with F.lazy():
        with pytest.raises(ValueError, match="not yet supported"):
            quant_ops.quantized_weight(_MOE_DIM, _HIDDEN_DIM, mxfp4)


# --------------------------------------------------------------------------- #
# stack / concat
# --------------------------------------------------------------------------- #


def _nvfp4_weight(rows: int = _MOE_DIM, cols: int = _HIDDEN_DIM) -> NVFP4Tensor:
    return NVFP4Tensor.zeros((rows, cols))


def test_stack_nvfp4_adds_expert_axis_to_every_leaf() -> None:
    """Stacking per-expert weights lifts all four leaves to a leading E axis."""
    with F.lazy():
        stacked = quant_ops.stack([_nvfp4_weight() for _ in range(4)], axis=0)

        assert isinstance(stacked, NVFP4Tensor)
        assert list(stacked.data.shape) == [4, _MOE_DIM, _HIDDEN_DIM // 2]
        assert list(stacked.weight_scale.shape) == [
            4,
            _MOE_DIM,
            _HIDDEN_DIM // 16,
        ]
        # The per-tensor scales become per-expert vectors.
        assert list(stacked.weight_scale_2.shape) == [4]
        assert list(stacked.input_scale.shape) == [4]
        assert stacked.block_size == (1, 16)


def test_concat_nvfp4_shares_the_global_scales() -> None:
    """Fusing gate||up concatenates the block leaves and shares the globals.

    modelopt emits one global scale for the fused projection, so the second
    weight's ``weight_scale_2`` / ``input_scale`` are intentionally dropped.
    """
    with F.lazy():
        gate, up = _nvfp4_weight(), _nvfp4_weight()
        fused = quant_ops.concat_weights(gate, up, axis=0)

        assert isinstance(fused, NVFP4Tensor)
        assert list(fused.data.shape) == [2 * _MOE_DIM, _HIDDEN_DIM // 2]
        assert list(fused.weight_scale.shape) == [
            2 * _MOE_DIM,
            _HIDDEN_DIM // 16,
        ]
        assert fused.weight_scale_2 is gate.weight_scale_2
        assert fused.input_scale is gate.input_scale


def test_concat_nvfp4_rejects_non_row_axis() -> None:
    with F.lazy():
        with pytest.raises(ValueError, match="only supports axis=0"):
            quant_ops.concat_nvfp4(_nvfp4_weight(), _nvfp4_weight(), axis=1)


def test_concat_nvfp4_rejects_mixed_block_sizes() -> None:
    with F.lazy():
        other = NVFP4Tensor.zeros((_MOE_DIM, _HIDDEN_DIM), block_size=(1, 32))
        with pytest.raises(ValueError, match="same block_size"):
            quant_ops.concat_nvfp4(_nvfp4_weight(), other)


def test_concat_nvfp4_requires_a_tensor() -> None:
    with pytest.raises(ValueError, match="at least one tensor"):
        quant_ops.concat_nvfp4()


# --------------------------------------------------------------------------- #
# Fused-SwiGLU sigma permutation
# --------------------------------------------------------------------------- #


def test_sigma_permute_interleaves_gate_and_up_rows() -> None:
    """The fused SwiGLU kernel wants ``[g0, u0, g1, u1, ...]`` on the N axis.

    Checked eagerly on CPU: the stacked ``[gate; up]`` rows come back
    row-interleaved, and the same permutation is applied to the block scales.
    """
    num_experts, gate_rows, cols = 2, 3, 4
    rows = 2 * gate_rows

    data = np.arange(num_experts * rows * cols, dtype=np.uint8).reshape(
        num_experts, rows, cols
    )
    scales = np.arange(num_experts * rows * 2, dtype=np.float32).reshape(
        num_experts, rows, 2
    )

    weight = NVFP4Tensor(
        data=Tensor(data, device=CPU()),
        weight_scale=Tensor(scales, device=CPU()),
        weight_scale_2=Tensor(np.ones(num_experts, np.float32), device=CPU()),
        input_scale=Tensor(np.ones(num_experts, np.float32), device=CPU()),
    )

    permuted = quant_ops.sigma_permute_gate_up_nvfp4(weight)

    # sigma(2i) = i (gate row i), sigma(2i + 1) = gate_rows + i (up row i).
    order = [i // 2 if i % 2 == 0 else gate_rows + i // 2 for i in range(rows)]
    np.testing.assert_array_equal(permuted.data.to_numpy(), data[:, order])
    np.testing.assert_array_equal(
        permuted.weight_scale.to_numpy(), scales[:, order]
    )
    # The global scales are untouched by the permutation.
    assert permuted.weight_scale_2 is weight.weight_scale_2
    assert permuted.input_scale is weight.input_scale
    assert permuted.block_size == weight.block_size


# --------------------------------------------------------------------------- #
# EPDispatchPayload
# --------------------------------------------------------------------------- #


def _ep_config(
    dispatch_dtype: DType, quant_config: QuantConfig | None
) -> EPConfig:
    return EPConfig(
        dispatch_dtype=dispatch_dtype,
        combine_dtype=DType.bfloat16,
        hidden_size=_HIDDEN_DIM,
        top_k=2,
        n_experts=_NUM_EXPERTS,
        max_tokens_per_rank=_TOKENS,
        n_gpus_per_node=2,
        n_nodes=1,
        dispatch_quant_config=quant_config,
    )


def _value(shape: list[int], dtype: DType) -> TensorValue:
    return TensorValue(Tensor.zeros(shape, dtype=dtype, device=CPU()))


def _metadata() -> TensorValue:
    """The host grouped-matmul metadata tuple element ``[max_m, n_active]``."""
    return TensorValue(Tensor.zeros([2], dtype=DType.uint32, device=CPU()))


def _bf16_dispatch() -> list[tuple[TensorValue, ...]]:
    return [
        (
            _value([_TOKENS, _HIDDEN_DIM], DType.bfloat16),
            _value([_NUM_LOCAL_EXPERTS + 1], DType.uint32),
            _value([_NUM_LOCAL_EXPERTS], DType.int32),
            _metadata(),
        )
        for _ in range(2)
    ]


def _fp8_dispatch() -> list[tuple[TensorValue, ...]]:
    return [
        (
            _value([_TOKENS, _HIDDEN_DIM], DType.float8_e4m3fn),
            _value([_TOKENS, _HIDDEN_DIM // 128], DType.float32),
            _value([_NUM_LOCAL_EXPERTS + 1], DType.uint32),
            _value([_NUM_LOCAL_EXPERTS], DType.int32),
            _metadata(),
        )
        for _ in range(2)
    ]


def _nvfp4_dispatch() -> list[tuple[TensorValue, ...]]:
    """The six NVFP4 dispatch outputs (see ``_ep_dispatch_output_types``)."""
    return [
        (
            _value([_TOKENS, _HIDDEN_DIM // 2], DType.uint8),
            _value([1, 1, 32, 4, 4], DType.float8_e4m3fn),
            _value([_NUM_LOCAL_EXPERTS + 1], DType.uint32),
            _value([_NUM_LOCAL_EXPERTS], DType.uint32),
            _value([_NUM_LOCAL_EXPERTS], DType.int32),
            _metadata(),
        )
        for _ in range(2)
    ]


def test_payload_bf16_has_no_scales() -> None:
    """A bf16 dispatch carries usage stats and no scale columns."""
    with F.lazy():
        config = _ep_config(DType.bfloat16, None)
        payload = quant_ops.EPDispatchPayload.from_dispatch(
            _bf16_dispatch(), None, config
        )

        assert payload.scales is None
        assert payload.scales_offset is None
        assert payload.usage_stats is not None
        assert len(payload.tokens) == 2
        assert list(payload.expert_start[0].shape) == [_NUM_LOCAL_EXPERTS + 1]
        assert list(payload.expert_ids[0].shape) == [_NUM_LOCAL_EXPERTS]
        # The bf16 grouped matmul reads num_active_experts off the host.
        assert payload.usage_stats[0].device.is_host

        assert payload.local_map_tokens(None) == payload.tokens


def test_payload_fp8_carries_activation_scales(
    fp8_quant_config: QuantConfig,
) -> None:
    """An FP8 dispatch yields pre-quantized FP8BlockTensor activations."""
    with F.lazy():
        config = _ep_config(DType.float8_e4m3fn, fp8_quant_config)
        payload = quant_ops.EPDispatchPayload.from_dispatch(
            _fp8_dispatch(), fp8_quant_config, config
        )

        assert payload.scales is not None
        assert payload.scales_offset is None
        assert payload.usage_stats is not None

        tokens = payload.local_map_tokens(fp8_quant_config)
        assert len(tokens) == 2
        first = tokens[0]
        assert isinstance(first, FP8BlockTensor)
        # Activations are quantized per-token along K, not on the weight grid.
        assert first.block_size == (
            1,
            fp8_quant_config.weight_scale.block_size[1],  # type: ignore[index]
        )


def test_payload_nvfp4_carries_scales_and_offsets(
    nvfp4_quant_config: QuantConfig,
) -> None:
    """An NVFP4 dispatch adds scale offsets and drops the host usage stats."""
    with F.lazy():
        config = _ep_config(DType.float8_e4m3fn, nvfp4_quant_config)
        payload = quant_ops.EPDispatchPayload.from_dispatch(
            _nvfp4_dispatch(), nvfp4_quant_config, config
        )

        assert payload.scales is not None
        assert payload.scales_offset is not None
        # The NVFP4 grouped matmul synthesizes its own host stats.
        assert payload.usage_stats is None

        global_scale = Tensor.zeros((), dtype=DType.float32, device=CPU())
        tokens = payload.local_map_tokens(
            nvfp4_quant_config, nvfp4_global_scale=global_scale
        )
        assert len(tokens) == 2
        first = tokens[0]
        assert isinstance(first, NVFP4Activation)
        assert first.data is payload.tokens[0]
        assert first.scales is payload.scales[0]
        assert first.scales_offset is payload.scales_offset[0]
        assert first.input_scale is global_scale


def test_payload_nvfp4_tokens_require_the_uniform_scale(
    nvfp4_quant_config: QuantConfig,
) -> None:
    """The dispatch quantized with one scale; wrapping without it is a bug."""
    with F.lazy():
        config = _ep_config(DType.float8_e4m3fn, nvfp4_quant_config)
        payload = quant_ops.EPDispatchPayload.from_dispatch(
            _nvfp4_dispatch(), nvfp4_quant_config, config
        )

        with pytest.raises(AssertionError, match="uniform scale"):
            payload.local_map_tokens(nvfp4_quant_config)
