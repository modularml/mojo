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
"""CPU/graph-build tests for pre-fused projection layers (DISTINF-194 v2).

A dedicated :class:`~max.nn.FusedMLP` declares one ``gate_up_proj_fused``
weight, and ``StackedLinear(stacked=True)`` declares one qkv/qk weight; both
expect the checkpoint to carry the pre-fused tensor. No process-global toggle.
The gemma4 weight-adapter fusing + strict round-trip live in the gemma4
integration tests (they depend on the pipeline package).
"""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, ShardingStrategy, TensorValue, Weight
from max.nn import FusedMLP, Module
from max.nn.stacked_linear import StackedLinear

_HIDDEN = 8
_FFL = 16
_Q, _KV = 12, 4


def _input(rows: int) -> TensorValue:
    return TensorValue(
        Weight("x", DType.float32, [rows, _HIDDEN], DeviceRef.CPU())
    )


# ------------------------------- FusedMLP -------------------------------- #


class _MLPWrapper(Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = FusedMLP(
            dtype=DType.float32,
            hidden_dim=_HIDDEN,
            feed_forward_length=_FFL,
            devices=[DeviceRef.CPU()],
        )

    def __call__(self) -> None:  # pragma: no cover
        raise NotImplementedError


def test_fused_mlp_declares_single_gate_up_weight() -> None:
    wrapper = _MLPWrapper()
    keys = set(wrapper.raw_state_dict().keys())
    assert keys == {"mlp.gate_up_proj_fused", "mlp.down_proj.weight"}
    assert list(wrapper.mlp.gate_up_proj_fused.shape) == [2 * _FFL, _HIDDEN]


def test_fused_mlp_forward_shape() -> None:
    """The fused gate/up weight flows straight into the matmul (no in-graph
    concat of separate gate/up weights)."""
    wrapper = _MLPWrapper()
    with Graph("fused_mlp", input_types=[]):
        out = wrapper.mlp(_input(3))
    assert list(out.shape) == [3, _HIDDEN]


def test_fused_mlp_survives_single_device_shard() -> None:
    """gemma4 always runs the shard, not the original. At single-device TP the
    shard shares the parent's fused weight and its forward still works."""
    wrapper = _MLPWrapper()
    wrapper.mlp.sharding_strategy = ShardingStrategy.tensor_parallel(1)
    shard = wrapper.mlp.shard([DeviceRef.CPU()])[0]
    assert shard.gate_up_proj_fused.name.startswith("gate_up_proj_fused")
    with Graph("fused_mlp_shard", input_types=[]):
        out = shard(_input(3))
    assert list(out.shape) == [3, _HIDDEN]


def test_fused_mlp_rejects_multiple_constructor_devices() -> None:
    with pytest.raises(ValueError, match="requires exactly one device"):
        FusedMLP(
            dtype=DType.float32,
            hidden_dim=_HIDDEN,
            feed_forward_length=_FFL,
            devices=[DeviceRef.CPU(), DeviceRef.CPU()],
        )


def test_fused_mlp_rejects_multi_device_tensor_parallelism() -> None:
    wrapper = _MLPWrapper()
    with pytest.raises(
        ValueError, match="does not support tensor parallelism across 2 devices"
    ):
        wrapper.mlp.sharding_strategy = ShardingStrategy.tensor_parallel(2)


# ----------------------- StackedLinear(stacked=True) --------------------- #


class _QKVWrapper(Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv_proj = StackedLinear(
            in_dim=_HIDDEN,
            out_dims=[_Q, _KV, _KV],
            names=["q_proj", "k_proj", "v_proj"],
            dtype=DType.float32,
            device=DeviceRef.CPU(),
            stacked=True,
        )

    def __call__(self) -> None:  # pragma: no cover
        raise NotImplementedError


class _QKWrapper(Module):
    def __init__(self) -> None:
        super().__init__()
        self.qk_proj = StackedLinear(
            in_dim=_HIDDEN,
            out_dims=[_Q, _KV],
            names=["q_proj", "k_proj"],
            dtype=DType.float32,
            device=DeviceRef.CPU(),
            stacked=True,
        )

    def __call__(self) -> None:  # pragma: no cover
        raise NotImplementedError


def test_stacked_qkv_is_single_named_weight() -> None:
    wrapper = _QKVWrapper()
    assert set(wrapper.raw_state_dict().keys()) == {"qkv_proj.weight"}
    assert list(wrapper.qkv_proj.weight.shape) == [_Q + 2 * _KV, _HIDDEN]


def test_stacked_qk_is_single_named_weight() -> None:
    wrapper = _QKWrapper()
    assert set(wrapper.raw_state_dict().keys()) == {"qk_proj.weight"}
    assert list(wrapper.qk_proj.weight.shape) == [_Q + _KV, _HIDDEN]
