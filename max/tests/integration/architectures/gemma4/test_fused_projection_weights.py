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
"""CPU tests for gemma4 pre-fused projection weights (DISTINF-194 v2).

Covers the fused adapter's key/byte contract and a strict-load round-trip
proving its output keys exactly match the fused model's expected weight names.
"""

from __future__ import annotations

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.graph import DeviceRef
from max.graph.type import Shape
from max.graph.weights import WeightData
from max.nn import FusedMLP, Module
from max.nn.layer import LayerList
from max.nn.stacked_linear import StackedLinear
from max.pipelines.architectures.gemma4.weight_adapters import (
    fuse_gemma4_projection_weights,
)

_HIDDEN = 8
_FFL = 16
_Q, _KV = 12, 4


def _wd(name: str, shape: list[int]) -> WeightData:
    arr = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    return WeightData(
        Buffer.from_dlpack(arr), name, DType.float32, Shape(shape)
    )


def _unfused_state_dict() -> dict[str, WeightData]:
    # Layer 0: sliding -> q,k,v.  Layer 1: global k_eq_v -> q,k only.
    return {
        "layers.0.mlp.gate_proj.weight": _wd(
            "layers.0.mlp.gate_proj.weight", [_FFL, _HIDDEN]
        ),
        "layers.0.mlp.up_proj.weight": _wd(
            "layers.0.mlp.up_proj.weight", [_FFL, _HIDDEN]
        ),
        "layers.0.mlp.down_proj.weight": _wd(
            "layers.0.mlp.down_proj.weight", [_HIDDEN, _FFL]
        ),
        "layers.0.self_attn.q_proj.weight": _wd(
            "layers.0.self_attn.q_proj.weight", [_Q, _HIDDEN]
        ),
        "layers.0.self_attn.k_proj.weight": _wd(
            "layers.0.self_attn.k_proj.weight", [_KV, _HIDDEN]
        ),
        "layers.0.self_attn.v_proj.weight": _wd(
            "layers.0.self_attn.v_proj.weight", [_KV, _HIDDEN]
        ),
        "layers.1.mlp.gate_proj.weight": _wd(
            "layers.1.mlp.gate_proj.weight", [_FFL, _HIDDEN]
        ),
        "layers.1.mlp.up_proj.weight": _wd(
            "layers.1.mlp.up_proj.weight", [_FFL, _HIDDEN]
        ),
        "layers.1.mlp.down_proj.weight": _wd(
            "layers.1.mlp.down_proj.weight", [_HIDDEN, _FFL]
        ),
        "layers.1.self_attn.q_proj.weight": _wd(
            "layers.1.self_attn.q_proj.weight", [_Q, _HIDDEN]
        ),
        "layers.1.self_attn.k_proj.weight": _wd(
            "layers.1.self_attn.k_proj.weight", [_KV, _HIDDEN]
        ),
    }


def test_fuse_adapter_produces_fused_keys_and_bytes() -> None:
    result = fuse_gemma4_projection_weights(_unfused_state_dict())
    assert set(result.keys()) == {
        "layers.0.mlp.gate_up_proj_fused",
        "layers.0.mlp.down_proj.weight",
        "layers.0.self_attn.qkv_proj.weight",
        "layers.1.mlp.gate_up_proj_fused",
        "layers.1.mlp.down_proj.weight",
        "layers.1.self_attn.qk_proj.weight",
    }
    gate_up = result["layers.0.mlp.gate_up_proj_fused"]
    assert list(gate_up.shape) == [2 * _FFL, _HIDDEN]
    assert list(result["layers.0.self_attn.qkv_proj.weight"].shape) == [
        _Q + 2 * _KV,
        _HIDDEN,
    ]
    assert list(result["layers.1.self_attn.qk_proj.weight"].shape) == [
        _Q + _KV,
        _HIDDEN,
    ]

    # Byte contract: fused = row-concat of sources in declaration order.
    src = _unfused_state_dict()
    expected = np.concatenate(
        [
            np.from_dlpack(
                Buffer.from_dlpack(src["layers.0.mlp.gate_proj.weight"].data)
            ),
            np.from_dlpack(
                Buffer.from_dlpack(src["layers.0.mlp.up_proj.weight"].data)
            ),
        ],
        axis=0,
    )
    actual = np.from_dlpack(Buffer.from_dlpack(gate_up.data))
    assert np.array_equal(actual, expected)


# ---------------- Strict round-trip model (correction #5) ---------------- #


class _SelfAttn(Module):
    def __init__(self, names: list[str], out_dims: list[int]) -> None:
        super().__init__()
        if len(names) == 3:
            self.qkv_proj = StackedLinear(
                in_dim=_HIDDEN,
                out_dims=out_dims,
                names=names,
                dtype=DType.float32,
                device=DeviceRef.CPU(),
                stacked=True,
            )
        else:
            self.qk_proj = StackedLinear(
                in_dim=_HIDDEN,
                out_dims=out_dims,
                names=names,
                dtype=DType.float32,
                device=DeviceRef.CPU(),
                stacked=True,
            )

    def __call__(self) -> None:  # pragma: no cover
        raise NotImplementedError


class _Layer(Module):
    def __init__(self, names: list[str], out_dims: list[int]) -> None:
        super().__init__()
        self.mlp = FusedMLP(
            dtype=DType.float32,
            hidden_dim=_HIDDEN,
            feed_forward_length=_FFL,
            devices=[DeviceRef.CPU()],
        )
        self.self_attn = _SelfAttn(names, out_dims)

    def __call__(self) -> None:  # pragma: no cover
        raise NotImplementedError


class _Model(Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = LayerList(
            [
                _Layer(["q_proj", "k_proj", "v_proj"], [_Q, _KV, _KV]),
                _Layer(["q_proj", "k_proj"], [_Q, _KV]),
            ]
        )

    def __call__(self) -> None:  # pragma: no cover
        raise NotImplementedError


def test_fused_dict_strict_loads_into_model() -> None:
    """The fused adapter output keys EXACTLY match the fused model's expected
    weight names, so strict loading succeeds (correction #5)."""
    model = _Model()
    fused = fuse_gemma4_projection_weights(_unfused_state_dict())
    assert set(fused.keys()) == set(model.raw_state_dict().keys())
    model.load_state_dict(fused, weight_alignment=1, strict=True)
