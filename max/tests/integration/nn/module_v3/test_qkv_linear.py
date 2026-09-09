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
"""Tests for the ModuleV3 :class:`QKVLinear` fused q/k/v projection."""

from __future__ import annotations

import numpy as np
from max.driver import CPU
from max.dtype import DType
from max.experimental.nn import Module, TransparentModule
from max.experimental.nn.common_layers.linear import QKVLinear
from max.experimental.tensor import Tensor, default_dtype

IN_DIM = 8
Q_DIM = 16
KV_DIM = 4  # GQA: q_dim != kv_dim.
OUT_DIM = Q_DIM + 2 * KV_DIM
TOKENS = 3


class _Wrapper(Module[[Tensor], Tensor]):
    """A parent so name-transparency of a child QKVLinear is observable."""

    def __init__(self, *, stacked: bool, bias: bool) -> None:
        self.qkv_proj = QKVLinear(
            IN_DIM, Q_DIM, KV_DIM, bias=bias, stacked=stacked
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.qkv_proj(x)


def test_records_dims_and_mode() -> None:
    separate = QKVLinear(IN_DIM, Q_DIM, KV_DIM)
    assert separate.q_dim == Q_DIM
    assert separate.kv_dim == KV_DIM
    assert isinstance(separate, TransparentModule)
    assert separate.name_transparent is True
    assert hasattr(separate, "q_proj")
    assert not hasattr(separate, "weight")

    stacked = QKVLinear(IN_DIM, Q_DIM, KV_DIM, stacked=True)
    assert isinstance(stacked, TransparentModule)
    assert stacked.name_transparent is False
    assert stacked.weight.shape == [OUT_DIM, IN_DIM]
    assert not hasattr(stacked, "q_proj")


def test_separate_child_exposes_native_projection_names() -> None:
    names = {name for name, _ in _Wrapper(stacked=False, bias=False).parameters}
    assert names == {
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
    }


def test_stacked_child_keeps_fused_name() -> None:
    names = {name for name, _ in _Wrapper(stacked=True, bias=False).parameters}
    assert names == {"qkv_proj.weight"}


def _forward_parity(*, stacked: bool, bias: bool) -> None:
    rng = np.random.default_rng(0)
    q = rng.standard_normal((Q_DIM, IN_DIM)).astype(np.float32)
    k = rng.standard_normal((KV_DIM, IN_DIM)).astype(np.float32)
    v = rng.standard_normal((KV_DIM, IN_DIM)).astype(np.float32)
    fused_w = np.concatenate([q, k, v], axis=0)
    x_np = rng.standard_normal((TOKENS, IN_DIM)).astype(np.float32)

    fused_b = np.zeros((OUT_DIM,), dtype=np.float32)
    if bias:
        fused_b = rng.standard_normal((OUT_DIM,)).astype(np.float32)

    with default_dtype(DType.float32):
        proj = QKVLinear(IN_DIM, Q_DIM, KV_DIM, bias=bias, stacked=stacked)
    if stacked:
        proj.weight = Tensor(fused_w, device=CPU())
        if bias:
            proj.bias = Tensor(fused_b, device=CPU())
    else:
        proj.q_proj.weight = Tensor(q, device=CPU())
        proj.k_proj.weight = Tensor(k, device=CPU())
        proj.v_proj.weight = Tensor(v, device=CPU())
        if bias:
            proj.q_proj.bias = Tensor(fused_b[:Q_DIM], device=CPU())
            proj.k_proj.bias = Tensor(
                fused_b[Q_DIM : Q_DIM + KV_DIM], device=CPU()
            )
            proj.v_proj.bias = Tensor(fused_b[Q_DIM + KV_DIM :], device=CPU())

    out = proj(Tensor(x_np, device=CPU())).to_numpy()
    expected = x_np @ fused_w.T + fused_b
    assert out.shape == (TOKENS, OUT_DIM)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)
    assert np.array_equal(proj.fused_weight.to_numpy(), fused_w)


def test_forward_separate() -> None:
    _forward_parity(stacked=False, bias=False)


def test_forward_separate_bias() -> None:
    _forward_parity(stacked=False, bias=True)


def test_forward_stacked() -> None:
    _forward_parity(stacked=True, bias=False)


def test_forward_stacked_bias() -> None:
    _forward_parity(stacked=True, bias=True)
