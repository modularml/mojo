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
"""Tests for `Tensor.from_dim`: materializing a (symbolic) dim as a scalar
tensor, and using it as a runtime predicate for `F.cond`. CPU-only."""

import max.experimental.functional as F
import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.experimental import nn
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, Dim, TensorType

CPU_REF = DeviceRef.CPU()


class Batch(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return Tensor.from_dim(x.shape[0])


class Product(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return Tensor.from_dim(x.shape[0] * x.shape[1])  # algebraic dim


class StaticSeven(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return Tensor.from_dim(7)  # plain int


class CondOnBatch(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        batch = x.shape[0]
        out_t = TensorType(x.dtype, x.shape, device=x.device)
        pred = Tensor.from_dim(batch) <= 2
        (out,) = F.cond(pred, [out_t], lambda: x * 2.0, lambda: x * 4.0)
        return out


def test_from_dim_symbolic_value_and_contract() -> None:
    """A symbolic dim materializes to its runtime value as a rank-0 int64 CPU
    tensor."""
    compiled = Batch().compile(
        TensorType(DType.float32, ["batch", 8], device=CPU_REF)
    )
    out = compiled(Tensor.ones([5, 8], dtype=DType.float32, device=CPU()))
    assert list(out.shape) == []  # rank-0 scalar
    assert out.dtype == DType.int64
    assert int(out.to_numpy()) == 5


def test_from_dim_algebraic_dim() -> None:
    """`batch * seq` materializes to the runtime product."""
    compiled = Product().compile(
        TensorType(DType.float32, ["batch", "seq"], device=CPU_REF)
    )
    out = compiled(Tensor.ones([5, 3], dtype=DType.float32, device=CPU()))
    assert int(out.to_numpy()) == 15


def test_from_dim_static_int() -> None:
    """A plain int materializes to a constant scalar."""
    compiled = StaticSeven().compile(
        TensorType(DType.float32, ["batch"], device=CPU_REF)
    )
    out = compiled(Tensor.ones([3], dtype=DType.float32, device=CPU()))
    assert int(out.to_numpy()) == 7


def test_from_dim_predicate_in_cond() -> None:
    """`Tensor.from_dim(dim) <= k` is a valid runtime `F.cond` predicate, and
    branch selection follows the runtime dim value."""
    compiled = CondOnBatch().compile(
        TensorType(DType.float32, ["batch", 4], device=CPU_REF)
    )

    # batch == 1 (<= 2)  -> then branch (x * 2)
    small = compiled(Tensor.ones([1, 4], dtype=DType.float32, device=CPU()))
    assert np.allclose(small.to_numpy(), 2.0)

    # batch == 5 (> 2)   -> else branch (x * 4)
    large = compiled(Tensor.ones([5, 4], dtype=DType.float32, device=CPU()))
    assert np.allclose(large.to_numpy(), 4.0)


# --- Eager mode (no compile) -------------------------------------------------


def test_from_dim_eager_static() -> None:
    """In eager mode a static dim materializes to its value (no graph needed).
    Covers both a literal int and a concrete tensor's (always-static) shape."""
    r = Tensor.from_dim(7)
    r._sync_realize()
    assert list(r.shape) == []
    assert r.dtype == DType.int64
    assert r.item() == 7

    x = Tensor(
        np.ones((5, 8), dtype=np.float32), dtype=DType.float32, device=CPU()
    )
    got = Tensor.from_dim(x.shape[0])  # x.shape[0] is a StaticDim in eager
    got._sync_realize()
    assert got.item() == 5


def test_from_dim_eager_symbolic_raises() -> None:
    """A symbolic dim has no value in eager mode -> clear ValueError."""
    with pytest.raises(ValueError, match="eager mode"):
        Tensor.from_dim(Dim("x"))


def test_from_dim_eager_algebraic_raises() -> None:
    """An algebraic dim has no value in eager mode -> clear ValueError."""
    with pytest.raises(ValueError, match="eager mode"):
        Tensor.from_dim(Dim("a") * Dim("b"))
