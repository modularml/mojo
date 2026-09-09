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
"""`F.cond` predicated on a symbolic dim, rewritten with `Tensor.from_dim`.

Predicating on a symbolic dim used to require
`int(batch) <= 2` (which fails — dims are dynamic). The supported form is now a
one-liner:

    pred = Tensor.from_dim(batch) <= 2

`out_t` keeps the composite `batch * seq` dim flowing through the `mo.if`
result; with `Tensor.from_dim` the predicate is ergonomic and the model
compiles and runs.
"""

import max.experimental.functional as F
import numpy as np
from max.driver import CPU
from max.dtype import DType
from max.experimental import nn
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType

GPU = DeviceRef.GPU()
HEADS, VDIM = 6, 256
D = HEADS * VDIM


class CondRepro(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        batch, seq = x.shape[0], x.shape[1]
        total = batch * seq
        x4 = x.reshape((total, HEADS, 1, VDIM))

        out_t = TensorType(x.dtype, [total, HEADS, 1, VDIM], device=x.device)
        pred = Tensor.from_dim(batch) <= 2
        (out,) = F.cond(pred, [out_t], lambda: x4 * 2.0, lambda: x4 * 4.0)

        return out.reshape((batch, seq, D)) * x


def test_cond_repro_compiles_and_runs() -> None:
    compiled = CondRepro().compile(
        TensorType(DType.bfloat16, ["batch", "seq", D], device=GPU)
    )
    print("compiled OK (no crash)")

    # Also exercise both branches at runtime via the symbolic-dim predicate.
    for batch, scale in ((2, 2.0), (5, 4.0)):  # batch<=2 -> *2, else *4
        x = Tensor.ones([batch, 4, D], dtype=DType.bfloat16, device=GPU)
        out = compiled(x)
        arr = out.to(CPU()).cast(DType.float32).to_numpy()
        assert arr.shape == (batch, 4, D)
        assert np.allclose(arr, scale)  # ones * scale * ones
