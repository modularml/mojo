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
"""Shared graph construction for the row-wise reduction ops.

Both the correctness test (``test_rowwise_reductions.py``) and the Kepler
bandwidth benchmark (``//utils/benchmarking/kepler/graph:reductions``) build the
same single-reduction graphs, so this module owns the op-name -> ``max.graph``
op mapping to keep the two in sync. Output shape, dtype and device are derived
from the input value, so one builder serves both the test's symbolic-dimension
graphs and the benchmark's concrete-shape graphs. Norm weights are passed in as
graph values -- the test feeds them as graph inputs, the benchmark bakes them as
constants -- since only their source differs, not the op call.
"""

from __future__ import annotations

from collections.abc import Sequence

from max.dtype import DType
from max.graph import TensorType, TensorValue, ops

# Pure reductions: true reductions defined on an arbitrary axis.
PURE_REDUCTIONS = (
    "reduce_sum",
    "reduce_max",
    "reduce_min",
    "reduce_mean",
    "reduce_product",
    "argmax",
    "argmin",
    "reduce_min_and_max",
)
# Norm-type ops: defined on the last axis.
NORM_OPS = (
    "softmax",
    "logsoftmax",
    "layer_norm",
    "rms_norm",
    "row_mean_of_squares",
)

LAYER_NORM_EPS = 1e-5
RMS_NORM_EPS = 1e-6

# Ops that are just `ops.<fn>(x, axis=axis)` with no extra plumbing.
_SIMPLE_OPS = {
    "reduce_sum": ops.sum,
    "reduce_max": ops.max,
    "reduce_min": ops.min,
    "reduce_mean": ops.mean,
    "reduce_product": ops.prod,
    "argmax": ops.argmax,
    "argmin": ops.argmin,
    "softmax": ops.softmax,
    "logsoftmax": ops.logsoftmax,
}


def build_reduction(
    op: str,
    x: TensorValue,
    axis: int,
    *,
    weights: Sequence[TensorValue] = (),
) -> TensorValue:
    """Builds the named row-wise reduction over ``axis`` of ``x``.

    Args:
        op: One of :data:`PURE_REDUCTIONS` or :data:`NORM_OPS`.
        x: The input tensor value.
        axis: The reduction axis (may be negative).
        weights: The norm parameters when ``op`` needs them -- ``(gamma, beta)``
            for ``layer_norm`` and ``(weight,)`` for ``rms_norm``. The caller
            supplies them however it likes (graph inputs or baked constants);
            unused for every other op.

    Returns:
        The reduction output. Its shape, dtype and device follow from ``x``.
    """
    if op in _SIMPLE_OPS:
        return _SIMPLE_OPS[op](x, axis=axis)
    if op == "reduce_min_and_max":
        # Reduced axis becomes length 2 ([min, max]) in the same rank as x.
        norm_axis = axis + x.rank if axis < 0 else axis
        out_shape = [
            2 if i == norm_axis else dim for i, dim in enumerate(x.shape)
        ]
        return ops.custom(
            "mo.reduce.reduce_min_and_max",
            device=x.device,
            values=[x],
            out_types=[TensorType(x.dtype, out_shape, device=x.device)],
            parameters={"axis": axis},
        )[0].tensor
    if op == "row_mean_of_squares":
        return ops.custom(
            "mo.reduce.row_mean_of_squares",
            device=x.device,
            values=[x],
            out_types=[
                TensorType(DType.float32, [x.shape[0], 1], device=x.device)
            ],
        )[0].tensor
    if op == "layer_norm":
        gamma, beta = weights
        return ops.layer_norm(x, gamma, beta, epsilon=LAYER_NORM_EPS)
    if op == "rms_norm":
        (weight,) = weights
        return ops.rms_norm(x, weight, epsilon=RMS_NORM_EPS)
    raise ValueError(f"unknown reduction {op!r}")
