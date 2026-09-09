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
"""Op implementation for band_part."""

from __future__ import annotations

from max._core.dialects import kgen, rmo
from max.dtype import DType

from ..dim import StaticDim
from ..graph import Graph
from ..type import DeviceRef
from ..value import TensorValue, TensorValueLike
from .constant import constant


def band_part(
    x: TensorValueLike,
    num_lower: int | None = None,
    num_upper: int | None = None,
    exclude: bool = False,
) -> TensorValue:
    """Masks out everything except a diagonal band of an input matrix.

    Copies a tensor, setting everything outside the central diagonal band of
    each sub-matrix to zero. All axes except the last two are treated as
    batch dimensions; the last two axes define the ``M x N`` sub-matrices.

    A sub-matrix element at row ``m`` and column ``n`` is kept when
    ``(m - n) <= num_lower`` (or ``num_lower`` is ``None``) and
    ``(n - m) <= num_upper`` (or ``num_upper`` is ``None``). With
    ``exclude=True`` the selection inverts: elements inside the band are
    zeroed and elements outside are kept.

    .. code-block:: python

        import numpy as np
        from max.driver import CPU
        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, TensorType, ops

        input_type = TensorType(
            DType.float32, [4, 4], device=DeviceRef.CPU()
        )
        with Graph("band_part", input_types=[input_type]) as graph:
            x = graph.inputs[0].tensor
            # Tridiagonal band: one sub- and one super-diagonal.
            tridiag = ops.band_part(x, num_lower=1, num_upper=1)
            # Lower triangle: all sub-diagonals, no super-diagonals.
            lower = ops.band_part(x, num_lower=None, num_upper=0)
            graph.output(tridiag, lower)

        model = InferenceSession(devices=[CPU()]).load(graph)
        mat = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
        tri_out, lower_out = model.execute(mat)
        # tri_out:
        #   [[ 1.,  2.,  0.,  0.],
        #    [ 5.,  6.,  7.,  0.],
        #    [ 0., 10., 11., 12.],
        #    [ 0.,  0., 15., 16.]]
        # lower_out:
        #   [[ 1.,  0.,  0.,  0.],
        #    [ 5.,  6.,  0.,  0.],
        #    [ 9., 10., 11.,  0.],
        #    [13., 14., 15., 16.]]

    .. invisible-code-block: python

        rows, cols = np.indices((4, 4))

        tri_mask = (rows - cols <= 1) & (cols - rows <= 1)
        np.testing.assert_allclose(
            tri_out.to_numpy(), np.where(tri_mask, mat, 0.0)
        )

        lower_mask = cols - rows <= 0
        np.testing.assert_allclose(
            lower_out.to_numpy(), np.where(lower_mask, mat, 0.0)
        )

    Args:
        x: The input tensor to mask.
        num_lower: The number of sub-diagonal bands to keep. If ``None``,
            the entire lower triangle is kept.
        num_upper: The number of super-diagonal bands to keep. If ``None``,
            the entire upper triangle is kept.
        exclude: If ``True``, invert the band selection so that elements
            inside the band are zeroed and elements outside are kept.

    Returns:
        A symbolic tensor value with the same shape as ``x``. Elements
        outside the selected band are zero; all other values are copied
        from ``x``.

    Raises:
        ValueError: If the input tensor rank is less than 2, or if
            ``num_lower`` or ``num_upper`` are out of bounds for a
            statically known dimension.
    """
    x = TensorValue(x)
    num_lower = -1 if num_lower is None else num_lower
    num_upper = -1 if num_upper is None else num_upper

    if num_lower < -1:
        raise ValueError(f"{num_lower=} must be non-negative")
    if num_upper < -1:
        raise ValueError(f"{num_upper=} must be non-negative")

    if x.rank < 2:
        raise ValueError(
            f"Input tensor {x.shape=} must have at least 2 dimensions"
        )

    # Check for out-of-bounds values for known static dimensions.
    # - m is the "vertical" dimension, and n is the "horizontal" dimension, visually
    # - num_lower is how far "down", so it is compared against m
    # - num_upper is how far "right", so it is compared against n
    *_, m, n = x.shape
    if isinstance(m, StaticDim) and num_lower >= int(m):
        raise ValueError(
            f"{num_lower=} is out of bounds for dimension size {int(m)}"
        )
    if isinstance(n, StaticDim) and num_upper >= int(n):
        raise ValueError(
            f"{num_upper=} is out of bounds for dimension size {int(n)}"
        )

    return Graph.current._add_op_generated(
        rmo.MoLinalgBandPartOp,
        x.type,
        x,
        constant(num_lower, DType.int64, DeviceRef.CPU()),
        constant(num_upper, DType.int64, DeviceRef.CPU()),
        constant(exclude, DType.bool, DeviceRef.CPU()),
        kgen.ParamDeclArrayAttr([]),
    )[0].tensor
