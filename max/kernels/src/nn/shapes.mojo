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
"""Provides output-shape computation utilities for sliding-window operations such as pooling and convolution."""

from std.math import ceildiv
from std.math.uutils import ufloordiv, umod
from std.utils import IndexList

from layout import Coord, CoordLike


# Static-divisor n-th subvolume start-index decomposition.
#
# The static-folding counterpart of
# `_get_start_indices_of_nth_subvolume[subvolume_rank](n, shape)`
# (`max.algorithm.functional`): same semantics for every `subvolume_rank` (the
# trailing `subvolume_rank` dims are the subvolume set by the caller; dims
# `1..rank-subvolume_rank-1` are decomposed; dim 0 is the final quotient), and
# bit-identical to it for the shapes that reach it. The difference is only in
# the shape argument: here it arrives as a `Coord` that carries BOTH the static
# dims (in its type) AND the runtime values, instead of a plain `IndexList`.
#
# For dims whose extent is statically known in the `Coord` type
# (`element_types[i].is_static_value`), the divisor becomes a compile-time
# literal, so the `divmod` strength-reduces to a magic-multiply + shift
# (verified: SASS `IMAD.HI`/`SHF`, no `IDIV`/`MUFU.RCP`) instead of the runtime
# Newton-reciprocal divide that an `IndexList` divisor forces. Dynamic dims fall
# back to the runtime value carried in the `Coord`, matching the MAX baseline.
#
# The flat index `n` stays runtime, so the *result* indices are runtime; only
# the DIVISORS fold. `rank` and the static dims are inferred from the `Coord`
# argument's type, so callers pass the `Coord` (e.g. `layout.shape_coord()`)
# directly without spelling out the type/rank params; `subvolume_rank` defaults
# to 1 (the row translation). The `Coord` is computed in-kernel (e.g. from the
# on-device layout) or copy-captured at the call site; only its dynamic-dim leaf
# values are read at runtime here.
#
# Uses unconditional `umod`/`ufloordiv` (unlike `_get_start_indices_of_nth_subvolume`,
# which picks signed/unsigned off `IndexList._int_type`): a flat index and
# tensor extents are always non-negative, so unsigned is not just safe but
# required for the compile-time-divisor magic-multiply fold to fire.
#
# Shared by the normalization (rms_norm / layer_norm) and concat kernels; lives
# here rather than in either kernel module so neither has to import the other.
@always_inline
def _get_start_indices_of_nth_subvolume_static[
    element_types: TypeList[Trait=CoordLike, ...], //, subvolume_rank: Int = 1
](n: Int, shape: Coord[*element_types]) -> IndexList[shape.rank]:
    comptime rank = shape.rank
    comptime assert (
        subvolume_rank <= rank
    ), "subvolume rank cannot be greater than indices rank"
    comptime assert subvolume_rank >= 0, "subvolume rank must be non-negative"

    var res = IndexList[rank]()

    # Match `_get_start_indices_of_nth_subvolume`'s fast paths so behavior is
    # bit-identical for the shapes that reach them.
    comptime if rank == 2 and subvolume_rank == 1:
        res[0] = n
        return res

    comptime if rank - 1 == subvolume_rank:
        res[0] = n
        return res

    comptime if rank == subvolume_rank:
        return res

    var curr = n

    comptime for i in reversed(range(1, rank - subvolume_rank)):
        comptime ElemT = element_types[i]
        comptime if ElemT.is_static_value:
            # Compile-time divisor -> magic-multiply + shift (no `IDIV`).
            comptime divisor = ElemT.static_value
            res[i] = umod(curr, divisor)
            curr = ufloordiv(curr, divisor)
        else:
            # Dynamic dim: read the divisor from the runtime leaf value carried
            # in the `Coord`. This path emits a runtime divide, same as the
            # `_get_start_indices_of_nth_subvolume` baseline.
            var divisor = Int(shape[i].value())
            res[i] = umod(curr, divisor)
            curr = ufloordiv(curr, divisor)

    res[0] = curr
    return res


@always_inline("nodebug")
def get_sliding_window_out_dim[
    ceil_mode: Bool = False,
](in_dim: Int, ft_dim: Int, dilation: Int, stride: Int, pad: Int) -> Int:
    """
    Return output dimension for a sliding window operation along some dimension.

    Parameters:
        ceil_mode: Define rounding mode for shape calculation.

    Args:
        in_dim: The size of the input dimension.
        ft_dim: The size of the corresponding filter dimension.
        dilation: The dilation for the sliding window operation.
        stride: The stride for the sliding window operation.
        pad: The total padding for the sliding window operation.

    Returns:
        The size of the output dimension.

    """

    comptime if ceil_mode:
        return 1 + ceildiv(in_dim + pad - (1 + dilation * (ft_dim - 1)), stride)
    else:
        return 1 + (in_dim + pad - (1 + dilation * (ft_dim - 1))) // stride
