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
"""Equivalence check for the static-folding subvolume decomposition helper.

`_get_start_indices_of_nth_subvolume_static` (in `nn.shapes`) is the
static-divisor-folding counterpart of
`_get_start_indices_of_nth_subvolume` (in `max.algorithm.functional`): for a
`Coord` whose outer dims are statically known, its per-element `divmod` folds to
a magic-multiply + shift, but the *result* must stay bit-identical to the
MAX algorithm baseline for every shape and `subvolume_rank`.

This is a pure host-side check (no device launch): it compares the two
functions' `IndexList` outputs directly. It covers `subvolume_rank` in
{0, 1, 2}, ranks 2/3/4, over a sweep of flat indices, and runs each shape both
as an all-static `Coord` (the fold fires) and an all-dynamic `Coord` (degrades
to the runtime-divide path). All four must agree with the MAX reference.
"""

from max.algorithm.functional import _get_start_indices_of_nth_subvolume

from layout import Coord, row_major
from nn.shapes import _get_start_indices_of_nth_subvolume_static

from std.testing import assert_equal
from std.utils import IndexList


def _check[
    rank: Int, //, subvolume_rank: Int
](
    static_shape: Coord,
    dynamic_shape: Coord,
    ref_shape: IndexList[rank],
    n_max: Int,
) raises:
    """Assert both `Coord` forms match the stdlib reference for all `n < n_max`.
    """
    for n in range(n_max):
        var expected = _get_start_indices_of_nth_subvolume[subvolume_rank](
            n, ref_shape
        )
        var got_static = _get_start_indices_of_nth_subvolume_static[
            subvolume_rank=subvolume_rank
        ](n, static_shape)
        var got_dynamic = _get_start_indices_of_nth_subvolume_static[
            subvolume_rank=subvolume_rank
        ](n, dynamic_shape)
        for d in range(rank):
            assert_equal(
                got_static[d],
                expected[d],
                msg="static-Coord fold diverged from stdlib reference",
            )
            assert_equal(
                got_dynamic[d],
                expected[d],
                msg="dynamic-Coord path diverged from stdlib reference",
            )


def test_rank2() raises:
    # rank-2: subvolume_rank 1 (row fast path) and 2 (whole-tensor fast path).
    comptime d0 = 5
    comptime d1 = 3
    var ref_il = IndexList[2](d0, d1)
    var static_c = row_major[d0, d1]().shape_coord()
    var dyn_c = row_major(Coord(ref_il)).shape_coord()
    _check[subvolume_rank=1](static_c, dyn_c, ref_il, d0 * d1)
    _check[subvolume_rank=2](static_c, dyn_c, ref_il, d0 * d1)


def test_rank3() raises:
    # rank-3: subvolume_rank 0 (per-element), 1 (row), 2 (rank-1==K fast path).
    comptime d0 = 4
    comptime d1 = 3
    comptime d2 = 6
    var ref_il = IndexList[3](d0, d1, d2)
    var static_c = row_major[d0, d1, d2]().shape_coord()
    var dyn_c = row_major(Coord(ref_il)).shape_coord()
    _check[subvolume_rank=0](static_c, dyn_c, ref_il, d0 * d1 * d2)
    _check[subvolume_rank=1](static_c, dyn_c, ref_il, d0 * d1 * d2)
    _check[subvolume_rank=2](static_c, dyn_c, ref_il, d0 * d1 * d2)


def test_rank4() raises:
    # rank-4: the general decomposition loop actually runs for K in {0, 1}.
    comptime d0 = 2
    comptime d1 = 7
    comptime d2 = 3
    comptime d3 = 5
    var ref_il = IndexList[4](d0, d1, d2, d3)
    var static_c = row_major[d0, d1, d2, d3]().shape_coord()
    var dyn_c = row_major(Coord(ref_il)).shape_coord()
    _check[subvolume_rank=0](static_c, dyn_c, ref_il, d0 * d1 * d2 * d3)
    _check[subvolume_rank=1](static_c, dyn_c, ref_il, d0 * d1 * d2 * d3)
    _check[subvolume_rank=2](static_c, dyn_c, ref_il, d0 * d1 * d2 * d3)


def main() raises:
    test_rank2()
    test_rank3()
    test_rank4()
