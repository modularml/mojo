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
#
# CPU unit test for the reusable fuzz harness (_fuzz.mojo). Validates the
# boundary-aware shape generator and the value-distribution fills -- especially
# the NaN/Inf special-value injection, which is the harness's main new
# numerical content. No GPU required.

from std.random import seed
from std.testing import assert_equal, assert_true
from std.utils.numerics import isinf, isnan

from _fuzz import (
    NUM_VALUE_DISTS,
    VD_SPECIALS,
    boundary_int,
    fill_all_equal,
    fill_by_dist,
    fill_sparse,
    fill_uniform,
    fill_with_specials,
    value_dist_name,
)


def test_boundary_int_in_range() raises:
    seed(1)
    for _ in range(2000):
        var v = boundary_int(4, 100, 16)
        assert_true(v >= 4 and v <= 100, "boundary_int out of [lo, hi]")
    # Degenerate range collapses to lo.
    assert_equal(boundary_int(7, 7, 16), 7)
    assert_equal(boundary_int(9, 3, 16), 9)


def test_boundary_int_hits_boundaries() raises:
    seed(2)
    # Over many draws it must land on the size-1 edge and on tile multiples,
    # which a uniform generator over [1, 4096] essentially never would.
    var saw_lo = False
    var saw_tile = False
    var saw_2tile = False
    for _ in range(4000):
        var v = boundary_int(1, 4096, 128)
        if v == 1:
            saw_lo = True
        if v == 128:
            saw_tile = True
        if v == 256:
            saw_2tile = True
    assert_true(saw_lo, "should hit lo edge")
    assert_true(saw_tile, "should hit tile")
    assert_true(saw_2tile, "should hit 2*tile")


def test_fill_uniform_in_range() raises:
    seed(3)
    var buf = Array[Float32, 512](uninitialized=True)
    var span = Span(buf)
    fill_uniform(span, -2.0, 3.0)
    for i in range(len(span)):
        assert_true(
            span[i] >= -2.0 and span[i] < 3.0, "uniform value out of range"
        )


def test_fill_all_equal() raises:
    var buf = Array[Float32, 64](uninitialized=True)
    var span = Span(buf)
    fill_all_equal(span, 3.5)
    for i in range(len(span)):
        assert_equal(span[i], Float32(3.5))


def test_fill_sparse_mostly_zero() raises:
    seed(4)
    var buf = Array[Float32, 4096](uninitialized=True)
    var span = Span(buf)
    fill_sparse(span, density=0.05)
    var nonzero = 0
    for i in range(len(span)):
        if span[i] != Float32(0):
            nonzero += 1
    # Expect ~5% nonzero; allow generous slack but assert it is sparse.
    assert_true(nonzero > 0, "sparse fill produced all zeros")
    assert_true(nonzero < len(span) // 2, "sparse fill is not sparse")


def test_fill_specials_injects_nan_and_inf() raises:
    seed(5)
    var buf = Array[Float32, 4096](uninitialized=True)
    var span = Span(buf)
    fill_with_specials(span, density=0.5)
    var has_nan = False
    var has_inf = False
    for i in range(len(span)):
        if isnan(span[i]):
            has_nan = True
        if isinf(span[i]):
            has_inf = True
    assert_true(has_nan, "specials fill must inject at least one NaN")
    assert_true(has_inf, "specials fill must inject at least one Inf")


def test_fill_by_dist_all_compile_and_run() raises:
    seed(6)
    for dist in range(NUM_VALUE_DISTS):
        var buf = Array[Float32, 128](uninitialized=True)
        var span = Span(buf)
        fill_by_dist(span, dist)
        # Just exercising every dispatch arm; name lookup must not be empty.
        assert_true(value_dist_name(dist).byte_length() > 0, "dist name empty")


def main() raises:
    test_boundary_int_in_range()
    test_boundary_int_hits_boundaries()
    test_fill_uniform_in_range()
    test_fill_all_equal()
    test_fill_sparse_mostly_zero()
    test_fill_specials_injects_nan_and_inf()
    test_fill_by_dist_all_compile_and_run()
    print("all _fuzz harness tests passed")
