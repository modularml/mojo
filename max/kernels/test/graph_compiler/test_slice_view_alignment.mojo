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
"""Tests the alignment a `mo.slice` view reports for itself.

The reported alignment promises that every address the view can produce is a
multiple of it, and consumers turn it straight into a vector-load width.
`get_view_alignment` is a pure function of strides, starts and steps, so these
cases pin it directly, with no device and no op execution.

Regression test for GEX-4058.
"""

from builtin_kernels.gather_scatter import Slice

from layout import IntTuple, UNKNOWN_VALUE

from std.testing import TestSuite, assert_equal

# A freshly allocated buffer is generously aligned. Every case starts here and
# can only shrink it.
comptime BUFFER_ALIGN = 256

comptime NO_OFFSET_2 = IntTuple(0, 0)
comptime UNIT_STEP_2 = IntTuple(1, 1)
comptime NO_OFFSET_3 = IntTuple(0, 0, 0)
comptime UNIT_STEP_3 = IntTuple(1, 1, 1)


def test_column_slice_bounded_by_row_pitch() raises:
    """`x[:, :8]` of a `[4, 10]` float32 tensor: rows are 40 bytes apart."""
    comptime strides = IntTuple(10, 1)
    var got = Slice.get_view_alignment[
        2, DType.float32, strides, NO_OFFSET_2, UNIT_STEP_2
    ](BUFFER_ALIGN)

    assert_equal(got, 8)


def test_aligned_row_pitch_is_a_no_op() raises:
    """`[4, 64]` float32 rows are 256 bytes, so the pitch costs nothing."""
    comptime strides = IntTuple(64, 1)
    var got = Slice.get_view_alignment[
        2, DType.float32, strides, NO_OFFSET_2, UNIT_STEP_2
    ](BUFFER_ALIGN)

    # Guards the vectorization the pitch rule exists to enable: clamping every
    # slice to the innermost stride would report 4.
    assert_equal(got, 256)


def test_moe_gate_row_pitch() raises:
    """Selecting 256 routed experts out of 258 logits: 1032-byte rows."""
    comptime strides = IntTuple(258, 1)
    var got = Slice.get_view_alignment[
        2, DType.float32, strides, NO_OFFSET_2, UNIT_STEP_2
    ](BUFFER_ALIGN)

    assert_equal(got, 8)


def test_step_scales_the_pitch() raises:
    """A step of 2 doubles what one move along that dimension covers."""
    comptime strides = IntTuple(10, 1)
    comptime steps = IntTuple(2, 1)
    var got = Slice.get_view_alignment[
        2, DType.float32, strides, NO_OFFSET_2, steps
    ](BUFFER_ALIGN)

    assert_equal(got, 16)


def test_innermost_start_offset_shrinks_alignment() raises:
    """`x[:, 2:]` of `[4, 64]` float32: an aligned pitch, an offset base."""
    comptime strides = IntTuple(64, 1)
    comptime starts = IntTuple(0, 2)
    var got = Slice.get_view_alignment[
        2, DType.float32, strides, starts, UNIT_STEP_2
    ](BUFFER_ALIGN)

    # The pitch is a no-op at 256 bytes, isolating the start-offset path.
    assert_equal(got, 8)


def test_dense_rank3_outer_strides_are_dominated() raises:
    """On a dense source each outer stride is a multiple of the next inner."""
    comptime strides = IntTuple(30, 10, 1)
    var got = Slice.get_view_alignment[
        3, DType.float32, strides, NO_OFFSET_3, UNIT_STEP_3
    ](BUFFER_ALIGN)

    assert_equal(got, 8)


def test_permuted_rank3_every_dimension_contributes() raises:
    """Strides stop nesting once a view is permuted, so all of them matter."""
    comptime strides = IntTuple(24, 64, 1)
    var got = Slice.get_view_alignment[
        3, DType.float32, strides, NO_OFFSET_3, UNIT_STEP_3
    ](BUFFER_ALIGN)

    # Folding only `stride[rank - 2]` would report 256, since 64 * 4 is already
    # a multiple of it. Dimension 0 is what brings this down.
    assert_equal(got, 32)


def test_dynamic_stride_bails_to_one() raises:
    """A stride unknown at compile time admits no promise at all."""
    comptime strides = IntTuple(UNKNOWN_VALUE, 1)
    var got = Slice.get_view_alignment[
        2, DType.float32, strides, NO_OFFSET_2, UNIT_STEP_2
    ](BUFFER_ALIGN)

    assert_equal(got, 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
