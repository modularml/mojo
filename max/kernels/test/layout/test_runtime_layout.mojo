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

from layout import (
    IntTuple,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
    UNKNOWN_VALUE,
)
from layout.layout import coalesce as coalesce_layout
from layout.layout import crd2idx
from layout.runtime_layout import coalesce, make_layout
from std.testing import assert_equal


def test_runtime_layout_const() raises:
    print("== test_runtime_layout_const")

    comptime shape = IntTuple(UNKNOWN_VALUE, 8)
    comptime stride = IntTuple(8, 1)

    comptime layout = Layout(shape, stride)

    var shape_runtime = RuntimeTuple[layout.shape, element_type=DType.uint32](
        16, 8
    )
    var stride_runtime = RuntimeTuple[
        layout.stride, element_type=DType.uint32
    ]()

    var layout_r = RuntimeLayout[
        layout, element_type=.uint32, linear_idx_type=.uint32
    ](shape_runtime, stride_runtime)

    assert_equal(String(materialize[layout_r.layout]()), "((-1, 8):(8, 1))")
    assert_equal(String(layout_r), "((16, 8):(8, 1))")


def test_static_and_dynamic_size() raises:
    print("== test_static_and_dynamic_size")
    comptime d_layout = Layout(IntTuple(UNKNOWN_VALUE, 4), IntTuple(4, 1))
    var layout = RuntimeLayout[
        d_layout, element_type=.uint32, linear_idx_type=.uint32
    ](
        RuntimeTuple[d_layout.shape, element_type=.uint32](4, 8),
        RuntimeTuple[d_layout.stride, element_type=.uint32](4, 8),
    )
    assert_equal(layout.size(), 32)


def test_tiled_layout_indexing() raises:
    print("== test_tiled_layout_indexing")

    comptime shape = IntTuple(IntTuple(2, 2), IntTuple(2, 2))
    comptime stride = IntTuple(IntTuple(1, 8), IntTuple(2, 4))

    comptime d_tuple = IntTuple(
        IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE),
        IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE),
    )
    comptime d_layout = Layout(d_tuple, d_tuple)

    var layout = RuntimeLayout[
        d_layout, element_type=.uint32, linear_idx_type=.uint32
    ](
        RuntimeTuple[d_layout.shape, element_type=.uint32](2, 2, 2, 2),
        RuntimeTuple[d_layout.stride, element_type=.uint32](1, 8, 2, 4),
    )

    for ii in range(2):
        for i in range(2):
            for jj in range(2):
                for j in range(2):
                    assert_equal(
                        crd2idx(
                            IntTuple(IntTuple(ii, i), IntTuple(jj, j)),
                            shape,
                            stride,
                        ),
                        Int(layout(RuntimeTuple[d_tuple](ii, i, jj, j))),
                    )


def test_tiled_layout_indexing_linear_idx() raises:
    print("== test_tiled_layout_indexing_linear_idx")

    comptime shape = IntTuple(IntTuple(2, 2), IntTuple(2, 2))
    comptime stride = IntTuple(IntTuple(1, 8), IntTuple(2, 4))

    comptime d_tuple = IntTuple(
        IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE),
        IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE),
    )
    comptime d_layout = Layout(d_tuple, d_tuple)

    var layout = RuntimeLayout[
        d_layout, element_type=.uint32, linear_idx_type=.uint32
    ](
        RuntimeTuple[d_layout.shape, element_type=.uint32](2, 2, 2, 2),
        RuntimeTuple[d_layout.stride, element_type=.uint32](1, 8, 2, 4),
    )

    for i in range(16):
        assert_equal(
            crd2idx(
                i,
                shape,
                stride,
            ),
            Int(layout(RuntimeTuple[IntTuple(UNKNOWN_VALUE)](i))),
        )


def test_sublayout_indexing() raises:
    print("== test_sublayout_indexing")
    comptime layout_t = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime layout = RuntimeLayout[
        layout_t, element_type=.uint32, linear_idx_type=.uint32
    ](
        RuntimeTuple[layout_t.shape, element_type=.uint32](8, 4),
        RuntimeTuple[layout_t.stride, element_type=.uint32](4, 1),
    )
    assert_equal(String(layout.sublayout[0]()), "(8:4)")
    assert_equal(String(layout.sublayout[1]()), "(4:1)")


def test_coalesce() raises:
    print("== test_coalesce")
    comptime layout_t = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    var layout = RuntimeLayout[
        layout_t, element_type=.uint32, linear_idx_type=.uint32
    ](
        RuntimeTuple[layout_t.shape, element_type=.uint32](8, 1),
        RuntimeTuple[layout_t.stride, element_type=.uint32](1, 1),
    )
    assert_equal(String(coalesce(layout)), "((8, 1):(1, 1))")
    assert_equal(
        String(coalesce_layout(materialize[layout_t]())), "((-1, -1):(-1, 1))"
    )

    comptime layout_t_2 = Layout(
        IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE, 8, 1),
        IntTuple(UNKNOWN_VALUE, 8, 1, 1),
    )
    var layout_2 = RuntimeLayout[
        layout_t_2, element_type=.uint32, linear_idx_type=.uint32
    ](
        RuntimeTuple[layout_t_2.shape, element_type=.uint32](32, 16, 8, 1),
        RuntimeTuple[layout_t_2.stride, element_type=.uint32](16, 8, 1, 1),
    )

    assert_equal(String(coalesce(layout_2)), "((32, 16, 8):(16, 8, 1))")
    assert_equal(
        String(coalesce_layout(materialize[layout_t_2]())),
        "((-1, -1, 8):(-1, 8, 1))",
    )


def test_make_layout() raises:
    print("== test_make_layout")
    comptime layout_t = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    var l_a = RuntimeLayout[
        layout_t, element_type=.uint32, linear_idx_type=.uint32
    ](
        RuntimeTuple[layout_t.shape, element_type=.uint32](2, 2),
        RuntimeTuple[layout_t.stride, element_type=.uint32](2, 1),
    )
    var l_b = RuntimeLayout[
        layout_t, element_type=.uint32, linear_idx_type=.uint32
    ](
        RuntimeTuple[layout_t.shape, element_type=.uint32](4, 4),
        RuntimeTuple[layout_t.stride, element_type=.uint32](4, 1),
    )
    assert_equal(
        String(make_layout(l_a, l_b)), "(((2, 2), (4, 4)):((2, 1), (4, 1)))"
    )


def test_large_layout_linear_index() raises:
    print("== test_large_layout_linear_index")
    # Shape size exceeds Int32 max but stays within UInt32 max.
    # This ensures the runtime layout uses Int64 for linear indexing.
    comptime large_layout = Layout.row_major(65536, 57344)
    comptime tensor_type = LayoutTensor[.uint8, large_layout, _]

    var shape = tensor_type.RuntimeLayoutType.ShapeType(65536, 57344)
    var stride = tensor_type.RuntimeLayoutType.StrideType(57344, 1)
    var runtime_layout = tensor_type.RuntimeLayoutType(shape, stride)

    var idx = runtime_layout(
        RuntimeTuple[IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE)](65535, 57343)
    )
    var expected = Int(65535) * 57344 + 57343
    assert_equal(Int(idx), expected)


def main() raises:
    test_runtime_layout_const()
    test_static_and_dynamic_size()
    test_tiled_layout_indexing()
    test_tiled_layout_indexing_linear_idx()
    test_sublayout_indexing()
    test_coalesce()
    test_make_layout()
    test_large_layout_linear_index()
