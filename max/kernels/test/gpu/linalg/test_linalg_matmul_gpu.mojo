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


from max.gpu.host import DeviceContext
from linalg.matmul import matmul
from layout import TileTensor, CoordLike, Coord, Idx, row_major
from linalg.matmul.gpu import _matmul_gpu
from std.testing import assert_almost_equal


def _linspace_fill[dtype: DType](mut buff: TileTensor[mut=True, dtype, ...]):
    for i in range(buff.num_elements()):
        buff.raw_store(i, Scalar[dtype](i))


def _get_test_name[
    dtype: DType
](shape_c: Coord, shape_a: Coord, shape_b: Coord,) -> String:
    return String(
        "test-case(",
        dtype,
        ") : ",
        shape_c[0].value(),
        (
            "_dynamic"
            + " x "
            + String(shape_b[1]) if not shape_c.element_types[
                0
            ].is_static_value else " x "
            + String(shape_b[1])
        ),
        (
            "_dynamic"
            + " x "
            + String(shape_a[1]) if not shape_b.element_types[
                1
            ].is_static_value else " x "
            + String(shape_a[1])
        ),
        "_dynamic" if not shape_a.element_types[1].is_static_value else "",
        ", ... ",
    )


def matmul_test_case[
    dtype: DType,
](shape_c: Coord, shape_a: Coord, shape_b: Coord, ctx: DeviceContext,) raises:
    print(
        _get_test_name[dtype](shape_c, shape_a, shape_b),
        end=" ",
    )

    var mat_a_dev = ctx.enqueue_create_buffer[dtype](Int(shape_a.product()))
    var mat_a_tensor = TileTensor(mat_a_dev, row_major(shape_a))
    var mat_b_dev = ctx.enqueue_create_buffer[dtype](Int(shape_b.product()))
    var mat_b_tensor = TileTensor(mat_b_dev, row_major(shape_b))
    var mat_c_dev = ctx.enqueue_create_buffer[dtype](Int(shape_c.product()))
    var mat_c_tensor = TileTensor(mat_c_dev, row_major(shape_c))

    var mat_a_host_buf = ctx.enqueue_create_host_buffer[dtype](
        Int(shape_a.product())
    )
    var mat_a_host = TileTensor(mat_a_host_buf, row_major(shape_a))
    var mat_b_host_buf = ctx.enqueue_create_host_buffer[dtype](
        Int(shape_b.product())
    )
    var mat_b_host = TileTensor(mat_b_host_buf, row_major(shape_b))
    var mat_c_host_buf = ctx.enqueue_create_host_buffer[dtype](
        Int(shape_c.product())
    )
    var mat_c_host = TileTensor(mat_c_host_buf, row_major(shape_c))
    var mat_c_ref_host_buf = ctx.enqueue_create_host_buffer[dtype](
        Int(shape_c.product())
    )
    var mat_c_ref_host = TileTensor(mat_c_ref_host_buf, row_major(shape_c))

    _linspace_fill(mat_a_host)
    _linspace_fill(mat_b_host)

    ctx.enqueue_copy(mat_a_dev, mat_a_host._storage)
    ctx.enqueue_copy(mat_b_dev, mat_b_host._storage)

    _matmul_gpu[use_tensor_core=True](
        mat_c_tensor, mat_a_tensor, mat_b_tensor, ctx
    )

    ctx.enqueue_copy(mat_c_host._storage, mat_c_dev)
    ctx.synchronize()

    # FIXME: We should run a reference gpu matmul, the reference should also
    # support applying the epilogue on the final result.
    matmul(mat_c_ref_host, mat_a_host, mat_b_host)

    comptime assert type_of(shape_c[0]).DTYPE.is_integral()
    comptime assert type_of(shape_c[1]).DTYPE.is_integral()

    for m in range(shape_c[0].value()):
        for n in range(shape_c[1].value()):
            comptime assert mat_c_ref_host.flat_rank == 2
            assert_almost_equal(mat_c_ref_host[m, n], mat_c_host[m, n])


def create_matmul_test_case[
    MType: CoordLike, NType: CoordLike, KType: CoordLike, //, dtype: DType
](ctx: DeviceContext, m: MType, n: NType, k: KType) raises:
    matmul_test_case[.float32,](Coord(m, n), Coord(m, k), Coord(k, n), ctx)


def main() raises:
    with DeviceContext() as ctx:
        create_matmul_test_case[.float32](ctx, Int(8), Idx[8], Idx[4])
        create_matmul_test_case[.float32](ctx, Int(16), Idx[16], Idx[8])
