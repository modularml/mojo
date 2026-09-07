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

from std.math import ceildiv, isclose
from std.memory import alloc
from std.random import rand, random_float64
from std.sys import size_of

from max.algorithm import sync_parallelize
from layout import (
    IntTuple,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
    UNKNOWN_VALUE,
    lt_to_tt,
)
from quantization.qmatmul import matmul_qint4, matmul_qint4_pack_b
from quantization.qmatmul_k import (
    _block_Q4_K,
    _block_Q6_K,
    _block_QK_K,
    matmul_Q4_K,
    matmul_Q4_K_pack_b,
    matmul_Q6_K,
    matmul_Q6_K_pack_b,
)

from std.utils.index import Index


def fill_random[dtype: DType](mut array: Array[Scalar[dtype], ...]):
    rand(array.unsafe_ptr(), len(array))


def random_float16(min: Float64 = 0, max: Float64 = 1) -> Float16:
    # Avoid pulling in a __truncdfhf2 dependency for a float64->float16
    # conversion by casting through float32 first.
    return random_float64(min=min, max=max).cast[.float32]().cast[.float16]()


def quantize_a_Q8[
    group_size: Int
](a: ImmPointer[Float32, ...], a_quant: MutPointer[Int8, ...]) -> Float32:
    var fp_data = a.load[width=group_size]()
    var max_value = abs(fp_data).reduce_max()
    var multiplier = 127.0 / max_value if max_value != 0.0 else 0.0
    var scale = (max_value / 127.0).cast[.float32]()
    var quant_data = round(fp_data * multiplier).cast[.int8]()

    a_quant.store(quant_data)
    return scale


def dot_product_QK_K[
    b_scales_type: DType,
    *,
    group_size: Int,
    b_zero_point: Int32 = 0,
](
    a_quant_data: ImmPointer[Int8, ...],
    b_quant_data: ImmPointer[UInt8, ...],
    b_scales: ImmPointer[Scalar[b_scales_type], ...],
) -> Int32:
    var sum: Int32 = 0
    for i in range(_block_QK_K.quantized_k):
        sum += (
            a_quant_data[i].cast[.int32]()
            * (b_quant_data[i].cast[.int32]() - b_zero_point)
            * b_scales[i // group_size].cast[.int32]()
        )
    return sum


trait QuantizedGemm:
    @staticmethod
    def k_group_size() -> Int:
        ...

    @staticmethod
    def build_b_buffer(
        N: Int, K: Int
    ) -> LayoutTensor[.uint8, Layout.row_major[2](), MutAnyOrigin]:
        ...

    @staticmethod
    def pack_b_buffer(
        b: LayoutTensor[mut=True, .uint8, Layout.row_major[2](), _],
        b_packed: LayoutTensor[mut=True, .uint8, Layout.row_major[2](), _],
    ) raises:
        ...

    @staticmethod
    def kernel(
        a: LayoutTensor[.float32, Layout.row_major[2](), _],
        b: LayoutTensor[.uint8, Layout.row_major[2](), _],
        c: LayoutTensor[mut=True, .float32, Layout.row_major[2](), _],
    ):
        ...

    @staticmethod
    def dot_product(
        a: LayoutTensor[.float32, Layout.row_major[2](), _],
        b: LayoutTensor[.uint8, Layout.row_major[2](), _],
        m: Int,
        n: Int,
        k: Int,
    ) -> Float32:
        ...


struct _block_Q4_0:
    comptime group_size = 32

    var base_scale: Float16
    var q_bits: Array[UInt8, Self.group_size // 2]


struct qgemm_Q4_0(QuantizedGemm):
    @staticmethod
    def k_group_size() -> Int:
        return _block_Q4_0.group_size

    @staticmethod
    def build_b_buffer(
        N: Int, K: Int
    ) -> LayoutTensor[.uint8, Layout.row_major[2](), MutUntrackedOrigin]:
        var k_groups = ceildiv(K, Self.k_group_size())
        var b_ptr = alloc[UInt8](N * k_groups * size_of[_block_Q4_0]())
        var block_ptr = b_ptr.bitcast[_block_Q4_0]()

        for _n in range(N):
            for _k in range(k_groups):
                block_ptr[].base_scale = random_float16(max=0.001)
                fill_random(block_ptr[].q_bits)
                block_ptr += 1

        return LayoutTensor[.uint8, Layout.row_major[2]()](
            b_ptr,
            RuntimeLayout[Layout.row_major[2]()].row_major(
                Index(N, k_groups * size_of[_block_Q4_0]())
            ),
        )

    @staticmethod
    def pack_b_buffer(
        b: LayoutTensor[mut=True, .uint8, Layout.row_major[2](), _],
        b_packed: LayoutTensor[mut=True, .uint8, Layout.row_major[2](), _],
    ) raises:
        matmul_qint4_pack_b[_block_Q4_0.group_size](
            lt_to_tt(b), lt_to_tt(b_packed)
        )

    @staticmethod
    def kernel(
        a: LayoutTensor[.float32, Layout.row_major[2](), _],
        b: LayoutTensor[.uint8, Layout.row_major[2](), _],
        c: LayoutTensor[mut=True, .float32, Layout.row_major[2](), _],
    ):
        matmul_qint4[_block_Q4_0.group_size](
            lt_to_tt(a), lt_to_tt(b), lt_to_tt(c)
        )

    @staticmethod
    def dot_product(
        a: LayoutTensor[.float32, Layout.row_major[2](), _],
        b: LayoutTensor[.uint8, Layout.row_major[2](), _],
        m: Int,
        n: Int,
        k: Int,
    ) -> Float32:
        var block_ptr = (b.ptr + b._offset(Index(n, 0))).bitcast[
            _block_Q4_0
        ]() + (k // Self.k_group_size())

        var a_quant_data = Array[Int8, _block_Q4_0.group_size](fill={})

        var a_scale = quantize_a_Q8[_block_Q4_0.group_size](
            a.ptr + a._offset(Index(m, k)), a_quant_data.unsafe_ptr()
        )

        var b_quant_data = Array[UInt8, _block_Q4_0.group_size](fill={})

        # Decode the bits of the weight data. The origin here is
        # parametrically mutable (it inherits `block_ptr`'s origin), so it can
        # be neither `MutPointer` (not statically mutable) nor `ImmPointer`
        # (not statically immutable); the bare `Pointer` keeps it parametric.
        var q_bits_ptr: Pointer[
            UInt8, origin_of(block_ptr[].q_bits)
        ] = block_ptr[].q_bits.unsafe_ptr()
        var q_packed_bits = q_bits_ptr.load[width=_block_Q4_0.group_size // 2]()

        var b_quant_data_ptr: MutPointer[
            UInt8, origin_of(b_quant_data)
        ] = b_quant_data.unsafe_ptr()
        for j in range(2):
            var idx = j * _block_Q4_0.group_size // 2
            var q_bits = (q_packed_bits >> UInt8(j * 4)) & 15
            b_quant_data_ptr.store(idx, q_bits)

        var sum: Int32 = 0

        comptime b_zero_point = 8

        for i in range(_block_Q4_0.group_size):
            sum += a_quant_data[i].cast[.int32]() * (
                (b_quant_data[i].cast[.int32]() - b_zero_point)
            )

        var sumf = (
            sum.cast[.float32]() * block_ptr[].base_scale.cast[.float32]()
        )

        return sumf * a_scale


struct qgemm_Q4_K(QuantizedGemm):
    @staticmethod
    def k_group_size() -> Int:
        return _block_QK_K.quantized_k

    @staticmethod
    def build_b_buffer(
        N: Int, K: Int
    ) -> LayoutTensor[.uint8, Layout.row_major[2](), MutUntrackedOrigin]:
        var k_groups = ceildiv(K, Self.k_group_size())
        var b_ptr = alloc[UInt8](N * k_groups * size_of[_block_Q4_K]())
        var block_ptr = b_ptr.bitcast[_block_Q4_K]()

        for _n in range(N):
            for _k in range(k_groups):
                block_ptr[].base_scale = random_float16(max=0.001)
                block_ptr[].base_min = random_float16(min=-0.01, max=0.01)
                fill_random(block_ptr[].q_scales_and_mins)
                fill_random(block_ptr[].q_bits)
                block_ptr += 1

        return LayoutTensor[.uint8, Layout.row_major[2]()](
            b_ptr,
            RuntimeLayout[Layout.row_major[2]()].row_major(
                Index(N, k_groups * size_of[_block_Q4_K]())
            ),
        )

    @staticmethod
    def pack_b_buffer(
        b: LayoutTensor[mut=True, .uint8, Layout.row_major[2](), _],
        b_packed: LayoutTensor[mut=True, .uint8, Layout.row_major[2](), _],
    ) raises:
        matmul_Q4_K_pack_b(lt_to_tt(b), lt_to_tt(b_packed))

    @staticmethod
    def kernel(
        a: LayoutTensor[.float32, Layout.row_major[2](), _],
        b: LayoutTensor[.uint8, Layout.row_major[2](), _],
        c: LayoutTensor[mut=True, .float32, Layout.row_major[2](), _],
    ):
        matmul_Q4_K(lt_to_tt(a), lt_to_tt(b), lt_to_tt(c))

    @staticmethod
    def dot_product(
        a: LayoutTensor[.float32, Layout.row_major[2](), _],
        b: LayoutTensor[.uint8, Layout.row_major[2](), _],
        m: Int,
        n: Int,
        k: Int,
    ) -> Float32:
        var block_ptr = (b.ptr + b._offset(Index(n, 0))).bitcast[
            _block_Q4_K
        ]() + (k // Self.k_group_size())

        var a_quant_data = Array[Int8, _block_QK_K.quantized_k](fill={})

        var a_scale = quantize_a_Q8[_block_QK_K.quantized_k](
            a.ptr + a._offset(Index(m, k)), a_quant_data.unsafe_ptr()
        )

        var a_quant_data_ptr: MutPointer[
            Int8, origin_of(a_quant_data)
        ] = a_quant_data.unsafe_ptr()
        var a_block_sums = Array[Int32, _block_Q4_K.group_count](
            fill_with=lambda (i: Int) -> Int32: (
                a_quant_data_ptr.load[width=_block_Q4_K.group_size](
                    i * _block_Q4_K.group_size
                )
                .cast[.int32]()
                .reduce_add()
            )
        )

        def b_scale_at(i: Int) {imm} -> UInt8:
            if i < 4:
                return block_ptr[].q_scales_and_mins[i] & 63
            return (block_ptr[].q_scales_and_mins[i + 4] & 15) | (
                (block_ptr[].q_scales_and_mins[i - 4] >> 6) << 4
            )

        def b_min_at(i: Int) {imm} -> UInt8:
            if i < 4:
                return block_ptr[].q_scales_and_mins[i + 4] & 63
            return (block_ptr[].q_scales_and_mins[i + 4] >> 4) | (
                (block_ptr[].q_scales_and_mins[i - 0] >> 6) << 4
            )

        var b_scales = Array[UInt8, _block_Q4_K.group_count](
            fill_with=b_scale_at
        )
        var b_mins = Array[UInt8, _block_Q4_K.group_count](fill_with=b_min_at)

        var b_quant_data = Array[UInt8, _block_QK_K.quantized_k](fill={})
        var b_quant_data_ptr: MutPointer[
            UInt8, origin_of(b_quant_data)
        ] = b_quant_data.unsafe_ptr()

        # Decode the bits of the weight data.
        for i in range(0, _block_QK_K.quantized_k // 2, 32):
            var q_bits_ptr: Pointer[
                UInt8, origin_of(block_ptr[].q_bits)
            ] = block_ptr[].q_bits.unsafe_ptr()
            var q_packed_bits = q_bits_ptr.load[width=32](i)

            for j in range(2):
                var idx = i * 2 + j * 32
                var q_bits = (q_packed_bits >> UInt8(j * 4)) & 15
                b_quant_data_ptr.store(idx, q_bits)

        var sum2: Int32 = 0

        for i in range(_block_Q4_K.group_count):
            sum2 += a_block_sums[i] * b_mins[i].cast[.int32]()

        var sum = dot_product_QK_K[group_size=_block_Q4_K.group_size](
            a_quant_data.unsafe_ptr(),
            b_quant_data.unsafe_ptr(),
            b_scales.unsafe_ptr(),
        )

        var sumf = (
            sum.cast[.float32]() * block_ptr[].base_scale.cast[.float32]()
        )
        sumf = (
            sumf - sum2.cast[.float32]() * block_ptr[].base_min.cast[.float32]()
        )

        return sumf * a_scale


struct qgemm_Q6_K(QuantizedGemm):
    @staticmethod
    def k_group_size() -> Int:
        return _block_QK_K.quantized_k

    @staticmethod
    def build_b_buffer(
        N: Int, K: Int
    ) -> LayoutTensor[.uint8, Layout.row_major[2](), MutUntrackedOrigin]:
        var k_groups = ceildiv(K, Self.k_group_size())
        var b_ptr = alloc[UInt8](N * k_groups * size_of[_block_Q6_K]())
        var block_ptr = b_ptr.bitcast[_block_Q6_K]()

        for _n in range(N):
            for _k in range(k_groups):
                fill_random(block_ptr[].q_bits_lo)
                fill_random(block_ptr[].q_bits_hi)
                fill_random(block_ptr[].q_scales)
                block_ptr[].base_scale = random_float16(max=0.001)
                block_ptr += 1

        return LayoutTensor[.uint8, Layout.row_major[2]()](
            b_ptr,
            RuntimeLayout[Layout.row_major[2]()].row_major(
                Index(N, k_groups * size_of[_block_Q6_K]())
            ),
        )

    @staticmethod
    def pack_b_buffer(
        b: LayoutTensor[mut=True, .uint8, Layout.row_major[2](), _],
        b_packed: LayoutTensor[mut=True, .uint8, Layout.row_major[2](), _],
    ) raises:
        matmul_Q6_K_pack_b(lt_to_tt(b), lt_to_tt(b_packed))

    @staticmethod
    def kernel(
        a: LayoutTensor[.float32, Layout.row_major[2](), _],
        b: LayoutTensor[.uint8, Layout.row_major[2](), _],
        c: LayoutTensor[mut=True, .float32, Layout.row_major[2](), _],
    ):
        matmul_Q6_K(lt_to_tt(a), lt_to_tt(b), lt_to_tt(c))

    @staticmethod
    def dot_product(
        a: LayoutTensor[.float32, Layout.row_major[2](), _],
        b: LayoutTensor[.uint8, Layout.row_major[2](), _],
        m: Int,
        n: Int,
        k: Int,
    ) -> Float32:
        var block_ptr = (b.ptr + b._offset(Index(n, 0))).bitcast[
            _block_Q6_K
        ]() + (k // Self.k_group_size())

        var a_quant_data = Array[Int8, _block_QK_K.quantized_k](fill={})

        var a_scale = quantize_a_Q8[_block_QK_K.quantized_k](
            a.ptr + a._offset(Index(m, k)), a_quant_data.unsafe_ptr()
        )

        var b_quant_data = Array[UInt8, _block_QK_K.quantized_k](fill={})
        var b_quant_data_ptr: MutPointer[
            UInt8, origin_of(b_quant_data)
        ] = b_quant_data.unsafe_ptr()

        # Decode the bottom bits of the weight data.
        for i in range(0, _block_QK_K.quantized_k // 2, 64):
            var q_bits_lo_ptr: Pointer[
                UInt8, origin_of(block_ptr[].q_bits_lo)
            ] = block_ptr[].q_bits_lo.unsafe_ptr()
            var q_packed_bits = q_bits_lo_ptr.load[width=64](i)

            for j in range(2):
                var idx = i * 2 + j * 64
                var q_bits = (q_packed_bits >> UInt8(j * 4)) & 15
                b_quant_data_ptr.store(idx, q_bits)

        # Decode the top bits of the weight data.
        for i in range(0, _block_QK_K.quantized_k // 4, 32):
            var q_bits_hi_ptr: Pointer[
                UInt8, origin_of(block_ptr[].q_bits_hi)
            ] = block_ptr[].q_bits_hi.unsafe_ptr()
            var q_packed_bits = q_bits_hi_ptr.load[width=32](i)

            for j in range(4):
                var idx = i * 4 + j * 32
                var q_bits_lo = b_quant_data_ptr.load[width=32](idx)
                var q_bits_hi = ((q_packed_bits >> UInt8(j * 2)) & 3) << 4
                b_quant_data_ptr.store(idx, q_bits_hi | q_bits_lo)

        var sum = dot_product_QK_K[
            group_size=_block_Q6_K.group_size, b_zero_point=32
        ](
            a_quant_data.unsafe_ptr(),
            b_quant_data.unsafe_ptr(),
            block_ptr[].q_scales.unsafe_ptr(),
        )

        var sumf = (
            sum.cast[.float32]() * block_ptr[].base_scale.cast[.float32]()
        )

        return sumf * a_scale


def reference_gemm[
    qgemm: QuantizedGemm
](
    a: LayoutTensor[.float32, Layout.row_major[2](), _],
    b: LayoutTensor[.uint8, Layout.row_major[2](), _],
    c: LayoutTensor[mut=True, .float32, Layout.row_major[2](), _],
):
    var M = a.dim[0]()
    var N = b.dim[0]()
    var K = a.dim[1]()

    comptime grain_size = 128

    var total_work = M * N
    var num_workers = ceildiv(total_work, grain_size)

    def task_func(task_id: Int) {var total_work, var N, var K, imm}:
        var task_start = task_id * grain_size
        var task_count = min(total_work - task_start, grain_size)

        for i in range(task_start, task_start + task_count):
            var m, n = divmod(i, N)

            var result: Float32 = 0

            for k in range(0, K, qgemm.k_group_size()):
                result += qgemm.dot_product(a, b, m, n, k)

            c.store(Index(m, n), result)

    sync_parallelize(task_func, num_workers)


struct GemmContext[qgemm: QuantizedGemm]:
    @__allow_legacy_any_origin_fields
    var a: LayoutTensor[.float32, Layout.row_major[2](), MutAnyOrigin]

    @__allow_legacy_any_origin_fields
    var b: LayoutTensor[.uint8, Layout.row_major[2](), MutAnyOrigin]

    @__allow_legacy_any_origin_fields
    var b_packed: LayoutTensor[.uint8, Layout.row_major[2](), MutAnyOrigin]

    @__allow_legacy_any_origin_fields
    var c: LayoutTensor[.float32, Layout.row_major[2](), MutAnyOrigin]

    @__allow_legacy_any_origin_fields
    var c_golden: LayoutTensor[.float32, Layout.row_major[2](), MutAnyOrigin]

    @staticmethod
    def _build_float_buffer(
        M: Int, N: Int
    ) raises -> LayoutTensor[
        .float32, Layout.row_major[2](), MutUntrackedOrigin
    ]:
        var ptr = alloc[Float32](M * N)
        for i in range(M * N):
            ptr[i] = random_float64(min=-1.0, max=+1.0).cast[.float32]()
        return LayoutTensor[.float32, Layout.row_major[2]()](
            ptr, RuntimeLayout[Layout.row_major[2]()].row_major(Index(M, N))
        )

    @staticmethod
    def _build_b_buffer(
        N: Int, K: Int
    ) raises -> LayoutTensor[.uint8, Layout.row_major[2](), MutAnyOrigin]:
        return Self.qgemm.build_b_buffer(N, K)

    @staticmethod
    def _pack_b_buffer(
        b: LayoutTensor[mut=True, .uint8, Layout.row_major[2](), _]
    ) raises -> LayoutTensor[.uint8, Layout.row_major[2](), MutUntrackedOrigin]:
        var b_packed_buffer = alloc[UInt8](b.size())
        var b_packed = LayoutTensor[.uint8, Layout.row_major[2]()](
            b_packed_buffer,
            RuntimeLayout[Layout.row_major[2]()].row_major(
                b.runtime_layout.shape.value.canonicalize()
            ),
        )
        Self.qgemm.pack_b_buffer(b, b_packed)
        return b_packed

    def __init__(out self, M: Int, N: Int, K: Int) raises:
        self.a = Self._build_float_buffer(M, K)
        self.b = Self._build_b_buffer(N, K)
        self.b_packed = Self._pack_b_buffer(self.b)
        self.c = Self._build_float_buffer(M, N)
        self.c_golden = Self._build_float_buffer(M, N)

    def free(mut self) raises:
        self.a.ptr.free()
        self.b.ptr.free()
        self.b_packed.ptr.free()
        self.c.ptr.free()
        self.c_golden.ptr.free()


def test_case[qgemm: QuantizedGemm](M: Int, N: Int, K: Int) raises:
    var ctx = GemmContext[qgemm](M, N, K)

    if K % qgemm.k_group_size() != 0:
        raise ("K must be a multiple of qgemm.k_group_size()")

    reference_gemm[qgemm](ctx.a, ctx.b, ctx.c_golden)
    qgemm.kernel(ctx.a, ctx.b_packed, ctx.c)

    var mismatch = False
    for i in range(ctx.c.size()):
        if not isclose(ctx.c.ptr[i], ctx.c_golden.ptr[i], atol=1e-4, rtol=1e-4):
            print(
                "MISMATCH",
                ctx.c.runtime_layout.idx2crd(
                    RuntimeTuple[IntTuple(UNKNOWN_VALUE)](i)
                ),
                ctx.c.ptr[i],
                ctx.c_golden.ptr[i],
            )
            mismatch = True
            break
    if mismatch:
        raise Error("found mismatch")

    ctx.free()


def test_cases[qgemm: QuantizedGemm]() raises:
    for m in range(1, 16):
        test_case[qgemm](m, 128, 256)
        test_case[qgemm](m, 128, 1024)

        comptime if qgemm.k_group_size() == 32:
            test_case[qgemm](m, 256, 32)

    # Typical LLM use case.
    test_case[qgemm](1, 4096, 4096)
    test_case[qgemm](160, 4096, 4096)


def main() raises:
    test_cases[qgemm_Q4_0]()
    test_cases[qgemm_Q4_K]()
    test_cases[qgemm_Q6_K]()
