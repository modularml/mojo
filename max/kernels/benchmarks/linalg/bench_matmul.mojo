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

from std.os import abort
from std.random import rand
from std.utils.numerics import get_accum_type

from std.benchmark import *
from std.benchmark import keep
from layout import Coord, TileTensor, row_major
from linalg.matmul import matmul
from linalg.packing import pack_b_ndbuffer, pack_matmul_b_shape_func
from std.memory import ThinAllocation, dealloc, Layout
from std.testing import assert_almost_equal


def _ri(v: Int) -> Int64:
    return Int64(v)


def gemm_naive(a: TileTensor, b: TileTensor, c: TileTensor[mut=True, ...]):
    """Naive reference matmul. Accumulates into an `acc_type` (f32 for f16/bf16
    output) scratch buffer so that verification matches the main matmul's
    internal fp32-scratch behavior."""
    var m = Int(c.dim[0]())
    var n = Int(c.dim[1]())
    var k = Int(a.dim[1]())

    comptime acc_type = get_accum_type[c.dtype]()

    var accum = List(length=m * n, fill=Scalar[acc_type](0))

    for i in range(m):
        for p in range(k):
            var a_val = a.raw_load(i * k + p).cast[acc_type]()
            for j in range(n):
                var b_val = b.raw_load(p * n + j).cast[acc_type]()
                accum[i * n + j] += a_val * b_val

    for idx in range(m * n):
        c.ptr[idx] = accum[idx].cast[c.dtype]()


def verify(a: TileTensor, b: TileTensor, c: TileTensor):
    var m = Int(c.dim[0]())
    var n = Int(c.dim[1]())

    var c_ref_ptr = List(length=m * n, fill=Scalar[c.dtype](0))
    var c_ref = TileTensor(c_ref_ptr, row_major(Coord(_ri(m), _ri(n))))
    gemm_naive(a, b, c_ref)

    for i in range(m):
        for j in range(n):
            try:
                assert_almost_equal(
                    c.raw_load(i * n + j), c_ref.raw_load(i * n + j)
                )
            except e:
                abort(String(e))


def bench_matmul_spec(mut m: Bench, spec: MatmulSpec) raises:
    # disatch to bench_matmul with concrete spec type
    m.bench_with_input(
        bench_matmul[spec.static_info],
        BenchId("matmul", String(spec)),
        spec,
        # TODO: Pick relevant benchmetric
        [ThroughputMeasure(BenchMetric.elements, spec.flops())],
    )


def bench_matmul[
    static: MatmulSpecStatic
](mut bencher: Bencher, spec: MatmulSpec[static]) raises:
    comptime a_type = spec.static_info.a_type
    comptime b_type = spec.static_info.b_type
    comptime c_type = spec.static_info.c_type
    comptime b_packed = spec.static_info.b_packed
    comptime alignment = 64
    var a_alloc = alloc(
        Layout[Scalar[a_type]].aligned[alignment](count=spec.m * spec.k)
    ).into_managed()
    var b_alloc = alloc(
        Layout[Scalar[b_type]].aligned[alignment](count=spec.k * spec.n)
    ).into_managed()
    var c_alloc = alloc(
        Layout[Scalar[c_type]].aligned[alignment](count=spec.m * spec.n)
    ).into_managed()
    var a = TileTensor(
        a_alloc.unsafe_ptr(), row_major(Coord(_ri(spec.m), _ri(spec.k)))
    )
    var b = TileTensor(
        b_alloc.unsafe_ptr(), row_major(Coord(_ri(spec.k), _ri(spec.n)))
    )
    var c = TileTensor(
        c_alloc.unsafe_ptr(), row_major(Coord(_ri(spec.m), _ri(spec.n)))
    )
    rand(a_alloc.unsafe_span())
    rand(b_alloc.unsafe_span())
    _ = c.fill(Scalar[c_type](0))

    var padded_n_k = pack_matmul_b_shape_func[a_type, c_type, False](b)

    var padded_n = padded_n_k[1] if b_packed else spec.n
    var padded_k = padded_n_k[0] if b_packed else spec.k

    var bp_alloc = alloc[Scalar[b_type]](
        {count = padded_k * padded_n, alignment = alignment}
    ).into_managed()
    var bp = TileTensor(
        bp_alloc.unsafe_ptr(), row_major(Coord(_ri(padded_k), _ri(padded_n)))
    )

    if b_packed:
        pack_b_ndbuffer[a_type, c_type](b, bp)

    @always_inline
    def bench_fn() raises {var a, var b, var c, var bp}:
        comptime bench_matmul = matmul[
            transpose_b=False, b_packed=b_packed, saturated_vnni=False
        ]
        if b_packed:
            bench_matmul(c, a, bp)
        else:
            bench_matmul(c, a, b)
        keep(c.ptr)

    bencher.iter(bench_fn)
    verify(a, b, c)

    dealloc(a_alloc^)
    dealloc(b_alloc^)
    dealloc(bp_alloc^)
    dealloc(c_alloc^)


@fieldwise_init
struct MatmulSpecStatic(ImplicitlyCopyable):
    var b_packed: Bool
    var a_type: DType
    var b_type: DType
    var c_type: DType


@fieldwise_init
struct MatmulSpec[static_info: MatmulSpecStatic](ImplicitlyCopyable, Writable):
    var m: Int
    var n: Int
    var k: Int

    def write_to(self, mut writer: Some[Writer]):
        """Writes a string representation of the matmul spec.

        Args:
            writer: The writer to write to.
        """
        writer.write(
            "m=",
            self.m,
            ";n=",
            self.n,
            ";k=",
            self.k,
            ";b_packed=",
            Self.static_info.b_packed,
            ";a_type=",
            Self.static_info.a_type,
            ";b_type=",
            Self.static_info.b_type,
            ";c_type=",
            Self.static_info.c_type,
        )

    def flops(self) -> Int:
        return 2 * self.m * self.n * self.k


def main() raises:
    var m = Bench(BenchConfig(num_repetitions=2))

    comptime packed_float32 = MatmulSpecStatic(
        b_packed=True,
        a_type=DType.float32,
        b_type=DType.float32,
        c_type=DType.float32,
    )
    comptime unpacked_float32 = MatmulSpecStatic(
        b_packed=False,
        a_type=DType.float32,
        b_type=DType.float32,
        c_type=DType.float32,
    )
    comptime unpacked_f16_f16 = MatmulSpecStatic(
        b_packed=False,
        a_type=DType.float16,
        b_type=DType.float16,
        c_type=DType.float16,
    )
    comptime unpacked_f16_f32 = MatmulSpecStatic(
        b_packed=False,
        a_type=DType.float16,
        b_type=DType.float16,
        c_type=DType.float32,
    )
    comptime unpacked_bf16_bf16 = MatmulSpecStatic(
        b_packed=False,
        a_type=DType.bfloat16,
        b_type=DType.bfloat16,
        c_type=DType.bfloat16,
    )
    comptime unpacked_bf16_f32 = MatmulSpecStatic(
        b_packed=False,
        a_type=DType.bfloat16,
        b_type=DType.bfloat16,
        c_type=DType.float32,
    )
    bench_matmul_spec(m, MatmulSpec[packed_float32](m=256, n=256, k=256))
    bench_matmul_spec(m, MatmulSpec[packed_float32](m=512, n=512, k=512))
    bench_matmul_spec(m, MatmulSpec[packed_float32](m=1024, n=1024, k=1024))
    bench_matmul_spec(m, MatmulSpec[unpacked_float32](m=256, n=256, k=256))

    # fp16/bf16 coverage — exercises the internal fp32 scratch-buffer path.
    bench_matmul_spec(m, MatmulSpec[unpacked_f16_f16](m=256, n=256, k=256))
    bench_matmul_spec(m, MatmulSpec[unpacked_f16_f16](m=512, n=512, k=512))
    bench_matmul_spec(m, MatmulSpec[unpacked_f16_f16](m=1024, n=1024, k=1024))
    bench_matmul_spec(m, MatmulSpec[unpacked_f16_f32](m=256, n=256, k=256))
    bench_matmul_spec(m, MatmulSpec[unpacked_f16_f32](m=512, n=512, k=512))
    bench_matmul_spec(m, MatmulSpec[unpacked_f16_f32](m=1024, n=1024, k=1024))
    bench_matmul_spec(m, MatmulSpec[unpacked_bf16_bf16](m=256, n=256, k=256))
    bench_matmul_spec(m, MatmulSpec[unpacked_bf16_bf16](m=512, n=512, k=512))
    bench_matmul_spec(m, MatmulSpec[unpacked_bf16_bf16](m=1024, n=1024, k=1024))
    bench_matmul_spec(m, MatmulSpec[unpacked_bf16_f32](m=256, n=256, k=256))
    bench_matmul_spec(m, MatmulSpec[unpacked_bf16_f32](m=512, n=512, k=512))
    bench_matmul_spec(m, MatmulSpec[unpacked_bf16_f32](m=1024, n=1024, k=1024))

    m.dump_report()
