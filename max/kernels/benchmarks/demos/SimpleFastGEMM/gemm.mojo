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

# Meant to be run on an AVX512 system

from std.math import align_up
from std.memory import dealloc
from std.sys import align_of, prefetch, simd_width_of
from std.sys.intrinsics import PrefetchOptions

import std.benchmark
from linalg.utils import (
    get_matmul_kernel_shape,
    get_matmul_prefetch_b_distance_k,
)


from layout import TileTensor, Coord, Idx, row_major
from layout.tensor_engine import DefaultEngine

comptime dtype = DType.float32
comptime simd_size = simd_width_of[dtype]()
comptime alignment = align_of[SIMD[dtype, simd_size]]()

comptime kernel_shape = get_matmul_kernel_shape[dtype, dtype, dtype, False]()
comptime MR = kernel_shape.simd_rows
comptime NR = kernel_shape.simd_cols * simd_size

# AVX512 values
# alias MR = 6
# alias NR = 64

comptime prefetch_distance = get_matmul_prefetch_b_distance_k()


def print_mat(a_ptr: ImmPointer[Scalar[dtype], _], m: Int, n: Int):
    var a = TileTensor(a_ptr, row_major(m, n))
    for i in range(m):
        for j in range(n):
            print(a[i, j], end=" ")
        print("")


def gemm_naive(
    a: TileTensor[dtype, Engine=DefaultEngine[element_width=1], ...],
    b: TileTensor[dtype, Engine=DefaultEngine[element_width=1], ...],
    c: TileTensor[mut=True, dtype, Engine=DefaultEngine[element_width=1], ...],
    m: Int,
    n: Int,
    k: Int,
):
    comptime assert a.flat_rank == 2
    comptime assert b.flat_rank == 2
    comptime assert c.flat_rank == 2

    for i in range(m):
        for p in range(k):
            for j in range(n):
                c[i, j] += a[i, p] * b[p, j]


def kernel(
    a_ptr: ImmPointer[Scalar[dtype], _],
    b_ptr: ImmPointer[Scalar[dtype], _],
    c_ptr: MutPointer[Scalar[dtype], _],
    n: Int,
    k: Int,
    kc: Int,
):
    var a = TileTensor(a_ptr, row_major(MR * k))
    var b = TileTensor(b_ptr, row_major(k * NR))
    var c = TileTensor(c_ptr, row_major(MR * n))

    var c_stack = Array[Scalar[dtype], align_up(MR * NR, alignment)](
        uninitialized=True
    )
    var c_local = TileTensor(c_stack, row_major[MR * NR]())

    comptime NR2 = NR // simd_size

    comptime for idx0 in range(MR):
        for idx1 in range(NR2):
            var cv = c.load[width=simd_size](Coord(n * idx0 + simd_size * idx1))
            c_local.store(Coord(NR * idx0 + simd_size * idx1), cv)

    for pr in range(kc):
        comptime for i in range(NR2):
            prefetch[
                PrefetchOptions().for_read().high_locality().to_data_cache()
            ](b_ptr + NR * pr + simd_size * (i + 16))

        comptime for idx0 in range(MR):
            for idx1 in range(NR2):
                var av = a[idx0 * k + pr].cast[dtype]()
                var bv = b.load[width=simd_size](
                    Coord(NR * pr + simd_size * idx1)
                )
                var cv = c_local.load[width=simd_size](
                    Coord(NR * idx0 + simd_size * idx1)
                )
                cv += av * bv
                c_local.store(Coord(NR * idx0 + simd_size * idx1), cv)

    comptime for idx0 in range(MR):
        for idx1 in range(NR2):
            var cv = c_local.load[width=simd_size](
                Coord(NR * idx0 + simd_size * idx1)
            )
            c.store(Coord(n * idx0 + simd_size * idx1), cv)


def pack_B(
    b_ptr: ImmPointer[Scalar[dtype], _],
    b2_ptr: MutPointer[Scalar[dtype], _],
    k: Int,
    n: Int,
    kc: Int,
    nc: Int,
):
    var b = TileTensor(b_ptr, row_major(k * n))
    var bc = TileTensor(b2_ptr, row_major(k * n))
    for pr in range(kc):
        for ir in range(nc // NR):
            for v in range(NR):
                bc[NR * (pr + kc * ir) + v] = b[pr * n + NR * ir + v]


def prepack_B(
    b_ptr: ImmPointer[Scalar[dtype], _],
    b2_ptr: MutPointer[Scalar[dtype], _],
    k: Int,
    n: Int,
    kc: Int,
    nc: Int,
):
    for pc in range(0, k, kc):
        for jc in range(0, n, nc):
            pack_B(b_ptr + pc * n + jc, b2_ptr + n * pc + jc * kc, k, n, kc, nc)


def gemm(
    a_ptr: ImmPointer[Scalar[dtype], _],
    b_ptr: ImmPointer[Scalar[dtype], _],
    c_ptr: MutPointer[Scalar[dtype], _],
    m: Int,
    n: Int,
    k: Int,
    mc: Int,
    nc: Int,
    kc: Int,
):
    for ic in range(0, m, mc):
        for pc in range(0, k, kc):
            for jc in range(0, n, nc):
                for ir in range(0, mc, MR):
                    for jr in range(0, nc, NR):
                        kernel(
                            a_ptr + (ic + ir) * k + pc,
                            b_ptr + n * pc + jc * kc + jr * kc,
                            c_ptr + (ic + ir) * n + jc + jr,
                            n,
                            k,
                            kc,
                        )


def main() raises:
    var m = align_up(1024, MR)
    var n = align_up(1024, NR)
    var k: Int = 1024
    var mc: Int = m
    var nc: Int = NR
    var kc: Int = k
    if m % MR != 0:
        print("m must be multiple of 6")
        return
    if n % NR != 0:
        print("n must be a multiple of 64")
        return

    print(m, end="")
    print("x", end="")
    print(n, end="")
    print("x", end="")
    print(k)

    var a_alloc = (
        alloc[Scalar[dtype]].aligned[alignment](count=m * k).into_managed()
    )
    var b_alloc = (
        alloc[Scalar[dtype]].aligned[alignment](count=k * n).into_managed()
    )
    var b2_alloc = (
        alloc[Scalar[dtype]].aligned[alignment](count=k * n).into_managed()
    )
    var c_alloc = (
        alloc[Scalar[dtype]].aligned[alignment](count=m * n).into_managed()
    )
    var c2_alloc = (
        alloc[Scalar[dtype]].aligned[alignment](count=m * n).into_managed()
    )

    var a = TileTensor(a_alloc.unsafe_ptr(), row_major(m * k))
    var b = TileTensor(b_alloc.unsafe_ptr(), row_major(k * n))
    var b2 = TileTensor(b2_alloc.unsafe_ptr(), row_major(k * n))
    var c = TileTensor(c_alloc.unsafe_ptr(), row_major(m * n))
    var c2 = TileTensor(c2_alloc.unsafe_ptr(), row_major(m * n))

    var am = TileTensor(a_alloc.unsafe_ptr(), row_major(m, k))
    var bm = TileTensor(b_alloc.unsafe_ptr(), row_major(k, n))
    var cm = TileTensor(c_alloc.unsafe_ptr(), row_major(m, n))

    for i in range(m * k):
        a[i] = Scalar[dtype](i)
    for i in range(k * n):
        b[i] = Scalar[dtype](i)
        b2[i] = Scalar[dtype](i)
    for i in range(m * n):
        c[i] = Scalar[dtype](i)
        c2[i] = Scalar[dtype](i)

    prepack_B(b.ptr, b2.ptr, k, n, kc, nc)

    gemm_naive(am, bm, cm, m, n, k)
    gemm(a.ptr, b2.ptr, c2.ptr, m, n, k, mc, nc, kc)
    var errors: Int = 0
    for i in range(m * n):
        if c[i] != c2[i]:
            errors += 1
    print(errors, end="")
    print("/", end="")
    print(m * n, end="")
    print(" errors")

    def bench_gemm() {var}:
        gemm(a.ptr, b2.ptr, c2.ptr, m, n, k, mc, nc, kc)

    var num_warmup: Int = 1
    var time = std.benchmark.run(bench_gemm, num_warmup).mean()
    var flops = 2.0 * Float64(m) * Float64(n) * Float64(k) / time / 1e9
    print(time, end="")
    print(" seconds")
    print(flops, end="")
    print(" GFLOPS")

    # assume turbo is disabled and the frequency set to 2.9 GHz
    var rpeak = flops / (2.9 * 64)
    print(rpeak, end="")
    print(" measured/peak FLOPS assuming 2.9 GHz")

    dealloc(a_alloc^)
    dealloc(b_alloc^)
    dealloc(b2_alloc^)
    dealloc(c_alloc^)
    dealloc(c2_alloc^)
