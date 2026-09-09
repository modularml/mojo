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
from std.sys import align_of, simd_width_of

import std.benchmark
from layout import *

comptime MR = 6
comptime NR = 64

comptime dtype = DType.float32
comptime simd_size = simd_width_of[dtype]()
comptime alignment = align_of[SIMD[dtype, simd_size]]()


def gemm_naive[
    layout_b: Layout, origin: Origin
](
    c: TileTensor[mut=True, dtype, ...],  # M x N
    a: TileTensor[dtype, ...],  # M x K
    b: LayoutTensor[dtype, layout_b, MutAnyOrigin],  # N x K
):
    var M = Int(c.dim[0]())
    var N = b.dim(1)
    var K = b.dim(0)

    for mm in range(M):
        for kk in range(K):
            for nn in range(N):
                c.ptr[mm * N + nn] += a.ptr[mm * K + kk] * b[kk, nn]


def kernel[
    layout_c: Layout,
    layout_a: Layout,
    layout_b: Layout,
](
    c: LayoutTensor[dtype, layout_c],  # MR, NR
    a: LayoutTensor[dtype, layout_a],  # MR, K
    b_packed: LayoutTensor[dtype, layout_b],  # 1, K * NR
):
    var K = a.dim(1)

    var c_cache = TensorBuilder[MR, NR, dtype].OnStackAligned[alignment]()

    comptime for m in range(MR):
        c_cache.store[NR](m, 0, c.load[NR](m, 0))

    for pr in range(K // NR):
        var a_tile = a.tile[MR, NR](0, pr)
        var b_row = b_packed.tile[1, NR * NR](0, pr)

        for k in range(NR):
            var b_next_tile = b_row.tile[1, NR](0, k + 4)

            comptime for n in range(0, NR, simd_size):
                b_next_tile.prefetch(0, n)

            var b_tile = b_row.tile[1, NR](0, k)

            comptime for m in range(MR):
                var av = a_tile[m, k]

                c_cache.store[NR](
                    m, 0, av * b_tile.load[NR](0, 0) + c_cache.load[NR](m, 0)
                )

    comptime for m in range(MR):
        c.store[NR](m, 0, c_cache.load[NR](m, 0))


def pack_b[
    layout_b: Layout,
    layout_packed: Layout,
](
    b: LayoutTensor[layout_b, dtype],  # K x N
    packed: LayoutTensor[layout_packed, dtype],  # N // NR x K * NR
):
    comptime K = b.dim[0]()
    comptime N = b.dim[1]()

    for jc in range(N // NR):
        for pr in range(K // NR):
            var b_tile = b.tile[NR, NR](pr, jc)
            var packed_row = packed.tile[1, NR * NR](jc, pr)

            for k in range(NR):
                var packed_tile = packed_row.tile[1, NR](0, k)
                for n in range(NR):
                    packed_tile[0, n] = b_tile[k, n]


def gemm[
    N: Int,
    K: Int,
    layout_b: Layout,
](
    c: TileTensor[mut=True, dtype, ...],  # M x N
    a: TileTensor[dtype, ...],  # M x K
    b_packed: LayoutTensor[layout_b, dtype],  # (N // NR) x (K * NR)
):
    var M = Int(c.dim[0]())

    for jc in range(N // NR):
        var b_tile = b_packed.tile[1, K * NR](jc, 0)

        for ir in range(M // MR):
            var a_tile = TensorBuilder[MR, K, dtype].Wrap(a.ptr + K * MR * ir)

            # Possibly a slightly more efficient way of building c_tile
            comptime c_tile_layout = Layout([MR, NR], [N, 1])
            var c_tile = LayoutTensor[c_tile_layout, dtype](
                c.ptr + N * MR * ir + NR * jc
            )

            kernel(c_tile, a_tile, b_tile)


# kgen --emit=asm max/kernels/benchmarks/demos/SimpleFastGEMM/gemm_layout.mojo >out.S
@export
def gemm_export_dynamic(
    a_ptr: ImmPointer[Scalar[dtype], _],
    b_packed_ptr: ImmPointer[Scalar[dtype], _],
    c_ptr: MutPointer[Scalar[dtype], _],
    M: Int,
) abi("C"):
    comptime N = 1024
    comptime K = 1024
    var a = TileTensor(a_ptr, row_major(M, Idx[N]))
    var b_packed = TensorBuilder[N // NR, K * NR, dtype].Wrap(b_packed_ptr)
    var c = TileTensor(
        c_ptr.bitcast[Scalar[dtype], mut=True](), row_major(M, Idx[N])
    )
    gemm[N, K](c, a, b_packed)


def main():
    comptime M = align_up(1024, MR)
    comptime N = align_up(1024, NR)
    comptime K: Int = 1024

    if M % MR != 0:
        print("M must be multiple of", MR)
        return
    if N % NR != 0:
        print("N must be a multiple of", NR)
        return

    print(M, end="")
    print("x", end="")
    print(N, end="")
    print("x", end="")
    print(K)

    # FIXME: Something causes sporadic crashes on intel with TensorBuilder.Build()
    var a_alloc = alloc[Float32].aligned[alignment](count=M * K).into_managed()
    var b_alloc = alloc[Float32].aligned[alignment](count=K * N).into_managed()
    var b_packed_alloc = (
        alloc[Float32].aligned[alignment](count=K * N).into_managed()
    )
    var c_alloc = alloc[Float32].aligned[alignment](count=M * N).into_managed()
    var c2_alloc = alloc[Float32].aligned[alignment](count=M * N).into_managed()

    var a = TileTensor(a_alloc.unsafe_ptr(), row_major[M, K]())

    var b = TensorBuilder[K, N, dtype].Wrap(b_alloc.unsafe_ptr())
    var b_packed = TensorBuilder[N // NR, K * NR, dtype].Wrap(
        b_packed_alloc.unsafe_ptr()
    )

    var c = TileTensor(c_alloc.unsafe_ptr(), row_major[M, N]())
    var c2 = TileTensor(c2_alloc.unsafe_ptr(), row_major[M, N]())

    for j in range(M):
        for i in range(K):
            a.ptr[j * K + i] = K * j + i

    for j in range(K):
        for i in range(N):
            b[j, i] = N * j + i

    for j in range(M):
        for i in range(N):
            c.ptr[j * N + i] = 0
            c2.ptr[j * N + i] = 0

    pack_b(b, b_packed)

    gemm_naive(c, a, b)
    gemm[N, K](c2, a, b_packed)
    var errors: Int = 0
    for j in range(M):
        for i in range(N):
            if c.ptr[j * N + i] != c2.ptr[j * N + i]:
                errors += 1

    print(errors)
    print("/", end="")
    print(M * N, end="")
    print(" errors")

    def bench_gemm() {var}:
        gemm[N, K](c2, a, b_packed)

    var num_warmup: Int = 1
    var time = std.benchmark.run(bench_gemm, num_warmup).mean()
    var flops = 2.0 * M * N * K / time / 1e9
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
    dealloc(b_packed_alloc^)
    dealloc(c_alloc^)
    dealloc(c2_alloc^)
