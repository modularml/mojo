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

from std.sys import prefetch
from std.sys.intrinsics import PrefetchOptions


comptime mr = 6
comptime nr = 64

comptime simd_size = 16


def kernel6x4(
    a_ptr: ImmPointer[Float32, _],
    b_ptr: ImmPointer[Float32, _],
    c_ptr: MutPointer[Float32, _],
    n: Int,
    k: Int,
    kc: Int,
):
    var cv0 = c_ptr.load[width=simd_size](n * 0 + simd_size * 0)
    var cv1 = c_ptr.load[width=simd_size](n * 0 + simd_size * 1)
    var cv2 = c_ptr.load[width=simd_size](n * 0 + simd_size * 2)
    var cv3 = c_ptr.load[width=simd_size](n * 0 + simd_size * 3)
    var cv4 = c_ptr.load[width=simd_size](n * 1 + simd_size * 0)
    var cv5 = c_ptr.load[width=simd_size](n * 1 + simd_size * 1)
    var cv6 = c_ptr.load[width=simd_size](n * 1 + simd_size * 2)
    var cv7 = c_ptr.load[width=simd_size](n * 1 + simd_size * 3)
    var cv8 = c_ptr.load[width=simd_size](n * 2 + simd_size * 0)
    var cv9 = c_ptr.load[width=simd_size](n * 2 + simd_size * 1)
    var cv10 = c_ptr.load[width=simd_size](n * 2 + simd_size * 2)
    var cv11 = c_ptr.load[width=simd_size](n * 2 + simd_size * 3)
    var cv12 = c_ptr.load[width=simd_size](n * 3 + simd_size * 0)
    var cv13 = c_ptr.load[width=simd_size](n * 3 + simd_size * 1)
    var cv14 = c_ptr.load[width=simd_size](n * 3 + simd_size * 2)
    var cv15 = c_ptr.load[width=simd_size](n * 3 + simd_size * 3)
    var cv16 = c_ptr.load[width=simd_size](n * 4 + simd_size * 0)
    var cv17 = c_ptr.load[width=simd_size](n * 4 + simd_size * 1)
    var cv18 = c_ptr.load[width=simd_size](n * 4 + simd_size * 2)
    var cv19 = c_ptr.load[width=simd_size](n * 4 + simd_size * 3)
    var cv20 = c_ptr.load[width=simd_size](n * 5 + simd_size * 0)
    var cv21 = c_ptr.load[width=simd_size](n * 5 + simd_size * 1)
    var cv22 = c_ptr.load[width=simd_size](n * 5 + simd_size * 2)
    var cv23 = c_ptr.load[width=simd_size](n * 5 + simd_size * 3)

    for pr in range(kc):
        var bv0 = b_ptr.load[width=simd_size](
            4 * simd_size * pr + simd_size * 0
        )
        var bv1 = b_ptr.load[width=simd_size](
            4 * simd_size * pr + simd_size * 1
        )
        var bv2 = b_ptr.load[width=simd_size](
            4 * simd_size * pr + simd_size * 2
        )
        var bv3 = b_ptr.load[width=simd_size](
            4 * simd_size * pr + simd_size * 3
        )
        prefetch[PrefetchOptions().for_read().high_locality().to_data_cache()](
            b_ptr + 4 * simd_size * pr + simd_size * 16
        )
        prefetch[PrefetchOptions().for_read().high_locality().to_data_cache()](
            b_ptr + 4 * simd_size * pr + simd_size * 17
        )
        prefetch[PrefetchOptions().for_read().high_locality().to_data_cache()](
            b_ptr + 4 * simd_size * pr + simd_size * 18
        )
        prefetch[PrefetchOptions().for_read().high_locality().to_data_cache()](
            b_ptr + 4 * simd_size * pr + simd_size * 19
        )

        var av = a_ptr[0 * k + pr].cast[.float32]()
        cv0 += av * bv0
        cv1 += av * bv1
        cv2 += av * bv2
        cv3 += av * bv3

        av = a_ptr[1 * k + pr].cast[.float32]()
        cv4 += av * bv0
        cv5 += av * bv1
        cv6 += av * bv2
        cv7 += av * bv3

        av = a_ptr[2 * k + pr].cast[.float32]()
        cv8 += av * bv0
        cv9 += av * bv1
        cv10 += av * bv2
        cv11 += av * bv3

        av = a_ptr[3 * k + pr].cast[.float32]()
        cv12 += av * bv0
        cv13 += av * bv1
        cv14 += av * bv2
        cv15 += av * bv3

        av = a_ptr[4 * k + pr].cast[.float32]()
        cv16 += av * bv0
        cv17 += av * bv1
        cv18 += av * bv2
        cv19 += av * bv3

        av = a_ptr[5 * k + pr].cast[.float32]()
        cv20 += av * bv0
        cv21 += av * bv1
        cv22 += av * bv2
        cv23 += av * bv3

    c_ptr.store[width=simd_size](n * 0 + simd_size * 0, cv0)
    c_ptr.store[width=simd_size](n * 0 + simd_size * 1, cv1)
    c_ptr.store[width=simd_size](n * 0 + simd_size * 2, cv2)
    c_ptr.store[width=simd_size](n * 0 + simd_size * 3, cv3)
    c_ptr.store[width=simd_size](n * 1 + simd_size * 0, cv4)
    c_ptr.store[width=simd_size](n * 1 + simd_size * 1, cv5)
    c_ptr.store[width=simd_size](n * 1 + simd_size * 2, cv6)
    c_ptr.store[width=simd_size](n * 1 + simd_size * 3, cv7)
    c_ptr.store[width=simd_size](n * 2 + simd_size * 0, cv8)
    c_ptr.store[width=simd_size](n * 2 + simd_size * 1, cv9)
    c_ptr.store[width=simd_size](n * 2 + simd_size * 2, cv10)
    c_ptr.store[width=simd_size](n * 2 + simd_size * 3, cv11)
    c_ptr.store[width=simd_size](n * 3 + simd_size * 0, cv12)
    c_ptr.store[width=simd_size](n * 3 + simd_size * 1, cv13)
    c_ptr.store[width=simd_size](n * 3 + simd_size * 2, cv14)
    c_ptr.store[width=simd_size](n * 3 + simd_size * 3, cv15)
    c_ptr.store[width=simd_size](n * 4 + simd_size * 0, cv16)
    c_ptr.store[width=simd_size](n * 4 + simd_size * 1, cv17)
    c_ptr.store[width=simd_size](n * 4 + simd_size * 2, cv18)
    c_ptr.store[width=simd_size](n * 4 + simd_size * 3, cv19)
    c_ptr.store[width=simd_size](n * 5 + simd_size * 0, cv20)
    c_ptr.store[width=simd_size](n * 5 + simd_size * 1, cv21)
    c_ptr.store[width=simd_size](n * 5 + simd_size * 2, cv22)
    c_ptr.store[width=simd_size](n * 5 + simd_size * 3, cv23)


def kernel6x4_naive(
    a_ptr: ImmPointer[Float32, _],
    b_ptr: ImmPointer[Float32, _],
    c_ptr: MutPointer[Float32, _],
    n: Int,
    k: Int,
    kc: Int,
):
    for ir in range(mr):
        for jr in range(nr):
            for p in range(kc):
                c_ptr[ir * n + jr] += a_ptr[ir * k + p] * b_ptr[p * nr + jr]
