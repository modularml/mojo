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


# start-take-simd
def take_simd(vec: SIMD[...]):
    print(vec.dtype)
    print(Int(vec.length))


# end-take-simd


def take_simd2[t: DType, s: SIMDLength, //](vec: SIMD[t, s]):
    print(t)
    print(Int(s))


# start-interleave
def interleave(v1: SIMD, v2: type_of(v1)) -> SIMD[v1.dtype, v1.length * 2]:
    var result = SIMD[v1.dtype, v1.length * 2]()

    comptime for i in range(v1.length):
        result[i * 2] = v1[i]
        result[i * 2 + 1] = v2[i]
    return result


# end-interleave


def foo[value: SIMD]():
    pass


def simd_param[value: SIMD[...]]():
    pass


def simd_param2[dtype: DType, size: SIMDLength, //, value: SIMD[dtype, size]]():
    pass


struct SomeStruct[s: SIMD[...]]:
    pass


struct SomeStruct2[dtype: DType, size: SIMDLength, //, s: SIMD[dtype, size]]:
    pass


# start-automatic-parameterization-comptime
comptime SomeComptime[s: SIMD[...]] = SomeStruct[s]

# Equivalent to:
comptime SomeComptime2[
    dtype: DType, size: SIMDLength, //, S: SIMD[dtype, size]
] = SomeStruct[S]
# end-automatic-parameterization-comptime


def main():
    # start-take-simd-usage
    var v = SIMD[.float64, 4](1.0, 2.0, 3.0, 4.0)
    take_simd(v)
    # end-take-simd-usage

    take_simd2(v)

    # start-interleave-usage
    var a = SIMD[.int16, 4](1, 2, 3, 4)
    var b = SIMD[.int16, 4](0, 0, 0, 0)
    var c = interleave(a, b)
    print(c)
    # end-interleave-usage
