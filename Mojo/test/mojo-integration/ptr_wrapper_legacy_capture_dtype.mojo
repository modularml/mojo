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
# RUN: %mojo %s

# Wrapping a nested def as a thin ImplicitlyCopyable value. Signature captures
# stay unbound on the wrapper Impl. A body ParamDeclRef whose type mentions a
# signature capture becomes an extra PtrWrapper param bound into Impl at
# `__call__`. One whose type does not is omitted and stays a leftover, partially
# bound ParamDeclRef on the nested symbol. `dtype` appears on a value argument
# so `__call__` must bind that capture from the method aux, not the struct.

from std.testing import assert_equal


def take_thin[
    dtype: DType,
    //,
    F: ImplicitlyCopyable
    & def[width: Int](SIMD[dtype, width]) -> SIMD[dtype, width],
](f: F) -> SIMD[dtype, 1]:
    return f[1](SIMD[dtype, 1](1))


def wrap[
    dtype: DType,
    input_fn: def[width: Int](SIMD[dtype, width]) capturing -> SIMD[
        dtype, width
    ],
]() -> SIMD[dtype, 1]:
    def unified[width: Int](val: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return input_fn[width](val)

    return take_thin(unified)


def wrap_unrelated_input_fn[
    dtype: DType,
    input_fn: def[width: Int](SIMD[DType.float32, width]) capturing -> SIMD[
        DType.float32, width
    ],
]() -> SIMD[dtype, 1]:
    def unified[width: Int](val: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return input_fn[width](val.cast[DType.float32]()).cast[dtype]()

    return take_thin(unified)


def ones[
    width: Int
](val: SIMD[DType.float32, width]) capturing -> SIMD[DType.float32, width]:
    return val


def main() raises:
    assert_equal(wrap[DType.float32, ones](), SIMD[DType.float32, 1](1))
    assert_equal(
        wrap_unrelated_input_fn[DType.float32, ones](),
        SIMD[DType.float32, 1](1),
    )
