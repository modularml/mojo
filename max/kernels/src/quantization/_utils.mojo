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
from std.sys.info import CompilationTarget, simd_width_of
from std.sys.intrinsics import llvm_intrinsic


@always_inline
def roundeven_to_int32[
    dtype: DType, simd_width: SIMDLength
](x: SIMD[dtype, simd_width]) -> SIMD[.int32, simd_width]:
    comptime native_width = simd_width_of[dtype]()

    # Use the AVX512 instruction `vcvtps2dq` with embedded rounding control
    # set to do rounding to nearest with ties to even (roundeven). This
    # replaces a `vrndscaleps` and `vcvttps2dq` instruction pair.
    comptime if (
        CompilationTarget.has_avx512f()
        and dtype == .float32
        and simd_width >= native_width
    ):
        var x_i32 = SIMD[.int32, simd_width]()

        comptime for i in range(0, simd_width, native_width):
            var part = llvm_intrinsic[
                "llvm.x86.avx512.mask.cvtps2dq.512",
                SIMD[.int32, native_width],
                has_side_effect=False,
            ](
                x.slice[native_width, offset=i](),
                SIMD[.int32, native_width](0),
                Int16(-1),  # no mask
                Int32(8),  # round to nearest
            )
            x_i32 = x_i32.insert[offset=i](part)

        return x_i32

    # Use the NEON instruction `fcvtns` to fuse the conversion to int32
    # with rounding to nearest with ties to even (roundeven). This
    # replaces a `frintn` and `fcvtzs` instruction pair.
    comptime if (
        CompilationTarget.has_neon()
        and dtype == .float32
        and simd_width >= native_width
    ):
        var x_i32 = SIMD[.int32, simd_width]()

        comptime for i in range(0, simd_width, native_width):
            var part = llvm_intrinsic[
                "llvm.aarch64.neon.fcvtns.v4i32.v4f32",
                SIMD[.int32, native_width],
                has_side_effect=False,
            ](x.slice[native_width, offset=i]())
            x_i32 = x_i32.insert[offset=i](part)

        return x_i32

    return round(x).cast[.int32]()
