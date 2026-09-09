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

"""Elementwise operation traits for MAX Graph kernel implementations.

Defines the `ElementwiseUnaryOp`, `ElementwiseBinaryOp`, and related traits
that structured kernel implementations satisfy so the graph compiler can fuse
and dispatch elementwise epilogues.
"""

# ===----------------------------------------------------------------------=== #
# Op implementation traits
# ===----------------------------------------------------------------------=== #


trait ElementwiseUnaryOp:
    """Requires an `elementwise` op that transforms one SIMD value into another of the same dtype.

    Structured kernel implementations conforming to this trait expose a single
    static `elementwise` entry point so the graph compiler can fuse and dispatch
    unary elementwise epilogues.
    """

    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDLength,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        ...


trait ElementwiseUnaryMixedOp:
    """Requires an `elementwise` op that produces an output dtype different from its input.

    Structured kernel implementations conforming to this trait expose a single
    static `elementwise` entry point so the graph compiler can fuse and dispatch
    unary elementwise epilogues whose result widens or narrows the element type.
    """

    @staticmethod
    def elementwise[
        dtype: DType,
        out_dtype: DType,
        width: SIMDLength,
    ](x: SIMD[dtype, width]) -> SIMD[out_dtype, width]:
        ...


trait ElementwiseBinaryOp:
    """Requires an `elementwise` op that combines two SIMD values of the same dtype.

    Structured kernel implementations conforming to this trait expose a single
    static `elementwise` entry point so the graph compiler can fuse and dispatch
    binary elementwise epilogues.
    """

    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDLength,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        ...


trait ElementwiseBinaryComparisonOp:
    """Requires an `elementwise` op that compares two SIMD values and yields a boolean mask.

    Structured kernel implementations conforming to this trait expose a single
    static `elementwise` entry point so the graph compiler can fuse and dispatch
    binary comparison epilogues whose result is a boolean SIMD vector.
    """

    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDLength,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
        DType.bool, width
    ]:
        ...
