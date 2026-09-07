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
"""Complex numbers: SIMD types, scalar types, and operations.

The `complex` package provides types and operations for complex number
arithmetic in Mojo. It supports both scalar complex values and SIMD vectors of
complex numbers, enabling efficient vectorized complex arithmetic. The package
implements standard complex operations including arithmetic, conjugation,
magnitude calculation, and exponential functions.

Use this package for numerical computing with complex numbers, signal
processing, Fourier transforms, or any algorithm requiring complex arithmetic.
The SIMD support enables efficient processing of complex number arrays in
scientific and engineering applications.
"""

from .complex import (
    ComplexScalar,
    ComplexFloat32,
    ComplexFloat64,
    ComplexSIMD,
    abs,
)
