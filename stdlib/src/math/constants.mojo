# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
"""Defines mathematical constants, accurate to 64 decimal places.

You can import these APIs from the `math` package. For example:

```mojo
from math import pi
```
"""

alias nan = FloatLiteral(__mlir_attr[`#kgen.float_literal<nan>`])
"""Not a Number. Represents mathematical degeneracy. Equal to nothing, including itself."""

alias inf = FloatLiteral(__mlir_attr[`#kgen.float_literal<inf>`])
"""Infinity. Represents an unbounded quantity larger than any number."""

alias e = 2.7182818284590452353602874713526624977572470936999595749669676277
"""The euler constant e = 2.718281... e is the natural base."""

alias pi = 3.1415926535897932384626433832795028841971693993751058209749445923
"""The mathematical constant π = 3.141592... pi is the area of the unit circle (τ/2)."""

alias tau = 6.2831853071795864769252867665590057683943387987502116419498891846
"""The mathematical constant τ = 6.283185... tau is the circumference of the unit circle (2π)."""

alias phi = 1.6180339887498948482045868343656381177203091798057628621354486227
"""The golden ratio φ = 1.618033... phi satisfies the equation: 1/φ = φ-1."""

alias omg = 0.5671432904097838729999686622103555497538157871865125081351310792
"""The omega constant Ω = 0.567143... omega satisfies the equation: Ωe**Ω = 1"""

alias egm = 0.5772156649015328606065120900824024310421593359399235988057672348
"""The gamma constant γ = 0.577215... gamma is related to the digamma function evaluated at 1."""
