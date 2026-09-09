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
"""Defines math utilities.

You can import these APIs from the `math` package. For example:

```mojo
from std.math import pi
```
"""


comptime pi = 3.1415926535897932384626433832795028841971693993751058209749445923
"""The mathematical constant π = 3.141592..."""

comptime e = 2.7182818284590452353602874713526624977572470936999595749669676277
"""The euler constant e = 2.718281..."""

comptime tau = 2 * pi
"""The mathematical constant τ = 6.283185.... Tau is a circumference of a circle (2π)."""

comptime log2e = 1.442695040888963407359924681001892137426646
"""The value of log2(e), where e is Euler's constant."""
