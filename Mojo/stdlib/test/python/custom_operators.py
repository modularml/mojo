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

"""Operands that only implement part of each binary operator protocol.

Each method returns its own name so a test can tell which one CPython chose.
"""


class OnlyReflected:
    """Implements only the reflected half of each binary operator.

    A builtin left operand therefore returns `NotImplemented` first, and the
    operation only succeeds if the reflected fallback is attempted.
    """

    def __radd__(self, other: object) -> str:
        return "radd"

    def __rsub__(self, other: object) -> str:
        return "rsub"

    def __rmul__(self, other: object) -> str:
        return "rmul"

    def __rtruediv__(self, other: object) -> str:
        return "rtruediv"

    def __rfloordiv__(self, other: object) -> str:
        return "rfloordiv"

    def __rmod__(self, other: object) -> str:
        return "rmod"

    def __rpow__(self, other: object) -> str:
        return "rpow"

    def __rlshift__(self, other: object) -> str:
        return "rlshift"

    def __rrshift__(self, other: object) -> str:
        return "rrshift"

    def __rand__(self, other: object) -> str:
        return "rand"

    def __ror__(self, other: object) -> str:
        return "ror"

    def __rxor__(self, other: object) -> str:
        return "rxor"


class OnlyForward:
    """Implements only the non-in-place half of each binary operator.

    An augmented assignment therefore has no in-place slot to call, and only
    succeeds if it falls back to the plain operator.
    """

    def __add__(self, other: object) -> str:
        return "add"

    def __sub__(self, other: object) -> str:
        return "sub"

    def __mul__(self, other: object) -> str:
        return "mul"

    def __truediv__(self, other: object) -> str:
        return "truediv"

    def __floordiv__(self, other: object) -> str:
        return "floordiv"

    def __mod__(self, other: object) -> str:
        return "mod"

    def __pow__(self, other: object) -> str:
        return "pow"

    def __lshift__(self, other: object) -> str:
        return "lshift"

    def __rshift__(self, other: object) -> str:
        return "rshift"

    def __and__(self, other: object) -> str:
        return "and"

    def __or__(self, other: object) -> str:
        return "or"

    def __xor__(self, other: object) -> str:
        return "xor"
