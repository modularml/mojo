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
"""Defines the builtin `NoneType`.

These are Mojo built-ins, so you don't need to import them.
"""


struct NoneType(
    Defaultable,
    ImplicitlyCopyable,
    TrivialRegisterPassable,
    Writable,
):
    """Represents the absence of a value."""

    comptime _mlir_type = __mlir_type.`!kgen.none`
    """Raw MLIR type of the `None` value."""

    var _value: Self._mlir_type

    @always_inline("builtin")
    def __init__(out self):
        """Construct an instance of the `None` type."""
        self._value = None

    @always_inline("builtin")
    @implicit
    def __init__(out self, value: Self._mlir_type):
        """Construct an instance of the `None` type.

        Args:
            value: The MLIR none type to construct from.
        """
        self._value = value

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        """Writes `None` to a writer.

        Args:
            writer: The object to write to.
        """
        writer.write_string("None")

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        """Writes `None` to a writer.

        Args:
            writer: The object to write to.
        """
        writer.write_string("None")
