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
"""Defines the symbols re-exported by __init__."""

comptime INLINED_ALIAS = 7
"""Docstring of the alias."""

comptime NOT_INLINED_ALIAS = 8
"""Docstring of the alias that is imported without the decorator."""


def inlined_func() -> Int:
    """Docstring of the function.

    Returns:
        Seven.
    """
    return INLINED_ALIAS


struct InlinedStruct:
    """Docstring of the struct."""

    var value: Int
    """Docstring of the field."""

    def __init__(out self):
        """Docstring of the constructor."""
        self.value = INLINED_ALIAS


@doc_hidden
struct HiddenStruct:
    """Docstring of the hidden struct."""

    pass
