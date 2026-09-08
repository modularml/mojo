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
"""Implements the `repr()` function for producing string representations of values."""


def repr[T: Writable](value: T) -> String:
    """Returns the string representation of the given value.

    Args:
        value: The value to get the string representation of.

    Parameters:
        T: The type of `value`. Must implement the `Writable` trait.

    Returns:
        The string representation of the given value.
    """
    var string = String()
    value.write_repr_to(string)
    return string^


def repr(value: None) -> String:
    """Returns the string representation of `None`.

    Args:
        value: A `None` value.

    Returns:
        The string representation of `None`.
    """
    return "None"
