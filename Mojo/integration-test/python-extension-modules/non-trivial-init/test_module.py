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

"""Test non-trivial initialization of MojoPair from Python."""

# Imports from 'mojo_module.so'
import mojo_module  # type: ignore[import-not-found]
import pytest


def test_non_trivial_initialization() -> None:
    """Test creating MojoPair with custom values MojoPair(42, 10)."""
    pair = mojo_module.MojoPair(42, 10)

    # Verify the values were set correctly
    assert pair.get_first() == 42
    assert pair.get_second() == 10


def test_swap_method() -> None:
    """Test the swap method works correctly."""
    pair = mojo_module.MojoPair(42, 10)

    # Before swap
    assert pair.get_first() == 42
    assert pair.get_second() == 10

    # Swap and verify
    result = pair.swap()
    assert result == pair  # Should return self
    assert pair.get_first() == 10
    assert pair.get_second() == 42


def test_error_handling() -> None:
    """Test error handling for invalid initialization arguments.

    A raising ``__init__`` flows through ``_py_init_function_wrapper``, which
    translates the Mojo ``Error`` into a Python ``ValueError`` (the wrapper's
    ``exc_type``). Assert the concrete type and message, not just
    ``Exception``, to cover that translation end-to-end.
    """
    with pytest.raises(
        ValueError, match="Failed to convert first argument to integer"
    ):
        mojo_module.MojoPair("not_a_number", 10)

    with pytest.raises(
        ValueError, match="Failed to convert second argument to integer"
    ):
        mojo_module.MojoPair(42, "not_a_number")


def test_keyword_arguments() -> None:
    """Test that keyword arguments work in __init__ methods."""
    # Test basic keyword arguments
    pair = mojo_module.MojoPair(first=42, second=10)
    assert pair.get_first() == 42
    assert pair.get_second() == 10

    # Test mixed positional and keyword arguments
    pair2 = mojo_module.MojoPair(100, second=200)
    assert pair2.get_first() == 100
    assert pair2.get_second() == 200

    # Test keyword-only arguments (all as kwargs)
    pair3 = mojo_module.MojoPair(second=999, first=888)
    assert pair3.get_first() == 888
    assert pair3.get_second() == 999

    # Test no keyword arguments
    pair4 = mojo_module.MojoPair(100, 200)
    assert pair4.get_first() == 100
    assert pair4.get_second() == 200
