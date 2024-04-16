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
"""Implements support functions for working with Windows."""

from sys import external_call, os_is_windows


@always_inline
fn _assert_windows():
    constrained[os_is_windows(), "This operation is only valid in Windows."]()


@always_inline
fn reset_last_error():
    """Resets any existing error code previous operations."""
    _assert_windows()
    external_call["SetLastError", NoneType](0)


@always_inline
fn get_last_error_code() -> Int:
    """
    Returns the error code from last operation.

    Returns:
      The error code from last completed operation. Returns 0
      if operation was success.
    """
    _assert_windows()
    return external_call["GetLastError", Int]()


@always_inline
fn last_operation_succeeded() -> Bool:
    """
    Returns if the last completed operation was a success or not.

    Returns:
      Returns True on success, else returns False.
    """
    _assert_windows()
    return get_last_error_code() == 0
