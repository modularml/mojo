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
"""Provides functions to examine build configuration."""

from .defines import get_defined_string, is_defined


@always_inline("nodebug")
def _build_type() -> StaticString:
    comptime assert is_defined["BUILD_TYPE"](), "the build type must be defined"
    return get_defined_string["BUILD_TYPE"]()


@always_inline("nodebug")
def is_debug_build() -> Bool:
    """
    Returns True if the build is in debug mode.

    Returns:
        Bool: True if the build is in debug mode and False otherwise.
    """

    comptime if is_defined["DEBUG"]():
        return True
    elif is_defined["BUILD_TYPE"]():
        return _build_type() == "debug"
    else:
        return False


@always_inline("nodebug")
def is_release_build() -> Bool:
    """
    Returns True if the build is in release mode.

    Returns:
        Bool: True if the build is in release mode and False otherwise.
    """

    comptime if is_defined["DEBUG"]():
        return False
    elif is_defined["BUILD_TYPE"]():
        comptime build_type = _build_type()
        return (
            build_type == "release"
            or build_type == "relwithdebinfo"
            or build_type == "minsizerel"
        )
    else:
        return True
