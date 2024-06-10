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
"""Provides functions to examine build configuration."""

from .param_env import env_get_string, is_defined


@always_inline("nodebug")
fn _build_type() -> StringLiteral:
    constrained[is_defined["BUILD_TYPE"](), "the build type must be defined"]()
    return env_get_string["BUILD_TYPE"]()


@always_inline("nodebug")
fn _kernels_build_type() -> StringLiteral:
    constrained[
        is_defined["KERNELS_BUILD_TYPE"](),
        "the kernels build type must be defined",
    ]()
    return env_get_string["KERNELS_BUILD_TYPE"]()


@always_inline("nodebug")
fn is_kernels_debug_build() -> Bool:
    """
    Returns True if the build is in debug mode.

    Returns:
        Bool: True if the build is in debug mode and False otherwise.
    """

    @parameter
    if is_defined["DEBUG"]():
        return True
    elif is_defined["KERNELS_BUILD_TYPE"]():
        return _kernels_build_type() == "debug"
    else:
        return False


@always_inline("nodebug")
fn is_debug_build() -> Bool:
    """
    Returns True if the build is in debug mode.

    Returns:
        Bool: True if the build is in debug mode and False otherwise.
    """

    @parameter
    if is_defined["DEBUG"]():
        return True
    elif is_defined["BUILD_TYPE"]():
        return _build_type() == "debug"
    else:
        return False


@always_inline("nodebug")
fn is_release_build() -> Bool:
    """
    Returns True if the build is in release mode.

    Returns:
        Bool: True if the build is in release mode and False otherwise.
    """

    @parameter
    if is_defined["DEBUG"]():
        return False
    elif is_defined["BUILD_TYPE"]():
        alias build_type: StringLiteral = _build_type()
        return (
            build_type == "release"
            or build_type == "relwithdebinfo"
            or build_type == "minsizerel"
        )
    else:
        return True


@always_inline("nodebug")
fn is_relwithdebinfo_build() -> Bool:
    """
    Returns True if the build is in relwithdebinfo mode.

    Returns:
        Bool: True if the build is in relwithdebinfo mode and False otherwise.
    """

    @parameter
    if is_defined["DEBUG"]():
        return True
    elif is_defined["BUILD_TYPE"]():
        return _build_type() == "relwithdebinfo"
    else:
        return False
