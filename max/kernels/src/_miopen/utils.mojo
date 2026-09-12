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

from std.pathlib import Path
from std.ffi import _find_dylib
from std.ffi import _get_dylib_function as _ffi_get_dylib_function
from std.ffi import _Global, OwnedDLHandle

from .types import Status

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#

comptime ROCM_MIOPEN_LIBRARY_PATHS: List[Path] = [
    "libMIOpen.so.1",
    "libMIOpen.so",
    "/opt/rocm/lib/libMIOpen.so.1",
    "/opt/rocm/lib/libMIOpen.so",
]


def _on_error_msg() -> Error:
    return Error(
        (
            "Cannot find the MIOpen library. Please make sure that "
            "the ROCm toolkit is installed and that the library path is "
            "correctly set in one of the following paths ["
        ),
        ", ".join(materialize[ROCM_MIOPEN_LIBRARY_PATHS]()),
        (
            "]. You may need to make sure that you are using the non-slim"
            " version of the MAX container."
        ),
    )


comptime ROCM_MIOPEN_LIBRARY = _Global[
    "ROCM_MIOPEN_LIBRARY", _init_dylib, on_error_msg=_on_error_msg
]()


def _init_dylib() -> OwnedDLHandle:
    return _find_dylib[abort_on_failure=False](
        materialize[ROCM_MIOPEN_LIBRARY_PATHS]()
    )


@inline(.always)
def _get_dylib_function[
    func_name: StaticString, result_type: TrivialRegisterPassable
]() raises -> result_type:
    return _ffi_get_dylib_function[
        ROCM_MIOPEN_LIBRARY,
        func_name,
        result_type,
    ]()


@inline(.always)
def check_error(stat: Status) raises:
    if stat != Status.SUCCESS:
        raise Error("MIOPEN ERROR:", stat)
