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

comptime CUDA_CUFFT_LIBRARY_PATHS: List[Path] = [
    "libcufft.so.12",
    "/usr/local/cuda/lib64/libcufft.so.12",
    "libcufft.so.11",
    "/usr/local/cuda/lib64/libcufft.so.11",
]


def _on_error_msg() -> Error:
    return Error(
        (
            "Cannot find the cuFFT libraries. Please make sure that "
            "the CUDA toolkit is installed and that the library path is "
            "correctly set in one of the following paths ["
        ),
        ", ".join(materialize[CUDA_CUFFT_LIBRARY_PATHS]()),
        (
            "]. You may need to make sure that you are using the non-slim"
            " version of the MAX container."
        ),
    )


comptime CUDA_CUFFT_LIBRARY = _Global[
    "CUDA_CUFFT_LIBRARY", _init_dylib, on_error_msg=_on_error_msg
]()


def _init_dylib() -> OwnedDLHandle:
    return _find_dylib[abort_on_failure=False](
        materialize[CUDA_CUFFT_LIBRARY_PATHS]()
    )


@inline(.always)
def _get_dylib_function[
    func_name: StaticString, result_type: TrivialRegisterPassable
]() raises -> result_type:
    return _ffi_get_dylib_function[
        CUDA_CUFFT_LIBRARY,
        func_name,
        result_type,
    ]()


@inline(.always)
def check_error(stat: Status) raises:
    if stat != Status.CUFFT_SUCCESS:
        raise Error("CUFFT ERROR: ", stat)
