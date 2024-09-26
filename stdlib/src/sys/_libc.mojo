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
"""Implements low-level bindings to functions from the C standard library.

The functions in this module are intended to be thin wrappers around their
C standard library counterparts. These are used to implement higher level
functionality in the rest of the Mojo standard library.
"""

from memory import UnsafePointer
from sys.ffi import c_char, c_int, OpaquePointer


# ===----------------------------------------------------------------------===#
# dlfcn.h — dynamic library operations
# ===----------------------------------------------------------------------===#


@always_inline
fn dlerror() -> UnsafePointer[c_char]:
    return external_call["dlerror", UnsafePointer[c_char]]()


@always_inline
fn dlopen(filename: UnsafePointer[c_char], flags: c_int) -> OpaquePointer:
    return external_call["dlopen", OpaquePointer](filename, flags)


fn dlclose(handle: OpaquePointer) -> c_int:
    return external_call["dlclose", c_int](handle)


@always_inline
fn dlsym[
    # Default `dlsym` result is an OpaquePointer.
    result_type: AnyType = NoneType
](
    handle: OpaquePointer,
    name: UnsafePointer[c_char],
) -> UnsafePointer[
    result_type
]:
    return external_call["dlsym", UnsafePointer[result_type]](handle, name)
