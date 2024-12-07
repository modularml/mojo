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

from sys import os_is_windows
from sys.ffi import OpaquePointer, c_char, c_int

from memory import UnsafePointer

# ===-----------------------------------------------------------------------===#
# stdlib.h — core C standard library operations
# ===-----------------------------------------------------------------------===#


@always_inline
fn free(ptr: OpaquePointer):
    external_call["free", NoneType](ptr)


@always_inline
fn exit(status: c_int):
    external_call["exit", NoneType](status)


# ===-----------------------------------------------------------------------===#
# stdio.h — input/output operations
# ===-----------------------------------------------------------------------===#

alias FILE_ptr = OpaquePointer


@always_inline
fn fdopen(fd: c_int, mode: UnsafePointer[c_char]) -> FILE_ptr:
    alias name = "_fdopen" if os_is_windows() else "fdopen"

    return external_call[name, FILE_ptr](fd, mode)


@always_inline
fn fclose(stream: FILE_ptr) -> c_int:
    return external_call["fclose", c_int](stream)


@always_inline
fn fflush(stream: FILE_ptr) -> c_int:
    return external_call["fflush", c_int](stream)


@always_inline
fn popen(
    command: UnsafePointer[c_char],
    type: UnsafePointer[c_char],
) -> FILE_ptr:
    return external_call["popen", FILE_ptr](command, type)


@always_inline
fn pclose(stream: FILE_ptr) -> c_int:
    return external_call["pclose", c_int](stream)


# ===-----------------------------------------------------------------------===#
# unistd.h
# ===-----------------------------------------------------------------------===#


@always_inline
fn dup(oldfd: c_int) -> c_int:
    alias name = "_dup" if os_is_windows() else "dup"

    return external_call[name, c_int](oldfd)


# ===-----------------------------------------------------------------------===#
# dlfcn.h — dynamic library operations
# ===-----------------------------------------------------------------------===#


@always_inline
fn dlerror() -> UnsafePointer[c_char]:
    return external_call["dlerror", UnsafePointer[c_char]]()


@always_inline
fn dlopen(filename: UnsafePointer[c_char], flags: c_int) -> OpaquePointer:
    return external_call["dlopen", OpaquePointer](filename, flags)


@always_inline
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
