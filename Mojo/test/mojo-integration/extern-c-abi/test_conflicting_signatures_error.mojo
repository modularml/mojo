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
# Tests that two external_call sites to the same C function with different
# argument types produce a clear error. The error points to the second
# (conflicting) call; the note points back to where the first declaration was
# registered.
#
# Note: the error is not necessarily at the "wrong" call site — both locations
# are reported so the user can judge which declaration is correct.
#
# RUN: not %mojo %s 2>&1 | FileCheck %s
# (not --verify-diagnostics: the error is emitted inside stdlib/ffi/__init__.mojo, not this file)

from std.ffi import external_call


def call_with_int32():
    external_call["c_takes_int64", NoneType](Int32(1))


def call_with_uint64():
    # CHECK: existing function with conflicting signature
    external_call["c_takes_int64", NoneType](UInt64(2))


def main():
    call_with_int32()
    call_with_uint64()
