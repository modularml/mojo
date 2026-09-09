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

# RUN: not %mojo %s 2>&1 | FileCheck %s

from std.ffi import _DLHandle


def main() raises:
    var handle = _DLHandle()
    # A plain Mojo function type passes struct arguments and return values
    # under the wrong convention, so it must be rejected rather than silently
    # miscompiled.
    # CHECK: result_type must be a C-ABI function pointer type
    _ = handle.get_function[def(Float64) thin -> Float64]("sqrt")
