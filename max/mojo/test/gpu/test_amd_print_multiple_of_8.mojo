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
# REQUIRES: AMD-GPU
# RUN: %mojo %s | FileCheck %s
"""Regression test for AMDGPU multiple-of-8 print corruption.

On AMDGPU, strings are packed into 64-bit words for the `hostcall` printf
interface, and the host reads each appended string up to its nul terminator.
When a string's byte length was an exact multiple of 8, the terminator was
dropped, so the host read past the payload and emitted nondeterministic
garbage. Both the `print()` buffer path and the low-level `_printf`
format-string path are exercised here with multiple-of-8 byte lengths.

Launched single-threaded (grid=1/block=1) for deterministic output. The
`{{^}}...{{$}}` full-line anchors reject leading/trailing garbage, and
`CHECK-NEXT` rejects stray lines bleeding in between.
"""

from max.gpu.host import DeviceContext
from std.io.io import _printf


def _kernel():
    # `print()` path: "test 12" + "\n" == 8 bytes.
    # CHECK: {{^}}test 12{{$}}
    print("test", 12)
    # `print()` path: "abcdefg" + "\n" == 8 bytes.
    # CHECK-NEXT: {{^}}abcdefg{{$}}
    print("abcdefg")
    # `_printf` format-string path: "1234567\n" == 8 bytes.
    # CHECK-NEXT: {{^}}1234567{{$}}
    _printf["1234567\n"]()
    # `_printf` format-string path: "0123456789abcde\n" == 16 bytes.
    # CHECK-NEXT: {{^}}0123456789abcde{{$}}
    _printf["0123456789abcde\n"]()


def main() raises:
    with DeviceContext() as ctx:
        ctx.enqueue_function[_kernel](grid_dim=1, block_dim=1)
        ctx.synchronize()
