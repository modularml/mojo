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

# RUN: not %mojo -DALLOC_TEST=1 %s 2>&1 | FileCheck %s --check-prefix=CHECK-TEST1
# RUN: not %mojo -DALLOC_TEST=2 %s 2>&1 | FileCheck %s --check-prefix=CHECK-TEST2

import std.sys

from std.memory import Layout


def main() raises:
    comptime if std.sys.get_defined_int["ALLOC_TEST"]() == 1:
        var too_small = 2
        # CHECK-TEST1: ABORT: {{.*}}: Alignment is invalid. Must be a power of two and >= to the types natural alignment.
        var _layout = Layout[Int32](count=1, alignment=too_small)
    elif std.sys.get_defined_int["ALLOC_TEST"]() == 2:
        var not_pow_of_2 = 127
        # CHECK-TEST2: ABORT: {{.*}}: Alignment is invalid. Must be a power of two and >= to the types natural alignment.
        var _layout = Layout[Int32](count=1, alignment=not_pow_of_2)
    else:
        comptime assert False, "unreachable!"
