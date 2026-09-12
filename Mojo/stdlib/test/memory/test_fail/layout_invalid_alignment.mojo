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

# RUN: not %mojo -D ALLOC_TEST=1 %s 2>&1 | FileCheck %s --check-prefix=CHECK-TEST1
# RUN: not %mojo -D ALLOC_TEST=2 %s 2>&1 | FileCheck %s --check-prefix=CHECK-TEST2

import std.sys

from std.memory import Layout


@align(128)
struct LargeAlign:
    var x: Int


def main() raises:
    comptime if std.sys.get_defined_int["ALLOC_TEST"]() == 1:
        comptime to_small = 2
        # CHECK-TEST1: note: constraint failed: `Layout` alignment must be at least `align_of[T]()`
        var _layout = Layout[LargeAlign, alignment=.of_bytes[to_small]()](
            count=1
        )
    comptime if std.sys.get_defined_int["ALLOC_TEST"]() == 2:
        comptime not_pow_of_2 = 129
        # CHECK-TEST2: note: constraint failed: alignment must be a power of two
        var _layout = Layout[LargeAlign, alignment=.of_bytes[not_pow_of_2]()](
            count=1
        )
    else:
        comptime assert False, "unreachable!"
