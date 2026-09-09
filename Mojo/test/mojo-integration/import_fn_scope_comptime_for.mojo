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

# RUN: %mojo %s | FileCheck %s

# An import inside a comptime for is instantiated once per unrolled trip
# during elaboration; the repeated bindings must not conflict and the name
# must resolve on every trip. Exact duplicate imports at function scope are
# de-duplicated. (Parse-level scoping of these imports is covered by
# KGEN/test/mojo-parser/import/import_fn_scope_comptime_blocks.mojo.)


def main():
    comptime for i in range(2):
        from std.math import sqrt

        print(sqrt(Float64(i)))
    # CHECK: 0.0
    # CHECK: 1.0

    from std.math import sqrt
    from std.math import sqrt

    print(sqrt(16.0))
    # CHECK: 4.0
