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

# Test that global_constant produces a clear error message when used with
# a type that has non-trivial copy/destroy semantics.

from std.builtin.globals import global_constant


# CHECK: global_constant requires a type with trivial copy and destroy semantics
def main() raises:
    comptime s = String("hello")
    ref global_ptr = global_constant[s]()
    print(global_ptr)
