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

# CHECK: compile_offload must reference a valid GeneratorOp
# CHECK-SAME: param_fn{{.*}}

from std.compile import compile_info


def param_fn[x: Int, y: Int]() -> Int:
    return x + y


def my_wrapper[f: def() thin -> Int]() -> def() thin -> Int:
    return f


def main() raises:
    # intentionally passing a function ptr instead of a generator
    print(compile_info[my_wrapper[param_fn[1, 2]]()]())
