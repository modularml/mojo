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

# CHECK: failed to infer parameter 'y', specify the parameter or use '_' or '...' to unbind the parameter explicitly

from std.compile import compile_info


def param_fn[x: Int, y: Int]() -> Int:
    return x + y


def main() raises:
    # intentionally missing one parameter
    comptime myInstantiatedFn = param_fn[2]
    print(compile_info[myInstantiatedFn]())
