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

# RUN: %mojo-build %s -o %t
# RUN: %t | FileCheck %s

# Test that we can build executables that rely on AsyncRT and the runtime
# libraries.

from std.runtime._asyncrt import create_task


# CHECK-LABEL: test_runtime_task
def main():
    print("== test_runtime_task")

    @__parameter
    async def test_asyncrt_add[lhs: Int](rhs: Int) -> Int:
        return lhs + rhs

    @__parameter
    async def test_asyncrt_add_two_of_them(a: Int, b: Int) -> Int:
        return await create_task(test_asyncrt_add[1](a)) + await create_task(
            test_asyncrt_add[2](b)
        )

    var task = create_task(test_asyncrt_add_two_of_them(10, 20))
    # CHECK: 33
    print(task.wait())
