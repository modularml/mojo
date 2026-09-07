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

from std.runtime._asyncrt import (
    create_raising_task,
    create_task,
    _create_task,
)

from std.testing import assert_equal, TestSuite


# CHECK-LABEL: test_runtime_task
def test_runtime_task() raises:
    print("== test_runtime_task")

    async def test_asyncrt_add[lhs: Int](rhs: Int) -> Int:
        return lhs + rhs

    async def test_asyncrt_add_two_of_them(a: Int, b: Int) -> Int:
        return await create_task(test_asyncrt_add[1](a)) + await create_task(
            test_asyncrt_add[2](b)
        )

    var task = create_task(test_asyncrt_add_two_of_them(10, 20))
    # CHECK: 33
    print(task.wait())


# CHECK-LABEL: test_runtime_taskgroup
def test_runtime_taskgroup() raises:
    print("== test_runtime_taskgroup")

    async def return_value[value: Int]() -> Int:
        return value

    async def run_as_group() -> Int:
        var t0 = create_task(return_value[1]())
        var t1 = create_task(return_value[2]())
        return await t0 + await t1

    var t0 = create_task(run_as_group())
    var t1 = create_task(run_as_group())
    # CHECK: 6
    print(t0.wait() + t1.wait())


# CHECK-LABEL: test_runtime_unified_async_memory_result_raises
def test_runtime_unified_async_memory_result_raises() raises:
    print("== test_runtime_unified_async_memory_result_raises")
    var prefix = String("hello")

    async def build_message() raises {mut prefix} -> String:
        return prefix + String(" world")

    var task = create_raising_task(build_message())
    var result = task^.wait()
    # CHECK: hello world
    print(result)
    assert_equal(result, "hello world")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
