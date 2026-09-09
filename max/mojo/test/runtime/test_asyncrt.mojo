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

from std.testing import assert_equal, assert_true, TestSuite

from std.runtime._asyncrt import _create_task

from max.runtime.asyncrt import task_id_for_device


def test_task_id_for_device_returns_int() raises:
    """Verify task_id_for_device returns a non-negative or -1 integer.

    The function delegates to MLRT_TaskIdForDevice.  Without a
    configured affinity map, the runtime may return -1 (no hint) or a valid
    worker index.  Either way, the return type must be Int and be >= -1.
    """
    print("== test_task_id_for_device_returns_int")
    var result = task_id_for_device(0)
    assert_true(result >= -1)


def test_create_task_with_affinity_runs_coroutine() raises:
    """Verify _create_task executes the coroutine to completion.

    The desired_worker_id hint is advisory; correctness must hold regardless
    of whether the runtime honours the hint.
    """
    print("== test_create_task_with_affinity_runs_coroutine")

    @__parameter
    async def compute() -> Int:
        return 42

    var worker_id = task_id_for_device(0)
    var task = _create_task(compute(), desired_worker_id=worker_id)
    # affinity task result == 42
    assert_equal(task.wait(), 42)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
