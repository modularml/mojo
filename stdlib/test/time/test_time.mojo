# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
# RUN: %mojo %s

from sys import os_is_windows
from time import now, sleep, time_function

from testing import assert_true


@always_inline
@parameter
fn time_me():
    sleep(1)


@always_inline
@parameter
fn time_me_templated[
    type: DType,
]():
    time_me()
    return


# Check that time_function works on templated function
fn time_templated_function[
    type: DType,
]() -> Int:
    return time_function[time_me_templated[type]]()


fn time_capturing_function(iters: Int) -> Int:
    @parameter
    fn time_fn():
        sleep(1)

    return time_function[time_fn]()


fn test_time() raises:
    alias ns_per_sec = 1_000_000_000

    assert_true(now() > 0)

    var t1 = time_function[time_me]()
    assert_true(t1 > 1 * ns_per_sec)
    assert_true(t1 < 10 * ns_per_sec)

    var t2 = time_templated_function[DType.float32]()
    assert_true(t2 > 1 * ns_per_sec)
    assert_true(t2 < 10 * ns_per_sec)

    var t3 = time_capturing_function(42)
    assert_true(t3 > 1 * ns_per_sec)
    assert_true(t3 < 10 * ns_per_sec)

    # test now() directly since time_function doesn't use now on windows
    var t4 = now()
    time_me()
    var t5 = now()
    assert_true((t5 - t4) > 1 * ns_per_sec)
    assert_true((t5 - t4) < 10 * ns_per_sec)


def main():
    test_time()
