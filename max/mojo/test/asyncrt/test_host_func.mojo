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
"""Tests for `DeviceStream.enqueue_host_func` (CUDA host callback)."""

from asyncrt_test_utils import create_test_device_context
from std.atomic import Atomic
from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_equal


def _bump_counter(user_data: OpaquePointer[MutAnyOrigin]):
    """Host-side CUDA callback: atomically increments the counter behind
    `user_data`. Runs on a driver thread with no GIL / CUDA context.
    """
    var counter_ptr = user_data.unsafe_bitcast[Int32]()
    _ = Atomic[Int32].fetch_add(counter_ptr, 1)


def test_enqueue_host_func() raises:
    var ctx = create_test_device_context()
    var stream = ctx.stream()

    var counter = Atomic[Int32](0)
    var counter_ptr = Pointer(to=counter).unsafe_bitcast[NoneType]()
    var counter_opaque = rebind[OpaquePointer[MutAnyOrigin]](counter_ptr)

    comptime N = 4
    for _ in range(N):
        stream.enqueue_host_func(_bump_counter, counter_opaque)

    stream.synchronize()
    assert_equal(Int(counter.load()), N)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
