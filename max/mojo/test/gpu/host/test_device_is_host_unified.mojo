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
"""Tests `DeviceContext.is_host_unified()`.

Only the Apple direction is asserted by value. Whether a CUDA or HIP part is
unified depends on the model, not the api, so that expectation is pinned in
`GPUDeviceContextTest` where it is keyed off the device name.
"""

from max.gpu.host import DeviceContext
from std.testing import assert_true, TestSuite


def test_apple_gpus_are_unified() raises:
    with DeviceContext() as ctx:
        if ctx.api() != "metal":
            return
        assert_true(ctx.is_host_unified())


def test_query_succeeds_on_any_device() raises:
    # Catches a backend that reaches the unimplemented default and raises.
    with DeviceContext() as ctx:
        assert_true(ctx.is_host_unified() == ctx.is_host_unified())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
