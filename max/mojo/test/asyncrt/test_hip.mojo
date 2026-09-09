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

from asyncrt_test_utils import create_test_device_context
from max.gpu.host import DeviceContext
from max.gpu.host._amdgpu_hip import HIP, hipDevice_t
from std.testing import TestSuite


def _run_hip_context(ctx: DeviceContext) raises:
    print("-")
    print("_run_hip_context()")

    var hip_ctx: hipDevice_t = HIP(ctx)
    print("hipDevice_t: " + String(hip_ctx))


def _run_hip_stream(ctx: DeviceContext) raises:
    print("-")
    print("_run_hip_stream()")

    print("Getting the stream.")
    var stream = ctx.stream()
    print("Synchronizing on `stream`.")
    stream.synchronize()
    var hip_stream = HIP(stream)
    print("hipStream_t: " + String(hip_stream))


def test_hip_context() raises:
    var ctx = create_test_device_context()
    _run_hip_context(ctx)


def test_hip_stream() raises:
    var ctx = create_test_device_context()
    _run_hip_stream(ctx)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
