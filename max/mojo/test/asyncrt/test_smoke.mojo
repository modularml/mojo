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
from max.gpu.host import DeviceAttribute, DeviceBuffer, DeviceContext
from std.testing import TestSuite, assert_equal


def _ownership_helper(
    var ctx: DeviceContext,
) raises -> DeviceContext:
    var ctx_copy = ctx
    print("local ctx_copy: " + ctx_copy.name())
    return ctx_copy


def _ownership_helper_buf[
    dtype: DType
](var buf: DeviceBuffer[dtype]) raises -> DeviceBuffer[dtype]:
    var buf_copy = buf
    print("local buf_copy: ", len(buf))
    return buf_copy


def _run_name(ctx: DeviceContext) raises:
    print("-")
    print("_run_name()")

    var ctx_name = ctx.name()
    print("ctx_name = ", ctx_name)

    var api = ctx.api()

    if api == "cpu" and "cpu" in ctx_name:
        return
    if api == "cuda" and "NVIDIA" in ctx_name:
        return
    if api == "hip" and "AMD" in ctx_name:
        return
    if api == "metal" and "Apple" in ctx_name:
        return

    raise "ctx_name malformed"


def _run_ownership_transfer(ctx: DeviceContext) raises:
    print("-")
    print("_run_ownership_transfer()")

    var ctx_copy = _ownership_helper(ctx)
    print("ctx_copy: ", ctx_copy.name())

    var buf = ctx.create_buffer_sync[.float32](32)
    print("buf: ", len(buf))
    var buf_copy = _ownership_helper_buf(buf)
    print("buf_copy: ", len(buf_copy))

    # Make sure buf survives to the end of the test function.
    _ = buf


def _run_device_info(ctx: DeviceContext) raises:
    print("-")
    print("_run_device_info()")

    var (free_before, total_before) = ctx.get_memory_info()

    var buf = ctx.create_buffer_sync[.float32](20 * 1024 * 1024)

    var (free_after, total_after) = ctx.get_memory_info()
    print(
        "Memory info (before -> after) - total: ",
        total_before,
        " -> ",
        total_after,
        " , free: ",
        free_before,
        " -> ",
        free_after,
    )

    # Make sure buf survives to the end of the test function.
    _ = buf


def _run_compute_capability(ctx: DeviceContext) raises:
    print("-")
    print("_run_compute_capability()")

    print("Compute capability: ", ctx.compute_capability())


def _run_get_attribute(ctx: DeviceContext) raises:
    print("-")
    print("_run_get_attribute()")
    print("clock_rate: ", ctx.get_attribute(DeviceAttribute.CLOCK_RATE))
    print("warp_size: ", ctx.get_attribute(DeviceAttribute.WARP_SIZE))


def _run_get_stream(ctx: DeviceContext) raises:
    print("-")
    print("_run_get_stream()")

    print("Getting the stream.")
    var stream = ctx.stream()
    print("Synchronizing on `stream`.")
    stream.synchronize()


def _run_peer_access(ctx: DeviceContext) raises:
    print("-")
    print("_run_peer_access()")

    assert_equal(ctx.can_access(ctx), False, "self access is not enabled")

    var num_gpus = DeviceContext.number_of_devices(api=ctx.api())
    print("Number of GPU devices: ", num_gpus)

    if num_gpus > 1:
        var peer = create_test_device_context(device_id=1)
        print("peer context on GPU[1]")

        if ctx.can_access(peer):
            ctx.enable_peer_access(peer)
            print("Enabled peer access.")


def test_smoke() raises:
    var ctx = create_test_device_context()
    print("-------")
    print("Running test_smoke(" + ctx.name() + "):")

    _run_name(ctx)
    _run_ownership_transfer(ctx)
    _run_device_info(ctx)
    _run_compute_capability(ctx)
    _run_get_attribute(ctx)
    _run_get_stream(ctx)
    _run_peer_access(ctx)

    print("Done.")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
