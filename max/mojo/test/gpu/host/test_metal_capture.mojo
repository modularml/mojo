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

from max.gpu.host import DeviceContext
from max.gpu.host._metal_capture import (
    _end_metal_trace_capture,
    _set_metal_gpu_print_enabled,
    _start_metal_trace_capture,
)
from std.pathlib import Path
from std.tempfile import gettempdir
from std.testing import assert_true


def test_end_without_start_raises(ctx: DeviceContext) raises:
    var raised = False
    var msg = String("")
    try:
        _end_metal_trace_capture(ctx)
    except e:
        raised = True
        msg = String(e)
    assert_true(raised, "end with no active capture must raise")
    # The message differs by backend ("no Metal capture is in progress" on a
    # Metal context, "Not a MetalContext" otherwise); assert only that a
    # non-empty, actionable message was raised.
    assert_true(msg.byte_length() > 0, "raised error must carry a message")


def test_print_lever_toggles(ctx: DeviceContext) raises:
    # Track disable and re-enable independently so a failure localizes to the
    # specific call rather than a conflated boolean.
    var disable_raised = False
    try:
        _set_metal_gpu_print_enabled(ctx, False)
    except:
        disable_raised = True

    var enable_raised = False
    try:
        _set_metal_gpu_print_enabled(ctx, True)
    except:
        enable_raised = True

    if ctx.api() == "metal":
        assert_true(not disable_raised, "disabling print must succeed on Metal")
        assert_true(
            not enable_raised, "re-enabling print must succeed on Metal"
        )
    else:
        assert_true(disable_raised, "disabling print must raise off Metal")
        assert_true(enable_raised, "re-enabling print must raise off Metal")


def test_capture_roundtrip(ctx: DeviceContext) raises:
    var tmp = gettempdir().value()
    var trace = tmp + "/modular_mojo_capture.gputrace"
    var started = True
    try:
        _start_metal_trace_capture(ctx, trace)
    except:
        # Capture unavailable (MTL_CAPTURE_ENABLED unset, or unsupported on
        # this OS) — skip gracefully.
        started = False
    if not started:
        return
    _end_metal_trace_capture(ctx)
    assert_true(Path(trace).exists(), "capture must produce a .gputrace file")


def main() raises:
    var ctx = DeviceContext()
    test_end_without_start_raises(ctx)
    test_print_lever_toggles(ctx)
    if ctx.api() == "metal":
        test_capture_roundtrip(ctx)
