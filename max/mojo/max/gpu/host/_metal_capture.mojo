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
"""Programmatic Metal GPU frame capture.

Brackets a region of GPU work and writes a `.gputrace` file for offline replay
in the Metal debugger. Capture requires `MTL_CAPTURE_ENABLED=1` set before the
process starts. While a capture is active (and while print is disabled via
`_set_metal_gpu_print_enabled`), Metal GPU `print` output is suppressed so the
captured command buffers are replayable.
"""

from std.ffi import CStringSlice, external_call

from max.gpu.host.device_context import (
    DeviceContext,
    _CString,
    _DeviceContextPtr,
    _checked,
)


def _start_metal_trace_capture(ctx: DeviceContext, path: String) raises:
    """Starts a Metal GPU capture writing to `path` (a `.gputrace` file).

    Args:
        ctx: The Metal device context whose work is captured.
        path: Output file path; use a `.gputrace` extension for replay.

    Raises:
        If capture is unavailable (set `MTL_CAPTURE_ENABLED=1` before launch),
        a capture is already active, or `ctx` is not a Metal context.
    """
    var path_copy = path
    _checked(
        external_call[
            "AsyncRT_DeviceContext_startMetalTraceCapture",
            _CString[],
            _DeviceContextPtr[mut=True],
            CStringSlice[origin_of(path_copy)],
        ](ctx._handle, path_copy.as_c_string_slice())
    )


def _end_metal_trace_capture(ctx: DeviceContext) raises:
    """Ends the active Metal GPU capture and finalizes the `.gputrace` file.

    Args:
        ctx: The Metal device context with an active capture.

    Raises:
        If no capture is active or `ctx` is not a Metal context.
    """
    _checked(
        external_call[
            "AsyncRT_DeviceContext_stopMetalTraceCapture",
            _CString[],
            _DeviceContextPtr[mut=True],
        ](ctx._handle)
    )


def _set_metal_gpu_print_enabled(ctx: DeviceContext, enabled: Bool) raises:
    """Enables or disables Metal GPU `print` (os_log) for this context.

    Disabling removes the per-command-buffer log state, which both enables
    replayable captures and avoids `os_log` overhead during perf measurement.

    Args:
        ctx: The Metal device context.
        enabled: `False` to disable GPU print, `True` to re-enable it.

    Raises:
        If `ctx` is not a Metal context.
    """
    _checked(
        external_call[
            "AsyncRT_DeviceContext_setMetalPrintEnabled",
            _CString[],
            _DeviceContextPtr[mut=True],
            Bool,
        ](ctx._handle, enabled)
    )
