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

"""Tests for `Trace`'s MAX profiler range branch and the external profiler
annotation bridge.

Exercises the runtime-gated `__enter__`/`__exit__` path that records spans
through the `KGEN_CompilerRT_Range*` FFI bridge, plus the suppressed path of
the `KGEN_CompilerRT_ExternalProfilerAnnotation*` external profiler annotation
bridge. No profiler backend is available in this test environment and
external profiler annotation is never requested, so the gates never rise
and these tests pin the safety properties: the disabled fast path, clean
unwind across raises, and profiler state staying undisturbed. Span content
is asserted end-to-end in `max/tests/internal/profiler/`.
"""

from std.ffi import external_call
from std.testing import assert_equal, assert_false, assert_true

from max.runtime.tracing import Trace, TraceLevel


def _is_recording() -> Bool:
    return external_call["KGEN_CompilerRT_RangeIsRecording", Int]() != 0


def test_not_recording_at_startup() raises:
    assert_false(_is_recording())


def _static_name_scope() raises:
    with Trace[TraceLevel.OP]("static-name-span"):
        pass


def _string_name_scope() raises:
    var name = String("string-name-span-", 42)
    with Trace[TraceLevel.OP](name):
        pass


def _nested_scope() raises:
    with Trace[TraceLevel.OP]("outer"):
        with Trace[TraceLevel.OP]("inner"):
            pass


def _thread_level_scope() raises:
    # THREAD-level traces compile the profiler-range branch out entirely;
    # entering one must still be safe.
    with Trace[TraceLevel.THREAD]("thread-span"):
        pass


def test_trace_scopes_while_not_recording() raises:
    """Trace scopes are no-ops for the profiler while nothing records."""
    assert_false(_is_recording())
    _static_name_scope()
    _string_name_scope()
    _nested_scope()
    _thread_level_scope()
    assert_false(_is_recording())


def test_trace_scopes_with_profiler_enabled() raises:
    """Enable intent alone must not make Trace open unpaired ranges.

    Without a backend the recording gate stays down even after enable(), so
    `__enter__` must not latch `_range_opened` and `__exit__` must not emit a
    stray RangeEnd. Observable here as: scopes run cleanly and enable state
    survives them.
    """
    external_call["KGEN_CompilerRT_RangeEnable", NoneType]()
    try:
        assert_false(_is_recording())
        _static_name_scope()
        _string_name_scope()
        _nested_scope()
        assert_equal(1, external_call["KGEN_CompilerRT_RangeIsEnabled", Int]())
    finally:
        external_call["KGEN_CompilerRT_RangeDisable", NoneType]()
    assert_equal(0, external_call["KGEN_CompilerRT_RangeIsEnabled", Int]())


def _raise_inside_scope() raises:
    with Trace[TraceLevel.OP]("raising-span"):
        raise Error("boom")


def test_trace_exits_when_body_raises() raises:
    """A raise inside the `with` block still runs `__exit__` cleanly."""
    var raised = False
    try:
        _raise_inside_scope()
    except:
        raised = True
    assert_true(raised)
    assert_false(_is_recording())


def test_external_profiler_annotation_suppressed_without_shim() raises:
    """With annotation never requested the bridge suppresses pushes.

    MODULAR_ENABLE_PROFILING is unset in this test's environment and no shim
    is available, so a push must report that nothing was emitted and a bare
    pop must be a safe no-op.
    """
    assert_equal(
        0,
        external_call[
            "KGEN_CompilerRT_ExternalProfilerAnnotationIsEnabled", Int
        ](),
    )
    var name = StaticString("tool-span")
    assert_equal(
        0,
        external_call["KGEN_CompilerRT_ExternalProfilerAnnotationPush", Int](
            name.as_bytes().unsafe_ptr(), name.byte_length(), UInt32(0)
        ),
    )
    external_call["KGEN_CompilerRT_ExternalProfilerAnnotationPop", NoneType]()


def main() raises:
    test_not_recording_at_startup()
    test_trace_scopes_while_not_recording()
    test_trace_scopes_with_profiler_enabled()
    test_trace_exits_when_body_raises()
    test_external_profiler_annotation_suppressed_without_shim()
