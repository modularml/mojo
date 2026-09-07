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
"""Provides tracing utilities."""


from std.collections.list import List
from std.utils.coord import Coord
from std.collections.optional import Optional, OptionalReg
from std.ffi import external_call
from std.sys import stderr
from std.sys.defines import get_defined_int, is_defined

import std.logger.logger as logger

import max.gpu.host._tracing as gpu_tracing
from max.gpu.host import DeviceContext
from max.gpu.host._tracing import Color
from max.gpu.host._tracing import _end_range as _end_gpu_range
from max.gpu.host._tracing import _is_enabled as _gpu_is_enabled
from max.gpu.host._tracing import _is_enabled_details as _gpu_is_enabled_details
from max.gpu.host._tracing import _mark as _mark_gpu
from max.gpu.host._tracing import _start_range as _start_gpu_range

from std.utils import IndexList, Variant
from std.os import abort

comptime log = logger.Logger[logger.Level.INFO](fd=stderr, prefix="[OP] ")
"""Logger instance for operation tracing with INFO level and [OP] prefix."""


def get_safe_task_id(ctx: DeviceContext) -> OptionalReg[Int]:
    """Safely extract task_id from DeviceContext, returning None if null/invalid.

    Args:
        ctx: The device context to extract the task ID from.

    Returns:
        An OptionalReg containing the task ID if valid, None otherwise.
    """
    try:
        return OptionalReg(Int(ctx.id()))
    except:
        return None


def get_safe_task_id(ctx: Optional[DeviceContext]) -> OptionalReg[Int]:
    """Safely extract task_id from an optional `DeviceContext`, returning
    `None` if the context is absent or invalid.

    Args:
        ctx: The optional device context to extract the task ID from.

    Returns:
        An `OptionalReg` containing the task ID if `ctx` is set and the
        underlying handle is valid, `None` otherwise.
    """
    if not ctx:
        return None
    return get_safe_task_id(ctx.value())


def _build_info_asyncrt_max_profiling_level() -> OptionalReg[Int]:
    comptime if not is_defined["MODULAR_ASYNCRT_MAX_PROFILING_LEVEL"]():
        return None
    return get_defined_int["MODULAR_ASYNCRT_MAX_PROFILING_LEVEL"]()


# ===-----------------------------------------------------------------------===#
# TraceCategory
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct TraceCategory(Equatable, Intable, TrivialRegisterPassable):
    """An enum-like struct specifying the type of tracing to perform."""

    comptime OTHER = Self(0)
    """Other or uncategorized trace events."""
    comptime ASYNCRT = Self(1)
    """Asynchronous runtime trace events."""
    comptime MEM = Self(2)
    """Memory-related trace events."""
    comptime Kernel = Self(3)
    """Kernel execution trace events."""
    comptime MAX = Self(4)
    """MAX framework trace events."""

    var value: Int
    """The integer value representing the trace category. Used for bitwise operations
    when determining if profiling is enabled for a specific category."""

    @always_inline("nodebug")
    def __eq__(self, rhs: Self) -> Bool:
        """Compares for equality.

        Args:
            rhs: The value to compare.

        Returns:
            True if they are equal.
        """
        return self.value == rhs.value

    @always_inline("nodebug")
    def __ne__(self, rhs: Self) -> Bool:
        """Compares for inequality.

        Args:
            rhs: The value to compare.

        Returns:
            True if they are not equal.
        """
        return self.value != rhs.value

    @always_inline("nodebug")
    def __int__(self) -> Int:
        """Converts the trace category to an integer.

        Returns:
            The integer value of the trace category.
        """
        return self.value


# ===-----------------------------------------------------------------------===#
# TraceLevel
# ===-----------------------------------------------------------------------===#


struct TraceLevel(Comparable, TrivialRegisterPassable):
    """An enum-like struct specifying the level of tracing to perform."""

    comptime ALWAYS = Self(0)
    """Always trace at this level."""
    comptime OP = Self(1)
    """Operation-level tracing."""
    comptime THREAD = Self(2)
    """Thread-level tracing."""

    var value: Int
    """The integer value representing the trace level.

    Lower values indicate higher priority trace levels:
    - 0 (ALWAYS): Always traced
    - 1 (OP): Operation-level tracing
    - 2 (THREAD): Thread-level tracing
    """

    @always_inline
    def __init__(out self, value: Int):
        """Initializes a TraceLevel with the given integer value.

        Args:
            value: The integer value for the trace level.
        """
        self.value = value

    @always_inline("nodebug")
    def __eq__(self, rhs: Self) -> Bool:
        """Compares for equality.

        Args:
            rhs: The value to compare.

        Returns:
            True if they are equal.
        """
        return self.value == rhs.value

    @always_inline("nodebug")
    def __lt__(self, rhs: Self) -> Bool:
        """Performs less than comparison.

        Args:
            rhs: The value to compare.

        Returns:
            True if this value is less than to `rhs`.
        """
        return self.value < rhs.value

    @always_inline("nodebug")
    def __int__(self) -> Int:
        """Converts the trace level to an integer.

        Returns:
            The integer value of the trace level.
        """
        return self.value


# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#


@always_inline
def is_profiling_enabled[type: TraceCategory, level: TraceLevel]() -> Bool:
    """Returns True if the profiling is enabled for that specific type and
    level and False otherwise.

    Parameters:
        type: The trace category to check.
        level: The trace level to check.

    Returns:
        True if profiling is enabled for the specified type and level.
    """
    comptime kProfilingTypeWidthBits = 3

    comptime if level == TraceLevel.ALWAYS:
        return True

    comptime max_profiling_level = _build_info_asyncrt_max_profiling_level()
    if not max_profiling_level:
        return False

    return level <= TraceLevel(
        (max_profiling_level.value() >> (type.value * kProfilingTypeWidthBits))
        & ((1 << kProfilingTypeWidthBits) - 1)
    )


@always_inline
def is_profiling_disabled[type: TraceCategory, level: TraceLevel]() -> Bool:
    """Returns False if the profiling is enabled for that specific type and
    level and True otherwise.

    Parameters:
        type: The trace category to check.
        level: The trace level to check.

    Returns:
        True if profiling is disabled for the specified type and level.
    """
    return not is_profiling_enabled[type, level]()


@always_inline
def _is_gpu_profiler_enabled[type: TraceCategory, level: TraceLevel]() -> Bool:
    """Returns True if the e2e kernel profiling is enabled. Note that we always
    prefer to use llcl profiling if they are enabled."""
    return (
        is_profiling_disabled[type, level]()
        and level <= TraceLevel.OP
        and _gpu_is_enabled()
    )


@always_inline
def _is_gpu_profiler_detailed_enabled[
    type: TraceCategory, level: TraceLevel
]() -> Bool:
    """Returns True if the e2e detailed kernel profiling is enabled. Note that
    we always prefer to use llcl profiling if they are enabled."""
    return (
        is_profiling_disabled[type, level]()
        and level <= TraceLevel.OP
        and _gpu_is_enabled_details()
    )


@always_inline
def _is_op_logging_enabled[level: TraceLevel]() -> Bool:
    comptime if logger.DEFAULT_LEVEL == logger.Level.NOTSET:
        return False

    return level <= TraceLevel.OP


@always_inline
def _is_tracy_enabled() -> Bool:
    """Returns whether the Tracy bridge is enabled in CompilerRT."""
    return external_call["KGEN_CompilerRT_TracyIsEnabled", Int]() != 0


@always_inline
def _is_range_recording() -> Bool:
    """Returns whether the MAX profiler is actively recording ranges.

    True while a profiler trace of either origin — an explicit session enable
    or an externally requested on-demand capture — is live. One relaxed
    atomic load of a process-global gate behind the FFI call, so it is safe
    on hot paths.
    """
    return external_call["KGEN_CompilerRT_RangeIsRecording", Int]() != 0


@always_inline
def _is_mojo_profiling_enabled[level: TraceLevel]() -> Bool:
    """Returns whether Mojo profiling is enabled for the specified level."""
    return is_profiling_enabled[TraceCategory.MAX, level]()


@always_inline
def _is_mojo_profiling_disabled[level: TraceLevel]() -> Bool:
    """Returns whether Mojo profiling is disabled for the specified level."""
    return is_profiling_disabled[TraceCategory.MAX, level]()


@always_inline
def _get_enabled_tracing_systems[level: TraceLevel]() -> List[String]:
    """Returns a list of enabled tracing system names.

    The MAX profiler and the external profiler annotation bridge are
    deliberately absent: neither participates in the one-system-at-a-time
    accounting this list backs, both recording additively alongside every
    system here except "Op Logging", whose branch short-circuits
    `__enter__` before their
    branches are reached.

    Returns:
        A list of strings naming the tracing systems that are enabled.
        Possible values: "AsyncRT", "GPU", "Tracy", "Op Logging".
    """
    var enabled_systems = List[String]()

    # Check AsyncRT profiling
    var asyncrt_level = _build_info_asyncrt_max_profiling_level()
    if asyncrt_level and asyncrt_level.value() > 0:
        enabled_systems.append("AsyncRT")

    # Check GPU profiling
    if _gpu_is_enabled():
        enabled_systems.append("GPU")

    # Check Tracy profiling
    if _is_tracy_enabled():
        enabled_systems.append("Tracy")

    # Check op logging
    if _is_op_logging_enabled[level]():
        enabled_systems.append("Op Logging")

    return enabled_systems^


@always_inline
def trace_arg(name: String, shape: IndexList) -> String:
    """Helper to stringify the type and shape of a kernel argument for tracing.

    Args:
        name: The name of the argument.
        shape: The shape of the argument.

    Returns:
        A string representation of the argument with its shape.
    """
    var s = name + "="
    for i in range(len(shape)):
        if i != 0:
            s += "x"
        s += String(shape[i])
    return s


@always_inline
def trace_arg(name: String, shape: Coord) -> String:
    """Helper to stringify the type and shape of a kernel argument for tracing.

    Args:
        name: The name of the argument.
        shape: The shape of the argument.

    Returns:
        A string representation of the argument with its shape.
    """
    var s = name + "="
    comptime for i in range(shape.rank):
        if i != 0:
            s += "x"
        s += String(shape[i].value())
    return s


@always_inline
def trace_arg(name: String, shape: IndexList, dtype: DType) -> String:
    """Helper to stringify the type and shape of a kernel argument for tracing.

    Args:
        name: The name of the argument.
        shape: The shape of the argument.
        dtype: The data type of the argument.

    Returns:
        A string representation of the argument with its shape and data type.
    """
    return String(t"{trace_arg(name, shape)}x{dtype}")


@always_inline
def trace_arg(name: String, shape: Coord, dtype: DType) -> String:
    """Helper to stringify the type and shape of a kernel argument for tracing.

    Args:
        name: The name of the argument.
        shape: The shape of the argument.
        dtype: The data type of the argument.

    Returns:
        A string representation of the argument with its shape and data type.
    """
    return String(t"{trace_arg(name, shape)}x{dtype}")


# ===-----------------------------------------------------------------------===#
# Trace
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct Trace[
    level: TraceLevel,
    *,
    category: TraceCategory = TraceCategory.MAX,
    target: Optional[StaticString] = None,
](ImplicitlyCopyable):
    """An object representing a specific trace.

    This struct provides functionality for creating and managing trace events
    for profiling and debugging purposes.

    Parameters:
        level: The trace level to use.
        category: The trace category to use (defaults to TraceCategory.MAX).
        target: Optional target information to include in the trace.
    """

    var _name_value: Variant[String, StaticString]
    var int_payload: OptionalReg[Int]
    """Optional integer payload, typically used for task IDs that are appended to trace names."""

    var detail: String
    """Additional details about the trace event, included when detailed tracing is enabled."""

    var event_id: Int
    """Unique identifier for the trace event, assigned when the trace begins."""

    var parent_id: Int
    """Identifier of the parent trace event, used for creating hierarchical trace relationships."""

    var color: Optional[Color]
    """Color of the trace span in NSight Systems viewer, only used for NVTX markers."""

    var _tracy_ctx: UInt64
    """Packed Tracy context id when a Tracy zone is active via CompilerRT."""

    var _range_opened: Bool
    """True iff `__enter__` opened a MAX profiler range; `__exit__` pairs it.

    Captured at `__enter__` so a profiler stop racing with the scope cannot
    leave the profiler's per-thread range stack unbalanced.

    The profiler pairs begin/end on a per-thread stack, so a scope that
    suspends and resumes on another thread (an `await` inside the `with`)
    records a dropped or misattributed span, and copying an entered `Trace`
    (as with `_tracy_ctx`) emits an extra end that can close an unrelated
    open range. Keep profiler-visible scopes on one thread and uncopied.
    """

    var _tool_pushed: Bool
    """True iff `__enter__` pushed an external profiler annotation range;
    `__exit__` pops it.

    External profiler annotations mark an external profiler's trace (e.g.
    an attached system profiler) rather than MAX's own. The vendor
    annotation APIs pair push/pop on a per-thread stack, so the same
    single-thread and no-copy caveats as `_range_opened` apply.
    """

    # This constructor is intentionally hidden because Variant is too flexible
    # about what it allows and we want to ensure that only StaticString or
    # String are used.
    @always_inline
    def __init__(
        out self,
        *,
        var _name_value: Variant[String, StaticString],
        detail: String = "",
        parent_id: Int = 0,
        task_id: OptionalReg[Int] = None,
        color: Optional[Color] = None,
    ):
        """Creates a Mojo trace with the given name.

        Args:
            _name_value: The name that is used to identify this Mojo trace.
            detail: Details of the trace entry.
            parent_id: Parent to associate the trace with. Trace name will be
                appended to parent name. 0 (default) indicates no parent.
            task_id: Int that is appended to name.
            color: Color of the trace span when visualized.
        """

        self.event_id = 0  # Known only when begin recording in __enter__
        self.parent_id = parent_id
        self.color = color

        # Debug assert the AsyncRT profiler => StaticString invariant for now,
        # to avoid making this raising.
        assert (
            is_profiling_disabled[Self.category, Self.level]()
            or _name_value.isa[StaticString]()
        ), "the AsyncRT profiler only supports `StaticString` names"

        # Validate that only one tracing system is enabled
        var enabled_systems = _get_enabled_tracing_systems[Self.level]()
        debug_assert(
            len(enabled_systems) <= 1,
            "only one tracing system should be enabled at a time, got: ",
            StaticString(", ").join(enabled_systems),
        )

        # Always initialize the tracy context to zero: it's set in __enter__.
        self._tracy_ctx = 0
        # Set in __enter__ iff a MAX profiler range was opened.
        self._range_opened = False
        # Set in __enter__ iff an external profiler annotation range was
        # pushed.
        self._tool_pushed = False

        comptime if _is_gpu_profiler_enabled[Self.category, Self.level]():
            self._name_value = _name_value^

            comptime if _gpu_is_enabled_details():
                self.detail = detail
            else:
                self.detail = ""
            self.int_payload = None
        elif (
            is_profiling_enabled[Self.category, Self.level]()
            or _is_op_logging_enabled[Self.level]()
        ):
            self._name_value = _name_value^
            self.detail = detail

            comptime if Self.target:
                if self.detail:
                    self.detail += ";"
                self.detail += String("target=", Self.target.value())
            self.int_payload = task_id
        else:
            self._name_value = _name_value^
            self.detail = ""
            self.int_payload = None

    @always_inline
    def __init__(
        out self,
        var name: String,
        detail: String = "",
        parent_id: Int = 0,
        color: Optional[Color] = None,
        *,
        task_id: OptionalReg[Int] = None,
    ):
        """Creates a Mojo trace with the given string name.

        Args:
            name: The name that is used to identify this Mojo trace.
            detail: Details of the trace entry.
            parent_id: Parent to associate the trace with. Trace name will be
                appended to parent name. 0 (default) indicates no parent.
            color: Color of the trace span when visualized.
            task_id: Int that is appended to name.
        """
        self = Self(
            _name_value=name^,
            detail=detail,
            parent_id=parent_id,
            task_id=task_id,
            color=color,
        )

    @always_inline
    def __init__(
        out self,
        name: StaticString,
        detail: String = "",
        parent_id: Int = 0,
        color: Optional[Color] = None,
        *,
        task_id: OptionalReg[Int] = None,
    ):
        """Creates a Mojo trace with the given static string name.

        Args:
            name: The name that is used to identify this Mojo trace.
            detail: Details of the trace entry.
            parent_id: Parent to associate the trace with. Trace name will be
                appended to parent name. 0 (default) indicates no parent.
            color: Color of the trace span when visualized.
            task_id: Int that is appended to name.
        """
        self = Self(
            _name_value=name,
            detail=detail,
            parent_id=parent_id,
            task_id=task_id,
            color=color,
        )

    @always_inline
    def __init__(
        out self,
        name: StringLiteral,
        detail: String = "",
        parent_id: Int = 0,
        color: Optional[Color] = None,
        *,
        task_id: OptionalReg[Int] = None,
    ):
        """Creates a Mojo trace with the given string literal name.

        Args:
            name: The name that is used to identify this Mojo trace.
            detail: Details of the trace entry.
            parent_id: Parent to associate the trace with. Trace name will be
                appended to parent name. 0 (default) indicates no parent.
            color: Color of the trace span when visualized.
            task_id: Int that is appended to name.
        """
        self = Self(
            _name_value=StaticString(name),
            detail=detail,
            parent_id=parent_id,
            task_id=task_id,
            color=color,
        )

    @always_inline
    def __enter__(mut self) raises:
        """Enters the trace context.

        This begins recording of the trace event.

        Raises:
            If the operation fails.
        """

        comptime if _is_op_logging_enabled[Self.level]():
            # Since Mojo does not support module-level globals yet, we need to
            # put this atomic counter variable in C++ code.
            self.event_id = external_call["KGEN_CompilerRT_GetNextOpId", Int]()
            self._emit_op_log("LAUNCH")
            return

        # Record a MAX profiler range while a profiler trace is live. Unlike
        # the branches below this one does not return: the profiler's trace
        # has a different audience (Perfetto/HTA/on-demand capture) than the
        # backends below (Tracy, NVTX/nsys, the AsyncRT time-trace), so
        # whichever of those is enabled must still record its own span.
        # THREAD-level spans are excluded — their per-task volume would swamp
        # the profiler's span budget.
        comptime if Self.level <= TraceLevel.OP:
            if _is_range_recording():
                var color_val = UInt32(
                    Int(self.color.value())
                ) if self.color else 0
                if self._name_value.isa[StaticString]():
                    external_call["KGEN_CompilerRT_RangeBegin", NoneType](
                        self._name_value[StaticString].as_bytes().unsafe_ptr(),
                        self._name_value[StaticString].byte_length(),
                        color_val,
                    )
                else:
                    external_call["KGEN_CompilerRT_RangeBegin", NoneType](
                        self._name_value[String].as_bytes().unsafe_ptr(),
                        self._name_value[String].byte_length(),
                        color_val,
                    )
                self._range_opened = True

        # Annotate an external profiler's trace through the
        # external profiler annotation bridge. Like the MAX profiler branch
        # above this is runtime-gated (the push reports whether anything was
        # emitted) and additive: the backends below still record their own
        # span. Compiled out when this build already emits vendor
        # annotations directly through the GPU profiler branch below, so
        # those builds never double-annotate.
        # Annotations carry the span name only; detail strings are not
        # forwarded.
        comptime if Self.level <= TraceLevel.OP and not _is_gpu_profiler_enabled[
            Self.category, Self.level
        ]():
            var tool_color = UInt32(
                Int(self.color.value())
            ) if self.color else 0
            if self._name_value.isa[StaticString]():
                self._tool_pushed = (
                    external_call[
                        "KGEN_CompilerRT_ExternalProfilerAnnotationPush", Int
                    ](
                        self._name_value[StaticString].as_bytes().unsafe_ptr(),
                        self._name_value[StaticString].byte_length(),
                        tool_color,
                    )
                    != 0
                )
            else:
                self._tool_pushed = (
                    external_call[
                        "KGEN_CompilerRT_ExternalProfilerAnnotationPush", Int
                    ](
                        self._name_value[String].as_bytes().unsafe_ptr(),
                        self._name_value[String].byte_length(),
                        tool_color,
                    )
                    != 0
                )

        # Start a Tracy zone if the bridge is available.
        if _is_tracy_enabled():
            var name_str = self.name()
            var color_val = UInt32(Int(self.color.value())) if self.color else 0
            self._tracy_ctx = external_call[
                "KGEN_CompilerRT_TracyZoneBegin", UInt64
            ](
                name_str.as_bytes().unsafe_ptr(),
                name_str.byte_length(),
                color_val,
            )
            return

        comptime if _is_gpu_profiler_enabled[Self.category, Self.level]():
            comptime if _gpu_is_enabled_details():
                # Convert to String since nvtx range APIs copy messages anyway.
                # TODO(KERN-1052): optimize by exposing explicit string
                # registration.
                self.event_id = Int(
                    _start_gpu_range(
                        message=String(
                            self.name(),
                            (String("/", self.detail) if self.detail else ""),
                        ),
                        category=Int(Self.category),
                        color=self.color,
                    )
                )
            else:
                self.event_id = Int(
                    _start_gpu_range(
                        message=self.name(),
                        category=Int(Self.category),
                        color=self.color,
                    )
                )
            return

        comptime if is_profiling_disabled[Self.category, Self.level]():
            return

        # The tracing builtins below expect the string to live beyond begin/end
        # calls, so we have to pass an inner pointer into the representation.
        #
        # IMPORTANT: since the AsyncRT profiler only supports `StaticString`
        # names, `self._name_value` must be `StaticString` when
        # `is_profiling_enabled()` is set.
        var name_str_ptr = (
            self._name_value[StaticString].as_bytes().unsafe_ptr()
        )
        var name_str_len = self._name_value[StaticString].byte_length()

        if self.detail:
            # 1. If there is a detail string we must heap allocate the string
            #    because it presumably contains information only known at
            #    runtime.

            # Begins recording the trace range from the stack. This is only enabled if the AsyncRT
            # profiling is enabled.
            self.event_id = external_call[
                "KGEN_CompilerRT_TimeTraceProfilerBeginDetail", Int
            ](
                name_str_ptr,
                name_str_len,
                self.detail.as_bytes().unsafe_ptr(),
                self.detail.byte_length(),
                self.parent_id,
            )
        elif self.int_payload:
            # 2. If there is a task id, use the profiler API to create task:id
            #    labels without copying.
            self.event_id = external_call[
                "KGEN_CompilerRT_TimeTraceProfilerBeginTask", Int
            ](
                name_str_ptr,
                name_str_len,
                self.parent_id,
                self.int_payload.value(),
            )
        else:
            # 3. In the common case without a task id or detail string, create
            #    a profiler event without copying until explicit intern call.
            self.event_id = external_call[
                "KGEN_CompilerRT_TimeTraceProfilerBegin", Int
            ](
                name_str_ptr,
                name_str_len,
                self.parent_id,
            )

        external_call[
            "KGEN_CompilerRT_TimeTraceProfilerSetCurrentId", NoneType
        ](self.event_id)

    @always_inline
    def __exit__(self):
        """Exits the trace context.

        This finishes recording of the trace event.
        """

        comptime if _is_op_logging_enabled[Self.level]():
            self._emit_op_log("COMPLETE")
            return

        # Close the MAX profiler range opened in __enter__ (see _range_opened).
        if self._range_opened:
            external_call["KGEN_CompilerRT_RangeEnd", NoneType]()

        # Pop the external profiler annotation range pushed in __enter__
        # (pop-iff-pushed: popping without a matching push would close an
        # unrelated outer range on this thread's vendor annotation stack).
        if self._tool_pushed:
            external_call[
                "KGEN_CompilerRT_ExternalProfilerAnnotationPop", NoneType
            ]()

        # End Tracy zone early to guarantee pairing even on early returns.
        if self._tracy_ctx != 0:
            external_call["KGEN_CompilerRT_TracyZoneEnd", NoneType](
                self._tracy_ctx
            )
            return

        comptime if _is_gpu_profiler_enabled[Self.category, Self.level]():
            try:
                _end_gpu_range(gpu_tracing.RangeID(self.event_id))
            except:
                abort("GPU tracing failure")
            return

        comptime if is_profiling_disabled[Self.category, Self.level]():
            return
        if self.event_id == 0:
            return
        external_call["KGEN_CompilerRT_TimeTraceProfilerEnd", NoneType](
            self.event_id
        )
        external_call[
            "KGEN_CompilerRT_TimeTraceProfilerSetCurrentId", NoneType
        ](0)

    @always_inline
    def mark(self) raises:
        """Marks the tracer with the info at the specific point of time.

        This creates a point event in the trace timeline rather than a range.
        Not recorded by the MAX profiler, whose range API has no point-event
        equivalent; forwarded to the external profiler annotation bridge when
        enabled.

        Raises:
            If the operation fails.
        """

        comptime if _is_gpu_profiler_enabled[Self.category, Self.level]():
            var message = self.name()

            comptime if _gpu_is_enabled_details():
                if self.detail:
                    message += String("/", self.detail)

            _mark_gpu(message=message)
        elif Self.level <= TraceLevel.OP:
            # Forward to the external profiler annotation bridge in builds
            # without the direct GPU-profiler marker above. The enabled
            # probe keeps the message materialization off the disabled path.
            if (
                external_call[
                    "KGEN_CompilerRT_ExternalProfilerAnnotationIsEnabled", Int
                ]()
                != 0
            ):
                var message = self.name()
                external_call[
                    "KGEN_CompilerRT_ExternalProfilerAnnotationMark", NoneType
                ](
                    message.as_bytes().unsafe_ptr(),
                    message.byte_length(),
                    UInt32(Int(self.color.value())) if self.color else 0,
                )

    @always_inline
    def name(self) -> String:
        """Returns the name of the trace.

        Returns:
            The name of the trace as a String.
        """
        return String(self._name_value[StaticString]) if self._name_value.isa[
            StaticString
        ]() else self._name_value[String]

    # WAR: passing detail_fn to __init__ causes internal compiler crash
    @staticmethod
    @always_inline
    def _get_detail_str(detail_fn: Some[def() -> String]) -> String:
        """Return the detail str when tracing is enabled and an empty string otherwise.
        """

        comptime if (
            is_profiling_enabled[Self.category, Self.level]()
            or _is_gpu_profiler_detailed_enabled[Self.category, Self.level]()
        ):
            return detail_fn()
        else:
            return ""

    @always_inline
    def start(mut self) raises:
        """Start recording trace event.

        This begins recording of the trace event, similar to __enter__.

        Raises:
            If the operation fails.
        """
        self.__enter__()

    @always_inline
    def end(mut self) raises:
        """End recording trace event.

        This finishes recording of the trace event, similar to __exit__.

        Raises:
            If the operation fails.
        """
        self.__exit__()

    def _emit_op_log(self, op_name: StringSlice):
        """
        Emit a log message for op tracing to stderr.
        """
        var detail = self.detail
        if self.int_payload:
            detail += String(":", self.int_payload.value())
        log.info(
            op_name,
            " ",
            self.name(),
            " [id=",
            self.event_id,
            "] ",
            detail,
            sep="",
        )


def get_current_trace_id[level: TraceLevel]() -> Int:
    """Returns the id of last created trace entry on the current thread.

    Parameters:
        level: The trace level to check.

    Returns:
        The ID of the current trace if profiling is enabled, otherwise 0.
    """

    comptime if _is_mojo_profiling_enabled[level]():
        return external_call[
            "KGEN_CompilerRT_TimeTraceProfilerGetCurrentId", Int
        ]()
    else:
        return 0
