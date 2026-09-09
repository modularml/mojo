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
"""Mojo `Range` context manager for libkineto activity spans.

`Range` wraps the `KGEN_CompilerRT_Range{Begin,End,IsEnabled}` FFI bridge from
`Mojo/lib/CompilerRT/RangeBridge.cpp`. Use it as a context manager to emit a
libkineto activity span: `__enter__` opens the span and `__exit__` closes it.

`Range` mirrors `max.runtime.tracing.Trace`: the span is opened in `__enter__`
and closed in `__exit__` rather than in `__init__`/`__deinit__`. This sidesteps
Mojo's ASAP destruction: a `var span = Range(...)` whose handle is never read
again would otherwise be destroyed immediately after construction, closing the
span before any work inside it ran. The `with` statement keeps the object alive
for the full block, so begin/end pair with block entry/exit.

Off-cost matches the C++ `M::Profiling::RangeScope`: one relaxed atomic load
through the FFI when the profiler is disabled. The `_began` flag is captured at
`__enter__` so a `disable()` racing with the scope cannot leave a libkineto
span begun without a matching end (mirrors the C++ RAII discipline).
"""

from std.ffi import external_call


@always_inline
def is_enabled() -> Bool:
    """Reports whether the libkineto profiler is currently enabled.

    Cheap enough to call on the hot path to elide expensive trace-name
    materialization on the disabled fast path. Wraps
    `KGEN_CompilerRT_RangeIsEnabled`.

    Returns:
        True while the libkineto profiler is enabled, False otherwise.

    Example:

    ```mojo
    from profiling_range import Range, is_enabled

    def dispatch_kernel() raises:
        if is_enabled():
            with Range("hot_kernel"):
                launch()
        else:
            launch()
    ```
    """
    return external_call["KGEN_CompilerRT_RangeIsEnabled", Int32]() != 0


struct Range(ImplicitlyCopyable):
    """A libkineto activity span recorded via the CompilerRT bridge.

    Use `Range` as a context manager: `__enter__` opens a libkineto activity
    span when the profiler is enabled and `__exit__` closes it. The span is
    bound to the `with` block, so it covers exactly the work inside the block:

    ```mojo
    from profiling_range import Range

    def build_graph() raises:
        with Range("graph_compile"):
            ...  # graph build code runs inside the span
    ```

    `max.runtime.tracing.Trace` now records libkineto spans natively when the
    profiler is live, so new instrumentation should prefer `Trace` — it also
    feeds the other tracing backends. `Range` remains for code that wants a
    profiler-only span with no `Trace` machinery.
    """

    var _name: StaticString
    """The span name, captured at construction and passed to `RangeBegin` in
    `__enter__`."""

    var _color: UInt32
    """24-bit RGB color forwarded to `RangeBegin` in `__enter__`."""

    var _began: Bool
    """Internal: true iff `__enter__` told libkineto to open a span.

    Captured in `__enter__` so `__exit__` pairs the begin/end calls
    symmetrically even if `M::Profiling::disable()` runs on another thread
    mid-scope. The underscore prefix marks this as private; direct mutation by
    external code would unbalance libkineto's bookkeeping.
    """

    @always_inline
    def __init__(out self, name: StaticString, color: UInt32 = 0):
        """Records the span name and color; opening is deferred to `__enter__`.

        Args:
            name: The span name. Passed straight through to libkineto in
                `__enter__`. `StaticString` accepts both string literals and
                runtime `StaticString` values; this matches the convention used
                by `max.runtime.tracing.Trace`.
            color: 24-bit RGB color for trace viewers that honor it.
                Defaults to 0 (libkineto's default color).
        """
        self._name = name
        self._color = color
        # `_began` starts False so a `Range` that is constructed but never
        # entered (or entered while disabled) never closes a span in __exit__.
        self._began = False

    @always_inline
    def __enter__(mut self):
        """Begin a libkineto activity span if the profiler is enabled.

        Pairs with `__exit__`. The enabled state is captured into `_began` here
        so a `disable()` racing with the scope cannot leave a span begun
        without a matching end.
        """
        if is_enabled():
            # RangeBridge.cpp requires `namePtr` non-null even when the length
            # is 0 (constructing a std::string_view from null is UB under
            # C++17). `StaticString.as_bytes().unsafe_ptr()` is backed by a
            # real symbol and is non-null even for `Range("")`, which upholds
            # that.
            external_call["KGEN_CompilerRT_RangeBegin", NoneType](
                self._name.as_bytes().unsafe_ptr(),
                self._name.byte_length(),
                self._color,
            )
            # Only mark begun after RangeBegin returns successfully — __exit__
            # uses `_began` to decide whether to call RangeEnd, so an unbegun
            # span must never set this to True.
            self._began = True

    @always_inline
    def __exit__(self):
        """Close the libkineto activity span if `__enter__` opened one.

        Pairs with the begin call in `__enter__`. Calling `disable()` between
        entry and exit does not unbalance the libkineto bookkeeping because
        `_began` was captured at entry time.
        """
        if self._began:
            external_call["KGEN_CompilerRT_RangeEnd", NoneType]()
