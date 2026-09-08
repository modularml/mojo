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

from std.collections import OptionalReg
from std.reflection.location import SourceLocation
from std.sys.info import _TargetType, _current_target

from std.utils.index import IndexList
from std.math.math import _ExpPluginHookFnType, _TanhPluginHookFnType
from std.memory.stack_allocation import _StackAllocationPluginHookFnType
from std.memory.unsafe_pointer import _UnsafeDanglingPluginHookFnType
from std.io.io import _PrintEmitPluginHookFnType
from std.collections.string.string_span import (
    _get_kgen_string,
)
from std.utils import StaticTuple

from .selector import Plugin

comptime _ReduceGeneratorPluginHookFnType = (
    def[
        num_reductions: Int,
        init_type: DType,
        input_0_fn: def[dtype: DType, width: Int, rank: Int](
            IndexList[rank]
        ) capturing[_] -> SIMD[dtype, width],
        output_0_fn: def[dtype: DType, width: SIMDLength, rank: Int](
            IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
        ) capturing[_] -> None,
        reduce_function: def[ty: DType, width: SIMDLength, reduction_idx: Int](
            SIMD[ty, width], SIMD[ty, width]
        ) capturing[_] -> SIMD[ty, width],
    ](
        shape: IndexList[_, element_type=DType.int64],
        init: StaticTuple[Scalar[init_type], num_reductions],
        reduce_dim: Int,
    ) thin
)
"""Plugin-hook signature for `PluginHooks.reduce_generator_fn`; keep in sync with `_reduce_generator`."""


trait PluginHooks(Plugin):
    """Compile-time hook interface for pluggable stdlib behavior.

    Most hooks are `comptime OptionalReg[Callable]` fields; call sites invoke
    `comptime if CurrentPlugin.xxx_fn: return comptime(CurrentPlugin.xxx_fn.value())(...)`,
    so implementors that leave a hook at `None` add zero cost.

    A few hooks (`abort_fn`, `debug_assert_emit_fn`) are required
    `@staticmethod` trait methods rather than `OptionalReg` fields, because
    their dispatch sites lie on `OptionalReg.value()`'s own instantiation
    path — an `OptionalReg` field would re-enter that template via its own
    `debug_assert` and deadlock comptime instantiation.
    """

    comptime name: __mlir_type.`!kgen.string`
    """Stable plugin identifier used by the selector to select this backend."""

    comptime exp_fn: OptionalReg[_ExpPluginHookFnType] = None
    """Elementwise exponential override.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The input SIMD vector.

    Returns:
        Elementwise `exp(x)` computed on the vendor backend.
    """

    comptime tanh_fn[dtype: DType, width: Int]: OptionalReg[
        _TanhPluginHookFnType
    ] = None
    """Elementwise hyperbolic tangent override.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The input SIMD vector.

    Returns:
        Elementwise `tanh(x)` computed on the vendor backend.
    """

    comptime stack_allocation_fn[address_space: AddressSpace]: OptionalReg[
        _StackAllocationPluginHookFnType[address_space]
    ] = None

    comptime address_space_fn[name: StaticString]: OptionalReg[
        AddressSpace
    ] = None
    """Target-specific named address-space lookup.

    Resolves an address-space *name* that has no built-in constant on
    `AddressSpace` (the GPU spaces `GENERIC`/`GLOBAL`/`SHARED`/...) to its
    target-specific value — for example an accelerator-specific scratchpad
    space. `AddressSpace.<NAME>` consults this hook for any such name; leaving it
    `None` (the default) makes the name a compile-time error. This keeps the
    set of valid address-space names open and target-extensible rather than a
    fixed portable enum.

    Parameters:
        name: The address-space name being looked up.

    Returns:
        The backend's `AddressSpace` for `name`, or `None` if the backend does
        not define it.
    """

    comptime unsafe_dangling_fn: OptionalReg[
        _UnsafeDanglingPluginHookFnType
    ] = None
    """`Pointer.unsafe_dangling()` address override.

    Parameters:
        alignment: The natural alignment of the pointee type, which the
            stdlib default uses as the dangling address.

    Returns:
        The raw integer address used to construct the dangling pointer.
    """

    comptime print_emit_fn: OptionalReg[_PrintEmitPluginHookFnType] = None
    """Plugin hook for emitting a `print()` UTF-8 byte buffer to a file
    descriptor."""

    comptime reduce_generator_fn: OptionalReg[
        _ReduceGeneratorPluginHookFnType
    ] = None

    @staticmethod
    def abort_fn():
        """`abort()` override, called before the default trap. If the hook
        doesn't return (e.g. via `longjmp`), the trap is dead code.

        The default is a no-op (the stdlib trap runs).
        """
        pass

    @staticmethod
    def debug_assert_emit_fn[
        O: Origin
    ](message: Pointer[Byte, O], length: Int, loc: SourceLocation):
        """Assertion-message emitter for targets without a usable `_printf`.

        Parameters:
            O: The origin of the message pointer.

        Args:
            message: Pointer to the nul-terminated message bytes.
            length: Length in bytes (excluding the trailing nul).
            loc: Source location of the failing assertion.

        Only invoked when `_handles_debug_assert` is `True`; the default is
        never called and is a no-op.
        """
        pass

    comptime _handles_debug_assert: Bool = False
    """If `True`, `_debug_assert_msg` dispatches to `debug_assert_emit_fn`
    and comptime-elides its `_printf` fallback. Required because the
    fallback's transitive `OptionalReg.value()` → `debug_assert` recurses
    back through `_debug_assert_msg` and deadlocks instantiation when
    assertions are enabled."""


# ===-----------------------------------------------------------------------===#
# DefaultPlugin
# ===-----------------------------------------------------------------------===#


struct DefaultPlugin(PluginHooks):
    """Default `PluginHooks` implementation used when no plugin is active.

    Every hook is left at its `PluginHooks` default, so the built-in stdlib
    code paths are preserved.
    """

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["default"]()
