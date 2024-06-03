# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
"""Implements classes and methods for coroutines.

These are Mojo built-ins, so you don't need to import them.
"""

from sys import sizeof

from memory import UnsafePointer

# ===----------------------------------------------------------------------=== #
# _suspend_async
# ===----------------------------------------------------------------------=== #


alias AnyCoroutine = __mlir_type.`!co.routine`


@always_inline
fn _suspend_async[body: fn (AnyCoroutine) capturing -> None]():
    __mlir_region await_body(hdl: AnyCoroutine):
        body(hdl)
        __mlir_op.`co.suspend.end`()

    __mlir_op.`co.suspend`[_region = "await_body".value]()


# ===----------------------------------------------------------------------=== #
# _CoroutineContext
# ===----------------------------------------------------------------------=== #


@register_passable("trivial")
struct _CoroutineContext:
    """The default context for a Coroutine, capturing the resume function
    callback and parent Coroutine. The resume function will typically just
    resume the parent. May be overwritten by other context types with different
    interpretations of the payload, but which nevertheless be the same size
    and contain the resume function and a payload pointer."""

    # Passed the coroutine being completed and its context's payload.
    alias _resume_fn_type = fn (AnyCoroutine, AnyCoroutine) -> None

    var _resume_fn: Self._resume_fn_type
    var _parent_hdl: AnyCoroutine


fn _coro_resume_callback(
    handle: AnyCoroutine,
    parent: AnyCoroutine,
):
    """Resume the parent Coroutine."""
    _coro_resume_fn(parent)


@always_inline
fn _coro_resume_fn(handle: AnyCoroutine):
    """This function is a generic coroutine resume function."""
    __mlir_op.`co.resume`(handle)


fn _coro_resume_noop_callback(handle: AnyCoroutine, null: AnyCoroutine):
    """Return immediately since nothing to resume."""
    return


# ===----------------------------------------------------------------------=== #
# Coroutine
# ===----------------------------------------------------------------------=== #


@register_passable
struct Coroutine[type: AnyType, lifetimes: LifetimeSet]:
    """Represents a coroutine.

    Coroutines can pause execution saving the state of the program (including
    values of local variables and the location of the next instruction to be
    executed). When the coroutine is resumed, execution continues from where it
    left off, with the saved state restored.

    Parameters:
        type: Type of value returned upon completion of the coroutine.
        lifetimes: The lifetime of the coroutine's captures.
    """

    var _handle: AnyCoroutine

    @always_inline
    fn _get_ctx[ctx_type: AnyType](self) -> UnsafePointer[ctx_type]:
        """Returns the pointer to the coroutine context.

        Parameters:
            ctx_type: The type of the coroutine context.

        Returns:
            The coroutine context.
        """
        constrained[
            sizeof[_CoroutineContext]() == sizeof[ctx_type](),
            "context size must be 16 bytes",
        ]()
        return __mlir_op.`co.get_callback_ptr`[
            _type = __mlir_type[`!kgen.pointer<`, ctx_type, `>`]
        ](self._handle)

    @always_inline
    fn _set_result_slot(self, slot: UnsafePointer[type]):
        __mlir_op.`co.set_byref_error_result`(
            self._handle,
            __mlir_attr.`#interp.pointer<0> : !kgen.pointer<none>`,
            slot.address,
        )

    @always_inline
    fn __init__(handle: AnyCoroutine) -> Self:
        """Construct a coroutine object from a handle.

        Args:
            handle: The init handle.

        Returns:
            The constructed coroutine object.
        """
        return Self {_handle: handle}

    @always_inline
    fn __del__(owned self):
        """Destroy the coroutine object."""
        __mlir_op.`co.destroy`(self._handle)

    @always_inline
    @__named_result(out)
    fn __await__(self) -> type:
        """Suspends the current coroutine until the coroutine is complete.

        Returns:
            The coroutine promise.
        """

        # Black magic! Internal implementation detail!
        # Don't you dare copy this code! 😤
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(out))
        self._set_result_slot(UnsafePointer.address_of(out))

        @always_inline
        @parameter
        fn await_body(parent_hdl: AnyCoroutine):
            LegacyPointer(self._get_ctx[_CoroutineContext]().address).store(
                _CoroutineContext {
                    _resume_fn: _coro_resume_callback, _parent_hdl: parent_hdl
                }
            )
            __mlir_op.`co.resume`(self._handle)

        _suspend_async[await_body]()

    # Never call this method.
    @__named_result(out)
    fn _deprecated_direct_resume(self) -> type:
        LegacyPointer(self._get_ctx[_CoroutineContext]().address).store(
            _CoroutineContext {
                _resume_fn: _coro_resume_noop_callback,
                _parent_hdl: self._handle,
            }
        )
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(out))
        __mlir_op.`co.set_byref_error_result`(
            self._handle,
            __mlir_attr.`#interp.pointer<0> : !kgen.pointer<none>`,
            UnsafePointer.address_of(out).address,
        )
        __mlir_op.`co.resume`(self._handle)


# ===----------------------------------------------------------------------=== #
# RaisingCoroutine
# ===----------------------------------------------------------------------=== #


@register_passable
struct RaisingCoroutine[type: AnyType, lifetimes: LifetimeSet]:
    """Represents a coroutine that can raise.

    Coroutines can pause execution saving the state of the program (including
    values of local variables and the location of the next instruction to be
    executed). When the coroutine is resumed, execution continues from where it
    left off, with the saved state restored.

    Parameters:
        type: Type of value returned upon completion of the coroutine.
        lifetimes: The lifetime set of the coroutine's captures.
    """

    var _handle: AnyCoroutine

    @always_inline
    fn _get_ctx[ctx_type: AnyType](self) -> UnsafePointer[ctx_type]:
        """Returns the pointer to the coroutine context.

        Parameters:
            ctx_type: The type of the coroutine context.

        Returns:
            The coroutine context.
        """
        constrained[
            sizeof[_CoroutineContext]() == sizeof[ctx_type](),
            "context size must be 16 bytes",
        ]()
        return __mlir_op.`co.get_callback_ptr`[
            _type = __mlir_type[`!kgen.pointer<`, ctx_type, `>`]
        ](self._handle)

    @always_inline
    fn _set_result_slot(
        self, slot: UnsafePointer[type], err: UnsafePointer[Error]
    ):
        __mlir_op.`co.set_byref_error_result`(
            self._handle, err.address, slot.address
        )

    @always_inline
    fn __init__(handle: AnyCoroutine) -> Self:
        """Construct a coroutine object from a handle.

        Args:
            handle: The init handle.

        Returns:
            An owning coroutine.
        """
        return Self {_handle: handle}

    @always_inline
    fn __del__(owned self):
        """Destroy the coroutine object."""
        __mlir_op.`co.destroy`(self._handle)

    @always_inline
    @__named_result(out)
    fn __await__(self) raises -> type:
        """Suspends the current coroutine until the coroutine is complete.

        Returns:
            The coroutine promise.
        """

        # Black magic! Internal implementation detail!
        # Don't you dare copy this code! 😤
        self._set_result_slot(
            __mlir_op.`lit.ref.to_pointer`(__get_mvalue_as_litref(out)),
            __mlir_op.`lit.ref.to_pointer`(
                __get_mvalue_as_litref(__get_nearest_error_slot())
            ),
        )

        @always_inline
        @parameter
        fn await_body(parent_hdl: AnyCoroutine):
            LegacyPointer(self._get_ctx[_CoroutineContext]().address).store(
                _CoroutineContext {
                    _resume_fn: _coro_resume_callback, _parent_hdl: parent_hdl
                }
            )
            __mlir_op.`co.resume`(self._handle)

        _suspend_async[await_body]()
        if __mlir_op.`co.get_results`[_type = __mlir_type.i1](self._handle):
            __mlir_op.`lit.ownership.mark_initialized`(
                __get_mvalue_as_litref(__get_nearest_error_slot())
            )
            __mlir_op.`lit.raise`()
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(out))
