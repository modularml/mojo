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
"""Implements classes and methods for coroutines.

Async support in Mojo is unfinished and these types are not ready for general
use. They are the return types the compiler synthesizes for `async def`, so they
appear in inferred types and diagnostics, but nothing here carries a stability
guarantee and the design is expected to change substantially. This module is
private for that reason: `Coroutine` and `RaisingCoroutine` are not exported
from the prelude, and code outside the standard library should not depend on
them.
"""

from std.sys import size_of

# ===----------------------------------------------------------------------=== #
# _suspend_async
# ===----------------------------------------------------------------------=== #


comptime AnyCoroutine = __mlir_type.`!co.routine`
"""The MLIR type representing a coroutine handle."""


@always_inline
def _suspend_async(body: Some[def(AnyCoroutine) -> None]):
    __mlir_region await_body(hdl: __mlir_type.`!co.routine`):
        body(hdl)
        __mlir_op.`co.suspend.end`()

    __mlir_op.`co.suspend`[_region="await_body".value]()


# ===----------------------------------------------------------------------=== #
# _CoroutineContext
# ===----------------------------------------------------------------------=== #


struct _CoroutineContext(TrivialRegisterPassable):
    """The default context for a Coroutine, capturing the resume function
    callback and parent Coroutine. The resume function will typically just
    resume the parent. May be overwritten by other context types with different
    interpretations of the payload, but which nevertheless be the same size
    and contain the resume function and a payload pointer."""

    # Passed the coroutine being completed and its context's payload.
    comptime _resume_fn_type = def(AnyCoroutine) thin -> None

    var _resume_fn: Self._resume_fn_type
    var _parent_hdl: AnyCoroutine


@always_inline
def _coro_get_resume_fn(handle: AnyCoroutine) -> def(AnyCoroutine) thin -> None:
    """This function is a generic coroutine resume function."""
    return __mlir_op.`co.resume`[_type=def(AnyCoroutine) thin -> None](handle)


@always_inline
def _coro_resume_fn(handle: AnyCoroutine):
    """This function is a generic coroutine resume function."""
    _coro_get_resume_fn(handle)(handle)


@always_inline
def _coro_destroy_fn(handle: AnyCoroutine):
    __mlir_op.`co.destroy`(handle)


def _coro_resume_noop_callback(null: AnyCoroutine):
    """Return immediately since nothing to resume."""
    return


# ===----------------------------------------------------------------------=== #
# Coroutine
# ===----------------------------------------------------------------------=== #


struct Coroutine[type: Deinitable, origins: OriginSet](
    Deinitable where False,
    RegisterPassable,
):
    """Represents a coroutine.

    Coroutines can pause execution saving the state of the program (including
    values of local variables and the location of the next instruction to be
    executed). When the coroutine is resumed, execution continues from where it
    left off, with the saved state restored.

    Parameters:
        type: Type of value returned upon completion of the coroutine.
        origins: The origin of the coroutine's captures.
    """

    var _handle: AnyCoroutine

    @always_inline
    def _get_ctx[
        ctx_type: AnyType
    ](self) -> Pointer[ctx_type, MutUntrackedOrigin]:
        """Returns the pointer to the coroutine context.

        Parameters:
            ctx_type: The type of the coroutine context.

        Returns:
            The coroutine context.
        """
        comptime assert (
            size_of[_CoroutineContext]() == size_of[ctx_type]()
        ), "context size must be 16 bytes"
        return {
            _mlir_value = __mlir_op.`co.get_callback_ptr`[
                _type=__mlir_type[`!kgen.pointer<`, ctx_type, `>`]
            ](self._handle)
        }

    @always_inline
    def _set_result_slot(self, slot: MutPointer[Self.type, ...]):
        __mlir_op.`co.set_byref_error_result`(
            self._handle, slot._get_kgen_pointer()
        )

    @always_inline
    def _set_noop_callback(self):
        """Set the resume function of the coroutine context to a no-op so it
        doesn't try to resume anything else after executing. This makes
        coroutines suitable for executing from external code (e.g. AsyncRT)
        using _coro_resume_fn.
        """
        self._get_ctx[
            _CoroutineContext
        ]()[]._resume_fn = _coro_resume_noop_callback

    @always_inline
    @implicit
    def __init__(out self, handle: AnyCoroutine):
        """Construct a coroutine object from a handle.

        Args:
            handle: The init handle.
        """
        self._handle = handle

    @always_inline
    def _unsafe_force_deinit(deinit self):
        """Destroy the coroutine object without running its body to completion.

        This is unsafe: if the coroutine is mid-suspension, any linear values
        still live inside its frame are dropped without cleanup. Only call this
        when the coroutine has run to completion (or was never resumed and holds
        no linear state).
        """
        __mlir_op.`co.destroy`(self._handle)

    @always_inline
    def _take_handle(deinit self) -> AnyCoroutine:
        """Take ownership of the raw handle."""
        return self._handle

    @always_inline
    def __await__(deinit self, out result: Self.type):
        """Suspends the current coroutine until the coroutine is complete.

        Returns:
            The coroutine promise.
        """

        # Black magic! Internal implementation detail!
        # Don't you dare copy this code! 😤
        var handle = self._handle
        __mlir_op.`co.await`[_type=NoneType](
            handle,
            __mlir_op.`lit.ref.to_pointer`(__get_mvalue_as_litref(result)),
        )
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(result)
        )


# ===----------------------------------------------------------------------=== #
# RaisingCoroutine
# ===----------------------------------------------------------------------=== #


struct RaisingCoroutine[type: AnyType, origins: OriginSet](
    Deinitable where False,
    RegisterPassable,
):
    """Represents a coroutine that can raise.

    Coroutines can pause execution saving the state of the program (including
    values of local variables and the location of the next instruction to be
    executed). When the coroutine is resumed, execution continues from where it
    left off, with the saved state restored.

    Parameters:
        type: Type of value returned upon completion of the coroutine.
        origins: The origin set of the coroutine's captures.
    """

    var _handle: AnyCoroutine

    @always_inline
    def _get_ctx[
        ctx_type: AnyType
    ](self) -> Pointer[ctx_type, MutUntrackedOrigin]:
        """Returns the pointer to the coroutine context.

        Parameters:
            ctx_type: The type of the coroutine context.

        Returns:
            The coroutine context.
        """
        comptime assert (
            size_of[_CoroutineContext]() == size_of[ctx_type]()
        ), "context size must be 16 bytes"
        return {
            _mlir_value = __mlir_op.`co.get_callback_ptr`[
                _type=__mlir_type[`!kgen.pointer<`, ctx_type, `>`]
            ](self._handle)
        }

    @always_inline
    def _set_result_slot(
        self,
        slot: MutPointer[Self.type, ...],
        err: ImmPointer[Error, ...],
    ):
        __mlir_op.`co.set_byref_error_result`(
            self._handle, slot._get_kgen_pointer(), err._get_kgen_pointer()
        )

    @always_inline
    @implicit
    def __init__(out self, handle: AnyCoroutine):
        """Construct a coroutine object from a handle.

        Args:
            handle: The init handle.
        """
        self._handle = handle

    @always_inline
    def _take_handle(deinit self) -> AnyCoroutine:
        """Take ownership of the raw handle."""
        return self._handle

    @always_inline
    def _unsafe_force_deinit(deinit self):
        """Destroy the coroutine object without running its body to completion.

        This is unsafe: if the coroutine is mid-suspension, any linear values
        still live inside its frame are dropped without cleanup. Only call this
        when the coroutine has run to completion (or was never resumed and holds
        no linear state).
        """
        __mlir_op.`co.destroy`(self._handle)

    @always_inline
    def __await__(var self, out result: Self.type) raises:
        """Suspends the current coroutine until the coroutine is complete.

        Returns:
            The result value from the completed coroutine.

        Raises:
            If the coroutine execution encounters an error.
        """

        # Black magic! Internal implementation detail!
        # Don't you dare copy this code! 😤
        var handle = self^._take_handle()
        var error: Error
        if Bool(
            # TODO: co.await should return scalar<bool>
            __mlir_op.`co.await`[_type=__mlir_type.i1](
                handle,
                __mlir_op.`lit.ref.to_pointer`(__get_mvalue_as_litref(result)),
                __mlir_op.`lit.ref.to_pointer`(__get_mvalue_as_litref(error)),
            )
        ):
            __mlir_op.`lit.ownership.mark_initialized`(
                __get_mvalue_as_litref(error)
            )
            raise error^
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(result)
        )
