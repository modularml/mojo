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

from memory import Pointer

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

    alias _opaque_handle = Pointer[__mlir_type.i8]
    # Passed the coroutine being completed and its context's payload.
    alias _resume_fn_type = fn (
        Self._opaque_handle, Self._opaque_handle
    ) -> None

    var _resume_fn: Self._resume_fn_type
    var _parent_hdl: Self._opaque_handle


fn _coro_resume_callback(
    handle: _CoroutineContext._opaque_handle,
    parent: _CoroutineContext._opaque_handle,
):
    """Resume the parent Coroutine."""
    _coro_resume_fn(parent)


@always_inline
fn _coro_resume_fn(handle: _CoroutineContext._opaque_handle):
    """This function is a generic coroutine resume function."""
    __mlir_op.`co.resume`(handle.address)


fn _coro_resume_noop_callback(
    handle: _CoroutineContext._opaque_handle,
    null: _CoroutineContext._opaque_handle,
):
    """Return immediately since nothing to resume."""
    return


# ===----------------------------------------------------------------------=== #
# Coroutine
# ===----------------------------------------------------------------------=== #


@register_passable
struct Coroutine[type: AnyRegType]:
    """Represents a coroutine.

    Coroutines can pause execution saving the state of the program (including
    values of local variables and the location of the next instruction to be
    executed). When the coroutine is resumed, execution continues from where it
    left off, with the saved state restored.

    Parameters:
        type: Type of value returned upon completion of the coroutine.
    """

    alias _handle_type = __mlir_type[`!co.routine<`, type, `>`]
    alias _promise_type = __mlir_type[`!kgen.struct<(`, type, `)>`]
    var _handle: Self._handle_type

    @always_inline
    fn _get_promise(self) -> Pointer[type]:
        """Return the pointer to the beginning of the memory where the async
        function results are stored.

        Returns:
            The coroutine promise.
        """
        var promise: Pointer[Self._promise_type] = __mlir_op.`co.promise`(
            self._handle
        )
        return promise.bitcast[type]()

    @always_inline
    fn get(self) -> type:
        """Get the value of the fulfilled coroutine promise.

        Returns:
            The value of the fulfilled promise.
        """
        return self._get_promise().load()

    @always_inline
    fn _get_ctx[ctx_type: AnyRegType](self) -> Pointer[ctx_type]:
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
        return self._get_promise().bitcast[ctx_type]() - 1

    @always_inline
    fn __init__(handle: Self._handle_type) -> Coroutine[type]:
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
    fn __call__(self) -> type:
        """Execute the coroutine synchronously.

        Returns:
            The coroutine promise.
        """

        self._get_ctx[_CoroutineContext]().store(
            _CoroutineContext {
                _resume_fn: _coro_resume_noop_callback,
                _parent_hdl: _CoroutineContext._opaque_handle.get_null(),
            }
        )
        __mlir_op.`co.resume`(self._handle)
        return self.get()

    @always_inline
    fn __await__(self) -> type:
        """Suspends the current coroutine until the coroutine is complete.

        Returns:
            The coroutine promise.
        """
        var parent_hdl = __mlir_op.`co.opaque_handle`()
        self._get_ctx[_CoroutineContext]().store(
            _CoroutineContext {
                _resume_fn: _coro_resume_callback, _parent_hdl: parent_hdl
            }
        )

        __mlir_region await_body():
            __mlir_op.`co.resume`(self._handle)
            __mlir_op.`co.await.end`()

        __mlir_op.`co.await`[_region = "await_body".value]()
        return self.get()


# ===----------------------------------------------------------------------=== #
# RaisingCoroutine
# ===----------------------------------------------------------------------=== #


@register_passable
struct RaisingCoroutine[type: AnyRegType]:
    """Represents a coroutine that can raise.

    Coroutines can pause execution saving the state of the program (including
    values of local variables and the location of the next instruction to be
    executed). When the coroutine is resumed, execution continues from where it
    left off, with the saved state restored.

    Parameters:
        type: Type of value returned upon completion of the coroutine.
    """

    alias _var_type = __mlir_type[`!kgen.variant<`, Error, `, `, type, `>`]
    alias _handle_type = __mlir_type[`!co.routine<`, Self._var_type, `>`]
    alias _promise_type = __mlir_type[`!kgen.struct<(`, Self._var_type, `)>`]
    var _handle: Self._handle_type

    @always_inline
    fn _get_promise(self) -> Pointer[Self._var_type]:
        """Return the pointer to the beginning of the memory where the async
        function results are stored.

        Returns:
            The coroutine promise.
        """
        var promise: Pointer[Self._promise_type] = __mlir_op.`co.promise`(
            self._handle
        )
        return promise.bitcast[Self._var_type]()

    @always_inline
    fn get(self) raises -> type:
        """Get the value of the fulfilled coroutine promise.

        Returns:
            The value of the fulfilled promise.
        """
        var variant = self._get_promise().load()
        if __mlir_op.`kgen.variant.is`[index = Int(0).value](variant):
            raise __mlir_op.`kgen.variant.take`[index = Int(0).value](variant)
        return __mlir_op.`kgen.variant.take`[index = Int(1).value](variant)

    @always_inline
    fn _get_ctx[ctx_type: AnyRegType](self) -> Pointer[ctx_type]:
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
        return self._get_promise().bitcast[ctx_type]() - 1

    @always_inline
    fn __init__(inout self, handle: Self._handle_type):
        """Construct a coroutine object from a handle.

        Args:
            handle: The init handle.
        """
        self = Self {_handle: handle}

    @always_inline
    fn __del__(owned self):
        """Destroy the coroutine object."""
        __mlir_op.`co.destroy`(self._handle)

    @always_inline
    fn __call__(self) raises -> type:
        """Execute the coroutine synchronously.

        Returns:
            The coroutine promise.
        """

        fn _coro_noop_fn(handle: _CoroutineContext._opaque_handle):
            return

        self._get_ctx[_CoroutineContext]().store(
            _CoroutineContext {
                _resume_fn: _coro_resume_noop_callback,
                _parent_hdl: _CoroutineContext._opaque_handle.get_null(),
            }
        )
        __mlir_op.`co.resume`(self._handle)
        return self.get()

    @always_inline
    fn __await__(self) raises -> type:
        """Suspends the current coroutine until the coroutine is complete.

        Returns:
            The coroutine promise.
        """
        var parent_hdl = __mlir_op.`co.opaque_handle`()
        self._get_ctx[_CoroutineContext]().store(
            _CoroutineContext {
                _resume_fn: _coro_resume_callback, _parent_hdl: parent_hdl
            }
        )

        __mlir_region await_body():
            __mlir_op.`co.resume`(self._handle)
            __mlir_op.`co.await.end`()

        __mlir_op.`co.await`[_region = "await_body".value]()
        return self.get()
