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
"""This module implements the low level concurrency library.

Mojo's async support is unfinished. The task primitives here — `create_task`,
`Task`, `RaisingTask`, `TaskGroup` — are the substrate that `parallelize()` is
built on, not a general-purpose async runtime, and their design is expected to
change substantially. This module is private for that reason; code outside the
standard library and the kernel library should not depend on it.
"""

from std.os import abort
from std.atomic import Atomic
from std.ffi import external_call
from std.memory.alloc import alloc, dealloc, ThinAllocation, Layout

from std.builtin._coroutine import (
    AnyCoroutine,
    Coroutine,
    RaisingCoroutine,
    _coro_resume_fn,
    _suspend_async,
)

# ===-----------------------------------------------------------------------===#
# _AsyncContext
# ===-----------------------------------------------------------------------===#


struct _Chain(Boolable, Defaultable, ImplicitlyCopyable, RegisterPassable):
    """A proxy for the C++ runtime's AsyncValueRef<_Chain> type."""

    # Actually an AsyncValueRef<_Chain>, which is just an AsyncValue*
    var storage: OptionalPointer[Int, MutUntrackedOrigin]

    def __init__(out self):
        self.storage = {}

    def __bool__(self) -> Bool:
        return Bool(self.storage)


struct _AsyncContext(ImplicitlyCopyable, RegisterPassable):
    """This struct models the coroutine context contained in every coroutine
    instance. The struct consists of a unary callback function that accepts a
    pointer argument. It is invoked with the second struct field, which is an
    opaque pointer. This struct is essentially a completion callback closure
    that is invoked by a coroutine when it completes and its results are made
    available.

    In async execution, a task's completion callback is to set its async token
    to available.
    """

    comptime callback_fn_type = def(_Chain) thin -> None

    var callback: Self.callback_fn_type
    var chain: _Chain

    @staticmethod
    def get_chain(
        ctx: MutPointer[_AsyncContext, _]
    ) -> Pointer[_Chain, origin_of(ctx[].chain)]:
        return Pointer(to=ctx[].chain)

    @staticmethod
    def complete(ch: _Chain):
        var tmp = ch
        _async_complete(Pointer(to=tmp))


# ===-----------------------------------------------------------------------===#
# AsyncRT C Shims
# ===-----------------------------------------------------------------------===#


def _init_asyncrt_chain(chain: MutPointer[_Chain, _]):
    external_call["KGEN_CompilerRT_AsyncRT_InitializeChain", NoneType](chain)


def _del_asyncrt_chain(chain: MutPointer[_Chain, _]):
    external_call["KGEN_CompilerRT_AsyncRT_DestroyChain", NoneType](chain)


def _async_and_then(hdl: AnyCoroutine, chain: MutPointer[_Chain, _]):
    external_call["KGEN_CompilerRT_AsyncRT_AndThen", NoneType](
        _coro_resume_fn, chain, hdl
    )


def _async_execute[type: AnyType](handle: AnyCoroutine, desired_worker_id: Int):
    external_call["KGEN_CompilerRT_AsyncRT_Execute", NoneType](
        _coro_resume_fn, handle, desired_worker_id
    )


def _async_wait(chain: MutPointer[_Chain, _]):
    external_call["KGEN_CompilerRT_AsyncRT_Wait", NoneType](chain)


def _async_complete(chain: MutPointer[_Chain, _]):
    external_call["KGEN_CompilerRT_AsyncRT_Complete", NoneType](chain)


def _async_wait_timeout(chain: MutPointer[_Chain, _], timeout: Int) -> Bool:
    return external_call["KGEN_CompilerRT_AsyncRT_Wait_Timeout", Bool](
        chain, timeout
    )


def create_task(
    var handle: Coroutine[...], out task: Task[handle.type, handle.origins]
):
    """Run the coroutine as a task on the AsyncRT Runtime.

    This function creates a task from a coroutine and schedules it for execution
    on the async runtime. The task will execute asynchronously without blocking
    the current execution context.

    Args:
        handle: The coroutine to execute as a task. Ownership is transferred.

    Returns:
        The `task` output parameter is initialized with the created task.
    """
    var ctx = handle._get_ctx[_AsyncContext]()
    _init_asyncrt_chain(_AsyncContext.get_chain(ctx))
    ctx[].callback = _AsyncContext.complete
    task = Task(handle^)
    _async_execute[handle.type](task._handle._handle, desired_worker_id=-1)


def _create_task(
    var handle: Coroutine[...],
    *,
    desired_worker_id: Int,
    out task: Task[handle.type, handle.origins],
):
    """Run the coroutine as a task on the AsyncRT Runtime with a worker affinity hint.

    Package-private variant of `create_task` that accepts a `desired_worker_id`
    so that callers can pin the task to a CPU worker with affinity for a
    particular device. Pass the return value of `task_id_for_device` here;
    pass -1 to get the same behavior as the public `create_task`.

    This is intentionally not part of the public API. Use `create_task` for
    unaffinitized launches. Use this function only when you have a concrete
    reason to pin to a specific worker (e.g. NUMA-aware GPU collective dispatch).

    Args:
        handle: The coroutine to execute as a task. Ownership is transferred.
        desired_worker_id: A preferred AsyncRT worker thread index. The runtime
            may ignore this hint if the requested worker is unavailable.

    Returns:
        The `task` output parameter is initialized with the created task.
    """
    var ctx = handle._get_ctx[_AsyncContext]()
    _init_asyncrt_chain(_AsyncContext.get_chain(ctx))
    ctx[].callback = _AsyncContext.complete
    task = Task(handle^)
    _async_execute[handle.type](task._handle._handle, desired_worker_id)


@always_inline
def _run(var handle: Coroutine[...], out result: handle.type):
    """Executes a coroutine and waits for its completion.
    This function runs the given coroutine on the async runtime and blocks until
    it completes. The result of the coroutine is stored in the output parameter.
    Args:
        handle: The coroutine to execute. Ownership is transferred.
    Returns:
        The `result` output parameter is initialized with the coroutine's result.
    """
    var ctx = handle._get_ctx[_AsyncContext]()
    _init_asyncrt_chain(_AsyncContext.get_chain(ctx))
    ctx[].callback = _AsyncContext.complete
    __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(result))
    handle._set_result_slot(Pointer(to=result))
    _async_execute[handle.type](handle._handle, -1)
    _async_wait(_AsyncContext.get_chain(ctx))
    _del_asyncrt_chain(_AsyncContext.get_chain(ctx))
    handle^._unsafe_force_deinit()


# ===-----------------------------------------------------------------------===#
# Task
# ===-----------------------------------------------------------------------===#


struct Task[type: Deinitable, origins: OriginSet](Movable where False):
    """Represents an asynchronous task that will produce a value of the specified type.

    A Task encapsulates a coroutine that is executing asynchronously and will eventually
    produce a result. Tasks can be awaited in async functions or waited on in synchronous code.

    Parameters:
        type: The type of value that this task will produce when completed.
        origins: The set of origins for the coroutine wrapped by this task.
    """

    var _handle: Coroutine[Self.type, Self.origins]
    """The underlying coroutine that executes the task."""

    var _result: Self.type
    """Storage for the result value produced by the task."""

    def __init__(out self, var handle: Coroutine[Self.type, Self.origins]):
        """Initialize a task with a coroutine.

        Takes ownership of the provided coroutine and sets up the task to receive
        its result when completed.

        Args:
            handle: The coroutine to execute as a task. Ownership is transferred.
        """
        self._handle = handle^
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self._result)
        )
        self._handle._set_result_slot(Pointer(to=self._result))

    def get(self) -> ref[self._result] Self.type:
        """Get the task's result value. Calling this on an incomplete task is
        undefined behavior.

        Returns:
            A reference to the result value produced by the task.
        """
        return self._result

    def __deinit__(deinit self):
        """Destroy the memory associated with a task. This must be manually
        called when a task goes out of scope.
        """
        var ctx = self._handle._get_ctx[_AsyncContext]()
        _del_asyncrt_chain(_AsyncContext.get_chain(ctx))
        self._handle^._unsafe_force_deinit()

    @always_inline
    def __await__(self) -> ref[self.get()] Self.type:
        """Suspend the current async function until the task completes and its
        result becomes available. This function must be force inlined into the
        calling async function.

        This method enables the use of the 'await' keyword with Task objects in
        async functions.

        Returns:
            A reference to the result value produced by the task.
        """

        var ctx = self._handle._get_ctx[_AsyncContext]()

        @always_inline
        def await_body(cur_hdl: AnyCoroutine) {var}:
            _async_and_then(
                cur_hdl,
                _AsyncContext.get_chain(ctx),
            )

        _suspend_async(await_body)
        return self.get()

    def wait(self) -> ref[self.get()] Self.type:
        """Block the current thread until the future value becomes available.

        This method is used in synchronous code to wait for an asynchronous task
        to complete. Unlike `__await__`, this method does not suspend the current
        coroutine but instead blocks the entire thread.

        Returns:
            A reference to the result value produced by the task.
        """
        _async_wait(
            _AsyncContext.get_chain(self._handle._get_ctx[_AsyncContext]())
        )
        return self.get()


# ===-----------------------------------------------------------------------===#
# RaisingTask
# ===-----------------------------------------------------------------------===#


def create_raising_task[
    type: Movable, origins: OriginSet
](
    var handle: RaisingCoroutine[type, origins],
    out task: RaisingTask[type, origins],
):
    """Run a raising coroutine as a task on the AsyncRT Runtime.

    Creates a task from a raising coroutine and schedules it for execution.
    The task may raise an error when waited on, propagating any error from
    the coroutine.

    Parameters:
        type: The result type, which must be `Movable` to extract the result.
        origins: The origin set from the coroutine's captures.

    Args:
        handle: The raising coroutine to execute as a task. Ownership is
            transferred.

    Returns:
        The `task` output parameter is initialized with the created task.
    """
    var ctx = handle._get_ctx[_AsyncContext]()
    _init_asyncrt_chain(_AsyncContext.get_chain(ctx))
    ctx[].callback = _AsyncContext.complete
    task = RaisingTask(handle^)
    _async_execute[type](task._handle._handle, desired_worker_id=-1)


struct RaisingTask[type: Movable, origins: OriginSet](
    Deinitable where False,
):
    """Represents an async task that may raise an error upon completion.

    Wraps a `RaisingCoroutine` that executes asynchronously and either
    produces a result value or raises an error. The error is propagated
    to the caller when `wait()` is called.

    This type does not conform to `Deinitable` because only one of the
    result or error slots is valid after completion. The caller must call
    `wait()` to consume the task.

    Parameters:
        type: The type of value produced on success.
        origins: The set of origins for the coroutine wrapped by this task.
    """

    var _handle: RaisingCoroutine[Self.type, Self.origins]
    """The underlying raising coroutine."""

    var _result_alloc: ThinAllocation[Self.type]
    """Heap-allocated storage for the result value."""

    var _error_alloc: ThinAllocation[Error]
    """Heap-allocated storage for the error value."""

    def __init__(
        out self, var handle: RaisingCoroutine[Self.type, Self.origins]
    ):
        """Initialize a raising task with a raising coroutine.

        Args:
            handle: The raising coroutine to execute. Ownership is transferred.
        """
        self._handle = handle^
        self._result_alloc = alloc(Layout[Self.type].single()).into_thin()
        self._error_alloc = alloc(Layout[Error].single()).into_thin()
        self._handle._set_result_slot(
            self._result_alloc.unsafe_ptr(), self._error_alloc.unsafe_ptr()
        )

    def _has_error(self) -> Bool:
        """Check whether the coroutine raised an error.

        Reads the error flag from the coroutine's continuation struct via
        the `co.get_results` MLIR op. Only valid after the task completes.

        Returns:
            True if the coroutine raised an error, False if it succeeded.
        """
        return __mlir_op.`co.get_results`[_type=__mlir_type.i1](
            self._handle._handle
        )

    def _into_result(deinit self, out result: Self.type) raises:
        """Consumes the finished task, returning its result or raising its error.

        Tears down the coroutine and the async-runtime chain, releases the heap
        storage for both the result and error slots, and then moves the produced
        value into `result` — or, if the coroutine failed, raises the stored
        error. The task must already have completed; callers block or suspend
        (via `wait` or `__await__`) before invoking this.

        Returns:
            The `result` output parameter receives the task's value on success.

        Raises:
            The error produced by the coroutine, if it raised.
        """
        var has_error = self._has_error()

        var ctx = self._handle._get_ctx[_AsyncContext]()
        _del_asyncrt_chain(_AsyncContext.get_chain(ctx))
        self._handle^._unsafe_force_deinit()

        if has_error:
            var err = self._error_alloc.unsafe_ptr().unsafe_take_pointee()
            dealloc(self._error_alloc^.unsafe_with_layout({count = 1}))
            dealloc(self._result_alloc^.unsafe_with_layout({count = 1}))
            raise err^
        result = self._result_alloc.unsafe_ptr().unsafe_take_pointee()
        dealloc(self._error_alloc^.unsafe_with_layout({count = 1}))
        dealloc(self._result_alloc^.unsafe_with_layout({count = 1}))

    @always_inline
    def __await__(deinit self, out result: Self.type) raises:
        """Suspend the current async function until the task completes.

        Consumes the task. On success, moves the result out. On failure,
        raises the error from the coroutine.

        This enables `await task^` syntax in async functions.

        Returns:
            The `result` output parameter receives the task's result value.

        Raises:
            If the underlying coroutine raised an error.
        """

        var ctx = self._handle._get_ctx[_AsyncContext]()

        @always_inline
        def await_body(cur_hdl: AnyCoroutine) {var}:
            _async_and_then(cur_hdl, _AsyncContext.get_chain(ctx))

        _suspend_async(await_body)
        result = self^._into_result()

    def wait(deinit self, out result: Self.type) raises:
        """Block until the task completes and return the result or raise.

        Consumes the task. On success, moves the result out. On failure,
        raises the error from the coroutine.

        Returns:
            The `result` output parameter receives the task's result value.

        Raises:
            If the underlying coroutine raised an error.
        """
        _async_wait(
            _AsyncContext.get_chain(self._handle._get_ctx[_AsyncContext]())
        )
        result = self^._into_result()

    # TODO: Add force_destroy() when we have a trait that combines
    # Movable and Deinitable. Currently, the caller must
    # call wait() to consume the task.


# ===-----------------------------------------------------------------------===#
# TaskGroup
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct TaskGroupContext(TrivialRegisterPassable):
    """Context structure for task group operations.

    This structure holds a callback function and a pointer to a TaskGroup,
    allowing asynchronous operations to interact with their parent TaskGroup
    when they complete.
    """

    comptime tg_callback_fn_type = def(mut TaskGroup) thin -> None
    """Type definition for callback functions that operate on TaskGroups."""

    var callback: Self.tg_callback_fn_type
    """Callback function to be invoked on the TaskGroup when an operation completes."""

    var task_group: Pointer[TaskGroup, MutUntrackedOrigin]
    """Pointer to the TaskGroup that owns or is associated with this context."""


struct _TaskGroupBox(Copyable, RegisterPassable):
    """This struct is a type-erased owning box for an opaque coroutine."""

    var handle: AnyCoroutine

    def __init__[type: Deinitable](out self, var coro: Coroutine[type, ...]):
        self.handle = coro^._take_handle()

    def __deinit__(deinit self):
        __mlir_op.`co.destroy`(self.handle)

    # FIXME(MSTDL-573): `List` requires copyability. Just crash here because it
    # should never get called.
    def __init__(out self, *, copy: Self):
        abort("_TaskGroupBox copy ctor should never get called")


struct TaskGroup(Defaultable):
    """A group of tasks that can be executed concurrently.

    TaskGroup manages a collection of coroutines that can be executed in parallel.
    It provides mechanisms to create, track, and wait for the completion of tasks.
    """

    var counter: Atomic[Int]
    """Atomic counter tracking the number of active tasks in the group."""

    var chain: _Chain
    """Chain used for asynchronous completion notification."""

    var tasks: List[_TaskGroupBox]
    """Collection of tasks managed by this TaskGroup."""

    def __init__(out self):
        """Initialize a new TaskGroup with an empty task list and initialized chain.
        """
        var chain = _Chain()
        _init_asyncrt_chain(Pointer(to=chain))
        self.counter = Atomic[Int](1)
        self.chain = chain
        self.tasks = List[_TaskGroupBox](capacity=16)

    def __deinit__(deinit self):
        """Clean up resources associated with the TaskGroup."""
        _del_asyncrt_chain(Pointer(to=self.chain))

    @always_inline
    def _counter_decr(mut self) -> Int:
        var prev: Int = self.counter.fetch_sub(1)
        return prev - 1

    @staticmethod
    def _task_complete_callback(mut tg: TaskGroup):
        tg._task_complete()

    def _task_complete(mut self):
        if self._counter_decr() == 0:
            _async_complete(Pointer(to=self.chain))

    def create_task(
        mut self,
        # FIXME(MSTDL-722): Avoid accessing ._mlir_type here, use `NoneType`.
        var task: Coroutine[NoneType._mlir_type, ...],
    ):
        """Add a new task to the TaskGroup for execution.

        Args:
            task: The coroutine to be executed as a task.
        """
        self._create_task(task^, desired_worker_id=-1)

    # Deprecated, use create_task() instead
    # Only sync_parallelize() uses this to pass desired_worker_id
    def _create_task(
        mut self,
        # FIXME(MSTDL-722): Avoid accessing ._mlir_type here, use `NoneType`.
        var task: Coroutine[NoneType._mlir_type, ...],
        desired_worker_id: Int = -1,
    ):
        # TODO(MOCO-771): Enforce that `task.origins` is a subset of
        # `Self.origins`.
        self.counter += 1
        task._get_ctx[TaskGroupContext]()[] = TaskGroupContext(
            Self._task_complete_callback,
            Pointer(to=self).unsafe_origin_cast[MutUntrackedOrigin](),
        )
        _async_execute[NoneType](task._handle, desired_worker_id)
        self.tasks.append(_TaskGroupBox(task^))

    @staticmethod
    def await_body_impl(hdl: AnyCoroutine, mut task_group: Self):
        """Implementation of the await functionality for TaskGroup.

        Args:
            hdl: The coroutine handle to be awaited.
            task_group: The TaskGroup to be awaited.
        """
        _async_and_then(hdl, Pointer(to=task_group.chain))
        task_group._task_complete()

    @always_inline
    def __await__(mut self):
        """Make TaskGroup awaitable in async contexts.

        This allows using 'await task_group' syntax in async functions.
        """

        @always_inline
        def await_body(cur_hdl: AnyCoroutine) {mut}:
            Self.await_body_impl(cur_hdl, self)

        _suspend_async(await_body)

    # FIXME: OriginSet isn't a first class type.  This API isn't very usable.
    def wait[origins: OriginSet = origin_of()._mlir_origin](mut self):
        """Wait for all tasks in the `TaskGroup` to complete.

        This is a blocking call that returns only when all tasks have finished.

        Parameters:
            origins: The origin set for the wait operation.
        """
        self._task_complete()
        _async_wait(Pointer(to=self.chain))
