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

"""This module provides functionality for building and replaying device graphs.
A device graph captures a sequence of GPU operations (such as kernel launches,
memory copies, and memsets) as a reusable graph that can be replayed at a lower
overhead than re-enqueueing each operation individually. The main entry point
is [`DeviceGraph.create()`](/api/mojo/max/gpu/host/device_graph/DeviceGraph/#create),
which hands a [`DeviceGraphBuilder`](/api/mojo/max/gpu/host/device_graph/DeviceGraphBuilder/)
to a scoped callback.

Graph capture is currently implemented for CUDA and HIP devices only.
Creating a graph on any other device, such as an Apple GPU or a CPU, raises."""

from max.gpu.host import (
    ConstantMemoryMapping,
    Dim,
    FuncAttribute,
    LaunchAttribute,
)

from std.collections.optional import OptionalReg
from std.ffi import c_size_t, external_call
from std.logger import Logger
from std.sys import align_of, bit_width_of, size_of
from std.memory import unsafe_memcpy
from std.memory.unsafe import bitcast
from std.reflection import call_location, SourceLocation
from std.utils.lock import BlockingScopedLock, BlockingSpinLock
from std.utils.static_tuple import StaticTuple
from std.builtin.device_passable import DevicePassable

from max.runtime.async_value import AnyAsyncValueRef

from max.gpu.host.device_context import (
    DeviceBuffer,
    DeviceContext,
    DeviceExternalFunction,
    DeviceFunction,
    HostBuffer,
    _check_dim,
    _checked,
    _CString,
    _DeviceBufferPtr,
    _DeviceContextPtr,
    _DeviceFunctionPtr,
    _DumpPath,
    _FunctionEnqueuer,
    _LaunchBits,
)

comptime _logger = Logger()


struct _DeviceGraphBuilderCpp:
    pass


struct _DeviceGraphCpp:
    pass


comptime _DeviceGraphBuilderPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = OptionalPointer[_DeviceGraphBuilderCpp, origin]

comptime _DeviceGraphPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = OptionalPointer[_DeviceGraphCpp, origin]


struct _DeviceGraphMemoryPoolCpp:
    pass


comptime _DeviceGraphMemoryPoolPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = OptionalPointer[_DeviceGraphMemoryPoolCpp, origin]


@fieldwise_init
struct DeviceGraphNode[arena_origin: ImmOrigin](
    TrivialRegisterPassable, Writable
):
    """A handle to a node in an under-construction device graph.

    Returned by node-adding methods on `DeviceGraphBuilder` such as
    `add_function`, `add_copy`, and `add_memset`. The handle can be used to
    refer to the node from later API calls (for example, when expressing
    explicit dependency edges).

    Parameters:
        arena_origin: Origin of the `DeviceGraph.create` scope that produced
            this handle. Branding ties the handle's usability to that scope,
            so a node cannot be used outside the builder callback or mixed
            into a different graph.
    """

    var id: Int32
    """Opaque integer identifier of the node within its graph builder."""

    @always_inline
    def write_to(self, mut writer: Some[Writer]):
        """Writes a human-readable representation of this node handle.

        Args:
            writer: The writer to output to.
        """
        writer.write("DeviceGraphNode(id=", self.id, ")")


@doc_hidden
@fieldwise_init
struct _GraphDepArgs[origin: ImmOrigin](TrivialRegisterPassable):
    """C ABI representation of the dependency list passed to the
    `AsyncRT_DeviceGraphBuilder_add*` exports.

    `count` is the (non-negative) number of `Int32` node ids that `ids`
    points to. When `count == 0`, `ids` may be a dangling pointer (the C
    side never dereferences it).
    """

    var ids: ImmPointer[Int32, Self.origin]
    var count: Int64


@doc_hidden
@always_inline
def _pack_dep_args[
    o: ImmOrigin
](deps: Span[DeviceGraphNode[o], _]) -> _GraphDepArgs[deps.origin]:
    """Packs an explicit dependency list into the (ids, count) pair used by
    the AsyncRT_DeviceGraphBuilder_add* C ABI exports.

    `DeviceGraphNode` is a single-Int32 struct, so `Span.unsafe_ptr()` can
    be bitcast directly to `Pointer[Int32]`. The matching C++ side
    static_asserts this layout invariant in MojoBindings.cpp.

    The returned `_GraphDepArgs` carries `deps.origin`, so the borrow
    checker keeps `deps` alive for as long as the packed `ids` pointer is
    in use.
    """
    return _GraphDepArgs[deps.origin](
        ids=deps.unsafe_ptr().unsafe_bitcast[Int32](),
        count=Int64(len(deps)),
    )


@doc_hidden
struct DeviceGraphMemoryPool(Equatable, ImplicitlyCopyable, Writable):
    """Owning handle to a device graph memory pool shared across graphs.

    Every allocation a device graph records must stay reserved for the graph's
    lifetime, since the graph bakes raw addresses into its nodes. By default
    each builder creates a private pool, so graphs never share activation
    memory; handing the same pool to several builders (via
    `DeviceGraphCache.get_or_create_pool()`) lets their graphs reuse one
    another's transient allocations instead.

    Sharing is sound only while the graphs sharing a pool replay serially in
    recording order and do not rely on transient allocations persisting across
    replays.

    Not re-exported from `max.gpu.host`: the HAL build shares that package's
    `__init__.mojo` but swaps this module for a stub without this type.
    """

    var _handle: _DeviceGraphMemoryPoolPtr[mut=True]
    var _ctx: DeviceContext

    def __init__(out self, ctx: DeviceContext):
        """Creates a fresh pool bound to the context's device.

        Args:
            ctx: The device context whose device the pool allocates from.
        """
        # DeviceGraphMemoryPool *AsyncRT_DeviceContext_createGraphMemoryPool(
        #     DeviceContext *ctx)
        self._handle = external_call[
            "AsyncRT_DeviceContext_createGraphMemoryPool",
            _DeviceGraphMemoryPoolPtr[mut=True],
            _DeviceContextPtr[mut=True],
        ](ctx._handle)
        self._ctx = ctx

    def __init__(out self, *, copy: Self):
        """Creates a copy of an existing pool handle by incrementing its
        reference count.

        Args:
            copy: The pool handle to copy.
        """
        # void AsyncRT_DeviceGraphMemoryPool_retain(DeviceGraphMemoryPool *pool)
        external_call[
            "AsyncRT_DeviceGraphMemoryPool_retain",
            NoneType,
            _DeviceGraphMemoryPoolPtr[mut=True],
        ](copy._handle)
        self._handle = copy._handle
        self._ctx = copy._ctx

    def __deinit__(deinit self):
        """Releases this reference to the pool."""
        # void AsyncRT_DeviceGraphMemoryPool_release(DeviceGraphMemoryPool *pool)
        external_call[
            "AsyncRT_DeviceGraphMemoryPool_release",
            NoneType,
            _DeviceGraphMemoryPoolPtr[mut=True],
        ](self._handle)

    def __eq__(self, other: DeviceGraphMemoryPool) -> Bool:
        return self._handle == other._handle

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "DeviceGraphCache(", self._handle, ", ", self._ctx.api(), ")"
        )


struct DeviceGraphCache(Movable):
    """Holds the device graphs a model has already built, keyed for reuse.

    A graph is expensive to record and instantiate, so
    [`DeviceGraph.create()`](/api/mojo/max/gpu/host/device_graph/DeviceGraph/#create)
    consults a cache before doing either. The key is derived from the graph's
    inputs, so a later call with equivalent inputs replays the graph the first
    call built.

    Each individual operation is safe to call concurrently. A miss is not
    exclusive, though: callers that miss together will each build a graph and the
    last one to `add()` wins, so concurrent first use of one key costs duplicated
    recording rather than a single shared graph.
    """

    var _cache: Dict[String, DeviceGraph]
    var _pools: Dict[Int, DeviceGraphMemoryPool]
    """Shared memory pools, one per device context the cache has seen, keyed
    by the context's pointer identity. Each pool retains its C++ context, so a
    key's referent stays alive for as long as its entry does."""

    var _lock: BlockingSpinLock
    """Lock used to allow safe mutation of this structure in a concurrent
    context. The general assumption this type makes is that locks are held for
    a very short duration."""

    def __init__(out self):
        """Creates an empty cache."""
        self._cache = {}
        self._pools = {}
        self._lock = BlockingSpinLock()

    @staticmethod
    def make_key[
        *Ts: DeviceGraphInput
    ](
        build: Some[def[o: ImmOrigin](mut DeviceGraphBuilder[o]) raises],
        *inputs: *Ts,
    ) -> String:
        """Derives the cache key for a graph built from the given inputs.

        Two calls agree on a key exactly when they pass the same work function
        and their inputs write the same contributions in the same order.

        Parameters:
            Ts: Types of the device graph inputs.

        Args:
            build: The work function that records the graph. Its type is what
                identifies the graph independently of its inputs; the function
                is never called here.
            inputs: The inputs whose contributions distinguish this graph.

        Returns:
            The cache key.
        """
        return Self._make_key(build, inputs)

    # Takes the pack itself so a variadic caller can forward its own inputs,
    # which the variadic spelling above cannot express.
    @staticmethod
    def _make_key[
        origin: ImmOrigin, //, *Ts: DeviceGraphInput
    ](
        build: Some[def[o: ImmOrigin](mut DeviceGraphBuilder[o]) raises],
        inputs: VariadicPack[
            origin=origin, element_trait=DeviceGraphInput, False, *Ts
        ],
    ) -> String:
        var key = String(reflect[type_of(build)].name())

        # Every contribution is separated, so no set of inputs can spell the
        # same key as a different set by running together -- inputs writing
        # `1` then `23` must not collide with `12` then `3`. Framing the key
        # here rather than in `write_graph_key` keeps that guarantee
        # independent of how each input chooses to describe itself.
        comptime for i in range(len(Ts)):
            key.write("|")
            inputs[i].write_graph_key(key)

        return key^

    def lookup(mut self, key: String) -> Optional[DeviceGraph]:
        """Returns the graph stored under a key, if there is one.

        Args:
            key: The cache key to look up.

        Returns:
            The cached graph, or `None` on a miss.
        """
        with BlockingScopedLock(self._lock):
            # A copy, so it stays valid once another thread may replace the
            # entry.
            return self._cache.find(key)

    def cache(mut self, var key: String, var graph: DeviceGraph) -> DeviceGraph:
        """Interns the device graph identified by the given key.

        If there is already a graph identified by the given key, that graph is
        returned. Otherwise, the supplied graph is inserted into the cache and
        returned.

        Args:
            key: The cache key to store the graph under.
            graph: The graph to store.

        Returns:
            The canonicalized graph in the cache.
        """
        with BlockingScopedLock(self._lock):
            var result = self._cache.find(key)
            if result:
                return result.take()

            self._cache[key^] = graph
            return graph^

    @doc_hidden
    def get_or_create_pool(
        mut self, ctx: DeviceContext
    ) -> DeviceGraphMemoryPool:
        """Returns the shared memory pool for a device context, creating it on
        first use.

        Graphs built through this cache allocate from one pool per context, so
        they share activation memory instead of each reserving its own. The
        graphs a model caches replay serially in recording order, which is
        what makes the sharing sound (see `DeviceGraphMemoryPool`).

        Args:
            ctx: The device context whose pool to return.

        Returns:
            The pool shared by every graph this cache builds against `ctx`.
        """
        var key = Int(ctx._handle.unsafe_value())
        # This is its own `_lock` acquisition: the lock is not reentrant, so
        # this method must never be called from inside `lookup` or `cache`.
        with BlockingScopedLock(self._lock):
            var found = self._pools.find(key)
            if found:
                return found.take()

            var pool = DeviceGraphMemoryPool(ctx)
            self._pools[key] = pool
            return pool^


trait DeviceGraphInput(ImplicitlyCopyable):
    """A device graph input that contributes to the graph's cache key."""

    def write_graph_key(self, mut writer: Some[Writer]):
        """Writes this input's contribution to a graph's cache key.

        Write whatever distinguishes graphs that must not be shared: two inputs
        that write the same bytes are treated as interchangeable. Contributions
        are separated from each other by
        [`DeviceGraphCache.make_key()`](/api/mojo/max/gpu/host/device_graph/DeviceGraphCache/#make_key),
        so there is no need to delimit or tag the output for that purpose.

        Args:
            writer: The writer accumulating the cache key.
        """
        ...

    def allocate_stable(self, mut builder: DeviceGraphBuilder) raises -> Self:
        """Returns a graph-owned value of this type for the graph to record.

        Allocate through
        [`DeviceGraphBuilder.create_input_buffer()`](/api/mojo/max/gpu/host/device_graph/DeviceGraphBuilder/#create_input_buffer),
        which reserves the address for the graph's lifetime and registers it, and
        describe the result with the same shape and dtype as `self`.

        Read only `self`'s shape, unless the input's data is host-resident,
        in which case also copy `self`'s contents into the stable allocation
        before returning. Recorded ops read host values at enqueue time, so
        the stable location must already hold the caller's bytes while the
        build closure records.

        A *mutable* input the graph writes in place (a KV cache, for example)
        must instead call
        [`DeviceGraphBuilder.register_in_place_input()`](/api/mojo/max/gpu/host/device_graph/DeviceGraphBuilder/#register_in_place_input)
        and return a value aliasing `self`: a private stable copy would
        silently discard the graph's writes. Such an implementation must
        include the input's address in `write_graph_key`, so a moved buffer
        misses the cache and forces a rebuild instead of replaying stale
        addresses.

        Args:
            builder: The builder for the graph under construction.

        Returns:
            A value of the same type backed by the graph's memory pool, or —
            for a mutable in-place input — a value aliasing `self`.

        Raises:
            If allocating from the graph's memory pool fails.
        """
        ...


struct DeviceGraph(ImplicitlyCopyable, Writable):
    """Represents an instantiated device graph that can be replayed.

    A `DeviceGraph` captures a sequence of GPU operations (such as kernel
    launches) as a reusable graph. Once instantiated from a
    `DeviceGraphBuilder`, the graph can be replayed multiple times at a
    lower overhead than re-enqueueing each operation individually.

    To obtain a `DeviceGraph`, use
    [`DeviceGraph.create()`](/api/mojo/max/gpu/host/device_graph/DeviceGraph/#create).

    Graph capture is currently implemented for CUDA and HIP devices only.
    Creating a graph on any other device, such as an Apple GPU or a CPU, raises.
    """

    var _handle: _DeviceGraphPtr[mut=True]

    @doc_hidden
    def __init__(out self, handle: _DeviceGraphPtr[mut=True]):
        self._handle = handle

    def __init__(out self, *, copy: Self):
        """Creates a copy of an existing device graph by incrementing its
        reference count.

        Args:
            copy: The device graph to copy.
        """
        # void AsyncRT_DeviceGraph_retain(DeviceGraph *graph)
        external_call[
            "AsyncRT_DeviceGraph_retain", NoneType, _DeviceGraphPtr[mut=True]
        ](copy._handle)
        self._handle = copy._handle

    def __deinit__(deinit self):
        """Releases resources associated with this device graph."""
        # void AsyncRT_DeviceGraph_release(DeviceGraph *graph)
        external_call[
            "AsyncRT_DeviceGraph_release", NoneType, _DeviceGraphPtr[mut=True]
        ](self._handle)

    @doc_hidden
    def take_handle(deinit self) -> _DeviceGraphPtr[mut=True]:
        """Surrenders the owning handle net-zero, suppressing the destructor.

        Returns:
            The owning `DeviceGraph*`; the caller must hand it to a runtime owner
            that adopts it without an extra reference.
        """
        return self._handle

    def replay(self) raises:
        """Replays the captured sequence of GPU operations.

        Submits the pre-captured sequence of operations for execution on the
        device. This is more efficient than re-enqueueing each operation
        individually because the graph has already been compiled and
        instantiated by the driver.

        Raises:
            If replay fails.

        Example:

        ```mojo
        from max.gpu.host import DeviceContext, DeviceGraph, DeviceGraphBuilder

        def kernel():
            print("replaying")

        with DeviceContext() as ctx:
            var compiled_fn = ctx.compile_function[kernel]()

            def build(mut builder: DeviceGraphBuilder) raises {imm}:
                _ = builder.add_function(
                    compiled_fn, grid_dim=1, block_dim=1, dependencies=[]
                )

            var graph = DeviceGraph.create(ctx, build)
            graph.replay()
            graph.replay()  # replay as many times as needed
            ctx.synchronize()
        ```
        """
        # const char *AsyncRT_DeviceGraph_replay(DeviceGraph *graph)
        _checked(
            external_call[
                "AsyncRT_DeviceGraph_replay",
                _CString[],
                _DeviceGraphPtr[mut=True],
            ](self._handle)
        )

    @staticmethod
    def create[
        *Ts: DeviceGraphInput,
    ](
        ctx: DeviceContext,
        build: Some[def[o: ImmOrigin](mut DeviceGraphBuilder[o]) raises],
        *inputs: *Ts,
        cache: Pointer[mut=True, DeviceGraphCache, _],
    ) raises -> DeviceGraph:
        """Builds and instantiates a device graph, reusing a cached one if it
        can.

        Behaves like the uncached overload, except that a graph an earlier call
        built from equivalent `inputs` is returned as-is and `build` is never
        called.

        Parameters:
            Ts: Types of the device graph inputs.

        Args:
            ctx: Device context for the target device.
            build: Callback that adds nodes to the supplied builder, called only
                on a cache miss.
            inputs: The device graph inputs the cache key is derived from.
            cache: The cache to consult, and to store a newly built graph in.

        Returns:
            The instantiated device graph, which may be one a previous call
            built.

        Raises:
            If `ctx` is on a device without graph support, or if graph builder
            creation, `build`, or instantiation fails.

        Example:

        ```mojo
        from max.gpu.host import (
            DeviceContext, DeviceGraph, DeviceGraphBuilder, DeviceGraphCache
        )

        def kernel():
            print("replaying")

        with DeviceContext() as ctx:
            var compiled_fn = ctx.compile_function[kernel]()
            var cache = DeviceGraphCache()

            def build(mut builder: DeviceGraphBuilder) raises {imm}:
                _ = builder.add_function(
                    compiled_fn, grid_dim=1, block_dim=1, dependencies=[]
                )

            # The second call reuses the graph the first one built.
            var graph = DeviceGraph.create(ctx, build, cache=Pointer(to=cache))
            var same = DeviceGraph.create(ctx, build, cache=Pointer(to=cache))
            ctx.synchronize()
        ```
        """
        # Caching of command buffers can easily exceed the maximum number of
        # supported command buffers. For now, restrict caching to genuine device
        # graph implementations.
        if ctx.api() not in ("cuda", "hip"):
            return Self.create(ctx, build)

        var key = DeviceGraphCache._make_key(build, inputs)

        var found = cache[].lookup(key)
        if found:
            _logger.info("found existing device graph for key", key)
            return found.take()
        _logger.info("recording new device graph for key", key)

        # Graphs built through the cache draw from one pool per device
        # context, so they share activation memory. Cached graphs replay
        # serially in recording order, which is what makes sharing sound.
        var graph = Self._create(ctx, build, cache[].get_or_create_pool(ctx))
        return cache[].cache(key^, graph^)

    @staticmethod
    def create(
        ctx: DeviceContext,
        build: Some[def[o: ImmOrigin](mut DeviceGraphBuilder[o]) raises],
    ) raises -> DeviceGraph:
        """Builds and instantiates a device graph within a scoped callback.

        Calls `build` with a fresh `DeviceGraphBuilder`, then instantiates the
        result into a replayable `DeviceGraph`. The builder, and any
        `DeviceGraphNode` handles obtained from it, are valid only for the
        duration of `build`: their origin is scoped to this call and cannot
        escape it, so a node handle cannot be stored beyond the callback or
        used with a different graph.

        Pass a `cache` to the overload above to reuse a previously built graph
        instead of recording one on every call.

        Graph capture is currently implemented for CUDA and HIP devices only.
        On any other device, such as an Apple GPU or a CPU, this raises before
        `build` runs.

        Args:
            ctx: Device context for the target device.
            build: Callback that adds nodes to the supplied builder. It
                receives the builder by mutable reference and therefore
                cannot instantiate it directly; instantiation happens here
                once the callback returns.

        Returns:
            The instantiated device graph.

        Raises:
            If `ctx` is on a device without graph support, or if graph builder
            creation, `build`, or instantiation fails.

        Example:

        ```mojo
        from max.gpu.host import DeviceContext, DeviceGraph, DeviceGraphBuilder

        def kernel(x: Int):
            print("Value:", x)

        with DeviceContext() as ctx:
            var compiled_fn = ctx.compile_function[kernel]()

            def build(mut builder: DeviceGraphBuilder) raises {imm}:
                _ = builder.add_function(
                    compiled_fn, 42, grid_dim=1, block_dim=1, dependencies=[]
                )

            var graph = DeviceGraph.create(ctx, build)
            graph.replay()
            ctx.synchronize()
        ```
        """
        var result: _DeviceGraphBuilderPtr[mut=True] = {}

        # const char *AsyncRT_DeviceContext_createGraphBuilder(
        #     DeviceGraphBuilder **result, DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_createGraphBuilder",
                _CString[],
                Pointer[_DeviceGraphBuilderPtr[mut=True], origin_of(result)],
                _DeviceContextPtr[mut=True],
            ](
                Pointer(to=result),
                ctx._handle,
            )
        )
        var arena: Int = 0
        var builder = DeviceGraphBuilder[origin_of(arena)](result, ctx)
        build(builder)

        return builder^.instantiate()

    @staticmethod
    def _create(
        ctx: DeviceContext,
        build: Some[def[o: ImmOrigin](mut DeviceGraphBuilder[o]) raises],
        pool: DeviceGraphMemoryPool,
    ) raises -> DeviceGraph:
        """Builds and instantiates a device graph that allocates from a shared
        memory pool.

        Behaves like the public uncached overload, except the builder draws
        its allocations from `pool` instead of a builder-private one, so the
        resulting graph shares activation memory with every other graph built
        against the same pool. Graphs sharing a pool must replay serially in
        recording order (see `DeviceGraphMemoryPool`).

        Args:
            ctx: Device context for the target device.
            build: Callback that adds nodes to the supplied builder.
            pool: The shared memory pool to allocate from; must belong to
                `ctx`'s device.

        Returns:
            The instantiated device graph.

        Raises:
            If graph builder creation, `build`, or instantiation fails.
        """
        var result: _DeviceGraphBuilderPtr[mut=True] = {}

        # const char *AsyncRT_DeviceContext_createGraphBuilderWithPool(
        #     DeviceGraphBuilder **result, DeviceContext *ctx,
        #     DeviceGraphMemoryPool *pool)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_createGraphBuilderWithPool",
                _CString[],
                Pointer[_DeviceGraphBuilderPtr[mut=True], origin_of(result)],
                _DeviceContextPtr[mut=True],
                _DeviceGraphMemoryPoolPtr[mut=True],
            ](
                Pointer(to=result),
                ctx._handle,
                pool._handle,
            )
        )
        var arena: Int = 0
        var builder = DeviceGraphBuilder[origin_of(arena)](result, ctx)
        build(builder)

        return builder^.instantiate()


struct DeviceGraphBuilder[arena_origin: ImmOrigin](Movable):
    """Builder for explicit device graph construction.

    A `DeviceGraphBuilder` is handed to the callback passed to
    [`DeviceGraph.create()`](/api/mojo/max/gpu/host/device_graph/DeviceGraph/#create).
    Callers add kernel nodes via `add_function()` from within that callback,
    which then instantiates a reusable `DeviceGraph`.

    The builder, and any `DeviceGraphNode` handles it produces, are valid only
    for the duration of the callback: their origin is scoped to the
    `DeviceGraph.create` call and cannot escape it.

    Parameters:
        arena_origin: Origin of the enclosing `DeviceGraph.create` scope.

    Example:

    ```mojo
    from max.gpu.host import DeviceContext, DeviceGraph, DeviceGraphBuilder

    def kernel(x: Int):
        print("Value:", x)

    with DeviceContext() as ctx:
        var compiled_fn = ctx.compile_function[kernel]()

        def build(mut builder: DeviceGraphBuilder) raises {imm}:
            _ = builder.add_function(
                compiled_fn, 42, grid_dim=1, block_dim=1, dependencies=[]
            )

        var graph = DeviceGraph.create(ctx, build)
        graph.replay()
        ctx.synchronize()
    ```
    """

    comptime Node = DeviceGraphNode[Self.arena_origin]
    """Node handle type produced by this builder, branded with the builder's
    `DeviceGraph.create` scope origin."""

    var _handle: _DeviceGraphBuilderPtr[mut=True]
    """Handle to the underlying ref-counted driver builder."""

    var _ctx: DeviceContext
    """The backing device context used to create the builder."""

    var _implicit_deps: List[Self.Node]
    """Predecessor edges the innermost active `region` scope injects into the
    *first* node added within it.

    Outside such a scope this is empty and node-adding methods behave exactly
    as their `dependencies` argument specifies. While a scope is active,
    `region` pushes the scope's predecessor handles here so the scope's first
    `add_*` call unions them into its own `dependencies`. Later nodes in the
    scope chain after their predecessor instead (see `_region_floor`), which
    already covers these edges transitively.
    """

    var _region_floor: Optional[Int32]
    """Id of the last node added before the innermost active `region` scope
    began, or `None` when no scope is active.

    A node whose id exceeds the floor was added by the active scope, so it is
    the chain predecessor of the next node the scope adds. `-1` is the floor
    for a scope entered before any node exists, matching the sentinel
    `AsyncRT_DeviceGraphBuilder_lastNodeIdOrNone` returns for an empty graph.
    """

    @doc_hidden
    def __init__(
        out self,
        handle: _DeviceGraphBuilderPtr[mut=True],
        ctx: DeviceContext,
    ):
        self._handle = handle
        self._ctx = ctx
        self._implicit_deps = []
        self._region_floor = None

    @always_inline
    def context(self) -> DeviceContext:
        """Returns the device context this builder records against.

        Unlike the `context()` accessors on buffer types, this is a non-raising
        read of the builder's stored device context (the `def` declares no
        `raises`).

        Returns:
            The `DeviceContext` backing this builder.
        """
        return self._ctx

    @no_inline
    def recording_context(self) raises -> DeviceContext:
        """Returns a `DeviceContext` view that records into this builder.

        Operations enqueued through the returned context (kernel launches,
        copies, memsets) are recorded as graph nodes in enqueue order,
        simulating stream ordering, instead of executing eagerly. This lets
        code written against `DeviceContext` — the bulk of the standard library
        and kernels — record into a device graph without modification.

        Setup and query calls (buffer allocation, function loading, attribute
        queries) forward to the builder's backing context. Host-visible waits
        (`synchronize()`, events, timers) raise, because they observe a
        device-side result that does not exist until replay.

        Returns:
            A `DeviceContext` that records into this builder.

        Raises:
            If the backing driver fails to create the recording context.
        """
        # Dependency tracking lives on the Mojo builder (`_implicit_deps` and
        # `region` scopes); the C++ builder does not model it. Materialize the
        # current implicit predecessor as a single empty "seed" node — its
        # dependencies come from `_merge_implicit` inside `add_empty` — and hand
        # that node across the boundary. Operations recorded through the
        # returned context chain after the seed, so they respect whatever
        # `region` scope is active when `recording_context` is called. Those
        # recorded nodes land in the same id space, so the last of them becomes
        # the chain predecessor a later `add_*` in the same scope picks up,
        # which keeps the scope serial across the boundary.
        var seed = self.add_empty()
        var result: _DeviceContextPtr[mut=True] = {}
        # const char *AsyncRT_DeviceGraphBuilder_recordingContext(
        #     DeviceContext **result, DeviceGraphBuilder *builder,
        #     int32_t seedNodeId)
        _checked(
            external_call[
                "AsyncRT_DeviceGraphBuilder_recordingContext",
                _CString[],
                Pointer[_DeviceContextPtr[mut=True], origin_of(result)],
                _DeviceGraphBuilderPtr[mut=True],
                Int32,
            ](
                Pointer(to=result),
                self._handle,
                seed.id,
            )
        )
        # `checkRef` transferred ownership of the fresh reference to us, so the
        # returned context must release it on destruction.
        var ctx = DeviceContext(ctx_ptr=result)
        ctx._owning = True
        return ctx^

    @doc_hidden
    @always_inline
    def _merge_implicit(
        self, var dependencies: List[Self.Node]
    ) -> List[Self.Node]:
        """Unions the active scope's implicit predecessor into `dependencies`.

        Returns `dependencies` unchanged when no `region` scope is active, so
        node-adding outside a scope is unaffected.

        Within a scope, a node depends on the node the scope added before it,
        which serializes the scope's nodes; the scope's incoming predecessors
        go to the first node only, since the chain covers them transitively
        from there. "The node added before it" is read from the builder rather
        than tracked here so that nodes recorded through a
        `recording_context()` — added by the C++ builder, never through this
        method — still chain correctly.

        The edge is unioned in (order is irrelevant: the dependency list is an
        unordered predecessor set).
        """
        var floor = self._region_floor
        if not floor:
            return dependencies^

        var last = self._last_node_id()
        if last and last.value() > floor.value():
            # If the lst dependency is named in the dependencies list, then we
            # don't need to merge. Merging would produce duplicated dependencies
            # which some graph APIs reject.
            for dep in dependencies:
                if dep.id == last.value():
                    return dependencies^
            dependencies.append(Self.Node(last.value()))
        else:
            dependencies.extend(Span(self._implicit_deps))
        return dependencies^

    def __deinit__(deinit self):
        """Releases resources associated with this graph builder."""
        # void AsyncRT_DeviceGraphBuilder_release(DeviceGraphBuilder *builder)
        external_call[
            "AsyncRT_DeviceGraphBuilder_release",
            NoneType,
            _DeviceGraphBuilderPtr[mut=True],
        ](self._handle)

    @doc_hidden
    def _last_node_id(self) -> Optional[Int32]:
        """Returns the id of the most recently added node, or None if no
        nodes have been added yet.

        Cannot fail. Used by `_last_node` and `region`
        to query the builder's current state.
        """
        # int32_t AsyncRT_DeviceGraphBuilder_lastNodeIdOrNone(
        #     DeviceGraphBuilder *builder)
        var id = external_call[
            "AsyncRT_DeviceGraphBuilder_lastNodeIdOrNone",
            Int32,
            _DeviceGraphBuilderPtr[mut=True],
        ](self._handle)

        if id < 0:
            return None
        return id

    @doc_hidden
    def _last_node(self) -> Optional[Self.Node]:
        """Returns a handle to the most recently added node, or `None`
        if no nodes have been added yet.

        Used internally by the public `add_*` methods to retrieve the
        handle of a node they just added; those call sites always expect
        a `Some` result and unwrap via `.value()`. The handle is branded
        with the builder's `arena_origin` (a stable struct parameter), so it
        ties to the enclosing `DeviceGraph.create` scope.
        """
        var id = self._last_node_id()
        if id:
            return Self.Node(id.value())
        return None

    @__parameter
    @always_inline
    def add_function[
        *Ts: DevicePassable
    ](
        self,
        f: DeviceFunction,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        var dependencies: List[Self.Node] = [],
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        location: Optional[SourceLocation] = None,
    ) raises -> Self.Node:
        """Adds a type-checked compiled kernel function as a node in this graph.

        Parameters:
            Ts: Argument types (must be `DevicePassable`).

        Args:
            f: The type-checked compiled function to add. Must have been
                compiled via `DeviceContext.compile_function()`.
            args: Arguments to pass to the kernel.
            grid_dim: Dimensions of the compute grid.
            block_dim: Dimensions of each thread block.
            dependencies: Explicit list of predecessor node handles, unioned
                with the implicit predecessor of the enclosing `region` scope
                if there is one. Outside such a scope an empty list makes the
                new node a graph root with no predecessors.
            cluster_dim: Cluster dimensions (optional).
            shared_mem_bytes: Amount of dynamic shared memory per block.
            attributes: Launch attributes.
            constant_memory: Constant memory mappings.
            location: Source location for the function call.

        Returns:
            A handle to the newly added kernel-dispatch node.

        Raises:
            If adding the node fails.
        """
        _check_dim["DeviceGraphBuilder.add_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceGraphBuilder.add_function", "block_dim"](
            block_dim, location=call_location()
        )
        dependencies = self._merge_implicit(dependencies^)
        # Build a transient enqueuer that pairs the builder handle with the
        # caller-supplied deps. It implements `_FunctionEnqueuer` so the
        # trait machinery in `_call_with_pack_checked` routes the call into
        # our C ABI, deps and all. (`_DeviceGraphBuilderEnqueuer` is defined
        # below `DeviceGraphBuilder` because it borrows `Self`.)
        var enqueuer = _DeviceGraphBuilderEnqueuer(self, dependencies^)
        f._call_with_pack_checked(
            enqueuer,
            *args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            location=location.or_else(call_location()),
        )
        return self._last_node().value()

    @always_inline
    def add_function[
        *Ts: AnyType
    ](
        self,
        f: DeviceExternalFunction,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        var dependencies: List[Self.Node] = [],
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        location: Optional[SourceLocation] = None,
    ) raises -> Self.Node:
        """Adds an external device function as a node in this graph.

        This overload accepts a `DeviceExternalFunction` that was loaded from
        assembly code (PTX/SASS) via `DeviceContext.load_function()`. Because
        the function was not compiled from Mojo source, its arguments cannot
        be type-checked against a declared signature.

        Parameters:
            Ts: Argument types to pass to the external function.

        Args:
            f: The external device function to add.
            args: Arguments to pass to the function.
            grid_dim: Dimensions of the compute grid.
            block_dim: Dimensions of each thread block.
            dependencies: Explicit list of predecessor node handles, unioned
                with the implicit predecessor of the enclosing `region` scope
                if there is one. Outside such a scope an empty list makes the
                new node a graph root with no predecessors.
            cluster_dim: Cluster dimensions (optional).
            shared_mem_bytes: Amount of dynamic shared memory per block.
            attributes: Launch attributes.
            constant_memory: Constant memory mappings.
            location: Source location for the function call.

        Returns:
            A handle to the newly added kernel-dispatch node.

        Raises:
            If adding the node fails.
        """
        _check_dim["DeviceGraphBuilder.add_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceGraphBuilder.add_function", "block_dim"](
            block_dim, location=call_location()
        )
        dependencies = self._merge_implicit(dependencies^)
        # Build a transient enqueuer that pairs the builder handle with the
        # caller-supplied deps. It implements `_FunctionEnqueuer` so the
        # trait machinery in `_call_with_pack` routes the call into our
        # C ABI, deps and all. (`_DeviceGraphBuilderEnqueuer` is defined
        # below `DeviceGraphBuilder` because it borrows `Self`.)
        var enqueuer = _DeviceGraphBuilderEnqueuer(self, dependencies^)
        f._call_with_pack(
            enqueuer,
            *args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            location=location.or_else(call_location()),
        )
        return self._last_node().value()

    @always_inline
    def add_function[
        declared_arg_types: TypeList[Trait=AnyType, ...],
        //,
        func: def(* args: * declared_arg_types) thin -> None,
        *actual_arg_types: DevicePassable,
        link_options: StaticString = "",
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
        _ptxas_info_verbose: Bool = False,
    ](
        self,
        *args: *actual_arg_types,
        grid_dim: Dim,
        block_dim: Dim,
        var dependencies: List[Self.Node] = [],
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        func_attribute: OptionalReg[FuncAttribute] = None,
        location: Optional[SourceLocation] = None,
    ) raises -> Self.Node:
        """Compiles a thin kernel and adds it as a node in this graph.

        Takes the kernel as a compile-time function pointer (`thin`) and
        compiles it with the `DeviceContext` that created this builder. This
        is the same identity as `DeviceContext.compile_function[kernel]()`.
        Capturing closures (unified closures with a `{}` capture list) use
        the `DevicePassable`- and `RegisterPassable`-constrained `add_function`
        overloads below.

        Parameters:
            declared_arg_types: Types of the arguments to pass to the device
                function.
            func: The thin kernel to compile and add as a graph node.
            actual_arg_types: The types of the arguments being passed to the
                function.
            link_options: Additional linker flags and options as a string.
            dump_asm: To dump the compiled assembly, pass `True`, or a file
                path to dump to, or a function returning a file path.
            dump_llvm: To dump the generated LLVM code, pass `True`, or a file
                path to dump to, or a function returning a file path.
            _dump_sass: Only runs on NVIDIA targets, and requires CUDA Toolkit
                to be installed. Pass `True`, or a file path to dump to, or a
                function returning a file path.
            _ptxas_info_verbose: Only runs on NVIDIA targets, and requires CUDA
                Toolkit to be installed. Changes `dump_asm` to output verbose
                PTX assembly (default `False`).

        Args:
            args: Variadic arguments which are passed to the `func`.
            grid_dim: Dimensions of the compute grid.
            block_dim: Dimensions of each thread block.
            dependencies: Explicit list of predecessor node handles, unioned
                with the implicit predecessor of the enclosing `region` scope
                if there is one. Outside such a scope an empty list makes the
                new node a graph root with no predecessors.
            cluster_dim: Cluster dimensions (optional).
            shared_mem_bytes: Amount of dynamic shared memory per block.
            attributes: Launch attributes.
            constant_memory: Constant memory mappings.
            func_attribute: `CUfunction_attribute` enum.
            location: Source location for the function call.

        Returns:
            A handle to the newly added kernel-dispatch node.

        Raises:
            If adding the node fails.

        Example:

        ```mojo
        from max.gpu.host import DeviceContext, DeviceGraph, DeviceGraphBuilder

        def kernel(x: Int):
            print("Value:", x)

        with DeviceContext() as ctx:
            def build(mut builder: DeviceGraphBuilder) raises {imm}:
                _ = builder.add_function[kernel](
                    42, grid_dim=1, block_dim=1, dependencies=[]
                )

            var graph = DeviceGraph.create(ctx, build)
            graph.replay()
            ctx.synchronize()
        ```
        """
        _check_dim["DeviceGraphBuilder.add_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceGraphBuilder.add_function", "block_dim"](
            block_dim, location=call_location()
        )

        var inferred_func_attribute = func_attribute
        if not func_attribute and shared_mem_bytes:
            var max_shared = self._ctx._get_max_dynamic_shared_memory_bytes(
                shared_mem_bytes.value()
            )
            if max_shared > 0:
                inferred_func_attribute = (
                    FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(max_shared)
                )

        var gpu_kernel = self._ctx.compile_function[
            func,
            dump_asm=dump_asm,
            dump_llvm=dump_llvm,
            link_options=link_options,
            _dump_sass=_dump_sass,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ](func_attribute=inferred_func_attribute)

        return self.add_function(
            gpu_kernel,
            *args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            dependencies=dependencies^,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            location=location.or_else(call_location()),
        )

    @no_inline
    def add_function[
        FuncType: DevicePassable & def() -> None,
        //,
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
        _ptxas_info_verbose: Bool = False,
    ](
        self,
        func: FuncType,
        grid_dim: Dim,
        block_dim: Dim,
        var dependencies: List[Self.Node] = [],
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        func_attribute: OptionalReg[FuncAttribute] = None,
        location: Optional[SourceLocation] = None,
    ) raises -> Self.Node:
        """Compiles a `DevicePassable` capturing closure and adds it as a node.

        This overload is for closures that conform to `DevicePassable`: the
        closure (a unified closure with a `{}` capture list) is encoded via
        its `device_type` so host handles such as `DevicePointer` become
        device addresses, and `FuncType.__call__` is compiled as the kernel.
        The encoded closure is the single launch argument. Captures flow from
        the host scope into the kernel through the `DevicePassable` encoding
        rather than through explicit positional arguments.

        Parameters:
            FuncType: The closure type (usually inferred). Must conform to
                `DevicePassable` and be callable with no arguments.
            dump_asm: To dump the compiled assembly, pass `True`, or a file
                path to dump to, or a function returning a file path.
            dump_llvm: To dump the generated LLVM code, pass `True`, or a file
                path to dump to, or a function returning a file path.
            _dump_sass: Only runs on NVIDIA targets, and requires CUDA Toolkit
                to be installed. Pass `True`, or a file path to dump to, or a
                function returning a file path.
            _ptxas_info_verbose: Only runs on NVIDIA targets, and requires CUDA
                Toolkit to be installed. Changes `dump_asm` to output verbose
                PTX assembly (default `False`).

        Args:
            func: The capturing closure to compile and add as a graph node.
            grid_dim: Dimensions of the compute grid.
            block_dim: Dimensions of each thread block.
            dependencies: Explicit list of predecessor node handles, unioned
                with the implicit predecessor of the enclosing `region` scope
                if there is one. Outside such a scope an empty list makes the
                new node a graph root with no predecessors.
            cluster_dim: Cluster dimensions (optional).
            shared_mem_bytes: Amount of dynamic shared memory per block.
            attributes: Launch attributes.
            constant_memory: Constant memory mappings.
            func_attribute: `CUfunction_attribute` enum.
            location: Source location for the function call.

        Returns:
            A handle to the newly added kernel-dispatch node.

        Raises:
            If compiling the closure or adding the node fails.
        """
        _check_dim["DeviceGraphBuilder.add_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceGraphBuilder.add_function", "block_dim"](
            block_dim, location=call_location()
        )
        dependencies = self._merge_implicit(dependencies^)

        # The compiled kernel is `FuncType.__call__`; the launch argument is
        # the encoded `FuncType.device_type` instance. Layout punning is only
        # safe while those sizes and alignments coincide on the launch target.
        comptime launch_target = DeviceContext.default_device_info.target()
        comptime host_size = size_of[FuncType, target=launch_target]()
        comptime device_size = size_of[
            FuncType.device_type, target=launch_target
        ]()
        comptime host_align = align_of[FuncType, target=launch_target]()
        comptime device_align = align_of[
            FuncType.device_type, target=launch_target
        ]()
        comptime assert host_size == device_size, String(
            "capturing add_function requires size_of[FuncType] to match",
            " size_of[FuncType.device_type] on the launch target; host=",
            host_size,
            " device=",
            device_size,
        )
        comptime assert host_align == device_align, String(
            "capturing add_function requires align_of[FuncType] to match",
            " align_of[FuncType.device_type] on the launch target; host=",
            host_align,
            " device=",
            device_align,
        )

        var gpu_kernel = DeviceFunction[
            FuncType.__call__,
            TypeList.of[Trait=AnyType, FuncType.device_type](),
            target=launch_target,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ](self._ctx, func_attribute=func_attribute)
        gpu_kernel.dump_rep[
            dump_asm=dump_asm,
            dump_llvm=dump_llvm,
            _dump_sass=_dump_sass,
        ]()

        # Build a transient enqueuer that pairs the builder handle with the
        # caller-supplied deps; see the other `add_function` overloads.
        var enqueuer = _DeviceGraphBuilderEnqueuer(self, dependencies^)
        gpu_kernel._call_with_pack_checked(
            enqueuer,
            func,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            location=location.or_else(call_location()),
        )
        return self._last_node().value()

    @no_inline
    def add_function[
        FuncType: RegisterPassable & def() -> None,
        //,
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
        _ptxas_info_verbose: Bool = False,
    ](
        self,
        func: FuncType,
        grid_dim: Dim,
        block_dim: Dim,
        var dependencies: List[Self.Node] = [],
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        func_attribute: OptionalReg[FuncAttribute] = None,
        location: Optional[SourceLocation] = None,
    ) raises -> Self.Node where not conforms_to(FuncType, DevicePassable):
        """Bit-copies a register-passable capturing closure and adds it as a
        node.

        Callable values parameterized by a capturing function type cannot
        implement `DevicePassable` (that bound makes every method `capturing
        thin`). This overload is for such closures: it bit-copies the closure
        into a `DevicePassable` bag (`_LaunchBits`), compiles
        `FuncType.__call__`, and launches the bag as the single argument.
        Capturing closures that do conform to `DevicePassable` use the
        encoding overload above instead.

        Parameters:
            FuncType: The closure type (usually inferred). Must conform to
                `RegisterPassable` and be callable with no arguments, and must
                not conform to `DevicePassable`.
            dump_asm: To dump the compiled assembly, pass `True`, or a file
                path to dump to, or a function returning a file path.
            dump_llvm: To dump the generated LLVM code, pass `True`, or a file
                path to dump to, or a function returning a file path.
            _dump_sass: Only runs on NVIDIA targets, and requires CUDA Toolkit
                to be installed. Pass `True`, or a file path to dump to, or a
                function returning a file path.
            _ptxas_info_verbose: Only runs on NVIDIA targets, and requires CUDA
                Toolkit to be installed. Changes `dump_asm` to output verbose
                PTX assembly (default `False`).

        Args:
            func: The register-passable capturing closure to compile and add
                as a graph node.
            grid_dim: Dimensions of the compute grid.
            block_dim: Dimensions of each thread block.
            dependencies: Explicit list of predecessor node handles, unioned
                with the implicit predecessor of the enclosing `region` scope
                if there is one. Outside such a scope an empty list makes the
                new node a graph root with no predecessors.
            cluster_dim: Cluster dimensions (optional).
            shared_mem_bytes: Amount of dynamic shared memory per block.
            attributes: Launch attributes.
            constant_memory: Constant memory mappings.
            func_attribute: `CUfunction_attribute` enum.
            location: Source location for the function call.

        Returns:
            A handle to the newly added kernel-dispatch node.

        Raises:
            If compiling the closure or adding the node fails.
        """
        _check_dim["DeviceGraphBuilder.add_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceGraphBuilder.add_function", "block_dim"](
            block_dim, location=call_location()
        )
        dependencies = self._merge_implicit(dependencies^)

        comptime launch_target = DeviceContext.default_device_info.target()
        comptime n = size_of[FuncType, target=launch_target]()
        comptime a = align_of[FuncType, target=launch_target]()
        var bits = _LaunchBits[n, a](StaticTuple[UInt8, n]())
        unsafe_memcpy(
            dest=Pointer(to=bits.storage).unsafe_bitcast[FuncType](),
            src=Pointer(to=func),
            count=1,
        )

        var gpu_kernel = DeviceFunction[
            FuncType.__call__,
            TypeList.of[Trait=AnyType, _LaunchBits[n, a]](),
            target=launch_target,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ](self._ctx, func_attribute=func_attribute)
        gpu_kernel.dump_rep[
            dump_asm=dump_asm,
            dump_llvm=dump_llvm,
            _dump_sass=_dump_sass,
        ]()

        # Build a transient enqueuer that pairs the builder handle with the
        # caller-supplied deps; see the other `add_function` overloads.
        var enqueuer = _DeviceGraphBuilderEnqueuer(self, dependencies^)
        gpu_kernel._call_with_pack_checked(
            enqueuer,
            bits,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            location=location.or_else(call_location()),
        )
        return self._last_node().value()

    @no_inline
    def add_copy[
        dtype: DType
    ](
        self,
        dst_buf: DeviceBuffer[dtype, ...],
        src_buf: HostBuffer[dtype, ...],
        *,
        var dependencies: List[Self.Node] = [],
    ) raises -> Self.Node:
        """Adds a host-to-device memcpy node to the graph.

        The number of bytes copied is determined by the size of the device
        buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src_buf: Host buffer to copy from.
            dependencies: Explicit list of predecessor node handles, unioned
                with the implicit predecessor of the enclosing `region` scope
                if there is one. Outside such a scope an empty list makes the
                new node a graph root with no predecessors.

        Returns:
            A handle to the newly added memcpy node.

        Raises:
            If adding the node fails.
        """
        dependencies = self._merge_implicit(dependencies^)
        var dep_args = _pack_dep_args(dependencies)
        # const char *AsyncRT_DeviceGraphBuilder_addCopyHostToDevice(
        #     DeviceGraphBuilder *builder, DeviceBuffer *dst, const void *src,
        #     const int32_t *depIds, int64_t numDeps)
        _checked(
            external_call[
                "AsyncRT_DeviceGraphBuilder_addCopyHostToDevice",
                _CString[],
            ](
                self._handle,
                dst_buf._handle,
                src_buf._host_ptr,
                dep_args.ids,
                dep_args.count,
            )
        )
        return self._last_node().value()

    @no_inline
    def add_copy[
        dtype: DType
    ](
        self,
        dst_buf: HostBuffer[dtype, ...],
        src_buf: DeviceBuffer[dtype, ...],
        *,
        var dependencies: List[Self.Node] = [],
    ) raises -> Self.Node:
        """Adds a device-to-host memcpy node to the graph.

        The number of bytes copied is determined by the size of the device
        buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Host buffer to copy to.
            src_buf: Device buffer to copy from.
            dependencies: Explicit list of predecessor node handles, unioned
                with the implicit predecessor of the enclosing `region` scope
                if there is one. Outside such a scope an empty list makes the
                new node a graph root with no predecessors.

        Returns:
            A handle to the newly added memcpy node.

        Raises:
            If adding the node fails.
        """
        dependencies = self._merge_implicit(dependencies^)
        var dep_args = _pack_dep_args(dependencies)
        # const char *AsyncRT_DeviceGraphBuilder_addCopyDeviceToHost(
        #     DeviceGraphBuilder *builder, void *dst, DeviceBuffer *src,
        #     const int32_t *depIds, int64_t numDeps)
        _checked(
            external_call[
                "AsyncRT_DeviceGraphBuilder_addCopyDeviceToHost",
                _CString[],
            ](
                self._handle,
                dst_buf._host_ptr,
                src_buf._handle,
                dep_args.ids,
                dep_args.count,
            )
        )
        return self._last_node().value()

    @no_inline
    def add_copy[
        dtype: DType
    ](
        self,
        dst_buf: DeviceBuffer[dtype, ...],
        src_buf: DeviceBuffer[dtype, ...],
        *,
        var dependencies: List[Self.Node] = [],
    ) raises -> Self.Node:
        """Adds a device-to-device memcpy node to the graph.

        Both buffers must belong to the same context as this builder;
        cross-context copies are not supported in graphs. The number of bytes
        copied is determined by the size of the source buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src_buf: Device buffer to copy from. Must be the same size as
                `dst_buf`.
            dependencies: Explicit list of predecessor node handles, unioned
                with the implicit predecessor of the enclosing `region` scope
                if there is one. Outside such a scope an empty list makes the
                new node a graph root with no predecessors.

        Returns:
            A handle to the newly added memcpy node.

        Raises:
            If adding the node fails.
        """
        dependencies = self._merge_implicit(dependencies^)
        var dep_args = _pack_dep_args(dependencies)
        # const char *AsyncRT_DeviceGraphBuilder_addCopyDeviceToDevice(
        #     DeviceGraphBuilder *builder, DeviceBuffer *dst, DeviceBuffer *src,
        #     const int32_t *depIds, int64_t numDeps)
        _checked(
            external_call[
                "AsyncRT_DeviceGraphBuilder_addCopyDeviceToDevice",
                _CString[],
            ](
                self._handle,
                dst_buf._handle,
                src_buf._handle,
                dep_args.ids,
                dep_args.count,
            )
        )
        return self._last_node().value()

    @no_inline
    def add_memset[
        dtype: DType
    ](
        self,
        dst: DeviceBuffer[dtype, ...],
        val: Scalar[dtype],
        *,
        var dependencies: List[Self.Node] = [],
    ) raises -> Self.Node:
        """Adds a memset node to the graph that sets all elements of `dst` to
        `val`.

        Parameters:
            dtype: Type of the data stored in the buffer.

        Args:
            dst: Destination buffer.
            val: Value to set all elements of `dst` to.
            dependencies: Explicit list of predecessor node handles, unioned
                with the implicit predecessor of the enclosing `region` scope
                if there is one. Outside such a scope an empty list makes the
                new node a graph root with no predecessors.

        Returns:
            A handle to the newly added memset node.

        Raises:
            If adding the node fails. The underlying graph APIs cannot express
            an 8-byte memset whose high and low 32-bit halves differ as a
            single node, so such patterns will return an error.
        """
        comptime bitwidth = bit_width_of[dtype]()
        comptime assert (
            bitwidth == 8 or bitwidth == 16 or bitwidth == 32 or bitwidth == 64
        ), "bitwidth of memset dtype must be one of [8,16,32,64]"
        var value: UInt64

        comptime if bitwidth == 8:
            value = UInt64(Int(bitcast[.uint8, 1](val)))
        elif bitwidth == 16:
            value = UInt64(Int(bitcast[.uint16, 1](val)))
        elif bitwidth == 32:
            value = UInt64(bitcast[.uint32, 1](val))
        else:
            value = bitcast[.uint64, 1](val)

        dependencies = self._merge_implicit(dependencies^)
        var dep_args = _pack_dep_args(dependencies)
        # const char *AsyncRT_DeviceGraphBuilder_addSetMemory(
        #     DeviceGraphBuilder *builder, DeviceBuffer *dst, uint64_t val,
        #     size_t valSize, const int32_t *depIds, int64_t numDeps)
        _checked(
            external_call[
                "AsyncRT_DeviceGraphBuilder_addSetMemory",
                _CString[],
                _DeviceGraphBuilderPtr[mut=True],
                _DeviceBufferPtr[mut=True],
                UInt64,
                c_size_t,
                Pointer[Int32, ImmutAnyOrigin],
                Int64,
            ](
                self._handle,
                dst._handle,
                value,
                c_size_t(size_of[dtype]()),
                dep_args.ids.as_unsafe_any_origin(),
                dep_args.count,
            )
        )
        return self._last_node().value()

    @no_inline
    def add_empty(
        self,
        *,
        var dependencies: List[Self.Node] = [],
    ) raises -> Self.Node:
        """Adds an empty (no-op) node to the graph.

        Empty nodes perform no work at execution time. They are used purely
        for transitive ordering: a single empty node fanned in from `m`
        predecessors and out to `n` successors expresses an `m`-to-`n`
        barrier using `m + n` edges instead of `m * n`, and serves as a
        stable handle for "the completion of this phase" when the producer
        set is not visible to the consumer.

        Args:
            dependencies: Explicit list of predecessor node handles, unioned
                with the implicit predecessor of the enclosing `region` scope
                if there is one. Outside such a scope an empty list makes the
                new node a graph root with no predecessors.

        Returns:
            A handle to the newly added empty node.

        Raises:
            If adding the node fails.
        """
        dependencies = self._merge_implicit(dependencies^)
        var dep_args = _pack_dep_args(dependencies)
        # const char *AsyncRT_DeviceGraphBuilder_addEmpty(
        #     DeviceGraphBuilder *builder, const int32_t *depIds,
        #     int64_t numDeps)
        _checked(
            external_call[
                "AsyncRT_DeviceGraphBuilder_addEmpty",
                _CString[],
                _DeviceGraphBuilderPtr[mut=True],
                Pointer[Int32, ImmutAnyOrigin],
                Int64,
            ](self._handle, dep_args.ids.as_unsafe_any_origin(), dep_args.count)
        )
        return self._last_node().value()

    @no_inline
    def region(
        mut self,
        work: Some[def[o: ImmOrigin](mut DeviceGraphBuilder[o]) raises],
        *,
        var dependencies: List[Self.Node] = [],
    ) raises -> Self.Node:
        """Runs `work` with its nodes chained in the order they were added,
        and returns the last of them.

        Each node `work` adds depends on the node `work` added before it, so
        the region's nodes execute in the order the closure recorded them.
        That makes the last node the region's sole sink, and the returned
        handle is suitable as a one-element `dependencies=` entry on a
        downstream `add_*` call: depending on it means depending on
        everything the region recorded.

        The predecessors named in `dependencies` gate the *first* node `work`
        adds, so the whole region runs after them without the closure having
        to thread the handles through to every `add_*` call. With the default
        (empty) `dependencies`, the region's nodes are unconstrained relative
        to earlier work.

        Chaining is what lets two regions in a linear dependency relationship
        be fused into one: concatenating their bodies preserves the ordering
        the dependency edge expressed. It also means a region cannot express
        two mutually independent nodes — record those in separate regions.

        Args:
            work: Closure whose effects on this builder are captured. The
                builder is passed as `work`'s sole argument; the closure
                must not capture the same builder, since doing so would
                alias with this method's receiver. The closure may add
                any number of nodes (zero or more) via any of the
                `add_*` methods.
            dependencies: Predecessor node handles the first node added by
                `work` should depend on, and therefore — through the chain —
                every node it adds. Defaults to empty (no added predecessors).

        Returns:
            A handle that successors can depend on to run after everything
            `work` added: the last node it added, or — when it added none — a
            fresh empty node depending on `dependencies`, so the region still
            chains correctly.

        Raises:
            Anything `work` itself raises, or anything raised while
            adding the fallback empty node.

        Example:

        ```mojo
        from max.gpu.host import DeviceContext, DeviceGraph, DeviceGraphBuilder

        with DeviceContext() as ctx:
            var buf_a = ctx.enqueue_create_buffer[.uint8](100)
            var buf_b = ctx.enqueue_create_buffer[.uint8](100)
            var host_src = ctx.enqueue_create_host_buffer[.uint8](100)

            def build(mut builder: DeviceGraphBuilder) raises {imm}:
                # The copy is chained after the memset, so it wins.
                def stage(mut b: DeviceGraphBuilder) raises {imm} -> None:
                    _ = b.add_memset(buf_a, UInt8(1))
                    _ = b.add_copy(buf_a, host_src)

                var staged = builder.region(stage)
                _ = builder.add_copy(buf_b, buf_a, dependencies=[staged])

            var graph = DeviceGraph.create(ctx, build)
            graph.replay()
        ```
        """
        # Resolve the incoming set in the *enclosing* scope, so a nested
        # region's first node still chains after whatever the enclosing region
        # added before it.
        var incoming = self._merge_implicit(dependencies^)

        var saved_deps = self._implicit_deps^
        var saved_floor = self._region_floor
        self._implicit_deps = incoming^
        self._region_floor = Optional(self._last_node_id().or_else(-1))

        var result: Self.Node
        try:
            work(self)

            # Chaining makes the last node the region's sole sink, so it needs
            # no join node. A region that added nothing falls back to an empty
            # node carrying the incoming predecessors, so a downstream consumer
            # still waits for them.
            var last = self._last_node_id()
            if last and last.value() > self._region_floor.value():
                result = Self.Node(last.value())
            else:
                result = self.add_empty()
        finally:
            self._implicit_deps = saved_deps^
            self._region_floor = saved_floor

        return result

    @no_inline
    def add_output(self, var output: AnyAsyncValueRef):
        """Add a value as an output for the resulting device graph.

        The graph records the output so its backing memory outlives the graph
        that references it. Ownership of the async value is transferred to the
        builder.

        Args:
            output: The async value to register as a graph output.
        """
        # void AsyncRT_DeviceGraphBuilder_addOutput(
        #     DeviceGraphBuilder *builder, AsyncValue *output)
        external_call[
            "AsyncRT_DeviceGraphBuilder_addOutput",
            NoneType,
        ](self._handle, output^.take_handle())

    @no_inline
    def num_outputs(self) -> Int:
        """Returns the number of outputs registered on the device graph.

        Returns:
            The number of outputs added via `add_output`.
        """
        # int64_t AsyncRT_DeviceGraphBuilder_numOutputs(
        #     DeviceGraphBuilder *builder)
        return Int(
            external_call[
                "AsyncRT_DeviceGraphBuilder_numOutputs",
                Int64,
                _DeviceGraphBuilderPtr[mut=True],
            ](self._handle)
        )

    @no_inline
    def add_input[T: DeviceGraphInput](mut self, input: T) raises -> T:
        """Gives an input a stable location the graph can record against.

        A recorded graph bakes in the addresses it was built with, so it cannot
        read a caller's buffer directly if it is to be replayed later. This
        allocates a graph-owned twin of `input` and registers it, so replay can
        be fed by copying the live input into that fixed location. Record the
        body against the returned value, not against `input`.

        Inputs must be added in graph-signature order.

        Parameters:
            T: The device graph input type.

        Args:
            input: The input to allocate a stable location for. Only its shape
                is read; its storage is not retained.

        Returns:
            A graph-owned value of the same type, backed by the graph's memory
            pool.

        Raises:
            If allocating the stable buffer fails.
        """
        return input.allocate_stable(self)

    @no_inline
    def num_inputs(self) -> Int:
        """Returns the number of stable inputs registered on the device graph.

        Returns:
            The number of inputs added via `add_input`, including in-place
            markers registered via `register_in_place_input`.
        """
        # int64_t AsyncRT_DeviceGraphBuilder_numInputs(
        #     DeviceGraphBuilder *builder)
        return Int(
            external_call[
                "AsyncRT_DeviceGraphBuilder_numInputs",
                Int64,
                _DeviceGraphBuilderPtr[mut=True],
            ](self._handle)
        )

    @no_inline
    def register_in_place_input(mut self):
        """Registers an in-place input marker at the current input position.

        Used by `DeviceGraphInput.allocate_stable` implementations for mutable
        inputs the graph writes in place (a KV cache, for example): the graph
        records the caller's live address directly and replay must not copy
        into that position, so no stable twin is allocated. Such an input must
        contribute its address to the graph cache key, so a moved buffer
        forces a rebuild instead of replaying stale addresses.

        Like `add_input`, markers must be registered in graph-signature order.
        """
        # void AsyncRT_DeviceGraphBuilder_addInPlaceInput(
        #     DeviceGraphBuilder *builder)
        external_call[
            "AsyncRT_DeviceGraphBuilder_addInPlaceInput",
            NoneType,
        ](self._handle)

    @doc_hidden
    @no_inline
    def instantiate(var self) raises -> DeviceGraph:
        """Instantiates the constructed graph into an executable device graph.

        Finalizes the graph construction and produces a `DeviceGraph` that
        can be replayed multiple times. Called by
        `DeviceGraph.create` once the builder callback returns;
        not part of the user-facing API (the callback receives the builder by
        reference and so cannot consume it to call this directly).

        Returns:
            The instantiated device graph.

        Raises:
            If instantiation fails.
        """
        var result: _DeviceGraphPtr[mut=True] = {}
        # const char *AsyncRT_DeviceGraphBuilder_instantiate(
        #     DeviceGraph **result, DeviceGraphBuilder *builder)
        _checked(
            external_call[
                "AsyncRT_DeviceGraphBuilder_instantiate",
                _CString[],
                Pointer[_DeviceGraphPtr[mut=True], origin_of(result)],
                _DeviceGraphBuilderPtr[mut=True],
            ](
                Pointer(to=result),
                self._handle,
            )
        )
        return DeviceGraph(result)

    @no_inline
    def create_buffer[
        dtype: DType
    ](self, size: Int, is_host: Bool, out result: DeviceBuffer[dtype]) raises:
        """Allocates a buffer for use in the current device graph.

        Device graph allocations have lifetimes tied to the device graph that
        creates this. All allocations created for a device graph should go
        through this method rather than `DeviceContext.enqueue_create_buffer`.

        Parameters:
            dtype: The element type of the resulting buffer.

        Args:
            size: The number of elements to allocate for the buffer.
            is_host: Allocate host-addressable memory instead of device memory.

        Returns:
            The newly allocated DeviceBuffer.

        Raises:
            If memory allocation fails.
        """

        comptime elem_size = size_of[dtype]()

        var cpp_handle: _DeviceBufferPtr[mut=True] = {}
        var device_ptr: Optional[result._DevicePtr] = {}

        # const char * AsyncRT_DeviceGraph_createBuffer(DeviceBuffer **result, void **devicePtr, DeviceGraphBuilder *builder, size_t len, size_t elemSize, bool isHost)
        _checked(
            external_call[
                "AsyncRT_DeviceGraph_createBuffer",
                _CString[],
            ](
                Pointer(to=cpp_handle),
                Pointer(to=device_ptr),
                self._handle,
                c_size_t(size),
                c_size_t(elem_size),
                is_host,
            ),
            location=call_location(),
        )

        result = {cpp_handle, device_ptr.value()}

    @no_inline
    def create_input_buffer[
        dtype: DType
    ](self, size: Int, is_host: Bool, out result: DeviceBuffer[dtype]) raises:
        """Allocates a buffer and registers it as a stable graph input.

        Allocation and registration are one step so a `DeviceGraphInput` cannot
        allocate a stable location without the graph learning about it, which
        would leave replay with nothing to copy into.

        Parameters:
            dtype: The element type of the resulting buffer.

        Args:
            size: The number of elements to allocate for the buffer.
            is_host: Allocate host-addressable memory instead of device memory.

        Returns:
            The newly allocated buffer, already registered with the graph.

        Raises:
            If memory allocation fails.
        """
        result = self.create_buffer[dtype](size, is_host=is_host)

        # void AsyncRT_DeviceGraphBuilder_addInput(
        #     DeviceGraphBuilder *builder, DeviceBuffer *input)
        #
        # The handle is borrowed rather than surrendered: the graph retains its
        # own reference so the pool cannot reissue this address, and the caller
        # keeps using the buffer as the input it records against.
        external_call[
            "AsyncRT_DeviceGraphBuilder_addInput",
            NoneType,
        ](self._handle, result._handle)


@doc_hidden
struct _DeviceGraphBuilderEnqueuer[
    arena_origin: ImmOrigin,
    builder_origin: ImmOrigin,
](_FunctionEnqueuer):
    """Transient `_FunctionEnqueuer` pairing a `DeviceGraphBuilder` borrow
    with the dependency list for a single node addition.

    Constructed locally inside `DeviceGraphBuilder.add_function` and passed
    to `DeviceFunction._call_with_pack[_checked]` so the explicit
    dependency list can flow through the trait machinery into the C ABI
    without becoming part of the trait surface or requiring mutable state
    on `DeviceGraphBuilder` itself.

    Parameters:
        arena_origin: Origin of the enclosing `DeviceGraph.create` scope,
            shared by the borrowed builder and the dependency handles.
        builder_origin: The origin of the borrow on the parent
            `DeviceGraphBuilder`. The borrow checker enforces that this
            enqueuer cannot outlive the originating builder.
    """

    comptime Node = DeviceGraphNode[Self.arena_origin]
    """Node handle type for this enqueuer's scope origin."""

    var _builder: Pointer[
        DeviceGraphBuilder[Self.arena_origin], Self.builder_origin
    ]
    """Borrowed reference to the parent graph builder. The Mojo borrow
    checker uses `builder_origin` to ensure this enqueuer cannot outlive
    the borrow."""

    var _dependencies: List[Self.Node]
    """Explicit dependency list for the node being added. An empty list
    creates a graph root; a non-empty list specifies exact predecessor
    edges."""

    @always_inline
    def __init__(
        out self,
        ref[Self.builder_origin] builder: DeviceGraphBuilder[Self.arena_origin],
        var dependencies: List[Self.Node],
    ):
        """Initializes the transient enqueuer with a borrowed builder and
        the dependency list to apply to the next node addition.

        Args:
            builder: The parent `DeviceGraphBuilder` whose handle is used
                for the C ABI call. Borrowed for the lifetime of this
                enqueuer.
            dependencies: Explicit dependency list for the node about to
                be added. See the field docstring on `_dependencies` for
                the meaning of each value.
        """
        self._builder = Pointer(to=builder)
        self._dependencies = dependencies^

    @always_inline
    def enqueue[
        args_origin: MutOrigin, //
    ](
        self,
        func_handle: _DeviceFunctionPtr[mut=True],
        grid_dim: Dim,
        block_dim: Dim,
        shared_mem_bytes: Int,
        attributes: Pointer[mut=True, LaunchAttribute, _],
        num_attributes: Int,
        args: Pointer[mut=True, OpaquePointer[args_origin], _],
        arg_count: UInt32,
        arg_sizes: OptionalPointer[mut=True, UInt64, _],
    ) -> _CString[]:
        """Adds a kernel-dispatch node to the borrowed graph builder.

        Forwards to `AsyncRT_DeviceGraphBuilder_addFunctionDirect`,
        attaching the dependency list captured at construction time so it
        is applied to the node being added. See `_FunctionEnqueuer.enqueue`
        for the full contract.

        Args:
            func_handle: Handle to the compiled `DeviceFunction` to launch.
            grid_dim: Grid dimensions (number of thread blocks).
            block_dim: Block dimensions (number of threads per block).
            shared_mem_bytes: Bytes of dynamic shared memory per block.
            attributes: Pointer to the launch attributes array.
            num_attributes: Number of entries in `attributes`.
            args: Pointer to the array of argument value pointers.
            arg_count: Number of entries in `args`.
            arg_sizes: Optional pointer to the per-argument sizes in bytes.

        Returns:
            A C-string carrying an error message on failure, or an empty
            string on success.
        """
        var dep_args = _pack_dep_args(self._dependencies)
        return external_call[
            "AsyncRT_DeviceGraphBuilder_addFunctionDirect", _CString[]
        ](
            self._builder[]._handle,
            func_handle,
            grid_dim.x(),
            grid_dim.y(),
            grid_dim.z(),
            block_dim.x(),
            block_dim.y(),
            block_dim.z(),
            shared_mem_bytes,
            attributes,
            num_attributes,
            args,
            arg_count,
            arg_sizes,
            dep_args.ids,
            dep_args.count,
        )
