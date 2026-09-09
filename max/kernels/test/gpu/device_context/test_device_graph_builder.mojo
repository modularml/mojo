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

from std.math import ceildiv
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.reflection import get_function_name
from std.testing import (
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_true,
)

from max.gpu.host import (
    DeviceBuffer,
    DeviceGraph,
    DeviceGraphBuilder,
    DeviceGraphCache,
    DeviceGraphInput,
)
from max.gpu.host.device_graph import DeviceGraphMemoryPool
from max.runtime.async_value import AnyAsyncValueRef


# GPU kernels live as static methods rather than module-level functions:
# `__functions_in_module()` in `main` materializes every module-level function
# as a host function value.
struct Kernels:
    @staticmethod
    def vec_add(
        output: Pointer[Float32, MutAnyOrigin],
        in0: Pointer[Float32, ImmutAnyOrigin],
        in1: Pointer[Float32, ImmutAnyOrigin],
        length_dev: Int32,
    ):
        var length = Int(length_dev)
        var tid = global_idx.x
        if tid >= length:
            return
        output[unsafe_offset=tid] = (
            in0[unsafe_offset=tid] + in1[unsafe_offset=tid]
        )

    @staticmethod
    def scaled_vec_add(
        output: Pointer[Float32, MutAnyOrigin],
        in0: Pointer[Float32, ImmutAnyOrigin],
        in1: Pointer[Float32, ImmutAnyOrigin],
        length_dev: Int32,
        scale: Float32,
    ):
        var length = Int(length_dev)
        var tid = global_idx.x
        if tid >= length:
            return
        output[unsafe_offset=tid] = (
            in0[unsafe_offset=tid] + in1[unsafe_offset=tid]
        ) * scale

    @staticmethod
    def fill_constant(
        output: Pointer[Float32, MutAnyOrigin],
        val_dev: Int32,
        length_dev: Int32,
    ):
        var val = Int(val_dev)
        var length = Int(length_dev)
        var tid = global_idx.x
        if tid >= length:
            return
        output[unsafe_offset=tid] = Float32(val)

    @staticmethod
    def add_in_place(
        buf: Pointer[Float32, MutAnyOrigin],
        delta_dev: Int32,
        length_dev: Int32,
    ):
        var delta = Int(delta_dev)
        var length = Int(length_dev)
        var tid = global_idx.x
        if tid >= length:
            return
        buf[unsafe_offset=tid] += Float32(delta)


def test_vec_add_kernel_node(ctx: DeviceContext) raises:
    print("Test capturing and replaying a vec_add kernel in a device graph.")
    comptime length = 1024
    comptime block_dim = 256

    var in0_dev = ctx.enqueue_create_buffer[.float32](length)
    var in1_dev = ctx.enqueue_create_buffer[.float32](length)
    var out_dev = ctx.enqueue_create_buffer[.float32](length)

    with in0_dev.map_to_host() as in0_host, in1_dev.map_to_host() as in1_host:
        for i in range(length):
            in0_host[i] = Float32(i)
            in1_host[i] = Float32(length - i)

    var func = ctx.compile_function[Kernels.vec_add]()

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        _ = builder.add_function(
            func,
            out_dev,
            in0_dev,
            in1_dev,
            Int32(length),
            grid_dim=ceildiv(length, block_dim),
            block_dim=block_dim,
        )

    var graph = DeviceGraph.create(ctx, build)
    graph.replay()

    # Check values and zero out buffer for next run
    with out_dev.map_to_host() as out_host:
        for i in range(length):
            assert_equal(out_host[i], Float32(length))
            out_host[i] = 0.0

    graph.replay()

    with out_dev.map_to_host() as out_host:
        for i in range(length):
            assert_equal(out_host[i], Float32(length))


def test_parameterized_kernel_node(ctx: DeviceContext) raises:
    print("Test add_function compiling a kernel without compile_function.")
    comptime length = 1024
    comptime block_dim = 256

    var in0_dev = ctx.enqueue_create_buffer[.float32](length)
    var in1_dev = ctx.enqueue_create_buffer[.float32](length)
    var out_dev = ctx.enqueue_create_buffer[.float32](length)

    with in0_dev.map_to_host() as in0_host, in1_dev.map_to_host() as in1_host:
        for i in range(length):
            in0_host[i] = Float32(i)
            in1_host[i] = Float32(length - i)

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        _ = builder.add_function[Kernels.vec_add](
            out_dev,
            in0_dev,
            in1_dev,
            Int32(length),
            grid_dim=ceildiv(length, block_dim),
            block_dim=block_dim,
        )

    var graph = DeviceGraph.create(ctx, build)
    graph.replay()

    with out_dev.map_to_host() as out_host:
        for i in range(length):
            assert_equal(out_host[i], Float32(length))


def test_scaled_kernel_node(ctx: DeviceContext) raises:
    print("Test add_function compiling a thin kernel with a scale argument.")
    comptime length = 1024
    comptime block_dim = 256
    var scale = Float32(3.0)

    var in0_dev = ctx.enqueue_create_buffer[.float32](length)
    var in1_dev = ctx.enqueue_create_buffer[.float32](length)
    var out_dev = ctx.enqueue_create_buffer[.float32](length)

    with in0_dev.map_to_host() as in0_host, in1_dev.map_to_host() as in1_host:
        for i in range(length):
            in0_host[i] = Float32(i)
            in1_host[i] = Float32(length - i)

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        _ = builder.add_function[Kernels.scaled_vec_add](
            out_dev,
            in0_dev,
            in1_dev,
            Int32(length),
            scale,
            grid_dim=ceildiv(length, block_dim),
            block_dim=block_dim,
        )

    var graph = DeviceGraph.create(ctx, build)
    graph.replay()

    with out_dev.map_to_host() as out_host:
        for i in range(length):
            assert_equal(out_host[i], Float32(length) * scale)


def test_closure_node(ctx: DeviceContext) raises:
    print("Test using a closure as a device graph node.")
    comptime length = 1024
    comptime block_dim = 256
    var scale = Float32(2.0)

    var in0_dev = ctx.enqueue_create_buffer[.float32](length)
    var in1_dev = ctx.enqueue_create_buffer[.float32](length)
    var out_dev = ctx.enqueue_create_buffer[.float32](length)

    with in0_dev.map_to_host() as in0_host, in1_dev.map_to_host() as in1_host:
        for i in range(length):
            in0_host[i] = Float32(i)
            in1_host[i] = Float32(length - i)

    var out_ptr = out_dev.unsafe_ptr()
    var in0_ptr = in0_dev.unsafe_ptr()
    var in1_ptr = in1_dev.unsafe_ptr()

    # Closure captures device pointers and scale from the enclosing scope.
    # Dispatches to the `DevicePassable` capturing-closure `add_function`
    # overload, which encodes the closure so the captured `DevicePointer`s
    # become device addresses at the launch boundary.
    def scaled_vec_add() {var scale, var out_ptr, var in0_ptr, var in1_ptr}:
        var tid = global_idx.x
        if tid >= length:
            return
        out_ptr[unsafe_offset=tid] = (
            in0_ptr[unsafe_offset=tid] + in1_ptr[unsafe_offset=tid]
        ) * scale

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        _ = builder.add_function(
            scaled_vec_add,
            grid_dim=ceildiv(length, block_dim),
            block_dim=block_dim,
        )

    var graph = DeviceGraph.create(ctx, build)
    graph.replay()

    _ = in0_dev^
    _ = in1_dev^

    with out_dev.map_to_host() as out_host:
        for i in range(length):
            assert_equal(out_host[i], Float32(length) * scale)


def test_add_copy_to_device(ctx: DeviceContext) raises:
    print("Test capturing a host-to-device memcpy node.")
    comptime length = 1024

    var host_src = ctx.enqueue_create_host_buffer[.float32](length)
    for i in range(length):
        host_src[i] = Float32(i) * 3.0
    var dev_buf = ctx.enqueue_create_buffer[.float32](length)

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        _ = builder.add_copy(dev_buf, host_src)

    var graph = DeviceGraph.create(ctx, build)
    graph.replay()
    ctx.synchronize()

    with dev_buf.map_to_host() as host_view:
        for i in range(length):
            assert_equal(host_view[i], Float32(i) * 3.0)


def test_add_copy_from_device(ctx: DeviceContext) raises:
    print("Test capturing a device-to-host memcpy node.")
    comptime length = 1024

    var dev_buf = ctx.enqueue_create_buffer[.float32](length)
    with dev_buf.map_to_host() as host_view:
        for i in range(length):
            host_view[i] = Float32(2 * i + 1)

    # Zero the host destination so we can detect that the graph wrote to it.
    var host_dst = ctx.enqueue_create_host_buffer[.float32](length)
    for i in range(length):
        host_dst[i] = 0.0

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        _ = builder.add_copy(host_dst, dev_buf)

    var graph = DeviceGraph.create(ctx, build)
    graph.replay()
    ctx.synchronize()

    for i in range(length):
        assert_equal(host_dst[i], Float32(2 * i + 1))


def test_add_copy_device_to_device(ctx: DeviceContext) raises:
    print("Test capturing a device-to-device memcpy node.")
    comptime length = 1024

    var src_dev = ctx.enqueue_create_buffer[.float32](length)
    var dst_dev = ctx.enqueue_create_buffer[.float32](length)

    with src_dev.map_to_host() as src_host:
        for i in range(length):
            src_host[i] = Float32(i * i)

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        _ = builder.add_copy(dst_dev, src_dev)

    var graph = DeviceGraph.create(ctx, build)
    graph.replay()
    ctx.synchronize()

    with dst_dev.map_to_host() as dst_host:
        for i in range(length):
            assert_equal(dst_host[i], Float32(i * i))


def test_add_memset(ctx: DeviceContext) raises:
    print("Test capturing memset nodes for 8/16/32/64-bit dtypes.")
    comptime length = 64

    var buf_u8 = ctx.enqueue_create_buffer[.uint8](length)
    var buf_u16 = ctx.enqueue_create_buffer[.uint16](length)
    var buf_u32 = ctx.enqueue_create_buffer[.uint32](length)
    var buf_u64 = ctx.enqueue_create_buffer[.uint64](length)

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        # The four memsets target disjoint buffers, so each can be an
        # independent graph root.
        _ = builder.add_memset(buf_u8, UInt8(123))
        _ = builder.add_memset(buf_u16, UInt16(0xBEEF))
        _ = builder.add_memset(buf_u32, UInt32(0xDEADBEEF))
        # Symmetric high/low halves so the graph builder can express it as a
        # single node.
        _ = builder.add_memset(buf_u64, UInt64(0x0101010101010101))

    var graph = DeviceGraph.create(ctx, build)
    graph.replay()
    ctx.synchronize()

    with buf_u8.map_to_host() as host_u8:
        for i in range(length):
            assert_equal(host_u8[i], UInt8(123))

    with buf_u16.map_to_host() as host_u16:
        for i in range(length):
            assert_equal(host_u16[i], UInt16(0xBEEF))

    with buf_u32.map_to_host() as host_u32:
        for i in range(length):
            assert_equal(host_u32[i], UInt32(0xDEADBEEF))

    with buf_u64.map_to_host() as host_u64:
        for i in range(length):
            assert_equal(host_u64[i], UInt64(0x0101010101010101))


def test_add_output(ctx: DeviceContext) raises:
    print("Test registering a graph output alongside a kernel node.")
    comptime length = 1024
    comptime block_dim = 256

    var in0_dev = ctx.enqueue_create_buffer[.float32](length)
    var in1_dev = ctx.enqueue_create_buffer[.float32](length)
    var out_dev = ctx.enqueue_create_buffer[.float32](length)

    with in0_dev.map_to_host() as in0_host, in1_dev.map_to_host() as in1_host:
        for i in range(length):
            in0_host[i] = Float32(i)
            in1_host[i] = Float32(length - i)

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        _ = builder.add_function[Kernels.vec_add](
            out_dev,
            in0_dev,
            in1_dev,
            Int32(length),
            grid_dim=ceildiv(length, block_dim),
            block_dim=block_dim,
        )
        # Register the result buffer as a graph output; the graph must still
        # instantiate and replay correctly. Copy the buffer handle so `out_dev`
        # remains valid for the host readback below.
        assert_equal(builder.num_outputs(), 0)
        builder.add_output(AnyAsyncValueRef(storage_buf=out_dev.copy()))
        assert_equal(builder.num_outputs(), 1)

    var graph = DeviceGraph.create(ctx, build)
    graph.replay()
    ctx.synchronize()

    with out_dev.map_to_host() as out_host:
        for i in range(length):
            assert_equal(out_host[i], Float32(length))


def test_add_function_with_dependencies(ctx: DeviceContext) raises:
    print(
        "Test add_function with explicit dependencies for two independent"
        " kernel chains."
    )
    comptime length = 1024
    comptime block_dim = 256
    comptime grid_dim = ceildiv(length, block_dim)

    var buf_a = ctx.enqueue_create_buffer[.float32](length)
    var buf_b = ctx.enqueue_create_buffer[.float32](length)

    var func = ctx.compile_function[Kernels.fill_constant]()

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        # Sequence A on `buf_a`: write 1, then 2, then 3, internally chained
        # via explicit dependencies. The first node is rooted with an empty
        # deps list; downstream nodes name their predecessor explicitly.
        var a0 = builder.add_function(
            func,
            buf_a,
            Int32(1),
            Int32(length),
            grid_dim=grid_dim,
            block_dim=block_dim,
        )
        var a1 = builder.add_function(
            func,
            buf_a,
            Int32(2),
            Int32(length),
            grid_dim=grid_dim,
            block_dim=block_dim,
            dependencies=[a0],
        )
        _ = builder.add_function(
            func,
            buf_a,
            Int32(3),
            Int32(length),
            grid_dim=grid_dim,
            block_dim=block_dim,
            dependencies=[a1],
        )

        # Sequence B on `buf_b`: write 4, then 5, then 6. Independent of
        # sequence A — also rooted explicitly.
        var b0 = builder.add_function(
            func,
            buf_b,
            Int32(4),
            Int32(length),
            grid_dim=grid_dim,
            block_dim=block_dim,
        )
        var b1 = builder.add_function(
            func,
            buf_b,
            Int32(5),
            Int32(length),
            grid_dim=grid_dim,
            block_dim=block_dim,
            dependencies=[b0],
        )
        _ = builder.add_function(
            func,
            buf_b,
            Int32(6),
            Int32(length),
            grid_dim=grid_dim,
            block_dim=block_dim,
            dependencies=[b1],
        )

    var graph = DeviceGraph.create(ctx, build)
    graph.replay()
    ctx.synchronize()

    with buf_a.map_to_host() as host_a:
        for i in range(length):
            assert_equal(host_a[i], Float32(3))
    with buf_b.map_to_host() as host_b:
        for i in range(length):
            assert_equal(host_b[i], Float32(6))


def test_add_memset_with_dependencies(ctx: DeviceContext) raises:
    print(
        "Test add_memset with explicit dependencies for two independent"
        " memset chains."
    )
    comptime length = 64

    var buf_a = ctx.enqueue_create_buffer[.uint8](length)
    var buf_b = ctx.enqueue_create_buffer[.uint8](length)

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        # Sequence A on `buf_a`: 0x11 -> 0x22 -> 0x33, internally chained.
        # First node is rooted with an empty deps list; sequences A and B are
        # independent because neither names a predecessor in the other chain.
        var a0 = builder.add_memset(buf_a, UInt8(0x11), dependencies=[])
        var a1 = builder.add_memset(buf_a, UInt8(0x22), dependencies=[a0])
        _ = builder.add_memset(buf_a, UInt8(0x33), dependencies=[a1])

        # Sequence B on `buf_b`: 0xAA -> 0xBB -> 0xCC. Independent of seq A.
        var b0 = builder.add_memset(buf_b, UInt8(0xAA), dependencies=[])
        var b1 = builder.add_memset(buf_b, UInt8(0xBB), dependencies=[b0])
        _ = builder.add_memset(buf_b, UInt8(0xCC), dependencies=[b1])

    var graph = DeviceGraph.create(ctx, build)
    graph.replay()
    ctx.synchronize()

    with buf_a.map_to_host() as host_a:
        for i in range(length):
            assert_equal(host_a[i], UInt8(0x33))
    with buf_b.map_to_host() as host_b:
        for i in range(length):
            assert_equal(host_b[i], UInt8(0xCC))


def test_add_copy_with_dependencies(ctx: DeviceContext) raises:
    print(
        "Test add_copy with explicit dependencies for two independent copy"
        " chains."
    )
    comptime length = 64

    var buf_a = ctx.enqueue_create_buffer[.uint32](length)
    var buf_b = ctx.enqueue_create_buffer[.uint32](length)

    # Distinct host-side payloads for each step. Pinned via
    # enqueue_create_host_buffer so the graph builder can reference them
    # directly via add_copy.
    var host_a1 = ctx.enqueue_create_host_buffer[.uint32](length)
    var host_a2 = ctx.enqueue_create_host_buffer[.uint32](length)
    var host_b1 = ctx.enqueue_create_host_buffer[.uint32](length)
    var host_b2 = ctx.enqueue_create_host_buffer[.uint32](length)
    for i in range(length):
        host_a1[i] = UInt32(0x11111111)
        host_a2[i] = UInt32(0x22222222)
        host_b1[i] = UInt32(0xAAAAAAAA)
        host_b2[i] = UInt32(0xBBBBBBBB)

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        # Sequence A: HtoD(host_a1) -> HtoD(host_a2). Final state of `buf_a`
        # is the second copy's payload (host_a2). First node rooted
        # explicitly.
        var a0 = builder.add_copy(buf_a, host_a1, dependencies=[])
        _ = builder.add_copy(buf_a, host_a2, dependencies=[a0])

        # Sequence B: HtoD(host_b1) -> HtoD(host_b2). Independent of seq A.
        var b0 = builder.add_copy(buf_b, host_b1, dependencies=[])
        _ = builder.add_copy(buf_b, host_b2, dependencies=[b0])

    var graph = DeviceGraph.create(ctx, build)
    graph.replay()
    ctx.synchronize()

    with buf_a.map_to_host() as host_a:
        for i in range(length):
            assert_equal(host_a[i], UInt32(0x22222222))
    # with buf_b.map_to_host() as host_b:
    #    for i in range(length):
    #        assert_equal(host_b[i], UInt32(0xBBBBBBBB))

    # FIXME(MSTDL-2742): HostBuffer is origin incorrect.
    _ = Pointer(to=host_a1).as_unsafe_any_origin()[]


def test_region(ctx: DeviceContext) raises:
    print(
        "Test region returns its last node, usable as a downstream node's"
        " sole dependency covering everything the scope added."
    )
    comptime length = 64

    var buf_a = ctx.enqueue_create_buffer[.uint8](length)
    var buf_b = ctx.enqueue_create_buffer[.uint8](length)
    var buf_c = ctx.enqueue_create_buffer[.uint8](length)

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        # Pre-existing root node added before the scope. The scope's
        # `dependencies=` names it, so the scope runs after it.
        var pre_scope = builder.add_memset(buf_a, UInt8(0x01), dependencies=[])

        # Two producer nodes added inside the scope. The work is a named
        # capturing def, passed as a callback to the scope method. A node
        # from the enclosing scope (`pre_scope`) cannot be named inside the
        # callback because the callback is generic over the scope origin, so
        # it is injected as a scope-level predecessor via `dependencies=`
        # instead; every node the callback adds runs after it.
        def add_producers(mut b: DeviceGraphBuilder) raises {imm}:
            _ = b.add_memset(buf_a, UInt8(0xAA), dependencies=[])
            _ = b.add_memset(buf_b, UInt8(0xBB), dependencies=[])

        var producers_join = builder.region(
            add_producers, dependencies=[pre_scope]
        )

        # Use the join node as the sole dependency of a memset on buf_c. The
        # final state of buf_c being 0xCC confirms that consumer ran; the
        # transitive order through the empty node enforces that the producers
        # have completed by then.
        _ = builder.add_memset(
            buf_c, UInt8(0xCC), dependencies=[producers_join]
        )

    var graph = DeviceGraph.create(ctx, build)
    graph.replay()
    ctx.synchronize()

    with buf_a.map_to_host() as host_a:
        for i in range(length):
            assert_equal(host_a[i], UInt8(0xAA))
    with buf_b.map_to_host() as host_b:
        for i in range(length):
            assert_equal(host_b[i], UInt8(0xBB))
    with buf_c.map_to_host() as host_c:
        for i in range(length):
            assert_equal(host_c[i], UInt8(0xCC))


def test_region_empty(ctx: DeviceContext) raises:
    print(
        "Test region still returns a usable node"
        " when the scope adds no nodes (empty node becomes a graph root)."
    )
    comptime length = 64
    var buf = ctx.enqueue_create_buffer[.uint8](length)

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        def add_nothing(mut b: DeviceGraphBuilder) raises {imm}:
            return

        var join = builder.region(add_nothing)

        # Hang a memset off the (rootless) empty node and verify the graph
        # still instantiates and replays successfully.
        _ = builder.add_memset(buf, UInt8(0xEE), dependencies=[join])

    var graph = DeviceGraph.create(ctx, build)
    graph.replay()
    ctx.synchronize()

    with buf.map_to_host() as host:
        for i in range(length):
            assert_equal(host[i], UInt8(0xEE))


def test_region_with_dependencies(ctx: DeviceContext) raises:
    print(
        "Test region(dependencies=...) injects predecessors so"
        " a consumer scope runs after a producer scope (RAW on one buffer)."
    )
    comptime length = 1024
    comptime block_dim = 256
    comptime grid_dim = ceildiv(length, block_dim)

    var buf = ctx.enqueue_create_buffer[.float32](length)

    var fill = ctx.compile_function[Kernels.fill_constant]()
    var incr = ctx.compile_function[Kernels.add_in_place]()

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        # Producer scope: fill `buf` with 5 (single kernel node, a graph root).
        def producer(mut b: DeviceGraphBuilder) raises {imm}:
            _ = b.add_function(
                fill,
                buf,
                Int32(5),
                Int32(length),
                grid_dim=grid_dim,
                block_dim=block_dim,
                dependencies=[],
            )

        var join_a = builder.region(producer)

        # Consumer scope: increment `buf` by 10. Passing dependencies=[join_a]
        # makes join_a a predecessor of the scope's first node, so it runs
        # strictly after the producer. Final value must be 15, not 10.
        def consumer(mut b: DeviceGraphBuilder) raises {imm}:
            _ = b.add_function(
                incr,
                buf,
                Int32(10),
                Int32(length),
                grid_dim=grid_dim,
                block_dim=block_dim,
                dependencies=[],
            )

        _ = builder.region(consumer, dependencies=[join_a])

    var graph = DeviceGraph.create(ctx, build)
    graph.replay()
    ctx.synchronize()

    with buf.map_to_host() as host:
        for i in range(length):
            assert_equal(host[i], Float32(15))


def test_region_passthrough_dependencies(
    ctx: DeviceContext,
) raises:
    print(
        "Test region returns a node that still gates on"
        " `dependencies` when the scope adds no nodes (zero-node fallback)."
    )
    comptime length = 1024
    comptime block_dim = 256
    comptime grid_dim = ceildiv(length, block_dim)

    var buf = ctx.enqueue_create_buffer[.float32](length)

    var fill = ctx.compile_function[Kernels.fill_constant]()
    var incr = ctx.compile_function[Kernels.add_in_place]()

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        # Producer scope: fill `buf` with 5.
        def producer(mut b: DeviceGraphBuilder) raises {imm}:
            _ = b.add_function(
                fill,
                buf,
                Int32(5),
                Int32(length),
                grid_dim=grid_dim,
                block_dim=block_dim,
                dependencies=[],
            )

        var join_a = builder.region(producer)

        # Empty scope depending on join_a: adds no nodes, so its returned
        # handle falls back to an empty node depending on join_a directly.
        def add_nothing(mut b: DeviceGraphBuilder) raises {imm}:
            return

        var passthrough = builder.region(add_nothing, dependencies=[join_a])

        # Increment by 10, gated on the passthrough join. Final value must be
        # 15, proving the empty scope still ordered the incr after the
        # producer.
        _ = builder.add_function(
            incr,
            buf,
            Int32(10),
            Int32(length),
            grid_dim=grid_dim,
            block_dim=block_dim,
            dependencies=[passthrough],
        )

    var graph = DeviceGraph.create(ctx, build)
    graph.replay()
    ctx.synchronize()

    with buf.map_to_host() as host:
        for i in range(length):
            assert_equal(host[i], Float32(15))


def test_region_node_structure(ctx: DeviceContext) raises:
    print(
        "Test region chains its nodes, returns the last one, and adds no"
        " join node."
    )
    comptime length = 64
    var buf = ctx.enqueue_create_buffer[.uint8](length)

    # Node ids are dense and monotonic, so the builder's last-node id doubles
    # as a node count. That is what pins the contract here: the *behaviour* of
    # a chain edge is not observable through replay (an unordered pair of root
    # nodes still runs in insertion order in practice), but the node count and
    # the returned handle are exact.
    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        def two_nodes(mut b: DeviceGraphBuilder) raises {imm}:
            _ = b.add_memset(buf, UInt8(0x11))
            _ = b.add_memset(buf, UInt8(0x22))

        var chained = builder.region(two_nodes)

        # Two nodes in, two nodes added: chaining makes the second the scope's
        # sole sink, so there is no third (join) node, and the handle returned
        # is that sink.
        assert_equal(Int(builder._last_node_id().value()), 1)
        assert_equal(Int(chained.id), 1)

        # An outer scope spanning a nested scope adds one node per `add_*` and
        # nothing more, which is what proves the floor bookkeeping nests.
        def outer(mut b: DeviceGraphBuilder) raises {imm}:
            _ = b.add_memset(buf, UInt8(0x33))

            def inner(mut inner_b: DeviceGraphBuilder) raises {imm}:
                _ = inner_b.add_memset(buf, UInt8(0x44))

            var inner_sink = b.region(inner)
            assert_equal(Int(inner_sink.id), 3)

            _ = b.add_memset(buf, UInt8(0x55))

        var outer_sink = builder.region(outer)
        assert_equal(Int(builder._last_node_id().value()), 4)
        assert_equal(Int(outer_sink.id), 4)

        # A scope that adds nothing still materializes its fallback empty node.
        def nothing(mut b: DeviceGraphBuilder) raises {imm}:
            return

        var empty_sink = builder.region(nothing)
        assert_equal(Int(builder._last_node_id().value()), 5)
        assert_equal(Int(empty_sink.id), 5)

    var graph = DeviceGraph.create(ctx, build)
    graph.replay()
    ctx.synchronize()

    with buf.map_to_host() as host:
        for i in range(length):
            assert_equal(host[i], UInt8(0x55))


def test_region_chain_edges(ctx: DeviceContext) raises:
    print(
        "Test the predecessor set a region hands each node: incoming"
        " predecessors for the first, the previous node for the rest."
    )
    comptime length = 64
    var buf = ctx.enqueue_create_buffer[.uint8](length)

    # `_merge_implicit` is the whole chain rule, and it is a pure function of
    # the builder's state, so probing it asserts the edges directly. Replay
    # cannot: two unordered root nodes still run in insertion order in
    # practice, so a missing edge is invisible end to end.
    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        var pre = builder.add_memset(buf, UInt8(0x01))

        # Outside a region there is no implicit predecessor at all.
        assert_equal(
            len(builder._merge_implicit(List[type_of(builder).Node]())), 0
        )

        def probe(mut b: DeviceGraphBuilder) raises {imm}:
            # First node of the scope: gated on the scope's incoming
            # predecessors.
            var first = b._merge_implicit(List[type_of(b).Node]())
            assert_equal(len(first), 1)
            assert_equal(Int(first[0].id), Int(pre.id))

            var n0 = b.add_memset(buf, UInt8(0x02))

            # Every later node chains after the one before it instead, so the
            # incoming predecessors are named once rather than on every node.
            var second = b._merge_implicit(List[type_of(b).Node]())
            assert_equal(len(second), 1)
            assert_equal(Int(second[0].id), Int(n0.id))

            var n1 = b.add_memset(buf, UInt8(0x03))

            # An explicit dependency is unioned with the chain edge, not
            # replaced by it.
            var both = b._merge_implicit([n0])
            assert_equal(len(both), 2)
            assert_equal(Int(both[0].id), Int(n0.id))
            assert_equal(Int(both[1].id), Int(n1.id))

            # Naming the chain predecessor explicitly does not repeat it: the
            # graph APIs reject a duplicated predecessor.
            var same = b._merge_implicit([n1])
            assert_equal(len(same), 1)
            assert_equal(Int(same[0].id), Int(n1.id))

        _ = builder.region(probe, dependencies=[pre])

        # The scope is popped again on the way out.
        assert_equal(
            len(builder._merge_implicit(List[type_of(builder).Node]())), 0
        )

    var graph = DeviceGraph.create(ctx, build)
    graph.replay()
    ctx.synchronize()

    with buf.map_to_host() as host:
        for i in range(length):
            assert_equal(host[i], UInt8(0x03))


def test_region_serializes_nodes(ctx: DeviceContext) raises:
    print(
        "Test a region whose nodes have a RAW dependency on one buffer"
        " replays to the chained result."
    )
    comptime length = 1024
    comptime block_dim = 256
    comptime grid_dim = ceildiv(length, block_dim)

    var buf = ctx.enqueue_create_buffer[.float32](length)

    var fill = ctx.compile_function[Kernels.fill_constant]()
    var incr = ctx.compile_function[Kernels.add_in_place]()

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        # Neither node names a dependency and both write `buf`, so only the
        # region's chaining orders them. See `test_region_node_structure` for
        # the assertion that actually pins the edge.
        def stage(mut b: DeviceGraphBuilder) raises {imm}:
            _ = b.add_function(
                fill,
                buf,
                Int32(5),
                Int32(length),
                grid_dim=grid_dim,
                block_dim=block_dim,
            )
            _ = b.add_function(
                incr,
                buf,
                Int32(10),
                Int32(length),
                grid_dim=grid_dim,
                block_dim=block_dim,
            )

        _ = builder.region(stage)

    var graph = DeviceGraph.create(ctx, build)
    graph.replay()
    ctx.synchronize()

    with buf.map_to_host() as host:
        for i in range(length):
            assert_equal(host[i], Float32(15))


def test_region_chains_after_recording_context(ctx: DeviceContext) raises:
    print(
        "Test a region's chain picks up nodes recorded through"
        " recording_context(), which the C++ builder adds directly."
    )
    comptime length = 1024
    comptime block_dim = 256
    comptime grid_dim = ceildiv(length, block_dim)

    var buf = ctx.enqueue_create_buffer[.float32](length)

    var fill = ctx.compile_function[Kernels.fill_constant]()
    var incr = ctx.compile_function[Kernels.add_in_place]()

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        # The fill is recorded through the recording context, so it never
        # passes through `_merge_implicit`. The following `add_function` must
        # still chain after it, which it does because the chain predecessor is
        # read from the builder's node ids rather than tracked on the Mojo
        # side: the seed plus the recorded launch are node 0 and 1, so the
        # increment lands on node 2 depending on node 1.
        def stage(mut b: DeviceGraphBuilder) raises {imm}:
            with b.recording_context() as rec:
                rec.enqueue_function(
                    fill,
                    buf,
                    Int32(5),
                    Int32(length),
                    grid_dim=grid_dim,
                    block_dim=block_dim,
                )

            _ = b.add_function(
                incr,
                buf,
                Int32(10),
                Int32(length),
                grid_dim=grid_dim,
                block_dim=block_dim,
            )

        var sink = builder.region(stage)
        assert_equal(Int(sink.id), 2)

    var graph = DeviceGraph.create(ctx, build)
    graph.replay()
    ctx.synchronize()

    with buf.map_to_host() as host:
        for i in range(length):
            assert_equal(host[i], Float32(15))


def test_as_context_kernel_chain(ctx: DeviceContext) raises:
    print(
        "Test recording a two-kernel chain through DeviceGraphBuilder"
        ".recording_context() using the ordinary enqueue_function API."
    )
    comptime length = 1024
    comptime block_dim = 256
    comptime grid_dim = ceildiv(length, block_dim)

    var buf = ctx.enqueue_create_buffer[.float32](length)

    var fill = ctx.compile_function[Kernels.fill_constant]()
    var incr = ctx.compile_function[Kernels.add_in_place]()

    # Record fill(=5) then incr(+10) on the same buffer through the recording
    # context. The facade chains each enqueue after the previous one, so a
    # correct final value of 15 proves the launches recorded AND stayed ordered
    # (incr must observe fill's write, which on CUDA/HIP relies on the recorded
    # dependency edge rather than any implicit stream FIFO).
    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        with builder.recording_context() as rec:
            rec.enqueue_function(
                fill,
                buf,
                Int32(5),
                Int32(length),
                grid_dim=grid_dim,
                block_dim=block_dim,
            )
            rec.enqueue_function(
                incr,
                buf,
                Int32(10),
                Int32(length),
                grid_dim=grid_dim,
                block_dim=block_dim,
            )

    var graph = DeviceGraph.create(ctx, build)

    graph.replay()
    ctx.synchronize()
    with buf.map_to_host() as host:
        for i in range(length):
            assert_equal(host[i], Float32(15))
            # Zero out so the second replay must recompute from scratch.
            host[i] = 0.0

    graph.replay()
    ctx.synchronize()
    with buf.map_to_host() as host:
        for i in range(length):
            assert_equal(host[i], Float32(15))


def test_create_buffer(ctx: DeviceContext) raises:
    print("Test allocating graph-owned device and host buffers.")
    comptime length = 1024

    var host_dst = ctx.enqueue_create_host_buffer[.uint8](length)
    for i in range(length):
        host_dst[i] = 0

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        # A device allocation is graph-scoped, so the graph copies it out to
        # storage the test owns rather than handing back the buffer itself.
        var dev_buf = builder.create_buffer[.uint8](length, is_host=False)
        var memset = builder.add_memset(dev_buf, UInt8(0x5A))
        _ = builder.add_copy(host_dst, dev_buf, dependencies=[memset])

        # A host allocation comes from the pool's other memory manager: writing
        # through it here would fault if it had been served as device memory.
        var pool_host = builder.create_buffer[.uint8](length, is_host=True)
        var ptr = pool_host.unsafe_ptr()

        # Slightly questionable way to check that this is a host allocation.
        for i in range(length):
            ptr[unsafe_offset=i] = UInt8(i % 251)

        for i in range(length):
            assert_equal(ptr[unsafe_offset=i], UInt8(i % 251))

    var graph = DeviceGraph.create(ctx, build)
    graph.replay()
    ctx.synchronize()

    for i in range(length):
        assert_equal(host_dst[i], UInt8(0x5A))


@fieldwise_init
struct _TaggedInput(DeviceGraphInput, ImplicitlyCopyable, Movable):
    """A graph input whose cache key is just its tag.

    Deliberately writes a bare, undelimited integer: framing is
    `make_key`'s job, so this is the adversarial case for it.
    """

    var tag: Int

    def write_graph_key(self, mut writer: Some[Writer]):
        writer.write(self.tag)

    def allocate_stable(self, mut builder: DeviceGraphBuilder) raises -> Self:
        # Backed by no memory, so there is nothing to give a stable location.
        return Self(copy=self)


@fieldwise_init
struct _BufferInput(DeviceGraphInput, ImplicitlyCopyable, Movable):
    """A graph input backed by a device buffer, for exercising `add_input`."""

    var buf: DeviceBuffer[.uint8]

    def write_graph_key(self, mut writer: Some[Writer]):
        writer.write(t"BufferInput({len(self.buf)})")

    def allocate_stable(self, mut builder: DeviceGraphBuilder) raises -> Self:
        return Self(
            builder.create_input_buffer[.uint8](len(self.buf), is_host=False)
        )


@fieldwise_init
struct _InPlaceBufferInput(DeviceGraphInput, ImplicitlyCopyable, Movable):
    """A mutable graph input recorded at the caller's live address.

    Models a buffer the graph writes in place (a KV cache): `allocate_stable`
    registers an in-place marker and returns an alias of the caller's buffer,
    and the buffer's address joins the cache key so a moved buffer forces a
    rebuild instead of replaying stale addresses.
    """

    var buf: DeviceBuffer[.uint8]

    def write_graph_key(self, mut writer: Some[Writer]):
        writer.write(
            t"InPlaceBufferInput({len(self.buf)}, {Int(self.buf.unsafe_ptr())})"
        )

    def allocate_stable(self, mut builder: DeviceGraphBuilder) raises -> Self:
        builder.register_in_place_input()
        return Self(copy=self)


def test_add_input(ctx: DeviceContext) raises:
    print("Test add_input gives the graph a stable location to record against.")
    comptime length = 64

    var host_dst = ctx.enqueue_create_host_buffer[.uint8](length)
    for i in range(length):
        host_dst[i] = 0

    var caller_buf = ctx.enqueue_create_buffer[.uint8](length)

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        assert_equal(builder.num_inputs(), 0)
        var stable = builder.add_input(_BufferInput(caller_buf.copy()))
        assert_equal(builder.num_inputs(), 1)

        # The stable location is the graph's own allocation, not the caller's.
        assert_true(stable.buf.unsafe_ptr() != caller_buf.unsafe_ptr())

        # Record against the stable location and copy it out, so the readback
        # observes what the graph actually wrote.
        var memset = builder.add_memset(stable.buf, UInt8(0x7E))
        _ = builder.add_copy(host_dst, stable.buf, dependencies=[memset])

    var graph = DeviceGraph.create(ctx, build)
    graph.replay()
    ctx.synchronize()

    for i in range(length):
        assert_equal(host_dst[i], UInt8(0x7E))


def test_add_in_place_input(ctx: DeviceContext) raises:
    print("Test in-place inputs alias the caller's buffer with no stable twin.")
    comptime length = 64

    var host_dst = ctx.enqueue_create_host_buffer[.uint8](length)
    for i in range(length):
        host_dst[i] = 0

    var caller_buf = ctx.enqueue_create_buffer[.uint8](length)
    with caller_buf.map_to_host() as host_view:
        for i in range(length):
            host_view[i] = 0x7E

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        assert_equal(builder.num_inputs(), 0)
        var in_place = builder.add_input(_InPlaceBufferInput(caller_buf.copy()))
        # The marker occupies an input position like a stable buffer would,
        # keeping the positional pairing with execute operands intact.
        assert_equal(builder.num_inputs(), 1)

        # The returned handle aliases the caller's buffer: the graph records
        # the live address rather than a graph-private twin.
        assert_true(in_place.buf.unsafe_ptr() == caller_buf.unsafe_ptr())

        _ = builder.add_copy(host_dst, in_place.buf)

    var graph = DeviceGraph.create(ctx, build)
    graph.replay()
    ctx.synchronize()

    for i in range(length):
        assert_equal(host_dst[i], UInt8(0x7E))

    # Mutate the caller's buffer and replay: the graph reads through the live
    # address, so the readback must see the new contents (a private snapshot
    # would keep returning the old pattern).
    with caller_buf.map_to_host() as host_view:
        for i in range(length):
            host_view[i] = 0x3C

    graph.replay()
    ctx.synchronize()

    for i in range(length):
        assert_equal(host_dst[i], UInt8(0x3C))


@fieldwise_init
struct _HostValueInput(DeviceGraphInput, ImplicitlyCopyable, Movable):
    """A host-resident graph input whose contents are baked into the graph.

    Models dispatch metadata (e.g. attention cache lengths): the build closure
    reads the value on the host at record time, freezing it into the recorded
    nodes, so the full contents join the cache key -- changed bytes must miss
    and re-record rather than replay stale dispatch. `allocate_stable` copies
    the caller's bytes into a graph-owned host allocation before returning,
    since recorded ops read the stable location while the build runs.
    """

    var data: Pointer[Scalar[DType.uint8], MutUntrackedOrigin]
    var size: Int

    def write_graph_key(self, mut writer: Some[Writer]):
        writer.write(t"HostValueInput({self.size}, ")
        for i in range(self.size):
            writer.write(Int(self.data[i]), ",")
        writer.write(")")

    def allocate_stable(self, mut builder: DeviceGraphBuilder) raises -> Self:
        var stable = builder.create_input_buffer[DType.uint8](
            self.size, is_host=True
        )
        # The graph retains the allocation (create_input_buffer registers it),
        # so returning just the pointer keeps the address reserved.
        var dst = stable.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        for i in range(self.size):
            dst[i] = self.data[i]
        return Self(dst, self.size)


def test_add_host_input(ctx: DeviceContext) raises:
    print("Test host inputs get a stable host twin whose value bakes in.")
    comptime length = 64

    var host_dst = ctx.enqueue_create_host_buffer[DType.uint8](length)
    for i in range(length):
        host_dst[i] = 0

    var device_buf = ctx.enqueue_create_buffer[DType.uint8](length)

    # The host "dispatch value" the recording bakes in.
    var caller_buf = ctx.enqueue_create_host_buffer[DType.uint8](1)
    caller_buf[0] = 0x2B
    var caller_input = _HostValueInput(caller_buf.unsafe_ptr(), 1)

    var cache = DeviceGraphCache()
    var build_count = 0
    var count = Pointer(to=build_count)

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        count[] += 1
        var stable = builder.add_input(caller_input)
        assert_equal(builder.num_inputs(), 1)

        # The stable twin is graph-owned host memory, not the caller's.
        assert_true(stable.data != caller_input.data)

        # Enqueue-time host read: the value steers what gets recorded, the
        # way dispatch metadata steers kernel selection and launch geometry.
        var pattern = stable.data[0]
        var memset = builder.add_memset(device_buf, pattern)
        _ = builder.add_copy(host_dst, device_buf, dependencies=[memset])

    var graph = DeviceGraph.create(
        ctx, build, caller_input, cache=Pointer(to=cache)
    )
    assert_equal(build_count, 1)
    graph.replay()
    ctx.synchronize()
    for i in range(length):
        assert_equal(host_dst[i], UInt8(0x2B))

    # Same bytes: cache hit, no rebuild.
    var again = DeviceGraph.create(
        ctx, build, caller_input, cache=Pointer(to=cache)
    )
    assert_equal(build_count, 1)
    _ = again^

    # Changed bytes: miss and re-record with the new baked value.
    caller_buf[0] = 0x71
    var rerecorded = DeviceGraph.create(
        ctx, build, caller_input, cache=Pointer(to=cache)
    )
    assert_equal(build_count, 2)
    rerecorded.replay()
    ctx.synchronize()
    for i in range(length):
        assert_equal(host_dst[i], UInt8(0x71))


def test_host_input_key_bakes_contents(ctx: DeviceContext) raises:
    print("Test host-input cache keys separate by contents.")

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        return

    var a = ctx.enqueue_create_host_buffer[DType.uint8](2)
    a[0] = 1
    a[1] = 2
    var b = ctx.enqueue_create_host_buffer[DType.uint8](2)
    b[0] = 1
    b[1] = 3

    var key_a = DeviceGraphCache.make_key(
        build, _HostValueInput(a.unsafe_ptr(), 2)
    )
    var key_b = DeviceGraphCache.make_key(
        build, _HostValueInput(b.unsafe_ptr(), 2)
    )
    assert_not_equal(key_a, key_b)

    # Same contents at a different address must agree: the key is the value.
    var a2 = ctx.enqueue_create_host_buffer[DType.uint8](2)
    a2[0] = 1
    a2[1] = 2
    assert_equal(
        key_a,
        DeviceGraphCache.make_key(build, _HostValueInput(a2.unsafe_ptr(), 2)),
    )


def test_cache_key_separates_inputs() raises:
    print("Test cache keys keep adjacent input contributions apart.")

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        return

    # Undelimited, these two would both spell "...123".
    var a = DeviceGraphCache.make_key(build, _TaggedInput(1), _TaggedInput(23))
    var b = DeviceGraphCache.make_key(build, _TaggedInput(12), _TaggedInput(3))
    assert_not_equal(a, b)

    # Equal inputs still agree, or nothing would ever hit.
    assert_equal(
        a, DeviceGraphCache.make_key(build, _TaggedInput(1), _TaggedInput(23))
    )

    # Arity is part of the key too: one input must not look like two.
    assert_not_equal(a, DeviceGraphCache.make_key(build, _TaggedInput(1)))

    # A different work function keys a different graph even with equal inputs.
    def other_build(mut builder: DeviceGraphBuilder) raises {imm}:
        return

    assert_not_equal(
        a,
        DeviceGraphCache.make_key(
            other_build, _TaggedInput(1), _TaggedInput(23)
        ),
    )


def test_cache_reuses_graph(ctx: DeviceContext) raises:
    print("Test a cache hit returns the prior graph without rebuilding it.")
    comptime length = 64

    var buf = ctx.enqueue_create_buffer[.uint8](length)
    var cache = DeviceGraphCache()

    # Counted through a pointer so the closure can stay `imm`-capturing.
    var build_count = 0
    var count = Pointer(to=build_count)

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        count[] += 1
        _ = builder.add_memset(buf, UInt8(0x5A))

    var first = DeviceGraph.create(ctx, build, cache=Pointer(to=cache))
    assert_equal(build_count, 1)

    # Same closure and same (empty) input list, so the key matches and the build
    # body must not run a second time.
    var second = DeviceGraph.create(ctx, build, cache=Pointer(to=cache))
    assert_equal(build_count, 1)

    # The graph handed back on a hit is a fully usable graph, not a husk.
    second.replay()
    ctx.synchronize()

    with buf.map_to_host() as host:
        for i in range(length):
            assert_equal(host[i], UInt8(0x5A))


def test_cache_distinguishes_inputs(ctx: DeviceContext) raises:
    print("Test inputs writing different cache keys do not share a graph.")
    comptime length = 64

    var buf = ctx.enqueue_create_buffer[.uint8](length)
    var cache = DeviceGraphCache()

    var build_count = 0
    var count = Pointer(to=build_count)

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        count[] += 1
        _ = builder.add_memset(buf, UInt8(0x11))

    _ = DeviceGraph.create(ctx, build, _TaggedInput(1), cache=Pointer(to=cache))
    assert_equal(build_count, 1)

    # A different key must miss and rebuild.
    _ = DeviceGraph.create(ctx, build, _TaggedInput(2), cache=Pointer(to=cache))
    assert_equal(build_count, 2)

    # Returning to the first key must hit the entry stored by the first call.
    _ = DeviceGraph.create(ctx, build, _TaggedInput(1), cache=Pointer(to=cache))
    assert_equal(build_count, 2)


def test_cache_without_cache_always_builds(ctx: DeviceContext) raises:
    print("Test omitting the cache rebuilds the graph on every call.")
    comptime length = 64

    var buf = ctx.enqueue_create_buffer[.uint8](length)

    var build_count = 0
    var count = Pointer(to=build_count)

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        count[] += 1
        _ = builder.add_memset(buf, UInt8(0x22))

    _ = DeviceGraph.create(ctx, build)
    _ = DeviceGraph.create(ctx, build)
    assert_equal(build_count, 2)


def test_cache_lookup_and_add(ctx: DeviceContext) raises:
    print("Test DeviceGraphCache lookup/add semantics directly.")
    comptime length = 64

    var buf = ctx.enqueue_create_buffer[.uint8](length)
    var cache = DeviceGraphCache()

    assert_false(Bool(cache.lookup("absent")))

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        _ = builder.add_memset(buf, UInt8(0x33))

    _ = cache.cache("key", DeviceGraph.create(ctx, build))

    var found = cache.lookup("key")
    assert_true(Bool(found))

    # The looked-up graph is an independent handle on the same graph, so it
    # replays even though the cache still holds its own reference.
    found.take().replay()
    ctx.synchronize()

    with buf.map_to_host() as host:
        for i in range(length):
            assert_equal(host[i], UInt8(0x33))

    # A second add under the same key replaces the entry rather than failing.
    _ = cache.cache("key", DeviceGraph.create(ctx, build))
    assert_true(Bool(cache.lookup("key")))
    assert_false(Bool(cache.lookup("other")))


def test_cache_shares_memory_pool(ctx: DeviceContext) raises:
    print("Test cached graph creations share one memory pool per context.")
    comptime length = 64

    var cache = DeviceGraphCache()

    # The same context maps to the same pool handle on every lookup.
    var p1 = cache.get_or_create_pool(ctx)
    var p2 = cache.get_or_create_pool(ctx)
    assert_equal(p1, p2)

    # Each build allocates transient scratch through the builder -- that is,
    # from the cache's shared pool -- memsets it, and copies it out to an
    # externally owned buffer. The scratch handle dies inside the closure, so
    # its block returns to the shared pool for the next cached build to reuse.
    var buf_a = ctx.enqueue_create_buffer[.uint8](length)
    var buf_b = ctx.enqueue_create_buffer[.uint8](length)

    # Capture the address of the scratch allocations inside each graph build
    # region. These pointers are not valid to access, outside the region and
    # we only use it to test the allocator handed back the same address both
    # times.
    var scratch_addr_a = 0
    var scratch_addr_b = 0

    def build_a(
        mut builder: DeviceGraphBuilder,
    ) raises {imm, mut scratch_addr_a}:
        var scratch = builder.create_buffer[.uint8](length, is_host=False)
        scratch_addr_a = Int(scratch.unsafe_ptr())
        var filled = builder.add_memset(scratch, UInt8(0x11))
        _ = builder.add_copy(buf_a, scratch, dependencies=[filled])

    def build_b(
        mut builder: DeviceGraphBuilder,
    ) raises {imm, mut scratch_addr_b}:
        var scratch = builder.create_buffer[.uint8](length, is_host=False)
        scratch_addr_b = Int(scratch.unsafe_ptr())
        var filled = builder.add_memset(scratch, UInt8(0x22))
        _ = builder.add_copy(buf_b, scratch, dependencies=[filled])

    var graph_a = DeviceGraph.create(ctx, build_a, cache=Pointer(to=cache))
    var graph_b = DeviceGraph.create(ctx, build_b, cache=Pointer(to=cache))

    # Both graphs' scratch came from the shared pool: A's block was freed when
    # its closure ended, so B's same-size request was handed the same address.
    assert_equal(scratch_addr_a, scratch_addr_b)

    # A's replay writes the scratch block B also uses; replaying serially in
    # recording order is what keeps both outputs correct.
    graph_a.replay()
    graph_b.replay()
    ctx.synchronize()

    with buf_a.map_to_host() as host:
        for i in range(length):
            assert_equal(host[i], UInt8(0x11))

    with buf_b.map_to_host() as host:
        for i in range(length):
            assert_equal(host[i], UInt8(0x22))


comptime TestFunc = def(DeviceContext) thin raises -> None
comptime TestFuncNoArgs = def() thin raises -> None


def main() raises:
    comptime funcs = __functions_in_module()

    with DeviceContext() as ctx:
        comptime for i in range(len(funcs)):
            comptime test_func = funcs[i]

            comptime if get_function_name[test_func]().startswith("test_"):
                comptime if type_of(test_func) == TestFunc:
                    comptime func = rebind[TestFunc](test_func)
                    func(ctx)
                elif type_of(test_func) == TestFuncNoArgs:
                    comptime func = rebind[TestFuncNoArgs](test_func)
                    func()
                else:
                    comptime assert False, String(
                        "test function '",
                        get_function_name[test_func](),
                        "' has a nonconforming signature",
                    )
