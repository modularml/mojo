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
"""Real GPU launch test for `Span`'s `address_space` parameter.

A kernel writes GPU shared memory through `Span` indexing and reads it back
through an address-space-generic helper, checking that slicing and `as_imm`
preserve the address space. Further kernels `fill` real shared and thread-local
memory and read the results back.
"""

from max.gpu import thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.memory import unsafe_stack_allocation
from std.sys import has_apple_gpu_accelerator
from std.testing import assert_equal

comptime TILE_SIZE = 32


@fieldwise_init
struct Vec2(ImplicitlyCopyable, TrivialRegisterPassable):
    """Register passable, so it may cross an address-space boundary."""

    var x: Float32
    var y: Float32


# Address-space-generic: the same body sums a shared-memory tile and a
# generic-address-space span.
def _tile_sum(tile: Span[Float32, _, address_space=_]) -> Float32:
    var acc = Float32(0)
    for i in range(len(tile)):
        acc += tile[i]
    return acc


def _kernel(out_ptr: Pointer[Float32, MutAnyOrigin]):
    var smem = unsafe_stack_allocation[
        TILE_SIZE, Float32, address_space=.SHARED
    ]()
    var tile = Span[
        mut=True,
        Float32,
        MutUntrackedOrigin,
        address_space=.SHARED,
    ](unsafe_ptr=smem, length=TILE_SIZE)

    # Write into shared memory through `Span` indexing.
    tile[thread_idx.x] = Float32(thread_idx.x + 1)
    barrier()

    if thread_idx.x == 0:
        out_ptr[unsafe_offset=0] = _tile_sum(tile)
        # Slicing preserves the address space.
        out_ptr[unsafe_offset=1] = _tile_sum(tile[0 : TILE_SIZE // 2])
        # `as_imm` preserves the address space.
        out_ptr[unsafe_offset=2] = tile.as_imm()[5]


def _fill_kernel(out_ptr: Pointer[Float32, MutAnyOrigin]):
    var f_smem = unsafe_stack_allocation[
        TILE_SIZE, Float32, address_space=.SHARED
    ]()
    var f_tile = Span[
        mut=True,
        Float32,
        MutUntrackedOrigin,
        address_space=.SHARED,
    ](unsafe_ptr=f_smem, length=TILE_SIZE)

    # Guards multi-instantiation: three element types force three copies of the
    # address-space-generic `fill` into one module, the shape a kernel with
    # mixed-dtype tiles produces.
    var b_smem = unsafe_stack_allocation[
        TILE_SIZE, UInt8, address_space=.SHARED
    ]()
    var b_tile = Span[
        mut=True, UInt8, MutUntrackedOrigin, address_space=.SHARED
    ](unsafe_ptr=b_smem, length=TILE_SIZE)

    var v_smem = unsafe_stack_allocation[
        TILE_SIZE, Vec2, address_space=.SHARED
    ]()
    var v_tile = Span[
        mut=True, Vec2, MutUntrackedOrigin, address_space=.SHARED
    ](unsafe_ptr=v_smem, length=TILE_SIZE)

    if thread_idx.x == 0:
        f_tile.fill(2.5)
        b_tile.fill(7)
        v_tile.fill(Vec2(0.5, 1.5))
    barrier()

    if thread_idx.x == 0:
        out_ptr[unsafe_offset=0] = _tile_sum(f_tile)
        out_ptr[unsafe_offset=1] = Float32(
            Int(b_tile[0]) + Int(b_tile[TILE_SIZE - 1])
        )
        out_ptr[unsafe_offset=2] = v_tile[TILE_SIZE - 1].x + v_tile[0].y


def _local_fill_kernel(out_ptr: Pointer[Float32, MutAnyOrigin]):
    # Local memory is thread-private, so every thread fills its own tile with a
    # distinct value and two of them are read back.
    var l_mem = unsafe_stack_allocation[
        TILE_SIZE, Float32, address_space=.LOCAL
    ]()
    var l_tile = Span[
        mut=True,
        Float32,
        MutUntrackedOrigin,
        address_space=.LOCAL,
    ](unsafe_ptr=l_mem, length=TILE_SIZE)
    l_tile.fill(Float32(thread_idx.x) + 0.5)

    if thread_idx.x == 0:
        out_ptr[unsafe_offset=0] = _tile_sum(l_tile)
    elif thread_idx.x == TILE_SIZE - 1:
        out_ptr[unsafe_offset=1] = _tile_sum(l_tile)


def main() raises:
    with DeviceContext() as ctx:
        var out_device = ctx.enqueue_create_buffer[.float32](3)
        var compiled = ctx.compile_function[_kernel]()
        ctx.enqueue_function(
            compiled, out_device, grid_dim=(1,), block_dim=(TILE_SIZE,)
        )
        ctx.synchronize()

        with out_device.map_to_host() as out_host:
            # 1 + 2 + ... + 32
            assert_equal(out_host[0], 528.0)
            # 1 + 2 + ... + 16
            assert_equal(out_host[1], 136.0)
            assert_equal(out_host[2], 6.0)

        var fill_device = ctx.enqueue_create_buffer[.float32](3)
        var fill_compiled = ctx.compile_function[_fill_kernel]()
        ctx.enqueue_function(
            fill_compiled, fill_device, grid_dim=(1,), block_dim=(TILE_SIZE,)
        )
        ctx.synchronize()

        with fill_device.map_to_host() as fill_host:
            # 2.5 * 32
            assert_equal(fill_host[0], 80.0)
            # 7 at both ends
            assert_equal(fill_host[1], 14.0)
            # 0.5 + 1.5
            assert_equal(fill_host[2], 2.0)

        comptime if not has_apple_gpu_accelerator():
            var local_device = ctx.enqueue_create_buffer[.float32](2)
            var local_compiled = ctx.compile_function[_local_fill_kernel]()
            ctx.enqueue_function(
                local_compiled,
                local_device,
                grid_dim=(1,),
                block_dim=(TILE_SIZE,),
            )
            ctx.synchronize()

            with local_device.map_to_host() as local_host:
                assert_equal(local_host[0], 16.0)
                assert_equal(local_host[1], 1008.0)
