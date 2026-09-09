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


from max.gpu.host import ConstantMemoryMapping, DeviceContext
from max.gpu.host.compile import _compile_code
from max.gpu import thread_idx
from std.memory import unsafe_stack_allocation
from std.testing import assert_equal, assert_true


def test_constant_memory_compile(ctx: DeviceContext) raises:
    def _alloc[
        n: Int
    ]() -> MutPointer[Float32, MutUntrackedOrigin, address_space=.CONSTANT]:
        return unsafe_stack_allocation[n, Float32, address_space=.CONSTANT]()

    assert_true(".const .align 4 .b8 " in _compile_code[_alloc[20]]())
    assert_true(
        "internal addrspace(4) global [20 x float]"
        in _compile_code[_alloc[20], emission_kind="llvm"]()
    )


def test_constant_mem(ctx: DeviceContext) raises:
    print("== test_constant_mem")

    def _fill_impl[
        n: Int
    ]() -> MutPointer[Float32, MutUntrackedOrigin, address_space=.CONSTANT]:
        var ptr = unsafe_stack_allocation[n, Float32, address_space=.CONSTANT]()

        comptime for i in range(n):
            ptr[i] = Float32(i)
        return ptr

    def static_constant_kernel[n: Int](data: MutPointer[Float32, MutAnyOrigin]):
        comptime val = _fill_impl[n]()
        data[thread_idx.x] = val[thread_idx.x]

    var res_device = ctx.enqueue_create_buffer[.float32](16)
    res_device.enqueue_fill(0)

    comptime kernel = static_constant_kernel[16]
    ctx.enqueue_function[kernel](res_device, grid_dim=1, block_dim=16)

    with res_device.map_to_host() as res_host:
        for i in range(16):
            assert_equal(res_host[i], Float32(i))


def test_constant_mem_via_func(ctx: DeviceContext) raises:
    print("== test_constant_mem_via_func")

    def _fill_impl[
        n: Int
    ]() -> MutPointer[Float32, MutUntrackedOrigin, address_space=.CONSTANT]:
        var ptr = unsafe_stack_allocation[n, Float32, address_space=.CONSTANT]()

        comptime for i in range(n):
            ptr[i] = Float32(i)
        return ptr

    def static_constant_kernel[
        get_constant_memory: def() thin -> MutPointer[
            Float32, MutUntrackedOrigin, address_space=.CONSTANT
        ]
    ](data: MutPointer[Float32, MutAnyOrigin]):
        comptime val = get_constant_memory()
        data[thread_idx.x] = val[thread_idx.x]

    var res_device = ctx.enqueue_create_buffer[.float32](16)
    res_device.enqueue_fill(0)

    comptime kernel = static_constant_kernel[_fill_impl[20]]
    ctx.enqueue_function[kernel](res_device, grid_dim=1, block_dim=16)

    with res_device.map_to_host() as res_host:
        for i in range(16):
            assert_equal(res_host[i], Float32(i))


def test_external_constant_mem(ctx: DeviceContext) raises:
    print("== test_external_constant_mem")

    def static_constant_kernel(data: MutPointer[Float32, MutAnyOrigin]):
        var static_constant = unsafe_stack_allocation[
            16,
            Float32,
            name=StaticString("static_constant"),
            address_space=.CONSTANT,
            alignment=8,
        ]()
        data[thread_idx.x] = static_constant[thread_idx.x]

    var constant_memory: List[Float32] = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
    ]

    var res_device = ctx.enqueue_create_buffer[.float32](16)
    res_device.enqueue_fill(0)

    var constant_memory_ptr: Pointer[
        constant_memory.T, origin_of(constant_memory)
    ] = constant_memory.unsafe_ptr()

    comptime kernel = static_constant_kernel
    ctx.enqueue_function[kernel](
        res_device,
        grid_dim=1,
        block_dim=16,
        constant_memory=[
            ConstantMemoryMapping(
                "static_constant",
                constant_memory_ptr.bitcast[NoneType]().unsafe_origin_cast[
                    MutUntrackedOrigin
                ](),
                constant_memory.byte_length(),
            )
        ],
    )

    _ = constant_memory^

    with res_device.map_to_host() as res_host:
        for i in range(16):
            assert_equal(res_host[i], Float32(i))


def main() raises:
    with DeviceContext() as ctx:
        test_constant_memory_compile(ctx)
        test_constant_mem(ctx)
        test_constant_mem_via_func(ctx)
        test_external_constant_mem(ctx)
