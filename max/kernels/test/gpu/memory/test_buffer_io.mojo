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

from std.math import align_down

from max.gpu import thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, get_gpu_target
from max.gpu.host.compile import _compile_code
from max.gpu.host.info import MI355X
from max.gpu.intrinsics import AMDBufferResource
from max.gpu.memory import CacheOperation
from std.memory import unsafe_stack_allocation
from std.testing import assert_equal, assert_true

comptime size = 257
comptime size_clip = size - 5


def kernel[
    dtype: DType, width: Int
](a: MutPointer[Scalar[dtype], MutAnyOrigin]):
    var aligned_size = align_down(size, width)
    var buffer = AMDBufferResource(a, size_clip)
    for i in range(0, aligned_size, width):
        var v = buffer.load[dtype, width](Int32(i))
        buffer.store[dtype, width](0, 2 * v, scalar_offset=Int32(i))
    for i in range(aligned_size, size):
        var v = buffer.load[dtype, 1](0, scalar_offset=Int32(i))
        buffer.store[dtype, 1](Int32(i), 2 * v)


def kernel_lds[
    dtype: DType, width: Int
](a: MutPointer[Scalar[dtype], MutAnyOrigin]):
    var a_shared = unsafe_stack_allocation[size, dtype, address_space=.SHARED]()

    var aligned_size = align_down(size, width)
    var buffer = AMDBufferResource(a, size_clip)
    for i in range(size):
        a_shared[i] = 0
    barrier()

    for i in range(0, aligned_size, width):
        buffer.load_to_lds[width=width](Int32(i), a_shared + i)
    for i in range(aligned_size, size):
        buffer.load_to_lds[width=1](Int32(i), a_shared + i)
    barrier()

    for i in range(size):
        a[i] = 2 * a_shared[i]


# Assembly test kernels for different cache policies
def cache_policy_kernel_always():
    var dummy_ptr = MutPointer[Float32, MutAnyOrigin].unsafe_dangling()
    var buffer = AMDBufferResource(dummy_ptr, 1024)
    var offset = Int32(thread_idx.x)  # Use dynamic offset to force offen mode
    var v = buffer.load[.float32, 4, cache_policy=CacheOperation.ALWAYS](offset)
    buffer.store[.float32, 4, cache_policy=CacheOperation.ALWAYS](offset, v)


def cache_policy_kernel_streaming():
    var dummy_ptr = MutPointer[Float32, MutAnyOrigin].unsafe_dangling()
    var buffer = AMDBufferResource(dummy_ptr, 1024)
    var offset = Int32(thread_idx.x)  # Use dynamic offset to force offen mode
    var v = buffer.load[
        DType.float32, 4, cache_policy=CacheOperation.STREAMING
    ](offset)
    buffer.store[.float32, 4, cache_policy=CacheOperation.STREAMING](offset, v)


def cache_policy_kernel_global():
    var dummy_ptr = MutPointer[Float32, MutAnyOrigin].unsafe_dangling()
    var buffer = AMDBufferResource(dummy_ptr, 1024)
    var offset = Int32(thread_idx.x)  # Use dynamic offset to force offen mode
    var v = buffer.load[.float32, 4, cache_policy=CacheOperation.GLOBAL](offset)
    buffer.store[.float32, 4, cache_policy=CacheOperation.GLOBAL](offset, v)


def cache_policy_kernel_volatile():
    var dummy_ptr = MutPointer[Float32, MutAnyOrigin].unsafe_dangling()
    var buffer = AMDBufferResource(dummy_ptr, 1024)
    var offset = Int32(thread_idx.x)  # Use dynamic offset to force offen mode
    var v = buffer.load[.float32, 4, cache_policy=CacheOperation.VOLATILE](
        offset
    )
    buffer.store[.float32, 4, cache_policy=CacheOperation.VOLATILE](offset, v)


@always_inline
def _verify_cache_bits_always(asm: StringSlice) raises -> None:
    # aux = 0x00 - no cache control bits set
    # Should generate: buffer_load_dwordx4 v[...], v..., s[...], 0 offen
    assert_true("buffer_load_dwordx4" in asm)
    assert_true("offen" in asm)
    # Should NOT have sc0, nt, or sc1
    assert_true("sc0" not in asm)
    assert_true(" nt" not in asm)  # Space prefix to avoid matching "int"
    assert_true("sc1" not in asm)


@always_inline
def _verify_cache_bits_streaming(asm: StringSlice) raises -> None:
    # aux = 0x02 - only NT bit set
    # Should generate: buffer_load_dwordx4 v[...], v..., s[...], 0 offen nt
    assert_true("buffer_load_dwordx4" in asm)
    assert_true("offen" in asm)
    assert_true(" nt" in asm)  # Space prefix to avoid matching "int"
    # Should NOT have sc0 or sc1
    assert_true("sc0" not in asm)
    assert_true("sc1" not in asm)


@always_inline
def _verify_cache_bits_global(asm: StringSlice) raises -> None:
    # aux = 0x10 - only SC1 bit set
    # Should generate: buffer_load_dwordx4 v[...], v..., s[...], 0 offen sc1
    assert_true("buffer_load_dwordx4" in asm)
    assert_true("offen" in asm)
    assert_true("sc1" in asm)
    # Should NOT have sc0 or nt
    assert_true("sc0" not in asm)
    assert_true(" nt" not in asm)  # Space prefix to avoid matching "int"


@always_inline
def _verify_cache_bits_volatile(asm: StringSlice) raises -> None:
    # aux = 0x11 - SC0 and SC1 bits set
    # Should generate: buffer_load_dwordx4 v[...], v..., s[...], 0 offen sc0 sc1
    assert_true("buffer_load_dwordx4" in asm)
    assert_true("offen" in asm)
    assert_true("sc0" in asm)
    assert_true("sc1" in asm)
    # Should NOT have nt
    assert_true(" nt" not in asm)  # Space prefix to avoid matching "int"


def test_cache_policy_assembly_always() raises:
    var asm = _compile_code[
        cache_policy_kernel_always, target=get_gpu_target["mi300x"]()
    ]().asm
    _verify_cache_bits_always(asm)


def test_cache_policy_assembly_streaming() raises:
    var asm = _compile_code[
        cache_policy_kernel_streaming, target=get_gpu_target["mi300x"]()
    ]().asm
    _verify_cache_bits_streaming(asm)


def test_cache_policy_assembly_global() raises:
    var asm = _compile_code[
        cache_policy_kernel_global, target=get_gpu_target["mi300x"]()
    ]().asm
    _verify_cache_bits_global(asm)


def test_cache_policy_assembly_volatile() raises:
    var asm = _compile_code[
        cache_policy_kernel_volatile, target=get_gpu_target["mi300x"]()
    ]().asm
    _verify_cache_bits_volatile(asm)


def test_buffer[dtype: DType, width: Int](ctx: DeviceContext) raises:
    var a_host_buf = alloc[Scalar[dtype]](size)
    var a_device_buf = ctx.enqueue_create_buffer[dtype](size)

    for i in range(size):
        a_host_buf[i] = Scalar[dtype](i + 1)

    ctx.enqueue_copy(a_device_buf, a_host_buf)

    comptime kernel_func = kernel[dtype, width]
    ctx.enqueue_function[kernel_func](a_device_buf, grid_dim=1, block_dim=1)
    ctx.enqueue_copy(a_host_buf, a_device_buf)

    ctx.synchronize()
    for i in range(size_clip):
        assert_equal(a_host_buf[i], Scalar[dtype](2 * (i + 1)))
    for i in range(size_clip, size):
        assert_equal(a_host_buf[i], Scalar[dtype](i + 1))

    a_host_buf.free()


def test_buffer_lds[dtype: DType, width: Int](ctx: DeviceContext) raises:
    var a_host_buf = alloc[Scalar[dtype]](size)
    var a_device_buf = ctx.enqueue_create_buffer[dtype](size)

    for i in range(size):
        a_host_buf[i] = Scalar[dtype](i + 1)

    ctx.enqueue_copy(a_device_buf, a_host_buf)

    comptime kernel_lds_func = kernel_lds[dtype, width]
    ctx.enqueue_function[kernel_lds_func](a_device_buf, grid_dim=1, block_dim=1)
    ctx.enqueue_copy(a_host_buf, a_device_buf)

    ctx.synchronize()
    for i in range(size_clip):
        assert_equal(a_host_buf[i], Scalar[dtype](2 * (i + 1)))
    for i in range(size_clip, size):
        assert_equal(a_host_buf[i], 0)


def main() raises:
    # Test assembly generation for cache policies (AMD GPU only)
    test_cache_policy_assembly_always()
    test_cache_policy_assembly_streaming()
    test_cache_policy_assembly_global()
    test_cache_policy_assembly_volatile()
    # test_cache_policy_assembly_volatile_streaming()  # Future test

    # Test functional behavior
    with DeviceContext() as ctx:
        comptime for width in [1, 2, 4, 8]:
            test_buffer[.bfloat16, width](ctx)

        comptime for width in [1, 2, 4, 8, 16]:
            test_buffer[.int8, width](ctx)

        test_buffer_lds[.float32, 1](ctx)

        comptime if ctx.default_device_info == MI355X:
            test_buffer_lds[.bfloat16, 8](ctx)
