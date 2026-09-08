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

from std.builtin.device_passable import DevicePassable
from std.math import align_up
from std.sys import size_of
from max.gpu.host import DeviceBuffer, DeviceContext
from internal_utils._utils import InitializationType

# 512 MiB — larger than 2x the infinity cache on MI300x (256 MiB)
# and larger than 2x the L2 cache on NVIDIA GPUs (A100=40MB, H100=50MB).
comptime CACHE_BUST_BYTES = 512 * 1024 * 1024


struct CacheBustingBuffer[dtype: DType](ImplicitlyCopyable):
    """Per-tensor cache busting buffer for GPU benchmarks.

    Owns a DeviceBuffer sized to exceed 2x the GPU cache. Each benchmark
    iteration uses a different offset into the buffer, preventing cache reuse
    and giving realistic bandwidth/latency numbers.

    When `enabled=False`, allocates only `tensor_size` elements and
    `offset()` always returns 0.
    """

    var _buf: DeviceBuffer[Self.dtype]
    var stride: Int
    var buffer_size: Int

    def __init__(
        out self, tensor_size: Int, ctx: DeviceContext, enabled: Bool = True
    ) raises:
        # A TMA descriptor's globalAddress must be 16-byte aligned.
        self = Self(tensor_size, 16 // size_of[Self.dtype](), ctx, enabled)

    def __init__(
        out self,
        tensor_size: Int,
        alignment: Int,
        ctx: DeviceContext,
        enabled: Bool = True,
        budget_bytes: Int = CACHE_BUST_BYTES,
    ) raises:
        # `budget_bytes` is the target buffer footprint. The default exceeds 2x
        # GPU cache; pass a larger value when one tensor copy is itself bigger
        # than the default (otherwise the buffer collapses to a single window
        # and `offset()` always returns 0 — no cache busting).
        self.stride = align_up(tensor_size, alignment)
        var full_buf_size = (
            align_up(budget_bytes, self.stride * size_of[Self.dtype]())
            // size_of[Self.dtype]()
        )
        self.buffer_size = full_buf_size if enabled else self.stride
        var alloc = full_buf_size if enabled else tensor_size
        self._buf = ctx.enqueue_create_buffer[Self.dtype](alloc)

    @always_inline
    def offset(self, iteration: Int) -> Int:
        """Element offset for a benchmark iteration. Returns 0 when disabled."""
        return (iteration * self.stride) % self.buffer_size

    @always_inline
    def unsafe_ptr(self) -> DeviceBuffer[Self.dtype]._DevicePtr:
        """Raw device pointer to base of buffer."""
        # TODO: This should properly keep/use origins.
        # `DeviceBuffer.unsafe_ptr()` ties the returned pointer's mutability
        # and origin to the borrow of the buffer.
        return (
            self._buf.unsafe_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutUntrackedOrigin]()
        )

    @always_inline
    def offset_ptr(self, iteration: Int) -> DeviceBuffer[Self.dtype]._DevicePtr:
        """Device pointer offset to the window for this iteration."""
        # TODO: This should properly keep/use origins.
        # `DeviceBuffer.unsafe_ptr()` ties the returned pointer's mutability
        # and origin to the borrow of the buffer.
        return (
            (self._buf.unsafe_ptr() + self.offset(iteration))
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutUntrackedOrigin]()
        )

    @always_inline
    def device_buffer(self) -> DeviceBuffer[Self.dtype]:
        """Access underlying DeviceBuffer (copy, since DeviceBuffer is
        ImplicitlyCopyable)."""
        return self._buf

    @always_inline
    def alloc_size(self) -> Int:
        """Number of elements allocated."""
        return len(self._buf)

    def init_on_device(
        self, init_type: InitializationType, ctx: DeviceContext
    ) raises where conforms_to(Scalar[Self.dtype], DevicePassable):
        """Initialize the entire buffer on the device."""
        from internal_utils._utils import init_vector_launch

        init_vector_launch[Self.dtype](
            self._buf, self.alloc_size(), init_type, ctx
        )

    def init_scales_on_device(
        self, init_type: InitializationType, ctx: DeviceContext
    ) raises:
        """Initialize the entire scales buffer on the device."""
        from internal_utils._utils import _init_block_scaled_scales_launch

        _init_block_scaled_scales_launch[Self.dtype](
            self._buf, self.alloc_size(), ctx
        )
