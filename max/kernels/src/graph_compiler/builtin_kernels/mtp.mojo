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
"""Graph-op bindings for multi-token-prediction draft layers.

The kernel math lives in `nn.mtp_eh_norm`. Registration must be declared inside
the built-in kernel library; importing the kernel from `nn` does not add the op
to the graph compiler's registry.
"""

import extensibility
from extensibility import InputTensor, OutputTensor
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_gpu
from max.gpu import WARP_SIZE

from nn.mtp_eh_norm import mtp_eh_norm_kernel


@extensibility.register("mo.mtp.eh_norm")
struct MTPEhNorm:
    """Registers the `mo.mtp.eh_norm` graph op with the graph compiler."""

    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
        //,
        hidden_size: Int,
        block_threads: Int,
    ](
        out_buf: OutputTensor[dtype=dtype, rank=2, ...],
        embed: InputTensor[dtype=dtype, rank=2, ...],
        prev: InputTensor[dtype=dtype, rank=2, ...],
        enorm_weight: InputTensor[dtype=dtype, rank=1, ...],
        hnorm_weight: InputTensor[dtype=dtype, rank=1, ...],
        epsilon: Float32,
        ctx: DeviceContext,
    ) raises:
        """Normalizes both draft inputs and writes them side by side.

        Parameters:
            dtype: Element type of the inputs, weights and output.
            target: Compilation target.
            hidden_size: Channels per input; the output row is twice this.
            block_threads: Threads per block.

        Args:
            out_buf: Output `[num_tokens, 2 * hidden_size]`.
            embed: Token embeddings `[num_tokens, hidden_size]`.
            prev: Target hidden states `[num_tokens, hidden_size]`.
            enorm_weight: Embedding norm weight `[hidden_size]`.
            hnorm_weight: Hidden-state norm weight `[hidden_size]`.
            epsilon: Added inside the square root.
            ctx: Device context.

        Raises:
            Error: If the target is not a GPU.
        """
        comptime assert is_gpu[target](), "mo.mtp.eh_norm is GPU only"
        comptime assert (
            block_threads % WARP_SIZE == 0
        ), "block_threads must be a whole number of warps"
        comptime assert (
            block_threads <= ctx.default_device_info.max_thread_block_size
        ), "block_threads exceeds the device's maximum block size"
        # `block_reduce_dual_sum` reduces over a power-of-two lane group, so
        # pass a power-of-two ceiling rather than this launch's warp count.
        # Slots past the real warp count hold zero.
        comptime max_warps = (
            ctx.default_device_info.max_thread_block_size // WARP_SIZE
        )

        var out_tt = out_buf.to_tile_tensor[.int64]()
        var embed_tt = embed.to_tile_tensor[.int64]()
        var prev_tt = prev.to_tile_tensor[.int64]()
        var ew_tt = enorm_weight.to_tile_tensor[.int64]()
        var hw_tt = hnorm_weight.to_tile_tensor[.int64]()

        var num_tokens = Int(out_tt.dim[0]())
        if num_tokens == 0:
            return

        comptime kernel = mtp_eh_norm_kernel[
            dtype,
            out_tt.LayoutType,
            out_tt.origin,
            type_of(embed_tt.as_immut()).LayoutType,
            ImmOrigin(embed_tt.origin),
            type_of(prev_tt.as_immut()).LayoutType,
            ImmOrigin(prev_tt.origin),
            type_of(ew_tt.as_immut()).LayoutType,
            ImmOrigin(ew_tt.origin),
            type_of(hw_tt.as_immut()).LayoutType,
            ImmOrigin(hw_tt.origin),
            hidden_size,
            max_warps,
            out_tt.Engine,
            type_of(embed_tt.as_immut()).Engine,
            type_of(prev_tt.as_immut()).Engine,
            type_of(ew_tt.as_immut()).Engine,
            type_of(hw_tt.as_immut()).Engine,
        ]
        ctx.enqueue_function[kernel](
            out_tt,
            embed_tt.as_immut(),
            prev_tt.as_immut(),
            ew_tt.as_immut(),
            hw_tt.as_immut(),
            epsilon,
            Int32(num_tokens),
            grid_dim=num_tokens,
            block_dim=block_threads,
        )
