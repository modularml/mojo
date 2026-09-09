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
"""Input projection for a multi-token-prediction draft layer.

An MTP draft layer predicts token `t+2` from the embedding of the token the
target just produced and the target's hidden state at `t`:

    eh_proj( concat( enorm(embedding), hnorm(hidden_state) ) )

This kernel does everything before the projection. Both halves reduce over the
same `hidden_size`, so one block computes both row sums and writes the whole
`[tokens, 2 * hidden_size]` row.

Normalization matches `ops.rms_norm` in its Llama-style configuration, which is
what every MTP draft in the tree uses:

    out = cast(x * rsqrt(mean(x^2) + eps)) * weight

The cast comes before the multiply (`multiply_before_cast=False`). Swapping that
order changes the low bits, so the kernel would no longer match
`ops.rms_norm`.

Every tensor must be contiguous and row-major with a last dimension of exactly
`hidden_size`. The kernel addresses elements by linear offset through
`raw_load` and `raw_store`, which bypass the layout, so a transposed or strided
view would read the wrong elements.

Callers own position handling. MAX shifts a draft's token stream left per
request and appends a bonus token (`eagle_prefill_shift_tokens`), so every row
arriving here already holds the embedding it needs. Masking the first position
here as well would blank a correct row.
"""

from std.math import rsqrt
from max.gpu import WARP_SIZE, block_dim, block_idx, thread_idx
from layout import TensorEngine, TensorLayout, TileTensor

from nn.normalization import block_reduce_dual_sum


@__name(t"mtp_eh_norm_{hidden_size}_{max_warps_per_block}")
def mtp_eh_norm_kernel[
    dtype: DType,
    OutLayoutType: TensorLayout,
    out_origin: MutOrigin,
    EmbedLayoutType: TensorLayout,
    embed_origin: ImmOrigin,
    PrevLayoutType: TensorLayout,
    prev_origin: ImmOrigin,
    EWLayoutType: TensorLayout,
    ew_origin: ImmOrigin,
    HWLayoutType: TensorLayout,
    hw_origin: ImmOrigin,
    hidden_size: Int,
    max_warps_per_block: Int,
    OutEngine: TensorEngine,
    EmbedEngine: TensorEngine,
    PrevEngine: TensorEngine,
    EWEngine: TensorEngine,
    HWEngine: TensorEngine,
](
    out_buf: TileTensor[dtype, OutLayoutType, out_origin, Engine=OutEngine],
    embed: TileTensor[
        mut=False, dtype, EmbedLayoutType, embed_origin, Engine=EmbedEngine
    ],
    prev: TileTensor[
        mut=False, dtype, PrevLayoutType, prev_origin, Engine=PrevEngine
    ],
    enorm_weight: TileTensor[
        mut=False, dtype, EWLayoutType, ew_origin, Engine=EWEngine
    ],
    hnorm_weight: TileTensor[
        mut=False, dtype, HWLayoutType, hw_origin, Engine=HWEngine
    ],
    epsilon: Float32,
    num_tokens: Int32,
):
    """Normalizes both inputs for one token and writes them side by side.

    Parameters:
        dtype: Element type of the inputs, the weights and the output.
        OutLayoutType: Layout of `out_buf`.
        out_origin: Origin of `out_buf`.
        EmbedLayoutType: Layout of `embed`.
        embed_origin: Origin of `embed`.
        PrevLayoutType: Layout of `prev`.
        prev_origin: Origin of `prev`.
        EWLayoutType: Layout of `enorm_weight`.
        ew_origin: Origin of `enorm_weight`.
        HWLayoutType: Layout of `hnorm_weight`.
        hw_origin: Origin of `hnorm_weight`.
        hidden_size: Channels per input; the output row is twice this.
        max_warps_per_block: Power-of-two ceiling on the block's warp count,
            not the launch's actual warp count. `block_reduce_dual_sum` reduces
            over a power-of-two lane group; slots past the real warp count hold
            zero.
        OutEngine: Engine policy of `out_buf`.
        EmbedEngine: Engine policy of `embed`.
        PrevEngine: Engine policy of `prev`.
        EWEngine: Engine policy of `enorm_weight`.
        HWEngine: Engine policy of `hnorm_weight`.

    Args:
        out_buf: Output `[num_tokens, 2 * hidden_size]`. Columns
            `[0, hidden_size)` hold the normalized embedding, the rest the
            normalized hidden state.
        embed: Token embeddings `[num_tokens, hidden_size]`.
        prev: Target hidden states `[num_tokens, hidden_size]`.
        enorm_weight: Embedding norm weight `[hidden_size]`.
        hnorm_weight: Hidden-state norm weight `[hidden_size]`.
        epsilon: Added inside the square root, matching `ops.rms_norm`.
        num_tokens: Rows to process.
    """
    var token = block_idx.x
    if token >= Int(num_tokens):
        return

    # `block_reduce_dual_sum` reads one partial sum per whole warp, so a block
    # width that is not a multiple of `WARP_SIZE` drops its last partial warp.
    debug_assert(
        block_dim.x % WARP_SIZE == 0,
        "mtp_eh_norm block width must be a whole number of warps",
    )

    var tid = thread_idx.x
    var stride = block_dim.x
    var row = token * hidden_size

    # Reduce in float32 regardless of the input dtype, matching `ops.rms_norm`.
    var acc_e = Float32(0)
    var acc_h = Float32(0)
    var col = tid
    while col < hidden_size:
        var e = embed.raw_load(row + col).cast[.float32]()
        var p = prev.raw_load(row + col).cast[.float32]()
        acc_e += e * e
        acc_h += p * p
        col += stride

    var sums = block_reduce_dual_sum[.float32, max_warps_per_block](
        acc_e, acc_h
    )
    var rrms_e = rsqrt(sums[0] / Float32(hidden_size) + epsilon)
    var rrms_h = rsqrt(sums[1] / Float32(hidden_size) + epsilon)

    var out_row = token * 2 * hidden_size
    col = tid
    while col < hidden_size:
        # Cast before the multiply: `multiply_before_cast=False`.
        var e = embed.raw_load(row + col).cast[.float32]() * rrms_e
        var p = prev.raw_load(row + col).cast[.float32]() * rrms_h
        out_buf.raw_store(
            out_row + col,
            e.cast[dtype]() * enorm_weight.raw_load(col),
        )
        out_buf.raw_store(
            out_row + hidden_size + col,
            p.cast[dtype]() * hnorm_weight.raw_load(col),
        )
        col += stride
