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
"""Provides token-sampling utilities including logit-penalty application and repetition-penalty kernels."""

from std.math import ceildiv, iota
from std.sys.info import simd_width_of

from std.math import isfinite
import max.gpu.primitives.block as block
from max.algorithm.functional import elementwise
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_gpu
from layout import TensorEngine, TensorLayout, TileTensor
from nn._ragged_utils import get_batch_from_row_offsets

from std.utils import IndexList
from std.utils.coord import Coord, coord_to_index_list


def apply_penalties_to_logits[
    logit_type: DType,
    penalty_type: DType,
    //,
    target: StaticString,
](
    logits: TileTensor[mut=True, logit_type, ...],
    compressed_frequency_data: TileTensor[mut=False, .int32, ...],
    frequency_offsets: TileTensor[mut=False, .uint32, ...],
    frequency_penalty: TileTensor[mut=False, penalty_type, ...],
    presence_penalty: TileTensor[mut=False, penalty_type, ...],
    repetition_penalty: TileTensor[mut=False, penalty_type, ...],
    ctx: DeviceContext,
) raises:
    """
    Apply penalties to the logits based on the frequency of the tokens in the batch.

    The frequency data is stored in a CSR format, where the frequency_offsets is the
    starting index of each sequence in the frequency_data array. The frequency_data
    array is a 2D array, where:
    - frequency_data[i, 0] is the token id
    - frequency_data[i, 1] is the frequency of the token in the sequence

    Parameters:
        logit_type: Element type of the `logits` tensor (inferred).
        penalty_type: Element type of the per-batch penalty tensors (inferred).
        target: Target device to dispatch the elementwise kernel to.

    Args:
        logits: 2D logits tensor of shape `[batch, vocab]` updated in place
            with the applied penalties.
        compressed_frequency_data: 2D CSR frequency data where column 0 is the
            token id and column 1 is the token count within the sequence.
        frequency_offsets: 1D tensor of starting indices into
            `compressed_frequency_data` for each sequence in the batch.
        frequency_penalty: Per-batch scalar multiplied by the token count and
            subtracted from each seen token's logit.
        presence_penalty: Per-batch scalar subtracted from each seen token's
            logit.
        repetition_penalty: Per-batch scalar dividing positive logits and
            multiplying non-positive logits of seen tokens; must be finite
            and positive.
        ctx: Device context used to dispatch the kernel.
    """

    comptime assert frequency_offsets.flat_rank == 1
    comptime assert compressed_frequency_data.flat_rank == 2
    comptime assert repetition_penalty.flat_rank == 1
    comptime assert presence_penalty.flat_rank == 1
    comptime assert frequency_penalty.flat_rank == 1
    comptime assert logits.flat_rank == 2

    # all scalars
    comptime assert frequency_offsets.element_size == 1
    comptime assert compressed_frequency_data.element_size == 1
    comptime assert repetition_penalty.element_size == 1
    comptime assert presence_penalty.element_size == 1
    comptime assert frequency_penalty.element_size == 1
    comptime assert logits.element_size == 1

    @always_inline
    def apply_penalties_fn[width: Int, alignment: Int = 1](idx: Coord) {var}:
        comptime assert idx.rank == 1, "apply_penalties_fn: rank must be 1"

        var batch_id = get_batch_from_row_offsets(
            frequency_offsets, Int(idx[0].value())
        )
        var token = Int(compressed_frequency_data[idx[0], 0])

        var repetition_penalty_val = repetition_penalty[batch_id][0]
        var presence_penalty_val = presence_penalty[batch_id][0]
        var frequency_penalty_val = frequency_penalty[batch_id][0]
        debug_assert(
            isfinite(presence_penalty_val) and isfinite(frequency_penalty_val),
            "frequency/presence penalty must be finite",
        )
        # skip padding tokens
        if token >= 0:
            var count = compressed_frequency_data[idx[0], 1][0].cast[
                logit_type
            ]()

            var logit = logits[batch_id, token][0]

            if logit > 0:
                debug_assert(
                    repetition_penalty_val[0] > 0
                    and isfinite(repetition_penalty_val[0]),
                    "repetition_penalty must be finite and > 0, was ",
                    repetition_penalty_val[0],
                )
                logit = logit / repetition_penalty_val.cast[logit_type]()
            else:
                logit = logit * repetition_penalty_val.cast[logit_type]()

            logit -= (
                frequency_penalty_val[0].cast[logit_type]() * count
                + presence_penalty_val[0].cast[logit_type]()
            )

            logits[batch_id, token] = logit

    var dispatch_shape = Coord(Int(compressed_frequency_data.dim[0]()))
    elementwise[
        simd_width=1,
        target=target,
        _trace_description="apply_penalties_to_logits",
    ](apply_penalties_fn, dispatch_shape, ctx)


@__name(t"update_frequency_data_{token_type}")
def update_frequency_data_kernel[
    freq_data_origin: MutOrigin,
    FreqDataLayoutType: TensorLayout,
    freq_offsets_origin: ImmOrigin,
    FreqOffsetsLayoutType: TensorLayout,
    new_tokens_origin: ImmOrigin,
    NewTokensLayoutType: TensorLayout,
    token_type: DType,
    block_size: Int,
    FreqDataEngine: TensorEngine,
    FreqOffsetsEngine: TensorEngine,
    NewTokensEngine: TensorEngine,
](
    compressed_frequency_data: TileTensor[
        .int32, FreqDataLayoutType, freq_data_origin, Engine=FreqDataEngine
    ],
    frequency_offsets: TileTensor[
        .uint32,
        FreqOffsetsLayoutType,
        freq_offsets_origin,
        Engine=FreqOffsetsEngine,
    ],
    new_tokens: TileTensor[
        token_type,
        NewTokensLayoutType,
        new_tokens_origin,
        Engine=NewTokensEngine,
    ],
):
    """
    GPU kernel to update token frequency data in CSR format.

    Searches for new tokens in existing frequency data and either increments
    their count or adds them to the first available padding slot.

    Parameters:
        freq_data_origin: Mutable origin of the `compressed_frequency_data`
            tensor.
        FreqDataLayoutType: Layout type of the
            `compressed_frequency_data` tensor.
        freq_offsets_origin: Immutable origin of the `frequency_offsets`
            tensor.
        FreqOffsetsLayoutType: Layout type of the `frequency_offsets` tensor.
        new_tokens_origin: Immutable origin of the `new_tokens` tensor.
        NewTokensLayoutType: Layout type of the `new_tokens` tensor.
        token_type: Element type of the `new_tokens` tensor.
        block_size: Number of threads per GPU block used to scan a sequence's
            frequency entries.
        FreqDataEngine: Engine policy of the `compressed_frequency_data`
            tensor.
        FreqOffsetsEngine: Engine policy of the `frequency_offsets` tensor.
        NewTokensEngine: Engine policy of the `new_tokens` tensor.

    Args:
        compressed_frequency_data: 2D CSR frequency data where column 0 is the
            token id and column 1 is the token count within the sequence,
            updated in place.
        frequency_offsets: 1D tensor of starting indices into
            `compressed_frequency_data` for each sequence in the batch.
        new_tokens: 1D tensor of new token ids, one per sequence in the batch.
    """

    comptime assert frequency_offsets.flat_rank == 1
    comptime assert compressed_frequency_data.flat_rank == 2
    comptime assert new_tokens.flat_rank == 1

    comptime assert FreqDataEngine.element_size == 1
    comptime assert FreqOffsetsEngine.element_size == 1
    comptime assert NewTokensEngine.element_size == 1

    comptime simd_width = simd_width_of[DType.int32]()
    comptime PADDING_TOKEN = -1

    var tid = thread_idx.x
    var batch_id = block_idx.x

    var tok_start = Int(frequency_offsets.load[width=1](Coord(batch_id)))
    var tok_end = Int(frequency_offsets.load[width=1](Coord(batch_id + 1)))
    var new_token = new_tokens.load[width=1](Coord(batch_id)).cast[.int32]()

    var num_scans = ceildiv(tok_end - tok_start, block_size * simd_width)

    # search if the new token is already in the frequency data
    for scan_idx in range(num_scans):
        var tok_idx = tok_start + ((tid + scan_idx * block_size) * simd_width)

        var val = SIMD[.int32, simd_width](0)

        comptime for i in range(simd_width):
            if tok_idx + i < tok_end:
                val[i] = compressed_frequency_data.load[width=1](
                    Coord(tok_idx + i, 0)
                )[0]
            else:
                val[i] = Int32.MAX_FINITE

        var if_found = val.eq(new_token).select(
            iota[.int32, simd_width](Int32(tok_idx)),
            SIMD[.int32, simd_width](Int32.MIN_FINITE),
        )
        var first_padding_idx = val.eq(PADDING_TOKEN).select(
            iota[.int32, simd_width](Int32(tok_idx)),
            SIMD[.int32, simd_width](Int32.MAX_FINITE),
        )

        var target_token_idx = block.max[block_size=block_size, broadcast=True](
            if_found.reduce_max()
        )
        var padding_token_idx = block.min[
            block_size=block_size, broadcast=True
        ](first_padding_idx.reduce_min())

        if target_token_idx != Int32.MIN_FINITE:
            # we found the target token, update the frequency data
            if tid == 0:
                compressed_frequency_data[Int(target_token_idx), 1] += 1
            return
        elif padding_token_idx != Int32.MAX_FINITE:
            # we don't find the target token, but we found a padding token
            if tid == 0:
                var tidx = Int(padding_token_idx)
                compressed_frequency_data.store[width=1](
                    Coord(tidx, 0), SIMD[new_token.dtype, 1](new_token)
                )
                compressed_frequency_data[Int(padding_token_idx), 1] = 1
            return


def update_frequency_data[
    token_type: DType,
    //,
    target: StaticString,
](
    compressed_frequency_data: TileTensor[
        mut=True, .int32, address_space=.GENERIC, ...
    ],
    frequency_offsets: TileTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    new_tokens: TileTensor[mut=False, token_type, address_space=.GENERIC, ...],
    ctx: DeviceContext,
) raises:
    """
    Update the frequency data for the given new tokens.

    The frequency data is stored in a CSR format. This kernel expects there will be
    enough padding for each sequence to store the new tokens.

    Parameters:
        token_type: Element type of the `new_tokens` tensor (inferred).
        target: Target device to dispatch the kernel to.

    Args:
        compressed_frequency_data: Mutable 2D CSR frequency data where column
            0 is the token id and column 1 is the token count within the
            sequence, updated in place.
        frequency_offsets: 1D tensor of starting indices into
            `compressed_frequency_data` for each sequence in the batch.
        new_tokens: 1D tensor of new token ids, one per sequence in the batch.
        ctx: Device context used to dispatch the kernel.
    """
    comptime assert frequency_offsets.flat_rank == 1
    comptime assert compressed_frequency_data.flat_rank == 2
    comptime assert new_tokens.flat_rank == 1
    comptime assert compressed_frequency_data.element_size == 1
    comptime assert new_tokens.element_size == 1

    comptime if is_gpu[target]():
        comptime block_size = 128

        var dev_ctx = ctx
        comptime kernel = update_frequency_data_kernel[
            freq_data_origin=compressed_frequency_data.origin,
            FreqDataLayoutType=compressed_frequency_data.LayoutType,
            freq_offsets_origin=ImmOrigin(frequency_offsets.origin),
            FreqOffsetsLayoutType=frequency_offsets.LayoutType,
            new_tokens_origin=ImmOrigin(new_tokens.origin),
            NewTokensLayoutType=new_tokens.LayoutType,
            token_type=token_type,
            block_size=block_size,
            FreqDataEngine=compressed_frequency_data.Engine,
            FreqOffsetsEngine=frequency_offsets.Engine,
            NewTokensEngine=new_tokens.Engine,
        ]
        dev_ctx.enqueue_function[kernel](
            compressed_frequency_data,
            frequency_offsets.as_immut(),
            new_tokens.as_immut(),
            grid_dim=new_tokens.dim[0](),
            block_dim=block_size,
        )

    else:

        @always_inline
        def update_frequency_data_fn[
            width: Int, alignment: Int = 1
        ](idx: Coord) {var}:
            comptime assert (
                idx.rank == 1
            ), "update_frequency_data_fn: rank must be 1"

            var tok_start = Int(frequency_offsets.load[width=1](idx)[0])
            var tok_end = Int(
                frequency_offsets.load[width=1](Coord(idx[0].value() + 1))[0]
            )

            var new_token = new_tokens.load[width=1](idx)[0].cast[.int32]()

            for tok_id in range(tok_start, tok_end):
                var cur_tok = compressed_frequency_data.load[width=1](
                    Coord(tok_id, 0)
                )[0]
                if cur_tok == new_token:
                    var cur_count = compressed_frequency_data.load[width=1](
                        Coord(tok_id, 1)
                    )[0]
                    compressed_frequency_data.store[width=1](
                        Coord(tok_id, 1),
                        SIMD[.int32, 1](Int32(cur_count) + 1),
                    )
                    break

                # if we encounter a padding token, add the new token to the
                # occurrences tensor
                elif cur_tok == Int32(-1):
                    compressed_frequency_data.store[width=1](
                        Coord(tok_id, 0), SIMD[.int32, 1](new_token)
                    )
                    compressed_frequency_data.store[width=1](
                        Coord(tok_id, 1), SIMD[.int32, 1](Int32(1))
                    )
                    break

        var dispatch_shape = Coord(new_tokens.num_elements())
        elementwise[
            simd_width=1,
            target=target,
            _trace_description="update_frequency_data",
        ](update_frequency_data_fn, dispatch_shape, ctx)
