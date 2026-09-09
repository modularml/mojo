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

"""Registers log-probability computation graph ops used by sampling pipelines."""

from std.math import ceildiv, exp, inf, log

from max.algorithm.functional import parallelize
from extensibility import register, register_shape_function
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_cpu, is_gpu
from nn._ragged_utils import get_batch_from_row_offsets

from extensibility import InputTensor, OutputTensor
from extensibility import Tensor, TileTensorable

from std.utils.coord import coord_to_index_list
from std.utils.index import IndexList


struct FixedHeightMinHeap[k_dtype: DType, v_dtype: DType, levels: Int]:
    """Maintains a fixed-capacity min-heap of key/value pairs used to track top-k logits.

    Parameters:
        k_dtype: Element type of the heap keys, the token id dtype.
        v_dtype: Element type of the heap values, the logit dtype.
        levels: Number of heap levels; the heap stores `2**levels - 1` entries.
    """

    comptime num_elements = 2**Self.levels - 1
    """Maximum number of entries the heap can hold."""

    var k_array: Array[Scalar[Self.k_dtype], Self.num_elements]
    """Inline array of heap keys, storing token ids."""

    var v_array: Array[Scalar[Self.v_dtype], Self.num_elements]
    """Inline array of heap values, storing logits."""

    def __init__(
        out self, *, fill_k: Scalar[Self.k_dtype], fill_v: Scalar[Self.v_dtype]
    ):
        """Initializes the heap with fill values for all slots.

        Args:
            fill_k: Key value to fill every heap slot with.
            fill_v: Value to fill every heap slot with.
        """
        self.k_array = Array[length=Self.num_elements](fill=fill_k)
        self.v_array = Array[length=Self.num_elements](fill=fill_v)

    @always_inline
    def swap(mut self, a: Int, b: Int) -> None:
        """Swaps the key/value pairs at two heap positions.

        Args:
            a: Index of the first element to swap.
            b: Index of the second element to swap.
        """
        self.k_array[a], self.k_array[b] = self.k_array[b], self.k_array[a]
        self.v_array[a], self.v_array[b] = self.v_array[b], self.v_array[a]

    def heap_down(mut self) -> None:
        """Restores the min-heap property by sifting the root down."""
        var current_index = 0

        comptime for level in range(Self.levels - 1):
            # Must ensure:
            # arr[cur] < arr[left] && arr[cur] < arr[right]
            var left_index = current_index * 2 + 1
            var right_index = current_index * 2 + 2
            var smaller_index = left_index
            if self.v_array[right_index] < self.v_array[left_index]:
                smaller_index = right_index
            if self.v_array[current_index] < self.v_array[smaller_index]:
                # Full heap property is satisfied. We could stop here,
                # but this is an unrolled loop, so just continue on.
                # (Useless but harmless work.)
                pass
            else:
                self.swap(current_index, smaller_index)
                current_index = smaller_index


comptime logit_dtype = DType.float32
"""Element type of logit tensors used by log-probability ops."""
comptime token_dtype = DType.uint32
"""Element type of token id tensors used by log-probability ops."""
comptime offset_dtype = DType.uint32
"""Element type of row offset tensors used by log-probability ops."""


def compute_log_probabilities_1tok[
    target: StaticString, levels: Int
](
    output_token_index: Int,
    lp_logits: OutputTensor[dtype=logit_dtype, rank=2, ...],
    lp_tokens: OutputTensor[dtype=token_dtype, rank=2, ...],
    logits: InputTensor[dtype=logit_dtype, rank=2, ...],
    tokens: InputTensor[dtype=token_dtype, rank=1, ...],
    sampled_tokens: InputTensor[dtype=token_dtype, rank=1, ...],
    logit_row_offsets: InputTensor[dtype=offset_dtype, rank=1, ...],
    token_row_offsets: InputTensor[dtype=offset_dtype, rank=1, ...],
    lp_output_offsets: InputTensor[dtype=offset_dtype, rank=1, ...],
) -> None:
    """Computes log probabilities for a single token position from its logits row.

    Parameters:
        target: Compilation target string, such as `"cpu"` or `"gpu"`.
        levels: Number of heap levels controlling the top-k width; the output
            stores `2**levels` entries per token. When `levels <= 0`, only
            the sampled token's log probability is emitted.

    Args:
        output_token_index: Global row index into `lp_logits` and `lp_tokens`
            for this output position.
        lp_logits: Output tensor of shape `[num_output_tokens, 2**levels]`
            receiving the log probabilities.
        lp_tokens: Output tensor of shape `[num_output_tokens, 2**levels]`
            receiving the token ids paired with `lp_logits`.
        logits: Input logits ragged by batch, shape
            `[total_rows, vocab_size]`.
        tokens: Previously generated tokens across all batches, ragged.
        sampled_tokens: Most recently sampled token for each batch, used when
            this position is the last token in its sequence.
        logit_row_offsets: Per-batch start offsets into the first axis of
            `logits`.
        token_row_offsets: Per-batch start offsets into `tokens`.
        lp_output_offsets: Per-batch start offsets into the output row axis,
            mapping each output index to its batch.
    """
    var vocab_size = logits.shape()[1]
    var batch_index = get_batch_from_row_offsets(
        lp_output_offsets.to_tile_tensor[.int64](), output_token_index
    )
    var reverse_index_in_seq = (
        lp_output_offsets[batch_index + 1] - UInt32(output_token_index) - 1
    )
    var token_end_index = token_row_offsets[batch_index + 1]
    var sampled_token: Scalar[token_dtype]
    if reverse_index_in_seq == 0:
        sampled_token = sampled_tokens[batch_index]
    else:
        sampled_token = tokens[Int(token_end_index - reverse_index_in_seq)]

    var logit_end_index = logit_row_offsets[batch_index + 1]
    var logit_index = Int(logit_end_index - reverse_index_in_seq - 1)
    var x_max = logits[logit_index, 0]
    for token_value in range(1, vocab_size):
        x_max = max(x_max, logits[logit_index, token_value])
    var sum_exp = Scalar[logit_dtype](0.0)
    for token_value in range(vocab_size):
        sum_exp += exp(logits[logit_index, token_value] - x_max)
    var log_sum_exp = log(sum_exp)
    var normalizer = -(x_max + log_sum_exp)

    var post_heap_idx: Int

    comptime if levels <= 0:
        post_heap_idx = 0
    else:
        var heap = FixedHeightMinHeap[token_dtype, logit_dtype, levels](
            fill_k=UInt32(vocab_size), fill_v=-inf[logit_dtype]()
        )
        for token_value in range(vocab_size):
            var logit_value = logits[logit_index, token_value]
            if logit_value > heap.v_array[0]:
                heap.k_array[0] = UInt32(token_value)
                heap.v_array[0] = logit_value
                heap.heap_down()
        for i in range(heap.num_elements):
            lp_tokens[output_token_index, i] = heap.k_array[i]
            lp_logits[output_token_index, i] = heap.v_array[i] + normalizer
        post_heap_idx = heap.num_elements
    lp_tokens[output_token_index, post_heap_idx] = sampled_token
    lp_logits[output_token_index, post_heap_idx] = (
        logits[logit_index, Int(sampled_token)] + normalizer
    )


@register("compute_log_probabilities_ragged")
struct LogProbabilitiesRagged:
    """Registers the `compute_log_probabilities_ragged` graph op computing per-token log probabilities over ragged batches.
    """

    @staticmethod
    def execute[
        target: StaticString, levels: Int
    ](
        lp_logits: OutputTensor[dtype=logit_dtype, rank=2, ...],
        lp_tokens: OutputTensor[dtype=token_dtype, rank=2, ...],
        logits: InputTensor[dtype=logit_dtype, rank=2, ...],
        tokens: InputTensor[dtype=token_dtype, rank=1, ...],
        sampled_tokens: InputTensor[dtype=token_dtype, rank=1, ...],
        logit_row_offsets: InputTensor[dtype=offset_dtype, rank=1, ...],
        token_row_offsets: InputTensor[dtype=offset_dtype, rank=1, ...],
        lp_output_offsets: InputTensor[dtype=offset_dtype, rank=1, ...],
        lp_output_offsets_host: InputTensor[dtype=offset_dtype, rank=1, ...],
        ctx: DeviceContext,
    ) raises -> None:
        """Executes the ragged log-probabilities computation across all batches.

        Parameters:
            target: Compilation target string, such as `"cpu"` or `"gpu"`.
            levels: Number of heap levels controlling the top-k width; the
                output stores `2**levels` entries per token.

        Args:
            lp_logits: Output tensor of shape `[num_output_tokens, 2**levels]`
                receiving the log probabilities.
            lp_tokens: Output tensor of shape `[num_output_tokens, 2**levels]`
                receiving the token ids paired with `lp_logits`.
            logits: Input logits ragged by batch, shape
                `[total_rows, vocab_size]`.
            tokens: Previously generated tokens across all batches, ragged.
            sampled_tokens: Most recently sampled token for each batch.
            logit_row_offsets: Per-batch start offsets into the first axis
                of `logits`.
            token_row_offsets: Per-batch start offsets into `tokens`.
            lp_output_offsets: Per-batch start offsets into the output row
                axis.
            lp_output_offsets_host: Host-resident copy of `lp_output_offsets`
                whose last element gives the total number of output tokens.
            ctx: Device context used to enqueue the computation.

        Raises:
            Error: If `lp_logits` and `lp_tokens` disagree on axis 0, or if
                either output's axis 1 does not match `2**levels`.
        """
        var num_output_tokens = lp_logits.shape()[0]
        if lp_tokens.shape()[0] != num_output_tokens:
            raise Error("Mismatch in axis 0 of lp_logits and lp_tokens")
        if lp_logits.shape()[1] != 2**levels:
            raise Error("Axis 1 of lp_logits inconsistent with level setting")
        if lp_tokens.shape()[1] != 2**levels:
            raise Error("Axis 1 of lp_tokens inconsistent with level setting")

        comptime if is_cpu[target]():

            def lp_idx_kernel(output_token_index: Int) {imm} -> None:
                compute_log_probabilities_1tok[target, levels](
                    output_token_index=output_token_index,
                    lp_logits=lp_logits,
                    lp_tokens=lp_tokens,
                    logits=logits,
                    tokens=tokens,
                    sampled_tokens=sampled_tokens,
                    logit_row_offsets=logit_row_offsets,
                    token_row_offsets=token_row_offsets,
                    lp_output_offsets=lp_output_offsets,
                )

            parallelize(
                lp_idx_kernel,
                num_output_tokens,
                ctx=Optional[DeviceContext](ctx),
            )
        elif is_gpu[target]():

            @__parameter
            @__copy_capture(num_output_tokens)
            @__name(t"log_probabilities_l{levels}")
            def raw_lp_kernel():
                var output_token_index = global_idx.x
                if output_token_index < num_output_tokens:
                    compute_log_probabilities_1tok[target, levels](
                        output_token_index=output_token_index,
                        lp_logits=lp_logits,
                        lp_tokens=lp_tokens,
                        logits=logits,
                        tokens=tokens,
                        sampled_tokens=sampled_tokens,
                        logit_row_offsets=logit_row_offsets,
                        token_row_offsets=token_row_offsets,
                        lp_output_offsets=lp_output_offsets,
                    )

            comptime block_size = 64
            ctx.enqueue_function[raw_lp_kernel](
                grid_dim=ceildiv(num_output_tokens, block_size),
                block_dim=block_size,
            )
        else:
            comptime assert False, "unsupported target"


@register_shape_function("compute_log_probabilities_ragged")
def compute_log_probabilities_ragged_shape[
    levels: Int,
](
    logits: Some[Tensor],
    tokens: Some[Tensor],
    sampled_tokens: Some[Tensor],
    logit_row_offsets: Some[Tensor],
    token_row_offsets: Some[Tensor],
    lp_output_offsets: Some[Tensor],
    lp_output_offsets_host: Some[TileTensorable],
) -> IndexList[2]:
    """Computes the output shapes for the ragged log-probabilities op.

    Parameters:
        levels: Number of heap levels; the output second dimension is
            `2**levels`.

    Args:
        logits: Input logits ragged by batch, shape
            `[total_rows, vocab_size]`.
        tokens: Previously generated tokens across all batches, ragged.
        sampled_tokens: Most recently sampled token for each batch.
        logit_row_offsets: Per-batch start offsets into the first axis of
            `logits`.
        token_row_offsets: Per-batch start offsets into `tokens`.
        lp_output_offsets: Per-batch start offsets into the output row axis.
        lp_output_offsets_host: Host-resident copy of `lp_output_offsets`
            whose last element gives the total number of output tokens.

    Returns:
        The output shape `[num_output_tokens, 2**levels]`.
    """
    comptime assert (
        type_of(logits).dtype == logit_dtype
    ), "logits dtype must be logit_dtype"
    comptime assert type_of(logits).rank == 2, "logits must be rank 2"
    comptime assert (
        type_of(tokens).dtype == token_dtype
    ), "tokens dtype must be token_dtype"
    comptime assert type_of(tokens).rank == 1, "tokens must be rank 1"
    comptime assert (
        type_of(sampled_tokens).dtype == token_dtype
    ), "sampled_tokens dtype must be token_dtype"
    comptime assert (
        type_of(sampled_tokens).rank == 1
    ), "sampled_tokens must be rank 1"
    comptime assert (
        type_of(logit_row_offsets).dtype == offset_dtype
    ), "logit_row_offsets dtype must be offset_dtype"
    comptime assert (
        type_of(logit_row_offsets).rank == 1
    ), "logit_row_offsets must be rank 1"
    comptime assert (
        type_of(token_row_offsets).dtype == offset_dtype
    ), "token_row_offsets dtype must be offset_dtype"
    comptime assert (
        type_of(token_row_offsets).rank == 1
    ), "token_row_offsets must be rank 1"
    comptime assert (
        type_of(lp_output_offsets).dtype == offset_dtype
    ), "lp_output_offsets dtype must be offset_dtype"
    comptime assert (
        type_of(lp_output_offsets).rank == 1
    ), "lp_output_offsets must be rank 1"
    comptime assert (
        type_of(lp_output_offsets_host).dtype == offset_dtype
    ), "lp_output_offsets_host dtype must be offset_dtype"
    comptime assert (
        type_of(lp_output_offsets_host).rank == 1
    ), "lp_output_offsets_host must be rank 1"
    var num_offsets = coord_to_index_list(
        lp_output_offsets_host.shape().tuple()
    )[0]
    return IndexList[2](
        Int(lp_output_offsets_host.to_tile_tensor().raw_load(num_offsets - 1)),
        2**levels,
    )
