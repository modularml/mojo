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

"""Implements ragged token merging for speculative decoding workflows."""

__all__ = [
    "RaggedTokenMerger",
    "compute_host_merged_offsets",
    "ragged_token_merger",
]

from max.dtype import DType
from max.graph import DeviceRef, Dim, Graph, TensorType, TensorValue, ops
from max.nn.kernels import merge_ragged_tensors
from max.nn.layer import Module


def _shape_to_scalar(
    x: Dim, device: DeviceRef, dtype: DType = DType.int64
) -> TensorValue:
    """An especially cursed way to turn a shape into a scalar on the GPU.

    Usage:
    >>> my_scalar = _shape_to_scalar(tensor.shape[3], DeviceRef.GPU())

    We previously did `ops.shape_to_tensor([x]).to(DeviceRef.GPU())` but this
    issues a h2d copy that is incompatible with CUDA Graphs.

    TODO: Delete this method asap! We should really support a device placement
    when using shape_to_tensor.
    """
    x_cpu = ops.shape_to_tensor([x])
    x_gpu = ops.range(
        start=x_cpu[0],
        stop=x_cpu[0] + 1,
        out_dim=1,
        dtype=dtype,
        device=device,
    )[0]
    return x_gpu


def ragged_token_merger(device: DeviceRef) -> Graph:
    """Builds a graph that merges prompt and draft tokens into a single ragged sequence.

    Args:
        device: Device for the graph inputs and merge op.

    Returns:
        A graph that takes prompt tokens, prompt row offsets, and draft tokens and
        outputs merged tokens and merged row offsets.
    """
    graph_inputs = [
        TensorType(DType.int64, ["batch_prompt_seq_len"], device=device),
        TensorType(DType.uint32, ["offsets_len"], device=device),
        TensorType(DType.int64, ["batch_size", "draft_seq_len"], device=device),
    ]

    with Graph("merge_prompt_draft_tokens", input_types=graph_inputs) as graph:
        prompt_tensor, prompt_row_offsets, draft_tensor = graph.inputs

        merge_op = RaggedTokenMerger(device)
        merged_tensor, merged_row_offsets = merge_op(
            prompt_tensor.tensor, prompt_row_offsets.tensor, draft_tensor.tensor
        )

        graph.output(merged_tensor, merged_row_offsets)

        return graph


class RaggedTokenMerger(Module):
    """Merges prompt and draft token sequences into a single ragged batch."""

    def __init__(self, device: DeviceRef) -> None:
        super().__init__()
        self.device = device

    def __call__(
        self,
        prompt_tokens: TensorValue,
        prompt_offsets: TensorValue,
        draft_tokens: TensorValue,
    ) -> tuple[TensorValue, TensorValue]:
        """Merges prompt and draft tokens into a single ragged token sequence.

        Args:
            prompt_tokens: The prompt tokens of shape [S].
            prompt_offsets: The prompt offsets of shape [B+1].
            draft_tokens: The draft tokens of shape [B, K].

        Returns:
            A tuple of two tensors:
                - The merged tokens of shape [S+B*K].
                - The merged offsets of shape [B+1].
        """
        device = prompt_tokens.device
        K = _shape_to_scalar(draft_tokens.shape[1], device, dtype=DType.uint32)
        draft_tokens_flattened = ops.reshape(draft_tokens, shape=(-1,))

        # Compute draft_offsets as [0, K, 2K, ..., N*K] where K=num_steps.
        # We use range(step=1) * num_steps instead of range(step=num_steps)
        # because step=0 (during prefill when draft_seq_len=0) is undefined.
        batch_size_plus_1 = ops.shape_to_tensor([prompt_offsets.shape[0]])[0]
        indices = ops.range(
            start=0,
            stop=batch_size_plus_1,
            out_dim=prompt_offsets.shape[0],
            device=device,
            dtype=DType.uint32,
        )
        draft_offsets = indices * K
        merged_tensor, merged_offsets = merge_ragged_tensors(
            prompt_tokens, prompt_offsets, draft_tokens_flattened, draft_offsets
        )

        return merged_tensor, merged_offsets


def compute_host_merged_offsets(
    host_input_row_offsets: TensorValue,
    draft_tokens: TensorValue,
) -> TensorValue:
    """Computes merged offsets on CPU, avoiding D2H copies.

    ``merged_offsets[i] = host_input_row_offsets[i] + i * K`` where ``K`` is
    the number of draft tokens per request. This mirrors the GPU-side merge
    logic in :class:`RaggedTokenMerger` but stays on CPU so CUDA graph capture
    is not blocked by a device-to-host transfer.
    """
    K = ops.shape_to_tensor([draft_tokens.shape[1]])[0].cast(DType.uint32)
    batch_size_plus_one = ops.shape_to_tensor(
        [host_input_row_offsets.shape[0]]
    )[0]
    indices = ops.range(
        start=0,
        stop=batch_size_plus_one,
        out_dim=host_input_row_offsets.shape[0],
        device=DeviceRef.CPU(),
        dtype=DType.uint32,
    )
    return host_input_row_offsets + indices * K
