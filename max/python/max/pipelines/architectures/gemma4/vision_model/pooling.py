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
"""Vision pooling for Gemma4."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from max.dtype import DType
from max.graph import TensorValue, ops
from max.nn.layer import Module


def avg_pool_by_positions(
    all_pixel_position_ids: list[npt.NDArray[np.integer]],
    output_lengths: list[int],
    k: int,
) -> npt.NDArray[np.floating]:
    """Compute sparse average-pooling weights for a ragged batch of images.

    Ports HuggingFace ``Gemma4VisionPooler._avg_pool_by_positions`` to NumPy,
    adapted for ragged (unpadded) multi-image inputs.  Returns a single
    block-diagonal weight matrix covering all images.

    Each real patch at grid position ``(x, y)`` contributes ``1/k**2`` to
    output bin ``(x // k) + (patch_width // k) * (y // k)``.  The
    resulting matrix has one block per image of shape
    ``[output_lengths[i], num_patches_i]``.

    Args:
        all_pixel_position_ids: Per-image list of integer ``(x, y)`` grid
            coordinates, each shaped ``[num_patches_i, 2]``.
        output_lengths: Number of output pooled tokens per image.
        k: Pooling kernel size (``pooling_kernel_size`` from config).

    Returns:
        Weight matrix of shape ``[total_output, total_patches]``, float32,
        where ``total_output = sum(output_lengths)`` and
        ``total_patches = sum(num_patches_i)``.
    """
    patch_counts = [pos.shape[0] for pos in all_pixel_position_ids]
    total_patches = sum(patch_counts)
    total_output = sum(output_lengths)

    weights = np.zeros((total_output, total_patches), dtype=np.float32)

    patch_offset = 0
    row_offset = 0
    inv_k2 = np.float32(1.0 / (k * k))

    for pos_ids, n_patches, n_output in zip(
        all_pixel_position_ids, patch_counts, output_lengths, strict=True
    ):
        max_x = int(pos_ids[:, 0].max()) + 1

        x_bin = pos_ids[:, 0] // k
        y_bin = pos_ids[:, 1] // k
        bin_idxs = x_bin + (max_x // k) * y_bin

        weights[
            row_offset + bin_idxs,
            patch_offset + np.arange(n_patches, dtype=np.intp),
        ] = inv_k2

        patch_offset += n_patches
        row_offset += n_output

    return weights


def compute_pool_gather_index(
    all_pixel_position_ids: list[npt.NDArray[np.integer]],
    output_lengths: list[int],
    k: int,
) -> npt.NDArray[np.int32]:
    """Build the per-output-bin gather index for average pooling.

    Returns a ``[total_output, max_per_bin]`` int32 table where row ``o`` lists
    the patch indices that pool into output token ``o``.  ``max_per_bin`` is the
    largest patch count of any bin (``k**2`` for image grids).  Unused slots are
    filled with the sentinel ``total_patches``, which the pooler maps to an
    appended zero row so they contribute nothing.

    Args:
        all_pixel_position_ids: Per-image list of integer ``(x, y)`` grid
            coordinates, each shaped ``[num_patches_i, 2]``.
        output_lengths: Number of output pooled tokens per image.
        k: Pooling kernel size (``pooling_kernel_size`` from config).

    Returns:
        Gather index, shape ``[sum(output_lengths), max_per_bin]``, int32.
    """
    patch_counts = [pos.shape[0] for pos in all_pixel_position_ids]
    total_patches = sum(patch_counts)
    total_output = sum(output_lengths)

    # For each patch, the global index of the output token it pools into.
    # The pooled cell within an image is (x//k, y//k); flatten it row-major to
    # a 1-D id (row stride max_x//k = pooled columns), then add row_offset to
    # make it global across the concatenated images.
    bins = np.empty(total_patches, dtype=np.int64)
    patch_offset = 0
    row_offset = 0
    for pos_ids, n_patches, n_output in zip(
        all_pixel_position_ids, patch_counts, output_lengths, strict=True
    ):
        max_x = int(pos_ids[:, 0].max()) + 1
        x_bin = pos_ids[:, 0] // k
        y_bin = pos_ids[:, 1] // k
        bins[patch_offset : patch_offset + n_patches] = row_offset + (
            x_bin + (max_x // k) * y_bin
        )
        patch_offset += n_patches
        row_offset += n_output

    # Count patches per bin. The table is rectangular, so its width must fit
    # the bin with the most patches (k**2 for images, variable for video).
    occupancy = np.bincount(bins, minlength=total_output)
    max_per_bin = int(occupancy.max()) if total_output else 1

    # Table pre-filled with the sentinel (the pooler's appended zero row) so
    # bins with fewer than max_per_bin patches stay padded.
    gather_index = np.full(
        (total_output, max_per_bin), total_patches, dtype=np.int32
    )
    # Stable-sort patches by bin so each bin's patches are contiguous and keep
    # input order, then place each at slot 0..count-1 within its bin's row.
    order = np.argsort(bins, kind="stable")  # patch indices, grouped by bin
    sorted_bins = bins[order]
    is_new = np.ones(total_patches, dtype=bool)
    is_new[1:] = sorted_bins[1:] != sorted_bins[:-1]  # first patch of each bin
    group_start = np.where(is_new)[0]  # sorted position where each bin starts
    group_id = np.cumsum(is_new) - 1  # which bin (0-based) each position is in
    slot = np.arange(total_patches) - group_start[group_id]  # within-bin slot
    gather_index[sorted_bins, slot] = order.astype(np.int32)

    return gather_index


class Gemma4VisionPooler(Module):
    """Position-based average pooling for the Gemma4 vision encoder.

    Reduces ``total_patches`` patch tokens to ``num_pooled_tokens`` by gathering
    each output bin's patches (see :func:`compute_pool_gather_index`), summing
    them, and scaling by ``1/k**2``.  No learnable parameters.
    """

    def __init__(self, hidden_size: int, pooling_kernel_size: int) -> None:
        super().__init__()
        self._scale: float = math.sqrt(hidden_size)
        # Fold the 1/k**2 averaging weight into the post-reduce scale.
        self._inv_k2: float = 1.0 / (pooling_kernel_size * pooling_kernel_size)

    def __call__(
        self,
        hidden_states: TensorValue,
        pool_gather_index: TensorValue,
    ) -> TensorValue:
        """Pool patch tokens by gathering each output bin's patches and summing.

        Args:
            hidden_states: Packed patch embeddings,
                shape ``[total_patches, hidden_size]``.
            pool_gather_index: Per-bin patch indices, shape
                ``[num_pooled_tokens, max_per_bin]``, dtype int32 (see
                :func:`compute_pool_gather_index`).  The sentinel value
                ``total_patches`` selects an appended zero row.

        Returns:
            Pooled embeddings, shape ``[num_pooled_tokens, hidden_size]``.
        """
        original_dtype = hidden_states.dtype
        hidden_states = hidden_states.cast(DType.float32)

        # Zero row selected by the gather sentinel, so under-filled bins add 0.
        zero_row = ops.broadcast_to(
            ops.constant(0.0, DType.float32, device=hidden_states.device),
            [1, hidden_states.shape[1]],
        )
        padded = ops.concat([hidden_states, zero_row], axis=0)

        # Gather each bin's patches -> [num_pooled, max_per_bin, hidden], sum.
        gathered = ops.gather(padded, pool_gather_index, axis=0)
        pooled = ops.squeeze(ops.sum(gathered, axis=1), 1)

        result = pooled * ops.constant(
            self._scale * self._inv_k2,
            DType.float32,
            device=hidden_states.device,
        )
        if original_dtype == DType.float16:
            return result
        return result.cast(original_dtype)
