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

"""Shared utilities for vision-language model (VLM) architectures."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from max.experimental import functional as F
from max.experimental.sharding.rules import ternary_rule
from max.graph import TensorValue, ops
from max.nn.kernels import scatter_nd_skip_oob_indices
from max.pipelines.context import TextAndVisionContext


def merge_multimodal_embeddings(
    inputs_embeds: TensorValue,
    multimodal_embeddings: TensorValue,
    image_token_indices: TensorValue,
) -> TensorValue:
    """Merges multimodal embeddings into text embeddings at pre-computed indices.

    This is the MAX Graph API implementation of the embedding merge operation.
    It returns an updated copy of inputs_embeds with multimodal embeddings
    at positions specified by the indices.

    Indices may be oob (out of bounds), in which case the corresponding update will be skipped.

    Args:
        inputs_embeds: Text embeddings with shape [num_tokens, hidden_size].
        multimodal_embeddings: Vision embeddings to insert with shape
            [num_multimodal_tokens, hidden_size].
        image_token_indices: Pre-computed indices where to insert multimodal embeddings,
            with shape [num_multimodal_tokens].

    Returns:
        Copy of the inputs_embeds tensor with multimodal embeddings merged in.
    """
    # Use scatter_nd_skip_oob_indices to directly place embeddings at specified indices.
    # Expand indices to 2D for scatter_nd_skip_oob_indices: [num_tokens, 1]
    indices_2d = ops.unsqueeze(image_token_indices, -1)

    if multimodal_embeddings.dtype != inputs_embeds.dtype:
        multimodal_embeddings = ops.cast(
            multimodal_embeddings, dtype=inputs_embeds.dtype
        )

    # Scatter the multimodal embeddings into inputs_embeds at the specified
    # indices. Any negative values in the indices means that the corresponding
    # update will be skipped.
    return scatter_nd_skip_oob_indices(
        input=inputs_embeds,
        updates=multimodal_embeddings,
        indices=indices_2d,
    )


# TODO(MXF-336): Use @F.functional decorator as the main entrypoint, instead of
# creating a separate `F_` function, once all models are using ModuleV3.
F_merge_multimodal_embeddings = F.functional(
    merge_multimodal_embeddings, rule=ternary_rule
)


def compute_windowed_merge_indices(
    batch: Sequence[TextAndVisionContext],
) -> npt.NDArray[np.int32]:
    """Compute dense scatter indices for image rows in each active window.

    Emits one index per image-placeholder token inside its context's
    window ``[processed_length, current_position)``, ordered to match the
    window-bounded assembly, with no out-of-bounds sentinels. Use
    :func:`compute_multimodal_merge_indices` when the paired embeddings
    cover every image row of the batch.

    Args:
        batch: Sequence of VLM contexts.

    Returns:
        npt.NDArray[np.int32]: Dense multimodal merge indices.
    """
    parts: list[npt.NDArray[np.int32]] = []
    total_active_tokens = 0

    for ctx in batch:
        if getattr(ctx, "needs_vision_encoding", False):
            relative = (
                np.asarray(ctx.image_token_indices, dtype=np.int64)
                - ctx.tokens.processed_length
            )
            in_window = (relative >= 0) & (relative < ctx.tokens.active_length)
            if in_window.any():
                parts.append(
                    (relative[in_window] + total_active_tokens).astype(np.int32)
                )
        total_active_tokens += ctx.tokens.active_length

    if not parts:
        return np.array([], dtype=np.int32)
    return np.concatenate(parts)


def compute_multimodal_merge_indices(
    batch: Sequence[TextAndVisionContext],
) -> npt.NDArray[np.int32]:
    """Compute scatter indices for merging vision embeddings into text embeddings.

    Accounts for ``processed_length`` so that indices from prior chunks are
    mapped to an out-of-bounds sentinel and only the indices falling within the
    current active window receive valid offsets.

    Args:
        batch: Sequence of VLM contexts.

    Returns:
        npt.NDArray[np.int32]: Multimodal merge indices, some of which may be negative.
    """
    # Calculate sentinel OOB index value by finding the largest negative int32 value.
    oob_idx = np.iinfo(np.int32).min

    # Collect indices and offsets.
    indices_list = []
    total_active_tokens = 0

    for ctx in batch:
        if getattr(ctx, "needs_vision_encoding", False):
            # This logic is quite tricky but is required for VLM prefix caching.
            # In the current approach, we run image decoding on all images.
            # We then select the rows of the image embeddings we want to use.
            # This may not be all of the rows in the event of a prefix cache
            # hit. This is done via a multimodal merge operation which filters
            # out negative indices.

            # First, get the pre-computed indices of where the image placeholder
            # tokens are in the prompt. This is populated by tokenizer.
            # eg: prompt = [0, 1, 2, 3, IMG, IMG, IMG, IMG, 8, 9, IMG, IMG]
            #    indices = [4, 5, 6, 7, 10, 11]
            indices = ctx.image_token_indices

            # Subtract all of the indices by the start_idx to get offsets
            # relative to the ragged next_tokens input sequence.
            # eg: start_idx = 6
            #     indices = [-2, -1, 0, 1, 4, 5]
            indices = indices - ctx.tokens.processed_length

            # Keep only indices within the current active window.
            # Negative indices (already processed) and indices beyond
            # the active length (not yet in this chunk) are mapped to
            # OOB. Valid indices are bumped by the accumulated offset
            # for the batch.
            active_len = ctx.tokens.active_length
            indices_filtered = [
                idx + total_active_tokens if 0 <= idx < active_len else oob_idx
                for idx in indices.tolist()
            ]

            # Final scatter indices assuming the batch has 10 image tokens so far.
            # eg: indices_filtered = [-999, -999, 10, 11, 14, 15]
            #     This means that we will copy 4 image embeddings to the rows
            #     10-11 and 14-15 of the text embeddings.
            indices_list.append(indices_filtered)

        total_active_tokens += ctx.tokens.active_length

    # scatter_nd_skip_oob_indices uses int32 indices.
    if indices_list:
        flat = [idx for chunk in indices_list for idx in chunk]
        return np.array(flat, dtype=np.int32)
    else:
        return np.array([], dtype=np.int32)
