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


"""Test compute_multimodal_merge_indices function."""

import copy
from unittest.mock import Mock

import numpy as np
from max.pipelines.architectures.qwen2_5vl.context import (
    Qwen2_5VLTextAndVisionContext,
)
from max.pipelines.context import (
    ImageMetadata,
    TokenBuffer,
)
from max.pipelines.lib.vlm_utils import compute_multimodal_merge_indices


def test_compute_multimodal_merge_indices() -> None:
    # Sentinel OOB index value
    OOB = np.iinfo(np.int32).min

    # Image token ID
    IMG = 99

    # These pixel values are arbitrary
    img0 = np.array([[-1, -2], [-3, -4]])
    img1 = np.array([[-5, -6], [-7, -8]])
    tokens = np.array(
        [0, 1, 2, 3, IMG, IMG, IMG, IMG, 8, 9, IMG, IMG, IMG, IMG, IMG, 15]
    )
    ctx = Qwen2_5VLTextAndVisionContext(
        max_length=50,
        tokens=TokenBuffer(tokens),
        images=[
            ImageMetadata(
                start_idx=4,
                end_idx=8,
                pixel_values=img0,
            ),
            ImageMetadata(
                start_idx=10,
                end_idx=15,
                pixel_values=img1,
            ),
        ],
        vision_token_ids=[IMG],
        # Qwen2.5VL-specific required fields
        image_token_indices=np.array(
            [4, 5, 6, 7, 10, 11, 12, 13, 14], dtype=np.int32
        ),
        spatial_merge_size=Mock(),
        rope_delta=Mock(),
        image_token_id=Mock(),
        video_token_id=Mock(),
        vision_start_token_id=Mock(),
        tokens_per_second=Mock(),
        decoder_position_ids=Mock(),
        vision_data=None,
    )

    # fmt: off

    # Check that the image token indices are correct
    precomputed = ctx.image_token_indices
    assert (precomputed == np.where(ctx.tokens.all == IMG)[0]).all()

    # Test normal case: start_idx = 0
    image_token_indices = compute_multimodal_merge_indices([ctx])
    # 9 img tokens (img0 + img1)
    assert image_token_indices.tolist() == [4, 5, 6, 7, 10, 11, 12, 13, 14]

    # Test prefix cache hit case: start_idx = 8
    ctx.tokens.skip_processing(8)
    image_token_indices = compute_multimodal_merge_indices([ctx])
    # 5 img tokens (img1)
    # 0-3 are skipped as img0 is not included
    assert image_token_indices.tolist() == [OOB, OOB, OOB, OOB, 2, 3, 4, 5, 6]

    # Test multiple contexts case
    # ctx0 (start_idx=0), ctx1 (start_idx=8)
    ctx0 = copy.deepcopy(ctx)
    ctx1 = copy.deepcopy(ctx)
    ctx0.tokens.rewind_processing(ctx0.tokens.processed_length)
    image_token_indices = compute_multimodal_merge_indices(
        [ctx0, ctx1]
    )
    # 9 (img0 + img1) + 5 (img1) = 14 img tokens
    # 9-12 are skipped as img0 of ctx1 is not included
    assert image_token_indices.tolist() == [4, 5, 6, 7, 10, 11, 12, 13, 14, OOB, OOB, OOB, OOB, 18, 19, 20, 21, 22]

    # Test empty case
    image_token_indices = compute_multimodal_merge_indices([])
    assert image_token_indices.dtype == np.int32
    assert image_token_indices.tolist() == []
