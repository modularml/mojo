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

"""Tests for Idefics3 tokenizer."""

import io

import pytest
from max.pipelines.architectures.idefics3.tokenizer import Idefics3Tokenizer
from max.pipelines.modeling.types import (
    ImageContentPart,
    RequestID,
    TextContentPart,
    TextGenerationRequest,
    TextGenerationRequestMessage,
)
from PIL import Image
from test_common.mocks import DummyPipelineConfig

IDEFICS3_REPO_ID = "HuggingFaceM4/Idefics3-8B-Llama3"


@pytest.mark.asyncio
async def test_idefics3_tokenizer_image_token_indices() -> None:
    """Test that the tokenizer correctly computes image token indices."""
    pipeline_config = DummyPipelineConfig(
        model_path=IDEFICS3_REPO_ID,
        max_batch_size=None,
        max_length=None,
        quantization_encoding="float32",
    )
    # DummyPipelineConfig seeds a MagicMock HuggingFace config; set the
    # `image_token_id` that Idefics3Tokenizer reads.
    assert pipeline_config.model.huggingface_config is not None
    pipeline_config.model.huggingface_config.image_token_id = 128257
    tokenizer = Idefics3Tokenizer(
        IDEFICS3_REPO_ID,
        pipeline_config=pipeline_config,
    )
    assert tokenizer.vision_token_ids == [128257]
    assert tokenizer.enable_prefix_caching is True

    # Create a real image for the test using config dimensions
    img_buffer = io.BytesIO()
    image_size = 448
    Image.new("RGB", (image_size, image_size), color="red").save(
        img_buffer, format="PNG"
    )
    test_image = img_buffer.getvalue()

    # Create request with image
    request = TextGenerationRequest(
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content=[
                    TextContentPart(text="test"),
                    ImageContentPart(),
                ],
            )
        ],
        images=[test_image],
        request_id=RequestID("test-id"),
        model_name=IDEFICS3_REPO_ID,
    )

    context = await tokenizer.new_context(request)
    assert context is not None
    # Idefics3 tokenizer turns this single image into 17 patch groups
    assert len(context.images) == 17
