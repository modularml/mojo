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

"""Tests for Gemma4Tokenizer — structural checks with mocked dependencies."""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, NonCallableMock

import numpy as np
import numpy.typing as npt
import pytest
from max.pipelines.architectures.gemma4.tokenizer import Gemma4Tokenizer
from max.pipelines.architectures.gemma4.video_processor import VideoMetadata
from max.pipelines.context import (
    SamplingParams,
    TextGenerationResponseFormat,
)
from max.pipelines.context.exceptions import PromptTooLongError
from max.pipelines.lib import KVCacheConfig, VisionPreprocessCache
from max.pipelines.modeling.types import (
    ImageContentPart,
    ReasoningPipelineTokenizer,
    RequestID,
    TextContentPart,
    TextGenerationRequest,
    TextGenerationRequestMessage,
    VideoContentPart,
)
from max.support.image import hash_image
from PIL import Image
from pytest_mock import MockerFixture

# Fake token IDs matching what the mock HF config exposes.
IMAGE_TOKEN_ID = 262144
BOI_TOKEN_ID = 255999
EOI_TOKEN_ID = 256000
VIDEO_TOKEN_ID = 256001  # assigned by add_special_tokens
EOS_TOKEN_ID = 1


def _create_mock_hf_config() -> NonCallableMock:
    """Create a mock HuggingFace config with Gemma4 vision attributes."""
    cfg = NonCallableMock()
    cfg.image_token_id = IMAGE_TOKEN_ID
    cfg.boi_token_id = BOI_TOKEN_ID
    cfg.eoi_token_id = EOI_TOKEN_ID
    cfg.eos_token_id = [EOS_TOKEN_ID, 106]
    return cfg


@pytest.fixture
def mock_pipeline_config() -> MagicMock:
    """Create a mock PipelineConfig for Gemma4 tests."""
    hf_config = _create_mock_hf_config()

    kv_cache_config = NonCallableMock(spec=KVCacheConfig)
    kv_cache_config.enable_prefix_caching = False

    model_config = MagicMock()
    model_config.huggingface_config = hf_config
    model_config.kv_cache = kv_cache_config

    runtime_config = MagicMock()
    runtime_config.vision_cache_utilization = 0.0
    runtime_config.max_vision_preprocess_cache_bytes = 0
    runtime_config.max_video_preprocess_cache_bytes = 0
    runtime_config.max_media_preprocess_cache_idle_seconds = 0.0

    pipeline_config = MagicMock()
    pipeline_config.model = model_config
    pipeline_config.runtime = runtime_config
    return pipeline_config


def _make_mock_delegate() -> MagicMock:
    """Build a mock AutoTokenizer delegate with Gemma4 special tokens."""
    delegate = MagicMock()
    delegate.eos_token_id = EOS_TOKEN_ID
    delegate.model_max_length = 4096
    delegate.image_token = "<image_soft_token>"
    delegate.boi_token = "<start_of_image>"
    delegate.eoi_token = "<end_of_image>"
    delegate.chat_template = None

    # convert_tokens_to_ids is called for <|video|>
    delegate.convert_tokens_to_ids.return_value = VIDEO_TOKEN_ID

    # decode fallback — not exercised when token attrs are set
    delegate.decode.side_effect = lambda ids: "".join(f"[tok_{t}]" for t in ids)

    return delegate


def _patch_tokenizer_deps(mocker: MockerFixture, delegate: MagicMock) -> None:
    """Patch external dependencies of Gemma4Tokenizer.__init__."""
    mocker.patch(
        "max.pipelines.architectures.gemma4.tokenizer.AutoTokenizer"
        ".from_pretrained",
        return_value=delegate,
    )
    mocker.patch(
        "max.pipelines.architectures.gemma4.tokenizer.load_processor_config",
        return_value={},
    )
    mocker.patch(
        "max.pipelines.architectures.gemma4.tokenizer.Gemma4ImageProcessor",
        return_value=MagicMock(
            return_value=([], [], []),
            pooling_kernel_size=3,
        ),
    )
    mocker.patch(
        "max.pipelines.architectures.gemma4.tokenizer.Gemma4VideoProcessor",
        return_value=MagicMock(
            return_value=([], [], [], []),
            pooling_kernel_size=3,
        ),
    )


def _image_key(tokenizer: Gemma4Tokenizer, raw_bytes: bytes) -> int:
    """The digest new_context computes for an image."""
    return hash_image(raw_bytes, tokenizer.img_processor.max_soft_tokens)


def _video_key(tokenizer: Gemma4Tokenizer, raw_bytes: bytes) -> int:
    """The digest new_context computes for a video."""
    return hash_image(raw_bytes, tokenizer._video_size_tier)


def _make_image_bytes(size: tuple[int, int] = (64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color="red").save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_only_smoke(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    """Text-only request: context is created, no image/video metadata."""
    delegate = _make_mock_delegate()
    text_tokens = np.array([2, 100, 200, 300, 3], dtype=np.int64)
    delegate.return_value = {"input_ids": [text_tokens.tolist()]}
    delegate.apply_chat_template.return_value = "Hello world"
    _patch_tokenizer_deps(mocker, delegate)

    tokenizer = Gemma4Tokenizer("test-model", mock_pipeline_config)

    request = TextGenerationRequest(
        messages=[
            TextGenerationRequestMessage(role="user", content="Hello world")
        ],
        request_id=RequestID("test-text"),
        model_name="test-model",
    )

    context = await tokenizer.new_context(request)

    assert context is not None
    assert len(context.images) == 0
    # All token_type_ids should be 0 (text)
    assert np.all(context.mm_token_type_ids == 0)


@pytest.mark.asyncio
async def test_response_format_without_json_schema_key(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    """``response_format`` without a ``json_schema`` value must yield
    ``json_schema=None`` on the context.
    """
    delegate = _make_mock_delegate()
    text_tokens = np.array([2, 100, 200, 300, 3], dtype=np.int64)
    delegate.return_value = {"input_ids": [text_tokens.tolist()]}
    delegate.apply_chat_template.return_value = "Hello world"
    _patch_tokenizer_deps(mocker, delegate)

    tokenizer = Gemma4Tokenizer("test-model", mock_pipeline_config)

    request = TextGenerationRequest(
        messages=[
            TextGenerationRequestMessage(role="user", content="Hello world")
        ],
        request_id=RequestID("test-rf-no-schema"),
        model_name="test-model",
        response_format=TextGenerationResponseFormat(type="json_object"),
    )

    context = await tokenizer.new_context(request)

    assert context.json_schema is None


@pytest.mark.asyncio
async def test_response_format_with_json_schema_key(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    """When ``response_format`` carries a real schema, it is serialized
    onto the context for the grammar compiler."""
    delegate = _make_mock_delegate()
    text_tokens = np.array([2, 100, 200, 300, 3], dtype=np.int64)
    delegate.return_value = {"input_ids": [text_tokens.tolist()]}
    delegate.apply_chat_template.return_value = "Hello world"
    _patch_tokenizer_deps(mocker, delegate)

    tokenizer = Gemma4Tokenizer("test-model", mock_pipeline_config)

    schema = {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
    }
    request = TextGenerationRequest(
        messages=[
            TextGenerationRequestMessage(role="user", content="Hello world")
        ],
        request_id=RequestID("test-rf-with-schema"),
        model_name="test-model",
        response_format=TextGenerationResponseFormat(
            type="json_schema",
            grammar=None,
            json_schema=schema,
            grammar_enforced=False,
            tools_forced=False,
        ),
    )

    context = await tokenizer.new_context(request)

    assert context.json_schema is not None
    assert json.loads(context.json_schema) == schema


@pytest.mark.asyncio
async def test_no_response_format(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    """Default request with no ``response_format`` yields ``json_schema=None``."""
    delegate = _make_mock_delegate()
    text_tokens = np.array([2, 100, 200, 300, 3], dtype=np.int64)
    delegate.return_value = {"input_ids": [text_tokens.tolist()]}
    delegate.apply_chat_template.return_value = "Hello world"
    _patch_tokenizer_deps(mocker, delegate)

    tokenizer = Gemma4Tokenizer("test-model", mock_pipeline_config)

    request = TextGenerationRequest(
        messages=[
            TextGenerationRequestMessage(role="user", content="Hello world")
        ],
        request_id=RequestID("test-no-rf"),
        model_name="test-model",
    )

    context = await tokenizer.new_context(request)

    assert context.json_schema is None


def test_gemma4_tokenizer_satisfies_reasoning_pipeline_tokenizer_protocol(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    """``Gemma4Tokenizer`` exposes ``reasoning_start_token_id`` and
    ``reasoning_end_token_id`` as instance attributes so it satisfies the
    ``ReasoningPipelineTokenizer`` ``@runtime_checkable`` ``Protocol``.

    This lets the overlap pipeline's thinking-mode temperature scaling
    resolve the per-model delimiter ids without depending on the reasoning
    parser registry.
    """
    delegate = _make_mock_delegate()
    # Distinct ids for <|channel> vs <channel|> so we can assert the values.

    def _convert(token: str) -> int:
        if token == "<|channel>":
            return 100
        if token == "<channel|>":
            return 101
        return VIDEO_TOKEN_ID

    delegate.convert_tokens_to_ids.side_effect = _convert
    _patch_tokenizer_deps(mocker, delegate)

    tokenizer = Gemma4Tokenizer("test-model", mock_pipeline_config)

    assert isinstance(tokenizer, ReasoningPipelineTokenizer)
    assert tokenizer.reasoning_start_token_id == 100
    assert tokenizer.reasoning_end_token_id == 101


@pytest.mark.asyncio
async def test_prompt_too_long(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    """Prompt exceeding max_length raises PromptTooLongError."""
    delegate = _make_mock_delegate()
    delegate.model_max_length = 5
    long_tokens = list(range(10))
    delegate.return_value = {"input_ids": [long_tokens]}
    delegate.apply_chat_template.return_value = "a very long prompt"
    _patch_tokenizer_deps(mocker, delegate)

    tokenizer = Gemma4Tokenizer("test-model", mock_pipeline_config)

    request = TextGenerationRequest(
        messages=[
            TextGenerationRequestMessage(
                role="user", content="a very long prompt"
            )
        ],
        request_id=RequestID("test-long"),
        model_name="test-model",
    )

    with pytest.raises(PromptTooLongError) as exc_info:
        await tokenizer.new_context(request)
    assert exc_info.value.num_tokens == 10
    assert exc_info.value.max_length == 5


@pytest.mark.asyncio
async def test_image_tokens_inserted(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    """Image request: image tokens appear in input_ids with correct
    token_type_ids and image metadata."""
    delegate = _make_mock_delegate()
    num_soft_tokens = 4

    # Simulate tokenized output after placeholder expansion:
    #   [BOS, ..text.., BOI, IMG, IMG, IMG, IMG, EOI, ..text.., EOS]
    input_ids = np.array(
        [2, 100, BOI_TOKEN_ID]
        + [IMAGE_TOKEN_ID] * num_soft_tokens
        + [EOI_TOKEN_ID, 200, 3],
        dtype=np.int64,
    )
    delegate.return_value = {"input_ids": [input_ids.tolist()]}
    # One placeholder per image; the tokenizer expands it to BOI+IMG*N+EOI
    delegate.apply_chat_template.return_value = (
        "Describe <image_soft_token> this."
    )
    _patch_tokenizer_deps(mocker, delegate)

    # Mock image processor to return one image with matching soft token count
    fake_pixels = np.zeros((num_soft_tokens * 9, 768), dtype=np.float32)
    fake_pos_ids = np.zeros((num_soft_tokens * 9, 2), dtype=np.int32)
    img_processor_mock = MagicMock(
        return_value=([fake_pixels], [fake_pos_ids], [num_soft_tokens]),
        pooling_kernel_size=3,
    )
    mocker.patch(
        "max.pipelines.architectures.gemma4.tokenizer.Gemma4ImageProcessor",
        return_value=img_processor_mock,
    )

    tokenizer = Gemma4Tokenizer("test-model", mock_pipeline_config)

    request = TextGenerationRequest(
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content=[
                    ImageContentPart(),
                    TextContentPart(text="Describe this."),
                ],
            )
        ],
        images=[_make_image_bytes()],
        request_id=RequestID("test-img"),
        model_name="test-model",
    )

    context = await tokenizer.new_context(request)

    tokens = context.tokens.all
    # Image tokens are present in the token sequence
    img_mask = tokens == IMAGE_TOKEN_ID
    assert img_mask.sum() == num_soft_tokens

    # token_type_ids mark image tokens as 1
    assert np.all(context.mm_token_type_ids[img_mask] == 1)
    # Non-image tokens remain 0
    assert np.all(context.mm_token_type_ids[~img_mask] == 0)

    # Image metadata is populated
    assert len(context.images) == 1
    meta = context.images[0]
    assert meta.pixel_values.shape == fake_pixels.shape
    assert meta.start_idx < meta.end_idx


@pytest.mark.asyncio
async def test_image_placement_multi_turn(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    """In a multi-turn conversation the image placeholder lands after earlier
    turns, not at the beginning."""
    delegate = _make_mock_delegate()

    # Chat template output with the image placeholder in the last user turn
    templated = (
        "<start_of_turn>user\nHello<end_of_turn>\n"
        "<start_of_turn>model\nHi!<end_of_turn>\n"
        "<start_of_turn>user\n\n\n<image_soft_token>\n\n"
        "What is this?<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    delegate.apply_chat_template.return_value = templated

    num_soft_tokens = 4

    # Build token IDs that mirror the expanded text structure
    pre_image = [2, 10, 11, 12, 13, 14]
    post_image = [15, 16, 17, 3]
    input_ids = np.array(
        pre_image
        + [BOI_TOKEN_ID]
        + [IMAGE_TOKEN_ID] * num_soft_tokens
        + [EOI_TOKEN_ID]
        + post_image,
        dtype=np.int64,
    )
    delegate.return_value = {"input_ids": [input_ids.tolist()]}

    _patch_tokenizer_deps(mocker, delegate)

    fake_pixels = np.zeros((num_soft_tokens * 9, 768), dtype=np.float32)
    fake_pos_ids = np.zeros((num_soft_tokens * 9, 2), dtype=np.int32)
    img_processor_mock = MagicMock(
        return_value=([fake_pixels], [fake_pos_ids], [num_soft_tokens]),
        pooling_kernel_size=3,
    )
    mocker.patch(
        "max.pipelines.architectures.gemma4.tokenizer.Gemma4ImageProcessor",
        return_value=img_processor_mock,
    )

    tokenizer = Gemma4Tokenizer("test-model", mock_pipeline_config)

    request = TextGenerationRequest(
        messages=[
            TextGenerationRequestMessage(role="user", content="Hello"),
            TextGenerationRequestMessage(role="assistant", content="Hi!"),
            TextGenerationRequestMessage(
                role="user",
                content=[
                    ImageContentPart(),
                    TextContentPart(text="What is this?"),
                ],
            ),
        ],
        images=[_make_image_bytes()],
        request_id=RequestID("test-multi"),
        model_name="test-model",
    )

    context = await tokenizer.new_context(request)

    tokens = context.tokens.all
    img_positions = np.where(tokens == IMAGE_TOKEN_ID)[0]
    assert len(img_positions) == num_soft_tokens
    # Image tokens should appear after the preamble, not at position 0
    assert img_positions[0] > 0

    # Verify the image block is contiguous
    assert np.all(np.diff(img_positions) == 1)


@pytest.mark.asyncio
async def test_video_tokens_inserted(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    """Video request: video tokens appear in input_ids with token_type_ids=2
    and video metadata is populated."""
    delegate = _make_mock_delegate()

    num_video_soft_tokens = 2
    num_frames = 2

    # Simulate tokenized output with video tokens
    input_ids = np.array(
        [2, 100]
        + [BOI_TOKEN_ID]
        + [VIDEO_TOKEN_ID] * num_video_soft_tokens
        + [EOI_TOKEN_ID]
        + [BOI_TOKEN_ID]
        + [VIDEO_TOKEN_ID] * num_video_soft_tokens
        + [EOI_TOKEN_ID]
        + [200, 3],
        dtype=np.int64,
    )
    delegate.return_value = {"input_ids": [input_ids.tolist()]}
    delegate.apply_chat_template.return_value = "Describe <|video|> this."

    _patch_tokenizer_deps(mocker, delegate)

    # Mock video processor to return frame data

    fake_frame_pv = np.zeros((num_frames, 36, 768), dtype=np.float32)
    fake_frame_pos = np.zeros((num_frames, 36, 2), dtype=np.int32)
    video_meta = VideoMetadata(
        timestamps=[0.0, 1.0],
    )

    video_processor_mock = MagicMock(
        return_value=(
            [fake_frame_pv],
            [fake_frame_pos],
            [num_video_soft_tokens],
            [video_meta],
        ),
        pooling_kernel_size=3,
    )
    mocker.patch(
        "max.pipelines.architectures.gemma4.tokenizer.Gemma4VideoProcessor",
        return_value=video_processor_mock,
    )

    tokenizer = Gemma4Tokenizer("test-model", mock_pipeline_config)

    request = TextGenerationRequest(
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content=[
                    VideoContentPart(),
                    TextContentPart(text="Describe this."),
                ],
            )
        ],
        videos=[_make_image_bytes()],
        request_id=RequestID("test-vid"),
        model_name="test-model",
    )

    context = await tokenizer.new_context(request)

    tokens = context.tokens.all
    vid_mask = tokens == VIDEO_TOKEN_ID
    assert vid_mask.sum() == num_video_soft_tokens * num_frames

    assert np.all(context.mm_token_type_ids[vid_mask] == 2)

    assert len(context.images) == num_frames
    assert len(context.pixel_position_ids) == num_frames
    for img in context.images:
        assert all(tokens[img.start_idx : img.end_idx] == VIDEO_TOKEN_ID)


@pytest.mark.asyncio
async def test_video_frames_get_per_frame_content_hashes(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    """With caching on, each frame entry carries a distinct non-zero hash."""
    mock_pipeline_config.model.kv_cache.enable_prefix_caching = True
    delegate = _make_mock_delegate()

    n_soft, n_frames = 2, 3
    frame_tokens = [
        tok
        for _ in range(n_frames)
        for tok in ([BOI_TOKEN_ID, *([VIDEO_TOKEN_ID] * n_soft), EOI_TOKEN_ID])
    ]
    input_ids = np.array(
        [2, 100, *frame_tokens, 3],
        dtype=np.int64,
    )
    delegate.return_value = {"input_ids": [input_ids.tolist()]}
    delegate.apply_chat_template.return_value = "Describe <|video|> this."
    _patch_tokenizer_deps(mocker, delegate)

    video_processor_mock = MagicMock(
        return_value=(
            [np.zeros((n_frames, 36, 768), dtype=np.float32)],
            [np.zeros((n_frames, 36, 2), dtype=np.int32)],
            [n_soft],
            [VideoMetadata(timestamps=[0.0, 1.0, 2.0])],
        ),
        pooling_kernel_size=3,
    )
    mocker.patch(
        "max.pipelines.architectures.gemma4.tokenizer.Gemma4VideoProcessor",
        return_value=video_processor_mock,
    )

    tokenizer = Gemma4Tokenizer("test-model", mock_pipeline_config)
    request = TextGenerationRequest(
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content=[VideoContentPart(), TextContentPart(text="hi")],
            )
        ],
        videos=[_make_image_bytes()],
        request_id=RequestID("test-vid-frame-hash"),
        model_name="test-model",
    )
    context = await tokenizer.new_context(request)

    hashes = [img.image_hash for img in context.images]
    assert len(hashes) == n_frames
    assert all(h is not None and h != 0 for h in hashes)  # content-derived
    assert len(set(hashes)) == n_frames  # distinct per frame (frame index)


@pytest.mark.asyncio
async def test_text_only_no_images_or_videos(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    """Pure text request has empty pixel_position_ids and video lists."""
    delegate = _make_mock_delegate()
    delegate.return_value = {"input_ids": [[2, 50, 51, 52, 3]]}
    delegate.apply_chat_template.return_value = "Just text"
    _patch_tokenizer_deps(mocker, delegate)

    tokenizer = Gemma4Tokenizer("test-model", mock_pipeline_config)

    request = TextGenerationRequest(
        messages=[
            TextGenerationRequestMessage(role="user", content="Just text")
        ],
        request_id=RequestID("test-no-mm"),
        model_name="test-model",
    )

    context = await tokenizer.new_context(request)

    assert len(context.pixel_position_ids) == 0
    assert len(context.images) == 0


@pytest.mark.asyncio
async def test_structured_output_json_schema(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    """Request with json_schema response_format sets context.json_schema."""
    delegate = _make_mock_delegate()
    delegate.return_value = {"input_ids": [[2, 100, 200, 3]]}
    delegate.apply_chat_template.return_value = "Extract name and age"
    _patch_tokenizer_deps(mocker, delegate)

    tokenizer = Gemma4Tokenizer("test-model", mock_pipeline_config)

    schema = {
        "title": "Person",
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
    }
    request = TextGenerationRequest(
        messages=[
            TextGenerationRequestMessage(
                role="user", content="Extract name and age"
            )
        ],
        request_id=RequestID("test-json-schema"),
        model_name="test-model",
        sampling_params=SamplingParams(max_new_tokens=50),
        response_format=TextGenerationResponseFormat(
            type="json_schema",
            json_schema=schema,
            grammar=None,
        ),
    )

    context = await tokenizer.new_context(request)

    assert context.json_schema is not None
    parsed = json.loads(context.json_schema)
    assert parsed["title"] == "Person"
    assert "name" in parsed["properties"]
    assert context.grammar is None


@pytest.mark.asyncio
async def test_structured_output_grammar(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    """Request with grammar response_format sets context.grammar."""
    delegate = _make_mock_delegate()
    delegate.return_value = {"input_ids": [[2, 100, 200, 3]]}
    delegate.apply_chat_template.return_value = "Generate output"
    _patch_tokenizer_deps(mocker, delegate)

    tokenizer = Gemma4Tokenizer("test-model", mock_pipeline_config)

    test_grammar = r'start: "yes" | "no"'
    request = TextGenerationRequest(
        messages=[
            TextGenerationRequestMessage(role="user", content="Generate output")
        ],
        request_id=RequestID("test-grammar"),
        model_name="test-model",
        sampling_params=SamplingParams(max_new_tokens=10),
        response_format=TextGenerationResponseFormat(
            type="grammar",
            json_schema={},
            grammar=test_grammar,
        ),
    )

    context = await tokenizer.new_context(request)

    assert context.grammar == test_grammar


@pytest.mark.asyncio
async def test_no_response_format_no_grammar(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    """Request without response_format has None for json_schema and grammar."""
    delegate = _make_mock_delegate()
    delegate.return_value = {"input_ids": [[2, 100, 200, 3]]}
    delegate.apply_chat_template.return_value = "Hello"
    _patch_tokenizer_deps(mocker, delegate)

    tokenizer = Gemma4Tokenizer("test-model", mock_pipeline_config)

    request = TextGenerationRequest(
        messages=[TextGenerationRequestMessage(role="user", content="Hello")],
        request_id=RequestID("test-no-fmt"),
        model_name="test-model",
    )

    context = await tokenizer.new_context(request)

    assert context.json_schema is None
    assert context.grammar is None


def test_apply_chat_template_prefills_reasoning_open_when_thinking(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    """With thinking enabled, the generation prompt is forced open with
    ``<|channel>thought`` so the model reasons on *every* assistant turn --
    including the turn after a ``tool`` result. Without it Gemma skips
    thinking post-tool and OpenRouter's reasoning-enabled-tool-call-step-5
    test fails, auto-disabling tools."""
    delegate = _make_mock_delegate()
    # Pretend the base template rendered a generation prompt that does NOT
    # open a reasoning block.
    delegate.apply_chat_template.return_value = "PROMPT<|model>"
    _patch_tokenizer_deps(mocker, delegate)

    tokenizer = Gemma4Tokenizer("test-model", mock_pipeline_config)
    msgs = [TextGenerationRequestMessage(role="user", content="hi")]

    # enable_thinking=True -> reasoning channel forced open at the tail.
    out = tokenizer.apply_chat_template(msgs, enable_thinking=True)
    assert out.endswith("<|channel>thought\n")

    # The tokenizer keys off ``enable_thinking`` only (matching the chat
    # template). OpenRouter's ``reasoning`` toggle is mapped to
    # ``enable_thinking`` upstream (#89137), so the bare ``thinking`` alias
    # alone does not force the channel open here.
    out_thinking_alias = tokenizer.apply_chat_template(msgs, thinking=True)
    assert "<|channel>thought" not in out_thinking_alias

    # Disabled -> no prefill.
    out_off = tokenizer.apply_chat_template(msgs, enable_thinking=False)
    assert "<|channel>thought" not in out_off


def test_apply_chat_template_reopens_turn_after_tool_response(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    """After a tool result the chat template leaves the model mid-turn, so the
    prompt ends with ``<tool_response|>``. The prefill must re-open a fresh
    model turn before the channel, matching the user-turn structure that
    actually reasons."""
    delegate = _make_mock_delegate()
    # Real post-tool render: turn left open, ends with the tool-response close.
    delegate.apply_chat_template.return_value = (
        "<|turn>model\n<|tool_call>call:f{}<tool_call|>"
        "<|tool_response>response:f{value:42}<tool_response|>"
    )
    _patch_tokenizer_deps(mocker, delegate)

    tokenizer = Gemma4Tokenizer("test-model", mock_pipeline_config)
    msgs = [TextGenerationRequestMessage(role="user", content="hi")]

    out = tokenizer.apply_chat_template(msgs, enable_thinking=True)

    # A fresh model turn is opened between the tool response and the channel,
    # mirroring the user-turn structure (`<|turn>model\n<|channel>thought\n`).
    assert out.endswith(
        "<tool_response|><turn|>\n<|turn>model\n<|channel>thought\n"
    )
    # The bare-channel-after-tool-response shape (which doesn't reason) is gone.
    assert not out.endswith("<tool_response|><|channel>thought\n")


def test_apply_chat_template_no_double_prefill(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    """If the base template already opened the reasoning channel, don't
    append a second opener."""
    delegate = _make_mock_delegate()
    delegate.apply_chat_template.return_value = "PROMPT<|channel>thought\n"
    _patch_tokenizer_deps(mocker, delegate)

    tokenizer = Gemma4Tokenizer("test-model", mock_pipeline_config)
    msgs = [TextGenerationRequestMessage(role="user", content="hi")]

    out = tokenizer.apply_chat_template(msgs, enable_thinking=True)
    assert out.count("<|channel>thought") == 1


@pytest.mark.asyncio
async def test_video_frame_count_mismatch_raises(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    """A prompt whose <video> runs don't match the processor's frame count
    raises (rather than silently indexing past the frame data)."""
    delegate = _make_mock_delegate()
    input_ids = np.array(
        [2, 100, BOI_TOKEN_ID, VIDEO_TOKEN_ID, VIDEO_TOKEN_ID, EOI_TOKEN_ID, 3],
        dtype=np.int64,
    )
    delegate.return_value = {"input_ids": [input_ids.tolist()]}
    delegate.apply_chat_template.return_value = "Describe <|video|> this."
    _patch_tokenizer_deps(mocker, delegate)

    video_processor_mock = MagicMock(
        return_value=(
            [np.zeros((2, 36, 768), dtype=np.float32)],
            [np.zeros((2, 36, 2), dtype=np.int32)],
            [2],
            [VideoMetadata(timestamps=[0.0, 1.0])],
        ),
        pooling_kernel_size=3,
    )
    mocker.patch(
        "max.pipelines.architectures.gemma4.tokenizer.Gemma4VideoProcessor",
        return_value=video_processor_mock,
    )

    tokenizer = Gemma4Tokenizer("test-model", mock_pipeline_config)
    request = TextGenerationRequest(
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content=[VideoContentPart(), TextContentPart(text="hi")],
            )
        ],
        videos=[_make_image_bytes()],
        request_id=RequestID("test-vid-mismatch"),
        model_name="test-model",
    )
    with pytest.raises(ValueError, match="Video placeholder mismatch"):
        await tokenizer.new_context(request)


# ---------------------------------------------------------------------------
# Preprocessed-image cache
# ---------------------------------------------------------------------------


def _counting_image_processor() -> MagicMock:
    """A ``Gemma4ImageProcessor`` stand-in that counts preprocessing calls.

    Each call returns a distinct fill value, so a cached result is
    distinguishable from a freshly preprocessed one.
    """
    calls = 0

    def process(
        images: list[Image.Image],
    ) -> tuple[
        list[npt.NDArray[np.float32]],
        list[npt.NDArray[np.int32]],
        list[int],
    ]:
        nonlocal calls
        calls += 1
        return (
            [np.full((4, 3), float(calls), dtype=np.float32) for _ in images],
            [np.zeros((4, 2), dtype=np.int32) for _ in images],
            [7] * len(images),
        )

    processor = MagicMock(side_effect=process)
    processor.max_soft_tokens = 280
    processor.pooling_kernel_size = 3
    return processor


def _tokenizer_with_processor(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
    processor: MagicMock,
) -> Gemma4Tokenizer:
    _patch_tokenizer_deps(mocker, _make_mock_delegate())
    mocker.patch(
        "max.pipelines.architectures.gemma4.tokenizer.Gemma4ImageProcessor",
        return_value=processor,
    )
    return Gemma4Tokenizer("test-model", mock_pipeline_config)


def test_preprocessing_repeats_when_the_cache_is_disabled(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    """The default configuration preserves today's behavior exactly."""
    processor = _counting_image_processor()
    tokenizer = _tokenizer_with_processor(
        mocker, mock_pipeline_config, processor
    )
    image = _make_image_bytes()

    key = _image_key(tokenizer, image)
    first = tokenizer._preprocess_image(key, image)
    second = tokenizer._preprocess_image(key, image)

    assert processor.call_count == 2
    # Distinct fill values confirm the second call really reprocessed.
    assert first[0][0][0] == 1.0
    assert second[0][0][0] == 2.0


def test_repeated_image_is_preprocessed_once(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    mock_pipeline_config.runtime.max_vision_preprocess_cache_bytes = 1 << 20
    processor = _counting_image_processor()
    tokenizer = _tokenizer_with_processor(
        mocker, mock_pipeline_config, processor
    )
    image = _make_image_bytes()

    key = _image_key(tokenizer, image)
    first = tokenizer._preprocess_image(key, image)
    second = tokenizer._preprocess_image(key, image)

    assert processor.call_count == 1
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2] == second[2] == 7


def test_distinct_images_do_not_share_a_cache_entry(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    mock_pipeline_config.runtime.max_vision_preprocess_cache_bytes = 1 << 20
    processor = _counting_image_processor()
    tokenizer = _tokenizer_with_processor(
        mocker, mock_pipeline_config, processor
    )
    first_image = _make_image_bytes((64, 64))
    second_image = _make_image_bytes((32, 48))

    tokenizer._preprocess_image(_image_key(tokenizer, first_image), first_image)
    tokenizer._preprocess_image(
        _image_key(tokenizer, second_image), second_image
    )

    assert processor.call_count == 2


def test_cached_arrays_are_frozen_against_in_place_writes(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    """A shared array must not be mutable by one of its many borrowers."""
    mock_pipeline_config.runtime.max_vision_preprocess_cache_bytes = 1 << 20
    processor = _counting_image_processor()
    tokenizer = _tokenizer_with_processor(
        mocker, mock_pipeline_config, processor
    )
    image = _make_image_bytes()

    pixels, position_ids, _ = tokenizer._preprocess_image(
        _image_key(tokenizer, image), image
    )

    with pytest.raises(ValueError):
        pixels[0][0] = 1.0
    with pytest.raises(ValueError):
        position_ids[0][0] = 1


def test_eviction_under_budget_forces_reprocessing(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    # Each entry is 4*3*4 + 4*2*4 = 80 bytes, so a 100-byte budget holds one.
    mock_pipeline_config.runtime.max_vision_preprocess_cache_bytes = 100
    processor = _counting_image_processor()
    tokenizer = _tokenizer_with_processor(
        mocker, mock_pipeline_config, processor
    )
    first_image = _make_image_bytes((64, 64))
    second_image = _make_image_bytes((32, 48))

    tokenizer._preprocess_image(_image_key(tokenizer, first_image), first_image)
    tokenizer._preprocess_image(
        _image_key(tokenizer, second_image), second_image
    )
    tokenizer._preprocess_image(_image_key(tokenizer, first_image), first_image)

    assert processor.call_count == 3


def test_both_caches_take_the_configured_idle_timeout(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    """Reclaim-on-idle is configured once and reaches both media kinds."""
    mock_pipeline_config.runtime.max_vision_preprocess_cache_bytes = 1 << 20
    mock_pipeline_config.runtime.max_video_preprocess_cache_bytes = 1 << 20
    mock_pipeline_config.runtime.max_media_preprocess_cache_idle_seconds = 60.0
    tokenizer = _tokenizer_with_processor(
        mocker, mock_pipeline_config, _counting_image_processor()
    )

    assert tokenizer._preprocess_cache.idle_seconds == 60.0
    assert tokenizer._video_preprocess_cache.idle_seconds == 60.0


def test_an_idle_image_is_reclaimed_and_reprocessed(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    """A conversation that stops resending an image stops costing host memory.

    The budget is a ceiling on size, not a bound on how long an entry lives, so
    without this a burst of distinct images holds its whole resident set for
    the rest of the process's life.
    """
    mock_pipeline_config.runtime.max_vision_preprocess_cache_bytes = 1 << 20
    processor = _counting_image_processor()
    tokenizer = _tokenizer_with_processor(
        mocker, mock_pipeline_config, processor
    )
    # A hand-advanced clock, so the entry can age a minute without the test
    # waiting one.
    now = 1000.0
    tokenizer._preprocess_cache = VisionPreprocessCache(
        1 << 20, idle_seconds=60.0, clock=lambda: now
    )
    image = _make_image_bytes()
    key = _image_key(tokenizer, image)

    tokenizer._preprocess_image(key, image)
    assert tokenizer._preprocess_cache.total_bytes > 0

    now += 61.0
    assert tokenizer._preprocess_cache.collect() > 0
    assert tokenizer._preprocess_cache.total_bytes == 0

    tokenizer._preprocess_image(key, image)

    assert processor.call_count == 2


# ---------------------------------------------------------------------------
# Preprocessed-video cache
# ---------------------------------------------------------------------------


def _counting_video_processor() -> MagicMock:
    """A ``Gemma4VideoProcessor`` stand-in that counts preprocessing calls."""
    calls = 0

    def process(
        videos: list[bytes],
    ) -> tuple[
        list[npt.NDArray[np.float32]],
        list[npt.NDArray[np.int32]],
        list[int],
        list[VideoMetadata],
    ]:
        nonlocal calls
        calls += 1
        return (
            [
                np.full((2, 6, 3), float(calls), dtype=np.float32)
                for _ in videos
            ],
            [np.zeros((2, 6, 2), dtype=np.int32) for _ in videos],
            [2] * len(videos),
            [VideoMetadata(fps=30.0, timestamps=[0.0, 1.0])] * len(videos),
        )

    processor = MagicMock(side_effect=process)
    processor.max_soft_tokens = 70
    processor.num_frames = 32
    processor.pooling_kernel_size = 3
    return processor


def _tokenizer_with_video_processor(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
    processor: MagicMock,
) -> Gemma4Tokenizer:
    _patch_tokenizer_deps(mocker, _make_mock_delegate())
    mocker.patch(
        "max.pipelines.architectures.gemma4.tokenizer.Gemma4VideoProcessor",
        return_value=processor,
    )
    return Gemma4Tokenizer("test-model", mock_pipeline_config)


def test_video_preprocessing_repeats_when_the_cache_is_disabled(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    processor = _counting_video_processor()
    tokenizer = _tokenizer_with_video_processor(
        mocker, mock_pipeline_config, processor
    )
    video = b"fake-video-bytes"

    key = _video_key(tokenizer, video)
    first = tokenizer._preprocess_video(key, video)
    second = tokenizer._preprocess_video(key, video)

    assert processor.call_count == 2
    assert first[0][0][0][0] == 1.0
    assert second[0][0][0][0] == 2.0


def test_repeated_video_is_preprocessed_once(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    mock_pipeline_config.runtime.max_video_preprocess_cache_bytes = 1 << 20
    processor = _counting_video_processor()
    tokenizer = _tokenizer_with_video_processor(
        mocker, mock_pipeline_config, processor
    )
    video = b"fake-video-bytes"

    key = _video_key(tokenizer, video)
    first = tokenizer._preprocess_video(key, video)
    second = tokenizer._preprocess_video(key, video)

    assert processor.call_count == 1
    np.testing.assert_array_equal(first[0], second[0])
    assert first[2] == second[2] == 2
    assert first[3].timestamps == [0.0, 1.0]


def test_distinct_videos_do_not_share_a_cache_entry(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    mock_pipeline_config.runtime.max_video_preprocess_cache_bytes = 1 << 20
    processor = _counting_video_processor()
    tokenizer = _tokenizer_with_video_processor(
        mocker, mock_pipeline_config, processor
    )

    tokenizer._preprocess_video(
        _video_key(tokenizer, b"video-one"), b"video-one"
    )
    tokenizer._preprocess_video(
        _video_key(tokenizer, b"video-two"), b"video-two"
    )

    assert processor.call_count == 2


def test_cached_video_arrays_are_frozen(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    mock_pipeline_config.runtime.max_video_preprocess_cache_bytes = 1 << 20
    processor = _counting_video_processor()
    tokenizer = _tokenizer_with_video_processor(
        mocker, mock_pipeline_config, processor
    )

    pixels, position_ids, _, _ = tokenizer._preprocess_video(
        _video_key(tokenizer, b"a-video"), b"a-video"
    )

    with pytest.raises(ValueError):
        pixels[0][0][0] = 1.0
    with pytest.raises(ValueError):
        position_ids[0][0][0] = 1


def test_video_and_image_caches_have_separate_budgets(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    """A video must not be able to evict images, or vice versa."""
    mock_pipeline_config.runtime.max_vision_preprocess_cache_bytes = 1 << 20
    mock_pipeline_config.runtime.max_video_preprocess_cache_bytes = 0
    processor = _counting_video_processor()
    tokenizer = _tokenizer_with_video_processor(
        mocker, mock_pipeline_config, processor
    )

    assert tokenizer._preprocess_cache.enabled
    assert not tokenizer._video_preprocess_cache.enabled


@pytest.mark.asyncio
async def test_image_bytes_are_hashed_once_per_request(
    mocker: MockerFixture,
    mock_pipeline_config: MagicMock,
) -> None:
    """The preprocess cache and the vision-cache key share one digest.

    Both are keyed on the same content hash, so computing it separately in
    each place would hash every image's bytes twice per request -- which on a
    preprocess-cache hit is most of the work that remains.
    """
    mock_pipeline_config.model.kv_cache.enable_prefix_caching = True
    mock_pipeline_config.runtime.max_vision_preprocess_cache_bytes = 1 << 20

    delegate = _make_mock_delegate()
    num_soft_tokens = 4
    input_ids = np.array(
        [2, 100, BOI_TOKEN_ID]
        + [IMAGE_TOKEN_ID] * num_soft_tokens
        + [EOI_TOKEN_ID, 200, 3],
        dtype=np.int64,
    )
    delegate.return_value = {"input_ids": [input_ids.tolist()]}
    delegate.apply_chat_template.return_value = (
        "Describe <image_soft_token> this."
    )
    _patch_tokenizer_deps(mocker, delegate)
    mocker.patch(
        "max.pipelines.architectures.gemma4.tokenizer.Gemma4ImageProcessor",
        return_value=MagicMock(
            return_value=(
                [np.zeros((num_soft_tokens * 9, 768), dtype=np.float32)],
                [np.zeros((num_soft_tokens * 9, 2), dtype=np.int32)],
                [num_soft_tokens],
            ),
            pooling_kernel_size=3,
            max_soft_tokens=280,
        ),
    )
    spy = mocker.patch(
        "max.pipelines.architectures.gemma4.tokenizer.hash_image",
        side_effect=hash_image,
    )

    tokenizer = Gemma4Tokenizer("test-model", mock_pipeline_config)
    image_bytes = _make_image_bytes()
    request = TextGenerationRequest(
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content=[
                    ImageContentPart(),
                    TextContentPart(text="Describe this."),
                ],
            )
        ],
        images=[image_bytes],
        request_id=RequestID("test-hash-once"),
        model_name="test-model",
    )

    context = await tokenizer.new_context(request)

    # One digest, reused for both the preprocess cache and the context.
    assert spy.call_count == 1
    assert context.images[0].image_hash == hash_image(image_bytes, 280)
