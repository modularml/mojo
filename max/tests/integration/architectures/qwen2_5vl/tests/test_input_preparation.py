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

"""Tests for Qwen2.5VL input preparation with decoder position IDs after context reset."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, Mock, NonCallableMock

import numpy as np
import pytest
from max.nn.kv_cache import KVCacheInputs
from max.nn.parallel import ParallelArrayOps
from max.pipelines.architectures.qwen2_5vl.batch_processor import (
    Qwen2_5VLBatchProcessor,
)
from max.pipelines.architectures.qwen2_5vl.context import (
    Qwen2_5VLTextAndVisionContext,
)
from max.pipelines.architectures.qwen2_5vl.model import Qwen2_5VLInputs
from max.pipelines.architectures.qwen2_5vl.tokenizer import Qwen2_5VLTokenizer
from max.pipelines.context import (
    EOSTracker,
    SamplingParams,
    TextAndVisionContext,
    TokenBuffer,
)
from max.pipelines.lib import KVCacheConfig, PipelineRuntimeConfig
from max.pipelines.modeling.types import (
    ImageContentPart,
    RequestID,
    TextContentPart,
    TextGenerationRequest,
    TextGenerationRequestMessage,
)
from pytest_mock import MockerFixture
from transformers import Qwen2_5_VLConfig


def create_mock_qwen_batch_processor(
    mocker: MockerFixture,
) -> Qwen2_5VLBatchProcessor:
    """Create a minimal mock batch processor instance for testing.

    This mocks dependencies to avoid loading actual weights and needing
    GPU/device setup, while keeping the actual prepare_initial_token_inputs
    logic intact.

    Args:
        mocker: pytest-mock fixture for patching

    Returns:
        Mocked Qwen2_5VLBatchProcessor instance ready for testing
    """

    def mock_tensor_from_numpy(arr: np.ndarray) -> Mock:
        """Mock Tensor.from_numpy to return an object with .to() and .to_numpy() methods.

        Args:
            arr: NumPy array to wrap in a mock Tensor

        Returns:
            Mock object simulating a Tensor with .to() and .to_numpy() methods
        """

        def create_device_tensor(data: np.ndarray) -> Mock:
            tensor = Mock()
            tensor.to_numpy = Mock(return_value=data)
            return tensor

        mock_tensor = create_device_tensor(arr)

        def mock_to(devices: Any) -> Mock | list[Mock]:
            if isinstance(devices, list):
                return [create_device_tensor(arr) for _ in devices]
            return mock_tensor

        mock_tensor.to = mock_to
        return mock_tensor

    mocker.patch(
        "max.driver.Buffer.from_numpy", side_effect=mock_tensor_from_numpy
    )
    mocker.patch("max.driver.Buffer.zeros", return_value=Mock())

    # Import after patching

    # Create minimal mock config with REAL integer values for calculations
    mock_config = Mock()
    mock_config.vision_config = Mock()
    mock_config.text_config = Mock()
    mock_config.text_config.max_position_embeddings = (
        32768  # Real value for Qwen2.5VL-3B
    )
    mock_config.text_config.hidden_size = 2048

    processor = Qwen2_5VLBatchProcessor.__new__(Qwen2_5VLBatchProcessor)

    mock_device = Mock()
    mock_pipeline_config = Mock()
    mock_pipeline_config.model.huggingface_config = mock_config
    processor.runtime = Mock()
    processor.runtime.devices = [mock_device]
    processor.runtime.signal_buffers = [Mock()]
    processor.runtime.pipeline_config = mock_pipeline_config
    processor._parallel_ops = ParallelArrayOps(max_workers=1)

    def mock_batch_image_token_indices(
        self: Qwen2_5VLBatchProcessor,
        context_batch: list[TextAndVisionContext],
    ) -> list[Mock]:
        """Return empty indices for scenarios without actual image processing.

        Args:
            self: The batch processor instance (required for instance method)
            context_batch: List of contexts to process

        Returns:
            Per-device scatter index buffers as mock tensor lists
        """
        mock_scatter = Mock()
        mock_scatter.to = Mock(return_value=mock_scatter)
        mock_scatter.shape = (0,)  # Empty
        return [mock_scatter]

    mocker.patch.object(
        Qwen2_5VLBatchProcessor,
        "_batch_image_token_indices",
        mock_batch_image_token_indices,
    )

    return processor


# ============================================================================
# Pytest Fixtures for Common Mock Setup
# ============================================================================


@pytest.fixture
def qwen_token_ids() -> dict[str, int]:
    """Common token IDs used in Qwen2.5VL tests."""
    return {
        "image_token_id": 151652,
        "video_token_id": 151653,
        "vision_start_token_id": 151655,
    }


@pytest.fixture
def round_trip_tokenizer_mock(
    mocker: MockerFixture,
) -> Callable[[list[int]], MagicMock]:
    """Create a mock tokenizer with consistent encode/decode round-trips.

    Returns a factory function that creates a tokenizer mock with the given
    base input_ids. The mock maintains bidirectional mapping between tokens
    and text to ensure decode(encode(tokens)) returns the same tokens.

    Args:
        base_input_ids: Default token IDs to use for new text inputs

    Returns:
        Factory function that takes base_input_ids and returns configured mock tokenizer
    """

    def _create_tokenizer(base_input_ids: list[int]) -> MagicMock:
        mock_tok = MagicMock()
        mock_tok.model_max_length = 2048
        mock_tok.eos_token_id = 2

        # Bidirectional mapping for consistent round-trips
        token_to_text_map: dict[tuple[int, ...], str] = {}
        text_to_token_map: dict[str, list[int]] = {}

        def mock_decode(
            token_ids: list[int], skip_special_tokens: bool = False
        ) -> str:
            """Mock decode that maintains consistent round-trip behavior.

            Args:
                token_ids: List of token IDs to decode
                skip_special_tokens: Whether to skip special tokens (unused in mock)

            Returns:
                Decoded text string
            """
            token_tuple = tuple(token_ids)
            if token_tuple in token_to_text_map:
                return token_to_text_map[token_tuple]
            # Create unique text for this token sequence
            text = f"text_for_tokens_{'_'.join(map(str, token_ids))}"
            token_to_text_map[token_tuple] = text
            text_to_token_map[text] = token_ids
            return text

        def mock_tokenizer_call(
            text_input: str | list[str],
            padding: bool = True,
            return_tensors: str | None = None,
        ) -> dict[str, list[list[int]]]:
            """Mock tokenizer call that maintains consistent round-trip behavior.

            Args:
                text_input: Text or list of texts to tokenize
                padding: Whether to pad (unused in mock)
                return_tensors: Tensor format to return (unused in mock)

            Returns:
                Dictionary with input_ids and attention_mask
            """
            text = text_input[0] if isinstance(text_input, list) else text_input

            # Use cached tokens if available, otherwise use base_input_ids
            if text in text_to_token_map:
                token_ids = text_to_token_map[text]
            else:
                token_ids = base_input_ids
                text_to_token_map[text] = token_ids
                token_to_text_map[tuple(token_ids)] = text

            return {
                "input_ids": [token_ids],
                "attention_mask": [[1] * len(token_ids)],
            }

        mock_tok.apply_chat_template.return_value = "test prompt"
        mock_tok.decode = mock_decode
        mock_tok.side_effect = mock_tokenizer_call

        mocker.patch(
            "max.pipelines.architectures.qwen2_5vl.tokenizer.AutoTokenizer.from_pretrained",
            return_value=mock_tok,
        )
        return mock_tok

    return _create_tokenizer


@pytest.fixture
def qwen_vision_config_mock(mocker: MockerFixture) -> MagicMock:
    """Mock Qwen2.5VL vision config with standard parameters.

    Creates a mock config containing typical Qwen2.5VL vision parameters.
    This is used to create the mock pipeline_config fixture.
    """
    cfg = MagicMock()
    cfg.vision_config = MagicMock()
    cfg.vision_config.patch_size = 14
    cfg.vision_config.temporal_patch_size = 2
    cfg.vision_config.spatial_merge_size = 2
    cfg.vision_config.window_size = 448
    cfg.vision_config.tokens_per_second = 2
    return cfg


def _create_mock_huggingface_config(
    qwen_token_ids: dict[str, int],
) -> NonCallableMock:
    """Create a mock HuggingFace config with spec=Qwen2_5_VLConfig.

    Using spec ensures that ONLY attributes present on the real Qwen2_5_VLConfig
    are accessible. This prevents tests from passing when code incorrectly
    accesses attributes that don't exist on the real config type.
    """
    mock_hf_config = NonCallableMock(spec=Qwen2_5_VLConfig)

    # Set up required attributes that the tokenizer accesses from HuggingFace config
    mock_hf_config.image_token_id = qwen_token_ids["image_token_id"]
    mock_hf_config.video_token_id = qwen_token_ids["video_token_id"]
    mock_hf_config.vision_start_token_id = qwen_token_ids[
        "vision_start_token_id"
    ]
    mock_hf_config.eos_token_id = 2

    # Create vision config
    vision_config = NonCallableMock()
    vision_config.patch_size = 14
    vision_config.temporal_patch_size = 2
    vision_config.spatial_merge_size = 2
    vision_config.window_size = 448
    vision_config.tokens_per_second = 2
    mock_hf_config.vision_config = vision_config

    return mock_hf_config


@pytest.fixture
def mock_pipeline_config(qwen_token_ids: dict[str, int]) -> MagicMock:
    """Create a mock PipelineConfig for Qwen2.5VL tests.

    Provides a mock pipeline config with HuggingFace config containing
    the required token IDs and vision config.
    """
    hf_config = _create_mock_huggingface_config(qwen_token_ids)

    kv_cache_config = NonCallableMock(spec=KVCacheConfig)
    kv_cache_config.enable_prefix_caching = False

    # Create mock model config
    model_config = MagicMock()
    model_config.huggingface_config = hf_config
    model_config.kv_cache = kv_cache_config

    # Create mock pipeline config
    pipeline_config = MagicMock()
    # A real runtime config, not a MagicMock: the tokenizer sizes its
    # preprocessed image cache from these budgets, and arithmetic on a
    # MagicMock raises.
    pipeline_config.runtime = PipelineRuntimeConfig()
    pipeline_config.model = model_config
    return pipeline_config


@pytest.fixture
def image_processor_mock() -> Callable[[np.ndarray, np.ndarray], MagicMock]:
    """Create a mock image processor for vision tests.

    Returns a factory function that creates an image processor mock with
    the specified pixel values and grid information.

    Returns:
        Factory function that takes (pixel_values, grid_thw) and returns configured mock
    """

    def _create_image_processor(
        pixel_values: np.ndarray,
        grid_thw: np.ndarray,
    ) -> MagicMock:
        mock_processor = MagicMock()
        mock_processor.merge_size = 2

        def mock_processor_call(
            images: list[str], return_tensors: str = "pt"
        ) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
            """Mock image processor that returns pixel values and grid information.

            Args:
                images: List of image URLs or paths
                return_tensors: Format for returned tensors (unused in mock)

            Returns:
                Tuple of (processed_dict, pixel_values_list)
            """
            processed = {
                "image_grid_thw": grid_thw,
                "concatenated_pixel_values": pixel_values,
            }
            pixel_values_list = [pixel_values]
            return processed, pixel_values_list

        mock_processor.side_effect = mock_processor_call
        return mock_processor

    return _create_image_processor


# ============================================================================
# Test Cases
# ============================================================================


@pytest.mark.asyncio
async def test_qwen_input_preparation__position_ids_after_reset(
    mocker: MockerFixture,
    round_trip_tokenizer_mock: Callable[[list[int]], MagicMock],
    mock_pipeline_config: MagicMock,
    qwen_token_ids: dict[str, int],
) -> None:
    """Test that decoder_position_ids are correctly recomputed after context reset (text-only).

    TEST SCENARIO:
    --------------
    1. Create a text-only context (no actual images)
    2. Generate 5 tokens (simulating normal generation)
    3. Reset the context (simulating preemption/restart)
    4. Compare position IDs from reset context vs a fresh context with same tokens

    WHY THIS MATTERS:
    -----------------
    After preemption, the context must recompute position IDs for all tokens including
    the newly generated ones. This test ensures the recomputation logic in
    prepare_initial_token_inputs produces identical results to initial computation,
    preventing subtle bugs in resumed generation.

    WHAT WE'RE TESTING:
    -------------------
    - The attention mask is recalculated correctly for the new token count
    - decoder_position_ids (3D RoPE) are recomputed accurately for text tokens
    - vision_position_ids is None for text-only inputs (no images)
    - The recomputation matches exactly what a fresh context would produce

    Related: FIXME E2EOPT-661
    """

    # ========================================================================
    # SECTION 1: Mock Setup
    # ========================================================================
    # Create test input (text-only, no actual images)
    test_input_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Configure mocks using fixtures
    round_trip_tokenizer_mock(test_input_ids)

    mocker.patch(
        "max.pipelines.architectures.qwen2_5vl.tokenizer.process_vision_info",
        return_value=(None, None, ""),  # No images for text-only test
    )

    # Create tokenizer with mock pipeline config
    tokenizer = Qwen2_5VLTokenizer("test-model", mock_pipeline_config)

    # ========================================================================
    # SECTION 2: Create Initial Context
    # ========================================================================
    request = TextGenerationRequest(
        messages=[
            TextGenerationRequestMessage(role="user", content="test with image")
        ],
        request_id=RequestID("test-id-1"),
        model_name="test-model",
    )

    context = await tokenizer.new_context(request)

    # Verify context was created correctly with required metadata
    assert context is not None
    assert isinstance(context, Qwen2_5VLTextAndVisionContext), (
        f"Expected Qwen2_5VLTextAndVisionContext, got {type(context)}"
    )

    initial_current_length = len(context.tokens)

    # ========================================================================
    # SECTION 3: Simulate Generation + Reset (Core Test Scenario)
    # ========================================================================
    # Add 5 tokens to simulate normal generation
    for i in range(5):
        context.update(100 + i)  # Add tokens 100, 101, 102, 103, 104

    assert len(context.tokens) == initial_current_length + 5

    # Reset the context (simulating preemption/restart)
    context.reset()

    # Verify reset state: indices reset, but tokens remain
    assert context.tokens.processed_length == 0
    assert (
        context.tokens.current_position == initial_current_length + 5
    )  # Includes the 5 added tokens
    assert len(context.tokens) == initial_current_length + 5

    # Key point: decoder_position_ids shape (initial_current_length) doesn't match
    # current_length (initial_current_length + 5) because we added tokens.
    # This triggers the recomputation branch in prepare_initial_token_inputs.

    # ========================================================================
    # SECTION 4: Get Position IDs from Reset Context
    # ========================================================================
    processor = create_mock_qwen_batch_processor(mocker)
    mock_kv_cache = Mock(spec=KVCacheInputs)

    # Get position IDs from reset context (should trigger recomputation)
    reset_qwen_inputs: Qwen2_5VLInputs = processor.prepare_initial_token_inputs(
        replica_batches=[[context]],
        kv_cache_inputs=mock_kv_cache,
        return_n_logits=1,
    )

    # ========================================================================
    # SECTION 5: Create Fresh Context for Comparison
    # ========================================================================
    # Create a fresh context with the same token sequence as if we were starting
    # from scratch with all tokens (initial + 5 generated)
    assert len(context.tokens.all) == initial_current_length + 5

    request2 = TextGenerationRequest(
        prompt=context.tokens.all.tolist(),  # Use all tokens
        request_id=RequestID("test-id-2"),
        model_name="test-model",
    )
    context2 = await tokenizer.new_context(request2)

    # Get position IDs from fresh context
    fresh_qwen_inputs: Qwen2_5VLInputs = processor.prepare_initial_token_inputs(
        replica_batches=[[context2]],
        kv_cache_inputs=mock_kv_cache,
        return_n_logits=1,
    )

    # ========================================================================
    # SECTION 6: Compare Position IDs (The Critical Assertions)
    # ========================================================================

    # Extract decoder position IDs for comparison
    reset_position_ids = reset_qwen_inputs.position_ids.to_numpy()
    fresh_position_ids = fresh_qwen_inputs.position_ids.to_numpy()

    # CRITICAL ASSERTION: decoder_position_ids must match exactly
    np.testing.assert_array_equal(
        reset_position_ids,
        fresh_position_ids,
        err_msg=(
            "decoder_position_ids from reset context do not match fresh context! "
            "This indicates a bug in the recomputation logic in prepare_initial_token_inputs."
        ),
    )

    # Verify vision_position_ids is None for text-only inputs
    assert reset_qwen_inputs.vision_position_ids is None, (
        "Text-only context should have no vision_position_ids"
    )
    assert fresh_qwen_inputs.vision_position_ids is None, (
        "Text-only context should have no vision_position_ids"
    )


@pytest.mark.asyncio
async def test_qwen_input_preparation__position_ids_after_reset_with_image(
    mocker: MockerFixture,
    round_trip_tokenizer_mock: Callable[[list[int]], MagicMock],
    mock_pipeline_config: MagicMock,
    qwen_token_ids: dict[str, int],
    image_processor_mock: Callable[[np.ndarray, np.ndarray], MagicMock],
) -> None:
    """Test decoder_position_ids recomputation after context reset with actual image data.

    TEST SCENARIO:
    --------------
    1. Create a context with an actual image input (with pixel values and grid info)
    2. Generate 3 tokens (simulating normal generation)
    3. Reset the context (simulating preemption/restart)
    4. Compare position IDs from reset context vs a fresh context with same tokens + images

    WHY THIS MATTERS:
    -----------------
    Multi-modal inputs (text + images) require complex position ID calculations for both
    the decoder (3D RoPE for text tokens) and vision encoder (1D RoPE for image patches).
    After preemption, we must recompute these correctly while preserving image metadata.
    This test ensures the recomputation produces identical results to initial computation.

    WHAT WE'RE TESTING:
    -------------------
    - The attention mask is recalculated correctly for the new token count
    - decoder_position_ids (3D RoPE) are recomputed accurately with image tokens
    - vision_position_ids (1D RoPE) are consistent for the same image data
    - Image metadata (pixel values, grid info) is preserved through reset
    - Both position ID types match exactly what a fresh context would produce

    Related: FIXME E2EOPT-661
    """

    # ========================================================================
    # SECTION 1: Mock Setup (Tokenizer, Image Processor, Image Data)
    # ========================================================================
    # Create test input with vision tokens
    # Format: [text] [vision_start] [image tokens] [text]
    base_input_ids = [
        1,
        2,
        3,
        4,  # Initial text tokens
        qwen_token_ids["vision_start_token_id"],  # vision_start
        qwen_token_ids["image_token_id"],  # image tokens (4x)
        qwen_token_ids["image_token_id"],
        qwen_token_ids["image_token_id"],
        qwen_token_ids["image_token_id"],
        5,
        6,
        7,
        8,
        9,  # More text tokens
    ]

    # Configure mocks using fixtures
    round_trip_tokenizer_mock(base_input_ids)

    # Setup image data: pixel values + grid layout
    mock_image_url = "data:image/png;base64,iVBORw0KGgoAAAANS"
    mock_pixel_values = np.random.rand(1, 3, 448, 448).astype(np.float32)
    mock_image_grid_thw = np.array([[1, 2, 2]], dtype=np.int32)  # t=1, h=2, w=2

    mocker.patch(
        "max.pipelines.architectures.qwen2_5vl.tokenizer.process_vision_info",
        return_value=(
            [mock_image_url],
            None,
            "",
        ),  # (images, videos, placeholder_text)
    )

    # Create tokenizer with mock pipeline config
    tokenizer = Qwen2_5VLTokenizer("test-model", mock_pipeline_config)
    tokenizer.img_processor = image_processor_mock(
        mock_pixel_values, mock_image_grid_thw
    )

    # ========================================================================
    # SECTION 2: Create Initial Context with Image
    # ========================================================================
    request = TextGenerationRequest(
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content=[
                    ImageContentPart(),
                    TextContentPart(text="What's in this image?"),
                ],
            )
        ],
        request_id=RequestID("test-id-with-image-1"),
        model_name="test-model",
    )

    context = await tokenizer.new_context(request)

    # Verify context was created with image metadata
    assert context is not None
    assert isinstance(context, Qwen2_5VLTextAndVisionContext), (
        f"Expected Qwen2_5VLTextAndVisionContext, got {type(context)}"
    )
    assert len(context.images) > 0, "Context should contain image metadata"

    initial_current_length = len(context.tokens)
    initial_images = context.images

    # ========================================================================
    # SECTION 3: Simulate Generation + Reset (Core Test Scenario)
    # ========================================================================
    # Add 3 tokens to simulate normal generation
    for i in range(3):
        context.update(200 + i)  # Add tokens 200, 201, 202

    assert len(context.tokens) == initial_current_length + 3

    # Reset the context (simulating preemption/restart)
    context.reset()

    # Verify reset state: indices reset, tokens and images remain
    assert context.tokens.processed_length == 0
    assert context.tokens.current_position == initial_current_length + 3
    assert len(context.tokens) == initial_current_length + 3
    assert len(context.images) == len(initial_images), (
        "Images should be preserved after reset"
    )

    # ========================================================================
    # SECTION 4: Get Position IDs from Reset Context
    # ========================================================================
    processor = create_mock_qwen_batch_processor(mocker)
    mock_kv_cache = Mock(spec=KVCacheInputs)

    # Get position IDs from reset context (should trigger recomputation)
    reset_qwen_inputs: Qwen2_5VLInputs = processor.prepare_initial_token_inputs(
        replica_batches=[[context]],
        kv_cache_inputs=mock_kv_cache,
        return_n_logits=1,
    )

    # ========================================================================
    # SECTION 5: Create Fresh Context for Comparison
    # ========================================================================
    # Create a fresh context with the same token sequence AND images
    # Note: We provide tokens as prompt (goes through decode->encode cycle)
    # but images are also provided to ensure they're processed identically
    request2 = TextGenerationRequest(
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content=[
                    ImageContentPart(),
                    TextContentPart(text="What's in this image?"),
                ],
            )
        ],
        prompt=context.tokens.all.tolist(),  # Override with full token sequence
        request_id=RequestID("test-id-with-image-2"),
        model_name="test-model",
    )

    context2 = await tokenizer.new_context(request2)

    # Verify fresh context has same properties
    assert len(context2.images) == len(context.images)
    assert len(context2.tokens) == len(context.tokens)

    # Get position IDs from fresh context
    fresh_qwen_inputs: Qwen2_5VLInputs = processor.prepare_initial_token_inputs(
        replica_batches=[[context2]],
        kv_cache_inputs=mock_kv_cache,
        return_n_logits=1,
    )

    # ========================================================================
    # SECTION 6: Compare Position IDs (The Critical Assertions)
    # ========================================================================

    # Extract decoder position IDs for comparison
    reset_position_ids = reset_qwen_inputs.position_ids.to_numpy()
    fresh_position_ids = fresh_qwen_inputs.position_ids.to_numpy()

    # CRITICAL ASSERTION 1: decoder_position_ids must match exactly
    np.testing.assert_array_equal(
        reset_position_ids,
        fresh_position_ids,
        err_msg=(
            "decoder_position_ids from reset context with images do not match fresh context! "
            "This indicates a bug in the recomputation logic for vision-enabled contexts."
        ),
    )

    # Verify vision_position_ids exist for both contexts
    assert reset_qwen_inputs.vision_position_ids is not None, (
        "Context with images should have vision_position_ids"
    )
    assert fresh_qwen_inputs.vision_position_ids is not None, (
        "Context with images should have vision_position_ids"
    )

    # Extract vision position IDs (list of Tensors, one per device)
    reset_vision_position_ids = reset_qwen_inputs.vision_position_ids[
        0
    ].to_numpy()
    fresh_vision_position_ids = fresh_qwen_inputs.vision_position_ids[
        0
    ].to_numpy()

    # CRITICAL ASSERTION 2: vision_position_ids must match exactly
    np.testing.assert_array_equal(
        reset_vision_position_ids,
        fresh_vision_position_ids,
        err_msg=(
            "vision_position_ids from reset context with images do not match fresh context! "
            "This indicates a bug in the vision position ID handling after context reset."
        ),
    )


def test_qwen_text_only_decoder_posids_increment_on_first_decode(
    mocker: MockerFixture,
) -> None:
    """Test decoder position IDs increment correctly from prefill to first decode step.

    This test verifies the fix for an off-by-one error in computing decoder position IDs
    during text-only token generation (commit a52fa38ed1).

    Verify the key invariant: "first decode position ID equals last prefill
    position ID + 1" for text-only inputs.

    TEST STRUCTURE:
    ---------------
    1. Prefill phase: Process a text-only prompt of length L
    2. First decode phase: Generate position IDs for single token (L+1)
    3. Assert: decode_position_ids[0] == prefill_position_ids[-1] + 1
    """
    # Create mock model and text-only context.
    processor = create_mock_qwen_batch_processor(mocker)

    # Create a text-only prompt of length L=33.
    L = 33
    tokens = np.arange(L, dtype=np.int64)

    # Create initial decoder position IDs for prefill (3D RoPE: temporal, height, width).
    # For text-only inputs, all 3 dimensions have identical position IDs.
    initial_position_ids = np.arange(L, dtype=np.int64)
    decoder_position_ids_3d = np.tile(initial_position_ids, (3, 1))

    # Create a minimal Qwen2_5VLTextAndVisionContext for text-only input.
    rope_delta = 0
    decoder_position_ids = decoder_position_ids_3d
    spatial_merge_size = 2
    image_token_id = 151652
    ctx = Qwen2_5VLTextAndVisionContext(
        request_id=RequestID("test-posid-increment"),
        tokens=TokenBuffer(tokens),
        max_length=L + 8,
        eos_tracker=EOSTracker(eos_token_ids={2}),
        sampling_params=SamplingParams(max_new_tokens=2),
        images=[],  # text-only
        vision_token_ids=[],  # text-only
        # Qwen2.5VL-specific fields
        rope_delta=rope_delta,
        decoder_position_ids=decoder_position_ids,
        spatial_merge_size=spatial_merge_size,
        image_token_id=image_token_id,
        video_token_id=Mock(),
        vision_start_token_id=Mock(),
        tokens_per_second=Mock(),
        image_token_indices=Mock(),
        vision_data=Mock(),
    )

    # Verify initial state: prefill phase (start_idx=0, active range covers all tokens).
    assert ctx.tokens.processed_length == 0
    assert ctx.tokens.current_position == L
    assert len(ctx.tokens) == L

    # Build inputs for prefill phase.
    dummy_kv_inputs = Mock(spec=KVCacheInputs)

    prefill_inputs: Qwen2_5VLInputs = processor.prepare_initial_token_inputs(
        replica_batches=[[ctx]],
        kv_cache_inputs=dummy_kv_inputs,
        return_n_logits=1,
    )

    # Extract prefill position IDs (shape: (3, L) for 3D RoPE).
    prefill_posids = prefill_inputs.position_ids.to_numpy()
    assert prefill_posids.shape == (3, L), (
        f"Expected prefill position IDs shape (3, {L}), got {prefill_posids.shape}"
    )

    # Get the last prefill position ID (same for all 3 RoPE dimensions in text-only)
    last_prefill = prefill_posids[0, -1]

    # Simulate first decode step (single-token generation).
    # Mimic the pipeline's next iteration: move to decode phase with single active token.
    ctx.tokens.advance_with_token(0)

    step1_inputs: Qwen2_5VLInputs = processor.prepare_initial_token_inputs(
        replica_batches=[[ctx]],
        kv_cache_inputs=dummy_kv_inputs,
        return_n_logits=1,
    )

    # Extract decode position IDs (shape: (3, 1) for single token).
    step1_posids = step1_inputs.position_ids.to_numpy()
    assert step1_posids.shape == (3, 1), (
        f"Expected decode position IDs shape (3, 1), got {step1_posids.shape}"
    )

    # Verify decode position ID = last prefill ID + 1.
    expected = last_prefill + 1

    # All 3 RoPE dimensions should have identical position IDs for text-only.
    for dim in range(3):
        actual = step1_posids[dim, 0]
        assert actual == expected, (
            f"Decode position ID mismatch in dimension {dim}! "
            f"Expected {expected} (last_prefill + 1 = {last_prefill} + 1), "
            f"but got {actual}. "
            f"This indicates the off-by-one bug: position IDs are not incrementing "
            f"correctly from prefill to decode phase."
        )
