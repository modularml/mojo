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
"""Unit tests for _build_token_frequency_csr function.

This module tests the token frequency CSR matrix building function used
in the sampling logits processor. The CSR format stores token frequencies
efficiently for penalty calculations.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import (
    CPU,
    Accelerator,
    Buffer,
    DevicePinnedBuffer,
    accelerator_count,
)
from max.dtype import DType
from max.pipelines.context import (
    SamplingParams,
    TextContext,
    TokenBuffer,
)
from max.pipelines.sampling.sampling_logits_processor import (
    FrequencyData,
    FusedSamplingProcessor,
    _build_token_frequency_csr,
)


def create_text_context(
    prompt_tokens: npt.NDArray[np.int64],
    generated_tokens: npt.NDArray[np.int64] | None = None,
) -> TextContext:
    """Create a TextContext with specified prompt and generated tokens.

    Args:
        prompt_tokens: Array of prompt token IDs.
        generated_tokens: Optional array of generated token IDs.

    Returns:
        A TextContext with the specified token configuration.
    """
    if generated_tokens is None:
        generated_tokens = np.array([], dtype=np.int64)

    total_length = len(prompt_tokens) + len(generated_tokens)

    # Create context with prompt tokens only - TextContext.__post_init__
    # will set _prompt_len, _active_idx, _end_idx based on initial tokens
    context = TextContext(
        max_length=total_length + 100,  # Allow some headroom
        tokens=TokenBuffer(prompt_tokens.copy()),
    )

    # Use update() to add each generated token, which properly manages indices
    for token in generated_tokens:
        context.update(int(token))

    return context


class TestBuildTokenFrequencyCSRBasic:
    """Basic functionality tests for _build_token_frequency_csr."""

    def test_single_context_generated_only(self) -> None:
        """Test CSR building with a single context, generated tokens only."""
        prompt = np.array([1, 2, 3], dtype=np.int64)
        generated = np.array([4, 5, 4, 6, 4], dtype=np.int64)  # 4 appears 3x

        context = create_text_context(prompt, generated)
        batch = [context]
        device = CPU()
        padding_size = 2

        result = _build_token_frequency_csr(
            batch, padding_size, device, include_prompt=False
        )

        assert isinstance(result, FrequencyData)
        assert result.data is not None
        assert result.offsets is not None

        # Convert to numpy for verification
        data_np = result.data.to_numpy()
        offsets_np = result.offsets.to_numpy()

        # Verify offsets shape: batch_size + 1
        assert offsets_np.shape == (2,)
        assert offsets_np[0] == 0

        # Extract token-frequency pairs for the context
        start, end = offsets_np[0], offsets_np[1]
        context_data = data_np[start:end]

        # Filter out padding entries (token_id == -1)
        valid_entries = context_data[context_data[:, 0] != -1]

        # Build frequency dict from CSR data
        freq_dict = {int(row[0]): int(row[1]) for row in valid_entries}

        # Verify frequencies for generated tokens only
        assert freq_dict.get(4) == 3, (
            f"Token 4 should appear 3 times, got {freq_dict}"
        )
        assert freq_dict.get(5) == 1, (
            f"Token 5 should appear 1 time, got {freq_dict}"
        )
        assert freq_dict.get(6) == 1, (
            f"Token 6 should appear 1 time, got {freq_dict}"
        )
        # Prompt tokens should NOT be included
        assert 1 not in freq_dict, (
            "Prompt token 1 should not be in generated-only CSR"
        )
        assert 2 not in freq_dict, (
            "Prompt token 2 should not be in generated-only CSR"
        )
        assert 3 not in freq_dict, (
            "Prompt token 3 should not be in generated-only CSR"
        )

    def test_single_context_include_prompt(self) -> None:
        """Test CSR building with a single context, including prompt tokens."""
        prompt = np.array([1, 2, 3, 1], dtype=np.int64)  # 1 appears 2x
        generated = np.array([4, 5, 1], dtype=np.int64)  # 1 appears 1x more

        context = create_text_context(prompt, generated)
        batch = [context]
        device = CPU()
        padding_size = 2

        result = _build_token_frequency_csr(
            batch, padding_size, device, include_prompt=True
        )

        data_np = result.data.to_numpy()
        offsets_np = result.offsets.to_numpy()

        start, end = offsets_np[0], offsets_np[1]
        context_data = data_np[start:end]
        valid_entries = context_data[context_data[:, 0] != -1]
        freq_dict = {int(row[0]): int(row[1]) for row in valid_entries}

        # Token 1 appears 2x in prompt + 1x in generated = 3x total
        assert freq_dict.get(1) == 3, (
            f"Token 1 should appear 3 times, got {freq_dict}"
        )
        assert freq_dict.get(2) == 1
        assert freq_dict.get(3) == 1
        assert freq_dict.get(4) == 1
        assert freq_dict.get(5) == 1

    def test_empty_generated_tokens(self) -> None:
        """Test CSR building with no generated tokens."""
        prompt = np.array([1, 2, 3], dtype=np.int64)

        context = create_text_context(prompt)
        batch = [context]
        device = CPU()
        padding_size = 2

        # With include_prompt=False, should have no actual frequency data
        result = _build_token_frequency_csr(
            batch, padding_size, device, include_prompt=False
        )

        data_np = result.data.to_numpy()
        offsets_np = result.offsets.to_numpy()

        start, end = offsets_np[0], offsets_np[1]
        context_data = data_np[start:end]
        valid_entries = context_data[context_data[:, 0] != -1]

        # No generated tokens means no valid entries
        assert len(valid_entries) == 0

    def test_empty_generated_tokens_include_prompt(self) -> None:
        """Test CSR building with no generated tokens but include_prompt=True."""
        prompt = np.array([1, 2, 3, 1], dtype=np.int64)

        context = create_text_context(prompt)
        batch = [context]
        device = CPU()
        padding_size = 2

        result = _build_token_frequency_csr(
            batch, padding_size, device, include_prompt=True
        )

        data_np = result.data.to_numpy()
        offsets_np = result.offsets.to_numpy()

        start, end = offsets_np[0], offsets_np[1]
        context_data = data_np[start:end]
        valid_entries = context_data[context_data[:, 0] != -1]
        freq_dict = {int(row[0]): int(row[1]) for row in valid_entries}

        # Should include prompt tokens
        assert freq_dict.get(1) == 2
        assert freq_dict.get(2) == 1
        assert freq_dict.get(3) == 1


class TestBuildTokenFrequencyCSRBatch:
    """Tests for batch processing in _build_token_frequency_csr."""

    def test_multiple_contexts(self) -> None:
        """Test CSR building with multiple contexts in batch."""
        # Context 1: generated tokens [10, 10, 11]
        context1 = create_text_context(
            prompt_tokens=np.array([1, 2], dtype=np.int64),
            generated_tokens=np.array([10, 10, 11], dtype=np.int64),
        )

        # Context 2: generated tokens [20, 21, 20, 20]
        context2 = create_text_context(
            prompt_tokens=np.array([3, 4, 5], dtype=np.int64),
            generated_tokens=np.array([20, 21, 20, 20], dtype=np.int64),
        )

        batch = [context1, context2]
        device = CPU()
        padding_size = 1

        result = _build_token_frequency_csr(
            batch, padding_size, device, include_prompt=False
        )

        data_np = result.data.to_numpy()
        offsets_np = result.offsets.to_numpy()

        # Verify offsets shape: batch_size + 1
        assert offsets_np.shape == (3,)

        # Check context 1
        start1, end1 = offsets_np[0], offsets_np[1]
        data1 = data_np[start1:end1]
        valid1 = data1[data1[:, 0] != -1]
        freq1 = {int(row[0]): int(row[1]) for row in valid1}

        assert freq1.get(10) == 2
        assert freq1.get(11) == 1
        assert 20 not in freq1  # From context 2

        # Check context 2
        start2, end2 = offsets_np[1], offsets_np[2]
        data2 = data_np[start2:end2]
        valid2 = data2[data2[:, 0] != -1]
        freq2 = {int(row[0]): int(row[1]) for row in valid2}

        assert freq2.get(20) == 3
        assert freq2.get(21) == 1
        assert 10 not in freq2  # From context 1

    def test_batch_with_varying_lengths(self) -> None:
        """Test CSR building with contexts of different lengths."""
        # Short context
        context1 = create_text_context(
            prompt_tokens=np.array([1], dtype=np.int64),
            generated_tokens=np.array([2], dtype=np.int64),
        )

        # Medium context
        context2 = create_text_context(
            prompt_tokens=np.array([10, 11, 12], dtype=np.int64),
            generated_tokens=np.array([13, 14, 15, 16], dtype=np.int64),
        )

        # Long context with repetitions
        context3 = create_text_context(
            prompt_tokens=np.array([20, 21], dtype=np.int64),
            generated_tokens=np.array([22, 22, 22, 23, 23, 24], dtype=np.int64),
        )

        batch = [context1, context2, context3]
        device = CPU()
        padding_size = 3

        result = _build_token_frequency_csr(
            batch, padding_size, device, include_prompt=False
        )

        offsets_np = result.offsets.to_numpy()

        # Verify we have correct number of offsets
        assert offsets_np.shape == (4,)

        # Verify offsets are monotonically increasing
        for i in range(len(offsets_np) - 1):
            assert offsets_np[i] < offsets_np[i + 1]


class TestBuildTokenFrequencyCSREdgeCases:
    """Edge case tests for _build_token_frequency_csr."""

    def test_all_same_token(self) -> None:
        """Test CSR building when all tokens are the same."""
        prompt = np.array([5, 5, 5], dtype=np.int64)
        generated = np.array([5, 5, 5, 5], dtype=np.int64)

        context = create_text_context(prompt, generated)
        batch = [context]
        device = CPU()
        padding_size = 2

        result = _build_token_frequency_csr(
            batch, padding_size, device, include_prompt=True
        )

        data_np = result.data.to_numpy()
        offsets_np = result.offsets.to_numpy()

        start, end = offsets_np[0], offsets_np[1]
        context_data = data_np[start:end]
        valid_entries = context_data[context_data[:, 0] != -1]

        # Should have exactly one unique token
        assert len(valid_entries) == 1
        assert valid_entries[0, 0] == 5
        assert valid_entries[0, 1] == 7  # 3 prompt + 4 generated

    def test_all_unique_tokens(self) -> None:
        """Test CSR building when all tokens are unique."""
        prompt = np.array([1, 2, 3], dtype=np.int64)
        generated = np.array([4, 5, 6, 7], dtype=np.int64)

        context = create_text_context(prompt, generated)
        batch = [context]
        device = CPU()
        padding_size = 2

        result = _build_token_frequency_csr(
            batch, padding_size, device, include_prompt=True
        )

        data_np = result.data.to_numpy()
        offsets_np = result.offsets.to_numpy()

        start, end = offsets_np[0], offsets_np[1]
        context_data = data_np[start:end]
        valid_entries = context_data[context_data[:, 0] != -1]

        # Should have 7 unique tokens, each with count 1
        assert len(valid_entries) == 7
        for entry in valid_entries:
            assert entry[1] == 1

    def test_single_token_prompt(self) -> None:
        """Test CSR building with a single token prompt."""
        prompt = np.array([42], dtype=np.int64)
        generated = np.array([43, 44], dtype=np.int64)

        context = create_text_context(prompt, generated)
        batch = [context]
        device = CPU()
        padding_size = 1

        result = _build_token_frequency_csr(
            batch, padding_size, device, include_prompt=True
        )

        data_np = result.data.to_numpy()
        offsets_np = result.offsets.to_numpy()

        start, end = offsets_np[0], offsets_np[1]
        context_data = data_np[start:end]
        valid_entries = context_data[context_data[:, 0] != -1]
        freq_dict = {int(row[0]): int(row[1]) for row in valid_entries}

        assert freq_dict.get(42) == 1
        assert freq_dict.get(43) == 1
        assert freq_dict.get(44) == 1

    def test_single_generated_token(self) -> None:
        """Test CSR building with a single generated token."""
        prompt = np.array([1, 2, 3], dtype=np.int64)
        generated = np.array([99], dtype=np.int64)

        context = create_text_context(prompt, generated)
        batch = [context]
        device = CPU()
        padding_size = 1

        result = _build_token_frequency_csr(
            batch, padding_size, device, include_prompt=False
        )

        data_np = result.data.to_numpy()
        offsets_np = result.offsets.to_numpy()

        start, end = offsets_np[0], offsets_np[1]
        context_data = data_np[start:end]
        valid_entries = context_data[context_data[:, 0] != -1]

        assert len(valid_entries) == 1
        assert valid_entries[0, 0] == 99
        assert valid_entries[0, 1] == 1

    def test_large_token_ids(self) -> None:
        """Test CSR building with large token IDs (like Llama-3 vocab size)."""
        vocab_size = 128256
        large_tokens = np.array(
            [vocab_size - 1, vocab_size - 2, vocab_size - 1], dtype=np.int64
        )

        context = create_text_context(
            prompt_tokens=np.array([0, 1], dtype=np.int64),
            generated_tokens=large_tokens,
        )
        batch = [context]
        device = CPU()
        padding_size = 2

        result = _build_token_frequency_csr(
            batch, padding_size, device, include_prompt=False
        )

        data_np = result.data.to_numpy()
        offsets_np = result.offsets.to_numpy()

        start, end = offsets_np[0], offsets_np[1]
        context_data = data_np[start:end]
        valid_entries = context_data[context_data[:, 0] != -1]
        freq_dict = {int(row[0]): int(row[1]) for row in valid_entries}

        assert freq_dict.get(vocab_size - 1) == 2
        assert freq_dict.get(vocab_size - 2) == 1


class TestBuildTokenFrequencyCSRPadding:
    """Tests for padding behavior in _build_token_frequency_csr."""

    def test_padding_size_zero(self) -> None:
        """Test CSR building with zero padding."""
        prompt = np.array([1, 2], dtype=np.int64)
        generated = np.array([3, 4], dtype=np.int64)

        context = create_text_context(prompt, generated)
        batch = [context]
        device = CPU()
        padding_size = 0

        result = _build_token_frequency_csr(
            batch, padding_size, device, include_prompt=False
        )

        data_np = result.data.to_numpy()
        offsets_np = result.offsets.to_numpy()

        start, end = offsets_np[0], offsets_np[1]
        context_data = data_np[start:end]

        # No padding means only valid entries
        assert len(context_data) == 2  # 2 unique tokens

    def test_padding_included_correctly(self) -> None:
        """Test that padding entries have token_id=-1 and count=0."""
        prompt = np.array([1], dtype=np.int64)
        generated = np.array([2], dtype=np.int64)

        context = create_text_context(prompt, generated)
        batch = [context]
        device = CPU()
        padding_size = 5

        result = _build_token_frequency_csr(
            batch, padding_size, device, include_prompt=False
        )

        data_np = result.data.to_numpy()
        offsets_np = result.offsets.to_numpy()

        start, end = offsets_np[0], offsets_np[1]
        context_data = data_np[start:end]

        # Find padding entries
        padding_entries = context_data[context_data[:, 0] == -1]

        # Should have exactly padding_size padding entries
        assert len(padding_entries) == padding_size

        # All padding entries should have count=0
        for entry in padding_entries:
            assert entry[1] == 0


class TestBuildTokenFrequencyCSRDataTypes:
    """Tests for data type handling in _build_token_frequency_csr."""

    def test_output_tensor_types(self) -> None:
        """Test that output tensors have correct data types."""
        prompt = np.array([1, 2, 3], dtype=np.int64)
        generated = np.array([4, 5], dtype=np.int64)

        context = create_text_context(prompt, generated)
        batch = [context]
        device = CPU()
        padding_size = 2

        result = _build_token_frequency_csr(
            batch, padding_size, device, include_prompt=False
        )

        # Verify data array dtype is int64
        data_np = result.data.to_numpy()
        assert data_np.dtype == np.int32

        # Verify offsets array dtype is int64
        offsets_np = result.offsets.to_numpy()
        assert offsets_np.dtype == np.uint32

    def test_data_shape(self) -> None:
        """Test that data array has correct shape (N, 2)."""
        prompt = np.array([1, 2, 3], dtype=np.int64)
        generated = np.array([4, 5, 4], dtype=np.int64)

        context = create_text_context(prompt, generated)
        batch = [context]
        device = CPU()
        padding_size = 2

        result = _build_token_frequency_csr(
            batch, padding_size, device, include_prompt=False
        )

        data_np = result.data.to_numpy()

        # Should be 2D with second dimension = 2 (token_id, count)
        assert data_np.ndim == 2
        assert data_np.shape[1] == 2


class TestFusedSamplingProcessorInit:
    """Tests for FusedSamplingProcessor.__init__ penalty input construction."""

    @pytest.mark.parametrize(
        ("enable_penalties", "expect_penalty_inputs"),
        [
            pytest.param(True, True, id="enabled"),
            pytest.param(False, False, id="disabled"),
        ],
    )
    def test_penalty_inputs_follows_enable_penalties_flag(
        self, enable_penalties: bool, expect_penalty_inputs: bool
    ) -> None:
        """penalty_inputs is constructed iff enable_penalties=True.

        Regression: the sampler graph is compiled with penalty buffer slots
        when enable_penalties=True, so PenaltyInputs must always be created
        regardless of per-request penalty values.
        """
        context = TextContext(
            max_length=100,
            tokens=TokenBuffer(np.array([1, 2, 3], dtype=np.int64)),
            sampling_params=SamplingParams(),
        )
        config = MagicMock()
        config.sampling.enable_penalties = enable_penalties
        config.sampling.enable_min_tokens = False

        device = CPU()
        processor = FusedSamplingProcessor(
            sampler=MagicMock(),
            pipeline_config=config,
            context_batch=[context],
            device=device,
        )

        assert (processor.penalty_inputs is not None) == expect_penalty_inputs


class TestFusedSamplingProcessorAsyncCopy:
    """Tests for async token copy methods in FusedSamplingProcessor."""

    def _create_processor(
        self, device: CPU, pinned_new_tokens: Buffer | None = None
    ) -> FusedSamplingProcessor:
        """Create a FusedSamplingProcessor for testing."""
        context = TextContext(
            max_length=100,
            tokens=TokenBuffer(np.array([1, 2, 3], dtype=np.int64)),
            sampling_params=SamplingParams(),
        )
        config = MagicMock()
        config.sampling.enable_penalties = False
        config.sampling.enable_min_tokens = False

        return FusedSamplingProcessor(
            sampler=MagicMock(),
            pipeline_config=config,
            context_batch=[context],
            device=device,
            pinned_new_tokens=pinned_new_tokens,
        )

    def test_cpu_device_no_pinned_buffers(self) -> None:
        """On CPU device, no pinned buffers should be created."""
        device = CPU()
        processor = self._create_processor(device)

        # CPU device should not create pinned buffers
        assert processor._pinned_new_tokens is None
        assert processor._d2h_copy_event is None

    def test_get_new_tokens_numpy_fallback_on_cpu(self) -> None:
        """On CPU, get_new_tokens_numpy falls back to synchronous copy."""

        processor = self._create_processor(CPU())

        # Simulate sampler output by setting new_tokens
        # Shape is (batch_size,) = (1,) since we have 1 context
        test_tokens = np.array([42], dtype=np.int64)
        processor.new_tokens = Buffer.from_numpy(test_tokens)

        # Should return the tokens via synchronous fallback
        result = processor.get_new_tokens_numpy()

        np.testing.assert_array_equal(result, test_tokens)

    def test_start_async_then_get_tokens_on_cpu(self) -> None:
        """Full round-trip test on CPU: start async copy, then get tokens."""

        processor = self._create_processor(CPU())

        # Simulate sampler output
        # Shape is (batch_size,) = (1,) since we have 1 context
        test_tokens = np.array([10], dtype=np.int64)
        processor.new_tokens = Buffer.from_numpy(test_tokens)

        # Start async copy (no-op on CPU since no pinned buffer)
        processor.start_async_token_copy()

        # _d2h_copy_event should remain None since no pinned buffer exists
        assert processor._d2h_copy_event is None

        # Get tokens (synchronous fallback on CPU)
        result = processor.get_new_tokens_numpy()

        np.testing.assert_array_equal(result, test_tokens)

    def test_get_new_tokens_numpy_without_start_async(self) -> None:
        """get_new_tokens_numpy works even if start_async_token_copy not called."""

        processor = self._create_processor(CPU())

        # Simulate sampler output
        # Shape is (batch_size,) = (1,) since we have 1 context
        test_tokens = np.array([99], dtype=np.int64)
        processor.new_tokens = Buffer.from_numpy(test_tokens)

        # Don't call start_async_token_copy, just get tokens directly
        result = processor.get_new_tokens_numpy()

        np.testing.assert_array_equal(result, test_tokens)

    def test_start_async_noop_when_no_new_tokens(self) -> None:
        """start_async_token_copy is a no-op when new_tokens is None."""
        device = CPU()
        processor = self._create_processor(device)

        # new_tokens is None by default
        assert processor.new_tokens is None

        # Should not raise
        processor.start_async_token_copy()
        assert processor._d2h_copy_event is None


@pytest.mark.skipif(accelerator_count() == 0, reason="No GPU available")
class TestFusedSamplingProcessorAsyncCopyGPU:
    """GPU-specific tests for async token copy methods."""

    def _create_processor(
        self, device: Accelerator, pinned_new_tokens: Buffer | None = None
    ) -> FusedSamplingProcessor:
        """Create a FusedSamplingProcessor for GPU testing."""
        context = TextContext(
            max_length=100,
            tokens=TokenBuffer(np.array([1, 2, 3], dtype=np.int64)),
            sampling_params=SamplingParams(),
        )
        config = MagicMock()
        config.sampling.enable_penalties = False
        config.sampling.enable_min_tokens = False

        return FusedSamplingProcessor(
            sampler=MagicMock(),
            pipeline_config=config,
            context_batch=[context],
            device=device,
            pinned_new_tokens=pinned_new_tokens,
        )

    def test_gpu_device_uses_pinned_buffer(self) -> None:
        """On GPU device, processor uses provided pinned buffer."""
        device = Accelerator()
        pinned_new_tokens = DevicePinnedBuffer(
            shape=(1,), dtype=DType.int64, device=device
        )

        # Processor should use the provided pinned buffer
        processor = self._create_processor(device, pinned_new_tokens)
        assert processor._pinned_new_tokens is not None

    def test_async_token_copy_round_trip(self) -> None:
        """Full async copy round-trip: GPU buffer -> pinned -> numpy."""

        device = Accelerator()
        pinned_new_tokens = DevicePinnedBuffer(
            shape=(1,), dtype=DType.int64, device=device
        )
        processor = self._create_processor(device, pinned_new_tokens)

        # Create test tokens on GPU (simulating sampler output)
        # Shape must be (batch_size,) = (1,) since we have 1 context
        test_tokens = np.array([42], dtype=np.int64)
        processor.new_tokens = Buffer.from_numpy(test_tokens).to(device)

        # Start D2H copy and record event
        processor.start_async_token_copy()
        assert processor._d2h_copy_event is not None

        # Get tokens (waits for copy event)
        result = processor.get_new_tokens_numpy()
        assert processor._d2h_copy_event is None

        np.testing.assert_array_equal(result, test_tokens)

    def test_async_copy_multiple_calls(self) -> None:
        """Multiple async copy cycles work correctly."""

        device = Accelerator()
        pinned_new_tokens = DevicePinnedBuffer(
            shape=(1,), dtype=DType.int64, device=device
        )
        processor = self._create_processor(device, pinned_new_tokens)

        # Each iteration simulates one sampling step
        # Shape must be (batch_size,) = (1,) since we have 1 context
        for i in range(3):
            test_tokens = np.array([i * 10], dtype=np.int64)
            processor.new_tokens = Buffer.from_numpy(test_tokens).to(device)

            processor.start_async_token_copy()
            result = processor.get_new_tokens_numpy()

            np.testing.assert_array_equal(result, test_tokens)

    def test_get_tokens_without_async_start_falls_back(self) -> None:
        """On GPU, get_new_tokens_numpy works even without start_async_token_copy."""

        device = Accelerator()
        processor = self._create_processor(device)

        # Create test tokens on GPU
        # Shape must be (batch_size,) = (1,) since we have 1 context
        test_tokens = np.array([99], dtype=np.int64)
        processor.new_tokens = Buffer.from_numpy(test_tokens).to(device)

        # Async token copy was never started
        assert processor._d2h_copy_event is None

        # Don't call start_async_token_copy - should fall back to sync copy
        result = processor.get_new_tokens_numpy()

        np.testing.assert_array_equal(result, test_tokens)
