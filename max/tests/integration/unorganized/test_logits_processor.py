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
"""Unit tests for logits processor functions using ProcessorInputs dataclass."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from max.driver import CPU, Buffer
from max.pipelines.context import (
    BatchProcessorInputs,
    ProcessorInputs,
    SamplingParams,
    TextContext,
    TokenBuffer,
)
from max.pipelines.sampling import SamplerInputs
from max.pipelines.sampling.logits_processor import apply_logits_processors


class MockTensor:
    """Mock tensor for testing purposes."""

    def __init__(self, shape: tuple[int, ...], dtype: str = "float32"):
        self.shape = shape
        self.dtype = dtype
        self.data = np.zeros(shape, dtype=dtype)

    def __getitem__(self, key: Any) -> Any:
        return self.data[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self.data[key] = value


def create_tensor(shape: tuple[int, ...], dtype: str = "float32") -> Buffer:
    return cast(Buffer, MockTensor(shape, dtype))


class TestLogitsProcessor:
    """Test suite for LogitsProcessor functions."""

    def test_simple_function_processor(self) -> None:
        """Test a simple function processor that modifies logits."""

        def simple_processor(inputs: ProcessorInputs) -> None:
            # Simple modification for testing
            assert inputs.logits is not None
            assert inputs.context is not None
            inputs.logits[0, 0] = -10000

        mock_tensor = create_tensor((2, 100))
        context = TextContext(
            max_length=100,
            tokens=TokenBuffer(np.array([42], dtype=np.int64)),
            sampling_params=SamplingParams(),
        )

        processor_inputs = ProcessorInputs(logits=mock_tensor, context=context)
        simple_processor(processor_inputs)

        assert mock_tensor[0][0] == -10000

    def test_processor_with_state(self) -> None:
        """Test LogitsProcessor with stateful callable (like the example in docstring)."""

        class SuppressBeginToken:
            def __init__(self, tokens_to_suppress: list[int], steps: int):
                self.tokens_to_suppress = tokens_to_suppress
                self.steps = steps
                self.step_counter = 0

            def __call__(self, inputs: ProcessorInputs) -> None:
                logits = inputs.logits

                logits_mock = cast(MockTensor, logits)
                if self.step_counter < self.steps:
                    logits_mock.data[:, self.tokens_to_suppress] = -10000
                    self.step_counter += 1

        processor = SuppressBeginToken([5, 7], steps=2)
        context = TextContext(
            max_length=100,
            tokens=TokenBuffer(np.array([42], dtype=np.int64)),
            sampling_params=SamplingParams(),
        )

        # First call should suppress tokens
        input_tensor = create_tensor((2, 100))
        processor_inputs = ProcessorInputs(logits=input_tensor, context=context)
        processor(processor_inputs)
        assert input_tensor[0, 0] == 0
        assert input_tensor[0, 5] == -10000
        assert input_tensor[0, 7] == -10000
        assert input_tensor[1, 0] == 0
        assert input_tensor[1, 5] == -10000
        assert input_tensor[1, 7] == -10000
        assert processor.step_counter == 1

        # Second call should still suppress tokens
        input_tensor = create_tensor((2, 100))
        processor_inputs = ProcessorInputs(logits=input_tensor, context=context)
        processor(processor_inputs)
        assert input_tensor[0, 0] == 0
        assert input_tensor[0, 5] == -10000
        assert input_tensor[0, 7] == -10000
        assert input_tensor[1, 0] == 0
        assert input_tensor[1, 5] == -10000
        assert input_tensor[1, 7] == -10000
        assert processor.step_counter == 2

        # Third call should not suppress tokens (exceeded steps)
        input_tensor = create_tensor((2, 100))
        processor_inputs = ProcessorInputs(logits=input_tensor, context=context)
        processor(processor_inputs)
        assert input_tensor[0, 0] == 0
        assert input_tensor[0, 5] == 0
        assert input_tensor[0, 7] == 0
        assert input_tensor[1, 0] == 0
        assert input_tensor[1, 5] == 0
        assert input_tensor[1, 7] == 0
        assert processor.step_counter == 2  # Should not increment further

    def test_multiple_processors_independence(self) -> None:
        """Test that multiple processors maintain independent state."""

        class CountingProcessor:
            def __init__(self):
                self.count = 0

            def __call__(self, inputs: ProcessorInputs) -> None:
                self.count += 1

        processor1 = CountingProcessor()
        processor2 = CountingProcessor()

        tensor = create_tensor((1, 100))
        context = TextContext(
            max_length=100,
            tokens=TokenBuffer(np.array([42], dtype=np.int64)),
            sampling_params=SamplingParams(),
        )

        processor_inputs = ProcessorInputs(logits=tensor, context=context)

        # Use processor1
        processor1(processor_inputs)
        processor1(processor_inputs)

        # Use processor2
        processor2(processor_inputs)

        assert processor1.count == 2
        assert processor2.count == 1


class TestApplyLogitsProcessors:
    """Test suite for apply_logits_processors function."""

    def create_context_batch(self):  # noqa: ANN201
        """Returns a batch of two contexts with different logits processors."""

        # In the functions below, each element must be individually assigned
        # because we don't support slicing assignment.

        def add_one(inputs: ProcessorInputs) -> None:
            logits = inputs.logits
            logits_np = logits.to_numpy()
            for i in range(logits.shape[0]):
                for j in range(logits.shape[1]):
                    logits[i, j] = logits_np[i, j] + 1

        def add_two(inputs: ProcessorInputs) -> None:
            logits = inputs.logits
            logits_np = logits.to_numpy()
            for i in range(logits.shape[0]):
                for j in range(logits.shape[1]):
                    logits[i, j] = logits_np[i, j] + 2

        def sub_one(inputs: ProcessorInputs) -> None:
            logits = inputs.logits
            logits_np = logits.to_numpy()
            for i in range(logits.shape[0]):
                for j in range(logits.shape[1]):
                    logits[i, j] = logits_np[i, j] - 1

        context_batch = [
            TextContext(
                max_length=100,
                tokens=TokenBuffer(np.array([42], dtype=np.int64)),
                sampling_params=SamplingParams(
                    logits_processors=[add_one, add_two]
                ),
            ),
            TextContext(
                max_length=100,
                tokens=TokenBuffer(np.array([42], dtype=np.int64)),
                sampling_params=SamplingParams(logits_processors=[sub_one]),
            ),
        ]
        return context_batch

    def test_apply_logits_processors_no_offsets(self) -> None:
        """Test apply_logits_processors with no offsets."""

        logits = Buffer.from_numpy(np.arange(10).reshape(2, 5))

        apply_logits_processors(self.create_context_batch(), logits, None)
        final_array = logits.to_numpy()
        expected_array = np.arange(10).reshape(2, 5)
        expected_array[0, :] += 3
        expected_array[1, :] -= 1

        assert np.all(final_array == expected_array)

    def test_apply_logits_processors_with_offsets(self) -> None:
        """Test apply_logits_processors with offsets."""

        # Assume these 3 logits are returned for the first context
        # and 2 logits are returned for the second context.
        logits = Buffer.from_numpy(np.arange(30).reshape(5, 6))
        logit_offsets = Buffer.from_numpy(np.array([0, 3, 5]))

        apply_logits_processors(
            self.create_context_batch(), logits, logit_offsets
        )
        final_array = logits.to_numpy()
        expected_array = np.arange(30).reshape(5, 6)
        expected_array[0:3, :] += 3
        expected_array[3:5, :] -= 1
        assert np.all(final_array == expected_array)

    def test_apply_logits_processors_with_batch_processors(self) -> None:
        """Test apply_logits_processors with batch processors."""

        logits = Buffer.from_numpy(np.arange(10).reshape(2, 5))
        logit_offsets = Buffer.from_numpy(np.array([0, 3, 5]))
        context_batch = [
            TextContext(
                max_length=100,
                tokens=TokenBuffer(np.array([42], dtype=np.int64)),
            ),
            TextContext(
                max_length=100,
                tokens=TokenBuffer(np.array([42], dtype=np.int64)),
            ),
        ]

        def add_one(inputs: BatchProcessorInputs) -> None:
            assert inputs.logits is logits
            assert inputs.logit_offsets is logit_offsets
            assert inputs.context_batch is context_batch
            logits_np = inputs.logits.to_numpy()
            for i in range(inputs.logits.shape[0]):
                for j in range(inputs.logits.shape[1]):
                    inputs.logits[i, j] = logits_np[i, j] + 1

        apply_logits_processors(context_batch, logits, logit_offsets, [add_one])
        final_array = logits.to_numpy()
        expected_array = np.arange(10).reshape(2, 5)
        expected_array[:] += 1

        assert np.all(final_array == expected_array)


def test_sampler_inputs_extracts_min_p_cpu() -> None:
    """Test SamplerInputs extracts per-request min_p values on CPU."""
    context_batch = [
        TextContext(
            max_length=16,
            tokens=TokenBuffer(np.array([1, 2, 3], dtype=np.int64)),
            sampling_params=SamplingParams(min_p=0.05),
        ),
        TextContext(
            max_length=16,
            tokens=TokenBuffer(np.array([4, 5], dtype=np.int64)),
            sampling_params=SamplingParams(min_p=0.3),
        ),
    ]

    sampler_inputs = SamplerInputs.create(context_batch, device=CPU())
    np.testing.assert_allclose(
        sampler_inputs.min_p.to_numpy(),
        np.array([0.05, 0.3], dtype=np.float32),
    )
