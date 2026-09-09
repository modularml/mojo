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

import numpy as np
import pytest
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    ops,
)
from max.pipelines.context import (
    BatchProcessorInputs,
    ProcessorInputs,
    SamplingParams,
    TextContext,
    TokenBuffer,
)
from max.pipelines.sampling.logits_processor import apply_logits_processors


@pytest.fixture(scope="module")
def update_gpu_logits(session: InferenceSession) -> Model:
    def _add_logits(logits: BufferValue, constant: TensorValue) -> None:
        logits_tensor = ops.buffer_load(logits)
        ops.buffer_store(logits, logits_tensor + constant.to(logits.device))

    replace_logits_graph = Graph(
        "add_logits",
        _add_logits,
        input_types=[
            BufferType(
                DType.float32, ("seq_len", "vocab_size"), DeviceRef.GPU()
            ),
            TensorType(DType.float32, (1,), DeviceRef.CPU()),
        ],
    )
    return session.load(replace_logits_graph)


class TestApplyLogitsProcessorsGPU:
    """Test suite for apply_logits_processors function."""

    def create_context_batch(
        self, update_gpu_logits: Model
    ) -> list[TextContext]:
        """Returns a batch of two contexts with different logits processors."""

        # In the functions below, each element must be individually assigned
        # because we don't support slicing assignment.

        def add_one(inputs: ProcessorInputs) -> None:
            print("before", inputs.logits.to_numpy())
            update_gpu_logits(
                inputs.logits,
                Buffer.from_numpy(np.array([1], dtype=np.float32)),
            )
            print("after", inputs.logits.to_numpy())

        def add_two(inputs: ProcessorInputs) -> None:
            update_gpu_logits(
                inputs.logits,
                Buffer.from_numpy(np.array([2], dtype=np.float32)),
            )

        def sub_one(inputs: ProcessorInputs) -> None:
            update_gpu_logits(
                inputs.logits,
                Buffer.from_numpy(np.array([-1], dtype=np.float32)),
            )

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

    def test_apply_logits_processors_no_offsets(
        self, update_gpu_logits: Model
    ) -> None:
        """Test apply_logits_processors with no offsets."""

        device = Accelerator()
        logits = Buffer.from_numpy(
            np.arange(10).reshape(2, 5).astype(np.float32)
        ).to(device)

        apply_logits_processors(
            self.create_context_batch(update_gpu_logits), logits, None
        )
        final_array = logits.to_numpy()
        expected_array = np.arange(10).reshape(2, 5)
        expected_array[0, :] += 3
        expected_array[1, :] -= 1

        assert np.all(final_array == expected_array)

    def test_apply_logits_processors_with_offsets(
        self, update_gpu_logits: Model
    ) -> None:
        """Test apply_logits_processors with offsets."""

        # Assume these 3 logits are returned for the first context
        # and 2 logits are returned for the second context.
        device = Accelerator()
        logits = Buffer.from_numpy(
            np.arange(30).reshape(5, 6).astype(np.float32)
        ).to(device)
        logit_offsets = Buffer.from_numpy(
            np.array([0, 3, 5]).astype(np.uint32)
        ).to(device)

        apply_logits_processors(
            self.create_context_batch(update_gpu_logits), logits, logit_offsets
        )
        final_array = logits.to_numpy()
        expected_array = np.arange(30).reshape(5, 6)
        expected_array[0:3, :] += 3
        expected_array[3:5, :] -= 1
        assert np.all(final_array == expected_array)

    def test_apply_logits_processors_with_batch_processors(
        self, update_gpu_logits: Model
    ) -> None:
        """Test apply_logits_processors with batch processors."""

        device = Accelerator()
        logits = Buffer.from_numpy(
            np.arange(10).reshape(2, 5).astype(np.float32),
        ).to(device)
        logit_offsets = Buffer.from_numpy(
            np.array([0, 3, 5]).astype(np.uint32)
        ).to(device)
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
            update_gpu_logits(
                inputs.logits,
                Buffer.from_numpy(np.array([1], dtype=np.float32)),
            )

        apply_logits_processors(context_batch, logits, logit_offsets, [add_one])
        final_array = logits.to_numpy()
        expected_array = np.arange(10).reshape(2, 5)
        expected_array[:] += 1

        assert np.all(final_array == expected_array)
