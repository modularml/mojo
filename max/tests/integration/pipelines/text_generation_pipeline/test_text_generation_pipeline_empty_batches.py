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
from typing import cast
from unittest.mock import MagicMock

import pytest
from max.driver import Buffer
from max.dtype import DType
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    KVCacheConfig,
    PipelineArgs,
    PipelineConfig,
    PipelineRuntimeConfig,
    TextGenerationPipeline,
)
from max.pipelines.lib.interfaces import ModelOutputs
from max.pipelines.lib.registry import PIPELINE_REGISTRY
from max.pipelines.modeling.types import PipelineTask, TextGenerationInputs


@pytest.fixture(scope="module")
def pipeline() -> TextGenerationPipeline[TextContext]:
    """Retrieve the text generation pipeline once for all tests in this module."""
    _, p = PIPELINE_REGISTRY.retrieve(
        PipelineConfig.from_args(
            PipelineArgs(
                model_path="kathywu95/deepseek-v3-small-random",
                kv_cache=KVCacheConfig(enable_prefix_caching=False),
                max_length=100,
                runtime=PipelineRuntimeConfig(max_batch_size=2),
            )
        ),
        task=PipelineTask.TEXT_GENERATION,
    )
    return cast(TextGenerationPipeline[TextContext], p)


@pytest.mark.parametrize(
    "use_mock_model", [True, False], ids=["mocked-model", "real-model"]
)
def test_text_generation_pipeline__empty_batches(
    use_mock_model: bool,
    monkeypatch: pytest.MonkeyPatch,
    pipeline: TextGenerationPipeline[TextContext],
) -> None:
    """Test that the DeepSeek V3 architecture and pipeline support empty batches, using both a mock model and the real DeepSeek V3 model.

    This test is designed to validate that the text generation pipeline robustly handles the case where empty batches are passed for text generation. It proceeds in two stages:
    1. It first runs the pipeline with the underlying model execution mocked, ensuring that pipeline logic alone correctly processes empty batches without errors.
    2. It then runs the same empty-batch scenario with the actual DeepSeek V3 model to confirm end-to-end support for empty batches.

    Both stages check that supplying empty batch dictionaries results in an empty output dictionary, confirming no errors are thrown and the API behaves as expected.

    To run this test:
    ```
    ./bazelw test //max/tests/integration/pipelines/text_generation_pipeline:test_empty_batches --test_output=all
    ```
    """

    # Optionally mock the underlying pipeline model's execute to avoid real forward pass
    if use_mock_model:

        def _fake_execute(
            *,
            model_inputs,  # noqa: ANN001
        ) -> ModelOutputs:  # matches call style execute(model_inputs=...)
            # Return empty-shaped tensors to mimic an empty batch execution
            empty_logits = Buffer.zeros((0, 0), DType.float32)
            return ModelOutputs(
                logits=empty_logits, next_token_logits=empty_logits
            )

        assert hasattr(pipeline, "_pipeline_model")
        assert hasattr(pipeline._pipeline_model, "execute")
        mock_execute = MagicMock(side_effect=_fake_execute)
        # Use monkeypatch to avoid assigning to method directly (keeps type-checkers happy)
        monkeypatch.setattr(
            pipeline._pipeline_model, "execute", mock_execute, raising=True
        )

    # Generate empty batches for inputs
    inputs: TextGenerationInputs[TextContext] = TextGenerationInputs(
        batches=[[], []],
    )

    # Execute the pipeline
    responses = pipeline.execute(inputs)

    # If we have no inputs, responses should be an empty dict.
    assert len(responses) == 0

    # If mocked, ensure our fake execute was used (and did not raise)
    if use_mock_model:
        # Ensure our fake execute was used at least once
        assert mock_execute.call_count >= 1
    else:
        # Ensure execute was not monkeypatched from a previous test run
        assert not isinstance(pipeline._pipeline_model.execute, MagicMock)
