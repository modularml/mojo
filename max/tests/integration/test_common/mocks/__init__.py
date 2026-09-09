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
"""Utilities for working with mocks for unit testing"""

from collections.abc import Generator
from contextlib import contextmanager

from max.driver import DeviceSpec, scan_available_devices
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    MemoryPlan,
    TextGenerationPipeline,
    generate_local_model_path,
)

from .pipeline_config import (
    DummyMAXModelConfig,
    DummyPipelineConfig,
    mock_hf_repo_access,
    mock_huggingface_config,
    mock_huggingface_hub_repo_exists_with_retry,
    mock_pipeline_config_hf_dependencies,
    mock_pipeline_config_resolve,
    mock_plan_from_sizes,
    patched_hf_construction,
)
from .pipeline_model import MOCK_MODEL_MAX_SEQ_LEN, MockPipelineModel
from .tokenizer import MockTextTokenizer

REPO_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


@contextmanager
def retrieve_mock_text_generation_pipeline(
    vocab_size: int,
    eos_token: int,
    seed: int = 42,
    eos_prob: float = 0.1,
    max_length: int | None = None,
    max_new_tokens: int | None = None,
    device_specs: list[DeviceSpec] | None = None,
) -> Generator[tuple[MockTextTokenizer, TextGenerationPipeline], None, None]:  # type: ignore[type-arg]
    if eos_token > vocab_size:
        raise ValueError(
            f"eos_token provided '{eos_token}' must be less than vocab_size provided '{vocab_size}'"
        )

    if not device_specs:
        device_specs = scan_available_devices()

    mock_config = DummyPipelineConfig(
        model_path=generate_local_model_path(REPO_ID),
        max_length=max_length,
        max_batch_size=None,
        device_specs=device_specs,
        quantization_encoding="float32",
        eos_prob=eos_prob,
        vocab_size=vocab_size,
        eos_token=eos_token,
    )

    tokenizer = MockTextTokenizer(
        max_new_tokens=max_new_tokens,
        seed=seed,
        vocab_size=vocab_size,
        max_length=max_length,
    )

    try:
        pipeline: TextGenerationPipeline[TextContext] = TextGenerationPipeline(
            pipeline_config=mock_config,
            pipeline_model=MockPipelineModel,
            weight_adapters={},
            tokenizer=tokenizer,
            # Plans are always populated with the model's effective bound;
            # mirror the mock model's clamp when the caller sets no length.
            memory_plan=MemoryPlan(
                planned_max_batch_size=mock_config.runtime.max_batch_size or 1,
                footprint=0,
                planned_max_length=(
                    max_length
                    if max_length is not None
                    else MOCK_MODEL_MAX_SEQ_LEN
                ),
                device_specs=tuple(device_specs),
            ),
        )

        yield tokenizer, pipeline
    finally:
        ...


__all__ = [
    "DummyMAXModelConfig",
    "DummyPipelineConfig",
    "MockTextTokenizer",
    "mock_hf_repo_access",
    "mock_huggingface_config",
    "mock_huggingface_hub_repo_exists_with_retry",
    "mock_pipeline_config_hf_dependencies",
    "mock_pipeline_config_resolve",
    "mock_plan_from_sizes",
    "patched_hf_construction",
    "retrieve_mock_text_generation_pipeline",
]
