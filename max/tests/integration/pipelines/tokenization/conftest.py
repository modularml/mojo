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

import logging
from os import getenv
from pathlib import Path

import max.driver as md
import pytest
from max.engine import InferenceSession
from max.pipelines.lib import generate_local_model_path

MODULAR_AI_LLAMA_3_1_HF_REPO_ID = "modularai/Llama-3.1-8B-Instruct-GGUF"
LLAMA_3_1_HF_REPO_ID = "meta-llama/Llama-3.1-8B-Instruct"
# used in cases where float32 is needed. SmolLM3 is bfloat16.
SMOLLM_HF_REPO_ID = "HuggingFaceTB/SmolLM-135M"
SMOLLM2_HF_REPO_ID = "HuggingFaceTB/SmolLM2-135M"
DEEPSEEK_R1_DISTILL_LLAMA_8B_HF_REPO_ID = (
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)
LMSTUDIO_DEEPSEEK_R1_DISTILL_LLAMA_8B_HF_REPO_ID = (
    "lmstudio-community/DeepSeek-R1-Distill-Llama-8B-GGUF"
)
TINY_RANDOM_LLAMA_HF_REPO_ID = (
    "trl-internal-testing/tiny-random-LlamaForCausalLM"
)
GEMMA_3_1B_IT_HF_REPO_ID = "google/gemma-3-1b-it"
LLAMA_3_1_LORA_HF_REPO_ID = "FinGPT/fingpt-mt_llama3-8b_lora"
TINY_LLAMA_1_1B_CHAT_V1_0_HF_REPO_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TINY_RANDOM_LLAMA_FOR_CAUSAL_LM_HF_REPO_ID = (
    "neubla/tiny-random-LlamaForCausalLM"
)
QWQ_32B_HF_REPO_ID = "Qwen/QwQ-32B"
MISTRAL_NEMO_INSTRUCT_2407_HF_REPO_ID = "mistralai/Mistral-Nemo-Instruct-2407"
GOOGLE_GEMMA_3_4B_IT_HF_REPO_ID = "google/gemma-3-4b-it"

logger = logging.getLogger("max.pipelines")


@pytest.fixture
def modular_path() -> Path:
    """Returns the path to the Modular .derived directory."""
    modular_path = getenv("MODULAR_PATH")
    assert modular_path is not None

    return Path(modular_path)


@pytest.fixture
def mo_model_path(modular_path: Path) -> Path:
    """Returns the path to the generated BasicMLP model."""
    return modular_path / "max/tests/integration/API/Inputs/mo-model.mlir"


@pytest.fixture
def dynamic_model_path(modular_path: Path) -> Path:
    """Returns the path to the dynamic shape model."""
    return modular_path / "max/tests/integration/API/Inputs/dynamic-model.mlir"


@pytest.fixture
def no_input_path(modular_path: Path) -> Path:
    """Returns the path to a model spec without inputs."""
    return modular_path / "max/tests/integration/API/Inputs/no-inputs.mlir"


@pytest.fixture
def scalar_input_path(modular_path: Path) -> Path:
    """Returns the path to a model spec with scalar inputs."""
    return modular_path / "max/tests/integration/API/Inputs/scalar-input.mlir"


@pytest.fixture
def aliasing_outputs_path(modular_path: Path) -> Path:
    """Returns the path to a model spec with outputs that alias each other."""
    return (
        modular_path / "max/tests/integration/API/Inputs/aliasing-outputs.mlir"
    )


@pytest.fixture
def named_inputs_path(modular_path: Path) -> Path:
    """Returns the path to a model spec that adds a series of named tensors."""
    return modular_path / "max/tests/integration/API/Inputs/named-inputs.mlir"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--custom-ops-path",
        type=str,
        default="",
        help="Path to custom Ops package",
    )


@pytest.fixture(scope="module")
def session() -> InferenceSession:
    devices: list[md.Device] = []
    for i in range(md.accelerator_count()):
        devices.append(md.Accelerator(i))

    devices.append(md.CPU())

    return InferenceSession(devices=devices)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    # Prevent pytest from trying to collect Click commands and dataclasses as tests
    for item in items:
        if item.name.startswith("Test"):
            item.add_marker(pytest.mark.skip)


@pytest.fixture
def graph_testdata() -> Path:
    """Returns the path to the Modular .derived directory."""
    path = getenv("GRAPH_TESTDATA")
    assert path is not None
    return Path(path)


@pytest.fixture
def llama_3_1_8b_instruct_local_path() -> str:
    try:
        model_path = generate_local_model_path(LLAMA_3_1_HF_REPO_ID)
    except FileNotFoundError as e:
        logger.warning(f"Failed to generate local model path: {e}")
        logger.warning(
            f"Falling back to repo_id: {LLAMA_3_1_HF_REPO_ID} as config to PipelineConfig"
        )
        model_path = LLAMA_3_1_HF_REPO_ID
    return model_path


@pytest.fixture
def smollm_135m_local_path() -> str:
    try:
        model_path = generate_local_model_path(SMOLLM_HF_REPO_ID)
    except FileNotFoundError as e:
        logger.warning(f"Failed to generate local model path: {e}")
        logger.warning(
            f"Falling back to repo_id: {SMOLLM_HF_REPO_ID} as config to PipelineConfig"
        )
        model_path = SMOLLM_HF_REPO_ID
    return model_path


@pytest.fixture
def smollm2_135m_local_path() -> str:
    try:
        model_path = generate_local_model_path(SMOLLM2_HF_REPO_ID)
    except FileNotFoundError as e:
        logger.warning(f"Failed to generate local model path: {e}")
        logger.warning(
            f"Falling back to repo_id: {SMOLLM2_HF_REPO_ID} as config to PipelineConfig"
        )
        model_path = SMOLLM2_HF_REPO_ID
    return model_path


@pytest.fixture
def deepseek_r1_distill_llama_8b_local_path() -> str:
    try:
        model_path = generate_local_model_path(
            DEEPSEEK_R1_DISTILL_LLAMA_8B_HF_REPO_ID
        )
    except FileNotFoundError as e:
        logger.warning(f"Failed to generate local model path: {e}")
        logger.warning(
            f"Falling back to repo_id: {DEEPSEEK_R1_DISTILL_LLAMA_8B_HF_REPO_ID} as config to PipelineConfig"
        )
        model_path = DEEPSEEK_R1_DISTILL_LLAMA_8B_HF_REPO_ID
    return model_path


@pytest.fixture
def lmstudio_deepseek_r1_distill_llama_8b_local_path() -> str:
    try:
        model_path = generate_local_model_path(
            LMSTUDIO_DEEPSEEK_R1_DISTILL_LLAMA_8B_HF_REPO_ID
        )
    except FileNotFoundError as e:
        logger.warning(f"Failed to generate local model path: {e}")
        logger.warning(
            f"Falling back to repo_id: {LMSTUDIO_DEEPSEEK_R1_DISTILL_LLAMA_8B_HF_REPO_ID} as config to PipelineConfig"
        )
        model_path = LMSTUDIO_DEEPSEEK_R1_DISTILL_LLAMA_8B_HF_REPO_ID
    return model_path


@pytest.fixture
def modular_ai_llama_3_1_local_path() -> str:
    try:
        model_path = generate_local_model_path(MODULAR_AI_LLAMA_3_1_HF_REPO_ID)
    except FileNotFoundError as e:
        logger.warning(f"Failed to generate local model path: {e}")
        logger.warning(
            f"Falling back to repo_id: {MODULAR_AI_LLAMA_3_1_HF_REPO_ID} as config to PipelineConfig"
        )
        model_path = MODULAR_AI_LLAMA_3_1_HF_REPO_ID
    return model_path


@pytest.fixture
def tiny_random_llama_local_path() -> str:
    try:
        model_path = generate_local_model_path(TINY_RANDOM_LLAMA_HF_REPO_ID)
    except FileNotFoundError as e:
        logger.warning(f"Failed to generate local model path: {e}")
        logger.warning(
            f"Falling back to repo_id: {TINY_RANDOM_LLAMA_HF_REPO_ID} as config to PipelineConfig"
        )
        model_path = TINY_RANDOM_LLAMA_HF_REPO_ID
    return model_path


@pytest.fixture
def gemma_3_1b_it_local_path() -> str:
    try:
        model_path = generate_local_model_path(GEMMA_3_1B_IT_HF_REPO_ID)
    except FileNotFoundError as e:
        logger.warning(f"Failed to generate local model path: {e}")
        logger.warning(
            f"Falling back to repo_id: {GEMMA_3_1B_IT_HF_REPO_ID} as config to PipelineConfig"
        )
        model_path = GEMMA_3_1B_IT_HF_REPO_ID
    return model_path


@pytest.fixture
def llama_3_1_8b_lora_local_path() -> str:
    try:
        model_path = generate_local_model_path(LLAMA_3_1_LORA_HF_REPO_ID)
    except FileNotFoundError as e:
        logger.warning(f"Failed to generate local model path: {e}")
        logger.warning(
            f"Falling back to repo_id: {LLAMA_3_1_LORA_HF_REPO_ID} as config to PipelineConfig"
        )
        model_path = LLAMA_3_1_LORA_HF_REPO_ID
    return model_path


@pytest.fixture
def tiny_llama_1_1b_chat_v1_0_local_path() -> str:
    try:
        model_path = generate_local_model_path(
            TINY_LLAMA_1_1B_CHAT_V1_0_HF_REPO_ID
        )
    except FileNotFoundError as e:
        logger.warning(f"Failed to generate local model path: {e}")
        logger.warning(
            f"Falling back to repo_id: {TINY_LLAMA_1_1B_CHAT_V1_0_HF_REPO_ID} as config to PipelineConfig"
        )
        model_path = TINY_LLAMA_1_1B_CHAT_V1_0_HF_REPO_ID
    return model_path


@pytest.fixture
def tiny_random_llama_for_causal_lm_local_path() -> str:
    try:
        model_path = generate_local_model_path(
            TINY_RANDOM_LLAMA_FOR_CAUSAL_LM_HF_REPO_ID
        )
    except FileNotFoundError as e:
        logger.warning(f"Failed to generate local model path: {e}")
        logger.warning(
            f"Falling back to repo_id: {TINY_RANDOM_LLAMA_FOR_CAUSAL_LM_HF_REPO_ID} as config to PipelineConfig"
        )
        model_path = TINY_RANDOM_LLAMA_FOR_CAUSAL_LM_HF_REPO_ID
    return model_path


@pytest.fixture
def qwq_32b_local_path() -> str:
    try:
        model_path = generate_local_model_path(QWQ_32B_HF_REPO_ID)
    except FileNotFoundError as e:
        logger.warning(f"Failed to generate local model path: {e}")
        logger.warning(
            f"Falling back to repo_id: {QWQ_32B_HF_REPO_ID} as config to PipelineConfig"
        )
        model_path = QWQ_32B_HF_REPO_ID
    return model_path


@pytest.fixture
def mistral_nemo_instruct_2407_local_path() -> str:
    try:
        model_path = generate_local_model_path(
            MISTRAL_NEMO_INSTRUCT_2407_HF_REPO_ID
        )
    except FileNotFoundError as e:
        logger.warning(f"Failed to generate local model path: {e}")
        logger.warning(
            f"Falling back to repo_id: {MISTRAL_NEMO_INSTRUCT_2407_HF_REPO_ID} as config to PipelineConfig"
        )
        model_path = MISTRAL_NEMO_INSTRUCT_2407_HF_REPO_ID
    return model_path


@pytest.fixture
def google_gemma_3_4b_it_local_path() -> str:
    try:
        model_path = generate_local_model_path(GOOGLE_GEMMA_3_4B_IT_HF_REPO_ID)
    except FileNotFoundError as e:
        logger.warning(f"Failed to generate local model path: {e}")
        logger.warning(
            f"Falling back to repo_id: {GOOGLE_GEMMA_3_4B_IT_HF_REPO_ID} as config to PipelineConfig"
        )
        model_path = GOOGLE_GEMMA_3_4B_IT_HF_REPO_ID
    return model_path
