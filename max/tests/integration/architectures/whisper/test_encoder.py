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
from __future__ import annotations

import logging

import numpy as np
import pytest
import torch
from datasets import load_dataset
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.graph.weights import WeightData
from max.pipelines.architectures.whisper.encoder import WhisperEncoder
from max.pipelines.lib import generate_local_model_path
from transformers import AutoConfig, AutoModelForSpeechSeq2Seq, AutoProcessor

ACCURACY_RTOL = 1e-4
ACCURACY_ATOL = 1e-6

WHISPER_LARGE_V3_HF_REPO_ID = "openai/whisper-large-v3"

logger = logging.getLogger("max.pipelines")


@pytest.fixture
def whisper_large_v3_local_path() -> str:
    try:
        model_path = generate_local_model_path(WHISPER_LARGE_V3_HF_REPO_ID)
    except FileNotFoundError as e:
        logger.warning(f"Failed to generate local model path: {str(e)}")
        logger.warning(
            f"Falling back to repo_id: {WHISPER_LARGE_V3_HF_REPO_ID} as config to PipelineConfig"
        )
        model_path = WHISPER_LARGE_V3_HF_REPO_ID
    return model_path


@pytest.fixture
def torch_inputs(whisper_large_v3_local_path: str) -> torch.Tensor:
    """Returns 2 audio files in a tensor of shape = (batch_size=2, n_features=128, seq_length=3000)"""
    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    )
    audio_samples = [
        ds[0]["audio"]["array"],  # type: ignore
        ds[1]["audio"]["array"],  # type: ignore
    ]

    processor = AutoProcessor.from_pretrained(whisper_large_v3_local_path)
    inputs = processor(
        audio_samples,
        return_attention_mask=True,
        sampling_rate=ds[0]["audio"]["sampling_rate"],  # type: ignore
        return_tensors="pt",
    )

    input_features = inputs["input_features"]
    return input_features


@pytest.fixture
def graph_api_inputs(torch_inputs: torch.Tensor) -> torch.Tensor:
    """Returns 2 audio files in a tensor of shape = (batch_size=2, seq_length=3000, n_features=128)"""
    return torch.permute(torch_inputs, (0, 2, 1)).contiguous()


@pytest.mark.skip(
    reason="We decided to postpone finishing Whisper bring up. Should debug if we come back to it."
)
def test_whisper_encoder(
    torch_inputs: torch.Tensor,
    graph_api_inputs: torch.Tensor,
    whisper_large_v3_local_path: str,
) -> None:
    huggingface_config = AutoConfig.from_pretrained(whisper_large_v3_local_path)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        whisper_large_v3_local_path
    )

    torch_outputs = model.model.encoder(torch_inputs).last_hidden_state

    # TODO: Need to construct state dict.
    state_dict: dict[str, WeightData] = {}

    graph_api_model = WhisperEncoder(
        huggingface_config=huggingface_config,
        dtype=DType.float32,
        device=DeviceRef.CPU(),
    )
    graph_api_model.load_state_dict(state_dict)

    session = InferenceSession(devices=[CPU()])
    with Graph(
        name="encoder",
        input_types=(
            TensorType(
                DType.from_numpy(graph_api_inputs.numpy().dtype),
                graph_api_inputs.shape,
                device=DeviceRef.CPU(),
            ),
        ),
    ) as graph:
        graph.output(graph_api_model(graph.inputs[0].tensor)[0])

    compiled = session.load(graph, weights_registry=state_dict)

    graph_api_output = compiled.execute(graph_api_inputs)[0]
    assert isinstance(graph_api_output, Buffer)

    np.testing.assert_allclose(
        graph_api_output.to_numpy(),
        torch_outputs.detach().numpy(),
        equal_nan=True,
        rtol=ACCURACY_RTOL,
        atol=ACCURACY_ATOL,
    )
