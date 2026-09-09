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
# DOC: max/develop/logit-comparison.mdx

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from max.pipelines import (
    PIPELINE_REGISTRY,
    GenerateMixin,
    PipelineArgs,
    PipelineConfig,
)
from max.pipelines.context import (
    ProcessorInputs,
    SamplingParams,
    TextContext,
)
from max.pipelines.modeling.types import RequestID, TextGenerationRequest
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "Qwen/Qwen3-0.6B-Base"
CUSTOM_ARCHITECTURES: list[str] = []
PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a faraway kingdom,",
    "The quick brown fox jumps over the lazy",
]

NUM_STEPS = 5


@dataclass
class LogitComparison:
    prompt: str
    avg_abs_mae: float
    avg_cos_dist: float
    avg_kl_div: float
    max_kl_div: float


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.sum(a * b, axis=-1)
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return float(np.mean(1.0 - dot / denom))


def smooth_softmax(logits: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=-1, keepdims=True)
    return (1 - logits.shape[-1] * eps) * probs + eps


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.sum(p * np.log(p / q), axis=-1)


class CaptureGeneratedLogits:
    """Captures up to `max_steps` next-token logits per request."""

    def __init__(self, max_steps: int) -> None:
        self.max_steps = max_steps
        self.captured: dict[RequestID, list[np.ndarray]] = {}

    def __call__(self, inputs: ProcessorInputs) -> None:
        bucket = self.captured.setdefault(inputs.context.request_id, [])
        if len(bucket) < self.max_steps:
            bucket.append(inputs.logits[-1, :].to_numpy().copy())


def max_logits_for(
    prompt: str,
    pipeline: GenerateMixin[TextContext, TextGenerationRequest],
) -> np.ndarray:
    capture = CaptureGeneratedLogits(NUM_STEPS)
    request = TextGenerationRequest(
        request_id=RequestID(),
        model_name=MODEL_PATH,
        prompt=prompt,
        sampling_params=SamplingParams(
            max_new_tokens=NUM_STEPS,
            ignore_eos=True,
            top_k=1,
            logits_processors=[capture],
        ),
    )
    pipeline.generate([request])
    return np.stack(capture.captured[request.request_id]).astype(np.float64)


def torch_logits_for(prompt: str, max_logits: np.ndarray) -> np.ndarray:
    inputs = hf_tokenizer(prompt, return_tensors="pt").to(device)
    step_logits: list[np.ndarray] = []

    for step in range(NUM_STEPS):
        with torch.no_grad():
            outputs = hf_model(**inputs)

        logits = outputs.logits[0, -1, :].to(torch.float64).cpu().numpy()
        step_logits.append(logits)

        next_token = int(np.argmax(max_logits[step]))
        next_token_tensor = torch.tensor(
            [[next_token]], device=inputs["input_ids"].device
        )
        inputs["input_ids"] = torch.cat(
            [inputs["input_ids"], next_token_tensor], dim=-1
        )
        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], torch.ones_like(next_token_tensor)],
            dim=-1,
        )

    return np.stack(step_logits)


def compare_prompt(
    prompt: str,
    pipeline: GenerateMixin[TextContext, TextGenerationRequest],
) -> LogitComparison:
    max_logits = max_logits_for(prompt, pipeline)
    hf_logits = torch_logits_for(prompt, max_logits)

    kl_by_step = kl_divergence(
        smooth_softmax(hf_logits),
        smooth_softmax(max_logits),
    )

    return LogitComparison(
        prompt=prompt,
        avg_abs_mae=float(np.mean(np.abs(max_logits - hf_logits))),
        avg_cos_dist=cosine_distance(max_logits, hf_logits),
        avg_kl_div=float(np.mean(kl_by_step)),
        max_kl_div=float(np.max(kl_by_step)),
    )


def print_report(results: list[LogitComparison]) -> None:
    header = (
        f"{'prompt':<45} {'avg_abs_mae':>12} "
        f"{'avg_cos_dist':>12} {'avg_kl_div':>12} {'max_kl_div':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.prompt[:45]:<45} {r.avg_abs_mae:>12.3e} "
            f"{r.avg_cos_dist:>12.3e} {r.avg_kl_div:>12.3e} {r.max_kl_div:>12.3e}"
        )


pipeline_config = PipelineArgs.from_flat_kwargs(
    model_path=MODEL_PATH,
    custom_architectures=CUSTOM_ARCHITECTURES,
)
_, retrieved_pipeline = PIPELINE_REGISTRY.retrieve(
    PipelineConfig.from_args(pipeline_config)
)
assert isinstance(retrieved_pipeline, GenerateMixin)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype="auto"
).to(device)
hf_model.eval()

results = [compare_prompt(prompt, retrieved_pipeline) for prompt in PROMPTS]
print_report(results)
