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
"""Utilities for evaluating models and comparing the logits."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Sequence
from dataclasses import replace
from io import BytesIO
from typing import Any, TypeVar

import numpy as np
from max import pipelines
from max.driver import Device
from max.dtype import DType
from max.engine import InferenceSession
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
    LogitsProcessor,
    ProcessorInputs,
    SamplingParams,
)
from max.pipelines.lib import PixelGenerationTokenizer
from max.pipelines.modeling.types import (
    PipelineTokenizer,
    PixelGenerationInputs,
    RequestID,
    TextGenerationRequest,
)
from max.support import fetch_bytes_from_s3
from PIL import Image
from transformers import PreTrainedTokenizerBase

from .model_output import (
    ModelOutput,
    TokenInfo,
)
from .numerics import log_softmax
from .test_data import MockPixelGenerationRequest, MockTextGenerationRequest


def _decode_base64_image_to_numpy(image_data: str) -> np.ndarray:
    """Decode base64-encoded image to numpy array (H, W, C) float32 [0, 1].

    Mirrors the decode logic used in simple_offline_generation.save_image,
    but returns numpy for evaluation/comparison instead of saving to file.
    """
    image_bytes = base64.b64decode(image_data)
    pil_image = Image.open(BytesIO(image_bytes))
    img_array = np.array(pil_image, dtype=np.float32) / 255.0
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    return img_array


def _load_input_image(image_uri: str) -> Image.Image:
    """Load an input image for image-to-image requests."""
    if image_uri.startswith("s3://"):
        image_bytes = fetch_bytes_from_s3(image_uri)
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    return Image.open(image_uri).convert("RGB")


NUM_STEPS = 10

T = TypeVar("T")


def _create_batches(
    requests: Sequence[T], batch_sizes: int | list[int] = 1
) -> list[Sequence[T]]:
    """Group requests into batches."""
    if isinstance(batch_sizes, list):
        if sum(batch_sizes) != len(requests):
            raise ValueError(
                "The sum of the batch sizes must be equal to the number of requests."
            )
    else:
        batch_sizes = [batch_sizes] * len(requests)
    batches = []

    start = 0
    for size in batch_sizes:
        batches.append(requests[start : start + size])
        start += size
    return batches


def _create_requests(
    ids: Sequence[RequestID],
    requests: Sequence[MockTextGenerationRequest],
    num_steps: int,
    reference_by_id: dict[RequestID, ModelOutput] | None = None,
    logits_processors: list[LogitsProcessor] | None = None,
) -> list[TextGenerationRequest]:
    """Create text generation requests.

    Will correctly set `max_new_tokens` if reference outputs are provided.
    """
    text_generation_requests = []
    for id, request in zip(ids, requests, strict=True):
        if reference_by_id:
            assert reference_by_id[id].values is not None
            max_new_tokens = len(reference_by_id[id]["values"])
        else:
            max_new_tokens = num_steps
        text_generation_requests.append(
            request.to_text_generation_request(
                id,
                SamplingParams(
                    ignore_eos=True,
                    top_k=1,
                    max_new_tokens=max_new_tokens,
                    logits_processors=logits_processors,
                ),
            )
        )
    return text_generation_requests


def run_model(
    pipeline: pipelines.TextGenerationPipelineInterface,  # type: ignore[type-arg]
    tokenizer: PipelineTokenizer,  # type: ignore[type-arg]
    requests: Sequence[MockTextGenerationRequest],
    num_steps: int = NUM_STEPS,
    print_outputs: bool = False,
    batch_size: int | list[int] = 1,
    reference: list[ModelOutput] | None = None,
    generate_logprobs: bool = False,
) -> list[dict[str, Any]]:
    """Runs the pipeline for N steps on each request provided."""
    assert hasattr(tokenizer, "delegate")
    hf_tokenizer = tokenizer.delegate
    assert isinstance(hf_tokenizer, PreTrainedTokenizerBase)

    ids = [RequestID() for _ in requests]
    prompts_by_id = {
        id: request.prompt for id, request in zip(ids, requests, strict=True)
    }
    stored_logits = StoreLogits(
        ids, tokenizer, generate_logprobs=generate_logprobs
    )

    logits_processors: list[LogitsProcessor]
    if reference:
        reference_by_id = {
            id: reference for id, reference in zip(ids, reference, strict=True)
        }
        replace_logits = ReplaceLogitsWithReference(
            pipeline._devices, reference_by_id
        )
        logits_processors = [stored_logits, replace_logits]
    else:
        logits_processors = [stored_logits]
        reference_by_id = None

    batched_requests = _create_batches(requests, batch_size)
    batched_ids = _create_batches(ids, batch_size)

    for ids_in_batch, requests_in_batch in zip(
        batched_ids, batched_requests, strict=True
    ):
        batch = _create_requests(
            ids_in_batch,
            requests_in_batch,
            num_steps,
            reference_by_id,
            logits_processors,
        )
        outputs = pipeline.generate(batch)
        if print_outputs:
            for j in range(len(batch)):
                request = requests_in_batch[j]
                prompt = request.prompt
                print(
                    "Prompt:",
                    f"{prompt[:100]}...{prompt[-100:]}"
                    if len(prompt) > 200
                    else prompt,
                )
                print(
                    "Output:",
                    tokenizer.delegate.decode(
                        outputs[j].tokens, skip_special_tokens=True
                    ),
                )

    results: list[dict[str, Any]] = []
    for req_id, values in stored_logits.values.items():
        results.append({"prompt": prompts_by_id[req_id], "values": values})
    return results


class StoreLogits:
    def __init__(
        self,
        ids: Sequence[RequestID],
        tokenizer: PipelineTokenizer,  # type: ignore[type-arg]
        generate_logprobs: bool = False,
    ) -> None:
        self.values: dict[RequestID, list[TokenInfo]] = {id: [] for id in ids}
        self.tokenizer = tokenizer
        self.generate_logprobs = generate_logprobs

    def __call__(self, inputs: ProcessorInputs) -> None:
        logits = inputs.logits
        context = inputs.context
        # Don't do anything if the context is already finished.
        # This code path is possible due to overlap scheduling possibly executing
        # one more forward pass than needed.
        if context.is_done:
            return
        next_token_logits = logits[-1, :].to_numpy().copy()
        next_token = next_token_logits.argmax(axis=-1)

        entry: TokenInfo = {
            # We record the base next_token here.
            # If it deviates from the reference, we want to see that.
            "next_token": next_token,
            "next_token_logits": next_token_logits[next_token],
            "logits": next_token_logits,
        }

        if self.generate_logprobs:
            next_token_logprobs_array = log_softmax(next_token_logits)
            entry["next_token_logprobs"] = float(
                next_token_logprobs_array[next_token]
            )
            entry["logprobs"] = next_token_logprobs_array

        self.values[context.request_id].append(entry)


class ReplaceLogitsWithReference:
    def __init__(
        self,
        devices: Sequence[Device],
        reference_by_id: dict[RequestID, ModelOutput],
    ) -> None:
        self.reference_by_id = reference_by_id
        self.step_by_id = {id: 0 for id in reference_by_id}
        device_ref = DeviceRef.from_device(devices[0])

        def _replace_logits(
            logits: BufferValue, next_token: TensorValue
        ) -> None:
            logits[-1, next_token] = ops.constant(
                1e5, dtype=DType.float32, device=device_ref
            )

        replace_logits_graph = Graph(
            "replace_logits",
            _replace_logits,
            input_types=[
                BufferType(
                    DType.float32, ("seq_len", "vocab_size"), device_ref
                ),
                TensorType(DType.int64, (), DeviceRef.CPU()),
            ],
        )
        session = InferenceSession(devices=devices)
        self.replace_logits = session.load(replace_logits_graph)

    def __call__(self, inputs: ProcessorInputs) -> None:
        logits = inputs.logits
        context = inputs.context
        # Don't do anything if the context is already finished.
        # This code path is possible due to overlap scheduling possibly executing
        # one more forward pass than needed.
        if context.is_done:
            return
        # Assign the argmax of the reference to the logits.
        reference = self.reference_by_id[context.request_id]
        step = self.step_by_id[context.request_id]
        next_token = reference["values"][step]["next_token"]
        self.replace_logits(logits, next_token)
        self.step_by_id[context.request_id] += 1


def run_pixel_generation(
    pipeline: pipelines.PixelGenerationPipeline,  # type: ignore[type-arg]
    tokenizer: PixelGenerationTokenizer,
    requests: list[MockPixelGenerationRequest],
    num_steps: int,
    print_outputs: bool = False,
) -> list[dict[str, Any]]:
    """Run pixel generation using MAX PixelGenerationPipeline.

    Args:
        pipeline: MAX PixelGenerationPipeline instance
        tokenizer: MAX PixelGenerationTokenizer instance
        requests: List of MockPixelGenerationRequest objects
        num_steps: Number of inference steps (denoising steps)
        print_outputs: Whether to print outputs

    Returns:
        List of dicts with prompt and generated images
    """

    results = []

    for mock_request in requests:
        prompt = mock_request.prompt
        # Override num_steps if explicitly provided (for verification with different step counts)
        if num_steps != mock_request.num_inference_steps:
            mock_request = replace(mock_request, num_inference_steps=num_steps)

        if print_outputs:
            print(
                f'Generating image for prompt (steps={num_steps}): "{prompt}"'
            )

        # Convert MockPixelGenerationRequest to OpenResponsesRequest
        request_id = RequestID()
        request = mock_request.to_open_responses_request(
            request_id=request_id,
            model_name=tokenizer.model_path,
        )

        # Create context from request using tokenizer
        input_image = (
            _load_input_image(mock_request.input_image)
            if mock_request.input_image is not None
            else None
        )
        context = asyncio.run(tokenizer.new_context(request, input_image))

        if print_outputs:
            print(
                f"Context created: {context.height}x{context.width}, "
                f"{context.num_inference_steps} steps, guidance={context.guidance_scale}"
            )

        # Prepare inputs for the pipeline
        inputs = PixelGenerationInputs(batch={context.request_id: context})

        # Execute the pipeline
        outputs = pipeline.execute(inputs)

        # Get the output for our request
        output = outputs[context.request_id]

        pixel_data_list: list[np.ndarray] = []
        for image_content in output.output:
            if image_content.type != "output_image":
                raise ValueError(
                    f"Unexpected output content type '{image_content.type}'"
                )

            if image_content.image_data:
                pixel_data_list.append(
                    _decode_base64_image_to_numpy(image_content.image_data)
                )
            elif image_content.image_url:
                raise ValueError("image_url not supported")
            else:
                raise ValueError("No image data or URL in output")

        if not pixel_data_list:
            raise ValueError("No pixel data in output")

        pixel_data = np.stack(pixel_data_list, axis=0)

        if print_outputs:
            print(
                f"Generated image shape: {pixel_data.shape}, "
                f"dtype: {pixel_data.dtype}, range: [{pixel_data.min():.3f}, {pixel_data.max():.3f}]"
            )

        # Take first image from batch dimension
        image_np = pixel_data[0]  # Shape: (H, W, C)

        results.append({"prompt": prompt, "images": image_np})

    return results
