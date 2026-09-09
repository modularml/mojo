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
"""Test data for pipeline evaluation.

Separates test data from business logic for better maintainability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from max.pipelines.context import SamplingParams
from max.pipelines.modeling.types import (
    ImageContentPart,
    RequestID,
    TextContentPart,
    TextGenerationRequest,
    TextGenerationRequestMessage,
)
from max.pipelines.request import (
    OpenResponsesRequest,
    OpenResponsesRequestBody,
)
from max.pipelines.request.provider_options import (
    ImageProviderOptions,
    ProviderOptions,
)
from test_common.storage import load_bytes


@dataclass(frozen=True)
class MockTextGenerationRequest:
    """Request for text generation testing, supporting both text-only and multimodal inputs."""

    prompt: str
    """The text prompt to be processed by the model."""

    images: list[str]
    """List of image URLs or file paths. Empty for text-only requests."""

    messages: list[TextGenerationRequestMessage]
    """List of messages to be processed by the model. If this is provided, the
    prompt is used to identify the request, while the messages are processed by
    the model."""

    is_multimodal: bool

    model_name: str = ""

    @classmethod
    def text_only(cls, prompt: str) -> MockTextGenerationRequest:
        """Creates a text-only generation request."""
        return cls(prompt=prompt, images=[], messages=[], is_multimodal=False)

    @classmethod
    def with_images(
        cls,
        prompt: str,
        images: list[str],
        messages: list[dict[str, str | Any]] | None = None,
    ) -> MockTextGenerationRequest:
        """Creates a multimodal generation request.

        Images are embedded in message content with their URLs/paths for
        extraction later when converting to TextGenerationRequest.
        """
        if messages is None:
            proper_messages = [
                TextGenerationRequestMessage(
                    role="user",
                    content=[
                        TextContentPart(text=prompt),
                        *(ImageContentPart() for _ in images),
                    ],
                )
            ]
        else:
            proper_messages = []
            for message in messages:
                if isinstance(message, dict):
                    proper_messages.append(
                        TextGenerationRequestMessage(**message)
                    )

        return cls(
            prompt=prompt,
            images=images,
            messages=proper_messages,
            is_multimodal=True,
        )

    @classmethod
    def with_messages(
        cls, prompt: str, messages: list[dict[str, Any]], is_multimodal: bool
    ) -> MockTextGenerationRequest:
        """Creates a generation request with messages.

        Note that the prompt still needs to be passed in since it is used to
        identify the request.
        """
        # Extract image URLs/paths from messages content

        return cls(
            prompt=prompt,
            images=[],
            messages=[
                cast(TextGenerationRequestMessage, message)
                for message in messages
            ],
            is_multimodal=is_multimodal,
        )

    def to_text_generation_request(
        self, request_id: RequestID, sampling_params: SamplingParams
    ) -> TextGenerationRequest:
        if self.messages:
            return TextGenerationRequest(
                request_id=request_id,
                model_name=self.model_name,
                sampling_params=sampling_params,
                messages=self.messages,
                # `load_bytes` rather than `max.support.fetch_bytes_from_s3`:
                # it caches under ~/.cache/modular/testdata and falls back to
                # anonymous credentials for this public bucket, which is what
                # the torch half of logit verification already does. Without it
                # the MAX half of a multimodal comparison needs an AWS SSO
                # session and the torch half does not.
                images=[load_bytes(img) for img in self.images],
            )
        else:
            return TextGenerationRequest(
                request_id=request_id,
                model_name=self.model_name,
                sampling_params=sampling_params,
                prompt=self.prompt,
            )


@dataclass(frozen=True)
class MockPixelGenerationRequest:
    """Request for pixel generation testing."""

    prompt: str
    """The text prompt for image generation."""

    secondary_prompt: str | None = None
    """Optional secondary text prompt for dual text encoders."""

    negative_prompt: str | None = None
    """Optional negative prompt to guide what NOT to generate."""

    secondary_negative_prompt: str | None = None
    """Optional secondary negative prompt."""

    height: int | None = None
    """Height of generated image in pixels."""

    width: int | None = None
    """Width of generated image in pixels."""

    num_inference_steps: int = 50
    """Number of denoising steps."""

    guidance_scale: float = 3.5
    """Guidance scale for classifier-free guidance."""

    true_cfg_scale: float = 1.0
    """True CFG scale."""

    seed: int | None = None
    """Random seed for reproducibility."""

    input_image: str | None = None
    """Optional input image URI for image-to-image generation."""

    model_name: str = ""
    """Model name for the request."""

    @classmethod
    def from_prompt(
        cls,
        prompt: str,
        *,
        secondary_prompt: str | None = None,
        negative_prompt: str | None = None,
        secondary_negative_prompt: str | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        true_cfg_scale: float = 1.0,
        seed: int | None = None,
        input_image: str | None = None,
        model_name: str = "",
    ) -> MockPixelGenerationRequest:
        """Creates a pixel generation request from a prompt."""
        return cls(
            prompt=prompt,
            secondary_prompt=secondary_prompt,
            negative_prompt=negative_prompt,
            secondary_negative_prompt=secondary_negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            true_cfg_scale=true_cfg_scale,
            seed=seed,
            input_image=input_image,
            model_name=model_name,
        )

    def to_open_responses_request(
        self, request_id: RequestID, model_name: str | None = None
    ) -> OpenResponsesRequest:
        """Convert to an OpenResponsesRequest for pixel generation."""
        body = OpenResponsesRequestBody(
            model=model_name or self.model_name,
            input=self.prompt,
            seed=self.seed,
            provider_options=ProviderOptions(
                image=ImageProviderOptions(
                    negative_prompt=self.negative_prompt,
                    secondary_prompt=self.secondary_prompt,
                    secondary_negative_prompt=self.secondary_negative_prompt,
                    height=self.height,
                    width=self.width,
                    steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    true_cfg_scale=self.true_cfg_scale,
                    # Use PNG (lossless) for verification so MAE/RMSE/SSIM/LPIPS
                    # compare actual VAE output, not JPEG-compressed bytes.  The
                    # serving default is still JPEG.
                    output_format="png",
                )
            ),
        )
        return OpenResponsesRequest(request_id=request_id, body=body)


# Existing test data extracted from evaluate.py
LONG_TEXT_PROMPT = """One of the most important aspects of performance benchmarking when it pertains to comparison of different implementations is making sure comparisons are fair. This is a place where most discussions occur, as deviation from best practices can make one’s performance claims easy to dismiss. For faster results of a given implementation (the Mojo implementation in our case) to be meaningful, the comparison needs to be apples-to-apples.
    * Make sure you use equivalent optimization flags across implementations; even though flags (like -O3 in C) that enable multiple optimizations at once cannot always be equivalent to another language’s -O3, make sure you don’t compare something like a debug build with an implementation that uses the fast optimization flag.
    * Make sure that if one implementation has auto-vectorization or automatic multithreading enabled the same applies to all implementations to be compared (unless for a given language one of these performs worse when turned-on, in which case one could keep the fastest implementation for comparison purposes).
    * Use the latest (or best) combination of compilers, libraries, etc. — an older compiler version (for example) may perform better for whatever reason; however it should be considered sufficient to test with the latest stable version. One can test with older or experimental versions if they are so inclined.
    * Use the same input file (if applicable) or same input data. Avoid random data generation that may stress different code paths.
    * Use the same algorithm (if applicable) across all your implementations.
    * Use equivalent error testing as it applies to different domains’ best practices (e.g., input sanitizing, corner case testing).
    * Remove any unnecessary I/O (e.g., writing to file/screen for debug purposes) and keep only what is practically necessary — make sure you do so in a manner that code is not optimized out (see #6)!
    * Try to apply the same level of manual optimization (within reason) — if you write multi-threaded/vectorized code in Mojo, you should try to compare it to an equivalent implementation of the other language. There is a case to be made here, however, if the other language does not have such capabilities or they are so difficult to use that implementing them is beyond what one can reasonably do. This can highlight the programmability aspect of Mojo (or one language against another more generally), but this fact should be listed so that people can take the performance claims under this light."""
SHORT_TEXT_PROMPTS = (
    "def is_prime(x):\n",
    "The meaning of life is ",
    """Translate the English text to Italian.
    Text: Sometimes, I've believed as many as six impossible things before breakfast.
    Translation:""",
)
MULTIMODAL_PROMPT = (
    "<|image|><|begin_of_text|>If I had to write a haiku for this one"
)


MULTIMODAL_IMAGE = "s3://modular-bazel-artifacts-public/artifacts/model_testdata/multimodal_image.jpg"

PIXTRAL_PROMPT = "<s>[INST]Describe the images.\n[IMG][/INST]"
PIXTRAL_MESSAGES = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the images."},
            {"type": "image"},
        ],
    }
]
PIXTRAL_IMAGE = "s3://modular-bazel-artifacts-public/artifacts/model_testdata/pixtral_image.jpg"

INTERNVL_INSTRUCT_PROMPT = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|image|> Describe the image; where are these people and what are they doing?<|im_end|>\n<|im_start|>assistant\n"
INTERNVL_INSTRUCT_MESSAGES = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": "Describe the image; where are these people and what are they doing?",
            },
        ],
    },
]
INTERNVL_INSTRUCT_IMAGE = "s3://modular-bazel-artifacts-public/artifacts/model_testdata/internvl_instruct_image.jpg"

IDEFICS3_INSTRUCT_PROMPT = "<|begin_of_text|>User:<image>Describe the image:<end_of_utterance>\nAssistant:"
IDEFICS3_INSTRUCT_MESSAGES = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe the image:"},
        ],
    }
]
IDEFICS3_INSTRUCT_IMAGE = "s3://modular-bazel-artifacts-public/artifacts/model_testdata/idefics3_instruct_image.jpg"


DEFAULT_PROMPTS = [LONG_TEXT_PROMPT, *SHORT_TEXT_PROMPTS]
DEFAULT_TEXT_ONLY = [
    MockTextGenerationRequest.text_only(prompt=prompt)
    for prompt in DEFAULT_PROMPTS
]

DEFAULT_MULTIMODAL = [
    MockTextGenerationRequest.with_images(
        prompt=MULTIMODAL_PROMPT,
        images=[MULTIMODAL_IMAGE],
    )
]

PIXTRAL_REQUESTS = [
    MockTextGenerationRequest.with_images(
        prompt=PIXTRAL_PROMPT,
        images=[PIXTRAL_IMAGE],
        messages=PIXTRAL_MESSAGES,
    )
]

INTERNVL_INSTRUCT_REQUESTS = [
    MockTextGenerationRequest.with_images(
        prompt=INTERNVL_INSTRUCT_PROMPT,
        images=[INTERNVL_INSTRUCT_IMAGE],
        messages=INTERNVL_INSTRUCT_MESSAGES,
    )
]

IDEFICS3_INSTRUCT_REQUESTS = [
    MockTextGenerationRequest.with_images(
        IDEFICS3_INSTRUCT_PROMPT,
        [MULTIMODAL_IMAGE],
        messages=IDEFICS3_INSTRUCT_MESSAGES,
    ),
    MockTextGenerationRequest.with_images(
        IDEFICS3_INSTRUCT_PROMPT,
        [IDEFICS3_INSTRUCT_IMAGE],
        messages=IDEFICS3_INSTRUCT_MESSAGES,
    ),
]

# Default pixel generation prompts
DEFAULT_PIXEL_GENERATION_PROMPTS = [
    "photography, soft natural textures, highly realistic light, editorial, A black panther stalking through the dense undergrowth of an Indian jungle, early evening with shadows from tall trees, captured with low light photography, high ISO setting to highlight the panther's muscles in motion",
    "A red fox sitting in fresh snow in a winter forest, looking directly at the camera, soft morning light filtering through the trees",
    "Full body shot of a handsome tattooed short dark haired man wearing a jean and a white tee-shirt in 'Chiaroscuro Chronicles', lost in a captivating, slate gray monochromatic realm of masterful lighting and careful shading, emphasizing the emotional depth of the narrative, abrasive authenticity, ambient occlusion",
    "A lighthouse on a rocky cliff at sunset with an orange and purple sky, calm ocean in the background, photorealistic",
    'Four elements split into quadrants: top-left shows splashing water forming the word "WATER" on a water background, top-right shows soil forming "EARTH" with planet earth behind, bottom-left shows colorful clouds forming "AIR" at sunset, bottom-right shows fiery lava forming "FIRE" against the sun',
]

DEFAULT_PIXEL_GENERATION = [
    MockPixelGenerationRequest.from_prompt(prompt=prompt, seed=42)
    for prompt in DEFAULT_PIXEL_GENERATION_PROMPTS
]

# The prompt contains Kimi-specific media tokens for the vLLM path, which
# uses it directly.  The messages are the clean (no special tokens) version
# that the MAX tokenizer runs through the HuggingFace chat template.
KIMIK2_5_PROMPT = (
    "<|im_user|>user<|media_begin|>image<|media_content|>"
    "<|media_pad|><|media_end|>Describe this image.<|im_end|>"
    "<|im_assistant|>assistant<|im_middle|></think>"
)
KIMIK2_5_MESSAGES = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
KIMIK2_5_IMAGE = "s3://modular-bazel-artifacts-public/artifacts/model_testdata/kimik2_5_image.jpg"

KIMIK2_5_REQUESTS = [
    MockTextGenerationRequest.with_images(
        prompt=KIMIK2_5_PROMPT,
        images=[KIMIK2_5_IMAGE],
        messages=KIMIK2_5_MESSAGES,
    ),
]

# Default negative prompt from the official Wan repo (sample_neg_prompt).
# The model was trained with this as the "empty" negative; omitting it produces
# over-saturated, jittery, low-detail output.
WAN_DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

# Single-frame (1 frame, 20 steps) verification requests for Wan2.1 T2V.
# Prompts follow the WAN guide formula:
#   Subject + Scene + Motion + Camera + Atmosphere + Styling
# Motion is the centerpiece — static/still framing fights the model's T2V
# prior and produces high-variance outputs, inflating error bounds.
WAN_PIXEL_GENERATION_T2I = [
    # Slot 0 — fox trotting through autumn forest (natural motion, landscape)
    MockPixelGenerationRequest.from_prompt(
        prompt=(
            "A red fox trots steadily along a leaf-covered path through an"
            " autumn forest at dawn, its tail swaying gently with each"
            " stride. Camera tracks smoothly alongside at ground level,"
            " keeping pace with the fox. Soft amber light filters through"
            " the canopy, mist rising between the trees. Cinematic nature"
            " documentary style."
        ),
        negative_prompt=WAN_DEFAULT_NEGATIVE_PROMPT,
        height=720,
        width=1280,
        num_inference_steps=20,
        guidance_scale=5.0,
        seed=42,
    ),
    # Slot 1 — man pausing at flower stall (human motion, portrait)
    MockPixelGenerationRequest.from_prompt(
        prompt=(
            "An elderly man strolls slowly through a quiet Kyoto side street"
            " at dusk and stops at a small flower stall, reaching out to"
            " touch a white chrysanthemum. Camera dollies in gently from a"
            " medium shot to a close-up on his hand. Warm lantern light,"
            " subtle film grain, contemplative mood."
        ),
        negative_prompt=WAN_DEFAULT_NEGATIVE_PROMPT,
        height=1280,
        width=720,
        num_inference_steps=20,
        guidance_scale=5.0,
        seed=42,
    ),
]

FLUX2_PIXEL_GENERATION_I2I = [
    MockPixelGenerationRequest.from_prompt(
        prompt="Transform this image into a cinematic nighttime scene with neon reflections, wet streets, and dramatic contrast.",
        seed=42,
        input_image=MULTIMODAL_IMAGE,
        height=1024,
        width=1024,
    ),
    MockPixelGenerationRequest.from_prompt(
        prompt="Restyle this image as a watercolor painting with soft edges, visible brush texture, and warm afternoon light.",
        seed=42,
        input_image=MULTIMODAL_IMAGE,
        height=1024,
        width=1024,
    ),
]
