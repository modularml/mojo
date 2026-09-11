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

"""Dataset-agnostic image augmentation for benchmark workloads.

Wraps the same random-image generation `random.py` uses with distributions
over whether a request/session gets images at all, how many, and what size,
so any dataset (not just `random`) can have images mixed into its requests
via `augment_samples_with_images`.
"""

from __future__ import annotations

import logging
import math
import random as _random
from collections.abc import Mapping
from typing import NamedTuple

import numpy as np
from PIL import Image
from typing_extensions import assert_never

from .chat_judge import ChatJudgeChatSamples
from .distribution import BaseDistribution, DistributionParameter
from .types import (
    ChatMessage,
    ChatSamples,
    ChatSession,
    ImageTurn,
    OpenAIImage,
    RequestSamples,
    Samples,
    encode_image,
)

logger = logging.getLogger(__name__)

# Reference point the token-length estimate is anchored on: a 512x512 image
# costs ~256 tokens for InternVL-style vision encoders. Real accounting is
# architecture-specific (see e.g. gemma4's patch-budget resize vs.
# qwen2_5vl's smart_resize); this is a coarse cross-architecture estimate for
# workload-shaping purposes only, not a claim about any particular model.
_REFERENCE_LONG_SIDE = 512
_REFERENCE_TOKENS = 256


def generate_random_image(height: int, width: int) -> Image.Image:
    """Generate a placeholder image with a limited color palette.

    Truly random images end up too large and incompressible; a block-based
    random image with a limited palette keeps encoded payload sizes
    realistic.
    """
    block_size = 16
    colors = np.array([0, 64, 128, 192, 255], dtype=np.uint8)

    blocks_h = (height + block_size - 1) // block_size
    blocks_w = (width + block_size - 1) // block_size

    block_colors = np.random.choice(len(colors), size=(blocks_h, blocks_w, 3))
    block_array = colors[block_colors]

    array = np.repeat(
        np.repeat(block_array, block_size, axis=0), block_size, axis=1
    )
    array = array[:height, :width]

    return Image.fromarray(array)


class ImageDimensions(NamedTuple):
    """Pixel dimensions of a generated image. Unpacks as (width, height)."""

    width: int
    height: int


def sample_image_dimensions(
    long_side_dist: BaseDistribution, aspect_ratio_dist: BaseDistribution
) -> ImageDimensions:
    """Sample pixel dimensions from long-side and aspect-ratio distributions.

    Args:
        long_side_dist: Distribution for the image's longer side, in pixels.
        aspect_ratio_dist: Distribution for width / height. Values >= 1
            produce a landscape image (long side is the width); values < 1
            produce a portrait image (long side is the height).

    Returns:
        An `ImageDimensions(width, height)`, each floored at 1 pixel.

    Raises:
        ValueError: If the sampled aspect ratio is not positive and finite.
    """
    long_side = max(round(long_side_dist.sample_value()), 1)
    aspect_ratio = aspect_ratio_dist.sample_value()
    if not math.isfinite(aspect_ratio) or aspect_ratio <= 0:
        raise ValueError(
            "--image-aspect-ratio must sample positive, finite values (width /"
            f" height); got {aspect_ratio}."
        )
    if aspect_ratio >= 1:
        width = long_side
        height = max(round(long_side / aspect_ratio), 1)
    else:
        height = long_side
        width = max(round(long_side * aspect_ratio), 1)
    return ImageDimensions(width=width, height=height)


def estimate_image_token_len(width: int, height: int) -> int:
    """Estimate the vision-token cost of one image, scaled by pixel area.

    Anchored on the same 512x512 -> 256-token reference the `random` dataset
    already used for a fixed image size. Scaling by area rather than by the
    long side alone keeps `--image-aspect-ratio` meaningful: a 1024x256 image
    costs a quarter of a 1024x1024 one, matching how patch-based vision
    encoders price an image.
    """
    scale = (width * height) / (_REFERENCE_LONG_SIDE**2)
    return max(round(_REFERENCE_TOKENS * scale), 1)


def _sample_one_image(
    long_side_dist: BaseDistribution, aspect_ratio_dist: BaseDistribution
) -> tuple[OpenAIImage, int]:
    width, height = sample_image_dimensions(long_side_dist, aspect_ratio_dist)
    image = encode_image(generate_random_image(height, width))
    return image, estimate_image_token_len(width, height)


def augment_samples_with_images(
    samples: Samples,
    *,
    image_fraction: float,
    image_count: DistributionParameter,
    image_long_side: DistributionParameter,
    image_aspect_ratio: DistributionParameter,
    image_turn: ImageTurn = "first",
    max_chat_len: int | None = None,
    run_prefix_len: int = 0,
) -> None:
    """Mix generated images into a fraction of already-sampled requests/sessions.

    Dataset-agnostic: operates on the `Samples` any dataset produces, after
    sampling, so datasets never need their own image-generation logic.

    Nothing is counted that will not be sent: a selected request or session
    whose images could not reach the wire -- no user message to attach them
    to, or a chat session the driver would abandon for exceeding
    `max_chat_len` -- is left untouched rather than augmented and reported.
    Chat-judge workloads are skipped outright, since their driver sends text
    only.

    Args:
        samples: Already-sampled requests or chat sessions, mutated in place.
        image_fraction: Fraction (0.0-1.0) of requests (single-turn) or
            sessions (multi-turn) selected for images. Selection is not a
            guarantee: see above for when a selected one is left alone.
        image_count: Distribution for the number of images on a
            request/turn that was selected to have images.
        image_long_side: Distribution for each image's longer side, in pixels.
        image_aspect_ratio: Distribution for each image's width / height.
        image_turn: Which user turn(s) in a multi-turn session get images:
            "first", "last", or "every".
        max_chat_len: Per-session token budget the multi-turn driver enforces.
            When given, sessions whose images would push them past it before
            their first measured turn are skipped rather than counted.
        run_prefix_len: Tokens the per-run unique prefix adds to the first
            measured turn under ``--force-unique-runs``.

    Raises:
        ValueError: If `image_fraction` is outside [0, 1], or if
            `image_aspect_ratio` samples a non-positive or non-finite value.
        TypeError: If `samples` is neither `RequestSamples` nor `ChatSamples`.
    """
    if not (0.0 <= image_fraction <= 1.0):
        raise ValueError(
            f"image_fraction must be in [0, 1], got {image_fraction}"
        )
    if image_fraction == 0:
        return

    count_dist = BaseDistribution.from_distribution_parameter(image_count)
    long_side_dist = BaseDistribution.from_distribution_parameter(
        image_long_side
    )
    aspect_ratio_dist = BaseDistribution.from_distribution_parameter(
        image_aspect_ratio
    )
    assert count_dist is not None
    assert long_side_dist is not None
    assert aspect_ratio_dist is not None

    if isinstance(samples, RequestSamples):
        _augment_request_samples(
            samples,
            image_fraction,
            count_dist,
            long_side_dist,
            aspect_ratio_dist,
        )
    elif isinstance(samples, ChatJudgeChatSamples):
        # chat_judge_session_driver sends text-only messages, so augmenting
        # here would inflate num_tokens and --dry-run image stats for images
        # that never reach the wire.
        logger.warning(
            "Image augmentation: skipping chat-judge workload; its driver does not send images."
        )
    elif isinstance(samples, ChatSamples):
        _augment_chat_samples(
            samples,
            image_fraction,
            count_dist,
            long_side_dist,
            aspect_ratio_dist,
            image_turn,
            max_chat_len,
            run_prefix_len,
        )
    else:
        raise TypeError(f"Unsupported samples type: {type(samples)}")


def _prompt_accepts_images(prompt: str | list[ChatMessage]) -> bool:
    """Whether the request driver will actually put images on the wire.

    It attaches them to the first message with role "user"; a prompt with no
    such message has them dropped at send time, so counting their tokens here
    would describe a workload that never runs.
    """
    if isinstance(prompt, str):
        return True
    return any(message.role == "user" for message in prompt)


def _augment_request_samples(
    samples: RequestSamples,
    image_fraction: float,
    count_dist: BaseDistribution,
    long_side_dist: BaseDistribution,
    aspect_ratio_dist: BaseDistribution,
) -> None:
    augmented = 0
    no_user_message = 0
    for request in samples.requests:
        if _random.random() >= image_fraction:
            continue
        if not _prompt_accepts_images(request.prompt_formatted):
            no_user_message += 1
            continue
        num_images = max(round(count_dist.sample_value()), 1)
        added_tokens = 0
        for _ in range(num_images):
            image, tokens = _sample_one_image(long_side_dist, aspect_ratio_dist)
            request.encoded_images.append(image)
            added_tokens += tokens
        request.prompt_len += added_tokens
        augmented += 1
    if no_user_message:
        logger.warning(
            "Image augmentation: skipped %d request(s) with no user message;"
            " the driver has nowhere to attach images on those.",
            no_user_message,
        )
    logger.info(
        "Image augmentation: added images to %d/%d requests",
        augmented,
        len(samples.requests),
    )


def _sends_no_measured_turns(
    session: ChatSession,
    max_chat_len: int,
    extra_tokens: Mapping[int, int] | None = None,
    run_prefix_len: int = 0,
) -> bool:
    """Mirror chat_session_driver's budget walk up to its first measured turn.

    ``extra_tokens`` maps a message index to tokens not yet added to it, so a
    caller can ask "would this session still send anything if I attached these
    images?" without mutating first.

    The driver builds ``prefix_turns`` warmup turns locally, accumulating
    tokens as it goes, and abandons the session outright if that prefix
    overruns the budget. Only afterwards does it send anything, so the first
    measured turn sits at ``prefix_turns * 2`` and carries the whole prefix
    with it.

    ``run_prefix_len`` mirrors the per-run unique prefix the driver charges to
    the first measured turn under ``--force-unique-runs``; like the driver, it
    only applies when there is no warmup prefix ahead of that turn.
    """
    messages = session.messages
    extra = extra_tokens or {}

    def tokens_at(i: int) -> int:
        return messages[i].num_tokens + extra.get(i, 0)

    chat_len = 0
    idx = 0
    prefix_end_idx = session.prefix_turns * 2
    while idx < prefix_end_idx and idx + 1 < len(messages):
        chat_len += tokens_at(idx)
        output_len = tokens_at(idx + 1)
        if chat_len + output_len > max_chat_len:
            return True
        chat_len += output_len
        idx += 2
    if idx < prefix_end_idx or idx + 1 >= len(messages):
        return True
    first_measured = tokens_at(idx) + tokens_at(idx + 1)
    if idx == 0:
        first_measured += run_prefix_len
    return chat_len + first_measured > max_chat_len


def _augment_chat_samples(
    samples: ChatSamples,
    image_fraction: float,
    count_dist: BaseDistribution,
    long_side_dist: BaseDistribution,
    aspect_ratio_dist: BaseDistribution,
    image_turn: ImageTurn,
    max_chat_len: int | None = None,
    run_prefix_len: int = 0,
) -> None:
    augmented = 0
    unmeasurable: list[int | None] = []
    for session in samples.chat_sessions:
        if _random.random() >= image_fraction:
            continue
        user_turn_indices = [
            i for i, m in enumerate(session.messages) if m.source == "user"
        ]
        if not user_turn_indices:
            continue
        if image_turn == "first":
            target_indices = [user_turn_indices[0]]
        elif image_turn == "last":
            target_indices = [user_turn_indices[-1]]
        elif image_turn == "every":
            target_indices = user_turn_indices
        else:
            assert_never(image_turn)

        # Generate first, attach second. Attaching is itself what can push a
        # session past the budget, so a session we end up rejecting must not
        # keep the images that caused the rejection -- otherwise num_tokens
        # and the --dry-run tables describe a request that never gets sent.
        pending: dict[int, tuple[list[OpenAIImage], int]] = {}
        for idx in target_indices:
            num_images = max(round(count_dist.sample_value()), 1)
            images: list[OpenAIImage] = []
            added_tokens = 0
            for _ in range(num_images):
                image, tokens = _sample_one_image(
                    long_side_dist, aspect_ratio_dist
                )
                images.append(image)
                added_tokens += tokens
            pending[idx] = (images, added_tokens)

        if max_chat_len is not None and _sends_no_measured_turns(
            session,
            max_chat_len,
            extra_tokens={idx: tokens for idx, (_, tokens) in pending.items()},
            run_prefix_len=run_prefix_len,
        ):
            unmeasurable.append(session.id)
            continue

        for idx, (images, added_tokens) in pending.items():
            session.messages[idx].images.extend(images)
            session.messages[idx].num_tokens += added_tokens
        augmented += 1
    if unmeasurable:
        logger.warning(
            "Image augmentation: skipped %d session(s) whose images would"
            " exceed the max chat length of %d before their first measured"
            " turn: %s. Lower --image-long-side or --image-count.",
            len(unmeasurable),
            max_chat_len,
            ", ".join(str(sid) for sid in unmeasurable),
        )
    logger.info(
        "Image augmentation: added images to %d/%d chat sessions",
        augmented,
        len(samples.chat_sessions),
    )
