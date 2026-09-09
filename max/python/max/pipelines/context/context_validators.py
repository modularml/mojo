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

"""Context validation functions for different pipeline architectures."""

from __future__ import annotations

from .context import PixelContext, TextAndVisionContext, TextContext
from .exceptions import InputError


def validate_requires_vision_context(
    context: TextContext | TextAndVisionContext | PixelContext,
) -> None:
    """Validates that the context is a TextAndVisionContext.

    Args:
        context: The context to validate.

    Raises:
        InputError: If the context is not a TextAndVisionContext.
    """
    if not isinstance(context, TextAndVisionContext):
        raise InputError("This model requires TextAndVisionContext")


def validate_only_one_image(
    context: TextContext | TextAndVisionContext | PixelContext,
) -> None:
    """Validates that at most one image is provided in the context.

    Args:
        context: The context to validate.

    Raises:
        InputError: If more than one image is provided.
    """
    if isinstance(context, TextAndVisionContext) and len(context.images) > 1:
        raise InputError(
            f"This model only supports one image per request, got {len(context.images)}"
        )


def validate_initial_prompt_has_image(
    context: TextContext | TextAndVisionContext | PixelContext,
) -> None:
    """Validates that initial prompts contain an image for vision models.

    Args:
        context: The context to validate.

    Raises:
        InputError: If the initial prompt doesn't contain an image.
    """
    if isinstance(context, TextAndVisionContext):
        if context.is_initial_prompt and not context.images:
            raise InputError(
                "This model requires a prompt with an image. "
                "Consider using text-only models for non-image prompts."
            )


def validate_aspect_ratio_args(
    context: TextContext | TextAndVisionContext | PixelContext,
) -> None:
    """Validates that required aspect ratio arguments are present for vision input.

    Args:
        context: The context to validate.

    Raises:
        InputError: If required aspect ratio arguments are missing.
    """
    if isinstance(context, TextAndVisionContext) and context.images:
        if "aspect_ratio_ids" not in context.extra_model_args:
            raise InputError(
                "aspect_ratio_ids is required in extra_model_args for vision model input"
            )

        if "aspect_ratio_mask" not in context.extra_model_args:
            raise InputError(
                "aspect_ratio_mask is required in extra_model_args for vision model input"
            )


def validate_image_shape_5d(
    context: TextContext | TextAndVisionContext | PixelContext,
) -> None:
    """Validates that images have the expected 5-dimensional shape.

    Args:
        context: The context to validate.

    Raises:
        InputError: If the image shape is not 5-dimensional.
    """
    if isinstance(context, TextAndVisionContext) and context.images:
        for img in context.images:
            image_shape = img.pixel_values.shape
            expected_dims = 5
            if len(image_shape) != expected_dims:
                raise InputError(
                    f"Invalid image shape: expected {expected_dims} dimensions, "
                    f"got {len(image_shape)} dimensions"
                )


def validate_image_grid_thw_args(
    context: TextContext | TextAndVisionContext | PixelContext,
) -> None:
    """Validates that image_grid_thw is present when vision encoding is needed.

    Args:
        context: The context to validate.

    Raises:
        InputError: If image_grid_thw is missing from extra_model_args when
            vision encoding is needed.
    """
    if (
        isinstance(context, TextAndVisionContext)
        and context.needs_vision_encoding
    ):
        if "image_grid_thw" not in context.extra_model_args:
            raise InputError(
                "image_grid_thw is required in extra_model_args for vision model input when vision encoding is needed"
            )


def validate_vision_position_ids(
    context: TextContext | TextAndVisionContext | PixelContext,
) -> None:
    """Validates that vision_position_ids is present when vision encoding is needed.

    Args:
        context: The context to validate.

    Raises:
        InputError: If vision_position_ids is missing from extra_model_args when
            vision encoding is needed.
    """
    if (
        isinstance(context, TextAndVisionContext)
        and context.needs_vision_encoding
    ):
        if "vision_position_ids" not in context.extra_model_args:
            raise InputError(
                "vision_position_ids is required in extra_model_args for vision model input when vision encoding is needed"
            )
