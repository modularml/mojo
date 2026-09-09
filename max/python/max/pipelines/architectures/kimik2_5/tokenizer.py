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

"""Kimi K2.5 tokenizer for multimodal (text + vision) inputs.

Follows the same pattern as ``Qwen2_5VLTokenizer``: extends
``TextAndVisionTokenizer`` and delegates vision preprocessing to the
custom ``KimiK2_5VisionProcessor`` (PIL + NumPy only, no torch).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from max.pipelines.context import (
    GrammarEnforcementState,
    ImageMetadata,
    TokenBuffer,
)
from max.pipelines.context.exceptions import PromptTooLongError
from max.pipelines.lib import (
    TextAndVisionTokenizer,
    VisionPreprocessCache,
    max_tokens_to_generate,
)
from max.pipelines.lib.tokenizer import (
    encode_dkv_cache_hint,
    resolve_single_special_token,
    run_with_default_executor,
)
from max.pipelines.modeling.types import (
    TextGenerationRequest,
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
)
from max.support.image import find_contiguous_ranges, hash_image
from PIL import Image
from transformers import AutoTokenizer

from .context import KimiK2_5TextAndVisionContext
from .layers.vision.data_processing import compute_position_ids
from .vision_processor import (
    KimiK2_5VisionProcessor,
    _to_pil,
)

if TYPE_CHECKING:
    from max.pipelines.lib.config import PipelineConfig

logger = logging.getLogger("max.pipelines")

MEDIA_PAD = "<|media_pad|>"
IM_END = "<|im_end|>"
THINK_START = "<think>"
THINK_END = "</think>"
TOOL_CALLS_SECTION_BEGIN = "<|tool_calls_section_begin|>"
TOOL_CALLS_SECTION_END = "<|tool_calls_section_end|>"
TOOL_CALL_BEGIN = "<|tool_call_begin|>"
TOOL_CALL_END = "<|tool_call_end|>"
TOOL_CALL_ARGUMENT_BEGIN = "<|tool_call_argument_begin|>"

# One image's patchified pixels and its (t, h, w) grid, as cached.
_PreprocessedImage = tuple[npt.NDArray[Any], npt.NDArray[np.int64]]


def _sanitize_kimi_tool_schemas(
    tools: list[TextGenerationRequestTool] | None,
) -> list[TextGenerationRequestTool] | None:
    """Rewrite tool schemas to use only constructs Kimi's HF tokenizer supports.

    The Kimi-bundled ``tool_declaration_ts.py:_parse_parameter_type``
    only recognizes ``$ref``, ``anyOf``, ``enum``, ``type``, and ``{}``.
    A tool schema containing ``oneOf`` or a bare ``{"const": X}`` makes
    it raise ``ValueError``; ``tokenization_kimi.py`` then catches the
    exception, prints a warning, and renders the prompt **without any
    tool declarations**. Tool calling silently fails for that request
    even though the server returned 200.

    This function walks each tool's schema (any depth, all standard
    JSON Schema combinator/container keys) and applies two
    equivalence-preserving rewrites:

      * ``oneOf`` → ``anyOf`` (concatenated if both keys are present).
      * ``{"const": X}`` → ``{"enum": [X]}``.

    For tool-call argument schemas these transforms preserve semantics:
    JSON Schema defines ``const: X`` as exactly equivalent to
    ``enum: [X]``, and a value that matches a ``oneOf`` branch also
    matches the corresponding ``anyOf`` (the difference between the
    combinators — exclusive vs. inclusive matching — is not enforced
    by either Kimi's argument grammar or downstream tool runtimes).
    """
    if not tools:
        return tools
    return [
        TextGenerationRequestTool(
            type=tool["type"],
            function={
                **tool["function"],
                "parameters": _sanitize_kimi_schema_node(
                    tool["function"].get("parameters", {})
                ),
            },
        )
        for tool in tools
    ]


def _sanitize_kimi_schema_node(node: Any) -> Any:
    """Recursive helper for :func:`_sanitize_kimi_tool_schemas`."""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        any_of_branches: list[Any] = []
        for key, value in node.items():
            if key in ("oneOf", "anyOf"):
                sanitized = _sanitize_kimi_schema_node(value)
                if isinstance(sanitized, list):
                    any_of_branches.extend(sanitized)
                continue
            if key == "const":
                # Defer to the ``enum`` conversion below so an explicit
                # ``enum`` wins if both are present.
                continue
            out[key] = _sanitize_kimi_schema_node(value)
        if any_of_branches:
            out["anyOf"] = any_of_branches
        if "const" in node and "enum" not in node:
            out["enum"] = [node["const"]]
        return out
    if isinstance(node, list):
        return [_sanitize_kimi_schema_node(item) for item in node]
    return node


class KimiK2_5VLTokenizer(TextAndVisionTokenizer):
    """Kimi K2.5 tokenizer for multimodal (text + vision) inputs.

    Extends ``TextAndVisionTokenizer`` with a custom vision processor
    (``KimiK2_5VisionProcessor``) instead of ``AutoProcessor``, following
    the same architecture as ``Qwen2_5VLTokenizer``.
    """

    def __init__(
        self,
        model_path: str,
        pipeline_config: PipelineConfig,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        trust_remote_code: bool = False,
        **unused_kwargs,
    ) -> None:
        self.model_path = model_path

        self.delegate = AutoTokenizer.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            model_max_length=max_length,
        )
        self.max_length = max_length or self.delegate.model_max_length

        config = pipeline_config.model.huggingface_config

        eos_token_id = self.delegate.eos_token_id
        self._eos_token_ids = (
            {eos_token_id} if eos_token_id is not None else set()
        )
        if eos_token_id := getattr(config, "eos_token_id", None):
            if isinstance(eos_token_id, int):
                self._eos_token_ids.add(eos_token_id)
            elif isinstance(eos_token_id, list):
                self._eos_token_ids.update(eos_token_id)

        im_end_id = self.delegate.convert_tokens_to_ids(IM_END)
        if isinstance(im_end_id, int):
            self._eos_token_ids.add(im_end_id)

        self.enable_prefix_caching = (
            pipeline_config.model.kv_cache.enable_prefix_caching
        )
        self.enable_vision_caching = (
            pipeline_config.runtime.vision_cache_utilization != 0
        )

        # Resolve the media pad token ID used as the vision placeholder.
        media_pad_id = self.delegate.convert_tokens_to_ids(MEDIA_PAD)
        if isinstance(media_pad_id, list):
            media_pad_id = media_pad_id[0]
        if media_pad_id == self.delegate.unk_token_id:
            raise ValueError(
                f"Token {MEDIA_PAD!r} not found in tokenizer vocabulary"
            )
        self.media_pad_token_id: int = media_pad_id
        self.vision_token_ids = [self.media_pad_token_id]

        self._reasoning_start_token_id: int = resolve_single_special_token(
            self.delegate, THINK_START
        )
        self._reasoning_end_token_id: int = resolve_single_special_token(
            self.delegate, THINK_END
        )

        # Build the custom vision processor from HF config.
        media_proc_cfg = getattr(config, "media_proc_cfg", None)
        self.vision_processor = KimiK2_5VisionProcessor(
            media_proc_cfg=media_proc_cfg
        )

        self._preprocess_cache: VisionPreprocessCache[_PreprocessedImage] = (
            VisionPreprocessCache.for_images(pipeline_config.runtime)
        )

        # rope_max_width is needed to compute per-image position IDs.
        vision_cfg = getattr(config, "vision_config", None)
        self.rope_max_width: int = int(
            getattr(vision_cfg, "rope_max_width", 512)
        )

    @property
    def reasoning_start_token_id(self) -> int:
        """Token id of ``<think>`` (opens a Kimi K2.5 reasoning span)."""
        return self._reasoning_start_token_id

    @property
    def reasoning_end_token_id(self) -> int:
        """Token id of ``</think>`` (closes a Kimi K2.5 reasoning span)."""
        return self._reasoning_end_token_id

    async def encode(
        self, prompt: str | Sequence[int], add_special_tokens: bool = True
    ) -> npt.NDArray[np.integer[Any]]:
        """Transforms the provided prompt into a token array.

        Kimi's ``TikTokenTokenizer.encode()`` accepts
        ``allow_special_tokens`` instead of the HuggingFace-standard
        ``add_special_tokens``.  Passing the unrecognised kwarg causes
        HF to silently fall back to the slow
        ``PreTrainedTokenizer.encode()`` path (actually slow in
        transformers v5+).  This override calls the delegate with
        ``allow_special_tokens=True`` so the fast tiktoken path is
        always used.

        The ``add_special_tokens`` parameter is accepted for interface
        compatibility but is effectively a no-op: the tiktoken fast
        path never prepends/appends BOS/EOS tokens regardless, and
        ``allow_special_tokens`` (whether tiktoken *recognises* special
        token strings in the input) must stay ``True`` so that
        chat-templated text is tokenised correctly.
        """
        encoded_prompt: npt.NDArray[np.integer[Any]]
        if isinstance(prompt, str):

            def _encode_fn(
                prompt: str,
            ) -> npt.NDArray[np.integer[Any]]:
                return np.array(
                    self.delegate.encode(prompt, allow_special_tokens=True)
                )

            encoded_prompt = await run_with_default_executor(_encode_fn, prompt)

            max_length = self.max_length or self.delegate.model_max_length
            if max_length and len(encoded_prompt) > max_length:
                raise PromptTooLongError(len(encoded_prompt), max_length)
        else:
            encoded_prompt = np.array(list(prompt))

        return encoded_prompt

    def apply_chat_template(
        self,
        messages: list[TextGenerationRequestMessage],
        tools: list[TextGenerationRequestTool] | None = None,
        **chat_template_options: Any,
    ) -> str:
        """Applies the tokenizer's chat template to messages.

        Tools are passed through :func:`_sanitize_kimi_tool_schemas` to
        rewrite JSON Schema constructs that Kimi's HF tokenizer code
        (``tool_declaration_ts.py``) does not recognize — without the
        rewrite, tools containing ``oneOf`` or a bare ``{"const": X}``
        cause the HF code to raise, swallow the exception, and render
        the prompt with no tool declarations. The model then has no
        idea those tools exist and silently fails to call them.

        Args:
            messages: List of messages for the chat template.
            tools: Optional tools available for the model to invoke.
            **chat_template_options: Template options to forward to the Jinja
                template. Merged with ``add_generation_prompt=True`` default.
                Kimi K2.5 uses ``thinking=False`` to disable thinking mode.

        Returns:
            The templated chat message as a string.
        """
        chat_template_options = {
            "add_generation_prompt": True,
            **chat_template_options,
        }
        templated = self.delegate.apply_chat_template(
            [msg.model_dump(exclude_none=True) for msg in messages],
            tokenize=False,
            tools=_sanitize_kimi_tool_schemas(tools),
            **chat_template_options,
        )
        assert isinstance(templated, str)
        return templated

    def _preprocess_image(
        self, image_hash: int | None, image: bytes | Image.Image
    ) -> _PreprocessedImage:
        """Preprocesses one image, reusing a cached result when available.

        The cache key is the digest of the raw encoded bytes plus the
        resolution size class -- the same key ``new_context`` hands to the
        vision encoder cache, computed once by the caller and shared by both.
        Hitting here skips the resize, normalize and patchify that the encoder
        cache cannot skip, since it is consulted only after that work has run.

        ``KimiK2_5VisionProcessor.preprocess`` resizes, normalizes and
        patchifies each item with no cross-item state and only concatenates at
        the end, so preprocessing one image at a time and concatenating
        afterwards is bit-identical to the batched call it replaces.

        Args:
            image_hash: The image's content digest, or ``None`` when no media
                caching needs one.
            image: The image as raw bytes, or already decoded by the API server.

        Returns:
            The image's patchified pixels and its ``(t, h, w)`` grid.
        """

        def preprocess() -> _PreprocessedImage:
            # _to_pil accepts both a PIL.Image and raw bytes.
            outputs = self.vision_processor.preprocess(
                [{"type": "image", "image": _to_pil(image)}]
            )
            return outputs["pixel_values"], outputs["grid_thws"][0]

        return self._preprocess_cache.get_or_preprocess(image_hash, preprocess)

    def _process_images(
        self,
        request: TextGenerationRequest,
        image_hashes: Sequence[int | None],
    ) -> dict[str, npt.NDArray[Any]]:
        """Converts raw image bytes from the request into preprocessed arrays.

        Args:
            request: The text generation request containing raw image bytes
                in ``request.images``.
            image_hashes: One content digest per image, positionally matching
                ``request.images``, or ``None`` where no digest was needed.

        Returns:
            Dictionary with ``pixel_values`` and ``grid_thws`` arrays,
            or empty dict when no images are provided.
        """
        if not request.images:
            return {}

        per_image = [
            self._preprocess_image(image_hash, image)
            for image_hash, image in zip(
                image_hashes, request.images_for_processing(), strict=True
            )
        ]
        # Reassembles exactly what the batched call returned, so every caller
        # downstream sees the same two arrays it always did -- including their
        # writability, since concatenating copies out of the frozen cached
        # payloads.
        return {
            "pixel_values": np.concatenate(
                [pixels for pixels, _ in per_image], axis=0
            ),
            "grid_thws": np.stack(
                [grid for _, grid in per_image], axis=0
            ).astype(np.int64),
        }

    async def new_context(
        self, request: TextGenerationRequest
    ) -> KimiK2_5TextAndVisionContext:
        """Creates a ``KimiK2_5TextAndVisionContext`` for the Kimi K2.5 model.

        Processes text through the delegate tokenizer and images through
        ``KimiK2_5VisionProcessor``, then assembles the context with
        ``ImageMetadata`` entries for each image.
        """
        prompt: str | Sequence[int]
        add_special_tokens = True
        if request.prompt is not None:
            prompt = request.prompt
        elif request.messages:
            prompt = self.apply_chat_template(
                request.messages,
                request.tools,
                **(request.chat_template_options or {}),
            )
            add_special_tokens = False
        else:
            raise ValueError(f"{request} does not provide messages or prompt.")

        encoded_prompt = np.array(await self.encode(prompt, add_special_tokens))

        placeholder_positions = np.where(
            encoded_prompt == self.media_pad_token_id
        )[0]
        num_placeholders = len(placeholder_positions)

        merge_len = self.vision_processor.cfg.merge_kernel_size**2
        image_hashes: list[int | None] = []
        if request.images:
            if num_placeholders != len(request.images):
                raise ValueError(
                    f"Number of <|media_pad|> placeholders ({num_placeholders}) "
                    f"must match number of images ({len(request.images)})"
                )

            # Key each image on its raw encoded bytes (+ the resolution size
            # class), not on hash_image(pixels): the raw-byte key is
            # byte-identical across torch/BLAS/device so a separate encoder can
            # reproduce it for cache-aware routing, whereas the post-resize
            # float hash cannot. The vision processor resizes deterministically
            # from the encoded bytes + fixed config, so the process-wide
            # total-patch budget is the size tier. request.images is 1:1 with
            # pixel_values and start_and_end_idxs (one processed chunk + one
            # contiguous media-pad run per image).
            #
            # One digest per image, shared by the preprocessed-tensor cache in
            # _process_images and the vision encoder cache downstream.
            # Computing it in both places would hash every image's bytes twice,
            # which on a cache hit is most of the work that remains.
            image_size_tier = self.vision_processor.cfg.in_patch_limit
            image_hashes = (
                [
                    hash_image(raw_bytes, image_size_tier)
                    for raw_bytes in request.images
                ]
                if self.enable_prefix_caching
                or self.enable_vision_caching
                or self._preprocess_cache.enabled
                else [None] * len(request.images)
            )

            # Process images through the custom vision processor.
            vision_outputs = self._process_images(request, image_hashes)

            grid_thws = np.empty((0, 3), dtype=np.int64)
            pixel_values: list[npt.NDArray[Any]] = []
            position_ids = np.empty(0, dtype=np.int64)

            if not vision_outputs:
                raise ValueError(
                    "Images provided but vision processor returned empty"
                )

            grid_thws = vision_outputs["grid_thws"].copy()
            all_pixels = vision_outputs["pixel_values"]
            # Split the concatenated pixel array into per-image chunks
            # using the (t, h, w) grid dimensions to compute patch counts.
            offsets = np.prod(grid_thws, axis=1).cumsum()
            pixel_values = np.split(all_pixels, offsets[:-1].tolist())

            grid_thws_list = [(int(t), int(h), int(w)) for t, h, w in grid_thws]
            position_ids = compute_position_ids(
                grid_thws_list, self.rope_max_width
            )
            max_h = int(grid_thws[:, 1].max())
            max_w = int(grid_thws[:, 2].max())

            # Expand each media placeholder to match the number of merged
            # vision tokens for its corresponding image.
            for i in range(len(grid_thws)):
                idx = int(placeholder_positions[-(i + 1)])
                t, h, w = grid_thws[-(i + 1)]
                num_img_tokens = int((t * h * w) // merge_len)
                encoded_prompt = np.insert(
                    encoded_prompt,
                    idx,
                    [self.media_pad_token_id] * (num_img_tokens - 1),
                )
        else:
            # No images: initialize empty arrays for text-only requests
            max_h = 0
            max_w = 0
            grid_thws = np.empty((0, 3), dtype=np.int64)
            pixel_values = []
            position_ids = np.empty(0, dtype=np.int64)

        max_new_tokens = None
        if request.sampling_params.max_new_tokens is not None:
            max_new_tokens = request.sampling_params.max_new_tokens

        max_gen_tokens = max_tokens_to_generate(
            encoded_prompt.shape[0], self.max_length, max_new_tokens
        )

        # Positions of the image-placeholder token within this context's
        # token buffer (relative to the start of encoded_prompt).
        image_token_indices = np.where(
            encoded_prompt == self.media_pad_token_id
        )[0].astype(np.int32)

        json_schema = (
            json.dumps(request.response_format.json_schema)
            if request.response_format
            and request.response_format.json_schema is not None
            else None
        )

        grammar = (
            request.response_format.grammar if request.response_format else None
        )

        # Carry grammar enforcement state (grammar_enforced, tools_forced,
        # has_json_schema) from the response format.
        # Mirrors TextTokenizer / TextAndVisionTokenizer.
        grammar_state = GrammarEnforcementState.from_response_format(
            request.response_format
        )

        if self.max_length and encoded_prompt.shape[0] > self.max_length:
            raise PromptTooLongError(encoded_prompt.shape[0], self.max_length)

        start_and_end_idxs = find_contiguous_ranges(
            encoded_prompt, self.vision_token_ids
        )

        token_buffer = TokenBuffer(
            array=encoded_prompt.astype(np.int64, copy=False),
        )

        context = KimiK2_5TextAndVisionContext(
            request_id=request.request_id,
            eos_tracker=await self.create_eos_tracker(request),
            tokens=token_buffer,
            vocab_size=self.tokenizer_vocab_size,
            max_length=encoded_prompt.shape[0] + max_gen_tokens
            if max_gen_tokens is not None
            else self.max_length,
            json_schema=json_schema,
            grammar=grammar,
            grammar_state=grammar_state,
            log_probabilities=request.logprobs,
            log_probabilities_echo=request.echo,
            sampling_params=request.sampling_params,
            target_endpoint=request.target_endpoint,
            dkv_cache_hint=encode_dkv_cache_hint(request.dkv_cache_hint),
            grid_thws=grid_thws,
            position_ids=position_ids,
            image_token_indices=image_token_indices,
            max_h=max_h,
            max_w=max_w,
            images=[
                ImageMetadata(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    pixel_values=pixels,
                    image_hash=image_hash
                    if self.enable_prefix_caching or self.enable_vision_caching
                    else None,
                )
                for (start_idx, end_idx), pixels, image_hash in zip(
                    start_and_end_idxs,
                    pixel_values,
                    image_hashes,
                    strict=True,
                )
            ],
            vision_token_ids=self.vision_token_ids,
            cache_salt=request.cache_salt,
        )

        return context
