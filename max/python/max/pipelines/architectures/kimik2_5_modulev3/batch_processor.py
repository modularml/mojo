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
"""Input batching for Kimi-K2.5 pipeline models."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from max.driver import Buffer
from max.engine import InferenceSession
from max.experimental.nn import CompiledModel
from max.graph.buffer_utils import cast_tensor_to
from max.nn.kv_cache import KVCacheInputsInterface
from max.pipelines.context import ImageMetadata
from max.pipelines.lib.vision_encoder_cache import concat_device_buffers

from ..deepseekV3_modulev3.batch_processor import (
    DeepseekV3ModuleV3BatchProcessor,
)
from .context import KimiK2_5TextAndVisionContext
from .memory_planner import _vision_encoder_token_budget, _vision_merge_sq
from .model_config import KimiK2_5Config

if TYPE_CHECKING:
    from .model import KimiK2_5ModelInputs

logger = logging.getLogger("max.pipelines")


class KimiK2_5BatchProcessor(DeepseekV3ModuleV3BatchProcessor):
    """Ragged batching for Kimi-K2.5 models.

    Vision is cached by the pipeline-owned ``VisionEncoderCache`` between
    prep and execute; this processor only builds the text inputs and runs
    the chunked vision encode (see :meth:`encode_uncached_chunked`).
    """

    _model_config: KimiK2_5Config | None = None
    _vision_model: CompiledModel[Any, Any] | None = None
    _session: InferenceSession | None = None

    def bind_model_config(self, model_config: KimiK2_5Config) -> None:
        """Wire the finalized Kimi-K2.5 config from ``load_model``."""
        self._model_config = model_config

    def bind_vision_encoder(
        self,
        *,
        vision_model: CompiledModel[Any, Any],
        session: InferenceSession,
    ) -> None:
        """Wire the compiled vision encoder and inference session."""
        self._vision_model = vision_model
        self._session = session

    def prepare_initial_token_inputs(  # type: ignore[override]
        self,
        replica_batches: Sequence[Sequence[KimiK2_5TextAndVisionContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> KimiK2_5ModelInputs:
        """Prepare inputs for the first execution pass of KimiK2.5.

        Images are selected, encoded (see :meth:`encode_uncached_chunked`),
        and cached by the pipeline-owned ``VisionEncoderCache`` between prep
        and execute; this processor only builds the text inputs.
        """
        assert self._model_config is not None, (
            "model_config must be bound; call bind_model_config() after"
            " load_model()"
        )
        assert self._vision_model is not None
        assert self._session is not None

        base = super().prepare_initial_token_inputs(
            replica_batches,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=return_n_logits,
        )

        from .model import KimiK2_5ModelInputs

        return KimiK2_5ModelInputs(
            tokens=base.tokens,
            input_row_offsets=base.input_row_offsets,
            return_n_logits=base.return_n_logits,
            kv_cache_inputs=base.kv_cache_inputs,
            batch_context_lengths=base.batch_context_lengths,
            data_parallel_splits=base.data_parallel_splits,
            input_row_offsets_i64=base.input_row_offsets_i64,
            ep_inputs=base.ep_inputs,
        )

    def _collect_uncached_image_inputs(
        self,
        selection: Sequence[
            tuple[KimiK2_5TextAndVisionContext, Sequence[ImageMetadata]]
        ],
    ) -> list[dict[str, npt.NDArray[Any]]]:
        """Collect per-image numpy inputs for the cache-selected images.

        Miss images are matched to ``ctx.images`` by ``start_idx`` (image
        spans within a context are non-overlapping, so it uniquely names
        an image) rather than object identity: ``miss_images`` may hold
        reconstructed ``ImageMetadata`` instances for the same logical
        image, and an identity match would silently collect nothing.
        """
        per_image: list[dict[str, npt.NDArray[Any]]] = []
        for ctx, miss_images in selection:
            uncached_start_indices = {img.start_idx for img in miss_images}
            pos_offset = 0
            for i, img in enumerate(ctx.images):
                thw = ctx.grid_thws[i]
                n_pos = int(thw[0] * thw[1] * thw[2])

                if img.start_idx in uncached_start_indices:
                    per_image.append(
                        {
                            "pixel_values": img.pixel_values,
                            "grid_thw": np.asarray(thw, dtype=np.int64),
                            "position_ids": np.asarray(
                                ctx.position_ids[
                                    pos_offset : pos_offset + n_pos
                                ],
                                dtype=np.int64,
                            ),
                        }
                    )

                pos_offset += n_pos
        return per_image

    @staticmethod
    def _batch_vision_input_shapes(
        per_image: Sequence[dict[str, npt.NDArray[Any]]],
    ) -> dict[str, list[int]]:
        """Return the shapes ``_build_vision_input_buffers`` would produce for the full batch."""
        if not per_image:
            return {
                "pixel_values": [],
                "grid_thws": [],
                "cu_seqlens": [],
                "max_seqlen": [],
                "vision_position_ids": [],
            }
        pv0 = per_image[0]["pixel_values"]
        n_patches = sum(int(e["pixel_values"].shape[0]) for e in per_image)
        pixel_values_shape = [n_patches, *(int(d) for d in pv0.shape[1:])]
        n_images = len(per_image)
        n_position_ids = sum(int(e["position_ids"].shape[0]) for e in per_image)
        return {
            "pixel_values": pixel_values_shape,
            "grid_thws": [n_images, 3],
            "cu_seqlens": [n_images + 1],
            "max_seqlen": [1],
            "vision_position_ids": [n_position_ids],
        }

    def _collect_vision_encoder_request_metadata(
        self,
        selection: Sequence[
            tuple[KimiK2_5TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        vision_input_shapes: dict[str, list[int]],
    ) -> dict[str, Any]:
        """Collect debug metadata for a vision-encoder invocation."""
        assert self._model_config is not None
        patch_size = int(self._model_config.vision_config.patch_size)
        merge_cfg = self._model_config.vision_config.merge_kernel_size
        if isinstance(merge_cfg, int):
            merge_h, merge_w = merge_cfg, merge_cfg
        else:
            merge_h = int(merge_cfg[0])
            merge_w = int(merge_cfg[1])
        merge_sq = _vision_merge_sq(self._model_config.vision_config)

        uncached_contexts = [ctx for ctx, _ in selection]
        request_ids = [str(ctx.request_id) for ctx in uncached_contexts]
        total_images = sum(len(ctx.images) for ctx in uncached_contexts)
        next_images_total = sum(
            len(ctx.next_images) for ctx in uncached_contexts
        )
        context_sequence_lengths = [
            {
                "request_id": str(ctx.request_id),
                "active_sequence_length": int(ctx.tokens.active_length),
                "processed_sequence_length": int(ctx.tokens.processed_length),
                "total_sequence_length": len(ctx.tokens),
                "context_needs_vision_encoding": bool(
                    ctx.needs_vision_encoding
                ),
                "image_count": len(ctx.images),
                "next_image_count": len(ctx.next_images),
            }
            for ctx in uncached_contexts
        ]

        per_image_metadata: list[dict[str, Any]] = []
        image_counter = 0
        for ctx, miss_images in selection:
            uncached_ids = {id(img) for img in miss_images}
            for i, (img, thw) in enumerate(
                zip(ctx.images, ctx.grid_thws, strict=True)
            ):
                if id(img) not in uncached_ids:
                    continue

                t = int(thw[0])
                h = int(thw[1])
                w = int(thw[2])
                pixel_values_shape = [int(x) for x in img.pixel_values.shape]
                per_image_metadata.append(
                    {
                        "request_id": str(ctx.request_id),
                        "image_index_in_request": i,
                        "image_index_in_uncached_batch": image_counter,
                        "image_hash_present": img.image_hash is not None,
                        "token_span": {
                            "start_idx": int(img.start_idx),
                            "end_idx": int(img.end_idx),
                            "length": int(img.end_idx - img.start_idx),
                        },
                        "size_before_patching": {
                            "temporal_frames": t,
                            "height_px": h * patch_size,
                            "width_px": w * patch_size,
                        },
                        "size_after_patching": {
                            "patch_grid_thw": [t, h, w],
                            "patch_tensor_shape": pixel_values_shape,
                        },
                        "size_after_patch_merger": {
                            "merge_kernel_hw": [merge_h, merge_w],
                            "merged_token_count": int((t * h * w) // merge_sq),
                        },
                    }
                )
                image_counter += 1

        images_needing_encoding = len(per_image_metadata)
        grid_thws_shape = vision_input_shapes.get("grid_thws") or []
        expected_from_inputs = int(grid_thws_shape[0]) if grid_thws_shape else 0

        return {
            "request_ids": request_ids,
            "batch_size": len(uncached_contexts),
            "uncached_context_count": len(uncached_contexts),
            "vision_input_dtype": str(self._model_config.vision_config.dtype),
            "vision_input_shapes": vision_input_shapes,
            "images": {
                "total_images_in_batch": total_images,
                "next_images_total": next_images_total,
                "images_needing_encoding": images_needing_encoding,
                "images_not_needing_encoding": (
                    total_images - images_needing_encoding
                ),
                "images_already_encoded_in_prior_steps": (
                    total_images - next_images_total
                ),
                "images_skipped_due_to_cache": max(
                    0, next_images_total - images_needing_encoding
                ),
                "images_needing_encoding_from_vision_input_shapes": (
                    expected_from_inputs
                ),
            },
            "context_sequence_lengths": context_sequence_lengths,
            "per_image_metadata": per_image_metadata,
        }

    @staticmethod
    def _patches_in_image(entry: dict[str, npt.NDArray[Any]]) -> int:
        """Number of patches contributed by a single image entry."""
        thw = entry["grid_thw"]
        return int(thw[0] * thw[1] * thw[2])

    @staticmethod
    def _chunk_image_inputs_by_token_budget(
        per_image: Sequence[dict[str, npt.NDArray[Any]]],
        token_budget: int,
        merge_sq: int,
    ) -> list[list[dict[str, npt.NDArray[Any]]]]:
        """Group per-image entries into post-merge-token-bounded chunks."""
        chunks: list[list[dict[str, npt.NDArray[Any]]]] = []
        current: list[dict[str, npt.NDArray[Any]]] = []
        current_tokens = 0
        for entry in per_image:
            tokens = KimiK2_5BatchProcessor._patches_in_image(entry) // merge_sq
            if current and current_tokens + tokens > token_budget:
                chunks.append(current)
                current = []
                current_tokens = 0
            current.append(entry)
            current_tokens += tokens
        if current:
            chunks.append(current)
        return chunks

    def _build_vision_input_buffers(
        self,
        per_image: Sequence[dict[str, npt.NDArray[Any]]],
    ) -> dict[str, Buffer]:
        """Build device-0 vision encoder Buffers from a chunk of images.

        The vision encoder runs single-device (device 0); its output is
        replicated to every device by the caller.
        """
        assert self._model_config is not None
        assert self._session is not None
        assert per_image, "per_image must not be empty"

        devices = self.runtime.devices

        all_pixel_values = np.concatenate(
            [e["pixel_values"] for e in per_image], axis=0
        )
        all_grid_thws_np = np.vstack([e["grid_thw"] for e in per_image]).astype(
            np.int64
        )

        seq_lens = [int(np.prod(g)) for g in all_grid_thws_np]
        cu_seqlens_np = np.zeros(len(seq_lens) + 1, dtype=np.uint32)
        np.cumsum(seq_lens, out=cu_seqlens_np[1:])

        max_seqlen_np = np.array([max(seq_lens)], dtype=np.uint32)

        position_ids_np = np.concatenate(
            [e["position_ids"] for e in per_image]
        ).astype(np.int64)

        device0 = devices[0]
        vision_dtype = self._model_config.vision_config.dtype
        pixel_values_f32 = Buffer.from_numpy(all_pixel_values).to(device0)
        pixel_values_buf = (
            pixel_values_f32
            if pixel_values_f32.dtype == vision_dtype
            else cast_tensor_to(
                pixel_values_f32, vision_dtype, session=self._session
            )
        )
        grid_thws_buf = Buffer.from_numpy(all_grid_thws_np).to(device0)
        cu_seqlens_buf = Buffer.from_numpy(cu_seqlens_np).to(device0)
        vision_position_ids_buf = Buffer.from_numpy(position_ids_np).to(device0)
        max_seqlen_buf = Buffer.from_numpy(max_seqlen_np)
        return {
            "pixel_values": pixel_values_buf,
            "grid_thws": grid_thws_buf,
            "cu_seqlens": cu_seqlens_buf,
            "max_seqlen": max_seqlen_buf,
            "vision_position_ids": vision_position_ids_buf,
        }

    def encode_uncached_chunked(
        self,
        selection: Sequence[
            tuple[KimiK2_5TextAndVisionContext, Sequence[ImageMetadata]]
        ],
    ) -> tuple[list[Buffer], list[int]]:
        """Run the chunked vision encoder over the cache-selected images.

        Runs the encoder in token-bounded chunks on device 0, replicates each
        chunk's embedding to every device, then re-concatenates the per-device
        outputs to per-image order (with matching per-image token counts) so
        the cache only ever sees per-image-ordered output.
        """
        assert self._model_config is not None
        assert self._vision_model is not None

        per_image = self._collect_uncached_image_inputs(selection)
        assert per_image, (
            "selection non-empty but no per-image entries collected"
        )

        pipeline_config = self.runtime.pipeline_config
        token_budget = _vision_encoder_token_budget(pipeline_config)
        merge_sq = _vision_merge_sq(self._model_config.vision_config)

        if token_budget is None:
            chunks = [list(per_image)]
        else:
            chunks = self._chunk_image_inputs_by_token_budget(
                per_image, token_budget, merge_sq
            )

        batch_input_shapes = self._batch_vision_input_shapes(per_image)
        total_patches = sum(self._patches_in_image(e) for e in per_image)

        if len(chunks) > 1:
            assert token_budget is not None
            logger.info(
                "Vision encoder splitting into %d chunks "
                "(image_token_budget=%d, total_image_tokens=%d, "
                "total_patches=%d, image_count=%d)",
                len(chunks),
                token_budget,
                total_patches // merge_sq,
                total_patches,
                len(per_image),
            )

        chunk_outputs: list[list[Buffer]] = []
        for chunk_idx, chunk in enumerate(chunks):
            chunk_inputs = self._build_vision_input_buffers(chunk)

            chunk_patches = sum(self._patches_in_image(e) for e in chunk)
            chunk_meta = {
                "chunk_index": chunk_idx,
                "num_chunks": len(chunks),
                "chunk_image_count": len(chunk),
                "chunk_image_tokens": chunk_patches // merge_sq,
                "chunk_total_patches": chunk_patches,
                "image_token_budget": token_budget,
            }

            try:
                # The vision encoder graph is single-device (device 0); run it
                # once, then replicate its embedding to every device so the
                # language graph's per-device (replicated) image inputs are
                # satisfied without redundantly recomputing on each GPU.
                (device0_embed,) = self._vision_model.execute_raw(
                    chunk_inputs["pixel_values"],
                    chunk_inputs["grid_thws"],
                    chunk_inputs["cu_seqlens"],
                    chunk_inputs["max_seqlen"],
                    chunk_inputs["vision_position_ids"],
                )
                chunk_embeds = [
                    device0_embed.to(d) for d in self.runtime.devices
                ]
            except Exception as err:
                vision_metadata = self._collect_vision_encoder_request_metadata(
                    selection=selection,
                    vision_input_shapes=batch_input_shapes,
                )
                failure_payload = {
                    "stage": "vision_execute",
                    "chunk": chunk_meta,
                    "error": {
                        "type": type(err).__name__,
                        "message": str(err),
                        "repr": repr(err),
                    },
                    "metadata": vision_metadata,
                }
                logger.exception(
                    "Vision encoder failed. Request metadata: %s",
                    json.dumps(failure_payload, sort_keys=True),
                )
                raise
            assert len(chunk_embeds) == len(self.runtime.devices)
            chunk_outputs.append(list(chunk_embeds))

        if len(chunk_outputs) == 1:
            vision_embeds = chunk_outputs[0]
        else:
            vision_embeds = [
                concat_device_buffers([chunk[d] for chunk in chunk_outputs])
                for d in range(len(self.runtime.devices))
            ]

        token_counts = [
            self._patches_in_image(entry) // merge_sq for entry in per_image
        ]
        return vision_embeds, token_counts
