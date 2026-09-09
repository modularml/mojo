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

"""Qwen2.5-VL multimodal text encoder helpers.

This module owns prompt/image encoding that combines the shared module-v2 text
encoder with the Qwen2.5-VL vision encoder. Pipelines such as QwenImageEdit
should import this helper instead of defining an architecture-local copy.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.graph.type import BufferType
from max.graph.weights import WeightData, Weights
from max.nn.comm import Signals
from max.pipelines.context import TokenBuffer
from max.pipelines.lib import SupportedEncoding
from max.pipelines.lib.bfloat16_utils import float32_to_bfloat16_as_uint16
from max.pipelines.modeling.config_enums import supported_encoding_dtype
from PIL import Image

from ..model_config import VisionConfig
from ..nn.data_processing import (
    get_seqlens,
    get_window_index,
    mrope_pos_ids_3d,
)
from ..nn.qwen_vl_utils import fetch_image
from ..nn.visual_transformer import VisionTransformer
from ..tokenizer import Qwen2_5VLImageProcessor
from ..weight_adapters import QWEN2_5_VL_MODEL_MAPPING
from .model import Qwen25VLEncoderModel

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_DROP_IDX = 64


class Qwen25VLMultimodalEncoderModel:
    """Multimodal prompt encoder built on the shared Qwen2.5-VL components."""

    def __init__(
        self,
        text_encoder: Qwen25VLEncoderModel,
        config: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
        weights: Weights,
        session: InferenceSession,
        tokenizer: Any,
    ) -> None:
        self.text_encoder = text_encoder
        self.devices = devices
        self.session = session
        self.tokenizer = tokenizer
        self.lang_config = text_encoder.config
        self._cached_vision_inputs: dict[
            tuple[int, ...],
            tuple[Buffer, Buffer, Buffer, Buffer, Buffer, Buffer, Buffer],
        ] = {}
        self._cached_scatter_indices: dict[tuple[int, ...], Buffer] = {}
        self._cached_token_buffers: dict[tuple[int, ...], Buffer] = {}

        self._image_token_id = self.tokenizer.convert_tokens_to_ids(
            "<|image_pad|>"
        )
        vision_cfg = config.get("vision_config", {})
        enc_dtype = supported_encoding_dtype(encoding)
        device_ref = DeviceRef.from_device(devices[0])
        self.vision_config = VisionConfig(
            dtype=enc_dtype,
            llm_dtype=enc_dtype,
            devices=[device_ref],
            patch_size=vision_cfg.get("patch_size", 14),
            temporal_patch_size=vision_cfg.get("temporal_patch_size", 2),
            in_channels=vision_cfg.get("in_channels", 3),
            hidden_size=vision_cfg.get("hidden_size", 1280),
            num_attention_heads=vision_cfg.get("num_heads", 16),
            depth=vision_cfg.get("depth", 32),
            intermediate_size=vision_cfg.get("intermediate_size", 5120),
            out_hidden_size=vision_cfg.get(
                "out_hidden_size", self.lang_config.hidden_size
            ),
            fullatt_block_indexes=vision_cfg.get(
                "fullatt_block_indexes",
                [7, 15, 23, 31],
            ),
            rms_norm_eps=vision_cfg.get("rms_norm_eps", 1e-6),
            window_size=vision_cfg.get("window_size", 112),
            spatial_merge_size=vision_cfg.get("spatial_merge_size", 2),
        )
        self.image_processor = Qwen2_5VLImageProcessor(
            patch_size=self.vision_config.patch_size,
            temporal_patch_size=self.vision_config.temporal_patch_size,
            merge_size=self.vision_config.spatial_merge_size,
        )

        self._compile_vision_encoder(weights)
        self._compile_hidden_state_trimmer()
        self._compile_vision_merger()
        self._compile_hidden_state_tiler()

    def _compile_vision_encoder(self, weights: Weights) -> None:
        device_ref = DeviceRef.from_device(self.devices[0])
        vc = self.vision_config
        patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size**2

        vision_state: dict[str, Any] = {}
        for key, value in weights.items():
            wd = value.data()
            if wd.dtype.is_float() and not wd.dtype.is_float8():
                is_scale = key.endswith((".weight_scale", ".input_scale"))
                if not is_scale:
                    wd = wd.astype(DType.bfloat16)

            if "patch_embed.proj." in key:
                buf = Buffer.from_dlpack(wd.data)
                oc, ic, kh, kw, kd = buf.shape
                buf = buf.view(dtype=buf.dtype, shape=(oc, ic * kh * kw * kd))
                wd = WeightData(
                    data=buf,
                    name=wd.name,
                    dtype=wd.dtype,
                    shape=wd.shape.__class__(buf.shape),
                    quantization_encoding=wd.quantization_encoding,
                )

            mapped = key
            for before, after in QWEN2_5_VL_MODEL_MAPPING.items():
                mapped = mapped.replace(before, after)

            if mapped.startswith("vision_encoder."):
                vision_state[mapped[len("vision_encoder.") :]] = wd
            elif mapped.startswith("merger."):
                vision_state[mapped] = wd

        vision_transformer = VisionTransformer(vc)
        vision_transformer.load_state_dict(
            vision_state, weight_alignment=1, strict=True
        )

        signals = Signals(devices=[device_ref])
        input_types: list[TensorType | BufferType] = [
            TensorType(
                vc.dtype,
                shape=["vision_seq_len", patch_dim],
                device=device_ref,
            ),
            TensorType(
                DType.int64, shape=["vision_seq_len", 2], device=device_ref
            ),
            TensorType(
                DType.int64, shape=["window_seq_len"], device=device_ref
            ),
            TensorType(
                DType.uint32, shape=["n_full_seqlens"], device=device_ref
            ),
            TensorType(
                DType.uint32, shape=["n_win_seqlens"], device=device_ref
            ),
            TensorType(DType.uint32, shape=[1], device=DeviceRef.CPU()),
            TensorType(DType.uint32, shape=[1], device=DeviceRef.CPU()),
            TensorType(DType.int32, shape=[], device=DeviceRef.CPU()),
            *signals.input_types(),
        ]

        with Graph("qwen_edit_vision", input_types=input_types) as vision_graph:
            ins = vision_graph.inputs
            signal_buffers = [inp.buffer for inp in ins[8:]]
            outputs = vision_transformer(
                pixel_values=[ins[0].tensor],
                rot_pos_ids=[ins[1].tensor],
                window_index=[ins[2].tensor],
                cu_seqlens=[ins[3].tensor],
                cu_window_seqlens=[ins[4].tensor],
                max_seqlen=[ins[5].tensor],
                max_window_seqlen=[ins[6].tensor],
                max_grid_size=[ins[7].tensor],
                signal_buffers=signal_buffers,
            )
            vision_graph.output(outputs[0])

        self._vision_model: Model = self.session.load(
            vision_graph, weights_registry=vision_transformer.state_dict()
        )
        self._vision_signals = signals

    def _compile_hidden_state_trimmer(self) -> None:
        device_ref = DeviceRef.from_device(self.devices[0])
        hidden_size = self.lang_config.hidden_size

        with Graph(
            "qwen_edit_trim_hidden_states",
            input_types=[
                TensorType(
                    self.lang_config.dtype,
                    shape=["total_seq_len", hidden_size],
                    device=device_ref,
                )
            ],
        ) as graph:
            hidden_states = graph.inputs[0].tensor
            trimmed = ops.slice_tensor(
                hidden_states,
                [slice(PROMPT_TEMPLATE_DROP_IDX, None), slice(None)],
            )
            graph.output(ops.unsqueeze(trimmed, 0))

        self._hidden_state_trimmer: Model = self.session.load(graph)

    def _compile_vision_merger(self) -> None:
        device_ref = DeviceRef.from_device(self.devices[0])
        hidden_size = self.lang_config.hidden_size

        with Graph(
            "qwen_edit_merge_vision_embeddings",
            input_types=[
                TensorType(
                    self.lang_config.dtype,
                    shape=["total_seq_len", hidden_size],
                    device=device_ref,
                ),
                TensorType(
                    self.lang_config.dtype,
                    shape=["num_image_tokens", hidden_size],
                    device=device_ref,
                ),
                TensorType(
                    DType.int64,
                    shape=["num_image_tokens", hidden_size],
                    device=device_ref,
                ),
            ],
        ) as graph:
            hidden_states = graph.inputs[0].tensor
            vision_embeds = graph.inputs[1].tensor
            image_token_indices = graph.inputs[2].tensor
            graph.output(
                ops.scatter(
                    input=hidden_states,
                    updates=vision_embeds,
                    indices=image_token_indices,
                    axis=0,
                )
            )

        self._vision_merger: Model = self.session.load(graph)

    def _compile_hidden_state_tiler(self) -> None:
        device_ref = DeviceRef.from_device(self.devices[0])
        hidden_size = self.lang_config.hidden_size

        with Graph(
            "qwen_edit_tile_hidden_states",
            input_types=[
                TensorType(
                    self.lang_config.dtype,
                    shape=[1, "trimmed_seq_len", hidden_size],
                    device=device_ref,
                )
            ],
        ) as graph:
            hidden_states = graph.inputs[0].tensor
            graph.output(ops.tile(hidden_states, (2, 1, 1)))

        self._repeat_two_hidden_states: Model = self.session.load(graph)

    def _prepare_images(
        self, images: list[npt.NDArray[np.uint8]]
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.uint16]]:
        processed_images = [
            fetch_image({"image": Image.fromarray(image).convert("RGB")})
            for image in images
        ]
        processed = self.image_processor(
            images=processed_images, return_tensors="np"
        )
        processed_dict = (
            processed[0] if isinstance(processed, tuple) else processed
        )
        image_grid_thw = np.asarray(
            processed_dict["image_grid_thw"], dtype=np.int64
        )
        pixel_values = np.asarray(
            processed_dict.get(
                "pixel_values",
                processed_dict.get("concatenated_pixel_values"),
            )
        )
        if pixel_values.dtype == np.uint16:
            pixel_values_u16 = pixel_values
        else:
            pixel_values_u16 = float32_to_bfloat16_as_uint16(
                np.ascontiguousarray(pixel_values.astype(np.float32))
            )
        return image_grid_thw, pixel_values_u16

    def _run_vision_encoder(
        self,
        image_grid_thw: npt.NDArray[np.int64],
        pixel_values_u16: npt.NDArray[np.uint16],
    ) -> Buffer:
        vc = self.vision_config
        device = self.devices[0]

        rot_pos_ids = mrope_pos_ids_3d(image_grid_thw, vc.spatial_merge_size)
        window_idx, cu_win_seqlens = get_window_index(
            image_grid_thw,
            window_size=vc.window_size,
            spatial_merge_size=vc.spatial_merge_size,
            patch_size=vc.patch_size,
            spatial_merge_unit=vc.spatial_merge_size**2,
        )
        cu_seqlens, cu_window_seqlens, max_seqlen, max_window_seqlen = (
            get_seqlens(image_grid_thw, cu_win_seqlens)
        )
        max_grid_size = int(image_grid_thw[:, 1:].max())
        grid_key = tuple(int(x) for x in image_grid_thw.reshape(-1))

        if grid_key not in self._cached_vision_inputs:
            self._cached_vision_inputs[grid_key] = (
                Buffer.from_numpy(
                    np.ascontiguousarray(rot_pos_ids.astype(np.int64))
                ).to(device),
                Buffer.from_numpy(
                    np.ascontiguousarray(window_idx.astype(np.int64))
                ).to(device),
                Buffer.from_numpy(
                    np.ascontiguousarray(cu_seqlens.astype(np.uint32))
                ).to(device),
                Buffer.from_numpy(
                    np.ascontiguousarray(cu_window_seqlens.astype(np.uint32))
                ).to(device),
                Buffer.from_numpy(np.array([max_seqlen], dtype=np.uint32)),
                Buffer.from_numpy(
                    np.array([max_window_seqlen], dtype=np.uint32)
                ),
                Buffer.from_numpy(np.array(max_grid_size, dtype=np.int32)),
            )
        (
            rot_pos_ids_buf,
            window_idx_buf,
            cu_seqlens_buf,
            cu_window_seqlens_buf,
            max_seqlen_buf,
            max_window_seqlen_buf,
            max_grid_size_buf,
        ) = self._cached_vision_inputs[grid_key]

        if vc.dtype == DType.bfloat16:
            pv_buf = Buffer.from_numpy(
                np.ascontiguousarray(pixel_values_u16)
            ).to(device)
            pv_buf = pv_buf.view(
                dtype=DType.bfloat16, shape=pixel_values_u16.shape
            )
        else:
            pixel_values = (pixel_values_u16.astype(np.uint32) << 16).view(
                np.float32
            )
            if vc.dtype == DType.float16:
                pixel_values = pixel_values.astype(np.float16)
            pv_buf = Buffer.from_numpy(np.ascontiguousarray(pixel_values)).to(
                device
            )

        result = self._vision_model.execute(
            pv_buf,
            rot_pos_ids_buf,
            window_idx_buf,
            cu_seqlens_buf,
            cu_window_seqlens_buf,
            max_seqlen_buf,
            max_window_seqlen_buf,
            max_grid_size_buf,
            *self._vision_signals.buffers(),
        )
        return result[0]

    def encode(
        self,
        tokens: TokenBuffer,
        images: list[npt.NDArray[np.uint8]] | None = None,
        num_images_per_prompt: int = 1,
    ) -> Buffer:
        device = self.devices[0]

        pixel_values_u16: npt.NDArray[np.uint16] | None = None
        image_grid_thw: npt.NDArray[np.int64] | None = None
        if images:
            image_grid_thw, pixel_values_u16 = self._prepare_images(images)

        input_ids = (
            np.asarray(tokens.array).flatten().astype(np.int64, copy=False)
        )
        token_key = tuple(int(token) for token in input_ids.tolist())
        if token_key not in self._cached_token_buffers:
            self._cached_token_buffers[token_key] = Buffer.from_numpy(
                np.ascontiguousarray(input_ids)
            ).to(device)
        token_buf = self._cached_token_buffers[token_key]
        embed_result = self.text_encoder._embed_model.execute(token_buf)
        lc = self.lang_config
        merged_buf = embed_result[0]

        if images:
            if image_grid_thw is None or pixel_values_u16 is None:
                raise ValueError("vision inputs are required when images exist")
            vision_emb = self._run_vision_encoder(
                image_grid_thw, pixel_values_u16
            )
            pad_positions = np.where(input_ids == self._image_token_id)[0]
            if len(pad_positions) == vision_emb.shape[0]:
                scatter_key = tuple(int(x) for x in pad_positions.tolist())
                if scatter_key not in self._cached_scatter_indices:
                    scatter_indices = np.tile(
                        pad_positions[:, np.newaxis],
                        (1, vision_emb.shape[1]),
                    ).astype(np.int64, copy=False)
                    self._cached_scatter_indices[scatter_key] = (
                        Buffer.from_numpy(
                            np.ascontiguousarray(scatter_indices)
                        ).to(device)
                    )
                merged_buf = self._vision_merger.execute(
                    merged_buf,
                    vision_emb,
                    self._cached_scatter_indices[scatter_key],
                )[0]
            else:
                logger.warning(
                    "Vision token mismatch: %d pads vs %d embeddings. Skipping merge.",
                    len(pad_positions),
                    vision_emb.shape[0],
                )

        hs_buf = self.text_encoder._transform_model.execute(merged_buf)[0]
        trimmed_buf = self._hidden_state_trimmer.execute(hs_buf)[0]
        if num_images_per_prompt == 1:
            return trimmed_buf
        if num_images_per_prompt == 2:
            return self._repeat_two_hidden_states.execute(trimmed_buf)[0]

        hs_cpu = hs_buf.to(CPU())
        if lc.dtype == DType.bfloat16:
            hs_u16 = np.from_dlpack(
                hs_cpu.view(dtype=DType.uint16, shape=hs_cpu.shape)
            )
            hs_np = (hs_u16.astype(np.uint32) << 16).view(np.float32)
        else:
            hs_np = np.from_dlpack(hs_cpu).astype(np.float32)

        hs_np = hs_np[PROMPT_TEMPLATE_DROP_IDX:]
        hs_np = hs_np[np.newaxis, :, :]
        hs_np = np.repeat(hs_np, num_images_per_prompt, axis=0)

        if lc.dtype == DType.bfloat16:
            result_u16 = float32_to_bfloat16_as_uint16(
                np.ascontiguousarray(hs_np)
            )
            buf = Buffer.from_numpy(result_u16).to(device)
            return buf.view(dtype=DType.bfloat16, shape=hs_np.shape)
        return Buffer.from_numpy(np.ascontiguousarray(hs_np)).to(device)
