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

"""Text encoder component for WanExecutor."""

from __future__ import annotations

import logging

import numpy as np
from max.driver import CPU, Buffer, load_devices
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.graph.weights import load_weights
from max.pipelines.lib.compiled_component import CompiledComponent
from max.pipelines.lib.config.model_config import (
    _resolve_component_encoding_and_weights,
)
from max.pipelines.lib.model_manifest import ModelManifest
from max.profiler import traced

from ...umt5.model import _prepare_state_dict
from ...umt5.model_config import UMT5Config
from ...umt5.umt5 import UMT5EncoderModel

logger = logging.getLogger("max.pipelines")


class TextEncoder(CompiledComponent):
    """UMT5 text encoder component.

    Compiles the UMT5 encoder model and provides prompt embedding
    generation with padding to a fixed sequence length.
    """

    _model: Model

    # Wan-AI's reference implementation pads tokens to 512 and feeds the
    # full 512-length context to cross-attention. The broader DiT inference
    # ecosystem follows the same convention:
    # - diffusers WanPipeline.__call__ default:
    #   https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/wan/pipeline_wan.py#L403
    # - cache-dit hardcoded 512:
    #   https://github.com/vipshop/cache-dit/blob/v1.3.9/src/cache_dit/distributed/transformers/wan.py#L80
    embed_seq_len: int = 512

    @traced(message="TextEncoder.__init__")
    def __init__(
        self,
        manifest: ModelManifest,
        session: InferenceSession,
    ) -> None:
        super().__init__(manifest, session)

        config = manifest["text_encoder"]
        config_dict = config.huggingface_config.to_dict()
        resolved_encoding, resolved_weight_path = (
            _resolve_component_encoding_and_weights(config)
        )
        encoding = resolved_encoding or "bfloat16"
        devices = load_devices(config.device_specs)

        umt5_config = UMT5Config.generate(config_dict, encoding, devices)
        # Force bfloat16 — some repos declare float32 but should run bf16.
        dtype = DType.bfloat16
        umt5_config.dtype = dtype

        self._device = devices[0]

        # Load weights.
        paths = config.resolved_weight_paths(resolved_weight_path)
        weights = load_weights(paths)
        state_dict = _prepare_state_dict(weights, target_dtype=dtype)

        # Build module.
        dev_ref = DeviceRef.from_device(self._device)
        module = UMT5EncoderModel(umt5_config, dtype=dtype, device=dev_ref)
        module.load_state_dict(state_dict, weight_alignment=1, strict=True)

        # Build graph with symbolic sequence length.
        # The attention mask is derived from non-zero input_ids inside
        # the graph so pad tokens are masked both during encoder attention
        # and during output zeroing. Post-processing (slice to
        # embed_seq_len, mask padding) is folded in so the entire
        # encode+postprocess runs on-device.
        input_types = [
            TensorType(DType.int64, ["batch", "seq_len"], device=self._device),
        ]
        embed_len = self.embed_seq_len
        with Graph("umt5_encoder", input_types=input_types) as graph:
            input_ids = graph.inputs[0].tensor

            # WAN tokenizer pads with id=0; mark valid tokens as 1.
            attention_mask = ops.cast(input_ids != 0, DType.int64)

            hidden_states = module(input_ids, attention_mask)

            # Slice to embed_seq_len along seq dim.
            # Tokenizer pads to 512, so seq_len >= embed_len is guaranteed.
            sliced = hidden_states[:, :embed_len, :]

            # Zero out padding positions using the derived mask.
            mask_sliced = ops.cast(
                attention_mask[:, :embed_len], hidden_states.dtype
            )
            result = sliced * ops.unsqueeze(mask_sliced, -1)

            graph.output(result)

        self._load_graph(graph, weights_registry=module.state_dict())

    @traced(message="TextEncoder.__call__")
    def __call__(
        self,
        token_ids: Buffer,
        num_videos_per_prompt: int,
        max_sequence_length: int | None = None,
    ) -> Buffer:
        """Encode text tokens into prompt embeddings.

        Runs the UMT5 encoder with on-device post-processing (slice to
        ``embed_seq_len``, mask padding positions). The attention mask
        is derived from non-zero ``token_ids`` inside the graph.

        Args:
            token_ids: Token IDs, shape ``(batch, seq)`` or ``(seq,)``
                int64 on device.
            num_videos_per_prompt: Number of videos per prompt for batching.
            max_sequence_length: Must equal ``embed_seq_len`` (512) or
                ``None``. Compiled into the graph at init time.

        Returns:
            Prompt embeddings, shape ``(B, embed_seq_len, hidden_dim)``
            in model dtype.
        """
        if (
            max_sequence_length is not None
            and max_sequence_length != self.embed_seq_len
        ):
            raise ValueError(
                f"max_sequence_length={max_sequence_length} != compiled "
                f"embed_seq_len={self.embed_seq_len}"
            )

        # Ensure 2D input using Buffer metadata (no D2H transfer).
        if token_ids.rank == 1:
            token_ids = token_ids.view(token_ids.dtype, [1, token_ids.shape[0]])

        # Run encoder + on-device post-processing (slice + mask).
        raw = self._model.execute(token_ids)
        result = raw[0] if isinstance(raw, (list, tuple)) else raw

        # Repeat for num_videos_per_prompt (rare, typically 1).
        if num_videos_per_prompt > 1:
            # Small tensor (1, embed_seq_len, hidden_dim) in bf16 — acceptable D2H/H2D.
            result_np = np.from_dlpack(result.to(CPU()))
            repeated = np.ascontiguousarray(
                np.repeat(result_np, num_videos_per_prompt, axis=0)
            )
            result = Buffer.from_numpy(repeated).to(self._device)

        return result
