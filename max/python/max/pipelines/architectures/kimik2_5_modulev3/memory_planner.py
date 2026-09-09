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

"""Memory planner for the Kimi K2.5 architecture."""

from __future__ import annotations

import logging

from max.dtype import DType
from max.nn.comm.ep.ep_config import (
    calculate_ep_max_tokens_per_rank,
    estimate_ep_memory_usage,
)
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.modeling.config_enums import (
    is_float4_encoding,
    supported_encoding_dtype,
)
from max.support.human_readable_formatter import to_human_readable_bytes
from transformers import AutoConfig

from .model_config import KimiK2_5Config, VisionConfig

logger = logging.getLogger(__name__)

# Empirical coefficient for peak activation bytes per vision-encoder patch.
# See KimiK2_5Model._VISION_PEAK_BYTES_PER_PATCH_COEFF for derivation notes.
_VISION_PEAK_BYTES_PER_PATCH_COEFF = 20


class KimiK25MemoryPlanner(PagedMemoryPlanner):
    """Memory planner for Kimi K2.5 (vision-language, MoE) models.

    Accounts for replicated vision encoder weights, expert-parallel routing,
    MLA up-projection buffers, and vision encoder activation memory.
    """

    _always_signal_buffers = True

    # ------------------------------------------------------------------
    # Public estimation methods
    # ------------------------------------------------------------------

    def estimate_weights_size(self, pipeline_config: PipelineConfig) -> int:
        """Estimates weight memory for Kimi K2.5 models.

        Accounts for MoE expert sharding, replicated vision encoder weights,
        and MLA up-projection buffers.

        Args:
            pipeline_config: Pipeline configuration.

        Returns:
            Estimated weight memory in bytes.
        """
        model_config = pipeline_config.model
        weights_size = model_config.weights_size()
        n_gpus_per_node = len(model_config.device_specs)

        encoding = _select_quantization_encoding(
            pipeline_config.model, KimiK2_5Config.DEFAULT_ENCODING
        )

        def _n_elems_to_bytes(n_elems: int) -> int:
            dtype = supported_encoding_dtype(encoding).size_in_bytes
            if is_float4_encoding(encoding):
                # Account for the scales. For NVFP4 format, every 16 FP4 elements
                # share one FP8 scale factor. The size of the scales is one
                # eighth of the size of the FP4 quants (8 bits / (16 * 4 bits)).
                return int(n_elems // 2 * dtype * 1.125)
            else:
                return n_elems * dtype

        assert model_config.huggingface_config is not None
        config = model_config.huggingface_config.text_config
        assert config is not None
        n_sparse_layers = (
            config.num_hidden_layers - config.first_k_dense_replace
        )
        n_mtp_layers = config.num_nextn_predict_layers

        # Note: All the following calculations are not exact, but they are
        # better than directly using the raw weights size.

        # First, calculate the lm_head/embed_tokens size.
        # These are always in BF16.
        lm_head_size = (
            config.vocab_size
            * config.hidden_size
            * DType.bfloat16.size_in_bytes
        )
        embed_tokens_size = lm_head_size

        # Subtract the lm_head/embed_tokens size from the weights size
        weights_size -= lm_head_size + embed_tokens_size
        weights_size -= (lm_head_size + embed_tokens_size) * n_mtp_layers

        # We don't use the MTP module for now, so subtract the MTP attn/moe size.
        # Estimate the MTP module size by assuming the MTP layer is of the same
        # size as a sparse model layer.
        weights_size = int(
            weights_size * n_sparse_layers / (n_sparse_layers + n_mtp_layers)
        )

        # Calculate the routing experts and the shared experts size.
        expert_elems = (
            config.moe_intermediate_size * config.hidden_size * 3
        )  # A factor of 3 accounts for the gate/up/down proj weights.
        expert_size = _n_elems_to_bytes(expert_elems)
        routing_experts_size = (
            n_sparse_layers * config.n_routed_experts * expert_size
        )
        shared_experts_size = (
            n_sparse_layers * config.n_shared_experts * expert_size
        )

        # The vision encoder (patch_embed + transformer encoder) is REPLICATED
        # on every device. The patch_merger is tensor-parallel, so it correctly
        # stays in the LM-attention TP pool below.
        hf_vision_cfg = getattr(
            model_config.huggingface_config, "vision_config", None
        )
        vision_config = (
            VisionConfig.initialize_from_config(
                pipeline_config,
                hf_vision_cfg,
                huggingface_config=model_config.huggingface_config,
            )
            if hf_vision_cfg is not None
            else None
        )
        replicated_vision_bytes = _estimate_replicated_vision_weights_bytes(
            vision_config
        )

        # Estimate the size of the attention weights.
        attn_weights_size = (
            weights_size
            - routing_experts_size
            - shared_experts_size
            - replicated_vision_bytes
        )

        # If we use DP attention, attention weights are duplicated on each DP rank.
        total_size = attn_weights_size * model_config.data_parallel_degree

        # The shared experts are duplicated on each device.
        total_size += shared_experts_size * n_gpus_per_node

        # Replicated vision encoder weights live on every GPU.
        total_size += replicated_vision_bytes * n_gpus_per_node

        ep_size = max(pipeline_config.runtime.ep_size, 1)
        if ep_size == 1:
            total_size += routing_experts_size
        else:
            # we don't support mixing EP and TP strategies yet.
            # ep_size must be equal to n_gpus_per_node * n_nodes
            assert ep_size % n_gpus_per_node == 0
            n_nodes = ep_size // n_gpus_per_node
            total_size += routing_experts_size // n_nodes

        # Add back the lm_head/embed_tokens size, they will never be duplicated.
        total_size += lm_head_size + embed_tokens_size

        if replicated_vision_bytes:
            logger.info(
                "Estimated replicated vision encoder weights: %s per device, "
                "%s cluster-wide (%d devices)",
                to_human_readable_bytes(replicated_vision_bytes),
                to_human_readable_bytes(
                    replicated_vision_bytes * n_gpus_per_node
                ),
                n_gpus_per_node,
            )

        return total_size

    def estimate_activation_memory(
        self,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        """Estimates activation memory for Kimi K2.5 models.

        Accounts for MLA up-projection, MoE EP routing buffers, vision encoder
        peak activations, and optional device-graph-capture headroom.

        Args:
            pipeline_config: Pipeline configuration.
            huggingface_config: HuggingFace model configuration.

        Returns:
            Estimated activation memory in bytes.
        """
        encoding = _select_quantization_encoding(
            pipeline_config.model, KimiK2_5Config.DEFAULT_ENCODING
        )
        mla_activation_memory: int = 0
        moe_activation_memory: int = 0
        ep_buffer_memory = 0

        # During prefill, up-project all KV cache for current requests.
        if pipeline_config.runtime.pipeline_role != "decode_only":
            max_kv_length: int = 0
            if pipeline_config.runtime.max_batch_total_tokens is None:
                max_kv_length = pipeline_config.model.max_length or 0
            else:
                max_kv_length = pipeline_config.runtime.max_batch_total_tokens

            cache_dtype = cache_dtype_for_encoding(
                encoding,
                pipeline_config.model.kv_cache.kv_cache_format,
            )
            mla_activation_memory += (
                pipeline_config.model.data_parallel_degree
                * 2  # 2 for K and V
                * max_kv_length
                * huggingface_config.text_config.num_attention_heads
                * huggingface_config.text_config.qk_nope_head_dim
                * cache_dtype.size_in_bytes
            )

        # Estimate buffer and activation memory during Expert Parallel MoE.
        if pipeline_config.runtime.ep_size > 1:
            n_gpus_per_node = len(pipeline_config.model.device_specs)

            ep_max_rank_send_tokens = calculate_ep_max_tokens_per_rank(
                max_batch_input_tokens=pipeline_config.runtime.max_batch_input_tokens,
                ep_size=pipeline_config.runtime.ep_size,
                data_parallel_degree=pipeline_config.model.data_parallel_degree,
                use_allreduce=pipeline_config.runtime.ep_use_allreduce,
            )

            max_recv_tokens_per_rank = ep_max_rank_send_tokens * min(
                huggingface_config.text_config.n_routed_experts,
                pipeline_config.runtime.ep_size
                * huggingface_config.text_config.num_experts_per_tok,
            )

            if pipeline_config.runtime.ep_use_allreduce:
                max_recv_tokens_per_rank = (
                    pipeline_config.runtime.max_batch_input_tokens
                    * min(
                        huggingface_config.text_config.n_routed_experts
                        // n_gpus_per_node,
                        huggingface_config.text_config.num_experts_per_tok,
                    )
                )

            moe_activation_memory += (
                max_recv_tokens_per_rank
                * huggingface_config.text_config.moe_intermediate_size
                * supported_encoding_dtype(encoding).size_in_bytes
            )
            moe_activation_memory += (
                max_recv_tokens_per_rank
                * huggingface_config.text_config.hidden_size
                * DType.bfloat16.size_in_bytes
            )
            moe_activation_memory += 256 * 1024 * 1024
            moe_activation_memory *= n_gpus_per_node

            n_nodes = pipeline_config.runtime.ep_size // n_gpus_per_node
            per_device_ep_memory = estimate_ep_memory_usage(
                hidden_size=huggingface_config.text_config.hidden_size,
                dispatch_dtype=supported_encoding_dtype(encoding),
                combine_dtype=DType.bfloat16,
                max_tokens_per_rank=ep_max_rank_send_tokens,
                n_experts=huggingface_config.text_config.n_routed_experts,
                n_nodes=n_nodes,
                n_gpus_per_node=n_gpus_per_node,
                top_k=huggingface_config.text_config.num_experts_per_tok,
                use_allreduce=pipeline_config.runtime.ep_use_allreduce,
            )
            ep_buffer_memory = per_device_ep_memory * n_gpus_per_node
            logger.info(
                "Estimated EP SHMEM buffer memory: %s",
                to_human_readable_bytes(ep_buffer_memory),
            )

        # Vision encoder activation memory.
        hf_vision_cfg = getattr(huggingface_config, "vision_config", None)
        vision_config = (
            VisionConfig.initialize_from_config(
                pipeline_config,
                hf_vision_cfg,
                huggingface_config=huggingface_config,
            )
            if hf_vision_cfg is not None
            else None
        )
        vision_activation_memory = _estimate_vision_activation_memory(
            pipeline_config=pipeline_config,
            vision_config=vision_config,
        )

        # MLA, MoE, and vision activations are transient and mutually
        # exclusive in time; EP SHMEM buffers are persistent and stack on top.
        activation_memory = max(
            mla_activation_memory,
            moe_activation_memory,
            vision_activation_memory,
        )
        activation_memory += ep_buffer_memory

        if vision_activation_memory:
            logger.info(
                "Estimated vision encoder activation memory: %s",
                to_human_readable_bytes(vision_activation_memory),
            )

        if activation_memory != 0:
            logger.info(
                "Estimated activation memory: %s",
                to_human_readable_bytes(activation_memory),
            )

        return activation_memory


# ------------------------------------------------------------------
# Module-level helpers (shared with model.py via import)
# ------------------------------------------------------------------


def _vision_encoder_token_budget(
    pipeline_config: PipelineConfig,
) -> int | None:
    """Return the per-call image-token ceiling for the vision encoder.

    Returns ``max_batch_input_tokens`` so the vision encoder respects the
    same input-token budget the LM forward pass honors under chunked prefill.
    Returns ``None`` when ``max_batch_input_tokens`` is unset, signalling
    that chunking should be disabled.
    """
    max_batch_input_tokens = int(
        pipeline_config.runtime.max_batch_input_tokens or 0
    )
    if max_batch_input_tokens <= 0:
        return None
    return max_batch_input_tokens


def _estimate_replicated_vision_weights_bytes(
    vision_config: VisionConfig | None,
) -> int:
    """Estimate per-device bytes for replicated vision encoder weights."""
    if vision_config is None:
        return 0

    encoder_params_per_layer = (
        4 * vision_config.vt_hidden_size * vision_config.vt_hidden_size
        + 2 * vision_config.vt_hidden_size * vision_config.vt_intermediate_size
    )
    encoder_params = (
        vision_config.vt_num_hidden_layers * encoder_params_per_layer
    )

    patch_embed_params = (
        vision_config.vt_hidden_size
        * vision_config.in_channels
        * vision_config.patch_size
        * vision_config.patch_size
        + vision_config.init_pos_emb_height
        * vision_config.init_pos_emb_width
        * vision_config.vt_hidden_size
    )

    total_params = encoder_params + patch_embed_params
    return int(total_params * DType.bfloat16.size_in_bytes)


def _vision_merge_sq(vision_config: VisionConfig) -> int:
    """Patches-per-output-token from the vision config's merge kernel."""
    merge_kernel_size = vision_config.merge_kernel_size
    return max(1, merge_kernel_size[0] * merge_kernel_size[1])


def _estimate_vision_activation_memory(
    pipeline_config: PipelineConfig,
    vision_config: VisionConfig | None,
) -> int:
    """Estimate vision encoder peak activation memory cluster-wide."""
    if vision_config is None:
        return 0

    token_budget = _vision_encoder_token_budget(pipeline_config)
    if token_budget is None:
        return 0

    merge_sq = _vision_merge_sq(vision_config)
    patches_per_call = token_budget * merge_sq

    per_device_bytes = (
        patches_per_call
        * vision_config.vt_hidden_size
        * _VISION_PEAK_BYTES_PER_PATCH_COEFF
    )

    n_devices = len(pipeline_config.model.device_specs)
    return int(per_device_bytes * n_devices)
