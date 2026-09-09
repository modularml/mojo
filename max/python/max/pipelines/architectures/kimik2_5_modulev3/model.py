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
"""Implements the Kimi-K2.5 pipeline model using the ModuleV3 (eager) API.

The vision tower runs single-device (V3 does not shard the vision encoder);
the language tower is the multi-GPU DeepseekV3 ModuleV3 model (TP attention +
EP MoE via a :class:`DeviceMesh`). The two towers are compiled as separate
callables, mirroring ``gemma3multimodal_modulev3``.

The text path is wired for bf16, FP8, and NVFP4 under the DeepseekV3 ModuleV3
DP-attention + EP-MoE ABI (:class:`KimiK2_5ModelInputs` below, produced by
``batch_processor.py``). Validated end to end on 8xB200 for NVFP4
(``nvidia/Kimi-K2.5-NVFP4``, DP=8/EP=8).

TODO(MODELS-kimi-v3): the multimodal (image) path under the multi-GPU V3 ABI
and EPLB are not yet validated; MXFP4 preshuffles are not ported (the example
checkpoints are NVFP4).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

from max.driver import Buffer, Device, DeviceSpec, is_virtual_device_mode
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental import functional as F
from max.experimental.nn import CompiledModel
from max.experimental.sharding import (
    DeviceMesh,
    DistributedTensorType,
    Replicated,
)
from max.experimental.tensor import default_dtype
from max.graph import DeviceRef, TensorType
from max.graph.weights import WeightData, Weights, WeightsAdapter
from max.nn.comm.ep import (
    EPBatchManager,
    EPCommInitializer,
    EPConfig,
    calculate_ep_max_tokens_per_rank,
)
from max.nn.kv_cache import KVCacheParamInterface
from max.nn.transformer import ReturnLogits
from max.pipelines.context import ImageMetadata
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    ModuleV3MultiGraphPipelineModelWithKVCache,
    PipelineConfig,
)
from max.pipelines.lib.interfaces.batch_processor import (
    modulev3_ragged_kv_symbolic_inputs,
)
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.pipelines.lib.vision_encoder_cache import VisionEncodeResult
from max.pipelines.weights.quant import parse_quant_config
from transformers import AutoConfig

from ..deepseekV3_modulev3.layers.quant_ops import (
    routed_weight_dtype,
)
from .batch_processor import KimiK2_5BatchProcessor
from .context import KimiK2_5TextAndVisionContext
from .kimi_nvfp4_policy import infer_kimi_nvfp4_weight_flags
from .layers.language_model import KimiK2_5MoEDecoder
from .layers.vision.transformer import Transformer
from .model_config import KimiK2_5Config, KimiK2_5TextConfig

logger = logging.getLogger("max.pipelines")


@dataclass
class KimiK2_5ModelInputs(ModelInputs):
    """Flat ModuleV3 inputs for the Kimi-K2.5 model.

    The language ABI is ``(tokens, return_n_logits, input_row_offsets,
    vision_embeddings, vision_scatter_indices, *kv, *ep)`` — the DeepseekV3
    ModuleV3 order with the two multimodal tensors spliced in after the row
    offsets. ``vision_embeddings``/``vision_scatter_indices`` are the base
    :class:`ModelInputs` fields, set by the pipeline's vision seam
    (``finalize_vision_inputs``); replicated per device (one ``Buffer`` per
    device, identical data). Shape ``[num_patches, hidden]`` /
    ``[num_image_tokens]`` during prefill, ``[0, hidden]`` / ``[0]`` otherwise.
    """

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer

    batch_context_lengths: list[Buffer] = field(kw_only=True)
    """Host (CPU) page-aligned KV context length, one per DP replica.

    Substituted for the planner's device-resident ``buffer_lengths`` so the
    per-layer ``.to(CPU())`` stays host-to-host and the graph is capturable."""

    data_parallel_splits: Buffer | None = field(default=None, kw_only=True)
    input_row_offsets_i64: Buffer | None = field(default=None, kw_only=True)
    ep_inputs: tuple[Buffer, ...] = field(default=(), kw_only=True)

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        """Flat language-model input tuple in compile ABI order."""
        dp_inputs: tuple[Buffer, ...] = ()
        if self.data_parallel_splits is not None:
            assert self.input_row_offsets_i64 is not None
            dp_inputs = (self.data_parallel_splits, self.input_row_offsets_i64)
        return (
            self.tokens,
            self.return_n_logits,
            self.input_row_offsets,
            *self.vision_embeddings,
            *self.vision_scatter_indices,
            *self.batch_context_lengths,
            *dp_inputs,
            *(
                self.kv_cache_inputs.flatten()
                if self.kv_cache_inputs is not None
                else ()
            ),
            *self.ep_inputs,
        )


class KimiK2_5Model(
    ModuleV3MultiGraphPipelineModelWithKVCache[KimiK2_5TextAndVisionContext]
):
    """A Kimi-K2.5 multimodal pipeline model (ModuleV3)."""

    model_config_cls: ClassVar[type[Any]] = KimiK2_5Config
    batch_processor_cls: ClassVar[type[KimiK2_5BatchProcessor]] = (
        KimiK2_5BatchProcessor
    )

    vision_model: CompiledModel[Any, Any] | None
    language_model: CompiledModel[Any, Any]

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        *,
        memory_plan: MemoryPlan,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        max_batch_size: int = 1,
    ) -> None:
        if pipeline_config.model.device_specs[0] == DeviceSpec.cpu():
            raise ValueError("Kimi-K2.5 is only supported on GPU.")
        self.session = session
        self._ep_batch_manager: EPBatchManager | None = None
        self.ep_comm_initializer: EPCommInitializer | None = None
        self._modulev3_extra_input_types: list[Any] = []
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter=adapter,
            return_logits=return_logits,
            max_batch_size=max_batch_size,
            memory_plan=memory_plan,
        )

        # The base hook is typed loosely (``Callable``); both towers here come
        # out of ``Module.compile``.
        vision_model, language_model = self.load_model()
        assert isinstance(language_model, CompiledModel)
        assert vision_model is None or isinstance(vision_model, CompiledModel)
        self.vision_model, self.language_model = vision_model, language_model

        if self._batch_processor is not None:
            assert isinstance(self._batch_processor, KimiK2_5BatchProcessor)
            assert self.model_config is not None
            self._batch_processor.bind_model_config(self.model_config)
            assert self.vision_model is not None
            self._batch_processor.bind_vision_encoder(
                vision_model=self.vision_model,
                session=self.session,
            )
            self._batch_processor.bind_ep_comm_initializer(
                self.ep_comm_initializer
            )

    @property
    def model(self) -> CompiledModel[Any, Any]:
        """Expose language model for graph capture/replay.

        Only the language model is captured since vision runs
        during prefill. The capture runner unwraps ``engine_model`` and
        appends ``signal_buffers`` for a compiled ModuleV3 model.
        """
        return self.language_model

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParamInterface:
        return KimiK2_5TextConfig.construct_kv_params(
            huggingface_config=huggingface_config.text_config,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        return KimiK2_5Config.get_num_layers(huggingface_config)

    def _load_state_dict(self) -> dict[str, Any]:
        if self.adapter:
            merged = self.adapter(
                dict(self.weights.items()),
                huggingface_config=self.huggingface_config,
                pipeline_config=self.pipeline_config,
            )
        else:
            merged = {key: value.data() for key, value in self.weights.items()}

        # The adapter emits ``vision_encoder.*`` and ``language_model.*`` keys.
        # Each tower compiles from its own root module, so strip the
        # ``vision_encoder.`` prefix for the vision weights and keep the
        # ``language_model.`` prefix for the language weights.
        self._vision_weights_dict = {
            key[len("vision_encoder.") :]: value
            for key, value in merged.items()
            if key.startswith("vision_encoder.")
        }
        self._language_weights_dict = {
            key: value
            for key, value in merged.items()
            if key.startswith("language_model.")
        }
        return merged

    def _create_model_config(
        self, state_dict: dict[str, WeightData]
    ) -> KimiK2_5Config:
        model_config = KimiK2_5Config.initialize_from_config(
            pipeline_config=self.pipeline_config,
            huggingface_config=self.huggingface_config,
            max_seq_len=self.max_seq_len,
            llm_config=KimiK2_5TextConfig.initialize(
                self.pipeline_config, max_seq_len=self.max_seq_len
            ),
        )
        llm = model_config.llm_config

        # Parse the checkpoint's quant config for FP8 and NVFP4 (float4 packs
        # into uint8) checkpoints, mirroring deepseekV3_modulev3. NVFP4 Kimi
        # checkpoints may leave the shared experts in bf16 (e.g.
        # nvidia/Kimi-K2.6-NVFP4); detect that and record it on the quant
        # config so MoEQuantized loads the shared expert at the right dtype.
        quant_config = None
        if self.dtype in (
            DType.float8_e4m3fn,
            DType.uint8,
            DType.float4_e2m1fn,
        ):
            quant_config = parse_quant_config(
                self.huggingface_config.text_config, state_dict, self.dtype
            )
            shared_experts_weight_dtype, dense_mlp_layers_without_quant = (
                infer_kimi_nvfp4_weight_flags(
                    state_dict,
                    first_k_dense_replace=llm.first_k_dense_replace,
                    quant_config=quant_config,
                )
            )
            if (
                quant_config is not None
                and shared_experts_weight_dtype is not None
            ):
                quant_config = replace(
                    quant_config,
                    shared_experts_weight_dtype=shared_experts_weight_dtype,
                )
            llm.dense_mlp_layers_without_quant = dense_mlp_layers_without_quant
        llm.quant_config = quant_config
        # Kimi K2.5 keeps the entire attention block (including o_proj) in bf16
        # (the checkpoint's modelopt ``ignore`` list covers all ``self_attn``),
        # unlike DeepSeek-V3 NVFP4 which quantizes o_proj.
        llm.mla_o_proj_quantized = False

        llm.max_batch_context_length = (
            self.planned_max_batch_total_tokens or llm.max_batch_context_length
        )

        if llm.topk_method == "noaux_tc":
            for key, value in state_dict.items():
                if key.endswith("e_score_correction_bias"):
                    llm.correction_bias_dtype = value.dtype
                    break

        n_devices = len(self.devices)
        dp_degree = self.pipeline_config.model.data_parallel_degree
        if dp_degree > 1 and dp_degree != n_devices:
            raise NotImplementedError(
                f"data_parallel_degree={dp_degree} must equal the device "
                f"count ({n_devices}); dp x tp meshes are not supported yet."
            )
        # DP attention needs a "dp"-named mesh axis so the tower replicates
        # (rather than head-shards) attention weights; pure TP uses "tp".
        axis_name = "dp" if dp_degree > 1 else "tp"
        llm.mesh = DeviceMesh(tuple(self.devices), (n_devices,), (axis_name,))

        self.model_config = model_config
        return model_config

    def _module_default_dtype(self, model_config: KimiK2_5Config) -> DType:
        # Quantized checkpoints keep norms/biases/embeddings in bf16.
        if model_config.llm_config.quant_config is not None:
            return DType.bfloat16
        return model_config.dtype

    def _init_distributed_runtime(self, model_config: KimiK2_5Config) -> None:
        self._modulev3_extra_input_types = []
        self._ep_batch_manager = None

        ep_size = self.pipeline_config.runtime.ep_size
        if ep_size <= 1:
            return

        n_devices = len(self.devices)
        if ep_size % n_devices != 0:
            raise ValueError(
                f"ep_size={ep_size} must be divisible by the number of GPUs on"
                f" this node ({n_devices}); for single-node set"
                f" ep_size={n_devices}."
            )
        llm = model_config.llm_config
        # Dispatch FP8/NVFP4 tokens across ranks when the checkpoint is
        # quantized; otherwise dispatch bf16. Mirrors deepseekV3_modulev3.
        quant_config = llm.quant_config
        quantized_dispatch = quant_config is not None and (
            self.dtype.is_float8() or quant_config.is_nvfp4
        )
        fused_shared_expert = llm.n_shared_experts == 1
        if quant_config is not None:
            fused_shared_expert = fused_shared_expert and (
                quant_config.shared_experts_use_quant(
                    routed_weight_dtype(quant_config)
                )
            )
        ep_config = EPConfig(
            dispatch_dtype=(
                self.dtype if quantized_dispatch else DType.bfloat16
            ),
            dispatch_quant_config=(
                quant_config if quantized_dispatch else None
            ),
            combine_dtype=DType.bfloat16,
            hidden_size=llm.hidden_size,
            top_k=llm.num_experts_per_tok,
            n_experts=llm.n_routed_experts,
            max_tokens_per_rank=calculate_ep_max_tokens_per_rank(
                max_batch_input_tokens=self.pipeline_config.runtime.max_batch_input_tokens,
                ep_size=ep_size,
                data_parallel_degree=self.pipeline_config.model.data_parallel_degree,
                use_allreduce=self.pipeline_config.runtime.ep_use_allreduce,
            ),
            n_gpus_per_node=n_devices,
            n_nodes=ep_size // n_devices,
            fused_shared_expert=fused_shared_expert,
            use_allreduce=self.pipeline_config.runtime.ep_use_allreduce,
        )
        llm.ep_config = ep_config
        self._ep_batch_manager = EPBatchManager(ep_config)
        self._modulev3_extra_input_types = list(
            self._ep_batch_manager.input_types()
        )
        if not is_virtual_device_mode():
            self.ep_comm_initializer = EPCommInitializer(ep_config)
            self.ep_comm_initializer.ep_init(self.session)
            ep_config.node_id = self.ep_comm_initializer.config.node_id

    def _compile_vision_model(
        self,
        model_config: KimiK2_5Config,
        state_dict: dict[str, WeightData],
    ) -> Callable[..., Any]:
        device_ref = DeviceRef.from_device(self.devices[0])
        vision_config = model_config.vision_config

        with F.lazy():
            vision_nn = Transformer(vision_config)
            vision_nn.to(self.devices[0])

        pixel_values_type = TensorType(
            vision_config.dtype,
            shape=[
                "n_patches",
                vision_config.in_channels,
                vision_config.patch_size,
                vision_config.patch_size,
            ],
            device=device_ref,
        )
        grid_thws_type = TensorType(
            DType.int64, shape=["n_images", 3], device=device_ref
        )
        cu_seqlens_type = TensorType(
            DType.uint32, shape=["n_seqlens"], device=device_ref
        )
        max_seqlen_type = TensorType(
            DType.uint32, shape=[1], device=DeviceRef.CPU()
        )
        position_ids_type = TensorType(
            DType.int64, shape=["n_patches"], device=device_ref
        )

        return vision_nn.compile(
            pixel_values_type,
            grid_thws_type,
            cu_seqlens_type,
            max_seqlen_type,
            position_ids_type,
            weights=state_dict,
        )

    def _compile_language_model(
        self,
        model_config: KimiK2_5Config,
        state_dict: dict[str, WeightData],
    ) -> Callable[..., Any]:
        llm = model_config.llm_config
        assert llm.mesh is not None

        with F.lazy(), default_dtype(self._module_default_dtype(model_config)):
            language_nn = KimiK2_5MoEDecoder(
                llm, self.kv_params, self._ep_batch_manager
            )
            language_nn.to(llm.mesh)

        # Base ModuleV3 ragged inputs: (tokens, return_n_logits,
        # input_row_offsets, *kv). Splice the two multimodal tensors in after
        # input_row_offsets and append the EP buffer types.
        base_inputs = list(
            modulev3_ragged_kv_symbolic_inputs(
                kv_params=self.kv_params,
                device_refs=self.device_refs,
            )
        )
        tokens_t, return_n_logits_t, input_row_offsets_t, *kv_types = (
            base_inputs
        )
        image_embeddings_type = DistributedTensorType(
            DType.bfloat16,
            shape=["vision_merged_seq_len", llm.hidden_size],
            mesh=llm.mesh,
            placements=(Replicated(),),
        )
        image_token_indices_type = DistributedTensorType(
            DType.int32,
            shape=["total_image_tokens"],
            mesh=llm.mesh,
            placements=(Replicated(),),
        )

        # Host per-replica KV context lengths, and (under data parallelism) the
        # CPU split boundaries + int64 row offsets, in the order the decoder's
        # forward peels them off its variadic args. Mirrors the shared
        # DeepseekV3 ModuleV3 DP ABI.
        dp_degree = self.pipeline_config.model.data_parallel_degree
        dp_types: list[TensorType] = [
            TensorType(DType.int32, shape=[1], device=DeviceRef.CPU())
            for _ in range(max(1, dp_degree))
        ]
        if dp_degree > 1:
            dp_types.append(
                TensorType(
                    DType.int64,
                    shape=[dp_degree + 1],
                    device=DeviceRef.CPU(),
                )
            )
            dp_types.append(
                TensorType(
                    DType.int64,
                    shape=["input_row_offsets_len"],
                    device=DeviceRef.CPU(),
                )
            )
        language_input_types = (
            tokens_t,
            return_n_logits_t,
            input_row_offsets_t,
            image_embeddings_type,
            image_token_indices_type,
            *dp_types,
            *kv_types,
            *self._modulev3_extra_input_types,
        )

        return language_nn.compile(
            *language_input_types,
            weights=state_dict,
        )

    def pack_vision_inputs(
        self,
        selection: Sequence[
            tuple[KimiK2_5TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
    ) -> None:
        """Kimi packs inline in :meth:`vision_execute` (chunked encode)."""
        return

    def vision_execute(
        self,
        selection: Sequence[
            tuple[KimiK2_5TextAndVisionContext, Sequence[ImageMetadata]]
        ],
        devices: list[Device],
        packed: None,
    ) -> VisionEncodeResult:
        """Run the chunked vision encoder over the cache-selected images.

        The chunked encode (packing + per-chunk graph runs + device-0-then-
        broadcast) stays encapsulated in the batch processor; the cache only
        ever sees the per-image-ordered output.
        """
        assert isinstance(self._batch_processor, KimiK2_5BatchProcessor)
        embeddings, token_counts = (
            self._batch_processor.encode_uncached_chunked(selection)
        )
        return VisionEncodeResult(
            embeddings=embeddings, per_image_token_counts=token_counts
        )

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Per-device ``[0, hidden]`` image embeddings for non-vision steps.

        Cached: this is hit on every text-only / decode step, so it must not
        allocate per call.
        """
        if not hasattr(self, "_cached_empty_vision_embeddings"):
            assert self.model_config is not None
            hidden_size = self.model_config.llm_config.hidden_size
            host = Buffer.zeros(shape=[0, hidden_size], dtype=DType.bfloat16)
            self._cached_empty_vision_embeddings = [host.to(d) for d in devices]
        return self._cached_empty_vision_embeddings

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, KimiK2_5ModelInputs)
        assert model_inputs.kv_cache_inputs is not None, (
            "KimiK2_5 requires KV cache inputs"
        )

        # The pipeline's vision seam (finalize_vision_inputs) has already set
        # model_inputs.vision_embeddings/vision_scatter_indices; the language
        # graph only scatters the precomputed embeddings.
        raw_outputs = self.language_model.execute_raw(*model_inputs.buffers)

        if len(raw_outputs) == 3:
            return ModelOutputs(
                logits=raw_outputs[1],
                next_token_logits=raw_outputs[0],
                logit_offsets=raw_outputs[2],
            )
        return ModelOutputs(
            logits=raw_outputs[0],
            next_token_logits=raw_outputs[0],
        )
