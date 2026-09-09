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

"""Wan transformer component with block-level compilation for WanExecutor."""

from __future__ import annotations

import logging
import threading
from typing import Any

from max.driver import Buffer, Device, load_devices
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, Module, TensorType
from max.graph.weights import load_weights
from max.pipelines.lib.compiled_component import CompiledComponent
from max.pipelines.lib.config.model_config import (
    _resolve_component_encoding_and_weights,
)
from max.pipelines.lib.model_manifest import ModelManifest
from max.profiler import Tracer, traced

from ..model import (
    BlockLevelModel,
    _compute_wan_rope_cached,
    _remap_state_dict,
)
from ..model_config import WanConfig
from ..wan_transformer import (
    WanTransformerBlock,
    WanTransformerBlockSequence,
    WanTransformerPostProcess,
    WanTransformerPreProcess,
)
from ..weight_adapters import adapt_wan_fp8_weights

logger = logging.getLogger("max.pipelines")


class WanTransformer(CompiledComponent):
    """Wan DiT transformer with block-level compilation.

    Each block is compiled independently so only one block's workspace
    is live at any time, keeping peak VRAM low for 14B-parameter models.

    Supports MoE (dual expert) by compiling both transformers on device.
    """

    # Wan-AI's reference implementation feeds 512-length text-encoder
    # context to cross-attention. See:
    # - diffusers WanPipeline.__call__ default:
    #   https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/wan/pipeline_wan.py#L403
    # - cache-dit hardcoded 512:
    #   https://github.com/vipshop/cache-dit/blob/v1.3.9/src/cache_dit/distributed/transformers/wan.py#L80
    embed_seq_len: int = 512

    # Default resolution for block graph compilation (height, width, frames).
    default_resolution: tuple[int, int, int] = (720, 1280, 81)

    @traced(message="WanTransformer.__init__")
    def __init__(
        self,
        manifest: ModelManifest,
        session: InferenceSession,
    ) -> None:
        super().__init__(manifest, session)

        config_entry = manifest["transformer"]
        config_dict = config_entry.huggingface_config.to_dict()
        resolved_encoding, resolved_weight_path = (
            _resolve_component_encoding_and_weights(config_entry)
        )
        encoding = resolved_encoding or "bfloat16"
        devices = load_devices(config_entry.device_specs)

        self.config = WanConfig.generate(config_dict, encoding, devices)
        self.config.dtype = DType.bfloat16
        self._device = devices[0]

        self._load_lock = threading.Lock()
        self._model: BlockLevelModel | None = None
        self._model_2: BlockLevelModel | None = None
        # Graph IR is built once on the primary compile and reused by the
        # MoE secondary compile so we skip rebuilding 40 layers of
        # transformer-block Python-side IR on the second pass. The pre,
        # blocks, and post graphs share one MLIR module so a single
        # ``session.load_all`` call compiles them together — the graph
        # compiler parallelizes and dedupes across the module's GraphOps.
        # cfg_unpack has no expert-specific weights and is a separate
        # one-graph module loaded once.
        self._graphs_module: Module | None = None
        self._pre_graph: Graph | None = None
        self._blocks_graph: Graph | None = None
        self._post_graph: Graph | None = None
        self._cfg_unpack_model: Any = None

        # Load and remap weights. FP8 (Comfy-Org ``fp8_scaled``) checkpoints
        # use native Wan-AI naming with per-tensor weight/input scales and a
        # dedicated adapter; the bfloat16 path uses the diffusers remap.
        self._fp8 = self.config.quant_config is not None
        paths = config_entry.resolved_weight_paths(resolved_weight_path)
        weights = load_weights(paths)
        if self._fp8:
            state_dict = adapt_wan_fp8_weights(weights)
        else:
            state_dict = _remap_state_dict(weights, target_dtype=DType.bfloat16)

        # MoE: load secondary transformer weights.
        state_dict_2 = self._load_moe_weights(manifest)

        # Compile block-level graphs.
        h, w, nf = self.default_resolution
        seq_len = self._compute_seq_len(h, w, nf)
        self._compile_model(
            state_dict=state_dict,
            seq_text_len=self.embed_seq_len,
            seq_len=seq_len,
        )

        # Compile secondary transformer for MoE.
        if state_dict_2 is not None:
            self._compile_secondary_transformer(state_dict_2, seq_len)

    def _load_moe_weights(
        self, manifest: ModelManifest
    ) -> dict[str, Any] | None:
        """Load optional transformer_2 weights for MoE."""
        if "transformer_2" not in manifest:
            return None

        config_entry_2 = manifest["transformer_2"]
        _, resolved_weight_path_2 = _resolve_component_encoding_and_weights(
            config_entry_2
        )
        paths_2 = config_entry_2.resolved_weight_paths(resolved_weight_path_2)
        weights_2 = load_weights(paths_2)
        if self._fp8:
            state_dict_2 = adapt_wan_fp8_weights(weights_2)
        else:
            state_dict_2 = _remap_state_dict(
                weights_2, target_dtype=DType.bfloat16
            )

        return state_dict_2

    @traced(message="WanTransformer._compile_model")
    def _compile_model(
        self,
        *,
        state_dict: dict[str, Any],
        seq_text_len: int,
        seq_len: int,
        batch_size: int = 1,
    ) -> None:
        """Compile the transformer as separate pre/block/post graphs.

        The four ``Graph`` IR objects (pre, blocks, post, cfg_unpack) are
        built on the first invocation and cached on the instance. The MoE
        secondary compile reuses them so we skip rebuilding 40 layers of
        transformer-block Python-side IR on the second pass.
        """
        with self._load_lock:
            if self._model is not None:
                return

            dim = (
                self.config.num_attention_heads * self.config.attention_head_dim
            )
            dtype = self.config.dtype
            dev = self._device
            dev_ref = DeviceRef.from_device(dev)

            pre_weights, block_weights_list, post_weights = (
                self._split_state_dict(state_dict)
            )

            # Build modules + load weights for THIS expert. Each compile
            # creates fresh module instances so primary and secondary
            # experts hold distinct weight buffers — ``session.load`` takes
            # shared DLPack refs to the tensors in ``weights_registry``.
            with Tracer("dit_build_modules"):
                pre_module = WanTransformerPreProcess(
                    self.config, dtype=dtype, device=dev_ref
                )
                pre_module.load_state_dict(
                    pre_weights, weight_alignment=1, strict=True
                )

                all_blocks = [
                    WanTransformerBlock(
                        dim=dim,
                        ffn_dim=self.config.ffn_dim,
                        num_heads=self.config.num_attention_heads,
                        head_dim=self.config.attention_head_dim,
                        text_dim=dim,
                        cross_attn_norm=self.config.cross_attn_norm,
                        eps=self.config.eps,
                        added_kv_proj_dim=self.config.added_kv_proj_dim,
                        dtype=dtype,
                        device=dev_ref,
                        quant_config=self.config.quant_config,
                    )
                    for _ in range(self.config.num_layers)
                ]
                block_sequence = WanTransformerBlockSequence(all_blocks)
                # Load per-block weights with LayerList prefix.
                combined_block_weights: dict[str, Any] = {}
                for i, block_weights in enumerate(block_weights_list):
                    for k, v in block_weights.items():
                        combined_block_weights[f"blocks.{i}.{k}"] = v
                block_sequence.load_state_dict(
                    combined_block_weights, weight_alignment=1, strict=True
                )

                post_module = WanTransformerPostProcess(
                    self.config, dtype=dtype, device=dev_ref
                )
                post_module.load_state_dict(
                    post_weights, weight_alignment=1, strict=True
                )

            # Build graphs only on the first compile. The pre, blocks,
            # and post graphs go into one shared MLIR module so the
            # follow-up ``session.load_all`` compiles all three together —
            # the graph compiler parallelizes across the module's GraphOps
            # and dedupes shared IR. Reusing the Graph IR across the MoE
            # primary/secondary compiles also skips the Python-side
            # traversal of the 40-layer combined block graph.
            if self._pre_graph is None:
                self._graphs_module = Module()
                # Pre-processor graph. Latents and timestep are pinned to
                # B=1 (the executor enforces single-prompt batches) so
                # that the pre module can ``ops.broadcast_to`` them up to
                # the text embedding's symbolic ``"batch"`` axis. This
                # serves:
                #   * non-CFG (text emb is also B=1; broadcast is identity)
                #   * batched CFG (text emb is B=2; broadcast expands
                #     without a separate pack graph).
                # I2V models concatenate a VAE-encoded conditioning latent
                # (in_channels - out_channels channels) onto the noise latent
                # inside the pre module, so the noise input carries
                # out_channels and a 4th i2v_condition input supplies the
                # rest. T2V (in_channels == out_channels) has no condition.
                cond_channels = (
                    self.config.in_channels - self.config.out_channels
                )
                latent_in_channels = (
                    self.config.out_channels
                    if cond_channels > 0
                    else self.config.in_channels
                )
                pre_input_types = [
                    TensorType(
                        DType.float32,
                        [
                            1,
                            latent_in_channels,
                            "frames",
                            "height",
                            "width",
                        ],
                        device=dev,
                    ),
                    TensorType(DType.float32, [1], device=dev),
                    TensorType(
                        dtype,
                        ["batch", seq_text_len, self.config.text_dim],
                        device=dev,
                    ),
                ]
                if cond_channels > 0:
                    # The condition is concatenated onto the noise latent
                    # *after* it is broadcast to the text embedding's batch
                    # axis, so the condition shares the same symbolic
                    # ``"batch"`` dim (1 for the sequential I2V path).
                    pre_input_types.append(
                        TensorType(
                            dtype,
                            [
                                "batch",
                                cond_channels,
                                "frames",
                                "height",
                                "width",
                            ],
                            device=dev,
                        )
                    )
                with Graph(
                    "wan_pre",
                    input_types=pre_input_types,
                    module=self._graphs_module,
                ) as pre_graph:
                    outs = pre_module(*(v.tensor for v in pre_graph.inputs))
                    pre_graph.output(*outs)
                self._pre_graph = pre_graph

                # Combined blocks graph: all blocks in a single Model so
                # the runtime allocates one shared workspace.
                #
                # Batch dim is symbolic so the same compiled graph serves
                # both the B=1 path (no CFG / i2v fallback) and the B=2
                # path used by batched CFG (``call_cfg_batched``).
                block_seq_len_dim: str = "seq_len"
                block_batch_dim: str = "batch"
                block_input_types = [
                    TensorType(
                        dtype,
                        [block_batch_dim, block_seq_len_dim, dim],
                        device=dev,
                    ),
                    TensorType(
                        dtype,
                        [block_batch_dim, seq_text_len, dim],
                        device=dev,
                    ),
                    TensorType(dtype, [block_batch_dim, 6, dim], device=dev),
                    TensorType(
                        DType.float32,
                        [block_seq_len_dim, self.config.attention_head_dim],
                        device=dev,
                    ),
                    TensorType(
                        DType.float32,
                        [block_seq_len_dim, self.config.attention_head_dim],
                        device=dev,
                    ),
                ]
                with Graph(
                    "wan_blocks_combined",
                    input_types=block_input_types,
                    module=self._graphs_module,
                ) as blocks_graph:
                    block_out = block_sequence(
                        *(v.tensor for v in blocks_graph.inputs)
                    )
                    blocks_graph.output(block_out)
                self._blocks_graph = blocks_graph
                logger.info(
                    "Built combined block graph IR (batch=%d, "
                    "seq_len=symbolic default=%d, seq_text=%d, %d layers)",
                    batch_size,
                    seq_len,
                    seq_text_len,
                    self.config.num_layers,
                )

                # Post-processor graph.
                post_input_types = [
                    TensorType(dtype, ["batch", "seq_len", dim], device=dev),
                    TensorType(dtype, ["batch", dim], device=dev),
                    TensorType(DType.int8, ["ppf", "pph", "ppw"], device=dev),
                ]
                with Graph(
                    "wan_post",
                    input_types=post_input_types,
                    module=self._graphs_module,
                ) as post_graph:
                    post_out = post_module(
                        *(v.tensor for v in post_graph.inputs)
                    )
                    post_graph.output(post_out)
                self._post_graph = post_graph

                # CFG unpack: split the B=2 noise prediction back into the
                # cond / uncond halves expected by the guidance scheduler.
                # The matching pack step (concat + tile of
                # latents/timestep) is folded into
                # ``WanTransformerPreProcess.__call__`` so we only need
                # this one helper graph for batched CFG. It has no
                # expert-specific weights, so the loaded Model is shared
                # across primary and secondary compiles.
                cfg_unpack_input_types = [
                    TensorType(
                        dtype,
                        [
                            "batch",
                            "channels",
                            "frames",
                            "height",
                            "width",
                        ],
                        device=dev,
                    ),
                ]
                with Graph(
                    "wan_cfg_unpack", input_types=cfg_unpack_input_types
                ) as cfg_unpack_graph:
                    noise_b2 = next(v.tensor for v in cfg_unpack_graph.inputs)
                    noise_cond = noise_b2[0:1, :, :, :, :]
                    noise_uncond = noise_b2[1:2, :, :, :, :]
                    cfg_unpack_graph.output(noise_cond, noise_uncond)
                with Tracer("dit_compile_cfg_unpack"):
                    self._cfg_unpack_model = self._session.load(
                        cfg_unpack_graph
                    )

            # Load all three weight-bearing Models in one parallel
            # compile pass. The combined registry covers every GraphOp in
            # the shared module; ``load_all`` returns a dict keyed by
            # each graph's sym_name.
            combined_registry: dict[str, Any] = {
                **pre_module.state_dict(),
                **block_sequence.state_dict(),
                **post_module.state_dict(),
            }
            assert self._graphs_module is not None
            with Tracer("dit_compile_load_all"):
                models = self._session.load_all(
                    self._graphs_module, weights_registry=combined_registry
                )
                assert self._pre_graph is not None
                assert self._blocks_graph is not None
                assert self._post_graph is not None
                pre_model = models[self._pre_graph.name]
                combined_blocks_model = models[self._blocks_graph.name]
                post_model = models[self._post_graph.name]
            self._model = BlockLevelModel(
                pre_model,
                post_model,
                combined_blocks=combined_blocks_model,
            )

    @traced(message="WanTransformer.__call__")
    def __call__(
        self,
        hidden_states: Buffer,
        timestep: Buffer,
        encoder_hidden_states: Buffer,
        rope_cos: Buffer,
        rope_sin: Buffer,
        spatial_shape: Buffer,
        i2v_condition: Buffer | None = None,
    ) -> Buffer:
        """Execute the block-level transformer forward pass.

        Args:
            hidden_states: Latents in model dtype,
                shape ``(B, C, T, H, W)``.
            timestep: Timestep, shape ``(B,)`` float32.
            encoder_hidden_states: Text embeddings,
                shape ``(B, seq_text, dim)``.
            rope_cos: RoPE cosine, shape ``(seq_len, head_dim)`` float32.
            rope_sin: RoPE sine, shape ``(seq_len, head_dim)`` float32.
            spatial_shape: Shape carrier ``(ppf, pph, ppw)`` int8.
            i2v_condition: Optional I2V condition tensor for image-to-video.

        Returns:
            Noise prediction, shape ``(B, C, T, H, W)`` in model dtype.
        """
        if self._model is None:
            raise RuntimeError(
                "WanTransformer model not compiled. "
                "This should not happen — __init__ compiles it."
            )
        return self._model(
            hidden_states,
            timestep,
            encoder_hidden_states,
            rope_cos,
            rope_sin,
            spatial_shape,
            i2v_condition=i2v_condition,
        )

    def call_secondary(
        self,
        hidden_states: Buffer,
        timestep: Buffer,
        encoder_hidden_states: Buffer,
        rope_cos: Buffer,
        rope_sin: Buffer,
        spatial_shape: Buffer,
        i2v_condition: Buffer | None = None,
    ) -> Buffer:
        """Execute the secondary (low-noise) transformer for dual-load MoE."""
        if self._model_2 is None:
            raise RuntimeError(
                "Secondary transformer not compiled for dual-load MoE."
            )
        return self._model_2(
            hidden_states,
            timestep,
            encoder_hidden_states,
            rope_cos,
            rope_sin,
            spatial_shape,
            i2v_condition=i2v_condition,
        )

    @traced(message="WanTransformer.call_cfg_batched")
    def call_cfg_batched(
        self,
        hidden_states: Buffer,
        timestep: Buffer,
        encoder_hidden_states_b2: Buffer,
        rope_cos: Buffer,
        rope_sin: Buffer,
        spatial_shape: Buffer,
        *,
        use_secondary_transformer: bool = False,
    ) -> tuple[Buffer, Buffer]:
        """Run conditional and unconditional DiT forwards in a single B=2 pass.

        Replaces two sequential B=1 ``__call__`` invocations (cond + uncond)
        with one B=2 call, halving kernel-launch overhead and improving SM
        occupancy on the per-block GEMMs and attention. The pre graph tiles
        ``hidden_states`` and ``timestep`` (provided at B=1) up to the batch
        dim of ``encoder_hidden_states_b2``; the unpack graph splits the
        B=2 noise output back into the cond / uncond halves.

        Args:
            encoder_hidden_states_b2: Text embedding pre-concatenated as
                ``[cond; uncond]`` along axis 0 — the caller is responsible
                for batching this once per denoising phase.

        Does not currently support I2V (``i2v_condition``) — callers should
        fall back to two ``__call__``s when an I2V condition is present.

        Returns:
            A tuple of B=1 noise predictions ``(cond, uncond)``.
        """
        model = self._model_2 if use_secondary_transformer else self._model
        if model is None:
            raise RuntimeError(
                "Transformer not compiled. "
                f"use_secondary_transformer={use_secondary_transformer}."
            )

        # Pre internally tiles latents/timestep to match text emb's batch.
        noise_b2 = model(
            hidden_states,
            timestep,
            encoder_hidden_states_b2,
            rope_cos,
            rope_sin,
            spatial_shape,
            i2v_condition=None,
        )

        unpack_outs = self._cfg_unpack_model.execute(noise_b2)
        return unpack_outs[0], unpack_outs[1]

    def compute_rope(
        self,
        num_frames: int,
        height: int,
        width: int,
    ) -> tuple[Buffer, Buffer]:
        """Compute 3D RoPE cos/sin tensors and transfer to device."""
        rope_cos_np, rope_sin_np = _compute_wan_rope_cached(
            num_frames,
            height,
            width,
            self.config.patch_size,
            self.config.attention_head_dim,
        )
        return (
            Buffer.from_numpy(rope_cos_np).to(self._device),
            Buffer.from_numpy(rope_sin_np).to(self._device),
        )

    @property
    def has_moe(self) -> bool:
        """Whether this transformer has MoE (dual expert) support."""
        return self._model_2 is not None

    @property
    def device(self) -> Device:
        return self._device

    # -- Internal helpers ---------------------------------------------------

    def _compute_seq_len(self, height: int, width: int, num_frames: int) -> int:
        """Compute the latent sequence length for a given resolution."""
        p_t, p_h, p_w = self.config.patch_size
        # Compute video latent shape.
        vae_scale_t = 4  # Default temporal scale factor
        vae_scale_s = 8  # Default spatial scale factor
        adjusted = max(1, num_frames)
        if adjusted > 1:
            remainder = (adjusted - 1) % vae_scale_t
            if remainder != 0:
                adjusted += vae_scale_t - remainder
        latent_frames = (adjusted - 1) // vae_scale_t + 1
        latent_h = 2 * (height // (vae_scale_s * 2))
        latent_w = 2 * (width // (vae_scale_s * 2))
        return (latent_frames // p_t) * (latent_h // p_h) * (latent_w // p_w)

    def _split_state_dict(
        self, state_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        """Split flat state dict into pre/block/post weight groups."""
        pre_weights: dict[str, Any] = {}
        post_weights: dict[str, Any] = {}
        block_weights_list: list[dict[str, Any]] = [
            {} for _ in range(self.config.num_layers)
        ]

        for key, value in state_dict.items():
            if key.startswith(("patch_embedding.", "condition_embedder.")):
                pre_weights[key] = value
            elif key.startswith("blocks."):
                rest = key[len("blocks.") :]
                dot = rest.index(".")
                block_idx = int(rest[:dot])
                sub_key = rest[dot + 1 :]
                block_weights_list[block_idx][sub_key] = value
            else:
                post_weights[key] = value

        return pre_weights, block_weights_list, post_weights

    @traced(message="WanTransformer._compile_secondary_transformer")
    def _compile_secondary_transformer(
        self, state_dict: dict[str, Any], seq_len: int
    ) -> None:
        """Compile the secondary transformer for MoE.

        Both expert models stay resident on GPU so the executor can
        dispatch to either without hot-swapping weights.
        """
        saved_model = self._model
        self._model = None
        try:
            self._compile_model(
                state_dict=state_dict,
                seq_text_len=self.embed_seq_len,
                seq_len=seq_len,
            )
            self._model_2 = self._model
        finally:
            self._model = saved_model
