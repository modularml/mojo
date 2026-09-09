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
"""Pipeline model for the DiffusionGemmaForBlockDiffusion port.

Builds three graphs in one ``Module`` so their merged weights registry shares
every duplicated FQN: the donor gemma4 vision tower, the causal encoder text
graph (cache prefill/commit), and the bidirectional decoder text graph (one
denoise step per call). The block-diffusion generation loop itself lives in
``pipeline.py``; this class only compiles graphs and exposes
``execute``/``execute_decoder_step`` over them.

The inherited donor ``execute``/``prepare_initial_token_inputs`` drive the
*encoder* graph — its input signature is identical to the donor language
graph (including the empty multimodal-merge inputs), so prompt prefill and
accepted-canvas commits reuse the donor paths unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph, Module, TensorType
from max.graph.weights import WeightData
from max.nn.kv_cache import KVCacheInputs
from max.pipelines.architectures.gemma4.model import Gemma3_MultiModalModel
from max.pipelines.lib import CompilationTimer

from .diffusion_gemma import (
    DiffusionGemmaDecoderTextModel,
    DiffusionGemmaEncoderTextModel,
)
from .model_config import DiffusionGemmaForBlockDiffusionConfig
from .weight_adapters import (
    convert_safetensor_language_state_dict,
    convert_safetensor_vision_state_dict,
)


class DiffusionGemmaForBlockDiffusionModel(Gemma3_MultiModalModel):
    """Compiles the vision/encoder/decoder graphs for block diffusion.

    Differences from the donor pipeline model:

    - ``load_model`` uses this port's weight converters (decoder-canonical
      checkpoint layout) and compiles an extra decoder graph.
    - ``execute_decoder_step`` runs one denoise step: canvas K/V are written
      into the cache slots after each request's committed length, so calling
      it repeatedly without advancing context lengths overwrites the same
      slots (read-only cache semantics from the encoder's perspective).
    """

    decoder_model: Model

    def load_model(self, session: InferenceSession) -> tuple[Model, Model]:
        assert self.max_batch_size, "Expected max_batch_size to be set"

        weights_dict = dict(self.weights.items())
        language_weights = convert_safetensor_language_state_dict(weights_dict)
        vision_weights = convert_safetensor_vision_state_dict(weights_dict)

        raw_state_dict = {k: v.data() for k, v in weights_dict.items()}
        model_config = DiffusionGemmaForBlockDiffusionConfig.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=raw_state_dict,
            return_logits=self.return_logits,
        )
        self.config = model_config

        input_row_offsets_prealloc_host = Buffer.from_numpy(
            np.arange(
                self.max_batch_size + 1,
                dtype=np.uint32,
            )
        )
        self._input_row_offsets_prealloc = [
            input_row_offsets_prealloc_host.to(dev) for dev in self.devices
        ]
        self._execution_input_buffers: dict[Any, Any] = {}
        self._scatter_buffers: dict[Any, Any] = {}

        with CompilationTimer("vision + encoder + decoder model") as timer:
            module = Module()

            vision_graph, vision_state_dict = self._build_vision_graph(
                model_config, vision_weights, module=module
            )
            encoder_graph, encoder_state_dict = self._build_encoder_graph(
                model_config, language_weights, module=module
            )
            decoder_graph, decoder_state_dict = self._build_decoder_graph(
                model_config, language_weights, module=module
            )
            timer.mark_build_complete()

            # Shared FQNs (every transformer weight) resolve to the same
            # backing tensor; the decoder adds only self_conditioning.*.
            combined_weights = {
                **vision_state_dict,
                **encoder_state_dict,
                **decoder_state_dict,
            }
            models = session.load_all(module, weights_registry=combined_weights)
            vision_model = models[vision_graph.name]
            encoder_model = models[encoder_graph.name]
            self.decoder_model = models[decoder_graph.name]

        # The donor base class calls the language graph `language_model` and
        # routes its `execute`/graph-capture paths through it.
        return vision_model, encoder_model

    def _build_encoder_graph(
        self,
        config: DiffusionGemmaForBlockDiffusionConfig,
        state_dict: dict[str, WeightData],
        module: Module | None = None,
    ) -> tuple[Graph, dict[str, Any]]:
        """Donor ``_build_language_graph`` with this port's encoder class."""
        # self_conditioning.* weights exist only in the decoder graph; keep
        # strict loading for everything else.
        encoder_state_dict = {
            k: v
            for k, v in state_dict.items()
            if not k.startswith("self_conditioning.")
        }
        with Graph(
            "diffusiongemma_encoder",
            input_types=self._language_model_input_types(config),
            module=module,
        ) as graph:
            encoder = DiffusionGemmaEncoderTextModel(config)
            encoder.load_state_dict(
                encoder_state_dict,
                weight_alignment=1,
                strict=self._strict_state_dict_loading,
            )

            (tokens, return_n_logits, *variadic_args) = graph.inputs
            n = len(self.devices)
            input_row_offsets = [v.tensor for v in variadic_args[:n]]
            variadic_args = variadic_args[n:]
            image_embeddings = [v.tensor for v in variadic_args[:n]]
            variadic_args = variadic_args[n:]
            image_token_indices = [v.tensor for v in variadic_args[:n]]
            variadic_args = variadic_args[n:]
            signal_buffers = [v.buffer for v in variadic_args[:n]]
            variadic_args = variadic_args[n:]
            kv_cache = self._unflatten_kv_inputs(variadic_args)
            kv_local, kv_global = (
                kv_cache[: len(kv_cache) // 2],
                kv_cache[len(kv_cache) // 2 :],
            )

            outputs = encoder(
                tokens=tokens.tensor,
                signal_buffers=signal_buffers,
                sliding_kv_collections=kv_local,
                global_kv_collections=kv_global,
                return_n_logits=return_n_logits.tensor,
                input_row_offsets=input_row_offsets,
                image_embeddings=image_embeddings,
                image_token_indices=image_token_indices,
            )
            graph.output(*outputs)
        return graph, encoder.state_dict()

    def _decoder_input_types(
        self, config: DiffusionGemmaForBlockDiffusionConfig
    ) -> list[TensorType | BufferType]:
        device_ref = DeviceRef.from_device(self.devices[0])
        vocab_size = config.text_config.vocab_size

        canvas_tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_types = [
            TensorType(
                DType.uint32,
                shape=["input_row_offsets_len"],
                device=DeviceRef.from_device(dev),
            )
            for dev in self.devices
        ]
        sc_logits_type = TensorType(
            DType.bfloat16,
            shape=["total_seq_len", vocab_size],
            device=device_ref,
        )
        sc_enabled_type = TensorType(
            DType.float32, shape=[1], device=device_ref
        )
        temperature_type = TensorType(
            DType.float32, shape=[1], device=device_ref
        )

        from max.nn.comm.allreduce import Signals

        signals = Signals(
            devices=(DeviceRef(d.label, d.id) for d in self.devices)
        )

        return [
            canvas_tokens_type,
            *input_row_offsets_types,
            sc_logits_type,
            sc_enabled_type,
            temperature_type,
            *signals.input_types(),
            *self.kv_params.get_symbolic_inputs().flatten(),
        ]

    def _build_decoder_graph(
        self,
        config: DiffusionGemmaForBlockDiffusionConfig,
        state_dict: dict[str, WeightData],
        module: Module | None = None,
    ) -> tuple[Graph, dict[str, Any]]:
        with Graph(
            "diffusiongemma_decoder",
            input_types=self._decoder_input_types(config),
            module=module,
        ) as graph:
            decoder = DiffusionGemmaDecoderTextModel(config)
            decoder.load_state_dict(
                state_dict,
                weight_alignment=1,
                strict=self._strict_state_dict_loading,
            )

            (canvas_tokens, *variadic_args) = graph.inputs
            n = len(self.devices)
            input_row_offsets = [v.tensor for v in variadic_args[:n]]
            variadic_args = variadic_args[n:]
            sc_logits = variadic_args[0].tensor
            sc_enabled = variadic_args[1].tensor
            temperature = variadic_args[2].tensor
            variadic_args = variadic_args[3:]
            signal_buffers = [v.buffer for v in variadic_args[:n]]
            variadic_args = variadic_args[n:]
            kv_cache = self._unflatten_kv_inputs(variadic_args)
            kv_local, kv_global = (
                kv_cache[: len(kv_cache) // 2],
                kv_cache[len(kv_cache) // 2 :],
            )

            outputs = decoder(
                canvas_tokens=canvas_tokens.tensor,
                signal_buffers=signal_buffers,
                sliding_kv_collections=kv_local,
                global_kv_collections=kv_global,
                input_row_offsets=input_row_offsets,
                sc_logits=sc_logits,
                sc_enabled=sc_enabled,
                temperature=temperature,
            )
            graph.output(*outputs)
        return graph, decoder.state_dict()

    def execute_decoder_step(
        self,
        canvas_tokens: Buffer,
        input_row_offsets: Buffer,
        sc_logits: Buffer,
        sc_enabled: Buffer,
        temperature: Buffer,
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer],
    ) -> tuple[Buffer, Buffer, Buffer, Buffer, Buffer]:
        """Runs one denoise step.

        Returns ``(sc_logits_out, argmax, topk_probs, topk_idx, entropy)``;
        ``sc_logits_out`` is device-resident bf16 and feeds the next step's
        ``sc_logits``; the [N, 64] top-k pair feeds host-side categorical
        sampling.
        """
        outputs = self.decoder_model.execute(
            canvas_tokens,
            input_row_offsets,
            sc_logits,
            sc_enabled,
            temperature,
            *self.signal_buffers,
            *kv_cache_inputs.flatten(),
        )
        sc_out, argmax, topk_probs, topk_idx, entropy = outputs[:5]
        # Populated only when the graph was built with DG_DUMP_HIDDEN=1
        # (debug-model per-layer comparator taps).
        self.last_decoder_taps = outputs[5:]
        assert isinstance(sc_out, Buffer)
        assert isinstance(argmax, Buffer)
        assert isinstance(topk_probs, Buffer)
        assert isinstance(topk_idx, Buffer)
        assert isinstance(entropy, Buffer)
        return sc_out, argmax, topk_probs, topk_idx, entropy
