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
"""Implements the Kimi K2.5 language model (ModuleV3).

The Kimi K2.5 text tower is the multi-GPU DeepseekV3 ModuleV3 model with a
multimodal-embedding merge inserted after token embedding. The heavy lifting
(MLA, MoE, EP, YaRN RoPE) is reused from ``deepseekV3_modulev3`` by composing a
:class:`DeepseekV3TextModel`; only the image-embedding merge and the two extra
graph inputs are added here.
"""

from __future__ import annotations

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.sharding import (
    DeviceMapping,
    NoReshard,
    PlacementMapping,
    Replicated,
    Sharded,
    mode,
)
from max.experimental.tensor import Tensor
from max.graph import TensorValue
from max.nn.comm.ep import EPBatchManager, EPCommBuffers
from max.nn.kv_cache import KVCacheInputs, KVCacheParamInterface
from max.pipelines.lib.vlm_utils import F_merge_multimodal_embeddings

from ...deepseekV3_modulev3.deepseekV3 import (
    DeepseekV3TextModel,
    gather_last_tokens,
    split_replicated_batch,
)
from ...deepseekV3_modulev3.model_config import DeepseekV3Config


class KimiK2_5MoEDecoder(Module[..., tuple[Tensor, ...]]):
    """Top-level Kimi K2.5 language model.

    Composes a :class:`DeepseekV3TextModel` and reimplements its decode loop
    with a multimodal-embedding merge inserted after ``embed_tokens``. Mirrors
    ``deepseekV3_modulev3.DeepseekV3`` for the variadic KV/EP unflattening.
    """

    def __init__(
        self,
        config: DeepseekV3Config,
        kv_params: KVCacheParamInterface,
        ep_batch_manager: EPBatchManager | None = None,
    ) -> None:
        super().__init__()
        self.language_model = DeepseekV3TextModel(config, ep_batch_manager)
        self.config = config
        self.kv_params = kv_params
        self.ep_batch_manager = ep_batch_manager

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        image_embeddings: Tensor,
        image_token_indices: Tensor,
        *variadic_args: Tensor,
    ) -> tuple[Tensor, ...]:
        mesh = self.config.mesh
        if mesh is None:
            raise ValueError("Mesh must be defined")

        # Peel the per-replica host batch context lengths and (under data
        # parallelism) the CPU split boundaries off the variadic args, matching
        # ``DeepseekV3.forward`` (the shared tower's top-level wrapper).
        dp_degree = self.config.data_parallel_degree
        batch_context_lengths = variadic_args[:dp_degree]
        variadic_args = variadic_args[dp_degree:]
        data_parallel_splits: Tensor | None = None
        input_row_offsets_i64: Tensor | None = None
        if dp_degree > 1:
            batch_context_length = Tensor.from_shard_values(
                [TensorValue(shard) for shard in batch_context_lengths],
                PlacementMapping(mesh, (Replicated(),) * mesh.ndim),
            )
            data_parallel_splits, input_row_offsets_i64, *rest = variadic_args
            variadic_args = tuple(rest)
        else:
            batch_context_length = batch_context_lengths[0]

        kv_inputs = iter(x._graph_value for x in variadic_args)
        kv_collections = self.kv_params.unflatten_kv_inputs(kv_inputs)
        assert isinstance(kv_collections, KVCacheInputs)

        comm_buffers: EPCommBuffers | None = None
        if self.ep_batch_manager is not None:
            # Any variadic graph values left after the KV cache are the EP
            # communication buffers. Wrap them as an explicit forward argument
            # so they thread through each MoE block's subgraph boundary.
            ep_buffers = list(kv_inputs)
            comm_buffers = self.ep_batch_manager.comm_buffers(ep_buffers)

        kv_collection = PagedCacheValues.from_upstream(
            kv_collections.inputs,
            PlacementMapping(mesh, (Replicated(),) * mesh.ndim),
        )
        return self._run_text_model(
            tokens,
            kv_collection,
            input_row_offsets,
            image_embeddings,
            image_token_indices,
            batch_context_length,
            data_parallel_splits,
            input_row_offsets_i64,
            comm_buffers,
        )

    @mode(NoReshard())
    def _run_text_model(
        self,
        tokens: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
        image_embeddings: Tensor,
        image_token_indices: Tensor,
        batch_context_length: Tensor,
        data_parallel_splits: Tensor | None,
        input_row_offsets_i64: Tensor | None,
        comm_buffers: EPCommBuffers | None,
    ) -> tuple[Tensor, ...]:
        lm = self.language_model
        mesh = lm.mesh
        if mesh is not None:
            tokens = F.distributed_broadcast(tokens, mesh)
            input_row_offsets = F.distributed_broadcast(input_row_offsets, mesh)

        h = lm.embed_tokens(tokens)
        # Scatter the vision embeddings into the (replicated) text stream at the
        # image-token positions, before the data-parallel batch split so the
        # scatter indices still address the full-batch row space.
        h = F_merge_multimodal_embeddings(
            h, image_embeddings, image_token_indices
        )

        # Under data parallelism, shard the replicated batch across the ``dp``
        # mesh axis; each replica then runs attention on its own rows.
        if self.config.data_parallel_degree > 1:
            assert data_parallel_splits is not None
            assert input_row_offsets_i64 is not None
            assert mesh is not None
            batch_placement = tuple(
                Sharded(0) if name == "dp" else Replicated()
                for name in mesh.axis_names
            )
            h, input_row_offsets = split_replicated_batch(
                h,
                input_row_offsets,
                input_row_offsets_i64,
                data_parallel_splits,
                DeviceMapping(mesh, batch_placement),
            )

        freqs_cis = F.cast(lm.rope.freqs_cis, h.dtype)
        if mesh is not None:
            freqs_cis = freqs_cis.to(mesh)
        else:
            freqs_cis = freqs_cis.to(h.device)

        # The MLA prefill plan depends only on the row offsets and KV cache, so
        # it is identical across layers. Compute it once.
        mla_prefill_metadata = None
        first_attn = lm.layers[0].self_attn
        if first_attn.graph_mode in ("prefill", "auto"):
            mla_prefill_metadata = first_attn.create_mla_prefill_metadata(
                input_row_offsets, kv_collection
            )
            # Host-substitute the per-layer D2H buffer_length copies with the
            # CPU batch_context_length so the graph stays capturable.
            mla_prefill_metadata.buffer_lengths = batch_context_length

        for idx, layer in enumerate(lm.layers):
            layer_idx_tensor = F.constant(idx, DType.uint32, device=CPU())
            h = layer(
                layer_idx_tensor,
                h,
                kv_collection,
                input_row_offsets,
                freqs_cis,
                mla_prefill_metadata,
                comm_buffers,
            )

        last_token_h = gather_last_tokens(h, input_row_offsets)
        if self.config.data_parallel_degree > 1:
            last_token_h = F.allgather(last_token_h)
        last_logits = lm.lm_head(lm.norm(last_token_h))
        if mesh is not None:
            last_logits = last_logits.to(mesh.devices[0])
        last_logits = F.cast(last_logits, DType.float32)
        return (last_logits,)
