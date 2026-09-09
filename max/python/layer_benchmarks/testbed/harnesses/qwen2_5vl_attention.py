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

"""Qwen2.5VL decoder attention harness for the model test bed.

Supports benchmarking of the Qwen25VLDecoderAttentionWithRope layer which
requires multi-axis RoPE position_ids (mrope_section).

Static params:
    hidden_size: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    max_seq_len: int
    rope_theta: float (default 1000000.0)
    mrope_section: list[int] (default [16, 24, 24])
    total_num_pages: int (optional, default 8192)

Dynamic params (per shape):
    batch_size: int
    seq_len: int
    ctx_len: int (default 0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, cast

import numpy as np

# torch is a caller-supplied dep, see BUILD.bazel
import torch  # type: ignore[import-not-found]
from max.driver import Accelerator, Buffer, DLPackArray
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kv_cache import MHAKVCacheParams
from max.nn.rotary_embedding import Llama3RotaryEmbedding
from max.pipelines.architectures.qwen2_5vl.nn.decoder import (
    Qwen25VLDecoderAttentionWithRope,
)
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.kv_cache import PagedKVCacheManager
from max.pipelines.modeling.types import RequestID

from testbed.harness import CompiledLayerBundle, LayerTestHarness
from testbed.harnesses.ragged_attention_harness import (
    AttentionDynamicParams,
    AttentionStaticParams,
)
from testbed.registry import register_harness


@dataclass
class Qwen25VLAttentionStaticParams(AttentionStaticParams):
    rope_theta: float = 1000000.0
    mrope_section: list[int] = field(default_factory=lambda: [16, 24, 24])


@register_harness("qwen2_5vl_attention")
class Qwen25VLAttentionHarness(
    LayerTestHarness[
        Qwen25VLAttentionStaticParams,
        AttentionDynamicParams,
        list[TextContext],
    ]
):
    """Harness for Qwen2.5VL decoder attention with mRoPE position_ids.

    Does not support correctness testing (no trivial HF reference for
    the Qwen2.5VL decoder attention with mRoPE).
    """

    @staticmethod
    def static_params_type() -> type:
        return Qwen25VLAttentionStaticParams

    @staticmethod
    def dynamic_params_type() -> type:
        return AttentionDynamicParams

    @property
    def name(self) -> str:
        return "qwen2_5vl_attention"

    def __init__(
        self,
        static_params: Qwen25VLAttentionStaticParams,
        session: Any,
        device: Accelerator,
    ) -> None:
        super().__init__(static_params, session, device)
        self._kv_params = MHAKVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=static_params.n_kv_heads,
            head_dim=static_params.head_dim,
            num_layers=1,
            devices=[DeviceRef.GPU()],
        )

    @cached_property
    def _kv_manager(self) -> PagedKVCacheManager:
        """The paged KV cache, allocated on first use.

        Deferred because it claims device memory, which building the graph does
        not need: the CPU precompile step constructs a harness with no GPU
        attached.
        """
        return PagedKVCacheManager(
            params=self._kv_params,
            total_num_pages=self.static_params.total_num_pages,
            session=self.session,
            max_batch_size=128,
        )

    def build_graph(self) -> tuple[Graph, dict[str, DLPackArray]]:
        p = self.static_params
        hidden_size = p.hidden_size
        n_heads = p.n_heads
        n_kv_heads = p.n_kv_heads
        head_dim = p.head_dim
        max_seq_len = p.max_seq_len
        rope_theta = p.rope_theta
        mrope_section = p.mrope_section
        num_sections = len(mrope_section)

        kv_params = self._kv_params

        rope = Llama3RotaryEmbedding(
            dim=hidden_size,
            n_heads=n_heads,
            theta=rope_theta,
            max_seq_len=max_seq_len,
            head_dim=head_dim,
            interleaved=False,
        )

        layer = Qwen25VLDecoderAttentionWithRope(
            rope=rope,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            hidden_size=hidden_size,
            kv_params=kv_params,
            dtype=DType.bfloat16,
            devices=[DeviceRef.GPU()],
        )

        # Generate random weights.
        q_dim = n_heads * head_dim
        kv_dim = n_kv_heads * head_dim
        std = 0.02
        weights: dict[str, torch.Tensor] = {
            "q_proj.weight": torch.randn(
                q_dim, hidden_size, dtype=torch.bfloat16
            )
            * std,
            "k_proj.weight": torch.randn(
                kv_dim, hidden_size, dtype=torch.bfloat16
            )
            * std,
            "v_proj.weight": torch.randn(
                kv_dim, hidden_size, dtype=torch.bfloat16
            )
            * std,
            "o_proj.weight": torch.randn(
                hidden_size, q_dim, dtype=torch.bfloat16
            )
            * std,
            "q_proj.bias": torch.randn(q_dim, dtype=torch.bfloat16) * std,
            "k_proj.bias": torch.randn(kv_dim, dtype=torch.bfloat16) * std,
            "v_proj.bias": torch.randn(kv_dim, dtype=torch.bfloat16) * std,
        }
        layer.load_state_dict(weights)

        device_ref = DeviceRef.GPU()
        input_type = TensorType(
            dtype=DType.bfloat16,
            shape=["total_seq_len", hidden_size],
            device=device_ref,
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )
        position_ids_type = TensorType(
            DType.uint32,
            [num_sections, "total_seq_len"],
            device=device_ref,
        )
        flattened_kv_types = kv_params.flattened_kv_inputs()

        with Graph(
            "Qwen25VLAttention",
            input_types=(
                input_type,
                input_row_offsets_type,
                position_ids_type,
                *flattened_kv_types,
            ),
        ) as graph:
            inputs, input_row_offsets, position_ids, *kv_cache = graph.inputs
            kv_collection = kv_params.unflatten_kv_inputs(
                iter(kv_cache)
            ).inputs[0]
            layer_idx = ops.constant(0, DType.uint32, device=DeviceRef.CPU())
            freqs_cis = rope.freqs_cis.to(device_ref)
            graph.output(
                layer(
                    layer_idx,
                    inputs.tensor,
                    kv_collection,
                    freqs_cis=freqs_cis,
                    input_row_offsets=input_row_offsets.tensor,
                    position_ids=position_ids.tensor,
                    mrope_section=mrope_section,
                )
            )

        return graph, layer.state_dict()

    def build_and_compile(self) -> CompiledLayerBundle:
        graph, weights_registry = self.build_graph()
        compiled = self.session.load(graph, weights_registry=weights_registry)
        return CompiledLayerBundle(
            compiled_model=compiled,
            device=self.device,
            session=self.session,
        )

    def prepare_inputs(
        self,
        bundle: CompiledLayerBundle,
        dynamic_params: AttentionDynamicParams,
    ) -> tuple[list[Buffer], list[TextContext]]:
        device = bundle.device
        p = self.static_params
        total_len = dynamic_params.ctx_len + dynamic_params.seq_len
        num_sections = len(p.mrope_section)

        batch: list[TextContext] = []
        for _ in range(dynamic_params.batch_size):
            ctx = TextContext(
                request_id=RequestID(),
                max_length=max(total_len, p.max_seq_len),
                tokens=TokenBuffer(np.empty(total_len, dtype=np.int64)),
            )
            self._kv_manager.claim(ctx)
            self._kv_manager.alloc(ctx)
            if dynamic_params.ctx_len > 0:
                ctx.tokens.skip_processing(dynamic_params.ctx_len)
            batch.append(ctx)

        kv_runtime = self._kv_manager.runtime_inputs(
            cast(list[list[TextContext]], [batch])
        )

        total_tokens = dynamic_params.batch_size * dynamic_params.seq_len
        torch_input = torch.randn(
            total_tokens, p.hidden_size, dtype=torch.bfloat16
        )
        input_tensor = Buffer.from_dlpack(torch_input).to(device)
        row_offsets = Buffer.from_numpy(
            np.array(
                [
                    i * dynamic_params.seq_len
                    for i in range(dynamic_params.batch_size + 1)
                ],
                dtype=np.uint32,
            )
        ).to(device)

        # Build position_ids [num_sections, total_tokens].
        position_ids_data = np.tile(
            np.arange(total_tokens, dtype=np.uint32), (num_sections, 1)
        )
        position_ids_buf = Buffer.from_numpy(position_ids_data).to(device)

        execute_args: list[Buffer] = [
            input_tensor,
            row_offsets,
            position_ids_buf,
            *kv_runtime.flatten(),
        ]

        return execute_args, batch

    def cleanup_inputs(
        self,
        bundle: CompiledLayerBundle,
        context: list[TextContext],
    ) -> None:
        for ctx in context:
            self._kv_manager.release(ctx)

    def cuda_graph_eligible(
        self, dynamic_params: AttentionDynamicParams
    ) -> bool:
        return dynamic_params.seq_len == 1

    def torch_reference_layer(self, device: str = "cuda") -> torch.nn.Module:
        raise NotImplementedError(
            "Correctness testing not supported for Qwen2.5VL attention "
            "(no trivial HF reference for mRoPE decoder attention)"
        )

    def prepare_torch_inputs(
        self,
        execute_args: list[Any],
        dynamic_params: AttentionDynamicParams,
        device: str = "cuda",
    ) -> list[Any]:
        raise NotImplementedError(
            "Correctness testing not supported for Qwen2.5VL attention"
        )
