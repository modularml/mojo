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
"""Shared graph construction for the sparse MLA prefill CPU/GPU split.

The CPU producer (``precompile_mla_sparse``) and the GPU consumer
(``test_mla_sparse_prefill_graph_gpu``) both import this module so they
build the *identical* graph -- the producer compiles it to a MEF with no GPU,
the consumer initializes that MEF and executes it. Keeping the construction in
one place is what guarantees the compiled artifact and the runtime inputs can't
drift apart.

Two attention classes route the prefill branch into ``mla_sm100_prefill_sparse``:
``SparseLatentAttentionWithRopeFp8`` (FP8 attention weights, the ``.fp8.sparse``
op) and ``SparseLatentAttentionWithRope`` (BF16 attention weights, the GLM
``.sparse`` op). ``PrefillSpec.route`` selects between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from _mla_sparse_test_utils import (
    paged_kv_from_flat_graph_inputs,
    random_weights,
)
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops
from max.graph.weights import WeightData
from max.nn.kv_cache import (
    KVCacheQuantizationConfig,
    MLAKVCacheParams,
    MultiKVCacheParams,
)
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.nn.rotary_embedding import (
    DeepseekYarnRopeScalingParams,
    DeepseekYarnRotaryEmbedding,
)
from max.pipelines.architectures.deepseekV3_2.layers.sparse_mla import (
    SparseLatentAttentionWithRope,
    SparseLatentAttentionWithRopeFp8,
)

# Fixed model dimensions shared by every spec (they differ only in route +
# num_heads today; kept as module constants so the producer and consumer agree).
PREFILL_LEN = 32
INDEX_TOPK = 64
HIDDEN_SIZE = 2048
Q_LORA_RANK = 256
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
PAGE_SIZE = 128
BUFFER_SIZE = 4096
TOTAL_NUM_PAGES = 32
MAX_BATCH_SIZE = 32
ROPE_MAX_SEQ_LEN = 2048
INDEX_HEAD_DIM = 128

SparseAttn = SparseLatentAttentionWithRope | SparseLatentAttentionWithRopeFp8


@dataclass(frozen=True)
class PrefillSpec:
    """One compiled-graph parametrization of the sparse MLA prefill path."""

    name: str
    """Stable identifier used for the MEF filename and the test id."""

    route: Literal["fp8", "bf16"]
    """``"fp8"`` for the FP8-attention class (``.fp8.sparse``) or ``"bf16"`` for
    the BF16-attention GLM class (``.sparse``)."""

    num_heads: int
    """Number of attention heads (the DeepSeek/GLM head counts and their
    TP-sharded values)."""

    graph_mode: str = "auto"
    """Attention routing: ``"auto"`` or the disaggregated ``"prefill"``; both
    assemble the same graph for a pure prefill batch."""


PREFILL_SPECS: tuple[PrefillSpec, ...] = (
    # DeepSeek V3.2 (128) + GLM 5.1/5.2 (64) + GLM TP-sharded 64//{2,4,8}.
    PrefillSpec("prefill_fp8_128h", "fp8", 128),
    PrefillSpec("prefill_fp8_64h", "fp8", 64),
    PrefillSpec("prefill_fp8_32h", "fp8", 32),
    PrefillSpec("prefill_fp8_16h", "fp8", 16),
    PrefillSpec("prefill_fp8_8h", "fp8", 8),
    # GLM BF16 attention: 64 unsharded + 8 per device at TP8.
    PrefillSpec("prefill_bf16_64h", "bf16", 64),
    PrefillSpec("prefill_bf16_8h", "bf16", 8),
)

# Disaggregated prefill_only (``graph_mode="prefill"``) variants of the FP8 head
# counts the auto-vs-prefill equivalence test compares. Precompiled alongside
# PREFILL_SPECS; each pairs against the same-head-count FP8 entry above.
PREFILL_ONLY_SPECS: tuple[PrefillSpec, ...] = (
    PrefillSpec("prefill_only_fp8_128h", "fp8", 128, graph_mode="prefill"),
    PrefillSpec("prefill_only_fp8_8h", "fp8", 8, graph_mode="prefill"),
)

SPECS_BY_NAME: dict[str, PrefillSpec] = {
    s.name: s for s in PREFILL_SPECS + PREFILL_ONLY_SPECS
}


def make_quant_config() -> QuantConfig:
    """Builds the block-scaled FP8 quant config (indexer stays FP8 in both routes)."""
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.DYNAMIC,
            dtype=DType.float32,
            block_size=(1, 128),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float32,
            block_size=(128, 128),
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        embedding_output_dtype=None,
        format=QuantFormat.BLOCKSCALED_FP8,
    )


def make_rope(num_heads: int) -> DeepseekYarnRotaryEmbedding:
    """Builds the DeepSeek YaRN rope for ``num_heads`` heads."""
    return DeepseekYarnRotaryEmbedding(
        dim=QK_ROPE_HEAD_DIM,
        n_heads=num_heads,
        theta=10000.0,
        max_seq_len=ROPE_MAX_SEQ_LEN,
        scaling_params=DeepseekYarnRopeScalingParams(
            scaling_factor=40.0,
            original_max_position_embeddings=4096,
            beta_fast=32,
            beta_slow=1,
            mscale=1.0,
            mscale_all_dim=1.0,
        ),
    )


def make_kv_params(
    num_heads: int,
) -> tuple[MLAKVCacheParams, MLAKVCacheParams]:
    """Builds the ``(mla, indexer)`` KV-cache params for ``num_heads`` heads."""
    mla = MLAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        num_layers=1,
        page_size=PAGE_SIZE,
        devices=[DeviceRef.GPU()],
        num_q_heads=num_heads,
    )
    indexer = MLAKVCacheParams(
        dtype=DType.float8_e4m3fn,
        head_dim=INDEX_HEAD_DIM,
        num_layers=1,
        page_size=PAGE_SIZE,
        devices=[DeviceRef.GPU()],
        num_q_heads=num_heads,
        kvcache_quant_config=KVCacheQuantizationConfig(
            scale_dtype=DType.float32,
            quantization_granularity=32,
        ),
    )
    return mla, indexer


def make_multi_kv(num_heads: int) -> MultiKVCacheParams:
    """Builds the combined ``{mla, indexer}`` :class:`MultiKVCacheParams`."""
    mla, indexer = make_kv_params(num_heads)
    return MultiKVCacheParams.from_params({"mla": mla, "indexer": indexer})


def make_attn(
    spec: PrefillSpec, rope: DeepseekYarnRotaryEmbedding
) -> SparseAttn:
    """Builds the sparse attention layer for ``spec`` sharing ``rope``.

    ``rope`` is passed in (not created here) so the same instance backs both the
    layer and the graph's ``freqs_cis`` constant, matching the e2e tests.
    ``spec.graph_mode`` selects the routing.
    """
    mla_kv_params, _ = make_kv_params(spec.num_heads)
    if spec.route == "fp8":
        return SparseLatentAttentionWithRopeFp8(
            rope=rope,
            num_attention_heads=spec.num_heads,
            num_key_value_heads=1,
            hidden_size=HIDDEN_SIZE,
            kv_params=mla_kv_params,
            quant_config=make_quant_config(),
            devices=[DeviceRef.GPU()],
            graph_mode=spec.graph_mode,
            q_lora_rank=Q_LORA_RANK,
            kv_lora_rank=KV_LORA_RANK,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            v_head_dim=V_HEAD_DIM,
            buffer_size=BUFFER_SIZE,
            index_topk=INDEX_TOPK,
        )
    return SparseLatentAttentionWithRope(
        rope=rope,
        num_attention_heads=spec.num_heads,
        num_key_value_heads=1,
        hidden_size=HIDDEN_SIZE,
        kv_params=mla_kv_params,
        dtype=DType.bfloat16,
        indexer_quant_config=make_quant_config(),
        devices=[DeviceRef.GPU()],
        graph_mode=spec.graph_mode,
        q_lora_rank=Q_LORA_RANK,
        kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        buffer_size=BUFFER_SIZE,
        index_topk=INDEX_TOPK,
    )


def weights_for(spec: PrefillSpec) -> dict[str, WeightData]:
    """Random weights for ``spec``'s attention layer, keyed by weight name.

    Weight names and shapes are ``graph_mode``-independent, so one registry
    initializes any graph built from this spec.
    """
    attn = make_attn(spec, make_rope(spec.num_heads))
    _ = attn.state_dict()
    return random_weights(attn)


def build_prefill_graph(spec: PrefillSpec) -> Graph:
    """Builds the sparse MLA prefill graph for ``spec`` (device-independent).

    This is the single construction path the CPU producer compiles and the GPU
    consumer initializes.

    Args:
        spec: The parametrization to build (including its ``graph_mode``).

    Returns:
        The constructed :class:`Graph`, ready to compile.
    """
    rope = make_rope(spec.num_heads)
    mla_kv_params, indexer_kv_params = make_kv_params(spec.num_heads)
    attn = make_attn(spec, rope)
    # Materialize the layer's weights once before graph construction; without
    # this the forward lazily double-registers them ("Weight already exists").
    _ = attn.state_dict()

    multi_kv = MultiKVCacheParams.from_params(
        {"mla": mla_kv_params, "indexer": indexer_kv_params}
    )
    len_mla_kv = len(mla_kv_params.get_symbolic_inputs().inputs[0].flatten())
    len_indexer_kv = len(
        indexer_kv_params.get_symbolic_inputs().inputs[0].flatten()
    )
    kv_sym = list(multi_kv.flattened_kv_inputs())
    hidden_type = TensorType(
        DType.bfloat16, ["total_seq_len", HIDDEN_SIZE], DeviceRef.GPU()
    )
    row_off_type = TensorType(
        DType.uint32, ["row_offsets_len"], DeviceRef.GPU()
    )

    with Graph(
        spec.name, input_types=[hidden_type, row_off_type, *kv_sym]
    ) as g:
        hidden = g.inputs[0].tensor
        input_row_offsets = g.inputs[1].tensor
        mla_in = g.inputs[2 : 2 + len_mla_kv]
        idx_in = g.inputs[2 + len_mla_kv : 2 + len_mla_kv + len_indexer_kv]
        kv_mla = paged_kv_from_flat_graph_inputs(mla_kv_params, list(mla_in))
        kv_idx = paged_kv_from_flat_graph_inputs(
            indexer_kv_params, list(idx_in)
        )
        layer_idx = ops.constant(0, DType.uint32, device=DeviceRef.CPU())
        freqs_cis = ops.cast(rope.freqs_cis, hidden.dtype).to(hidden.device)
        out, _topk = attn(
            layer_idx,
            hidden,
            kv_mla,
            kv_idx,
            freqs_cis,
            input_row_offsets,
            None,
        )
        g.output(out)
    return g
