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
"""Tests for GLM-5.1 (GlmMoeDsa) DSA indexer vs MAX Indexer."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer, accelerator_api
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kv_cache import (
    KVCacheQuantizationConfig,
    MHAKVCacheParams,
    PagedCacheValues,
)
from max.nn.rotary_embedding import RotaryEmbedding
from max.pipelines.architectures.deepseekV3_2.layers import Indexer
from max.pipelines.kv_cache import PagedKVCacheManager
from test_common.context_utils import create_text_context
from test_common.graph_utils import is_h100_h200
from torch.utils.dlpack import from_dlpack
from torch_reference.configuration_glm import GlmMoeDsaConfig
from torch_reference.modeling_glm import (
    GlmMoeDsaIndexer,
    GlmMoeDsaRotaryEmbedding,
)

pytestmark = pytest.mark.skipif(
    not __import__(
        "torch_reference.modeling_glm", fromlist=["TORCH_REFERENCE_READY"]
    ).TORCH_REFERENCE_READY,
    reason="GLM torch reference not installed",
)

# Small dimensions; rope_head_dim must equal head_dim/2; head_dim must be
# divisible by act_quant block_size (128).
dim: int = 128
index_n_heads: int = 32
index_head_dim: int = 128
qk_rope_head_dim: int = 64
index_topk: int = 4
q_lora_rank: int = 128
batch_size: int = 2
seq_len: int = 8
page_size: int = 128

seed: int = 42


def _indexer_config(rope_dim: int = qk_rope_head_dim) -> GlmMoeDsaConfig:
    return GlmMoeDsaConfig(
        hidden_size=dim,
        q_lora_rank=q_lora_rank,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        qk_rope_head_dim=rope_dim,
        max_position_embeddings=1024,
        num_attention_heads=8,
        num_hidden_layers=1,
    )


@pytest.fixture
def x() -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(
        batch_size, seq_len, dim, dtype=torch.bfloat16
    ) / math.sqrt(dim)


@pytest.fixture
def qr() -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(
        batch_size, seq_len, q_lora_rank, dtype=torch.bfloat16
    ) / math.sqrt(q_lora_rank)


@pytest.fixture
def input_row_offsets() -> torch.Tensor:
    input_row_offsets = torch.zeros(batch_size + 1, dtype=torch.uint32)
    for i in range(batch_size):
        input_row_offsets[i] = i * seq_len
    input_row_offsets[batch_size] = batch_size * seq_len
    return input_row_offsets


@pytest.fixture
def state_dict() -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {
        "wq_b.weight": (
            torch.randn(index_n_heads * index_head_dim, q_lora_rank)
            / math.sqrt(q_lora_rank)
        ).to(torch.float8_e4m3fn),
        "wq_b.weight_scale": torch.ones(index_n_heads, 1, dtype=torch.float32),
        "wk.weight": (torch.randn(index_head_dim, dim) / math.sqrt(dim)).to(
            torch.float8_e4m3fn
        ),
        "wk.weight_scale": torch.tensor([[1.0]], dtype=torch.float32),
        "k_norm.weight": torch.randn(index_head_dim, dtype=torch.float32)
        / math.sqrt(index_head_dim),
        "k_norm.bias": torch.zeros(index_head_dim, dtype=torch.float32),
        "weights_proj.weight": (
            torch.randn(index_n_heads, dim) / math.sqrt(dim)
        ).to(torch.bfloat16),
    }


def run_torch_indexer(
    x: torch.Tensor,
    qr: torch.Tensor,
    state_dict: dict[str, torch.Tensor],
    mask: bool = False,
    device: str = "cuda",
    rope_dim: int = qk_rope_head_dim,
) -> torch.Tensor:
    """Run :class:`GlmMoeDsaIndexer` with fixture inputs and weights."""
    config = _indexer_config(rope_dim)
    indexer = GlmMoeDsaIndexer(config, layer_idx=0)
    indexer.load_state_dict(
        {k: v for k, v in state_dict.items() if "scale" not in k}, strict=True
    )
    indexer = indexer.to(device=device, dtype=torch.bfloat16).eval()

    batch_size_local, seq_len_local, _ = x.shape
    position_ids = torch.arange(seq_len_local, device=device).unsqueeze(0)
    position_ids = position_ids.expand(batch_size_local, -1)

    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None
    if rope_dim:
        rotary_emb = GlmMoeDsaRotaryEmbedding(config).to(device)
        probe = torch.zeros(
            batch_size_local,
            seq_len_local,
            config.index_n_heads,
            config.qk_rope_head_dim,
            dtype=x.dtype,
            device=device,
        )
        position_embeddings = rotary_emb(probe, position_ids)

    attention_mask = None
    if mask:
        attention_mask = torch.full(
            (batch_size_local, seq_len_local, seq_len_local),
            float("-inf"),
            device=device,
        ).triu_(diagonal=1)

    with torch.inference_mode():
        return indexer(
            x.to(device),
            qr.to(device),
            position_embeddings,
            attention_mask,
            use_cache=False,
        )


def run_max_indexer(
    x: torch.Tensor,
    qr: torch.Tensor,
    input_row_offsets: torch.Tensor,
    state_dict: dict[str, torch.Tensor],
    mask_variant: MHAMaskVariant = MHAMaskVariant.NULL_MASK,
    rope_dim: int = qk_rope_head_dim,
) -> torch.Tensor:
    device = Accelerator()
    session = InferenceSession(devices=[device])

    input_spec = InputScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        origin=ScaleOrigin.DYNAMIC,
        dtype=DType.float32,
        block_size=(1, 128),
    )
    weight_spec = WeightScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        dtype=DType.float32,
        block_size=(128, 128),
    )
    quant_config = QuantConfig(
        input_scale=input_spec,
        weight_scale=weight_spec,
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        embedding_output_dtype=None,
        format=QuantFormat.BLOCKSCALED_FP8,
    )

    kv_params = MHAKVCacheParams(
        dtype=DType.float8_e4m3fn,
        n_kv_heads=1,
        head_dim=index_head_dim,
        num_layers=1,
        page_size=page_size,
        devices=[DeviceRef.GPU()],
        kvcache_quant_config=KVCacheQuantizationConfig(
            scale_dtype=DType.float32,
            quantization_granularity=128,
        ),
    )

    kv_manager = PagedKVCacheManager(
        params=kv_params,
        total_num_pages=8,
        session=session,
        max_batch_size=128,
    )

    # A NoPE indexer never reads ``freqs_cis``, so the table stays full width
    # and simply goes unused.
    rope = RotaryEmbedding(
        dim=qk_rope_head_dim,
        n_heads=index_n_heads,
        theta=10000.0,
        max_seq_len=1024,
        head_dim=qk_rope_head_dim,
        interleaved=False,
    )

    indexer = Indexer(
        dim=dim,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        qk_rope_head_dim=rope_dim,
        index_topk=index_topk,
        q_lora_rank=q_lora_rank,
        devices=[DeviceRef.GPU()],
        quant_config=quant_config,
    )

    indexer.load_state_dict(state_dict, strict=True)

    x_type = TensorType(DType.bfloat16, ["total_seq_len", dim], DeviceRef.GPU())
    qr_type = TensorType(
        DType.bfloat16, ["total_seq_len", q_lora_rank], DeviceRef.GPU()
    )
    input_row_offsets_type = TensorType(
        DType.uint32, ["batch_size_plus_1"], DeviceRef.GPU()
    )

    with Graph(
        "GlmIndexerTest",
        input_types=(
            x_type,
            qr_type,
            input_row_offsets_type,
            *kv_params.flattened_kv_inputs(),
        ),
    ) as graph:
        x_in = graph.inputs[0].tensor
        qr_in = graph.inputs[1].tensor
        input_row_offsets_in = graph.inputs[2].tensor

        indexer_k_collection = PagedCacheValues(
            kv_blocks=graph.inputs[3].buffer,
            cache_lengths=graph.inputs[4].tensor,
            lookup_table=graph.inputs[5].tensor,
            max_prompt_length=graph.inputs[6].tensor,
            max_cache_length=graph.inputs[7].tensor,
            kv_scales=graph.inputs[8].buffer,
        )

        layer_idx = ops.constant(0, DType.uint32, device=DeviceRef.CPU())

        result = indexer(
            x_in,
            qr_in,
            rope.freqs_cis[:seq_len],
            input_row_offsets_in,
            indexer_k_collection,
            layer_idx,
            mask_variant,
        )

        graph.output(result)

    compiled = session.load(graph, weights_registry=indexer.state_dict())

    prompt_lens = [seq_len] * batch_size
    batch_contexts = []
    for prompt_len in prompt_lens:
        context = create_text_context(np.empty(prompt_len, dtype=np.int64))
        kv_manager.claim(context)
        kv_manager.alloc(context)
        batch_contexts.append(context)

    kv_inputs = kv_manager.runtime_inputs_for_leaf([batch_contexts]).inputs[0]

    x_flat = x.view(-1, dim)
    qr_flat = qr.view(-1, q_lora_rank)

    x_device = (
        Buffer.from_numpy(x_flat.view(torch.float16).numpy())
        .view(DType.bfloat16)
        .to(device)
    )
    qr_device = (
        Buffer.from_numpy(qr_flat.view(torch.float16).numpy())
        .view(DType.bfloat16)
        .to(device)
    )
    input_row_offsets_device = Buffer.from_numpy(input_row_offsets.numpy()).to(
        device
    )

    assert kv_inputs.kv_scales is not None

    output_result: list[Buffer] = compiled.execute(
        x_device,
        qr_device,
        input_row_offsets_device,
        *kv_inputs.flatten(),
    )

    output_tensor = from_dlpack(output_result[0])
    return output_tensor.cpu()


def _sorted_rows(indices: torch.Tensor) -> torch.Tensor:
    """Selected indices sorted within each row, as int32 on the CPU."""
    rows = indices.reshape(-1, index_topk).to("cpu").to(torch.int32)
    return rows.sort(dim=-1).values


@pytest.mark.skipif(
    accelerator_api() == "hip",
    reason="Memory access fault by GPU node-2 (Agent handle: 0x49c8e0a0) on address 0x10e2bfcf8000. Reason: Unknown.",
)
@pytest.mark.skipif(
    is_h100_h200(),
    reason="CUDA call failed: CUDA_ERROR_ILLEGAL_ADDRESS (an illegal memory access was encountered)",
)
@pytest.mark.parametrize(
    "rope_dim", [qk_rope_head_dim, 0], ids=["rope", "nope"]
)
def test_indexer_no_mask(
    x: torch.Tensor,
    qr: torch.Tensor,
    input_row_offsets: torch.Tensor,
    state_dict: dict[str, torch.Tensor],
    rope_dim: int,
) -> None:
    total_seq_len = batch_size * seq_len
    torch_output = run_torch_indexer(x, qr, state_dict, rope_dim=rope_dim)
    max_output = run_max_indexer(
        x, qr, input_row_offsets, state_dict, rope_dim=rope_dim
    )

    assert max_output.shape[0] == total_seq_len, (
        f"Expected first dimension {total_seq_len}, got {max_output.shape[0]}"
    )
    assert not torch.all(max_output == 0), "Output should not be all zeros"
    assert not torch.all(max_output == -1), (
        "Output should not be all -1 (invalid)"
    )

    # Top-k index order is not part of the indexer's contract, because the
    # consumer scatters these positions into a mask. Compare the sorted rows so
    # that reordered ties do not count as mismatches.
    total_equal = torch.sum(
        torch.eq(
            _sorted_rows(torch_output).view(-1),
            _sorted_rows(max_output).view(-1),
        )
    )
    assert total_equal / float(total_seq_len * index_topk) >= 0.89


@pytest.mark.skipif(
    accelerator_api() == "hip",
    reason="Memory access fault by GPU node-2 (Agent handle: 0x49c8e0a0) on address 0x10e2bfcf8000. Reason: Unknown.",
)
@pytest.mark.skipif(
    is_h100_h200(),
    reason="CUDA call failed: CUDA_ERROR_ILLEGAL_ADDRESS (an illegal memory access was encountered)",
)
@pytest.mark.parametrize(
    "rope_dim", [qk_rope_head_dim, 0], ids=["rope", "nope"]
)
def test_indexer_causal_mask(
    x: torch.Tensor,
    qr: torch.Tensor,
    input_row_offsets: torch.Tensor,
    state_dict: dict[str, torch.Tensor],
    rope_dim: int,
) -> None:
    total_seq_len = batch_size * seq_len
    torch_output = run_torch_indexer(
        x, qr, state_dict, mask=True, rope_dim=rope_dim
    )
    max_output = run_max_indexer(
        x,
        qr,
        input_row_offsets,
        state_dict,
        MHAMaskVariant.CAUSAL_MASK,
        rope_dim=rope_dim,
    )

    assert max_output.shape[0] == total_seq_len, (
        f"Expected first dimension {total_seq_len}, got {max_output.shape[0]}"
    )
    assert not torch.all(max_output == 0), "Output should not be all zeros"
    assert not torch.all(max_output == -1), (
        "Output should not be all -1 (invalid)"
    )

    valid_ids = (
        torch.arange(0, seq_len)
        .unsqueeze(-1)
        .tile((1, index_topk))
        .view(-1)
        .tile(batch_size)
    )
    torch_output_flat = torch_output.view(-1).to("cpu").to(torch.int32)
    max_output_flat = max_output.view(-1).to(torch.int32)

    # Collapse out-of-range picks to -1 before sorting. Sorting first would
    # place the two sides' sentinels at different positions.
    total_equal = torch.sum(
        torch.eq(
            _sorted_rows(
                torch_output_flat.where(torch_output_flat <= valid_ids, -1)
            ),
            _sorted_rows(
                max_output_flat.where(max_output_flat <= valid_ids, -1)
            ),
        )
    )
    assert total_equal / float(total_seq_len * index_topk) >= 0.89


def test_indexer_rejects_partial_rope_width() -> None:
    """A rope width that is neither 0 nor half the head dim cannot split."""
    with pytest.raises(ValueError, match="rope width must be 0 or half"):
        Indexer(
            dim=dim,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            qk_rope_head_dim=index_head_dim // 4,
            index_topk=index_topk,
            q_lora_rank=q_lora_rank,
            devices=[DeviceRef.GPU()],
            quant_config=QuantConfig(
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
            ),
        )
