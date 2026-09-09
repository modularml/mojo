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
"""Tests for DeepseekV3.2 Indexer layer."""

from __future__ import annotations

import math
import os

import conftest as _deepseek_v32_torch
import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer, accelerator_api
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, Shape, TensorType, ops
from max.graph.weights import WeightData
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
from max.nn.rotary_embedding import (
    DeepseekYarnRopeScalingParams,
    DeepseekYarnRotaryEmbedding,
)
from max.pipelines.architectures.deepseekV3_2.layers import Indexer
from max.pipelines.kv_cache import PagedKVCacheManager
from test_common.context_utils import create_text_context
from test_common.graph_utils import is_h100_h200
from torch.utils.dlpack import from_dlpack

# Small dimensions; rope_head_dim must equal head_dim/2; head_dim must be
# divisible by act_quant block_size (128)
dim: int = 128
index_n_heads: int = 64
index_head_dim: int = 128
qk_rope_head_dim: int = 64
index_topk: int = 4
q_lora_rank: int = 128  # must be divisible by block_size for fp8 act_quant
batch_size: int = 2
seq_len: int = 8
num_keys: int = 8  # cache size (same as seq_len for prefill-like test)
page_size: int = 128

seed: int = 42


@pytest.fixture
def x() -> torch.Tensor:
    """Generate dummy input tensor."""
    torch.manual_seed(seed)
    return torch.randn(
        batch_size,
        seq_len,
        dim,
        dtype=torch.bfloat16,
    ) / math.sqrt(dim)


@pytest.fixture
def qr() -> torch.Tensor:
    """Generate dummy input tensor."""
    torch.manual_seed(seed)
    return torch.randn(
        batch_size,
        seq_len,
        q_lora_rank,
        dtype=torch.bfloat16,
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
        "k_norm.bias": torch.randn(index_head_dim, dtype=torch.float32)
        / math.sqrt(index_head_dim),
        "weights_proj.weight": (
            torch.randn(index_n_heads, dim) / math.sqrt(dim)
        ).to(torch.bfloat16),
    }


def run_torch_indexer(
    x: torch.Tensor,
    qr: torch.Tensor,
    state_dict: dict[str, torch.Tensor],
    mask: bool | None = None,
    device: str = "cuda",
) -> torch.Tensor:
    """Run Indexer forward using fixture inputs and fixture state_dict."""
    args = _deepseek_v32_torch.ModelArgs(
        dim=dim,
        q_lora_rank=q_lora_rank,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        qk_rope_head_dim=qk_rope_head_dim,
    )
    _deepseek_v32_torch.Linear.dtype = (
        torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
    )
    _deepseek_v32_torch.Linear.scale_fmt = args.scale_fmt
    indexer = _deepseek_v32_torch.Indexer(args)
    indexer.load_state_dict(
        {
            k: v
            for k, v in state_dict.items()
            if k != "wq_b.weight_scale" and k != "wk.weight_scale"
        },
        strict=True,
    )
    indexer = indexer.to(device).eval()
    freqs_cis = _deepseek_v32_torch.precompute_freqs_cis(args).to(device)
    start_pos = 0
    seqlen = x.size(1)
    freqs_slice = freqs_cis[start_pos : start_pos + seqlen]

    if mask:
        mask = torch.full((seqlen, seqlen), float("-inf"), device=device).triu_(
            1
        )
    with torch.inference_mode():
        return indexer(
            x.to(device), qr.to(device), start_pos, freqs_slice.to(device), mask
        )


def run_max_indexer(
    x: torch.Tensor,
    qr: torch.Tensor,
    input_row_offsets: torch.Tensor,
    state_dict: dict[str, torch.Tensor],
    mask_variant: MHAMaskVariant = MHAMaskVariant.NULL_MASK,
    *,
    slice_freqs_cis: bool = False,
) -> torch.Tensor:
    """Tests that the indexer layer can execute and outputs are not all zeros.

    Args:
        slice_freqs_cis: If True, slice ``rope.freqs_cis`` to ``[:seq_len]``
            before passing it to the indexer.  This is the buggy codepath
            that triggers a timing-sensitive OOB in the mo.slice +
            mo.composite.rope.ragged fusion (see ``test_indexer_race_demo``
            and TODO(GEX-3777)).  Production code (``deepseekV3_2.py:504``)
            passes the full tensor, so this defaults to False.
    """
    device = Accelerator()
    session = InferenceSession(devices=[device])

    # Block-scaled FP8 matmul kernels require f32 scale tensors; Linear scale
    # Weights use WeightScaleSpec / InputScaleSpec dtype.
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

    # Create KV cache params for FP8 indexer cache with scales
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

    # Create RoPE embedding
    scaling_params = DeepseekYarnRopeScalingParams(
        scaling_factor=40.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1.0,
        mscale_all_dim=1.0,
    )
    rope = DeepseekYarnRotaryEmbedding(
        dim=qk_rope_head_dim,
        n_heads=index_n_heads,
        theta=10000.0,
        max_seq_len=1024,
        head_dim=qk_rope_head_dim,
        interleaved=False,
        scaling_params=scaling_params,
    )

    # Create Indexer
    indexer = Indexer(
        dim=dim,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        index_topk=index_topk,
        q_lora_rank=q_lora_rank,
        devices=[DeviceRef.GPU()],
        quant_config=quant_config,
    )

    # Convert state_dict to WeightData format
    weights_registry: dict[str, WeightData] = {}
    for key, tensor in state_dict.items():
        if tensor.dtype == torch.float8_e4m3fn:
            weights_registry[key] = WeightData(
                Buffer.from_dlpack(tensor.view(torch.uint8)).view(
                    DType.float8_e4m3fn
                ),
                key,
                DType.float8_e4m3fn,
                Shape(tensor.shape),
            )
        elif tensor.dtype == torch.bfloat16:
            weights_registry[key] = WeightData(
                tensor,
                key,
                DType.bfloat16,
                Shape(tensor.shape),
            )
        else:
            weights_registry[key] = WeightData(
                tensor,
                key,
                DType.float32,
                Shape(tensor.shape),
            )
    indexer.load_state_dict(weights_registry, strict=True)

    # Define input types
    x_type = TensorType(DType.bfloat16, ["total_seq_len", dim], DeviceRef.GPU())
    qr_type = TensorType(
        DType.bfloat16, ["total_seq_len", q_lora_rank], DeviceRef.GPU()
    )
    input_row_offsets_type = TensorType(
        DType.uint32, ["batch_size_plus_1"], DeviceRef.GPU()
    )

    # Build the graph
    with Graph(
        "IndexerTest",
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

        # Pass the full freqs_cis (matching production usage in
        # deepseekV3_2.py:504).  Slicing freqs_cis to [:seq_len] before
        # passing it in triggers a timing-sensitive OOB inside the
        # mo.slice + mo.composite.rope.ragged fusion under noisy-neighbor
        # host jitter on BuildBuddy remote-b200; production never slices.
        # TODO(GEX-3777): when the fusion bug is fixed, drop
        # ``slice_freqs_cis`` and just pass ``rope.freqs_cis``.
        freqs_cis_arg = (
            rope.freqs_cis if not slice_freqs_cis else rope.freqs_cis[:seq_len]
        )
        result = indexer(
            x_in,
            qr_in,
            freqs_cis_arg,
            input_row_offsets_in,
            indexer_k_collection,
            layer_idx,
            mask_variant,
        )

        graph.output(result)

    # Compile the graph
    compiled = session.load(graph, weights_registry=indexer.state_dict())

    # Allocate KV cache for batch
    prompt_lens = [seq_len] * batch_size
    batch_contexts = []
    for prompt_len in prompt_lens:
        context = create_text_context(np.empty(prompt_len, dtype=np.int64))
        kv_manager.claim(context)
        kv_manager.alloc(context)
        batch_contexts.append(context)

    kv_inputs = kv_manager.runtime_inputs_for_leaf([batch_contexts]).inputs[0]

    # Prepare input tensors - flatten batch dimension for ragged format
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

    # Execute the graph
    output_result: list[Buffer] = compiled.execute(
        x_device,
        qr_device,
        input_row_offsets_device,
        *kv_inputs.flatten(),
    )

    # Verify output is not all zeros
    output_tensor = from_dlpack(output_result[0])
    output_tensor = output_tensor.cpu()
    return output_tensor


@pytest.mark.skip(
    reason="Disabled due to nondeterminism. "
    "https://linear.app/modularml/issue/GEX-3777 "
    "https://linear.app/modularml/issue/QUA-448 "
    "Re-enable tracking: https://linear.app/modularml/issue/MODELS-1543"
)
@pytest.mark.skipif(
    accelerator_api() == "hip",
    reason="Memory access fault by GPU node-2 (Agent handle: 0x49c8e0a0) on address 0x10e2bfcf8000. Reason: Unknown.",
)
@pytest.mark.skipif(
    is_h100_h200(),
    reason="CUDA call failed: CUDA_ERROR_ILLEGAL_ADDRESS (an illegal memory access was encountered)",
)
def test_indexer_no_mask(
    x: torch.Tensor,
    qr: torch.Tensor,
    input_row_offsets: torch.Tensor,
    state_dict: dict[str, torch.Tensor],
) -> None:
    total_seq_len = batch_size * seq_len
    torch_output = run_torch_indexer(x, qr, state_dict)
    max_output = run_max_indexer(x, qr, input_row_offsets, state_dict)

    assert max_output.shape[0] == total_seq_len, (
        f"Expected first dimension {total_seq_len}, got {max_output.shape[0]}"
    )
    assert not torch.all(max_output == 0), "Output should not be all zeros"
    assert not torch.all(max_output == -1), (
        "Output should not be all -1 (invalid)"
    )

    total_equal = torch.sum(
        torch.eq(
            torch_output.view(-1).to("cpu").to(torch.int32),
            max_output.view(-1).to(torch.int32),
        )
    )
    assert total_equal / float(total_seq_len * index_topk) >= 0.89


@pytest.mark.skip(
    reason="Disabled due to nondeterminism. "
    "https://linear.app/modularml/issue/GEX-3777 "
    "https://linear.app/modularml/issue/QUA-448 "
    "Re-enable tracking: https://linear.app/modularml/issue/MODELS-1543"
)
@pytest.mark.skipif(
    accelerator_api() == "hip",
    reason="Memory access fault by GPU node-2 (Agent handle: 0x49c8e0a0) on address 0x10e2bfcf8000. Reason: Unknown.",
)
@pytest.mark.skipif(
    is_h100_h200(),
    reason="CUDA call failed: CUDA_ERROR_ILLEGAL_ADDRESS (an illegal memory access was encountered)",
)
def test_indexer_causal_mask(
    x: torch.Tensor,
    qr: torch.Tensor,
    input_row_offsets: torch.Tensor,
    state_dict: dict[str, torch.Tensor],
) -> None:
    total_seq_len = batch_size * seq_len
    torch_output = run_torch_indexer(x, qr, state_dict, mask=True)
    max_output = run_max_indexer(
        x, qr, input_row_offsets, state_dict, MHAMaskVariant.CAUSAL_MASK
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

    total_equal = torch.sum(
        torch.eq(
            torch_output_flat.where(torch_output_flat <= valid_ids, -1),
            max_output_flat.where(max_output_flat <= valid_ids, -1),
        )
    )
    assert total_equal / float(total_seq_len * index_topk) >= 0.89


# ------------------------------------------------------------------
# Race-condition demonstration (manual / skipped in CI)
# ------------------------------------------------------------------
#
# Passing ``rope.freqs_cis[:seq_len]`` (a CPU-side strided view) into
# the indexer's rope_ragged op fuses with ``mo.slice`` to produce a
# kernel that races on the freqs_cis transfer/lifetime under host
# jitter.  Empirically:
#
#   * remote-b200 (BuildBuddy, shared host, noisy neighbors):
#     3-5/15 fresh-process reruns crash with CUDA_ERROR_ILLEGAL_ADDRESS
#     inside a fused kernel whose op chain is:
#         mo.slice : !mo.tensor<[1024,64], f32> -> !mo.tensor<[8,64], f32>
#         custom__mo.rope.ragged : (q_pe, row_offsets, cache_lengths,
#                                   sliced_freqs_cis) -> !mo.tensor<[dyn,64,64], bf16>
#
#   * Same code with the slice removed (matching the production pattern
#     in ``deepseekV3_2.py:504``): 15/15 pass.
#
# This test is gated on the ``DSV32_RUN_RACE_DEMO`` env var so CI
# stays green, but is preserved as a minimal reproducer for the
# underlying kernel-level bug.  TODO(GEX-3777): once the
# mo.slice + mo.composite.rope.ragged fusion is fixed, delete this test
# (and the ``slice_freqs_cis`` kwarg on ``run_max_indexer``).
#
# NOTE: the op was renamed from the ``mo.rope.ragged`` custom op to the
# typed ``mo.composite.rope.ragged`` op; the captured crash trace below
# predates that rename and still shows the old ``custom__mo.rope.ragged``
# dispatch-summary name. A fresh repro will show the new op name instead.
#
# Run manually with::
#
#     DSV32_RUN_RACE_DEMO=1 ./bazelw test --config=remote-b200 \
#         --test_env=DSV32_RUN_RACE_DEMO=1 \
#         --test_arg=-k --test_arg=test_indexer_race_demo \
#         --runs_per_test=20 \
#         //max/tests/integration/architectures/deepseekV3_2:test_indexer
@pytest.mark.skipif(
    not os.environ.get("DSV32_RUN_RACE_DEMO"),
    reason="Race demo: triggers CUDA_ERROR_ILLEGAL_ADDRESS under host "
    "jitter via the mo.slice + mo.composite.rope.ragged fusion.  Set "
    "DSV32_RUN_RACE_DEMO=1 to opt in.  See TODO(GEX-3777).",
)
def test_indexer_race_demo(
    x: torch.Tensor,
    qr: torch.Tensor,
    input_row_offsets: torch.Tensor,
    state_dict: dict[str, torch.Tensor],
) -> None:
    """Manual repro for the rope-slice fusion race (see comment above)."""
    out = run_max_indexer(
        x, qr, input_row_offsets, state_dict, slice_freqs_cis=True
    )
    assert out.shape[0] == batch_size * seq_len
