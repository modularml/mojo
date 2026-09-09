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
"""GPU correctness test for Nemotron-H FP8 KV-cache attention.

Exercises the native unscaled FP8 MHA path added to ``NemotronHAttention``:
when ``kv_params.is_fp8_kv_dtype``, Q/K/V are cast to the FP8 cache dtype,
stored, and fed to ``flash_attention_ragged`` with a BF16 ``output_dtype`` so
``o_proj`` stays BF16. This compares the FP8-cache attention output against the
existing BF16-cache behavior on identical random weights and inputs.

Excluded from the fast CPU-only ``tests`` target's ``glob(["test_*.py"])`` and
run only under the dedicated GPU target (see BUILD.bazel).

Tolerance rationale: FP8 (E4M3) quantization of Q/K/V perturbs the QK^T and
P@V dots. On a random-weight smoke this is a coarse gate, not an authoritative
accuracy check (end-to-end gsm8k under the real checkpoint is authoritative).
We require cosine >= 0.99 against the BF16 baseline, matching the gemma4 /
laguna native-FP8 MHA smoke bar.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kv_cache import MHAKVCacheParams
from max.pipelines.architectures.nemotron_h.model_config import NemotronHConfig
from max.pipelines.architectures.nemotron_h.nemotron_h import NemotronHAttention
from max.pipelines.kv_cache import PagedKVCacheManager
from test_common.context_utils import create_text_context
from test_common.graph_utils import is_b100_b200
from torch.utils.dlpack import from_dlpack

# Small attention geometry (GQA): 8 query heads, 2 KV heads, head_dim 128.
HIDDEN_SIZE = 1024
NUM_ATTENTION_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 128
SEQ_LEN = 11
COSINE_THRESHOLD = 0.99

MAX_DTYPE = DType.bfloat16
TORCH_DTYPE = torch.bfloat16


class CompiledAttention(NamedTuple):
    compiled: Model
    kv_manager: PagedKVCacheManager


@pytest.fixture(scope="module")
def device() -> Device:
    return Accelerator()


@pytest.fixture(scope="module")
def session(device: Device) -> InferenceSession:
    return InferenceSession(devices=[device])


@pytest.fixture(scope="module")
def attention_weights() -> dict[str, torch.Tensor]:
    """Random bf16 weights for the fused QKV and output projections."""
    torch.manual_seed(42)
    q_dim = NUM_ATTENTION_HEADS * HEAD_DIM
    kv_dim = NUM_KV_HEADS * HEAD_DIM
    proj_std = 1.0 / math.sqrt(HIDDEN_SIZE)
    return {
        "qkv_proj.weight": (
            torch.randn(q_dim + 2 * kv_dim, HIDDEN_SIZE) * proj_std
        ).to(TORCH_DTYPE),
        "o_proj.weight": (torch.randn(HIDDEN_SIZE, q_dim) * proj_std).to(
            TORCH_DTYPE
        ),
    }


@pytest.fixture(scope="module")
def input_tensor() -> torch.Tensor:
    torch.manual_seed(0)
    x = torch.randn(SEQ_LEN, HIDDEN_SIZE) * (1.0 / math.sqrt(HIDDEN_SIZE))
    return x.to(TORCH_DTYPE)


def _build_config(device_ref: DeviceRef, cache_dtype: DType) -> NemotronHConfig:
    kv_params = MHAKVCacheParams(
        dtype=cache_dtype,
        devices=[device_ref],
        n_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        num_layers=1,
        page_size=256,
    )
    # Only the attention-relevant fields matter; mamba/MLP fields are set to
    # valid placeholders so the (kw_only) dataclass constructs.
    return NemotronHConfig(
        hidden_size=HIDDEN_SIZE,
        vocab_size=256,
        num_hidden_layers=1,
        layer_norm_epsilon=1e-5,
        max_seq_len=1024,
        dtype=MAX_DTYPE,
        devices=[device_ref],
        layer_kinds=["attention"],
        num_attention_heads=NUM_ATTENTION_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        attention_head_dim=HEAD_DIM,
        intermediate_size=HIDDEN_SIZE,
        mamba_num_heads=1,
        mamba_head_dim=HEAD_DIM,
        n_groups=1,
        ssm_state_size=16,
        conv_kernel=4,
        chunk_size=128,
        kv_params=kv_params,
    )


def _build_attention(
    session: InferenceSession,
    attention_weights: dict[str, torch.Tensor],
    cache_dtype: DType,
) -> CompiledAttention:
    device_ref = DeviceRef.GPU()
    config = _build_config(device_ref, cache_dtype)

    attention = NemotronHAttention(config, kv_layer_idx=0)
    attention.load_state_dict(
        {name: value.cpu() for name, value in attention_weights.items()}
    )

    kv_params = config.kv_params
    kv_manager = PagedKVCacheManager(
        params=kv_params,
        total_num_pages=8,
        session=session,
        max_batch_size=128,
    )

    input_type = TensorType(
        MAX_DTYPE, ["total_seq_len", HIDDEN_SIZE], device=device_ref
    )
    input_row_offsets_type = TensorType(
        DType.uint32, shape=["input_row_offsets_len"], device=device_ref
    )
    flattened_kv_types = kv_params.flattened_kv_inputs()

    with Graph(
        "NemotronHAttentionFP8KV",
        input_types=(input_type, input_row_offsets_type, *flattened_kv_types),
    ) as graph:
        x, input_row_offsets, *kv_cache = graph.inputs
        kv_collection = kv_params.unflatten_kv_inputs(iter(kv_cache)).inputs[0]
        layer_idx = ops.constant(0, DType.uint32, device=DeviceRef.CPU())
        graph.output(
            attention(
                layer_idx,
                x.tensor,
                kv_collection,
                input_row_offsets.tensor,
            )
        )

    compiled = session.load(graph, weights_registry=attention.state_dict())
    return CompiledAttention(compiled=compiled, kv_manager=kv_manager)


def _execute(
    compiled_attention: CompiledAttention,
    input_tensor: torch.Tensor,
    device: Device,
) -> torch.Tensor:
    kv_manager = compiled_attention.kv_manager
    compiled = compiled_attention.compiled
    batch = [create_text_context(np.empty(SEQ_LEN))]
    kv_manager.claim(batch[0])
    try:
        kv_manager.alloc(batch[0])
        kv_runtime_inputs = kv_manager.runtime_inputs([batch])
        output = compiled.execute(
            Buffer.from_dlpack(input_tensor).to(device),
            Buffer.from_numpy(np.array([0, SEQ_LEN], dtype=np.uint32)).to(
                device
            ),
            *kv_runtime_inputs.flatten(),
        )[0]
    finally:
        kv_manager.release(batch[0])
    return output


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.to(torch.float32).flatten()
    bf = b.to(torch.float32).flatten()
    return float(
        torch.dot(af, bf) / (torch.linalg.norm(af) * torch.linalg.norm(bf))
    )


def test_nemotron_h_fp8_kv_matches_bf16(
    session: InferenceSession,
    attention_weights: dict[str, torch.Tensor],
    input_tensor: torch.Tensor,
    device: Device,
) -> None:
    """FP8-cache attention output matches the BF16-cache baseline (cos>=0.99)."""
    if not is_b100_b200():
        pytest.skip("Native FP8 MHA requires B200 (SM100)")

    bf16 = _build_attention(session, attention_weights, MAX_DTYPE)
    fp8 = _build_attention(session, attention_weights, DType.float8_e4m3fn)

    bf16_out = from_dlpack(_execute(bf16, input_tensor, device)).to(
        torch.float32
    )
    fp8_out = from_dlpack(_execute(fp8, input_tensor, device)).to(torch.float32)

    cos = _cosine(bf16_out, fp8_out)
    max_abs_diff = float((bf16_out - fp8_out).abs().max())
    print(
        f"[nemotron_h fp8_vs_bf16] cosine={cos:.6f} "
        f"max_abs_diff={max_abs_diff:.4f}"
    )
    assert cos >= COSINE_THRESHOLD, (
        f"fp8 KV attention diverged from bf16: cosine={cos:.4f} "
        f"< {COSINE_THRESHOLD}"
    )
