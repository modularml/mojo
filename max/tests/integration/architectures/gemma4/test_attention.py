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
"""Gemma4 attention tests (bf16 and fp8-KV variants).

Module-scoped fixtures ensure each unique graph compiles once per shard.
See `_attention_helpers.py` for build/execute helpers and `conftest.py`
for shared fixtures.
"""

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from _attention_helpers import (
    MAX_DTYPE,
    TORCH_DTYPE,
    CompiledAttention,
    _attention_test_tensor,
    build_max_attention_graph,
    make_attention_weights_global,
    make_attention_weights_local,
    make_text_config,
)
from conftest import (
    Gemma4RotaryEmbedding,
    Gemma4TextAttention,
)
from max.driver import Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.pipelines.kv_cache import PagedKVCacheManager
from test_common.context_utils import create_text_context
from test_common.graph_utils import is_b100_b200
from test_common.mef_precompile import init_from_mef, mefs_from_env
from torch.utils.dlpack import from_dlpack
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig


@pytest.fixture(scope="module")
def session(device: Device) -> InferenceSession:
    return InferenceSession(devices=[device])


@pytest.fixture(scope="module")
def device() -> Device:
    return Accelerator()


@pytest.fixture(scope="module")
def text_config() -> Gemma4TextConfig:
    return make_text_config()


@pytest.fixture(scope="module")
def input_tensor(text_config: Gemma4TextConfig) -> torch.Tensor:
    torch.manual_seed(42)
    return _attention_test_tensor((1, 11, text_config.hidden_size)).to("cuda")


@pytest.fixture(scope="module")
def attention_weights_local(
    text_config: Gemma4TextConfig,
) -> dict[str, torch.Tensor]:
    return make_attention_weights_local(text_config)


@pytest.fixture(scope="module")
def attention_weights_global(
    text_config: Gemma4TextConfig,
) -> dict[str, torch.Tensor]:
    return make_attention_weights_global(text_config)


def _get_position_embeddings(
    text_config: Gemma4TextConfig,
    input_tensor: torch.Tensor,
    use_global_rope: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates rotary position embeddings based on the input tensor shape."""
    seq_len = input_tensor.shape[1]
    position_ids = torch.arange(
        seq_len, dtype=torch.long, device="cuda"
    ).unsqueeze(0)

    rope_params = getattr(text_config, "rope_parameters", None)
    if isinstance(rope_params, dict) and "sliding_attention" in rope_params:
        # v5: single embedding handles both layer types natively
        rotary_emb = Gemma4RotaryEmbedding(config=text_config, device="cuda")
        layer_type = (
            "full_attention" if use_global_rope else "sliding_attention"
        )
        cos, sin = rotary_emb(input_tensor, position_ids, layer_type=layer_type)
    else:
        # v4: need separate embedding with hacked config for local rope
        if use_global_rope:
            rotary_emb = Gemma4RotaryEmbedding(
                config=text_config, device="cuda"
            )
        else:
            config = copy.deepcopy(text_config)
            config.rope_theta = config.rope_local_base_freq
            config.rope_scaling = {"rope_type": "default"}
            rotary_emb = Gemma4RotaryEmbedding(config=config, device="cuda")
        cos, sin = rotary_emb(input_tensor, position_ids)

    return cos.to(TORCH_DTYPE).to("cuda"), sin.to(TORCH_DTYPE).to("cuda")


def _causal_attention_mask(seq_len: int) -> torch.Tensor:
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device="cuda"),
        diagonal=1,
    )
    attention_mask = torch.zeros(
        1, 1, seq_len, seq_len, dtype=TORCH_DTYPE, device="cuda"
    )
    attention_mask = attention_mask.masked_fill(
        causal_mask[None, None, :, :], torch.finfo(TORCH_DTYPE).min
    )
    return attention_mask


@torch.no_grad()
def generate_torch_outputs(
    text_config: Gemma4TextConfig,
    input_tensor: torch.Tensor,
    attention_weights: dict[str, torch.Tensor],
    layer_idx: int,
) -> torch.Tensor:
    """Generates the outputs of the MAX and PyTorch attention layers.

    `layer_idx` affects whether the local or global `RoPE` is used. When
    `layer_idx % 6 == 5`, the global `RoPE` is used. Otherwise, the local `RoPE`
    is used.
    """
    layer = (
        Gemma4TextAttention(
            text_config,
            layer_idx=layer_idx,
        )
        .to(TORCH_DTYPE)
        .to("cuda")
    )

    for name, param in layer.named_parameters():
        param.data = attention_weights[name].to(TORCH_DTYPE).to("cuda")

    attention_mask = _causal_attention_mask(input_tensor.shape[1])
    use_global_rope = layer_idx % 6 == 5
    position_embeddings = _get_position_embeddings(
        text_config, input_tensor, use_global_rope
    )

    return layer(input_tensor, position_embeddings, attention_mask)[0]


def build_max_attention(
    mef_path: Path,
    session: InferenceSession,
    text_config: Gemma4TextConfig,
    attention_weights: dict[str, torch.Tensor],
    dtype: DType,
    device_ref: DeviceRef,
    layer_idx: int,
    *,
    cache_dtype: DType | None = None,
) -> CompiledAttention:
    _, attention, kv_params = build_max_attention_graph(
        text_config,
        attention_weights,
        dtype,
        device_ref,
        layer_idx,
        cache_dtype=cache_dtype,
    )

    # Set up blank KV cache.
    kv_manager = PagedKVCacheManager(
        params=kv_params,
        total_num_pages=8,
        session=session,
        max_batch_size=128,
    )
    compiled = init_from_mef(session, mef_path, attention.state_dict())
    return CompiledAttention(compiled=compiled, kv_manager=kv_manager)


def execute_max_attention(
    compiled_attention: CompiledAttention,
    input_tensor: torch.Tensor,
    device: Device,
) -> torch.Tensor:
    """Runs a previously compiled attention graph against a fresh KV claim.

    Releases the request after execution so the shared kv_manager doesn't
    accumulate state across test invocations.
    """
    input_seq_len = input_tensor.shape[1]
    kv_manager = compiled_attention.kv_manager
    compiled = compiled_attention.compiled

    batch = [create_text_context(np.empty(input_seq_len))]
    kv_manager.claim(batch[0])
    try:
        kv_manager.alloc(batch[0])
        kv_runtime_inputs = kv_manager.runtime_inputs([batch])

        # Under fp8 KV the kv_params.get_symbolic_inputs() expands with
        # `kv_scales` buffer inputs.  Mirror that on the runtime side by
        # including them in the execute call when present.
        execute_args: list[Any] = [
            Buffer.from_dlpack(input_tensor[0]).to(device),
            Buffer.from_numpy(np.array([0, input_seq_len], dtype=np.uint32)).to(
                device
            ),
            *kv_runtime_inputs.flatten(),
        ]
        output = compiled.execute(*execute_args)[0]
    finally:
        kv_manager.release(batch[0])
    return output


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two flat tensors (cast to fp32)."""
    af = a.to(torch.float32).flatten()
    bf = b.to(torch.float32).flatten()
    return float(
        torch.dot(af, bf) / (torch.linalg.norm(af) * torch.linalg.norm(bf))
    )


def assert_fp8_matches_bf16(
    bf16_compiled: CompiledAttention,
    fp8_compiled: CompiledAttention,
    input_tensor: torch.Tensor,
    device: Device,
    layer_idx: int,
    head_dim_for_log: int,
) -> None:
    """Shared helper: execute the bf16 reference and fp8 paths on the same
    inputs from already-compiled attention graphs; assert cosine >= 0.99.

    The bf16 reference uses the dtype = MAX_DTYPE cache (= bf16). The fp8
    path uses `cache_dtype=float8_e4m3fn` + per-block fp32 scales at
    granularity=64 (production Gemma4 wiring).  Both paths use
    `rope.interleaved=False` (the trained Gemma4 RoPE convention).
    """
    bf16_out = execute_max_attention(bf16_compiled, input_tensor, device)
    fp8_out = execute_max_attention(fp8_compiled, input_tensor, device)

    bf16_t = from_dlpack(bf16_out).to(torch.float32)
    fp8_t = from_dlpack(fp8_out).to(torch.float32)
    cos = _cosine_similarity(bf16_t, fp8_t)
    max_abs_diff = float((bf16_t - fp8_t).abs().max())
    print(
        f"[fp8_vs_bf16] layer_idx={layer_idx} head_dim={head_dim_for_log} "
        f"cosine={cos:.6f} max_abs_diff={max_abs_diff:.4f}"
    )
    assert cos >= 0.99, (
        "fp8 KV attention output diverged from bf16 baseline: "
        f"cosine={cos:.4f} < 0.99 (layer_idx={layer_idx} "
        f"head_dim={head_dim_for_log})"
    )


# ---------------------------------------------------------------------------
# BF16 compiled fixtures (shared by bf16 tests and as baselines for fp8)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compiled_local_bf16(
    session: InferenceSession,
    text_config: Gemma4TextConfig,
    attention_weights_local: dict[str, torch.Tensor],
) -> CompiledAttention:
    mef_path = mefs_from_env("MEF_RLOCATIONS")["compiled_local_bf16.mef"]
    return build_max_attention(
        mef_path,
        session,
        text_config,
        attention_weights_local,
        MAX_DTYPE,
        DeviceRef.GPU(),
        layer_idx=0,
    )


@pytest.fixture(scope="module")
def compiled_global_bf16(
    session: InferenceSession,
    text_config: Gemma4TextConfig,
    attention_weights_global: dict[str, torch.Tensor],
) -> CompiledAttention:
    mef_path = mefs_from_env("MEF_RLOCATIONS")["compiled_global_bf16.mef"]
    return build_max_attention(
        mef_path,
        session,
        text_config,
        attention_weights_global,
        MAX_DTYPE,
        DeviceRef.GPU(),
        layer_idx=5,
    )


# ---------------------------------------------------------------------------
# Native pure-fp8 compiled fixtures (no per-block scales, scale=1)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compiled_local_native_fp8(
    session: InferenceSession,
    text_config: Gemma4TextConfig,
    attention_weights_local: dict[str, torch.Tensor],
) -> CompiledAttention:
    if not is_b100_b200():
        pytest.skip("Native FP8 MHA requires B200 (SM100)")
    mef_path = mefs_from_env("MEF_RLOCATIONS")["compiled_local_native_fp8.mef"]
    return build_max_attention(
        mef_path,
        session,
        text_config,
        attention_weights_local,
        MAX_DTYPE,
        DeviceRef.GPU(),
        layer_idx=0,
        cache_dtype=DType.float8_e4m3fn,
    )


@pytest.fixture(scope="module")
def compiled_global_native_fp8(
    session: InferenceSession,
    text_config: Gemma4TextConfig,
    attention_weights_global: dict[str, torch.Tensor],
) -> CompiledAttention:
    if not is_b100_b200():
        pytest.skip("Native FP8 MHA requires B200 (SM100)")
    mef_path = mefs_from_env("MEF_RLOCATIONS")["compiled_global_native_fp8.mef"]
    return build_max_attention(
        mef_path,
        session,
        text_config,
        attention_weights_global,
        MAX_DTYPE,
        DeviceRef.GPU(),
        layer_idx=5,
        cache_dtype=DType.float8_e4m3fn,
    )


# ---------------------------------------------------------------------------
# BF16 attention tests
# ---------------------------------------------------------------------------


def test_attention_local(
    text_config: Gemma4TextConfig,
    input_tensor: torch.Tensor,
    attention_weights_local: dict[str, torch.Tensor],
    compiled_local_bf16: CompiledAttention,
    device: Device,
) -> None:
    max_output = execute_max_attention(
        compiled_local_bf16, input_tensor, device
    )

    torch_output = generate_torch_outputs(
        text_config, input_tensor, attention_weights_local, layer_idx=0
    )

    torch.testing.assert_close(
        torch_output.squeeze(0).to(TORCH_DTYPE),
        from_dlpack(max_output).to(TORCH_DTYPE),
        rtol=2 * torch.finfo(TORCH_DTYPE).eps,
        atol=8 * torch.finfo(TORCH_DTYPE).eps,
    )


def test_attention_global(
    text_config: Gemma4TextConfig,
    input_tensor: torch.Tensor,
    attention_weights_global: dict[str, torch.Tensor],
    compiled_global_bf16: CompiledAttention,
    device: Device,
) -> None:
    max_output = execute_max_attention(
        compiled_global_bf16, input_tensor, device
    )
    torch_output = generate_torch_outputs(
        text_config,
        input_tensor,
        attention_weights_global,
        layer_idx=5,
    )

    torch.testing.assert_close(
        torch_output.squeeze(0).to(TORCH_DTYPE),
        from_dlpack(max_output).to(TORCH_DTYPE),
        rtol=2 * torch.finfo(TORCH_DTYPE).eps,
        atol=8 * torch.finfo(TORCH_DTYPE).eps,
    )


# ---------------------------------------------------------------------------
# Native pure-fp8 regression tests
# ---------------------------------------------------------------------------
# These exercise the native pure-fp8 path: fp8 Q/K/V read directly from the
# paged cache, Q@K^T and P@V both raw fp8 MMAs at tensorwise scale=1, bf16
# output. Routes through `mo.rope_split_store.ragged.paged` (outputs roped Q
# as fp8) and `mo.mha.ragged.paged` (fp8 Q+KV in, bf16 out).
# Cosine vs the bf16 baseline must clear the same 0.99 smoke bar. Same
# sensitivity caveat: random-weight smoke gate, not a sufficient bug-detector
# — end-to-end serving accuracy (e.g. gsm8k under a real checkpoint) is the
# authoritative correctness gate.


def test_attention_native_fp8_matches_bf16_local(
    text_config: Gemma4TextConfig,
    input_tensor: torch.Tensor,
    compiled_local_bf16: CompiledAttention,
    compiled_local_native_fp8: CompiledAttention,
    device: Device,
) -> None:
    """Native pure-fp8 (no scales) sliding (head_dim=256) layer vs bf16."""
    assert_fp8_matches_bf16(
        compiled_local_bf16,
        compiled_local_native_fp8,
        input_tensor,
        device,
        layer_idx=0,
        head_dim_for_log=text_config.head_dim,
    )


def test_attention_native_fp8_matches_bf16_global(
    text_config: Gemma4TextConfig,
    input_tensor: torch.Tensor,
    compiled_global_bf16: CompiledAttention,
    compiled_global_native_fp8: CompiledAttention,
    device: Device,
) -> None:
    """Native pure-fp8 (no scales) global (head_dim=512) layer vs bf16."""
    assert_fp8_matches_bf16(
        compiled_global_bf16,
        compiled_global_native_fp8,
        input_tensor,
        device,
        layer_idx=5,
        head_dim_for_log=text_config.global_head_dim,
    )
