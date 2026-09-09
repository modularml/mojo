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

"""Named layer specs: the graphs the harness tests compile.

A spec pairs a harness class with one set of static params, which is exactly
what determines a graph. Both sides of the compile-on-CPU/run-on-GPU split
import from here so the artifact and the test can't drift: the producer
(``precompile_layer``) compiles ``SPECS[name]`` to ``<name>.mef`` on a CPU
build action, and the GPU test looks that MEF up by the same name (see
``docs/internal/CompileOnCpuRunOnGpu.md``).

Nothing here may create a device or read ``accelerator_count()`` at import
time -- the producer sets the virtual-device knobs after importing this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Generic

from testbed.harness import (
    ContextT,
    DynamicParamsT,
    LayerTestHarness,
    StaticParamsT,
)
from testbed.harnesses.attention_with_rope import (
    AttentionWithRopeHarness,
    AttentionWithRopeStaticParams,
)
from testbed.harnesses.gemma3_attention import (
    Gemma3AttentionHarness,
    Gemma3AttentionStaticParams,
)
from testbed.harnesses.gemma4_attention import (
    Gemma4AttentionHarness,
    Gemma4AttentionStaticParams,
)
from testbed.harnesses.gpt_oss_attention import (
    GptOssAttentionHarness,
    GptOssAttentionStaticParams,
)
from testbed.harnesses.olmo2_attention import (
    Olmo2AttentionHarness,
    Olmo2AttentionStaticParams,
)
from testbed.harnesses.qwen2_5vl_attention import (
    Qwen25VLAttentionHarness,
    Qwen25VLAttentionStaticParams,
)
from testbed.harnesses.qwen3_attention import (
    Qwen3AttentionHarness,
    Qwen3AttentionStaticParams,
)
from testbed.harnesses.rms_norm import RMSNormHarness, RMSNormStaticParams
from testbed.harnesses.text_encoder import (
    TextEncoderHarness,
    TextEncoderStaticParams,
)
from testbed.harnesses.wan_vae_attention import (
    WanVaeAttentionHarness,
    WanVaeAttentionStaticParams,
)


@dataclass(frozen=True)
class LayerSpec(Generic[StaticParamsT, DynamicParamsT, ContextT]):
    """One compilable graph: a harness class plus the static params to build it.

    ``name`` is the identity of the compiled artifact -- the ``--spec`` the
    producer receives and the ``<name>.mef`` the test resolves.
    """

    name: str
    harness_type: type[
        LayerTestHarness[StaticParamsT, DynamicParamsT, ContextT]
    ]
    static_params: StaticParamsT


# Llama-3-8B-like config.
ATTENTION_WITH_ROPE_BF16 = LayerSpec(
    "attention_with_rope_bf16",
    AttentionWithRopeHarness,
    AttentionWithRopeStaticParams(
        hidden_size=4096,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        max_seq_len=65536,
        rope_theta=500000.0,
        dtype="bf16",
    ),
)

# Same config as ATTENTION_WITH_ROPE_BF16 with the rope+store fusion off, so
# the graph carries separate rope and kv-cache-store ops.
ATTENTION_UNFUSED_ROPE = LayerSpec(
    "attention_unfused_rope",
    AttentionWithRopeHarness,
    AttentionWithRopeStaticParams(
        hidden_size=4096,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        max_seq_len=65536,
        rope_theta=500000.0,
        dtype="bf16",
        _fuse_rope_and_store=False,
    ),
)

ATTENTION_WITH_ROPE_FP8 = LayerSpec(
    "attention_with_rope_fp8",
    AttentionWithRopeHarness,
    AttentionWithRopeStaticParams(
        hidden_size=16384,
        n_heads=16,
        n_kv_heads=1,
        head_dim=128,
        max_seq_len=131072,
        rope_theta=500000.0,
        dtype="fp8",
    ),
)

ATTENTION_WITH_ROPE_FP4 = LayerSpec(
    "attention_with_rope_fp4",
    AttentionWithRopeHarness,
    AttentionWithRopeStaticParams(
        hidden_size=4096,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        max_seq_len=131072,
        rope_theta=500000.0,
        dtype="fp4",
    ),
)

# google/gemma-3-1b-it config.
GEMMA3_ATTENTION = LayerSpec(
    "gemma3_attention",
    Gemma3AttentionHarness,
    Gemma3AttentionStaticParams(
        hidden_size=2304,
        n_heads=8,
        n_kv_heads=4,
        head_dim=256,
        max_seq_len=32768,
        rope_theta=1000000.0,
        qk_norm_eps=1e-6,
        sliding_window_pattern=6,
        local_window_size=1024,
        # layer_idx=5 => (5+1) % 6 == 0 => global/causal attention (no sliding).
        layer_idx=5,
    ),
)

# gemma4-12b sliding window config (reduced max_seq_len for test GPU memory).
GEMMA4_ATTENTION = LayerSpec(
    "gemma4_attention",
    Gemma4AttentionHarness,
    Gemma4AttentionStaticParams(
        hidden_size=3840,
        n_heads=16,
        n_kv_heads=8,
        n_global_kv_heads=4,
        head_dim=256,
        global_head_dim=512,
        max_seq_len=16384,
        rope_theta=1000000.0,
        is_sliding=True,
        attention_k_eq_v=True,
        local_window_size=1024,
        total_num_pages=2048,
    ),
)

# openai/gpt-oss-120b config.
GPT_OSS_ATTENTION = LayerSpec(
    "gpt_oss_attention",
    GptOssAttentionHarness,
    GptOssAttentionStaticParams(
        hidden_size=2880,
        n_heads=64,
        n_kv_heads=8,
        head_dim=64,
        max_seq_len=131072,
        rope_theta=150000.0,
        has_bias=True,
        layer_type="full_attention",
        local_window_size=128,
        rope_factor=32.0,
        rope_beta_fast=32.0,
        rope_beta_slow=1.0,
        rope_original_max_pos=4096,
        rope_truncate=False,
    ),
)

# OLMo-2-7B-like config.
OLMO2_ATTENTION = LayerSpec(
    "olmo2_attention",
    Olmo2AttentionHarness,
    Olmo2AttentionStaticParams(
        hidden_size=4096,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        max_seq_len=4096,
        rope_theta=500000.0,
        rms_norm_eps=1e-5,
    ),
)

# Qwen2.5-VL-3B config.
QWEN2_5VL_ATTENTION = LayerSpec(
    "qwen2_5vl_attention",
    Qwen25VLAttentionHarness,
    Qwen25VLAttentionStaticParams(
        hidden_size=2048,
        n_heads=16,
        n_kv_heads=2,
        head_dim=128,
        max_seq_len=1024,
        rope_theta=1000000.0,
        mrope_section=[16, 24, 24],
    ),
)

# Qwen3-8B-like config.
QWEN3_ATTENTION = LayerSpec(
    "qwen3_attention",
    Qwen3AttentionHarness,
    Qwen3AttentionStaticParams(
        hidden_size=4096,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        max_seq_len=4096,
        rope_theta=1000000.0,
        qk_norm_eps=1e-6,
    ),
)

RMS_NORM = LayerSpec(
    "rms_norm", RMSNormHarness, RMSNormStaticParams(dim=4096, eps=1e-6)
)

# Small smoke config; the production WAN 2.2 encoder is dialled in via env
# vars, which diverge from this spec and so compile in the test (see
# `layer_mefs.create_runner`).
TEXT_ENCODER = LayerSpec(
    "text_encoder",
    TextEncoderHarness,
    TextEncoderStaticParams(
        vocab_size=1000,
        d_model=256,
        d_kv=32,
        d_ff=512,
        num_layers=2,
        num_heads=8,
        embed_seq_len=226,
    ),
)

# Smoke config; production tiers come from env vars, as for TEXT_ENCODER.
WAN_VAE_ATTENTION = LayerSpec(
    "wan_vae_attention",
    WanVaeAttentionHarness,
    WanVaeAttentionStaticParams(dim=96),
)

SPECS: Mapping[str, LayerSpec[Any, Any, Any]] = {
    spec.name: spec
    for spec in (
        ATTENTION_WITH_ROPE_BF16,
        ATTENTION_UNFUSED_ROPE,
        ATTENTION_WITH_ROPE_FP8,
        ATTENTION_WITH_ROPE_FP4,
        GEMMA3_ATTENTION,
        GEMMA4_ATTENTION,
        GPT_OSS_ATTENTION,
        OLMO2_ATTENTION,
        QWEN2_5VL_ATTENTION,
        QWEN3_ATTENTION,
        RMS_NORM,
        TEXT_ENCODER,
        WAN_VAE_ATTENTION,
    )
}
