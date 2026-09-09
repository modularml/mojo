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
"""Kernel-level fuzz orchestrator (see gpu-kernels-fuzzing-design.md).

The Python side owns search/orchestration; each per-kernel Mojo target owns
execution of one case. The flow per target:

  1. build the target once (bazel),
  2. enumerate boundary-aware specs from a seed (target `--mode list-specs`),
  3. run each spec in its own subprocess (`--mode single`) under the chosen
     oracle, with a per-case timeout so a hanging case only kills its own
     process (it does not wedge the whole run),
  4. classify PASS / FAIL / HANG / ERROR, log JSONL, write a corpus entry for
     every non-PASS, and exit non-zero if anything failed.

Conventions (Verdict, result records, registry, CLI, exit code) mirror the
in-tree API fuzzer at max/tests/integration/accuracy/llm_fuzz/.

Examples:
    # Memory-safety fuzz of the MHA causal-padding kernel under memcheck:
    python3 max/kernels/test/gpu/fuzz/fuzz.py --target mha_causal \\
        --oracle memcheck --budget 32 --seed 12345

    # Quick diff-oracle smoke (catches hangs/crashes; no sanitizer):
    python3 max/kernels/test/gpu/fuzz/fuzz.py --target mha_causal \\
        --oracle diff --budget 24
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ===----------------------------------------------------------------------=== #
# Target registry
# ===----------------------------------------------------------------------=== #


@dataclasses.dataclass(frozen=True)
class FuzzTarget:
    """A per-kernel Mojo fuzz target and how to build/run it."""

    name: str
    bazel_target: str
    binary: str  # path under the repo root, relative to bazel-bin via symlink
    description: str
    default_oracle: str


_TARGETS: dict[str, FuzzTarget] = {
    "attn_res_mix": FuzzTarget(
        name="attn_res_mix",
        bazel_target=(
            "//max/kernels/test/gpu/fuzz:fuzz_attn_res_mix.mojo.test"
        ),
        binary="bazel-bin/max/kernels/test/gpu/fuzz/fuzz_attn_res_mix.mojo.test",
        description=(
            "attn_res_mix fused attention-residual softmax mixture "
            "(fp64-reference oracle)"
        ),
        default_oracle="ref",
    ),
    "mha_causal": FuzzTarget(
        name="mha_causal",
        bazel_target="//max/kernels/test/gpu/fuzz:fuzz_mha_causal.mojo.test",
        binary="bazel-bin/max/kernels/test/gpu/fuzz/fuzz_mha_causal.mojo.test",
        description=(
            "MHA CausalPaddingMask boundary fuzz (memory-safety oracle)"
        ),
        default_oracle="memcheck",
    ),
    "mha_nullmask": FuzzTarget(
        name="mha_nullmask",
        bazel_target=(
            "//max/kernels/test/gpu/fuzz:fuzz_mha_nullmask.mojo.test"
        ),
        binary="bazel-bin/max/kernels/test/gpu/fuzz/fuzz_mha_nullmask.mojo.test",
        description="MHA NullMask (non-causal) boundary fuzz (memory-safety + ref)",
        default_oracle="memcheck",
    ),
    "moe_indices": FuzzTarget(
        name="moe_indices",
        bazel_target=("//max/kernels/test/gpu/fuzz:fuzz_moe_indices.mojo.test"),
        binary="bazel-bin/max/kernels/test/gpu/fuzz/fuzz_moe_indices.mojo.test",
        description="MoE moe_create_indices (uninitialized-read oracle)",
        default_oracle="initcheck",
    ),
    "softmax": FuzzTarget(
        name="softmax",
        bazel_target="//max/kernels/test/gpu/fuzz:fuzz_softmax.mojo.test",
        binary="bazel-bin/max/kernels/test/gpu/fuzz/fuzz_softmax.mojo.test",
        description="softmax _softmax_gpu boundary fuzz (memory-safety oracle)",
        default_oracle="memcheck",
    ),
    "reductions": FuzzTarget(
        name="reductions",
        bazel_target="//max/kernels/test/gpu/fuzz:fuzz_reductions.mojo.test",
        binary="bazel-bin/max/kernels/test/gpu/fuzz/fuzz_reductions.mojo.test",
        description=(
            "algorithm.reductions sum/max/min/mean/argmax/argmin over the"
            " rowwise GPU scaffolder; reduced axis drawn from 0, outputs"
            " sentinel-checked for a missing write (ref oracle)"
        ),
        default_oracle="ref",
    ),
    "rms_norm": FuzzTarget(
        name="rms_norm",
        bazel_target="//max/kernels/test/gpu/fuzz:fuzz_rms_norm.mojo.test",
        binary="bazel-bin/max/kernels/test/gpu/fuzz/fuzz_rms_norm.mojo.test",
        description="rms_norm rms_norm_gpu boundary fuzz (memory-safety oracle)",
        default_oracle="memcheck",
    ),
    "layer_norm": FuzzTarget(
        name="layer_norm",
        bazel_target="//max/kernels/test/gpu/fuzz:fuzz_layer_norm.mojo.test",
        binary="bazel-bin/max/kernels/test/gpu/fuzz/fuzz_layer_norm.mojo.test",
        description=(
            "layer_norm layer_norm boundary fuzz (memory-safety + ref)"
        ),
        default_oracle="memcheck",
    ),
    "matmul": FuzzTarget(
        name="matmul",
        bazel_target="//max/kernels/test/gpu/fuzz:fuzz_matmul.mojo.test",
        binary="bazel-bin/max/kernels/test/gpu/fuzz/fuzz_matmul.mojo.test",
        description=(
            "matmul _matmul_gpu tuned SM100 bf16 (memory-safety; also"
            " determinism + batch_variance negative control)"
        ),
        default_oracle="memcheck",
    ),
    "gemv_split_k": FuzzTarget(
        name="gemv_split_k",
        bazel_target="//max/kernels/test/gpu/fuzz:fuzz_gemv_split_k.mojo.test",
        binary="bazel-bin/max/kernels/test/gpu/fuzz/fuzz_gemv_split_k.mojo.test",
        description=(
            "GEMV split-K (SM100 GEMV_SPLIT_K, FP32 decode router GEMM)"
            " run-to-run determinism"
        ),
        default_oracle="determinism",
    ),
    "numeric_canary": FuzzTarget(
        name="numeric_canary",
        bazel_target=(
            "//max/kernels/test/gpu/fuzz:fuzz_numeric_canary.mojo.test"
        ),
        binary=(
            "bazel-bin/max/kernels/test/gpu/fuzz/fuzz_numeric_canary.mojo.test"
        ),
        description=(
            "Deliberate wrong-answer canary -- positive control for ref oracle"
        ),
        default_oracle="ref",
    ),
    "block_scaled_fp4": FuzzTarget(
        name="block_scaled_fp4",
        bazel_target=(
            "//max/kernels/test/gpu/fuzz:fuzz_block_scaled_fp4.mojo.test"
        ),
        binary=(
            "bazel-bin/max/kernels/test/gpu/fuzz/fuzz_block_scaled_fp4.mojo.test"
        ),
        description=("block-scaled FP4 SM100 matmul small_bn (memcheck)"),
        default_oracle="memcheck",
    ),
    "block_scaled_mxfp8": FuzzTarget(
        name="block_scaled_mxfp8",
        bazel_target=(
            "//max/kernels/test/gpu/fuzz:fuzz_block_scaled_mxfp8.mojo.test"
        ),
        binary=(
            "bazel-bin/max/kernels/test/gpu/fuzz/"
            "fuzz_block_scaled_mxfp8.mojo.test"
        ),
        description=(
            "block-scaled MXFP8 SM100 dense matmul (MiniMax-M3-MXFP8 attention"
            " + shared-expert GEMM): e4m3 + E8M0 scales vs cuBLAS ref. ref/"
            "memcheck"
        ),
        default_oracle="ref",
    ),
    "grouped_matmul_mxfp8": FuzzTarget(
        name="grouped_matmul_mxfp8",
        bazel_target=(
            "//max/kernels/test/gpu/fuzz:fuzz_grouped_matmul_mxfp8.mojo.test"
        ),
        binary=(
            "bazel-bin/max/kernels/test/gpu/fuzz/"
            "fuzz_grouped_matmul_mxfp8.mojo.test"
        ),
        description=(
            "grouped block-scaled MXFP8 SM100 matmul (MiniMax-M3-MXFP8 routed"
            " MoE experts): ragged per-expert token distribution fuzz vs"
            " per-expert cuBLAS ref. ref/memcheck"
        ),
        default_oracle="ref",
    ),
    "grouped_matmul_sm100_w4a8": FuzzTarget(
        name="grouped_matmul_sm100_w4a8",
        bazel_target=(
            "//max/kernels/test/gpu/fuzz:"
            "fuzz_grouped_matmul_sm100_w4a8.mojo.test"
        ),
        binary=(
            "bazel-bin/max/kernels/test/gpu/fuzz/"
            "fuzz_grouped_matmul_sm100_w4a8.mojo.test"
        ),
        description=(
            "grouped block-scaled W4A8 SM100 matmul (E4M3 activations against"
            " nibble-packed E2M1 weights): ragged per-expert token distribution"
            " fuzz vs an exact host FP32 ref at one BF16 ulp."
            " ref/determinism/memcheck"
        ),
        default_oracle="ref",
    ),
    "moe_router": FuzzTarget(
        name="moe_router",
        bazel_target=("//max/kernels/test/gpu/fuzz:fuzz_moe_router.mojo.test"),
        binary="bazel-bin/max/kernels/test/gpu/fuzz/fuzz_moe_router.mojo.test",
        description=(
            "single_group_router: MiniMax-M3 MoE top-k routing (select by"
            " score+bias, unbiased normalized scaled weights). ref (host"
            " top-k) + contract (indices in-range/distinct under NaN/Inf)"
        ),
        default_oracle="ref",
    ),
    "mxfp8_quantize": FuzzTarget(
        name="mxfp8_quantize",
        bazel_target=(
            "//max/kernels/test/gpu/fuzz:fuzz_mxfp8_quantize.mojo.test"
        ),
        binary=(
            "bazel-bin/max/kernels/test/gpu/fuzz/fuzz_mxfp8_quantize.mojo.test"
        ),
        description=(
            "quantize_dynamic_scaled_fp4fp8 (MXFP8): BF16 -> fp8_e4m3fn + E8M0"
            " block scales (MiniMax-M3-MXFP8 activation narrowing). contract"
            " (finite input -> finite output, SERVOPT-1420 clamp guard) + ref"
            " (coarse dequant round-trip)"
        ),
        default_oracle="contract",
    ),
    "fused_swiglu_mxfp8": FuzzTarget(
        name="fused_swiglu_mxfp8",
        bazel_target=(
            "//max/kernels/test/gpu/fuzz:fuzz_fused_swiglu_mxfp8.mojo.test"
        ),
        binary=(
            "bazel-bin/max/kernels/test/gpu/fuzz/"
            "fuzz_fused_swiglu_mxfp8.mojo.test"
        ),
        description=(
            "fused_silu_mxfp8_interleaved: MiniMax-M3-MXFP8 MoE up-projection"
            " SwiGLU + MXFP8 re-quantize epilogue (swigluoai(g,u) -> fp8 + E8M0"
            " block scales). ref (host SwiGLU dequant round-trip) + contract"
            " (finiteness)"
        ),
        default_oracle="ref",
    ),
    "fused_swiglu_dispatch": FuzzTarget(
        name="fused_swiglu_dispatch",
        bazel_target=(
            "//max/kernels/test/gpu/fuzz:fuzz_fused_swiglu_dispatch.mojo.test"
        ),
        binary=(
            "bazel-bin/max/kernels/test/gpu/fuzz/"
            "fuzz_fused_swiglu_dispatch.mojo.test"
        ),
        description=(
            "grouped_matmul_swiglu_mxfp8_dispatch: the fully-fused MiniMax-M3"
            " MoE up-projection (matmul+SwiGLU+requant). ref = byte-exact"
            " fusion-equivalence vs the unfused matmul+epilogue chain"
        ),
        default_oracle="ref",
    ),
    "msa_decode": FuzzTarget(
        name="msa_decode",
        bazel_target="//max/kernels/test/gpu/fuzz:fuzz_msa_decode.mojo.test",
        binary="bazel-bin/max/kernels/test/gpu/fuzz/fuzz_msa_decode.mojo.test",
        description=(
            "msa_sm100_decode: MiniMax-M3 block-sparse decode attention"
            " (attends indexer-selected blocks). ref = f64 block-max softmax"
            " masking the partial-local-block tail (CENG-639 runaway:"
            " non-BN-aligned cache_length + 1e4 poison tail)"
        ),
        default_oracle="ref",
    ),
    "msa_prefill": FuzzTarget(
        name="msa_prefill",
        bazel_target="//max/kernels/test/gpu/fuzz:fuzz_msa_prefill.mojo.test",
        binary="bazel-bin/max/kernels/test/gpu/fuzz/fuzz_msa_prefill.mojo.test",
        description=(
            "msa_sm100_prefill_plan + _run: MiniMax-M3 block-sparse PREFILL"
            " attention (ragged Q, on-device reverse-CSR + block-major fwd +"
            " combine). Fuzzes the ragged shape (per-batch seqlen_q/seqlen_k +"
            " q2k selection incl. C==0). ref = f64 block-sparse oracle (the"
            " kernel's authority); memcheck (CSR build + scatter OOB)"
        ),
        default_oracle="ref",
    ),
    "mla_decode": FuzzTarget(
        name="mla_decode",
        bazel_target=("//max/kernels/test/gpu/fuzz:fuzz_mla_decode.mojo.test"),
        binary=(
            "bazel-bin/max/kernels/test/gpu/fuzz/fuzz_mla_decode.mojo.test"
        ),
        description=(
            "generic_flare_mla_decode_kv_cache_ragged: ragged paged-KV MLA"
            " decode (hang/OOB/ref/schedule)"
        ),
        default_oracle="memcheck",
    ),
    "fused_rope_rmsnorm": FuzzTarget(
        name="fused_rope_rmsnorm",
        bazel_target=(
            "//max/kernels/test/gpu/fuzz:fuzz_fused_rope_rmsnorm.mojo.test"
        ),
        binary=(
            "bazel-bin/max/kernels/test/gpu/fuzz/"
            "fuzz_fused_rope_rmsnorm.mojo.test"
        ),
        description=(
            "mla_fused_rope_rmsnorm_quantization: fused MLA RoPE + KV-cache"
            " RMSNorm + quantize-write; ragged shape fuzz (memcheck/ref)"
        ),
        default_oracle="memcheck",
    ),
    "fused_qk_rms_norm": FuzzTarget(
        name="fused_qk_rms_norm",
        bazel_target=(
            "//max/kernels/test/gpu/fuzz:fuzz_fused_qk_rms_norm.mojo.test"
        ),
        binary=(
            "bazel-bin/max/kernels/test/gpu/fuzz/"
            "fuzz_fused_qk_rms_norm.mojo.test"
        ),
        description=(
            "fused_qk_rms_norm_ragged_paged: MiniMax-M3 per-head QK RMSNorm of"
            " Q + the in-place K rows of the paged KV cache (weight_offset"
            " runtime, multiply_before_cast True). ragged shape fuzz;"
            " memcheck/ref (verifies q_output + the in-place K-cache rows)"
        ),
        default_oracle="ref",
    ),
    "fused_qk_rope": FuzzTarget(
        name="fused_qk_rope",
        bazel_target=(
            "//max/kernels/test/gpu/fuzz:fuzz_fused_qk_rope.mojo.test"
        ),
        binary=(
            "bazel-bin/max/kernels/test/gpu/fuzz/fuzz_fused_qk_rope.mojo.test"
        ),
        description=(
            "fused_qk_rope_ragged: MiniMax-M3 interleaved full-head_dim RoPE of"
            " Q + the in-place K rows of the paged KV cache. ragged shape fuzz;"
            " memcheck/ref (verifies output + the in-place-roped K-cache rows)"
        ),
        default_oracle="ref",
    ),
    "fused_qkv_matmul_mxfp8": FuzzTarget(
        name="fused_qkv_matmul_mxfp8",
        bazel_target=(
            "//max/kernels/test/gpu/fuzz:fuzz_fused_qkv_matmul_mxfp8.mojo.test"
        ),
        binary=(
            "bazel-bin/max/kernels/test/gpu/fuzz/"
            "fuzz_fused_qkv_matmul_mxfp8.mojo.test"
        ),
        description=(
            "generic_fused_qkv_matmul_kv_cache_paged_ragged_scale_float4"
            " (MXFP8): MiniMax-M3 fused QKV projection (e4m3 + E8M0 scales),"
            " returns Q + scatters K/V into the paged cache. ragged shape fuzz;"
            " memcheck (cache-scatter OOB) + ref (host dequant GEMM, Q + K/V)"
        ),
        default_oracle="memcheck",
    ),
    "fused_qkv_index_matmul_mxfp8": FuzzTarget(
        name="fused_qkv_index_matmul_mxfp8",
        bazel_target=(
            "//max/kernels/test/gpu/fuzz:"
            "fuzz_fused_qkv_index_matmul_mxfp8.mojo.test"
        ),
        binary=(
            "bazel-bin/max/kernels/test/gpu/fuzz/"
            "fuzz_fused_qkv_index_matmul_mxfp8.mojo.test"
        ),
        description=(
            "generic_fused_qkv_index_matmul_kv_cache_paged_ragged_scale_float4"
            " (MXFP8): MiniMax-M3 5-way dual-cache fused projection. Returns Q +"
            " IndexQ, scatters K/V to the MAIN cache + IndexK to the INDEX (MLA)"
            " cache. ragged shape fuzz; memcheck (two cache-scatter OOB) + ref"
            " (host dequant GEMM: Q+IndexQ + main K/V + index K)"
        ),
        default_oracle="memcheck",
    ),
    "topk_sampling": FuzzTarget(
        name="topk_sampling",
        bazel_target=(
            "//max/kernels/test/gpu/fuzz:fuzz_topk_sampling.mojo.test"
        ),
        binary=(
            "bazel-bin/max/kernels/test/gpu/fuzz/fuzz_topk_sampling.mojo.test"
        ),
        description=(
            "fused_token_sampling_gpu: token sampler unwritten-output-row"
            " (per-row top_k==0); initcheck/poison/memcheck"
        ),
        default_oracle="initcheck",
    ),
    "topk_topp_masked_probs": FuzzTarget(
        name="topk_topp_masked_probs",
        bazel_target=(
            "//max/kernels/test/gpu/fuzz:fuzz_topk_topp_masked_probs.mojo.test"
        ),
        binary=(
            "bazel-bin/max/kernels/test/gpu/fuzz/"
            "fuzz_topk_topp_masked_probs.mojo.test"
        ),
        description=(
            "topk_topp_masked_probs: spec-decode target-side masked softmax"
            " (cluster + single-block dispatch); ref validates the"
            " tie-tolerant accept-predicate contract vs an f64 recompute"
        ),
        default_oracle="ref",
    ),
    "topk_topp_sampling_dist": FuzzTarget(
        name="topk_topp_sampling_dist",
        bazel_target=(
            "//max/kernels/test/gpu/fuzz:fuzz_topk_topp_sampling_dist.mojo.test"
        ),
        binary=(
            "bazel-bin/max/kernels/test/gpu/fuzz/"
            "fuzz_topk_topp_sampling_dist.mojo.test"
        ),
        description=(
            "topk_topp_sampling_from_prob + emit_dist: spec-decode draft"
            " sampler; token-in-nucleus / degenerate-row / emit-inertness"
            " contracts, dist vs the accept predicate"
        ),
        default_oracle="ref",
    ),
    "ep_combine": FuzzTarget(
        name="ep_combine",
        bazel_target=("//max/kernels/test/gpu/fuzz:fuzz_ep_combine.mojo.test"),
        binary=(
            "bazel-bin/max/kernels/test/gpu/fuzz/fuzz_ep_combine.mojo.test"
        ),
        description=(
            "EP MoE combine send_tokens_back: unvalidated src_info write"
            " offset -> wild P2P write (SERVOPT-1458); memcheck/redzone"
        ),
        default_oracle="memcheck",
    ),
    "sparse_indexer": FuzzTarget(
        name="sparse_indexer",
        bazel_target=(
            "//max/kernels/test/gpu/fuzz:fuzz_sparse_indexer.mojo.test"
        ),
        binary=(
            "bazel-bin/max/kernels/test/gpu/fuzz/fuzz_sparse_indexer.mojo.test"
        ),
        description=(
            "block_select_topk: MiniMax-M3 MSA indexer block-selection core;"
            " validity contract (index in-range / distinct / -1-tail) + ref"
            " (top-k invariant). diff/memcheck/redzone/ref/contract"
        ),
        default_oracle="ref",
    ),
    "sparse_indexer_decode": FuzzTarget(
        name="sparse_indexer_decode",
        bazel_target=(
            "//max/kernels/test/gpu/fuzz:fuzz_sparse_indexer_decode.mojo.test"
        ),
        binary=(
            "bazel-bin/max/kernels/test/gpu/fuzz/"
            "fuzz_sparse_indexer_decode.mojo.test"
        ),
        description=(
            "sparse_indexer_decode_score + _topk: MiniMax-M3 MSA indexer decode"
            " path (block-max QK scoring, split-K, init/local forcing, top-k);"
            " ref (f32 block-max oracle) + validity contract + schedule"
        ),
        default_oracle="ref",
    ),
    "sparse_indexer_prefill": FuzzTarget(
        name="sparse_indexer_prefill",
        bazel_target=(
            "//max/kernels/test/gpu/fuzz:fuzz_sparse_indexer_prefill.mojo.test"
        ),
        binary=(
            "bazel-bin/max/kernels/test/gpu/fuzz/"
            "fuzz_sparse_indexer_prefill.mojo.test"
        ),
        description=(
            "sparse_indexer_prefill_score + _topk: MiniMax-M3 MSA indexer"
            " prefill path (SM100 tensor-core causal block-max scoring, ragged"
            " Q, init/local forcing, top-k); ref (f32 causal block-max oracle +"
            " planted-key causality probe) + validity contract + schedule"
        ),
        default_oracle="ref",
    ),
    "oob_canary": FuzzTarget(
        name="oob_canary",
        bazel_target=("//max/kernels/test/gpu/fuzz:fuzz_oob_canary.mojo.test"),
        binary="bazel-bin/max/kernels/test/gpu/fuzz/fuzz_oob_canary.mojo.test",
        description=(
            "Deliberate OOB-write canary -- positive control for the oracles"
        ),
        default_oracle="redzone",
    ),
}


# ===----------------------------------------------------------------------=== #
# Verdicts + results (mirrors llm_fuzz)
# ===----------------------------------------------------------------------=== #


class Verdict(str, enum.Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    HANG = "HANG"
    ERROR = "ERROR"


@dataclasses.dataclass
class CaseResult:
    target: str
    oracle: str
    spec: dict[str, int]
    verdict: Verdict
    elapsed_s: float
    exit_code: int | None
    detail: str = ""
    findings: list[str] = dataclasses.field(default_factory=list)

    def to_record(self) -> dict[str, object]:
        return {
            "target": self.target,
            "oracle": self.oracle,
            "spec": self.spec,
            "verdict": self.verdict.value,
            "elapsed_s": round(self.elapsed_s, 3),
            "exit_code": self.exit_code,
            "detail": self.detail,
            "findings": self.findings,
        }


# Markers that indicate a real memory-safety finding in tool output. Kept in
# sync with run_sanitizer.sh; deliberately precise (never the bare "ERROR
# SUMMARY: N" count, which also tallies internal-sanitizer noise).
_FINDING_MARKERS = re.compile(
    r"Invalid __(global|shared|local|device)__"
    r"|Race reported"
    r"|Barrier error"
    r"|Uninitialized __global__"
    r"|misaligned address"
    r"|is out of bounds"
    r"|MemoryManager detected a device buffer (under|over)flow"
    r"|CUDA_EXCEPTION"
    r"|illegal memory access"
    r"|FUZZ_NUMERIC_FAIL"
    r"|FUZZ_CONTRACT_FAIL"
)


# ===----------------------------------------------------------------------=== #
# Oracle configuration: how each oracle wraps/env-decorates a single case
# ===----------------------------------------------------------------------=== #


def _compute_sanitizer_path() -> str:
    return (
        shutil.which("compute-sanitizer")
        or "/usr/local/cuda/bin/compute-sanitizer"
    )


def _oracle_command_and_env(
    oracle: str, base_cmd: list[str], gpu: int
) -> tuple[list[str], dict[str, str]]:
    """Return the (command, extra-env) for running one case under `oracle`.

    diff        -- run as-is; catches hangs (timeout) and crashes (exit code).
    ref         -- numerical correctness: run with `--check 1` so the target
                 compares its output to a higher-precision CPU reference and
                 emits FUZZ_NUMERIC_FAIL on a wrong answer. No sanitizer.
    determinism -- run-to-run bit-stability: `--rerun 8` re-runs the same input
                 N times (no forced split-K) and flags any non-bit-exact output.
    batch_invariance -- `--batch-invariance 1`: run a probe under two different
                 co-batch compositions and flag if the probe's output changes.
    batch_variance -- `--batch-variance 1`: the negative control (inverse of
                 batch_invariance) -- run the probe across a batch composition
                 that straddles an M-keyed dispatch breakpoint (dense matmul
                 M=1 GEMV vs M>1 tile GEMM; attention default partition
                 heuristic) and require the probe's output to DIVERGE. Proves
                 the batch_invariance oracle has teeth; a bit-match is the FAIL.
    redzone   -- MAX redzone allocator: catches OOB *writes* at free (~native).
    poison    -- MAX poison allocator (`poison-all`) + `--check`: every device
                 allocation is NaN-filled, so an unwritten/uninitialized output
                 survives as NaN and the reference comparison flags it. No
                 sanitizer, no kernel instrumentation.
    memcheck  -- compute-sanitizer memcheck + device pool disabled so small OOB
                 reads/writes are not masked by the caching allocator.
    initcheck -- compute-sanitizer initcheck + pool disabled.
    racecheck -- compute-sanitizer racecheck (pool-independent).
    synccheck -- compute-sanitizer synccheck (pool-independent).
    """
    # Set every vendor's device-visibility var; the runtime that's actually
    # present reads its own and ignores the others, so this pins the device
    # correctly on both NVIDIA and AMD without needing to detect which vendor
    # built the target.
    env: dict[str, str] = {
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "HIP_VISIBLE_DEVICES": str(gpu),
        "ROCR_VISIBLE_DEVICES": str(gpu),
    }
    cs = _compute_sanitizer_path()
    cs_common = [
        cs,
        "--target-processes",
        "all",
        "--launch-timeout",
        "0",
        "--error-exitcode",
        "1",
    ]
    # Disabling the device caching allocator so the sanitizer sees true
    # per-buffer bounds. Running the binary directly (not via `bazel test`)
    # means the rule env does not pin these, so the process env wins.
    pool_off = {
        "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE": "0",
        "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_ONLY": "false",
    }

    if oracle == "diff":
        return base_cmd, env
    if oracle == "ref":
        return base_cmd + ["--check", "1"], env
    if oracle == "schedule":
        # Schedule amplification: the target re-runs the same input across a
        # varied launch decomposition (e.g. forced split-K) N times and checks
        # the output is bit-stable; divergence = an inter-block race.
        return base_cmd + ["--schedule", "8"], env
    if oracle == "determinism":
        # Run-to-run determinism: the target re-runs the SAME input N times and
        # checks the output is bit-stable (atol=rtol=0). Like `schedule` but
        # WITHOUT forcing a split-K decomposition -- it exercises the kernel's
        # default launch, catching races / order-dependent atomics that show up
        # without any forced amplification.
        return base_cmd + ["--rerun", "8"], env
    if oracle == "batch_invariance":
        # Batch invariance: the target runs a probe token under two different
        # co-batch compositions (different filler count / distribution / expert
        # set for the OTHER tokens; same expert + same input for the probe) and
        # checks the probe's output rows are bit-identical (atol=rtol=0). A
        # difference means a dispatch/reduction path keyed on the batch.
        return base_cmd + ["--batch-invariance", "1"], env
    if oracle == "batch_variance":
        # Negative control (the inverse of batch_invariance): the target runs
        # the SAME probe row in two batches whose composition straddles an
        # M-keyed dispatch breakpoint (dense matmul M=1 GEMV_SPLIT_K vs M>1 tile
        # GEMM; attention decode's default partition heuristic, which keys the
        # partition count on the batch size) and asserts the probe's output rows
        # DIVERGE bit-for-bit -- PASS iff divergence is observed, FAIL if they
        # bit-match (the control lost its teeth / the path became invariant, or
        # the heuristic collapsed to one count for the shapes). This proves the
        # invariance oracles above have real sensitivity to a dispatch switch,
        # rather than passing vacuously. A bit-match is reported as a finding,
        # not silently swallowed.
        return base_cmd + ["--batch-variance", "1"], env
    if oracle == "contract":
        # Special-value contract: the target injects NaN/Inf/large inputs and
        # checks a finiteness/propagation contract (not a tolerance diff).
        return base_cmd + ["--contract", "1"], env
    if oracle == "redzone":
        env["MODULAR_DEBUG_DEVICE_ALLOCATOR"] = "out-of-bounds"
        return base_cmd, env
    if oracle == "poison":
        # The debug allocator exposes two complementary poison tiers (see
        # MLRT/docs/Driver/MemoryManagerOverview.md): `poison-all` fills *every*
        # memory-manager allocation with a raw NaN byte (0xFF default) at the
        # allocator layer, while `uninitialized-poison` fills only graph-driver
        # tensors (`createDeviceMemory`) with a type-aware, non-NaN sentinel
        # detected by an instrumented Mojo load check. The fuzz targets
        # allocate device buffers directly via
        # `DeviceContext.enqueue_create_buffer` (the allocator layer, not the
        # graph driver), so `poison-all` is the tier that covers them. The
        # bare `poison` token (the old value) matches neither tier and silently
        # no-ops.
        #
        # `--check` is what observes the poison: poison-all NaN-fills each
        # output before the kernel runs, then the reference comparison flags a
        # surviving NaN (an unwritten output element / uninitialized read) as a
        # numeric mismatch. Without it the case would run but never inspect the
        # poisoned result.
        env["MODULAR_DEBUG_DEVICE_ALLOCATOR"] = "poison-all"
        return base_cmd + ["--check", "1"], env
    if oracle == "memcheck":
        env.update(pool_off)
        cmd = cs_common + ["--tool", "memcheck", "--leak-check", "no"]
        return cmd + base_cmd, env
    if oracle == "initcheck":
        env.update(pool_off)
        return cs_common + ["--tool", "initcheck"] + base_cmd, env
    if oracle == "racecheck":
        cmd = cs_common + ["--tool", "racecheck", "--racecheck-report", "all"]
        return cmd + base_cmd, env
    if oracle == "synccheck":
        return cs_common + ["--tool", "synccheck"] + base_cmd, env
    raise ValueError(f"unknown oracle: {oracle}")


# ===----------------------------------------------------------------------=== #
# Build / enumerate / run
# ===----------------------------------------------------------------------=== #


def _repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(out.stdout.strip())


def build_target(root: Path, target: FuzzTarget, line_info: bool) -> None:
    cmd = [
        "./bazelw",
        "build",
        target.bazel_target,
        "--curses=no",
        "--noshow_progress",
    ]
    if line_info:
        # Mojo-source attribution for compute-sanitizer findings.
        cmd += ["--mojocopt=--debug-level", "--mojocopt=line-tables"]
    print(f"[build] {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, cwd=root, check=True)


def list_specs(
    root: Path, target: FuzzTarget, seed: int, budget: int
) -> list[dict[str, int]]:
    out = subprocess.run(
        [
            target.binary,
            "--mode",
            "list-specs",
            "--seed",
            str(seed),
            "--budget",
            str(budget),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    specs: list[dict[str, int]] = []
    for line in out.stdout.splitlines():
        if not line.startswith("FUZZ_SPEC"):
            continue
        kv = dict(re.findall(r"(\w+)=\s*(-?\d+)", line))
        kv.pop("idx", None)  # generator index, not a spec field
        specs.append({k: int(v) for k, v in kv.items()})
    return specs


def run_case(
    root: Path,
    target: FuzzTarget,
    spec: dict[str, int],
    oracle: str,
    gpu: int,
    timeout_s: float,
) -> CaseResult:
    # Generic: pass each spec field as `--<key> <value>`; the Mojo target reads
    # them by name. Spec keys == FUZZ_SPEC keys == the target's single-mode flags.
    base_cmd = [target.binary, "--mode", "single"]
    for key, value in spec.items():
        base_cmd += [f"--{key}", str(value)]
    cmd, extra_env = _oracle_command_and_env(oracle, base_cmd, gpu)
    env = {**os.environ, **extra_env}

    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - start
        out = _as_text(exc.stdout) + _as_text(exc.stderr)
        return CaseResult(
            target=target.name,
            oracle=oracle,
            spec=spec,
            verdict=Verdict.HANG,
            elapsed_s=elapsed,
            exit_code=None,
            detail=f"timed out after {timeout_s}s",
            findings=_extract_findings(out),
        )

    elapsed = time.monotonic() - start
    out = proc.stdout + proc.stderr
    findings = _extract_findings(out)

    if findings:
        verdict = Verdict.FAIL
        detail = findings[0]
    elif proc.returncode != 0:
        verdict = Verdict.FAIL
        detail = f"nonzero exit {proc.returncode}"
    elif "FUZZ_RESULT verdict=PASS" in out:
        verdict = Verdict.PASS
        detail = ""
    else:
        verdict = Verdict.ERROR
        detail = "no PASS marker and no finding (unexpected output)"

    return CaseResult(
        target=target.name,
        oracle=oracle,
        spec=spec,
        verdict=verdict,
        elapsed_s=elapsed,
        exit_code=proc.returncode,
        detail=detail,
        findings=findings,
    )


def _as_text(x: object) -> str:
    """Coerce captured output to str (TimeoutExpired may carry bytes)."""
    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode("utf-8", "replace")
    assert isinstance(x, str)
    return x


_FINDING_KINDS = (
    "invalid __global__ write",
    "invalid __global__ read",
    "invalid __shared__",
    "invalid __local__",
    "race reported",
    "barrier error",
    "uninitialized __global__",
    "memorymanager detected a device buffer overflow",
    "memorymanager detected a device buffer underflow",
    "misaligned address",
    "illegal memory access",
    "is out of bounds",
    "fuzz_numeric_fail",
    "fuzz_contract_fail",
)


def _finding_kind(findings: list[str]) -> str:
    """Normalize the first finding to a stable bug 'kind' (no addresses).

    Used to keep shrinking verdict- AND bug-preserving: a FAIL caused by an OOB
    finding must not be 'minimized' into a FAIL caused by an unrelated crash on
    a degenerate input.
    """
    if not findings:
        return ""
    low = findings[0].lower()
    for kw in _FINDING_KINDS:
        if kw in low:
            return kw
    return "other-finding"


def _extract_findings(output: str) -> list[str]:
    seen: list[str] = []
    for line in output.splitlines():
        if (
            _FINDING_MARKERS.search(line)
            and "ERROR SUMMARY: 0 errors" not in line
        ):
            stripped = line.strip()
            if stripped not in seen:
                seen.append(stripped)
            if len(seen) >= 8:
                break
    return seen


# ===----------------------------------------------------------------------=== #
# Shrinking: minimize a failing spec to the smallest case with the same verdict
# ===----------------------------------------------------------------------=== #


def shrink(
    root: Path,
    target: FuzzTarget,
    spec: dict[str, int],
    oracle: str,
    gpu: int,
    timeout_s: float,
    target_verdict: Verdict,
    target_kind: str,
    max_rounds: int = 4,
) -> tuple[dict[str, int], CaseResult | None]:
    """Greedy coordinate-descent shrink: reduce each spec field to the smallest
    of {0, 1, half} that still reproduces the SAME bug, looping to a fixpoint.
    Returns (minimal_spec, last_reproducing_result-or-None).

    Bug-preserving: a reduction is accepted only if its verdict equals
    `target_verdict` AND its finding kind equals `target_kind` (so an OOB FAIL
    is not 'minimized' into an unrelated crash on a degenerate input).
    """
    current = dict(spec)
    last: CaseResult | None = None
    for _ in range(max_rounds):
        improved = False
        for key in sorted(current):
            candidates = sorted(
                {c for c in (0, 1, current[key] // 2) if c < current[key]}
            )
            for cand in candidates:
                trial = dict(current)
                trial[key] = cand
                res = run_case(root, target, trial, oracle, gpu, timeout_s)
                if (
                    res.verdict == target_verdict
                    and _finding_kind(res.findings) == target_kind
                ):
                    current[key] = cand
                    last = res
                    improved = True
                    break  # take the smallest reproducing value for this field
        if not improved:
            break
    return current, last


# ===----------------------------------------------------------------------=== #
# Corpus
# ===----------------------------------------------------------------------=== #


def _signature(target: str, spec: dict[str, int], verdict: Verdict) -> str:
    parts = "_".join(f"{k}{spec[k]}" for k in sorted(spec))
    return f"{target}_{verdict.value.lower()}_{parts}"


def write_corpus_entry(corpus_dir: Path, result: CaseResult) -> Path:
    tgt_dir = corpus_dir / result.target
    tgt_dir.mkdir(parents=True, exist_ok=True)
    sig = _signature(result.target, result.spec, result.verdict)
    path = tgt_dir / f"{sig}.json"
    path.write_text(
        json.dumps(
            {
                "target": result.target,
                "spec": result.spec,
                "verdict": result.verdict.value,
                "oracle": result.oracle,
                "detail": result.detail,
                "findings": result.findings,
            },
            indent=2,
        )
        + "\n"
    )
    return path


def replay_corpus(
    root: Path,
    corpus_root: Path,
    only_target: str | None,
    gpu: int,
    timeout_s: float,
    do_build: bool,
) -> int:
    """Re-run every corpus entry under its recorded oracle and check the verdict
    is unchanged. Returns 1 if any entry's verdict drifted (a regression, a
    fixed bug whose corpus entry needs updating, or a broken oracle), else 0.

    This is the deterministic, fast gate: same seed/spec -> same verdict.
    """
    loaded: list[tuple[Path, dict[str, object]]] = []
    targets_needed: set[str] = set()
    for path in sorted(corpus_root.glob("*/*.json")):
        data = json.loads(path.read_text())
        if only_target and data["target"] != only_target:
            continue
        loaded.append((path, data))
        targets_needed.add(str(data["target"]))

    if not loaded:
        print("[replay] no corpus entries found")
        return 0

    if do_build:
        for tname in sorted(targets_needed):
            build_target(root, _TARGETS[tname], line_info=False)

    mismatches = 0
    for _path, data in loaded:
        tgt = _TARGETS[str(data["target"])]
        spec_data = data["spec"]
        assert isinstance(spec_data, dict)
        spec = {str(k): int(v) for k, v in spec_data.items()}
        oracle = str(data["oracle"])
        expected = str(data["verdict"])
        res = run_case(root, tgt, spec, oracle, gpu, timeout_s)
        ok = res.verdict.value == expected
        mismatches += 0 if ok else 1
        print(
            f"[replay] {'OK' if ok else 'MISMATCH':<8} {tgt.name} {spec} "
            f"oracle={oracle} expected={expected} got={res.verdict.value}"
        )
    print(f"[replay] {len(loaded)} entries, {mismatches} mismatch(es)")
    return 1 if mismatches else 0


# ===----------------------------------------------------------------------=== #
# CLI
# ===----------------------------------------------------------------------=== #


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--target",
        default=None,
        choices=sorted(_TARGETS),
        help="Which kernel fuzz target to run (default: mha_causal; in "
        "--replay-corpus mode, an optional filter, default all targets).",
    )
    p.add_argument(
        "--replay-corpus",
        action="store_true",
        help="Re-run every corpus entry and check its verdict is stable "
        "(the deterministic regression gate). Ignores --seed/--budget/--spec.",
    )
    p.add_argument(
        "--oracle",
        default=None,
        choices=[
            "diff",
            "ref",
            "schedule",
            "determinism",
            "batch_invariance",
            "batch_variance",
            "contract",
            "redzone",
            "poison",
            "memcheck",
            "initcheck",
            "racecheck",
            "synccheck",
        ],
        help="Bug oracle (default: the target's default_oracle).",
    )
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--budget", type=int, default=32, help="Number of cases.")
    p.add_argument(
        "--spec",
        action="append",
        default=[],
        metavar="SL,NK,VL",
        help=(
            "Explicit 'seq_len,num_keys,valid_length' case (repeatable); when "
            "given, skips generation and runs exactly these specs. Used for "
            "deterministic repro and corpus replay."
        ),
    )
    p.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device index (NVIDIA or AMD).",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-case wall-clock timeout (s); exceeding it = HANG.",
    )
    p.add_argument("--no-build", action="store_true")
    p.add_argument(
        "--no-shrink",
        action="store_true",
        help="Skip minimizing each failing spec to a minimal repro.",
    )
    p.add_argument(
        "--out-dir",
        default="max/kernels/test/gpu/fuzz/.fuzzruns",
        help="Where to write the JSONL run log (relative to repo root).",
    )
    p.add_argument(
        "--corpus-dir",
        default="max/kernels/test/gpu/fuzz/corpus",
        help="Where to persist failing specs (relative to repo root).",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    root = _repo_root()

    if args.replay_corpus:
        return replay_corpus(
            root,
            root / args.corpus_dir,
            args.target,  # None => replay all targets
            args.gpu,
            args.timeout,
            do_build=not args.no_build,
        )

    target = _TARGETS[args.target or "mha_causal"]
    oracle = args.oracle or target.default_oracle

    if not args.no_build:
        build_target(
            root, target, line_info=(oracle in ("memcheck", "initcheck"))
        )

    if args.spec:
        specs = []
        for s in args.spec:
            spec: dict[str, int] = {}
            for pair in s.split(","):
                k, v = pair.split("=")
                spec[k.strip()] = int(v)
            specs.append(spec)
        source = "explicit --spec"
    else:
        specs = list_specs(root, target, args.seed, args.budget)
        source = f"seed={args.seed}"
    print(
        f"[fuzz] target={target.name} oracle={oracle} {source} "
        f"budget={len(specs)} timeout={args.timeout}s",
        file=sys.stderr,
    )

    run_dir = root / args.out_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"run-{int(time.time())}-{target.name}-{oracle}.jsonl"
    corpus_dir = root / args.corpus_dir

    results: list[CaseResult] = []
    counts: dict[str, int] = {v.value: 0 for v in Verdict}
    with log_path.open("w") as log:
        log.write(
            json.dumps(
                {
                    "event": "run_start",
                    "target": target.name,
                    "oracle": oracle,
                    "seed": args.seed,
                    "budget": len(specs),
                }
            )
            + "\n"
        )
        for i, spec in enumerate(specs):
            res = run_case(root, target, spec, oracle, args.gpu, args.timeout)
            results.append(res)
            counts[res.verdict.value] += 1
            rec = res.to_record()
            rec["event"] = "case"
            rec["idx"] = i
            log.write(json.dumps(rec) + "\n")
            log.flush()
            marker = "" if res.verdict == Verdict.PASS else "  <-- "
            spec_str = " ".join(f"{k}={v}" for k, v in spec.items())
            print(
                f"[{i:>3}] {res.verdict.value:<11} {spec_str} "
                f"({res.elapsed_s:.1f}s) {marker}{res.detail}"
            )
            if res.verdict != Verdict.PASS:
                final = res
                if not args.no_shrink and len(spec) > 0:
                    print(f"      shrinking {res.verdict.value} ...")
                    min_spec, min_res = shrink(
                        root,
                        target,
                        spec,
                        oracle,
                        args.gpu,
                        args.timeout,
                        res.verdict,
                        _finding_kind(res.findings),
                    )
                    if min_res is not None and min_spec != spec:
                        final = min_res
                        log.write(
                            json.dumps(
                                {
                                    "event": "shrink",
                                    "idx": i,
                                    "from": spec,
                                    "to": min_spec,
                                    "verdict": final.verdict.value,
                                }
                            )
                            + "\n"
                        )
                        log.flush()
                        print(
                            "      minimal: "
                            + " ".join(f"{k}={v}" for k, v in min_spec.items())
                        )
                cpath = write_corpus_entry(corpus_dir, final)
                print(f"      corpus: {cpath.relative_to(root)}")
        log.write(json.dumps({"event": "run_end", "counts": counts}) + "\n")

    n_bad = sum(c for v, c in counts.items() if v != "PASS")
    print(
        "\n[fuzz] done: "
        + " ".join(f"{v}={counts[v]}" for v in counts)
        + f"\n[fuzz] log: {log_path.relative_to(root)}"
    )
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
