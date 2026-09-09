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
"""Device-timed TFLOPS bench for `MlaPrefillV2` (fresh reference MLA port).

Targets the `MlaPrefillV2` struct (`mla_prefill_v2.mojo`). The
buffer-setup recipe is `CacheBustingBuffer` for Q / latent-K / V / O with
per-iter offsets to defeat L2 reuse, and the FLOPs formula below, so the
TFLOPS number is directly comparable across runs on the same node at a
fixed shape.

Timing is **device time**: `bencher_iter_custom()` forwards to
`DeviceContext.execution_time()`, which brackets the launches with GPU events
(NOT wall-clock around `enqueue_function()`). This is the trustworthy path — a
prior coarse harness that wall-clocked the launch produced non-physical TFLOPS.

`MlaPrefillV2` delegates every numeric step to `MlaPrefillV2Core[config]`'s
FP32-scores / reference-cadence path, gated behind
`MlaPrefillV2Core._FP32_SOFTMAX_SCORES` (default OFF in shipping source). The
BUILD target compiles this bench with:

    -D fp32_scores=true -D cadence=true

so that gated path is enabled for THIS compilation only — neither
`mla_prefill_v2.mojo` nor `mla_components.mojo` (nor any shared
file) is modified; all stay byte-identical to HEAD. Without those defines the
kernel will not compile (`_SOFTMAX_DTYPE` resolves to FP16 and the
delegated in-place-FP32 helpers are not wired). `MlaPrefillV2`
*only* supports the FP8 / KV>=128 / 32x32x64 shape (the reference integrated
cadence target), so the bench fixes `dtype=float8_e4m3fn`, `kv_block=128`.

Shape (DeepSeek-V3 MLA archetype, DSV-TP4 per-device shard — matches the
`MlaPrefillV2` Phase-1 correctness test):

- Q: `(batch, seq_len, num_heads, d_qk)` at `dtype`. `d_qk = 192`
  (`d_nope=128` + `d_rope=64`).
- Latent KV cache: `(batch, cache_seq_len, num_kv_heads, cache_depth)`
  at `dtype`. `cache_depth = 576` with `k_nope` at `[:, :128]` and
  `k_rope` at `[:, 512:576]`. `num_kv_heads = 1` (DSV MLA at TP=4 —
  one latent KV head, GQA group = num_heads).
- Output: `(batch, seq_len, num_heads, d_pv)` at BF16. `d_pv = 128`.

Mask: `NullMask` (full attention; matches the MlaPrefillV2 test's
must-pass NullMask KV=128 case).

FLOPs formula:

    flops = 2 * B * H * N * (NK * d_qk + NK * d_pv)

QK at `d_qk = 192` and PV at `d_pv = 128` summed over the full key
sequence (NullMask — full attention, not halved for causal).

Run
---

```bash
./bazelw run //max/kernels/benchmarks:gpu/nn/bench_mla_prefill_v2 \
    -- seq_len=8192 batch_size=1
```

Defaults: FP8 e4m3fn, KV=128, seq_len=8192, num_heads=32, num_kv_heads=1
(DSV-TP4, GQA group=32), q_block_size=32, cache_seq_len=seq_len
(self-attention). The reference target at this shape is ~1666 TFLOPS.
"""

from std.math import ceildiv
from std.random import seed
from std.sys import (
    get_defined_bool,
    get_defined_dtype,
    get_defined_int,
    get_defined_string,
    size_of,
)

from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu import *
from max.gpu.host import DeviceContext
from std.utils import StaticTuple

from internal_utils import CacheBustingBuffer, arg_parse
from internal_utils._utils import InitializationType
from layout import Idx, LayoutTensor, TileTensor, row_major
from layout.coord import Coord
from layout.runtime_layout import RuntimeLayout

from nn.attention.mha_mask import CausalMask, MHAMask, NullMask
from nn.attention.mha_operand import LayoutTensorMHAOperand, MHAOperand

from nn.attention.gpu.amd_structured.mha_mma_op import MlaConfigV2
from nn.attention.gpu.amd_structured.mla_prefill_v2 import MlaPrefillV2
from nn.attention.gpu.amd_structured.ps_metadata import build_uniform


comptime _Q_BLOCK_SIZE = 32
comptime _NUM_WARPS = 8
comptime _BM = _NUM_WARPS * _Q_BLOCK_SIZE  # 256

# DeepSeek-V3 MLA shape constants. These are not knobs — the latent
# cache geometry is fixed by the architecture and must match
# `mla_prefill.mojo` (cache_depth=576, rope segment at offset 512).
comptime _D_NOPE = 128
comptime _D_ROPE = 64
comptime _D_QK = _D_NOPE + _D_ROPE  # 192
comptime _CACHE_DEPTH = 576
comptime _ROPE_CACHE_OFFSET = 512  # = cache_depth - d_rope

# Per-kernel LLVM tuning forwarded as `-mllvm` flags via the launcher's
# `compile_options` param:
# - `amdgpu-igrouplp-exact-solver=true`: enable exponential solver.
# - `...-max-branches=10000`: cap branches (avoid exponential blowup).
# - `...-cost-heur=false`: node-order priority (over-declared hints fit
#   better).
# `exact_solver` (default 1 = the hardcoded exact-solver opts above):
# set `-D exact_solver=0` to drop the exponential IGLP solver and let
# the heuristic (greedy) IGLP scheduler run. The exact solver was found to
# front-load a 32-`ds_read` burst in the fused steady-state block; the
# greedy scheduler may stream the loads instead.
comptime _EXACT_SOLVER = get_defined_int["exact_solver", 1]()
# `-D llvm_preset=N` appends an LLVM-backend scheduler flag for the
# reference-gap experiments. Full-literal presets (no comptime StaticString
# concat). 0 = prior behavior. 1 = mfma-vgpr-form=false (o-acc -> AGPR, frees
# ~64 arch-VGPR). 2 = disable load-sinking UnclusteredHighRP reschedule.
# 3 = amdgpu RP trackers. 4 = metric-bias=0. 6 = mfma-vgpr-form WITHOUT exact
# solver.
comptime _LLVM_PRESET = get_defined_int["llvm_preset", 0]()
# `-D bench_null_mask=true` runs the kernel under `NullMask` (full
# attention) instead of the default `CausalMask`. The steady-state K/V
# `ds_read` cadence is mask-independent (the mask only gates score VALU),
# so this exists to report BOTH mask perf numbers for the cadence A/B; the
# FLOP convention is full-attention in both cases (see `compute_flops`).
comptime _NULL_MASK = get_defined_bool["bench_null_mask", False]()
comptime _PREFILL_IGLP_OPTS: StaticString = (
    "" if (
        _EXACT_SOLVER == 0 and _LLVM_PRESET == 0
    ) else "amdgpu-igrouplp-exact-solver=true,amdgpu-igrouplp-exact-solver-max-branches=10000,amdgpu-igrouplp-exact-solver-cost-heur=false" if _LLVM_PRESET
    == 0 else "amdgpu-igrouplp-exact-solver=true,amdgpu-igrouplp-exact-solver-max-branches=10000,amdgpu-igrouplp-exact-solver-cost-heur=false,amdgpu-mfma-vgpr-form=false" if _LLVM_PRESET
    == 1 else "amdgpu-igrouplp-exact-solver=true,amdgpu-igrouplp-exact-solver-max-branches=10000,amdgpu-igrouplp-exact-solver-cost-heur=false,amdgpu-disable-unclustered-high-rp-reschedule=true" if _LLVM_PRESET
    == 2 else "amdgpu-igrouplp-exact-solver=true,amdgpu-igrouplp-exact-solver-max-branches=10000,amdgpu-igrouplp-exact-solver-cost-heur=false,amdgpu-use-amdgpu-trackers=true" if _LLVM_PRESET
    == 3 else "amdgpu-igrouplp-exact-solver=true,amdgpu-igrouplp-exact-solver-max-branches=10000,amdgpu-igrouplp-exact-solver-cost-heur=false,amdgpu-schedule-metric-bias=0" if _LLVM_PRESET
    == 4 else "amdgpu-mfma-vgpr-form=false" if _LLVM_PRESET
    == 6 else "amdgpu-igrouplp-exact-solver=true,amdgpu-igrouplp-exact-solver-max-branches=10000,amdgpu-igrouplp-exact-solver-cost-heur=false"
)


def run_mla_prefill_v2[
    qkv_type: DType,
    out_type: DType,
    num_heads: Int,
    num_kv_heads: Int,
    kv_block: Int,
    mask_t: MHAMask,
    q_block_size: Int = _Q_BLOCK_SIZE,
    cache_busting: Bool = True,
](
    mut m: Bench,
    mask: mask_t,
    seq_len: Int,
    cache_seq_len: Int,
    batch_size: Int,
    bench: Bool,
    ctx: DeviceContext,
) raises:
    """Run `MlaPrefillV2` over `CacheBustingBuffer`-backed Q/K/V/O.

    `cache_seq_len` is the K/V sequence length materialized in the latent
    cache (>= `seq_len`); for self-attention `cache_seq_len == seq_len`.
    `num_keys` passed to the kernel is `cache_seq_len` and `start_pos` is
    `0` (prefill-from-scratch).
    """
    comptime scale = Float32(0.125)  # ~rsqrt(64); the reference is
    # depth-agnostic for the scale param. Matches the `bench_mla.mojo` bench.
    comptime assert (
        num_heads % num_kv_heads == 0
    ), "num_heads must be a multiple of num_kv_heads"

    var q_size = batch_size * num_heads * seq_len * _D_QK
    var k_cache_size = batch_size * num_kv_heads * cache_seq_len * _CACHE_DEPTH
    var o_size = batch_size * num_heads * seq_len * _D_NOPE

    # Cache busting: oversize allocation, defeat L2 reuse across iters.
    comptime simd_size = 4
    var cb_q = CacheBustingBuffer[qkv_type](
        q_size, simd_size, ctx, cache_busting
    )
    # Single latent cache buffer backs both k_nope and k_rope (the kernel
    # selects the column range via `head_dim_idx` at the per-tile DMA).
    var cb_k = CacheBustingBuffer[qkv_type](
        k_cache_size, simd_size, ctx, cache_busting
    )
    # V is the K_nope segment of the latent cache in production. Separate
    # `cb_v` buffer so the V tile's HBM footprint isn't covered by K's L2
    # residency (matches the `bench_mla.mojo` bench).
    var cb_v = CacheBustingBuffer[qkv_type](
        batch_size * num_kv_heads * cache_seq_len * _D_NOPE,
        simd_size,
        ctx,
        cache_busting,
    )
    var cb_o = CacheBustingBuffer[out_type](
        o_size, simd_size, ctx, cache_busting
    )

    comptime random_distribution = InitializationType.uniform_distribution
    cb_q.init_on_device(random_distribution, ctx)
    cb_k.init_on_device(random_distribution, ctx)
    cb_v.init_on_device(random_distribution, ctx)

    comptime _config = MlaConfigV2(
        q_block_size=q_block_size,
        kv_block=kv_block,
        depth=_D_NOPE,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        d_qk=_D_QK,
        d_rope=_D_ROPE,
        cache_depth=_CACHE_DEPTH,
        rope_cache_offset=_ROPE_CACHE_OFFSET,
        num_warps=_NUM_WARPS,
        dtype=qkv_type,
        output_dtype=out_type,
    )

    if bench:
        # ---- Hoisted compile: compile the kernel ONCE here, not inside the
        # timed `_kernel_launch` closure (which `iter_custom` calls ~100x).
        # Build a representative operand set (iteration 0) purely to form the
        # comptime `kernel_run` signature, compile + dump here, and let the
        # timed closure below only `enqueue_function` the cached handle. This
        # makes "compiled exactly once" structural rather than relying on
        # `compile_function`'s cache. ----
        comptime _kernel = MlaPrefillV2[_config]
        var _q0 = TileTensor(
            cb_q.offset_ptr(0).bitcast[Scalar[qkv_type]](),
            row_major(
                Coord(
                    Int32(batch_size),
                    Int32(seq_len),
                    Idx[num_heads],
                    Idx[_D_QK],
                )
            ),
        )
        var _o0 = TileTensor(
            cb_o.offset_ptr(0).bitcast[Scalar[out_type]](),
            row_major(
                Coord(
                    Int32(batch_size),
                    Int32(seq_len),
                    Idx[num_heads],
                    Idx[_D_NOPE],
                )
            ),
        )
        var _k0 = TileTensor(
            cb_k.offset_ptr(0).bitcast[Scalar[qkv_type]](),
            row_major(
                Coord(
                    Int32(batch_size),
                    Int32(cache_seq_len),
                    Idx[num_kv_heads],
                    Idx[_CACHE_DEPTH],
                )
            ),
        )
        var _knope0 = LayoutTensorMHAOperand(
            _k0.as_immut().as_unsafe_any_origin()
        )
        var _krope0 = LayoutTensorMHAOperand(
            _k0.as_immut().as_unsafe_any_origin()
        )
        var _v0 = TileTensor(
            cb_v.offset_ptr(0).bitcast[Scalar[qkv_type]](),
            row_major(
                Coord(
                    Int32(batch_size),
                    Int32(cache_seq_len),
                    Idx[num_kv_heads],
                    Idx[_D_NOPE],
                )
            ),
        )
        var _vop0 = LayoutTensorMHAOperand(
            _v0.as_immut().as_unsafe_any_origin()
        )
        comptime _kernel_run = _kernel.run[
            type_of(_knope0),
            type_of(_krope0),
            type_of(_vop0),
            mask_t,
            _q0.dtype,
            _o0.dtype,
            _q0.LayoutType,
            _o0.LayoutType,
            _q0.Engine,
            _o0.Engine,
            ragged=False,
        ]
        comptime _DUMP_ASM: StaticString = get_defined_string["dump_asm", ""]()
        var compiled = ctx.compile_function[
            _kernel_run,
            compile_options=_PREFILL_IGLP_OPTS,
            dump_asm=_DUMP_ASM,
        ]()
        # The representative operands existed only to form `kernel_run`;
        # discard their runtime values (their TYPES were used at comptime).
        _ = _q0
        _ = _o0
        _ = _knope0
        _ = _krope0
        _ = _vop0

        # ---- Persistent work partition (host-built, uploaded once).
        # With `-D persistent=true` the kernel consumes it via the
        # (num_cu,) grid; otherwise it is threaded-but-unused (static grid).
        # `available_tgs = 256` = MI355X CU count (the perf target partition).
        # Built once here, NOT per timed iteration. ----
        comptime _PERSISTENT = get_defined_bool["persistent", False]()
        var md = build_uniform(
            batch_size, Int32(seq_len), Int32(num_heads), 256
        )
        var num_cu = len(md.work_indptr) - 1
        var n_indptr = len(md.work_indptr)
        var n_info = len(md.work_info)
        var host_work_indptr = ctx.enqueue_create_host_buffer[.int32](n_indptr)
        var host_work_info = ctx.enqueue_create_host_buffer[.int32](n_info)
        ctx.synchronize()
        for i in range(n_indptr):
            host_work_indptr[i] = md.work_indptr[i]
        for i in range(n_info):
            host_work_info[i] = md.work_info[i]
        var dev_work_indptr = ctx.enqueue_create_buffer[.int32](n_indptr)
        var dev_work_info = ctx.enqueue_create_buffer[.int32](n_info)
        ctx.enqueue_copy(dev_work_indptr, host_work_indptr)
        ctx.enqueue_copy(dev_work_info, host_work_info)
        ctx.synchronize()
        var work_indptr_ptr = dev_work_indptr.unsafe_ptr()
        var work_info_ptr = dev_work_info.unsafe_ptr()
        var num_works = md.num_works

        @always_inline
        def bench_func(
            mut b: Bencher,
        ) raises {
            var cb_q,
            var cb_k,
            var cb_v,
            var cb_o,
            var compiled,
            var mask,
            var work_indptr_ptr,
            var work_info_ptr,
            var num_works,
            var num_cu,
            imm,
        }:
            @always_inline
            def _kernel_launch(ctx: DeviceContext, iteration: Int) raises {imm}:
                var q_ptr = cb_q.offset_ptr(iteration).bitcast[
                    Scalar[qkv_type]
                ]()
                var k_ptr = cb_k.offset_ptr(iteration).bitcast[
                    Scalar[qkv_type]
                ]()
                var v_ptr = cb_v.offset_ptr(iteration).bitcast[
                    Scalar[qkv_type]
                ]()
                var o_ptr = cb_o.offset_ptr(iteration).bitcast[
                    Scalar[out_type]
                ]()

                var q_tt = TileTensor(
                    q_ptr,
                    row_major(
                        Coord(
                            Int32(batch_size),
                            Int32(seq_len),
                            Idx[num_heads],
                            Idx[_D_QK],
                        )
                    ),
                )
                var o_tt = TileTensor(
                    o_ptr,
                    row_major(
                        Coord(
                            Int32(batch_size),
                            Int32(seq_len),
                            Idx[num_heads],
                            Idx[_D_NOPE],
                        )
                    ),
                )

                # Latent K cache TileTensor at cache_depth=576. Two
                # separate `LayoutTensorMHAOperand` values back the same
                # physical buffer; the borrow checker rejects passing one
                # operand SSA value twice through `enqueue_function`, so we
                # materialize two distinct values (same pointer, distinct
                # value-parameters).
                var k_tt = TileTensor(
                    k_ptr,
                    row_major(
                        Coord(
                            Int32(batch_size),
                            Int32(cache_seq_len),
                            Idx[num_kv_heads],
                            Idx[_CACHE_DEPTH],
                        )
                    ),
                )
                var k_nope_op = LayoutTensorMHAOperand(
                    k_tt.as_immut().as_unsafe_any_origin()
                )
                var k_rope_op = LayoutTensorMHAOperand(
                    k_tt.as_immut().as_unsafe_any_origin()
                )

                var v_tt = TileTensor(
                    v_ptr,
                    row_major(
                        Coord(
                            Int32(batch_size),
                            Int32(cache_seq_len),
                            Idx[num_kv_heads],
                            Idx[_D_NOPE],
                        )
                    ),
                )
                var v_op = LayoutTensorMHAOperand(
                    v_tt.as_immut().as_unsafe_any_origin()
                )

                # Enqueue ONLY — the kernel was compiled once above and the
                # handle captured. Persistent grid `(num_cu,)` under
                # `-D persistent=true`; else the static 3D grid.
                var gx: Int
                var gy: Int
                var gz: Int
                comptime if _PERSISTENT:
                    gx = num_cu
                    gy = 1
                    gz = 1
                else:
                    gx = num_heads
                    gy = ceildiv(seq_len, _kernel.BM)
                    gz = batch_size
                ctx.enqueue_function(
                    compiled,
                    q_tt,
                    k_nope_op,
                    k_rope_op,
                    v_op,
                    o_tt,
                    mask,
                    scale,
                    Int32(cache_seq_len),  # num_keys (self-attention)
                    Int32(0),  # start_pos
                    work_indptr_ptr,
                    work_info_ptr,
                    Int32(num_works),
                    grid_dim=(gx, gy, gz),
                    block_dim=_kernel.NUM_THREADS,
                )

            bencher_iter_custom(b, _kernel_launch, ctx)

        def compute_flops() {imm} -> Int:
            # MLA prefill FLOPs (NullMask — full attention, NOT half for
            # causal): one two-segment QK at `d_qk = d_nope + d_rope = 192`
            # plus one PV at `d_pv = d_nope = 128`.
            return (
                2
                * batch_size
                * num_heads
                * seq_len
                * (cache_seq_len * _D_QK + cache_seq_len * _D_NOPE)
            )

        def compute_hbm_bytes() {imm} -> Int:
            # HBM footprint per launch (NullMask, full attention):
            # - Q: B * H * N * d_qk    elts of `qkv_type`
            # - K read: B * H_kv * NK * d_qk    elts of `qkv_type`
            # - V read: B * H_kv * NK * d_pv    elts of `qkv_type`
            # - O write: B * H * N * d_pv      elts of `out_type`
            var qkv_b = size_of[qkv_type]()
            var out_b = size_of[out_type]()
            var q_bytes = batch_size * num_heads * seq_len * _D_QK * qkv_b
            var k_bytes = (
                batch_size * num_kv_heads * cache_seq_len * _D_QK * qkv_b
            )
            var v_bytes = (
                batch_size * num_kv_heads * cache_seq_len * _D_NOPE * qkv_b
            )
            var o_bytes = batch_size * num_heads * seq_len * _D_NOPE * out_b
            return q_bytes + k_bytes + v_bytes + o_bytes

        m.bench_function(
            bench_func,
            BenchId(
                "mla_prefill_v2",
                # fmt: off
                input_id=String(
                    "qkv_type=", qkv_type,
                    "/out_type=", out_type,
                    "/d_qk=", _D_QK,
                    "/d_pv=", _D_NOPE,
                    "/cache_depth=", _CACHE_DEPTH,
                    "/num_heads=", num_heads,
                    "/num_kv_heads=", num_kv_heads,
                    "/kv_block=", kv_block,
                    "/seq_len=", seq_len,
                    "/cache_seq_len=", cache_seq_len,
                    "/batch_size=", batch_size,
                    "/cache_busting=", cache_busting,
                ),
                # fmt: on
            ),
            [
                ThroughputMeasure(BenchMetric.flops, compute_flops()),
                ThroughputMeasure(BenchMetric.bytes, compute_hbm_bytes()),
            ],
        )
        # Keep the uploaded work-partition buffers alive past the timed iters
        # (only their raw ptrs were captured into `_kernel_launch`).
        _ = dev_work_indptr
        _ = dev_work_info
        ctx.synchronize()

    _ = cb_q
    _ = cb_k
    _ = cb_v
    _ = cb_o


@fieldwise_init
struct MlaPrefillV2Cfg(ImplicitlyCopyable, Writable):
    var qkv_type: DType
    var out_type: DType
    var num_heads: Int
    var num_kv_heads: Int
    var kv_block: Int
    var q_block_size: Int
    var cache_busting: Bool


def main() raises:
    # Fixed seed so random Q/K/V fills are reproducible across runs.
    seed(0)

    # `MlaPrefillV2` only supports the FP8 / KV>=128 / 32x32x64 shape.
    # The defaults below pin that shape (FP8 e4m3fn, kv_block=128) at the
    # DSV-TP4 head count (num_heads=32, num_kv_heads=1) — the reference target
    # config and the MlaPrefillV2 Phase-1 correctness shape.
    comptime qkv_type = get_defined_dtype["dtype", .float8_e4m3fn]()
    comptime out_type = get_defined_dtype["out_type", .bfloat16]()
    comptime num_heads = get_defined_int["num_heads", 32]()
    comptime num_kv_heads = get_defined_int["num_kv_heads", 1]()
    comptime kv_block = get_defined_int["kv_block", 128]()
    comptime q_block_size = get_defined_int["q_block_size", _Q_BLOCK_SIZE]()
    comptime cache_busting = get_defined_bool["cache_busting", True]()

    var seq_len = Int(arg_parse("seq_len", 8192))
    var cache_seq_len = Int(arg_parse("cache_seq_len", seq_len))
    var batch_size = Int(arg_parse("batch_size", 1))
    var bench = arg_parse("benchmark", True)

    comptime cfg = MlaPrefillV2Cfg(
        qkv_type=qkv_type,
        out_type=out_type,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        kv_block=kv_block,
        q_block_size=q_block_size,
        cache_busting=cache_busting,
    )

    print("Running MlaPrefillV2 benchmark with config:")
    print("  qkv_type:    ", cfg.qkv_type)
    print("  out_type:    ", cfg.out_type)
    print("  d_qk:        ", _D_QK)
    print("  d_pv:        ", _D_NOPE)
    print("  cache_depth: ", _CACHE_DEPTH)
    print("  num_heads:   ", cfg.num_heads)
    print("  num_kv_heads:", cfg.num_kv_heads)
    print(
        "  group:       ",
        cfg.num_heads // cfg.num_kv_heads,
    )
    print("  kv_block:    ", cfg.kv_block)
    print("  q_block_size:", cfg.q_block_size)
    print("  cache_busting:", cfg.cache_busting)
    print(
        "  seq_len:     ",
        seq_len,
        " cache_seq_len:",
        cache_seq_len,
        " batch_size:",
        batch_size,
    )

    var m = Bench()
    with DeviceContext() as ctx:
        # `-D bench_null_mask=true` -> NullMask (full attention); default
        # CausalMask. `mask_t` is inferred from the runtime mask arg. Both
        # report the full-attention FLOP convention.
        comptime if _NULL_MASK:
            run_mla_prefill_v2[
                cfg.qkv_type,
                cfg.out_type,
                cfg.num_heads,
                cfg.num_kv_heads,
                cfg.kv_block,
                type_of(NullMask()),
                cfg.q_block_size,
                cfg.cache_busting,
            ](m, NullMask(), seq_len, cache_seq_len, batch_size, bench, ctx)
        else:
            run_mla_prefill_v2[
                cfg.qkv_type,
                cfg.out_type,
                cfg.num_heads,
                cfg.num_kv_heads,
                cfg.kv_block,
                type_of(CausalMask()),
                cfg.q_block_size,
                cfg.cache_busting,
            ](m, CausalMask(), seq_len, cache_seq_len, batch_size, bench, ctx)
    m.dump_report()
