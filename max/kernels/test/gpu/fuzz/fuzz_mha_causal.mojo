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
#
# Per-kernel fuzz target: MHA CausalPaddingMask (see gpu-kernels-fuzzing-design.md).
#
# A seeded, boundary-aware shape generator drives flash_attention +
# mha_gpu_naive with a CausalPaddingMask. The oracle is memory safety
# (run under run_sanitizer.sh memcheck / the redzone or poison allocator), so
# input *values* are irrelevant -- only the *shape* and the mask matter.
#
# This target supports three argv modes so the Python orchestrator can drive it
# with per-case timeout + process isolation (a hanging case only kills its own
# subprocess), and so a single build serves generation, single-case execution,
# and standalone fuzzing:
#
#   --mode list-specs --seed S --budget B
#       Print the generated specs (machine-readable `FUZZ_SPEC ...` lines), no
#       GPU work. The orchestrator enumerates these, then runs each via `single`.
#
#   --mode single --seq-len S --num-keys N --valid-length V
#       Run exactly one case (for orchestration, shrinking, and corpus replay).
#       Prints `FUZZ_RESULT verdict=PASS` on success; a hang times out; a crash
#       exits non-zero.
#
#   --mode fuzz --seed S --budget B   (default)
#       Generate + run a batch in-process (standalone convenience). Uses the
#       SAFE spec space (excludes the known num_keys==1 & valid_length==0 hang)
#       so a single in-process run does not wedge; use the orchestrator for the
#       full space.
#
# Compile-time defaults (overridable by argv): `-D fuzz_seed=<n>`,
# `-D budget=<n>`, `-D depth=<n>`.

from std.math import ceildiv, max, min
from std.random import rand, random_ui64, seed
from std.sys.defines import get_defined_int

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from layout import (
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from std.utils.index import IndexList
from nn.attention.gpu.mha import (
    flash_attention,
    get_mha_decoding_num_partitions,
    mha_gpu_naive,
)
from nn.attention.mha_mask import CausalPaddingMask

from _fuzz import boundary_int, collect_args, flag, flag_int, numeric_check


# Fixed configuration. These match a historical OOB-trigger config (bf16, depth=128,
# num_heads=32, group=1) so decode cases land on the SM100 1q path. `depth` is a
# compile-time define because attention kernels specialize on it.
comptime qkv_type = DType.bfloat16
comptime depth = get_defined_int["depth", 128]()
comptime num_heads = 32
comptime group = 1
comptime kv_num_heads = num_heads // group
comptime scale = Float32(0.125)

comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()
comptime TILE = 128  # attention BN boundary -- the interesting modulus.

# Batch-invariance oracle knobs.
comptime BI_PINNED_PARTITIONS = 2  # pinned split for the positive gate.
# Negative control: a long cache so the decode partition heuristic
# (mha_decode_partition_heuristic.mojo) picks MORE than one partition at bs=1
# (SM100: min(sm_count // (bs*heads_per_group), num_keys // 512)) but collapses
# to a single partition at NEG_BS_HI, so the probe's split-K reduction tree --
# and thus its output -- must differ across the two batch sizes.
comptime NEG_NUM_KEYS = 2048
comptime NEG_BS_HI = 8


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    """One fuzz case: the runtime-varied attention shape."""

    var seq_len: Int
    var num_keys: Int
    var valid_length: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "seq_len=",
            self.seq_len,
            " num_keys=",
            self.num_keys,
            " valid_length=",
            self.valid_length,
        )


def gen_specs(n: Int, safe: Bool) -> List[CaseSpec]:
    """Generate `n` boundary-aware cases (decode-biased).

    When `safe` is True, excludes the degenerate num_keys==1 / valid_length==0
    region (which hangs the decode kernel) so an in-process run does not wedge.
    When False, generates the full space so an isolated orchestrator can explore
    (and surface) those degenerate configs.
    """
    var specs = List[CaseSpec]()
    var nk_lo = 16 if safe else 1
    var vl_lo = 1 if safe else 0
    for _ in range(n):
        var num_keys = boundary_int(nk_lo, 1024, TILE)
        # Bias 3/4 of cases to decode (seq_len == 1) -- the SM100 1q path. Keep
        # prefill small and within [1, num_keys] (valid causal range): attention
        # is O(seq_len*num_keys), so big prefill is slow under memcheck without
        # adding boundary coverage small shapes don't already give.
        var seq_len: Int
        if Int(random_ui64(0, 3)) != 0:
            seq_len = 1
        else:
            seq_len = boundary_int(1, min(num_keys, 256), TILE)
        # valid_length in [vl_lo, num_keys], boundary-biased (full / full-1 /
        # half / edge) where the padding boundary meets the causal boundary.
        var vroll = Int(random_ui64(0, 5))
        var valid_length: Int
        if vroll == 0:
            valid_length = vl_lo
        elif vroll == 1:
            valid_length = num_keys
        elif vroll == 2:
            valid_length = max(vl_lo, num_keys - 1)
        elif vroll == 3:
            valid_length = max(vl_lo, num_keys // 2)
        else:
            valid_length = Int(random_ui64(UInt64(vl_lo), UInt64(num_keys)))
        specs.append(CaseSpec(seq_len, num_keys, valid_length))
    return specs^


def run_one_case(
    ctx: DeviceContext, spec: CaseSpec, check: Bool = False
) raises:
    """Allocate, fill with random data, and launch attention once."""
    comptime batch_size = 1
    var seq_len = spec.seq_len
    var num_keys = spec.num_keys

    var q_size = batch_size * num_heads * seq_len * depth
    var kv_size = batch_size * kv_num_heads * num_keys * depth

    var q_host = ctx.enqueue_create_host_buffer[qkv_type](q_size)
    var k_host = ctx.enqueue_create_host_buffer[qkv_type](kv_size)
    var v_host = ctx.enqueue_create_host_buffer[qkv_type](kv_size)
    rand(q_host.as_span())
    rand(k_host.as_span())
    rand(v_host.as_span())

    var q_dev = ctx.enqueue_create_buffer[qkv_type](q_size)
    var k_dev = ctx.enqueue_create_buffer[qkv_type](kv_size)
    var v_dev = ctx.enqueue_create_buffer[qkv_type](kv_size)
    var o_dev = ctx.enqueue_create_buffer[qkv_type](q_size)
    ctx.enqueue_copy(q_dev, q_host)
    ctx.enqueue_copy(k_dev, k_host)
    ctx.enqueue_copy(v_dev, v_host)

    var q = TileTensor(
        q_dev, row_major((batch_size, seq_len, Idx[num_heads], Idx[depth]))
    )
    var k = TileTensor(
        k_dev, row_major((batch_size, num_keys, Idx[kv_num_heads], Idx[depth]))
    )
    var v = TileTensor(
        v_dev, row_major((batch_size, num_keys, Idx[kv_num_heads], Idx[depth]))
    )
    var o = TileTensor(
        o_dev, row_major((batch_size, seq_len, Idx[num_heads], Idx[depth]))
    )

    # `valid_lengths` is a 1-element [num_seqs=1] uint32 tensor, kept alive on
    # `vl_dev` across the launches (so a finding is the real OOB, not a UAF).
    var vl_dev = ctx.enqueue_create_buffer[.uint32](1)
    ctx.enqueue_memset(vl_dev, UInt32(spec.valid_length))
    var vl = LayoutTensor[.uint32, Layout.row_major(1)](vl_dev.unsafe_ptr())
    var mask = CausalPaddingMask(vl)

    flash_attention(o, q, k, v, mask, scale, ctx)
    ctx.synchronize()

    # Also exercise the naive reference path (mha.mojo bmm0).
    var o_ref_dev = ctx.enqueue_create_buffer[qkv_type](q_size)
    var o_ref = TileTensor(
        o_ref_dev, row_major((batch_size, seq_len, Idx[num_heads], Idx[depth]))
    )
    mha_gpu_naive(
        q,
        k,
        v,
        mask,
        o_ref,
        scale,
        batch_size,
        seq_len,
        num_keys,
        num_heads,
        depth,
        group,
        ctx,
    )
    ctx.synchronize()

    if check:
        # Numerical oracle: flash_attention (o) vs the naive reference (o_ref),
        # bf16 tolerances (matching the differential test's split-K allowance).
        var o_h = ctx.enqueue_create_host_buffer[qkv_type](q_size)
        var o_ref_h = ctx.enqueue_create_host_buffer[qkv_type](q_size)
        ctx.enqueue_copy(o_h, o_dev)
        ctx.enqueue_copy(o_ref_h, o_ref_dev)
        ctx.synchronize()
        if not numeric_check(
            o_h.as_span(), o_ref_h.as_span(), atol=2e-2, rtol=2e-2
        ):
            raise Error("flash_attention vs naive mismatch")

    _ = q_dev
    _ = k_dev
    _ = v_dev
    _ = o_dev
    _ = o_ref_dev
    _ = vl_dev


def run_schedule_case(ctx: DeviceContext, spec: CaseSpec, repeats: Int) raises:
    """Schedule amplification: force split-K decode (num_partitions=2) and run
    `repeats` times on the same input, flagging any non-bit-exact output. A
    difference means the inter-block split-K reduction is order-dependent (a
    race / nondeterminism) -- which racecheck (intra-block only) cannot see.
    """
    comptime batch_size = 1
    comptime seq_len = 1  # decode -> the split-K path
    var num_keys = max(2, spec.num_keys)  # large enough to split; avoid hang

    var q_size = batch_size * num_heads * seq_len * depth
    var kv_size = batch_size * kv_num_heads * num_keys * depth

    var q_host = ctx.enqueue_create_host_buffer[qkv_type](q_size)
    var k_host = ctx.enqueue_create_host_buffer[qkv_type](kv_size)
    var v_host = ctx.enqueue_create_host_buffer[qkv_type](kv_size)
    rand(q_host.as_span())
    rand(k_host.as_span())
    rand(v_host.as_span())

    var q_dev = ctx.enqueue_create_buffer[qkv_type](q_size)
    var k_dev = ctx.enqueue_create_buffer[qkv_type](kv_size)
    var v_dev = ctx.enqueue_create_buffer[qkv_type](kv_size)
    var o_dev = ctx.enqueue_create_buffer[qkv_type](q_size)
    ctx.enqueue_copy(q_dev, q_host)
    ctx.enqueue_copy(k_dev, k_host)
    ctx.enqueue_copy(v_dev, v_host)

    var q = TileTensor(
        q_dev, row_major((batch_size, seq_len, Idx[num_heads], Idx[depth]))
    )
    var k = TileTensor(
        k_dev, row_major((batch_size, num_keys, Idx[kv_num_heads], Idx[depth]))
    )
    var v = TileTensor(
        v_dev, row_major((batch_size, num_keys, Idx[kv_num_heads], Idx[depth]))
    )
    var o = TileTensor(
        o_dev, row_major((batch_size, seq_len, Idx[num_heads], Idx[depth]))
    )

    var vl_dev = ctx.enqueue_create_buffer[.uint32](1)
    ctx.enqueue_memset(vl_dev, UInt32(num_keys))  # full (no padding)
    var vl = LayoutTensor[.uint32, Layout.row_major(1)](vl_dev.unsafe_ptr())
    var mask = CausalPaddingMask(vl)
    var np = Optional[Int](2)  # force split-K

    flash_attention(o, q, k, v, mask, scale, ctx, num_partitions=np)
    ctx.synchronize()
    var first_h = ctx.enqueue_create_host_buffer[qkv_type](q_size)
    ctx.enqueue_copy(first_h, o_dev)
    ctx.synchronize()

    for _ in range(repeats - 1):
        flash_attention(o, q, k, v, mask, scale, ctx, num_partitions=np)
        ctx.synchronize()
        var rep_h = ctx.enqueue_create_host_buffer[qkv_type](q_size)
        ctx.enqueue_copy(rep_h, o_dev)
        ctx.synchronize()
        # Bit-exact (atol=rtol=0): any difference is nondeterminism.
        if not numeric_check(
            rep_h.as_span(), first_h.as_span(), atol=0.0, rtol=0.0
        ):
            raise Error("flash_attention split-K nondeterminism (schedule)")

    _ = q_dev
    _ = k_dev
    _ = v_dev
    _ = o_dev
    _ = vl_dev


def run_determinism_case(
    ctx: DeviceContext, spec: CaseSpec, repeats: Int
) raises:
    """Run-to-run determinism under the DEFAULT decode partition heuristic.

    Decode (seq_len=1) with num_partitions left UNSET, so the launch uses
    whatever `mha_decode_partition_heuristic.mojo` picks for this shape (unlike
    `run_schedule_case`, which pins num_partitions=2). All `repeats` launches
    are enqueued to DISTINCT output buffers before a single synchronize --
    mirroring test_split_k_determinism_amd.mojo, this keeps several decode
    kernels (and their split-K scratch) queued back-to-back so the GPU stays
    under concurrent-kernel load, the regime where an order-dependent atomic /
    scratch-reuse race surfaces (KERN-3129). The split-K combine
    (`mha_splitk_reduce`) is a fixed sequential loop, so a correct kernel is
    bit-stable; any divergence is a REAL race finding, not a legitimate
    reduction-order FP wobble.
    """
    comptime batch_size = 1
    comptime seq_len = 1  # decode -> the split-K decode path
    var num_keys = max(2, spec.num_keys)

    var q_size = batch_size * num_heads * seq_len * depth
    var kv_size = batch_size * kv_num_heads * num_keys * depth

    var q_host = ctx.enqueue_create_host_buffer[qkv_type](q_size)
    var k_host = ctx.enqueue_create_host_buffer[qkv_type](kv_size)
    var v_host = ctx.enqueue_create_host_buffer[qkv_type](kv_size)
    rand(q_host.as_span())
    rand(k_host.as_span())
    rand(v_host.as_span())

    var q_dev = ctx.enqueue_create_buffer[qkv_type](q_size)
    var k_dev = ctx.enqueue_create_buffer[qkv_type](kv_size)
    var v_dev = ctx.enqueue_create_buffer[qkv_type](kv_size)
    ctx.enqueue_copy(q_dev, q_host)
    ctx.enqueue_copy(k_dev, k_host)
    ctx.enqueue_copy(v_dev, v_host)

    var q = TileTensor(
        q_dev, row_major((batch_size, seq_len, Idx[num_heads], Idx[depth]))
    )
    var k = TileTensor(
        k_dev, row_major((batch_size, num_keys, Idx[kv_num_heads], Idx[depth]))
    )
    var v = TileTensor(
        v_dev, row_major((batch_size, num_keys, Idx[kv_num_heads], Idx[depth]))
    )

    var vl_dev = ctx.enqueue_create_buffer[.uint32](1)
    ctx.enqueue_memset(vl_dev, UInt32(num_keys))  # full (no padding)
    var vl = LayoutTensor[.uint32, Layout.row_major(1)](vl_dev.unsafe_ptr())
    var mask = CausalPaddingMask(vl)

    # Distinct output buffers so reruns do not serialize on one buffer (WAW)
    # and stay live concurrently -- the concurrent-load pattern.
    var outs = List[DeviceBuffer[qkv_type]](capacity=repeats)
    for _ in range(repeats):
        var o_dev = ctx.enqueue_create_buffer[qkv_type](q_size)
        var o = TileTensor(
            o_dev, row_major((batch_size, seq_len, Idx[num_heads], Idx[depth]))
        )
        # No num_partitions => the decode heuristic picks it.
        flash_attention(o, q, k, v, mask, scale, ctx)
        outs.append(o_dev)
    ctx.synchronize()

    var first_h = ctx.enqueue_create_host_buffer[qkv_type](q_size)
    ctx.enqueue_copy(first_h, outs[0])
    ctx.synchronize()
    for r in range(1, repeats):
        var rep_h = ctx.enqueue_create_host_buffer[qkv_type](q_size)
        ctx.enqueue_copy(rep_h, outs[r])
        ctx.synchronize()
        # Bit-exact (atol=rtol=0): any difference is nondeterminism.
        if not numeric_check(
            rep_h.as_span(), first_h.as_span(), atol=0.0, rtol=0.0
        ):
            raise Error(
                "flash_attention decode run-to-run nondeterminism"
                " (determinism oracle, default partition heuristic)"
            )

    _ = q_dev
    _ = k_dev
    _ = v_dev
    _ = vl_dev
    _ = outs^


# ===----------------------------------------------------------------------=== #
# Batch-invariance oracle
# ===----------------------------------------------------------------------=== #
#
# The invariant: a decode token's output must not depend on what OTHER
# sequences it is co-batched with, once the split-K partition count is fixed.
# The dense decode path derives its split-K partition boundaries from the
# PADDED num_keys (k.dim[1](), passed as max_cache_valid_length -- see
# mha.mojo flash_attention_dispatch), NOT from the per-sequence valid length,
# so with a fixed num_keys and a pinned num_partitions the probe (slot 0) has
# an identical reduction tree across compositions and its output must be
# bit-exact. The NEGATIVE control removes the pin: the heuristic keys the
# partition count on batch_size, so the probe's reduction tree -- and output --
# MUST change across batch sizes (proving the oracle has teeth).


def _bit_equal[
    dtype: DType
](a: Span[Scalar[dtype], _], b: Span[Scalar[dtype], _]) -> Bool:
    """Bit-exact element comparison with NO output side effect.

    Unlike `numeric_check(atol=0, rtol=0)` (which prints FUZZ_NUMERIC_FAIL on a
    mismatch -- a marker the orchestrator treats as a finding), this is silent.
    The negative control needs a MISMATCH to be its PASS, so it must not emit a
    finding marker on the expected divergence. Returns True iff every element's
    bits match.
    """
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i].to_bits() != b[i].to_bits():
            return False
    return True


def _run_mha_composition(
    ctx: DeviceContext,
    batch_size: Int,
    num_keys: Int,
    probe_seed: Int,
    filler_seed: Int,
    np: Optional[Int],
) raises -> HostBuffer[qkv_type]:
    """Build one dense decode batch and return the probe's output rows.

    Slot 0 is the probe: its Q and its K/V rows sit at buffer offset 0 and are
    filled from `probe_seed`, so they are BYTE-IDENTICAL across compositions;
    slots 1.. are fillers filled from `filler_seed`. The padded `num_keys` is
    fixed by the caller, so max_cache_valid_length -- and therefore the probe's
    split-K partition boundaries -- match across compositions. `np` pins
    num_partitions when set (positive gate); when None the decode heuristic
    picks it (negative control). Returns `o[0]` flattened to num_heads*depth.
    """
    comptime seq_len = 1  # decode
    var q_size = batch_size * seq_len * num_heads * depth
    var kv_size = batch_size * num_keys * kv_num_heads * depth
    var probe_q = seq_len * num_heads * depth
    var probe_kv = num_keys * kv_num_heads * depth

    var q_host = ctx.enqueue_create_host_buffer[qkv_type](q_size)
    var k_host = ctx.enqueue_create_host_buffer[qkv_type](kv_size)
    var v_host = ctx.enqueue_create_host_buffer[qkv_type](kv_size)

    # Probe (slot 0) from `probe_seed` -> identical bytes across compositions.
    seed(probe_seed)
    rand(q_host.unsafe_ptr(), probe_q)
    rand(k_host.unsafe_ptr(), probe_kv)
    rand(v_host.unsafe_ptr(), probe_kv)
    # Fillers (slots 1..) from `filler_seed`.
    if batch_size > 1:
        seed(filler_seed)
        rand(q_host.unsafe_ptr() + probe_q, q_size - probe_q)
        rand(k_host.unsafe_ptr() + probe_kv, kv_size - probe_kv)
        rand(v_host.unsafe_ptr() + probe_kv, kv_size - probe_kv)

    # valid_lengths: the probe attends to all keys (full); fillers get varied
    # lengths (they exercise different masking but are not compared).
    var vl_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    vl_host[0] = UInt32(num_keys)
    if batch_size > 1:
        seed(filler_seed ^ 0x5A5A5A5A)
        for i in range(1, batch_size):
            vl_host[i] = UInt32(boundary_int(1, num_keys, TILE))

    var q_dev = ctx.enqueue_create_buffer[qkv_type](q_size)
    var k_dev = ctx.enqueue_create_buffer[qkv_type](kv_size)
    var v_dev = ctx.enqueue_create_buffer[qkv_type](kv_size)
    var o_dev = ctx.enqueue_create_buffer[qkv_type](q_size)
    var vl_dev = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(q_dev, q_host)
    ctx.enqueue_copy(k_dev, k_host)
    ctx.enqueue_copy(v_dev, v_host)
    ctx.enqueue_copy(vl_dev, vl_host)

    var q = TileTensor(
        q_dev, row_major((batch_size, seq_len, Idx[num_heads], Idx[depth]))
    )
    var k = TileTensor(
        k_dev, row_major((batch_size, num_keys, Idx[kv_num_heads], Idx[depth]))
    )
    var v = TileTensor(
        v_dev, row_major((batch_size, num_keys, Idx[kv_num_heads], Idx[depth]))
    )
    var o = TileTensor(
        o_dev, row_major((batch_size, seq_len, Idx[num_heads], Idx[depth]))
    )
    # Runtime-sized [batch_size] valid_lengths (mask indexes valid_lengths[b]).
    comptime vl_layout = Layout(UNKNOWN_VALUE)
    var vl = LayoutTensor[.uint32, vl_layout](
        vl_dev.unsafe_ptr(),
        RuntimeLayout[vl_layout].row_major(IndexList[1](batch_size)),
    )
    var mask = CausalPaddingMask(vl)

    flash_attention(o, q, k, v, mask, scale, ctx, num_partitions=np)
    ctx.synchronize()

    var o_host = ctx.enqueue_create_host_buffer[qkv_type](q_size)
    ctx.enqueue_copy(o_host, o_dev)
    ctx.synchronize()
    # Slice the probe's output rows (o[0], i.e. the first num_heads*depth elems).
    var probe_host = ctx.enqueue_create_host_buffer[qkv_type](probe_q)
    for i in range(probe_q):
        probe_host[i] = o_host[i]

    _ = q_dev
    _ = k_dev
    _ = v_dev
    _ = o_dev
    _ = vl_dev
    return probe_host^


def run_batch_invariance_case(ctx: DeviceContext, spec: CaseSpec) raises:
    """Batch-invariance gate for MHA decode with PINNED num_partitions.

    The same probe (slot 0) runs under two co-batch compositions that differ in
    filler count / lengths / values; num_partitions is PINNED to the same value
    and the padded num_keys is fixed, so the probe's split-K reduction tree is
    identical across the two runs. The probe's output rows must be bit-exact
    (atol=rtol=0). A divergence is a REAL batch-variance finding (decode started
    depending on the batch beyond the pinned partition count), not a flake.
    """
    # TILE-aligned, >= 2*TILE so the pinned 2-way split is non-degenerate.
    var num_keys = TILE * max(2, ceildiv(max(1, spec.num_keys), TILE))
    var probe_seed = spec.num_keys * 131 + spec.valid_length * 17 + 1
    var pinned = Optional[Int](BI_PINNED_PARTITIONS)

    # Composition A: probe + 1 filler.
    var probe_a = _run_mha_composition(
        ctx, 2, num_keys, probe_seed, 0x0A11CE, pinned
    )
    # Composition B: probe + 4 fillers (different count, filler seed, lengths).
    var probe_b = _run_mha_composition(
        ctx, 5, num_keys, probe_seed, 0x0B0BB1E, pinned
    )
    if not numeric_check(
        probe_a.as_span(), probe_b.as_span(), atol=0.0, rtol=0.0
    ):
        raise Error(
            "MHA decode is NOT batch-invariant under pinned num_partitions:"
            " the probe's output changed with the co-batch composition"
        )
    _ = probe_a^
    _ = probe_b^


def run_batch_variance_negctl_case(ctx: DeviceContext, spec: CaseSpec) raises:
    """Negative control: MHA decode across batch sizes under the DEFAULT
    heuristic.

    The same probe runs at bs=1 vs bs=NEG_BS_HI with a long cache
    (NEG_NUM_KEYS), for which `get_mha_decoding_num_partitions` picks DIFFERENT
    partition counts. The probe's split-K reduction tree therefore differs and
    its output MUST diverge. This proves the batch_invariance oracle has teeth:
    a bit-match means the control lost its sensitivity (or the heuristic
    collapsed to one count for these shapes) and is reported as a FAILURE
    rather than a silent pass (mirrors the positive-control canary framing).
    """
    var np_lo = get_mha_decoding_num_partitions[num_heads, group](
        1, NEG_NUM_KEYS, ctx
    )
    var np_hi = get_mha_decoding_num_partitions[num_heads, group](
        NEG_BS_HI, NEG_NUM_KEYS, ctx
    )
    print(
        "FUZZ_NEGCTL num_keys=",
        NEG_NUM_KEYS,
        "np(bs=1)=",
        np_lo,
        "np(bs=",
        NEG_BS_HI,
        ")=",
        np_hi,
    )
    if np_lo == np_hi:
        raise Error(
            "negative-control setup: the decode heuristic gives the SAME"
            " num_partitions ("
            + String(np_lo)
            + ") for bs=1 and bs="
            + String(NEG_BS_HI)
            + " at num_keys="
            + String(NEG_NUM_KEYS)
            + "; adjust the shapes so the counts differ"
        )

    var probe_seed = spec.num_keys * 131 + spec.valid_length * 17 + 7
    var probe_a = _run_mha_composition(
        ctx, 1, NEG_NUM_KEYS, probe_seed, 0x0A11CE, None
    )
    var probe_b = _run_mha_composition(
        ctx, NEG_BS_HI, NEG_NUM_KEYS, probe_seed, 0x0B0BB1E, None
    )
    # Inverted check: a MISMATCH is the PASS (different partition counts must
    # change the reduction). `_bit_equal` avoids emitting FUZZ_NUMERIC_FAIL so
    # the expected divergence is not mis-parsed by the driver as a finding.
    if _bit_equal(probe_a.as_span(), probe_b.as_span()):
        raise Error(
            "negative control LOST TEETH: MHA decode probe bit-matched across"
            " bs=1 (np="
            + String(np_lo)
            + ") and bs="
            + String(NEG_BS_HI)
            + " (np="
            + String(np_hi)
            + ") despite different partition counts"
        )
    _ = probe_a^
    _ = probe_b^


# ===----------------------------------------------------------------------=== #
# Mode dispatch (argv handling shared from _fuzz)
# ===----------------------------------------------------------------------=== #


def main() raises:
    var args = collect_args()
    var mode = flag(args, "--mode", "fuzz")
    var the_seed = flag_int(args, "--seed", fuzz_seed)
    var the_budget = flag_int(args, "--budget", budget)
    var check = flag_int(args, "--check", 0) == 1
    var schedule_repeats = flag_int(args, "--schedule", 0)
    var rerun = flag_int(args, "--rerun", 0)
    var batch_invariance = flag_int(args, "--batch-invariance", 0) == 1
    var batch_negctl = flag_int(args, "--batch-variance", 0) == 1
    seed(the_seed)

    if mode == "list-specs":
        # Generation only -- the orchestrator enumerates these, no GPU work.
        var specs = gen_specs(the_budget, safe=False)
        for i in range(len(specs)):
            print(
                "FUZZ_SPEC idx=",
                i,
                "seq_len=",
                specs[i].seq_len,
                "num_keys=",
                specs[i].num_keys,
                "valid_length=",
                specs[i].valid_length,
            )
        return

    if mode == "single":
        # Flag names match the FUZZ_SPEC keys (underscored) so the orchestrator
        # can drive any target generically with `--<spec_key> <value>`.
        var sl = flag_int(args, "--seq_len", 1)
        var nk = flag_int(args, "--num_keys", 16)
        var vl = flag_int(args, "--valid_length", 8)
        print("FUZZ_SINGLE seq_len=", sl, "num_keys=", nk, "valid_length=", vl)
        with DeviceContext() as ctx:
            if batch_negctl:
                run_batch_variance_negctl_case(ctx, CaseSpec(sl, nk, vl))
            elif batch_invariance:
                run_batch_invariance_case(ctx, CaseSpec(sl, nk, vl))
            elif rerun > 0:
                run_determinism_case(ctx, CaseSpec(sl, nk, vl), rerun)
            elif schedule_repeats > 0:
                run_schedule_case(ctx, CaseSpec(sl, nk, vl), schedule_repeats)
            else:
                run_one_case(ctx, CaseSpec(sl, nk, vl), check)
        print("FUZZ_RESULT verdict=PASS")
        return

    # Default: in-process standalone fuzz over the SAFE space.
    print(
        "=== fuzz_mha_causal seed=",
        the_seed,
        "budget=",
        the_budget,
        "depth=",
        depth,
        "(in-process, safe space) ===",
    )
    var specs = gen_specs(the_budget, safe=True)
    with DeviceContext() as ctx:
        for i in range(len(specs)):
            print("case", i, ":", specs[i])
            if batch_negctl:
                run_batch_variance_negctl_case(ctx, specs[i])
            elif batch_invariance:
                run_batch_invariance_case(ctx, specs[i])
            elif rerun > 0:
                run_determinism_case(ctx, specs[i], rerun)
            elif schedule_repeats > 0:
                run_schedule_case(ctx, specs[i], schedule_repeats)
            else:
                run_one_case(ctx, specs[i], check)
    print("=== done:", len(specs), "cases ===")
