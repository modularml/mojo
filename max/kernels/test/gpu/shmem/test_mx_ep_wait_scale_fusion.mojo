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
"""Checks that `ep_wait` emits `scale_4d` directly, for both MX quant formats.

The MX up/gate-projection grouped matmul consumes the activation E8M0 scale in
the per-expert fixed-stride `scale_4d` slot layout
(`Shuffler.scale_4d_slot_byte_off`). Without the fold that layout comes from a
standalone kernel (`preshuffle_grouped_scale_4d_gpu`, which traces as
`block_scaled_preshuffle_grouped_scale_4d_kernel_KS<scale_K>` for both formats —
the name follows the `Shuffler`, not the activation dtype) running *serially*
after `ep_wait` (dispatch_wait) has written the per-token scale row-major. With
`fuse_a_scale_preshuffle`, `MXTokenFormat.copy_msg_to_output_tensor` writes the
scale straight into the slot layout, deleting the separate kernel from the
decode critical path.

Both formats run the same checks, parameterized on `quant_dtype`: the format
enters `MXTokenFormat` only through `elems_per_byte`, and the copy-out scale
path is identical at both because `group_size` is 32 either way.

Two gates per case:

1. Byte-compare the whole slot region between the fold off (raw row-major scales
   plus the standalone `preshuffle_grouped_scale_4d_gpu`) and on (direct slot
   write). This validates the fold's address arithmetic.
2. Decode the copied message against the host pattern. Gate 1 is blind to a
   wrong `elems_per_byte`: its two arms share `quant_dtype` and `hidden_size`
   and read the scale region at the same `scales_offset()`, so a region read
   from the wrong place lands at the same wrong slot on each side and cancels.
   The pattern's quant and scale bytes occupy disjoint non-zero ranges ([1,127]
   vs [128,254]), so neither can be mistaken for the other or for an untouched
   byte.

The launcher replays `ep_wait`'s copy granularity — ONE WARP PER TOKEN, warps
spread across blocks — so rows `r` and `r+16`, which pack into the same
`scale_4d` i32 cell since `S_MN_PACK` is 2 over 16 MFMA lanes, are written by
DIFFERENT warps/SMs. Each store is an independent single E8M0 byte (no
read-modify-write), so they cannot clobber each other; were one ever to lower to
an i32 RMW, this layout would surface it as a byte mismatch.

Both slot buffers are zero-filled first: the standalone kernel zero-fills the
pad rows of the last partial 32-row block while the fold writes only real token
rows, so a dirty buffer would read as a stride bug.

This test always passes `shared_expert_offset=0`, so `expert_slot == expert_id`;
production sets it to 1 when the shared expert is fused, which only shifts the
slot base.

Single-GPU is sufficient: nothing the fold changes crosses ranks. MI355X-only.

Go through bazel rather than plain `mojo`: `mojo` cannot resolve `max.*` here,
and an installed `shmem` package shadows `ep_comm.mojo` on the import path, so a
plain run can pass green against a stale artifact that never saw the edit.
  ./bazelw build --config=local-mi355 \\
      //max/kernels/test/gpu/shmem:test_mx_ep_wait_scale_fusion.mojo.test
  ./bazel-bin/max/kernels/test/gpu/shmem/test_mx_ep_wait_scale_fusion.mojo.test
"""

from max.gpu import block_idx
from max.gpu.host import DeviceContext, HostBuffer
from max.gpu.host.info import MI355X
from max.gpu.primitives import warp_id
from std.math import align_up

from layout import Coord, Idx, TileTensor, row_major
from layout.tile_layout import TensorLayout
from linalg.fp4_utils import MXFP4_SF_VECTOR_SIZE, MXFP8_SF_VECTOR_SIZE
from linalg.matmul.gpu.amd import Shuffler

from shmem.ep_comm import MXTokenFormat

from std.testing import assert_equal, assert_true

comptime WARP_SIZE = 64  # gfx950


# ===----------------------------------------------------------------------=== #
# Input helpers
# ===----------------------------------------------------------------------=== #


# Disjoint ranges, neither containing 0: quant byte, scale byte and untouched 0
# are mutually distinguishable. The modulus is prime and coprime to both
# multipliers, so a permutation hides only at displacements that are multiples
# of 127 — never at the power-of-two stride a `scale_4d` tiling bug takes.
def _quant_byte(token: Int, j: Int) -> UInt8:
    return UInt8((token * 131 + j * 37) % 127 + 1)


def _scale_byte(token: Int, k: Int) -> UInt8:
    return UInt8((token * 97 + k * 37) % 127 + 128)


def _build_routing(
    a_offsets_host: HostBuffer[.uint32],
    num_tokens_by_expert: List[Int],
):
    """Fills `a_offsets_host` with the ragged per-expert prefix sums: [0] is 0
    and [e+1] is the token count of the first e+1 experts."""
    a_offsets_host[0] = UInt32(0)
    for i in range(len(num_tokens_by_expert)):
        a_offsets_host[i + 1] = a_offsets_host[i] + UInt32(
            num_tokens_by_expert[i]
        )


# ===----------------------------------------------------------------------=== #
# Launcher kernel: one warp per token, calling the REAL format copy.
# ===----------------------------------------------------------------------=== #


def _ep_wait_copy_kernel[
    quant_dtype: DType,
    scales_dtype: DType,
    out_layout: TensorLayout,
    scales_layout: TensorLayout,
    aoff_layout: TensorLayout,
    hidden_size: Int,
    top_k: Int,
    fuse_a_scale_preshuffle: Bool,
    warps_per_block: Int,
](
    output_tokens: TileTensor[quant_dtype, out_layout, MutUntrackedOrigin],
    output_scales: TileTensor[scales_dtype, scales_layout, MutUntrackedOrigin],
    recv_buf: Pointer[UInt8, MutUntrackedOrigin],
    a_offsets: TileTensor[.uint32, aoff_layout, ImmutAnyOrigin],
    msg_bytes_dev: Int32,
    num_active_dev: Int32,
    total_tokens_dev: Int32,
    max_padded_M_dev: Int32,
):
    var msg_bytes = Int(msg_bytes_dev)
    var num_active = Int(num_active_dev)
    var total_tokens = Int(total_tokens_dev)
    var max_padded_M = Int(max_padded_M_dev)
    var fmt = MXTokenFormat[
        hidden_size, top_k, fuse_a_scale_preshuffle=fuse_a_scale_preshuffle
    ](output_tokens, output_scales, max_padded_M)

    var warp_global = Int(block_idx.x) * warps_per_block + Int(warp_id())
    var token = warp_global
    if token >= total_tokens:
        return

    # The scan stands in for the expert identity a real dispatch-wait comm SM
    # already derives from its block index.
    var expert_slot = 0
    while (
        expert_slot < num_active - 1
        and Int(a_offsets[Coord(expert_slot + 1)]) <= token
    ):
        expert_slot += 1
    var expert_start = Int(a_offsets[Coord(expert_slot)])

    var msg_ptr = recv_buf.unsafe_offset(token * msg_bytes)
    fmt.copy_msg_to_output_tensor(msg_ptr, token, expert_slot, expert_start)


# ===----------------------------------------------------------------------=== #
# The two gates, run per (format, shape) case.
# ===----------------------------------------------------------------------=== #


def _run_fusion_check[
    quant_dtype: DType,
    hidden_size: Int,
    NUM_ACTIVE: Int,
    top_k: Int = 8,
](
    name: String,
    num_tokens_by_expert: List[Int],
    inflate_padded_M: Int,
    ctx: DeviceContext,
) raises:
    # `MXTokenFormat.group_size` is `MXFP4_SF_VECTOR_SIZE` for both formats;
    # pin that the MXFP8 alias has not drifted away from it.
    comptime assert MXFP4_SF_VECTOR_SIZE == MXFP8_SF_VECTOR_SIZE
    comptime assert hidden_size % MXFP4_SF_VECTOR_SIZE == 0
    # Mirrors `MXTokenFormat.elems_per_byte`, the only place the quant format
    # enters; the bracket assert below pins the two derivations together.
    comptime elems_per_byte = 2 if quant_dtype == DType.uint8 else 1
    comptime output_dim = hidden_size // elems_per_byte
    comptime scale_K = hidden_size // MXFP4_SF_VECTOR_SIZE
    comptime n_off = NUM_ACTIVE + 1
    comptime warps_per_block = 4
    comptime fmt_name = "MXFP4" if elems_per_byte == 2 else "MXFP8"

    var num_active = NUM_ACTIVE
    # Not `debug_assert`: a short hand-written list would leave the tail of
    # `a_offsets` uninitialized and the kernel would route tokens off the end.
    assert_equal(
        len(num_tokens_by_expert),
        NUM_ACTIVE,
        "num_tokens_by_expert must have one entry per active expert",
    )
    var total_tokens = 0
    var max_tokens = 0
    for ne in num_tokens_by_expert:
        total_tokens += ne
        max_tokens = max(max_tokens, ne)
    # Slot stride the matmul reads with. `inflate_padded_M` drives a build-time
    # bound strictly larger than the runtime max — the common decode case.
    var max_padded_M = max(align_up(max_tokens, 32), inflate_padded_M)
    var slot_bytes = num_active * max_padded_M * scale_K

    print(
        "  ",
        fmt_name,
        " ",
        name,
        " active=",
        num_active,
        " total_tokens=",
        total_tokens,
        " hidden=",
        hidden_size,
        " scale_K=",
        scale_K,
        " max_padded_M=",
        max_padded_M,
    )

    comptime hw = ctx.default_device_info

    # The token output is compared too, so each path gets its own zeroed buffer
    # (gate 2 needs an untouched byte to stay 0).
    var ref_tok_d = ctx.enqueue_create_buffer[quant_dtype](
        total_tokens * output_dim
    )
    var fused_tok_d = ctx.enqueue_create_buffer[quant_dtype](
        total_tokens * output_dim
    )
    ref_tok_d.enqueue_fill(Scalar[quant_dtype](0))
    fused_tok_d.enqueue_fill(Scalar[quant_dtype](0))
    var ref_tok_tt = TileTensor[origin=MutAnyOrigin](
        ref_tok_d, row_major(Coord(total_tokens, Idx[output_dim]))
    )
    var fused_tok_tt = TileTensor[origin=MutAnyOrigin](
        fused_tok_d, row_major(Coord(total_tokens, Idx[output_dim]))
    )

    # Message per token: [quants | E8M0 scales]. A throwaway instance yields the
    # real offsets; passing the token tensor is what resolves `elems_per_byte`.
    var dummy_scales_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](scale_K)
    var dummy_fmt = MXTokenFormat[hidden_size, top_k](
        ref_tok_tt,
        TileTensor[origin=MutAnyOrigin](
            dummy_scales_d, row_major(Coord(1, Idx[scale_K]))
        ),
    )
    comptime FmtType = type_of(dummy_fmt)
    var scales_off = FmtType.scales_offset()
    var msg_bytes = FmtType.token_size()
    # A wrong `elems_per_byte` shifts the slot writer and reader together, so
    # gate 1 cannot see it; pin the quant extent directly instead.
    assert_true(
        output_dim <= scales_off and scales_off < 2 * output_dim,
        "scales_offset() is not the aligned quant width for this format",
    )

    # --- Host inputs: synthesized recv buffer + routing. ---
    var recv_h = ctx.enqueue_create_host_buffer[.uint8](
        total_tokens * msg_bytes
    )
    var a_off_h = ctx.enqueue_create_host_buffer[.uint32](n_off)
    ctx.synchronize()
    for i in range(total_tokens * msg_bytes):
        recv_h[i] = UInt8(0)
    for t in range(total_tokens):
        var base = t * msg_bytes
        for j in range(output_dim):
            recv_h[base + j] = _quant_byte(t, j)
        for k in range(scale_K):
            recv_h[base + scales_off + k] = _scale_byte(t, k)
    _build_routing(a_off_h, num_tokens_by_expert)

    var recv_d = ctx.enqueue_create_buffer[.uint8](total_tokens * msg_bytes)
    var a_off_d = ctx.enqueue_create_buffer[.uint32](n_off)
    ctx.enqueue_copy(recv_d, recv_h)
    ctx.enqueue_copy(a_off_d, a_off_h)
    var a_off_tt = TileTensor[origin=ImmutAnyOrigin](
        a_off_d, row_major[n_off]()
    )

    var grid_blocks = (total_tokens + warps_per_block - 1) // warps_per_block
    if grid_blocks == 0:
        grid_blocks = 1

    # ---- Path A (reference): copy writes raw [tokens, scale_K], then the
    #      standalone preshuffle kernel rearranges into slots. ----
    var raw_scales_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](
        total_tokens * scale_K
    )
    var ref_d = ctx.enqueue_create_buffer[.uint8](slot_bytes)
    ref_d.enqueue_fill(UInt8(0))

    var raw_scales_tt = TileTensor[origin=MutAnyOrigin](
        raw_scales_d, row_major(Coord(total_tokens, Idx[scale_K]))
    )

    # Every launcher param is explicit, so both scales layouts must be pinned —
    # they differ between the raw and the slot buffer.
    comptime kernel_ref = _ep_wait_copy_kernel[
        quant_dtype,
        DType.float8_e8m0fnu,
        ref_tok_tt.LayoutType,
        raw_scales_tt.LayoutType,
        a_off_tt.LayoutType,
        hidden_size=hidden_size,
        top_k=top_k,
        fuse_a_scale_preshuffle=False,
        warps_per_block=warps_per_block,
    ]
    ctx.enqueue_function[kernel_ref](
        ref_tok_tt,
        raw_scales_tt,
        recv_d.unsafe_ptr(),
        a_off_tt,
        Int32(msg_bytes),
        Int32(num_active),
        Int32(total_tokens),
        Int32(max_padded_M),
        grid_dim=grid_blocks,
        block_dim=warps_per_block * WARP_SIZE,
    )

    var raw_scales_bytes = TileTensor[origin=ImmutAnyOrigin](
        raw_scales_d.unsafe_ptr()
        .unsafe_bitcast[UInt8]()
        .as_unsafe_any_origin(),
        row_major(Coord(total_tokens, Idx[scale_K])),
    )
    var ref_slots_tt = TileTensor[origin=MutAnyOrigin](
        ref_d, row_major(Coord(num_active * max_padded_M, Idx[scale_K]))
    )
    Shuffler[1].preshuffle_grouped_scale_4d_gpu[K_SCALES=scale_K](
        raw_scales_bytes,
        ref_slots_tt,
        a_off_tt,
        num_active,
        # Must be the same slot stride the fused path uses, not the runtime
        # `max_tokens`, or the whole-buffer byte compare is invalid.
        max_padded_M,
        hw.sm_count * 2,
        ctx,
    )

    # ---- Path B (fused): copy writes the slot layout directly. ----
    var fused_scales_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](slot_bytes)
    fused_scales_d.enqueue_fill(Float8_e8m0fnu(0))
    var fused_slots_tt = TileTensor[origin=MutAnyOrigin](
        fused_scales_d,
        row_major(Coord(num_active * max_padded_M, Idx[scale_K])),
    )
    comptime kernel_fused = _ep_wait_copy_kernel[
        quant_dtype,
        DType.float8_e8m0fnu,
        fused_tok_tt.LayoutType,
        fused_slots_tt.LayoutType,
        a_off_tt.LayoutType,
        hidden_size=hidden_size,
        top_k=top_k,
        fuse_a_scale_preshuffle=True,
        warps_per_block=warps_per_block,
    ]
    ctx.enqueue_function[kernel_fused](
        fused_tok_tt,
        fused_slots_tt,
        recv_d.unsafe_ptr(),
        a_off_tt,
        Int32(msg_bytes),
        Int32(num_active),
        Int32(total_tokens),
        Int32(max_padded_M),
        grid_dim=grid_blocks,
        block_dim=warps_per_block * WARP_SIZE,
    )

    # --- Gate 1: fused slots == reference slots, byte for byte. ---
    var ref_host = ctx.enqueue_create_host_buffer[.uint8](slot_bytes)
    var fused_host = ctx.enqueue_create_host_buffer[.uint8](slot_bytes)
    ctx.enqueue_copy(ref_host, ref_d)
    ctx.enqueue_copy(
        fused_host, fused_scales_d.unsafe_ptr().unsafe_bitcast[UInt8]()
    )

    # --- Gate 2 inputs: the copied quant region and the unfused row-major
    #     scales, both decoded against the host pattern. ---
    var tok_bytes = total_tokens * output_dim
    var ref_tok_h = ctx.enqueue_create_host_buffer[.uint8](tok_bytes)
    var fused_tok_h = ctx.enqueue_create_host_buffer[.uint8](tok_bytes)
    var raw_sc_h = ctx.enqueue_create_host_buffer[.uint8](
        total_tokens * scale_K
    )
    ctx.enqueue_copy(ref_tok_h, ref_tok_d.unsafe_ptr().unsafe_bitcast[UInt8]())
    ctx.enqueue_copy(
        fused_tok_h, fused_tok_d.unsafe_ptr().unsafe_bitcast[UInt8]()
    )
    ctx.enqueue_copy(
        raw_sc_h, raw_scales_d.unsafe_ptr().unsafe_bitcast[UInt8]()
    )
    ctx.synchronize()

    var mismatches = 0
    var nonzero_ref = 0
    var nonzero_fused = 0
    for i in range(slot_bytes):
        if ref_host[i] != fused_host[i]:
            mismatches += 1
        if ref_host[i] != 0:
            nonzero_ref += 1
        if fused_host[i] != 0:
            nonzero_fused += 1
    assert_equal(mismatches, 0, "fused slot bytes differ from the reference")
    # Non-vacuity: real rows are a tiny fraction of the slot region, so two
    # silently-empty buffers would compare equal. Scale bytes are all >= 128.
    assert_equal(
        nonzero_ref, total_tokens * scale_K, "reference wrote no scale rows"
    )
    assert_equal(
        nonzero_fused, total_tokens * scale_K, "fold wrote no scale rows"
    )

    # A too-narrow quant copy leaves row tails at 0; a too-wide one or a shifted
    # `scales_offset()` drags scale-range bytes in. Both decode as a mismatch.
    var quant_mismatches = 0
    var fold_perturbed_quant = 0
    for t in range(total_tokens):
        for j in range(output_dim):
            var want = _quant_byte(t, j)
            if ref_tok_h[t * output_dim + j] != want:
                quant_mismatches += 1
            if fused_tok_h[t * output_dim + j] != want:
                fold_perturbed_quant += 1
    assert_equal(quant_mismatches, 0, "copied quant bytes do not decode")
    assert_equal(fold_perturbed_quant, 0, "the fold perturbed the quant copy")

    var scale_mismatches = 0
    for t in range(total_tokens):
        for k in range(scale_K):
            if raw_sc_h[t * scale_K + k] != _scale_byte(t, k):
                scale_mismatches += 1
    assert_equal(
        scale_mismatches,
        0,
        "row-major E8M0 scales do not decode (scales_offset)",
    )

    print(
        "    OK: ",
        slot_bytes,
        " slot bytes match; ",
        tok_bytes,
        " quant + ",
        total_tokens * scale_K,
        " scale bytes decode",
    )


def main() raises:
    var ctx = DeviceContext()
    comptime assert (
        ctx.default_device_info == MI355X
    ), "test_mx_ep_wait_scale_fusion currently requires MI355X"

    print("===> ep_wait MXFP4: fold equivalence + message decode")
    # KS224 up/gate proj: hidden 7168 → scale_K 224. The cases cover decode
    # (few tokens/expert), prefill, empty experts, and multi-tile m_blocks.
    _run_fusion_check[.uint8, hidden_size=7168, NUM_ACTIVE=1](
        "up-proj-single-tiny", [1], 0, ctx
    )
    _run_fusion_check[.uint8, hidden_size=7168, NUM_ACTIVE=5](
        "up-proj-decode", [1, 3, 0, 2, 5], 0, ctx
    )
    # >=17 tokens/expert makes rows r and r+16 both real inside one cell,
    # written by different warps (race-stress); >32 spans multiple m_blocks.
    _run_fusion_check[.uint8, hidden_size=7168, NUM_ACTIVE=4](
        "up-proj-multi-tile", [37, 64, 100, 5], 0, ctx
    )
    # Inflated max_padded_M (> runtime max) with >1 expert: guards that both
    # paths use the SAME build-time slot stride, so experts >= 1 land right.
    _run_fusion_check[.uint8, hidden_size=7168, NUM_ACTIVE=3](
        "up-proj-inflated-stride", [5, 20, 12], 128, ctx
    )
    # A down-proj-sized check too (KS64) to confirm the same code path is
    # K_SCALES-agnostic.
    _run_fusion_check[.uint8, hidden_size=2048, NUM_ACTIVE=4](
        "down-proj-decode", [1, 2, 0, 4], 0, ctx
    )

    print("===> ep_wait MXFP8: fold equivalence + message decode")
    # MiniMax-M3 up/gate proj: hidden 6144 → scale_K 192.
    _run_fusion_check[.float8_e4m3fn, hidden_size=6144, NUM_ACTIVE=1](
        "up-proj-single-tiny", [1], 0, ctx
    )
    _run_fusion_check[.float8_e4m3fn, hidden_size=6144, NUM_ACTIVE=5](
        "up-proj-decode", [1, 3, 0, 2, 5], 0, ctx
    )
    _run_fusion_check[.float8_e4m3fn, hidden_size=6144, NUM_ACTIVE=4](
        "up-proj-multi-tile", [37, 64, 100, 5], 0, ctx
    )
    _run_fusion_check[.float8_e4m3fn, hidden_size=6144, NUM_ACTIVE=3](
        "up-proj-inflated-stride", [5, 20, 12], 128, ctx
    )
    # M3's down-proj size (intermediate 3072 → KS96).
    _run_fusion_check[.float8_e4m3fn, hidden_size=3072, NUM_ACTIVE=4](
        "down-proj-ks96", [1, 2, 0, 4], 0, ctx
    )
    # Production EP8 (tp4dp2ep8, 8x MI355): 128 routed experts / 8 ranks = 16
    # local, +1 fused shared. Stride is `align_up(max_tokens_per_rank * n_ranks,
    # 32)`, and the dispatch branch takes `ceildiv(4096, ep_size // dp)` = 1024.
    _run_fusion_check[.float8_e4m3fn, hidden_size=6144, NUM_ACTIVE=17](
        "up-proj-ep8-dispatch",
        [3, 0, 1, 5, 2, 0, 7, 1, 0, 4, 2, 6, 0, 1, 3, 0, 9],
        8192,
        ctx,
    )
    # The allreduce-backed branch keeps the full 4096, giving 4096 * 8 = 32768:
    # the widest production stride, a 102 MiB slot region.
    _run_fusion_check[.float8_e4m3fn, hidden_size=6144, NUM_ACTIVE=17](
        "up-proj-ep8-wide-stride",
        [3, 0, 1, 5, 2, 0, 7, 1, 0, 4, 2, 6, 0, 1, 3, 0, 9],
        32768,
        ctx,
    )
    print("All ep_wait MX scale-fusion checks passed.")
