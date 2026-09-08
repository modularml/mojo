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

from std.math import isclose
from std.random import rand
from std.sys import argv, get_defined_bool


from max.gpu import *
from max.gpu.host import DeviceContext
from layout import (
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    row_major,
)
from nn.attention.gpu.mha import (
    _mha_decode_fold_ok,
    flash_attention,
    mha_gpu_naive,
)
from nn.attention.gpu.amd_structured.config import (
    mha_decode_fold_tile_q_seq_len,
    mha_decode_fold_wide_mma,
)
from nn.attention.mha_mask import CausalMask
from std.sys.info import _is_amd_rdna
from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_true,
)

from std.utils.index import Index
from std.utils.numerics import min_or_neg_inf


def is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


def test[
    qkv_type: DType,
    depth: Int,
    num_heads: Int,
    group: Int = 1,
    batch_size: Int = 1,
](
    seq_len: Int,
    num_keys: Int,
    ctx: DeviceContext,
    is_benchmark: Bool = False,
) raises:
    print("test_mha_causal_mask")
    print(
        "qkv_type:",
        qkv_type,
        "depth:",
        depth,
        "num_heads:",
        num_heads,
        "group:",
        group,
        "seq_len:",
        seq_len,
        "num_keys:",
        num_keys,
        "batch_size:",
        batch_size,
    )
    # Query, key, value dimensions.
    comptime scale = Float32(0.125)  # rsqrt[type, 1](Float32(depth))
    comptime kv_num_heads = num_heads // group
    comptime mask_type = DType.bfloat16
    comptime output_type = DType.bfloat16

    # Q, K, V shapes.
    var q_size = batch_size * num_heads * seq_len * depth
    var k_size = batch_size * kv_num_heads * num_keys * depth
    var v_size = k_size
    var o_size = q_size
    var mask_size = batch_size * num_heads * seq_len * num_keys

    # Allocate memory for all variables.
    var q_ptr = ctx.enqueue_create_host_buffer[qkv_type](q_size)
    var k_ptr = ctx.enqueue_create_host_buffer[qkv_type](k_size)
    var v_ptr = ctx.enqueue_create_host_buffer[qkv_type](v_size)
    var mask_ptr = ctx.enqueue_create_host_buffer[mask_type](mask_size)
    var output_ptr = ctx.enqueue_create_host_buffer[output_type](o_size)
    var flash_output_ptr = ctx.enqueue_create_host_buffer[output_type](o_size)

    for i in range(o_size):
        output_ptr[i] = Scalar[output_type](0)

    # Construct mask buffer for causal mask initialization.
    comptime layout_4d = Layout.row_major[4]()
    var mask = LayoutTensor[mask_type, layout_4d](
        mask_ptr,
        RuntimeLayout[layout_4d].row_major(
            Index(batch_size, num_heads, seq_len, num_keys)
        ),
    )

    # Initialize Q, K, V in bf16, then roundtrip through qkv_type so the
    # naive bf16 reference sees identical values (matters for fp8).
    var q_bf16_ptr = ctx.enqueue_create_host_buffer[.bfloat16](q_size)
    var k_bf16_ptr = ctx.enqueue_create_host_buffer[.bfloat16](k_size)
    var v_bf16_ptr = ctx.enqueue_create_host_buffer[.bfloat16](v_size)
    rand(q_bf16_ptr.as_span())
    rand(k_bf16_ptr.as_span())
    rand(v_bf16_ptr.as_span())
    for i in range(q_size):
        var val = q_bf16_ptr[i].cast[qkv_type]()
        q_ptr[i] = val
        q_bf16_ptr[i] = val.cast[.bfloat16]()
    for i in range(k_size):
        var val = k_bf16_ptr[i].cast[qkv_type]()
        k_ptr[i] = val
        k_bf16_ptr[i] = val.cast[.bfloat16]()
    for i in range(v_size):
        var val = v_bf16_ptr[i].cast[qkv_type]()
        v_ptr[i] = val
        v_bf16_ptr[i] = val.cast[.bfloat16]()

    # Initialize causal mask.
    for b in range(batch_size):
        for h in range(num_heads):
            for q_idx in range(seq_len):
                for k_idx in range(num_keys):
                    mask.store(
                        Index(b, h, q_idx, k_idx),
                        0 if q_idx + num_keys - seq_len
                        >= k_idx else min_or_neg_inf[mask_type](),
                    )

    # Device pointers
    var q_device_ptr = ctx.enqueue_create_buffer[qkv_type](q_size)
    var k_device_ptr = ctx.enqueue_create_buffer[qkv_type](k_size)
    var v_device_ptr = ctx.enqueue_create_buffer[qkv_type](v_size)
    var mask_device_ptr = ctx.enqueue_create_buffer[mask_type](mask_size)
    var output_device_ptr = ctx.enqueue_create_buffer[output_type](o_size)

    # Copy from host to device
    ctx.enqueue_copy(q_device_ptr, q_ptr)
    ctx.enqueue_copy(k_device_ptr, k_ptr)
    ctx.enqueue_copy(v_device_ptr, v_ptr)
    ctx.enqueue_copy(mask_device_ptr, mask_ptr)

    # Construct device buffers.
    var q_device = TileTensor(
        q_device_ptr.unsafe_ptr(),
        row_major((batch_size, seq_len, Idx[num_heads], Idx[depth])),
    )
    var k_device = TileTensor(
        k_device_ptr.unsafe_ptr(),
        row_major((batch_size, num_keys, Idx[kv_num_heads], Idx[depth])),
    )
    var v_device = TileTensor(
        v_device_ptr.unsafe_ptr(),
        row_major((batch_size, num_keys, Idx[kv_num_heads], Idx[depth])),
    )
    var mask4d = TileTensor(
        mask_device_ptr.unsafe_ptr(),
        row_major((batch_size, num_heads, seq_len, num_keys)),
    )
    var output_device = TileTensor(
        output_device_ptr.unsafe_ptr(),
        row_major((batch_size, seq_len, Idx[num_heads], Idx[depth])),
    )

    @always_inline
    def kernel_launch(
        ctx: DeviceContext,
    ) raises {
        var q_device,
        var k_device,
        var v_device,
        var mask4d,
        var output_device,
        imm,
    }:
        flash_attention(
            output_device,
            q_device,
            k_device,
            v_device,
            CausalMask(),
            scale,
            ctx,
        )

    if is_benchmark:
        comptime nrun = 50

        # Warmup
        kernel_launch(ctx)

        var nstime = Float64(ctx.execution_time(kernel_launch, nrun)) / Float64(
            nrun
        )
        var sectime = nstime / 1000000
        print(nrun, "runs avg", sectime, "ms")

    else:
        kernel_launch(ctx)

    ctx.synchronize()

    ctx.enqueue_copy(flash_output_ptr, output_device_ptr)

    # Naive reference: use roundtripped bf16 values so both flash and
    # naive see identical input data.
    var q_ref_device_ptr = ctx.enqueue_create_buffer[.bfloat16](q_size)
    var k_ref_device_ptr = ctx.enqueue_create_buffer[.bfloat16](k_size)
    var v_ref_device_ptr = ctx.enqueue_create_buffer[.bfloat16](v_size)
    ctx.enqueue_copy(q_ref_device_ptr, q_bf16_ptr)
    ctx.enqueue_copy(k_ref_device_ptr, k_bf16_ptr)
    ctx.enqueue_copy(v_ref_device_ptr, v_bf16_ptr)

    var q_ref_device = TileTensor(
        q_ref_device_ptr.unsafe_ptr(),
        row_major((batch_size, seq_len, Idx[num_heads], Idx[depth])),
    )
    var k_ref_device = TileTensor(
        k_ref_device_ptr.unsafe_ptr(),
        row_major((batch_size, num_keys, Idx[kv_num_heads], Idx[depth])),
    )
    var v_ref_device = TileTensor(
        v_ref_device_ptr.unsafe_ptr(),
        row_major((batch_size, num_keys, Idx[kv_num_heads], Idx[depth])),
    )
    var output_ref_device_ptr = ctx.enqueue_create_buffer[output_type](o_size)
    ctx.enqueue_copy(output_ref_device_ptr, output_ptr)
    var output_device_ref = TileTensor(
        output_ref_device_ptr.unsafe_ptr(),
        row_major((batch_size, seq_len, Idx[num_heads], Idx[depth])),
    )

    mha_gpu_naive(
        q_ref_device,
        k_ref_device,
        v_ref_device,
        mask4d,
        output_device_ref,
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
    ctx.enqueue_copy(output_ptr, output_ref_device_ptr)
    ctx.synchronize()
    _ = output_ref_device_ptr
    _ = q_ref_device_ptr
    _ = k_ref_device_ptr
    _ = v_ref_device_ptr

    var rtol = 6e-2 if qkv_type.is_float8() else 3e-2
    var atol = 1e-3 if qkv_type.is_float8() else 1e-5
    # Batch is outermost on purpose: a query-row stride taken from the tile's
    # height leaves batch 0 intact and lands its pad rows on batch 1, which a
    # batch-0-only check cannot see.
    for b in range(batch_size):
        for h in range(num_heads):
            for s in range(seq_len):
                for d in range(depth):
                    var idx = d + depth * (h + num_heads * (s + seq_len * b))
                    var expect = output_ptr[idx].cast[.float64]()
                    var actual = flash_output_ptr[idx].cast[.float64]()
                    if not isclose(actual, expect, atol=atol, rtol=rtol):
                        var rerr = abs((actual - expect) / expect)
                        print(b, h, s, d, actual, expect, rerr)
                    assert_almost_equal(actual, expect, atol=atol, rtol=rtol)

    _ = q_device_ptr
    _ = k_device_ptr
    _ = v_device_ptr
    _ = mask_device_ptr
    _ = output_device_ptr


comptime USE_FP8 = get_defined_bool["USE_FP8", False]()


def test_helper[depth: Int](ctx: DeviceContext) raises:
    comptime dtype = DType.float8_e4m3fn if USE_FP8 else DType.bfloat16
    test[dtype, depth=depth, num_heads=1](128, 128, ctx)
    test[dtype, depth=depth, num_heads=1](384, 384, ctx)
    test[dtype, depth=depth, num_heads=24, group=3](1024, 1024, ctx)
    test[dtype, depth=depth, num_heads=16, group=16](128, 128, ctx)
    test[dtype, depth=depth, num_heads=16, group=16](1024, 1024, ctx)
    comptime if depth == 128:
        # MiniMax-M3 shape: head_dim=128, 64 query heads, 4 KV heads.
        test[dtype, depth=depth, num_heads=64, group=16](1024, 1024, ctx)
    # Sequence length not multiple of 128
    test[dtype, depth=depth, num_heads=3, group=3](128, 128, ctx)
    test[dtype, depth=depth, num_heads=3, group=3](102, 102, ctx)
    test[dtype, depth=depth, num_heads=1](14, 14, ctx)
    test[dtype, depth=depth, num_heads=1](528, 528, ctx)
    # Token gen
    test[dtype, depth=depth, num_heads=32](1, 512, ctx, is_benchmark())
    test[dtype, depth=depth, num_heads=11](1, 256, ctx)
    test[dtype, depth=depth, num_heads=1](1, 11, ctx)
    test[dtype, depth=depth, num_heads=2](1, 523, ctx)
    test[dtype, depth=depth, num_heads=24, group=3](1, 29, ctx)
    test[dtype, depth=depth, num_heads=3, group=3](1, 156, ctx)
    test[dtype, depth=depth, num_heads=3, group=3](1, 208, ctx)
    test[dtype, depth=depth, num_heads=32, group=4](1, 1208, ctx)
    test[dtype, depth=depth, num_heads=32, group=4](1, 2008, ctx)
    test[dtype, depth=depth, num_heads=32, group=4](1, 5000, ctx)
    test[dtype, depth=depth, num_heads=16, group=16](1, 128, ctx)
    test[dtype, depth=depth, num_heads=16, group=16](1, 1024, ctx)
    test[dtype, depth=depth, num_heads=16, group=16](1, 5000, ctx)
    comptime if depth == 128:
        # MiniMax-M3 shape: head_dim=128, 64 query heads, 4 KV heads.
        test[dtype, depth=depth, num_heads=64, group=16](1, 1024, ctx)
    # Speculative-verify query lengths. On AMD these take the decode kernel's
    # query-token fold, one instantiation per S; where the fold does not apply
    # they take prefill, so gate on the predicate itself rather than restating
    # its dtype/depth window here — `test_fold_eligibility` is what stops that
    # gate from going quietly false and dropping the coverage. The num_keys span
    # a single partition (29), a BN=128 straddle (208), an aligned length
    # (1024), and split-K-heavy (5000).
    comptime if _mha_decode_fold_ok[dtype, depth, 16, 1, 4]():
        # Single KV head: rows are contiguous heads, BM = num_heads*S = 32 to
        # 128, stacked over 2 to 8 warp M-tiles. S=8 is the 1+7 verify step.
        for seq_len in [2, 3, 4, 5, 6, 7, 8]:
            test[dtype, depth=depth, num_heads=16, group=16](seq_len, 29, ctx)
            test[dtype, depth=depth, num_heads=16, group=16](seq_len, 208, ctx)
            test[dtype, depth=depth, num_heads=16, group=16](seq_len, 1024, ctx)
            test[dtype, depth=depth, num_heads=16, group=16](seq_len, 5000, ctx)
            # Batch > 1 is what makes the query-row STRIDES observable: every
            # address is `<per-token stride> * batch_idx`, so a stride taken
            # from the tile's slot count is a no-op at batch 1. `num_keys` past
            # one partition reaches the split-K stat planes, same stride.
            test[dtype, depth=depth, num_heads=16, group=16, batch_size=3](
                seq_len, 208, ctx
            )
            test[dtype, depth=depth, num_heads=16, group=16, batch_size=3](
                seq_len, 5000, ctx
            )
        # 32 heads exercises row -> (token, head) at num_heads != MMA_M, and
        # S=3 is the only ODD width reaching the wide MMA (96 rows tile 32
        # without tiling S). Gate on S=3 too, or an eligibility change there
        # would silently drop it to prefill and still pass.
        comptime if _mha_decode_fold_ok[
            dtype, depth, 32, 32, 3
        ]() and _mha_decode_fold_ok[dtype, depth, 32, 32, 4]():
            for seq_len in [3, 4]:
                test[dtype, depth=depth, num_heads=32, group=32](
                    seq_len, 208, ctx
                )
                test[dtype, depth=depth, num_heads=32, group=32](
                    seq_len, 1024, ctx
                )
        # 24 heads is OUTSIDE the pad's divisor set (128 % 24 != 0), so S=4
        # keeps its own 96 rows — on fp8 the only way to execute the THREE-warp
        # wide-MFMA CTA (BM=96, WM=32, 192 threads), since every 96-row width at
        # 16 or 32 heads pads onto the four-warp tile. It is also the only fold
        # shape whose heads-inner divisor is not a power of two, the divide
        # behind `store_partition_info`'s row split and the mask's row -> token.
        # 5000 keys puts it through split-K, so the odd warp count reaches the
        # stat writes.
        comptime if _mha_decode_fold_ok[dtype, depth, 24, 24, 4]():
            test[dtype, depth=depth, num_heads=24, group=24](4, 208, ctx)
            test[dtype, depth=depth, num_heads=24, group=24](4, 5000, ctx)
            test[dtype, depth=depth, num_heads=24, group=24, batch_size=3](
                4, 5000, ctx
            )
        # One query head per KV head: rows are the S tokens and the Q/O row
        # stride becomes the BSHD token stride, which only num_heads > 1
        # exercises; the KV-head base term in the split-K stat write needs
        # num_keys past one partition (1024/5000). S=9 is an EAGLE3 step-0
        # width at K=7.
        for seq_len in [2, 3, 4, 5, 6, 7, 8, 9]:
            test[dtype, depth=depth, num_heads=16, group=1](seq_len, 29, ctx)
            test[dtype, depth=depth, num_heads=16, group=1](seq_len, 208, ctx)
            test[dtype, depth=depth, num_heads=16, group=1](seq_len, 1024, ctx)
            test[dtype, depth=depth, num_heads=16, group=1](seq_len, 5000, ctx)
            test[dtype, depth=depth, num_heads=4, group=1](seq_len, 1024, ctx)
            # The one shape both arms' group predicate admits, and the only one
            # where the token stride equals the head stride.
            test[dtype, depth=depth, num_heads=1, group=1](seq_len, 1024, ctx)


def test_fold_eligibility() raises:
    """Pin `_mha_decode_fold_ok`'s answer per shape.

    Both the `is_token_generation` routing decision and the dispatch ladder call
    it, and a shape it rejects just takes prefill — slower but correct, so
    every numerical case in this file passes either way. This table is the only
    signal that a shape meant to fold silently does not.
    """
    comptime bf16 = DType.bfloat16
    # A rejection is arch-independent (the arch gate only ever rejects more), so
    # the inadmissible half is asserted outright and the admissible half against
    # a reference shape. That pins the shape logic without pinning which AMD
    # parts carry the fold (RDNA carries none).
    comptime available = _mha_decode_fold_ok[bf16, 128, 16, 1, 4]()
    # Pin the reference itself on the parts that carry the fold, else every
    # admissible assertion below passes vacuously wherever it is False.
    comptime if not _is_amd_rdna():
        assert_true(available)

    # One query head per KV head: rows are the S tokens of one head, so any
    # num_heads works and S is bounded only by the cap.
    assert_equal(_mha_decode_fold_ok[bf16, 128, 16, 1, 2](), available)
    assert_equal(_mha_decode_fold_ok[bf16, 128, 16, 1, 5](), available)
    assert_equal(_mha_decode_fold_ok[bf16, 128, 4, 1, 5](), available)
    assert_equal(_mha_decode_fold_ok[bf16, 64, 16, 1, 4](), available)
    # The EAGLE3 step-0 width at K=7, the reason the cap is 9.
    assert_equal(_mha_decode_fold_ok[bf16, 128, 16, 1, 9](), available)
    # num_heads == 1 satisfies both arms' group predicate; the stacked one
    # rejects it (`1*S % 16 != 0`) so this is the narrow arm, where the token
    # and head row strides coincide (`_q_stride0` collapses to `q_depth`).
    assert_equal(_mha_decode_fold_ok[bf16, 128, 1, 1, 4](), available)
    # Single KV head: rows are contiguous heads, so num_heads*S must tile 2 to 8
    # 16-row warp M-tiles — 128 rows at the 1+7 verify step, 64 at half the heads.
    assert_equal(_mha_decode_fold_ok[bf16, 128, 16, 16, 2](), available)
    assert_equal(_mha_decode_fold_ok[bf16, 128, 16, 16, 4](), available)
    assert_equal(_mha_decode_fold_ok[bf16, 128, 16, 16, 8](), available)
    assert_equal(_mha_decode_fold_ok[bf16, 128, 8, 8, 8](), available)
    # The head counts the numerical cases run, which are gated on THIS answer:
    # without an entry each, tightening eligibility would silently delete those
    # cases and stay green.
    assert_equal(_mha_decode_fold_ok[bf16, 128, 32, 32, 3](), available)
    assert_equal(_mha_decode_fold_ok[bf16, 128, 32, 32, 4](), available)
    assert_equal(_mha_decode_fold_ok[bf16, 128, 24, 24, 4](), available)
    # Ragged exempts the padded-batch exclusion.
    assert_equal(
        _mha_decode_fold_ok[
            bf16, 128, 16, 1, 4, use_valid_length=True, ragged=True
        ](),
        available,
    )
    # FP8 folds on both arms wherever its decode MFMA is the 16-row 16x16x128.
    comptime fp8 = DType.float8_e4m3fn
    assert_equal(_mha_decode_fold_ok[fp8, 128, 16, 1, 4](), available)
    assert_equal(_mha_decode_fold_ok[fp8, 128, 16, 16, 4](), available)
    # The fp8 arm of the 24-head case, which is the one that reaches the wide
    # MMA's three-warp geometry.
    assert_equal(_mha_decode_fold_ok[fp8, 128, 24, 24, 4](), available)

    # S=1 is plain decode, not a fold arm.
    assert_false(_mha_decode_fold_ok[bf16, 128, 16, 1, 1]())
    # Past `_MHA_DECODE_FOLD_MAX_S`.
    assert_false(_mha_decode_fold_ok[bf16, 128, 16, 1, 10]())
    # 1 < group < num_heads would need a token stride and a head stride at once.
    assert_false(_mha_decode_fold_ok[bf16, 128, 32, 4, 4]())
    # Single KV head at S=9: num_heads*S = 144 exceeds 8 warp M-tiles.
    assert_false(_mha_decode_fold_ok[bf16, 128, 16, 16, 9]())
    # And 32 query heads exceed it from S=5 (160 rows).
    assert_false(_mha_decode_fold_ok[bf16, 128, 32, 32, 5]())
    # depth > 128 is rejected for both arms by one shared conjunct, though the
    # reason is the single-KV-head arm's WN == BN register pressure.
    assert_false(_mha_decode_fold_ok[bf16, 256, 16, 16, 4]())
    assert_false(_mha_decode_fold_ok[bf16, 256, 16, 1, 4]())
    # fp16 shares the MMA shape but is untested through the fold.
    assert_false(_mha_decode_fold_ok[.float16, 128, 16, 1, 4]())
    # FP8 at depth 64 picks the 32x32x64 decode MFMA, whose MMA_M outruns the
    # fold's 16-row warp M-tile.
    assert_false(_mha_decode_fold_ok[fp8, 64, 16, 1, 4]())
    assert_false(_mha_decode_fold_ok[fp8, 64, 16, 16, 4]())
    # A sink lookup is the folded head only while num_heads == MMA_M.
    assert_false(_mha_decode_fold_ok[bf16, 128, 16, 1, 4, sink=True]())
    # A padded batch can hold a sequence shorter than S, whose split-K stats go
    # unwritten and un-skipped.
    assert_false(
        _mha_decode_fold_ok[bf16, 128, 16, 1, 4, use_valid_length=True]()
    )
    # A mask needing a per-tile decode check reads an S-row span that underflows.
    assert_false(
        _mha_decode_fold_ok[
            bf16, 128, 16, 1, 4, check_mask_during_decoding=True
        ]()
    )


def test_fold_wide_mma() raises:
    """Pin `mha_decode_fold_wide_mma`'s answer per shape.

    The wide arm is a pure perf choice — every width is numerically correct on
    either MFMA — so the numerical cases above pass whether or not the predicate
    fires, and a predicate gone quietly false is a silent no-op. This table is
    the only signal of which widths take it.

    These are the predicate's answers on RAW widths, several of which dispatch
    never passes: `mha_decode_fold_tile_q_seq_len` pads first, so at 16 heads
    every width from S=3 up arrives here as 8. The composed assertion at the end
    is what ties the table to the widths that ship.
    """
    comptime fp8 = DType.float8_e4m3fn
    comptime bf16 = DType.bfloat16

    # 32 rows per warp, so 96 rows (S=6 at 16 heads) is the narrowest width
    # leaving the three warps the split-K partition count assumes.
    assert_true(mha_decode_fold_wide_mma[fp8, 16, 16, 6]())
    assert_true(mha_decode_fold_wide_mma[fp8, 16, 16, 8]())
    # 128 and 96 rows at 32 heads — the widths that carry an odd S.
    assert_true(mha_decode_fold_wide_mma[fp8, 32, 32, 3]())
    assert_true(mha_decode_fold_wide_mma[fp8, 32, 32, 4]())
    # 96 rows at a head count the pad cannot reach, which is the only way a
    # 3-warp CTA actually launches. Covered numerically in `test_helper`.
    assert_true(mha_decode_fold_wide_mma[fp8, 24, 24, 4]())
    assert_equal(mha_decode_fold_tile_q_seq_len[fp8, 24, 24, 4](), 4)

    # Two warps: the kernel is fine, the partition count callers pass is not.
    assert_false(mha_decode_fold_wide_mma[fp8, 16, 16, 4]())
    assert_false(mha_decode_fold_wide_mma[fp8, 16, 16, 2]())
    # Rows that do not tile 32 would leave a warp a partial M-tile.
    assert_false(mha_decode_fold_wide_mma[fp8, 16, 16, 7]())
    # The narrow arm owns S token rows, not `num_heads * S` of them, and keeps
    # the four-way N split whose P cannot be warp-local.
    assert_false(mha_decode_fold_wide_mma[fp8, 16, 1, 6]())
    assert_false(mha_decode_fold_wide_mma[fp8, 16, 1, 8]())
    # bf16 stays on its 16-row shape: it doubles every SMEM term, so halving the
    # warps would only halve the waves.
    assert_false(mha_decode_fold_wide_mma[bf16, 16, 16, 6]())
    assert_false(mha_decode_fold_wide_mma[bf16, 16, 16, 8]())
    # S=1 is plain decode.
    assert_false(mha_decode_fold_wide_mma[fp8, 16, 16, 1]())

    # Compose the two in the order dispatch does, so the `False` rows above
    # cannot be read as "S=4 at 16 heads launches a 2-warp CTA" — the pad takes
    # that width to 8 slots first, and 8 IS wide.
    comptime for s in [3, 4, 5, 6, 7, 8]:
        assert_true(
            mha_decode_fold_wide_mma[
                fp8, 16, 16, mha_decode_fold_tile_q_seq_len[fp8, 16, 16, s]()
            ]()
        )
    assert_equal(mha_decode_fold_tile_q_seq_len[fp8, 16, 16, 2](), 2)
    assert_false(
        mha_decode_fold_wide_mma[
            fp8, 16, 16, mha_decode_fold_tile_q_seq_len[fp8, 16, 16, 2]()
        ]()
    )


def main() raises:
    test_fold_eligibility()
    test_fold_wide_mma()
    with DeviceContext() as ctx:
        comptime if USE_FP8:
            comptime for depth in [128, 256]:
                test_helper[depth](ctx)
        else:
            comptime for depth in [64, 80, 128, 256]:
                test_helper[depth](ctx)
