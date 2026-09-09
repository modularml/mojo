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

from std.collections import Optional
from std.math import exp, isclose
from std.random import rand, seed

from std.collections import Optional
from layout import (
    IntTuple,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
    UNKNOWN_VALUE,
)
from nn.attention.cpu.mha import flash_attention, flash_attention_split_kv
from nn.attention.mha_mask import NullMask
from std.testing import assert_equal

from std.utils import IndexList
from std.utils.index import Index


def reference_attention_bshd[
    dtype: DType
](
    q_nd: LayoutTensor[dtype, address_space=.GENERIC, ...],
    k_nd: LayoutTensor[dtype, address_space=.GENERIC, ...],
    v_nd: LayoutTensor[dtype, address_space=.GENERIC, ...],
    mask_nd: LayoutTensor[dtype, address_space=.GENERIC, ...],
    output_nd: LayoutTensor[mut=True, dtype, address_space=.GENERIC, ...],
    scale: Float32,
) raises:
    comptime assert dtype.is_floating_point(), "dtype must be floating point"
    comptime layout_4d = Layout.row_major[4]()

    def reshape_4d(
        buf: LayoutTensor[dtype, ...]
    ) -> LayoutTensor[
        dtype, layout_4d, buf.origin, address_space=buf.address_space
    ]:
        var shape = buf.runtime_layout.shape.value.canonicalize()
        var num_heads = shape[buf.rank - 2] if buf.rank == 4 else 1
        var shape_4d = Index(shape[0], shape[1], num_heads, shape[buf.rank - 1])
        return LayoutTensor[dtype, layout_4d, address_space=buf.address_space](
            buf.ptr, RuntimeLayout[layout_4d].row_major(shape_4d)
        )

    def reshape_mask_4d(
        buf: LayoutTensor[dtype, ...]
    ) -> LayoutTensor[
        dtype, layout_4d, buf.origin, address_space=buf.address_space
    ]:
        var shape = buf.runtime_layout.shape.value.canonicalize()
        var num_heads = shape[1] if buf.rank == 4 else 1
        var shape_4d = Index(
            shape[0], num_heads, shape[buf.rank - 2], shape[buf.rank - 1]
        )
        return LayoutTensor[dtype, layout_4d, address_space=buf.address_space](
            buf.ptr, RuntimeLayout[layout_4d].row_major(shape_4d)
        )

    var q_4d = reshape_4d(q_nd)
    var k_4d = reshape_4d(k_nd)
    var v_4d = reshape_4d(v_nd)
    var mask_4d = reshape_mask_4d(mask_nd)
    var output_4d = reshape_4d(output_nd)

    var batch_count = q_4d.dim(0)
    var seq_len = q_4d.dim(1)
    var num_heads = q_4d.dim(2)
    var depth_dim = q_4d.dim(3)
    var kv_seq_len = v_4d.dim(1)
    var kv_num_heads = v_4d.dim(2)

    assert_equal(num_heads % kv_num_heads, 0)

    var kv_group_count = num_heads // kv_num_heads

    assert_equal(batch_count, k_4d.dim(0))
    assert_equal(kv_seq_len, k_4d.dim(1))
    assert_equal(kv_num_heads, k_4d.dim(2))
    assert_equal(depth_dim, k_4d.dim(3))

    assert_equal(batch_count, v_4d.dim(0))
    assert_equal(depth_dim, v_4d.dim(3))

    assert_equal(batch_count, mask_4d.dim(0))
    assert_equal(num_heads, mask_4d.dim(1))
    assert_equal(seq_len, mask_4d.dim(2))
    assert_equal(kv_seq_len, mask_4d.dim(3))

    assert_equal(
        q_4d.runtime_layout.shape.value.canonicalize(),
        output_4d.runtime_layout.shape.value.canonicalize(),
    )

    comptime layout_2d = Layout.row_major[2]()
    var score_ptr = List(length=seq_len * kv_seq_len, fill=Scalar[dtype](0))
    var score_2d = LayoutTensor[dtype, layout_2d](
        score_ptr,
        RuntimeLayout[layout_2d].row_major(Index(seq_len, kv_seq_len)),
    )

    for b in range(batch_count):
        for h in range(num_heads):
            var kv_h = h // kv_group_count

            # Compute: `score = Q @ K`
            for m in range(seq_len):
                for n in range(kv_seq_len):
                    var accum = Scalar[dtype](0)
                    for k in range(depth_dim):
                        accum = q_4d[b, m, h, k][0].fma(
                            k_4d[b, n, kv_h, k][0], accum
                        )
                    score_2d[m, n] = accum

            # Apply scaling and masking to the score buffer
            for m in range(seq_len):
                for n in range(kv_seq_len):
                    score_2d[m, n] = (
                        score_2d[m, n][0] * scale.cast[dtype]()
                        + mask_4d[b, h, m, n][0]
                    )

            # Compute: `score = softmax(score)`
            for m in range(seq_len):
                var max_val = Scalar[dtype].MIN
                for n in range(kv_seq_len):
                    max_val = max(max_val, score_2d[m, n][0])

                var sum_val = Scalar[dtype](0)
                for n in range(kv_seq_len):
                    var exp_val = exp(score_2d[m, n][0] - max_val)
                    score_2d[m, n] = exp_val
                    sum_val += exp_val

                for n in range(kv_seq_len):
                    score_2d[m, n] = score_2d[m, n][0] / sum_val

            # Compute: `output = score @ V`
            for m in range(seq_len):
                for n in range(depth_dim):
                    var accum = Scalar[dtype](0)
                    for k in range(kv_seq_len):
                        accum = score_2d[m, k][0].fma(
                            v_4d[b, k, kv_h, n][0], accum
                        )
                    output_4d[b, m, h, n] = accum
    _ = score_ptr^


def reference_attention_bshd_with_sinks[
    dtype: DType
](
    q_nd: LayoutTensor[dtype, ...],
    k_nd: LayoutTensor[dtype, ...],
    v_nd: LayoutTensor[dtype, ...],
    mask_nd: LayoutTensor[dtype, ...],
    sink_weights_nd: LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), _],
    output_nd: LayoutTensor[mut=True, dtype, ...],
    scale: Float32,
) raises:
    """Reference implementation of attention with sink weights."""
    comptime assert dtype.is_floating_point(), "dtype must be floating point"

    comptime layout_4d = Layout.row_major[4]()

    def reshape_4d(
        buf: LayoutTensor[dtype, ...]
    ) -> LayoutTensor[
        dtype, layout_4d, buf.origin, address_space=buf.address_space
    ]:
        var shape = buf.runtime_layout.shape.value.canonicalize()
        var num_heads = shape[buf.rank - 2] if buf.rank == 4 else 1
        var shape_4d = Index(shape[0], shape[1], num_heads, shape[buf.rank - 1])
        return LayoutTensor[dtype, layout_4d, address_space=buf.address_space](
            buf.ptr, RuntimeLayout[layout_4d].row_major(shape_4d)
        )

    def reshape_mask_4d(
        buf: LayoutTensor[dtype, ...]
    ) -> LayoutTensor[
        dtype, layout_4d, buf.origin, address_space=buf.address_space
    ]:
        var shape = buf.runtime_layout.shape.value.canonicalize()
        var num_heads = shape[1] if buf.rank == 4 else 1
        var shape_4d = Index(
            shape[0], num_heads, shape[buf.rank - 2], shape[buf.rank - 1]
        )
        return LayoutTensor[dtype, layout_4d, address_space=buf.address_space](
            buf.ptr, RuntimeLayout[layout_4d].row_major(shape_4d)
        )

    var q_4d = reshape_4d(q_nd)
    var k_4d = reshape_4d(k_nd)
    var v_4d = reshape_4d(v_nd)
    var mask_4d = reshape_mask_4d(mask_nd)
    var output_4d = reshape_4d(output_nd)

    var batch_count = q_4d.dim(0)
    var seq_len = q_4d.dim(1)
    var num_heads = q_4d.dim(2)
    var depth_dim = q_4d.dim(3)
    var kv_seq_len = v_4d.dim(1)
    var kv_num_heads = v_4d.dim(2)
    # Note: sink_weights has one weight per head, not per token
    _ = min(sink_weights_nd.dim(0), num_heads)

    assert_equal(num_heads % kv_num_heads, 0)

    var kv_group_count = num_heads // kv_num_heads

    comptime layout_2d = Layout.row_major[2]()
    var score_ptr = List(length=seq_len * kv_seq_len, fill=Scalar[dtype](0))
    var score_2d = LayoutTensor[dtype, layout_2d](
        score_ptr,
        RuntimeLayout[layout_2d].row_major(Index(seq_len, kv_seq_len)),
    )

    for b in range(batch_count):
        for h in range(num_heads):
            var kv_h = h // kv_group_count

            # Compute: `score = Q @ K`
            for m in range(seq_len):
                for n in range(kv_seq_len):
                    var accum = Scalar[dtype](0)
                    for k in range(depth_dim):
                        accum = q_4d[b, m, h, k][0].fma(
                            k_4d[b, n, kv_h, k][0], accum
                        )
                    score_2d[m, n] = accum

            # Apply scaling and masking to the score buffer
            for m in range(seq_len):
                for n in range(kv_seq_len):
                    var score = score_2d[m, n][0] * scale.cast[dtype]()
                    score += mask_4d[b, h, m, n][0]
                    score_2d[m, n] = score

            # Compute softmax with sink tokens (following PyTorch reference)
            for m in range(seq_len):
                # Find max among attention logits
                var logits_max = Scalar[dtype].MIN
                for n in range(kv_seq_len):
                    logits_max = max(logits_max, score_2d[m, n][0])

                # Get sink logit for this head and compute joint max
                var sink_logit = sink_weights_nd[h][0]
                var joint_max = max(logits_max, sink_logit)

                # Compute normalized scores including sink in denominator
                var attention_sum = Scalar[dtype](0)
                for n in range(kv_seq_len):
                    var exp_val = exp(score_2d[m, n][0] - joint_max)
                    score_2d[m, n] = exp_val
                    attention_sum += exp_val

                # Add sink contribution to normalizer
                var sink_contribution = exp(sink_logit - joint_max)
                var normalizer = attention_sum + sink_contribution

                # Normalize only the attention scores (sink doesn't contribute to output)
                for n in range(kv_seq_len):
                    score_2d[m, n] = score_2d[m, n] / normalizer

            # Compute: `output = score @ V`
            for m in range(seq_len):
                for n in range(depth_dim):
                    var accum = Scalar[dtype](0)
                    for k in range(kv_seq_len):
                        accum = score_2d[m, k][0].fma(
                            v_4d[b, k, kv_h, n][0], accum
                        )
                    output_4d[b, m, h, n] = accum
    _ = score_ptr^


@fieldwise_init
struct TestCaseConfig[batch_rank: Int](TrivialRegisterPassable):
    """Test case workload configuration hyperparameters."""

    comptime rank = Self.batch_rank + 2
    comptime kv_cache_rank = Self.rank + 1

    var batch_dims: IndexList[Self.batch_rank]
    var seq_len: Int
    var kv_num_heads: Int
    var kv_seq_len: Int
    """Total KV sequence length including previous and current forwards."""
    var depth_dim: Int
    var scale: Float32

    @always_inline
    def prev_seq_len(self) -> Int:
        """Returns the KV cache length from previous iterations."""
        return self.kv_seq_len - self.seq_len

    @always_inline
    def build_shape[
        *, shape_rank: Int = Self.rank, is_kv: Bool = False
    ](self, x: Int, y: Int) -> IndexList[shape_rank]:
        var shape = IndexList[shape_rank]()

        comptime if shape_rank == self.kv_cache_rank:
            # Unsqueeze the output shape with a 1-dim.
            shape[0] = 1

            comptime for i in range(Self.batch_rank):
                shape[i + 1] = self.batch_dims[i]
        else:
            # Copy the batch dims without unsqueezing.
            comptime for i in range(Self.batch_rank):
                shape[i] = self.batch_dims[i]

        # Replace the number of query heads with the number of KV heads.
        comptime if is_kv and Self.batch_rank == 2:
            shape[shape_rank - 3] = self.kv_num_heads

        shape[shape_rank - 2] = x
        shape[shape_rank - 1] = y

        return shape

    @always_inline
    def build_shape_bshd[
        *, shape_rank: Int = Self.rank, is_kv: Bool = False
    ](self, x: Int, y: Int) -> IndexList[shape_rank]:
        var shape = IndexList[shape_rank]()

        comptime if shape_rank == self.kv_cache_rank:
            # Unsqueeze the output shape with a 1-dim.
            shape[0] = 1
            shape[1] = self.batch_dims[0]

            comptime for i in range(1, Self.batch_rank):
                shape[i + 2] = self.batch_dims[i]
        else:
            shape[0] = self.batch_dims[0]

            # Copy the batch dims without unsqueezing.
            comptime for i in range(1, Self.batch_rank):
                shape[i + 1] = self.batch_dims[i]

        # Replace the number of query heads with the number of KV heads.
        comptime if is_kv and Self.batch_rank == 2:
            shape[shape_rank - 2] = self.kv_num_heads

        shape[1] = x
        shape[shape_rank - 1] = y

        return shape


def verify_output[
    dtype: DType, batch_rank: Int
](
    output: LayoutTensor[dtype, ...],
    ref_output: LayoutTensor[dtype, ...],
    cfg: TestCaseConfig[batch_rank],
) raises -> None:
    """Compares `output` and `ref_output` elementwise, printing up to 5 mismatches.
    """
    var mismatches = 0
    for i in range(output.size()):
        var idx = output._offset(
            output.runtime_layout.idx2crd(
                RuntimeTuple[IntTuple(UNKNOWN_VALUE)](i)
            ).value
        )
        if not isclose(
            output.ptr[idx], ref_output.ptr[idx], atol=1e-5, rtol=1e-4
        ):
            if mismatches == 0:
                print(
                    "Found mismatches for",
                    cfg.batch_dims,
                    cfg.seq_len,
                    cfg.kv_num_heads,
                    cfg.kv_seq_len,
                    cfg.depth_dim,
                )

            print(
                "Mismatch at",
                idx,
                output.ptr[idx],
                ref_output.ptr[idx],
            )

            mismatches = mismatches + 1
            if mismatches > 5:
                break


def build_ndbuffer[
    dtype: DType,
    rank: Int,
    *,
    static_shape: IndexList[rank] = IndexList[rank](fill=UNKNOWN_VALUE),
](shape: IndexList[rank]) raises -> LayoutTensor[
    dtype, Layout.row_major(static_shape), MutUntrackedOrigin
]:
    var ptr = alloc[Scalar[dtype]](shape.flattened_length())
    rand(ptr, shape.flattened_length())
    return LayoutTensor[dtype, Layout.row_major(static_shape)](
        ptr, RuntimeLayout[Layout.row_major(static_shape)].row_major(shape)
    )


def test_case[
    dtype: DType,
    batch_rank: Int,
    *,
    output_static_shape: IndexList[batch_rank + 2] = IndexList[batch_rank + 2](
        fill=UNKNOWN_VALUE
    ),
](cfg: TestCaseConfig[batch_rank]) raises:
    seed(42)

    # Allocate the QKV tensors.
    var q_shape = cfg.build_shape_bshd(cfg.seq_len, cfg.depth_dim)
    var kv_shape = cfg.build_shape_bshd[is_kv=True](
        cfg.kv_seq_len, cfg.depth_dim
    )
    var q = build_ndbuffer[dtype](q_shape)
    var k = build_ndbuffer[dtype](kv_shape)
    var v = build_ndbuffer[dtype](kv_shape)

    # Allocate the attention mask.
    var mask = build_ndbuffer[dtype](
        cfg.build_shape(cfg.seq_len, cfg.kv_seq_len)
    )

    # Allocate output and reference output buffers.
    var output = build_ndbuffer[dtype, static_shape=output_static_shape](
        q_shape
    )
    var ref_output = build_ndbuffer[dtype](q_shape)

    reference_attention_bshd(q, k, v, mask, ref_output, cfg.scale)

    @__parameter
    @always_inline
    def input_k_fn[
        simd_width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[dtype, simd_width]:
        return k.load[width=simd_width](rebind[IndexList[k.rank]](idx))

    @__parameter
    @always_inline
    def input_v_fn[
        simd_width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[dtype, simd_width]:
        return v.load[width=simd_width](rebind[IndexList[v.rank]](idx))

    @__parameter
    @always_inline
    def mask_fn[
        simd_width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[dtype, simd_width]:
        return mask.load[width=simd_width](rebind[IndexList[mask.rank]](idx))

    flash_attention[input_k_fn, input_v_fn, mask_fn](
        q,
        k.runtime_layout.shape.value.canonicalize(),
        v.runtime_layout.shape.value.canonicalize(),
        mask.runtime_layout.shape.value.canonicalize(),
        LayoutTensor[output.dtype, Layout.row_major[output.rank]()](
            output.ptr,
            RuntimeLayout[Layout.row_major[output.rank]()].row_major(
                output.runtime_layout.shape.value.canonicalize()
            ),
        ),
        cfg.scale,
    )

    verify_output(output, ref_output, cfg)

    q.ptr.free()
    k.ptr.free()
    v.ptr.free()
    mask.ptr.free()
    output.ptr.free()
    ref_output.ptr.free()


def test_flash_attention[dtype: DType]() raises:
    test_case[dtype](
        TestCaseConfig(
            batch_dims=Index(1, 8),
            seq_len=1,
            kv_num_heads=8,
            kv_seq_len=503,
            depth_dim=128,
            scale=0.125,
        )
    )
    test_case[dtype](
        TestCaseConfig(
            batch_dims=Index(4, 12),
            seq_len=1,
            kv_num_heads=4,
            kv_seq_len=503,
            depth_dim=64,
            scale=0.125,
        )
    )
    test_case[dtype](
        TestCaseConfig(
            batch_dims=Index(2, 3),
            seq_len=128,
            kv_num_heads=3,
            kv_seq_len=128,
            depth_dim=63,
            scale=0.25,
        )
    )
    test_case[dtype](
        TestCaseConfig(
            batch_dims=Index(8),
            seq_len=64,
            kv_num_heads=1,
            kv_seq_len=64,
            depth_dim=384,
            scale=0.25,
        )
    )
    test_case[
        dtype,
        batch_rank=1,
        output_static_shape=IndexList[3](UNKNOWN_VALUE, UNKNOWN_VALUE, 128),
    ](
        TestCaseConfig(
            batch_dims=Index(1),
            seq_len=55,
            kv_num_heads=1,
            kv_seq_len=127,
            depth_dim=128,
            scale=0.2,
        )
    )
    test_case[
        dtype,
        batch_rank=1,
        output_static_shape=IndexList[3](UNKNOWN_VALUE, UNKNOWN_VALUE, 160),
    ](
        TestCaseConfig(
            batch_dims=Index(1),
            seq_len=100,
            kv_num_heads=1,
            kv_seq_len=300,
            depth_dim=160,
            scale=0.1,
        )
    )
    test_case[
        dtype,
        batch_rank=1,
        output_static_shape=IndexList[3](UNKNOWN_VALUE, UNKNOWN_VALUE, 300),
    ](
        TestCaseConfig(
            batch_dims=Index(1),
            seq_len=100,
            kv_num_heads=1,
            kv_seq_len=64,
            depth_dim=300,
            scale=0.1,
        )
    )


def test_case_split_kv[
    dtype: DType,
    batch_rank: Int,
    output_static_shape: IndexList[batch_rank + 2] = IndexList[batch_rank + 2](
        fill=UNKNOWN_VALUE
    ),
](cfg: TestCaseConfig[batch_rank]) raises:
    # For now only allow Q.shape = [B, S, H, D].
    comptime assert batch_rank == 2

    seed(42)

    # Allocate the QKV tensors.
    var q_shape = cfg.build_shape_bshd(cfg.seq_len, cfg.depth_dim)
    var kv_shape = cfg.build_shape_bshd[is_kv=True](
        cfg.kv_seq_len, cfg.depth_dim
    )
    var q = build_ndbuffer[dtype](q_shape)
    var k = build_ndbuffer[dtype](kv_shape)
    var v = build_ndbuffer[dtype](kv_shape)

    # Allocate the attention mask.
    var mask = build_ndbuffer[dtype](
        cfg.build_shape(cfg.seq_len, cfg.kv_seq_len)
    )

    # Allocate output and reference output buffers.
    var output = build_ndbuffer[dtype, static_shape=output_static_shape](
        q_shape
    )
    var ref_output = build_ndbuffer[dtype](q_shape)

    # Compute reference outputs for comparison.
    reference_attention_bshd(q, k, v, mask, ref_output, cfg.scale)

    # Define input lambdas for split KV cache attn `flash_attention_split_kv`.
    @__parameter
    @always_inline
    def input_k_fn[
        simd_width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[dtype, simd_width]:
        return k.load[width=simd_width](
            IndexList[k.rank](
                idx[0], idx[1] + cfg.prev_seq_len(), idx[2], idx[3]
            )
        )

    @__parameter
    @always_inline
    def input_v_fn[
        simd_width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[dtype, simd_width]:
        return v.load[width=simd_width](
            IndexList[v.rank](
                idx[0], idx[1] + cfg.prev_seq_len(), idx[2], idx[3]
            )
        )

    @__parameter
    @always_inline
    def input_k_cache_fn[
        simd_width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[dtype, simd_width]:
        return k.load[width=simd_width](
            IndexList[k.rank](idx[1], idx[3], idx[2], idx[4])
        )

    @__parameter
    @always_inline
    def input_v_cache_fn[
        simd_width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[dtype, simd_width]:
        return v.load[width=simd_width](
            IndexList[v.rank](idx[1], idx[3], idx[2], idx[4])
        )

    @__parameter
    @always_inline
    def mask_fn[
        simd_width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[dtype, simd_width]:
        return mask.load[width=simd_width](rebind[IndexList[mask.rank]](idx))

    var kv_present_shape = cfg.build_shape_bshd[is_kv=True](
        cfg.seq_len, cfg.depth_dim
    )
    var kv_past_shape = cfg.build_shape[
        shape_rank=cfg.kv_cache_rank, is_kv=True
    ](cfg.prev_seq_len(), cfg.depth_dim)

    flash_attention_split_kv[
        dtype=dtype,
        rank=batch_rank + 2,
        input_k_fn,
        input_v_fn,
        input_k_cache_fn,
        input_v_cache_fn,
        mask_fn,
    ](
        q,
        kv_present_shape,
        kv_present_shape,
        rebind[IndexList[cfg.rank + 1]](kv_past_shape),
        rebind[IndexList[cfg.rank + 1]](kv_past_shape),
        mask.runtime_layout.shape.value.canonicalize(),
        LayoutTensor[output.dtype, Layout.row_major[output.rank]()](
            output.ptr,
            RuntimeLayout[Layout.row_major[output.rank]()].row_major(
                output.runtime_layout.shape.value.canonicalize()
            ),
        ),
        cfg.scale,
    )

    verify_output(output, ref_output, cfg)

    q.ptr.free()
    k.ptr.free()
    v.ptr.free()
    mask.ptr.free()
    output.ptr.free()
    ref_output.ptr.free()


def test_flash_attention_split_kv[dtype: DType]() raises:
    for kv_seq_len in range(1, 128):
        test_case_split_kv[dtype](
            TestCaseConfig(
                batch_dims=Index(1, 1),
                seq_len=1,
                kv_num_heads=1,
                kv_seq_len=kv_seq_len,
                depth_dim=1,
                scale=0.125,
            )
        )
    test_case_split_kv[dtype](
        TestCaseConfig(
            batch_dims=Index(1, 8),
            seq_len=1,
            kv_num_heads=8,
            kv_seq_len=503,
            depth_dim=128,
            scale=0.125,
        )
    )
    test_case_split_kv[dtype](
        TestCaseConfig(
            batch_dims=Index(1, 8),
            seq_len=1,
            kv_num_heads=2,
            kv_seq_len=503,
            depth_dim=128,
            scale=0.125,
        )
    )
    test_case_split_kv[dtype](
        TestCaseConfig(
            batch_dims=Index(5, 24),
            seq_len=23,
            kv_num_heads=8,
            kv_seq_len=57,
            depth_dim=128,
            scale=0.125,
        )
    )
    test_case_split_kv[dtype](
        TestCaseConfig(
            batch_dims=Index(2, 3),
            seq_len=128,
            kv_num_heads=3,
            kv_seq_len=128,
            depth_dim=63,
            scale=0.25,
        )
    )


def test_flash_attention_with_sinks[dtype: DType]() raises:
    """Test flash attention with and without sink weights."""
    print("Testing flash attention with sink weights...")

    # Simple test configuration
    var batch_size = 1
    var seq_len = 4
    var kv_seq_len = 8
    var num_heads = 2
    var depth_dim = 16
    var scale = Float32(0.125)

    seed(42)

    # Create test tensors in BSHD format
    var q_shape = Index(batch_size, seq_len, num_heads, depth_dim)
    var kv_shape = Index(batch_size, kv_seq_len, num_heads, depth_dim)
    var mask_shape = Index(batch_size, num_heads, seq_len, kv_seq_len)

    var q = build_ndbuffer[dtype](q_shape)
    var k = build_ndbuffer[dtype](kv_shape)
    var v = build_ndbuffer[dtype](kv_shape)
    var mask = build_ndbuffer[dtype](mask_shape)

    # Create sink weights (one per head)
    var sink_weights_shape = Index(num_heads)
    var sink_weights = build_ndbuffer[dtype](sink_weights_shape)

    # Fill sink weights with known values
    for i in range(num_heads):
        sink_weights[i] = Scalar[dtype](
            0.5 * Float64((i + 1))
        )  # 0.5, 1.0 for heads 0, 1

    # Test 1: Regular attention without sinks
    var output_no_sinks = build_ndbuffer[dtype](q_shape)
    var ref_output_no_sinks = build_ndbuffer[dtype](q_shape)

    # Compute reference without sinks
    reference_attention_bshd(q, k, v, mask, ref_output_no_sinks, scale)

    # Test flash attention without sinks
    @__parameter
    @always_inline
    def input_k_fn[
        simd_width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[dtype, simd_width]:
        return k.load[width=simd_width](rebind[IndexList[k.rank]](idx))

    @__parameter
    @always_inline
    def input_v_fn[
        simd_width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[dtype, simd_width]:
        return v.load[width=simd_width](rebind[IndexList[v.rank]](idx))

    @__parameter
    @always_inline
    def mask_fn[
        simd_width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[dtype, simd_width]:
        return mask.load[width=simd_width](rebind[IndexList[mask.rank]](idx))

    # Call without sink weights
    flash_attention[input_k_fn, input_v_fn, mask_fn](
        q,
        k.runtime_layout.shape.value.canonicalize(),
        v.runtime_layout.shape.value.canonicalize(),
        mask.runtime_layout.shape.value.canonicalize(),
        output_no_sinks,
        scale,
    )

    # Verify no-sinks result
    var mismatches_no_sinks = 0
    for i in range(output_no_sinks.size()):
        if not isclose(
            output_no_sinks.ptr[i],
            ref_output_no_sinks.ptr[i],
            atol=1e-5,
            rtol=1e-4,
        ):
            if mismatches_no_sinks < 3:
                print(
                    "No-sinks mismatch at",
                    i,
                    ":",
                    output_no_sinks.ptr[i],
                    "vs",
                    ref_output_no_sinks.ptr[i],
                )
            mismatches_no_sinks += 1

    if mismatches_no_sinks == 0:
        print("✓ Flash attention without sinks passed")
    else:
        print(
            "✗ Flash attention without sinks failed with",
            mismatches_no_sinks,
            "mismatches",
        )

    # Test 2: Attention with sinks
    var output_with_sinks = build_ndbuffer[dtype](q_shape)
    var ref_output_with_sinks = build_ndbuffer[dtype](q_shape)

    # Compute reference with sinks
    reference_attention_bshd_with_sinks(
        q,
        k,
        v,
        mask,
        LayoutTensor[sink_weights.dtype, Layout.row_major(UNKNOWN_VALUE), _](
            sink_weights.ptr,
            RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
                sink_weights.runtime_layout.shape.value
            ),
        ),
        ref_output_with_sinks,
        scale,
    )

    # Call with sink weights (pass the data pointer wrapped in Optional)
    flash_attention[input_k_fn, input_v_fn, mask_fn](
        q,
        k.runtime_layout.shape.value.canonicalize(),
        v.runtime_layout.shape.value.canonicalize(),
        mask.runtime_layout.shape.value.canonicalize(),
        output_with_sinks,
        scale,
        sink_weights=LayoutTensor[
            sink_weights.dtype,
            Layout.row_major(UNKNOWN_VALUE),
        ](
            sink_weights.ptr.as_imm().as_unsafe_any_origin(),
            RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
                sink_weights.runtime_layout.shape.value
            ),
        ),
    )

    # Verify sinks result
    var mismatches_with_sinks = 0
    for i in range(output_with_sinks.size()):
        if not isclose(
            output_with_sinks.ptr[i],
            ref_output_with_sinks.ptr[i],
            atol=1e-5,
            rtol=1e-4,
        ):
            if mismatches_with_sinks < 3:
                print(
                    "With-sinks mismatch at",
                    i,
                    ":",
                    output_with_sinks.ptr[i],
                    "vs",
                    ref_output_with_sinks.ptr[i],
                )
            mismatches_with_sinks += 1

    if mismatches_with_sinks == 0:
        print("✓ Flash attention with sinks passed")
    else:
        print(
            "✗ Flash attention with sinks failed with",
            mismatches_with_sinks,
            "mismatches",
        )

    # Free memory
    q.ptr.free()
    k.ptr.free()
    v.ptr.free()
    mask.ptr.free()
    sink_weights.ptr.free()
    output_no_sinks.ptr.free()
    ref_output_no_sinks.ptr.free()
    output_with_sinks.ptr.free()
    ref_output_with_sinks.ptr.free()


def main() raises:
    test_flash_attention[.float32]()
    test_flash_attention_split_kv[.float32]()
    test_flash_attention_with_sinks[.float32]()
