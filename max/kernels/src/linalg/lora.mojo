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

"""Provides LoRA (Low-Rank Adaptation) grouped matmul kernels for SM100 GPUs."""

from max.gpu.host import DeviceContext
from linalg.grouped_matmul import grouped_matmul, naive_grouped_matmul

from std.utils import Index, IndexList
from layout import (
    Coord,
    Idx,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)


@always_inline
def shrink_qkv_permute_3mn_sm100(
    c_lora: TileTensor[mut=True, address_space=.GENERIC, ...],
    a: TileTensor[mut=False, address_space=.GENERIC, ...],
    b: TileTensor[mut=False, address_space=.GENERIC, ...],
    a_offsets: TileTensor[mut=False, .uint32, address_space=.GENERIC, ...],
    expert_ids: TileTensor[mut=False, .int32, address_space=.GENERIC, ...],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    """TileTensor primary implementation of `shrink_qkv_permute_3mn_sm100`.

    LoRA shrink GMM with planar Q/K/V output on SM100.

    Performs the LoRA 'shrink' grouped matmul for routed tokens:
    computes `[M, K] @ [G, 3N, K]^T` per active expert, then **permutes**
    the flat `[M, 3N]` result into a planar layout `[3, M, N]` (Q, K, V)
    using an elementwise epilogue, while reusing the same storage.

    Args:
        c_lora: Output tensor with planar Q/K/V layout, shape (3, M, N).
                Backed by row-major storage, used both as a 3D view and as a
                temporary 2D view (M, 3N) during compute.
        a:      Routed activation matrix, shape (M, K).
        b:      Shrink weights per expert, shape (G, 3N, K).
        a_offsets: Inclusive prefix sums of tokens per (active) expert,
                length (num_experts + 1). Defines per-expert [start, end) in A/C.
        expert_ids: Expert indices for the active groups, length >= num_active_experts.
        max_num_tokens_per_expert: Upper bound on tokens for any active expert.
        num_active_experts: Number of experts participating in this call.
        ctx:    DeviceContext used for enqueues and synchronization.

    Constraints:
        - c_lora must be rank 3 with static first dimension B == 3.
        - a must be rank 2 with trailing dimension K that matches b[..., K].
        - b must be rank 3 with shape (G, 3N, K).
        - The temporary 2D view of c_lora is (M, 3N) in row-major order and
        **aliases the same storage** as c_lora.
        - a_offsets is non-decreasing with a_offsets[0] == 0 and
        a_offsets[num_active_experts] == M.
        - expert_ids[i] in [0, G) for valid experts; kernel may treat -1 as inactive.
        - The epilogue assumes `N % vector_width == 0` for aligned vector stores.
    """
    comptime assert c_lora.rank == 3 and c_lora.flat_rank == 3
    comptime assert a.rank == 2 and a.flat_rank == 2
    comptime assert b.rank == 3 and b.flat_rank == 3
    comptime assert a_offsets.rank == 1 and a_offsets.flat_rank == 1
    comptime assert expert_ids.rank == 1 and expert_ids.flat_rank == 1

    comptime c_type = c_lora.dtype

    comptime N = c_lora.static_shape[2]
    comptime B = c_lora.static_shape[0]
    comptime assert N != UNKNOWN_VALUE and B == 3, String(
        "the outer dimension of c_shape must be known and equal to 3",
    )

    var M = Int(c_lora.dim(1))
    var c_tensor_lora = c_lora.to_layout_tensor()
    comptime N_Total = B * N
    # Create a dangling TileTensor for C. This ensures GroupGEMM does NOT
    # write into C directly; any changes to the final C output must happen
    # exclusively via the epilogue function.
    var c = TileTensor(
        UnsafePointer[Scalar[c_type], MutUntrackedOrigin].unsafe_dangling(),
        row_major(Coord(M, N_Total)),
    )

    @always_inline
    @__copy_capture(c_tensor_lora, M)
    @__parameter
    def permute_dim_lora_bmn[
        dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width]) -> None:
        """Epilogue: permute flat (M, 3N) columns to planar (3, M, N) tiles.
        Maps a flat column index `j` into `(head, n)` via `divmod(j, N)` and
        stores the SIMD vector `val` into the original 3D layout tensor at
        `[head, m, n + lane]`. Used as the elementwise epilogue for the
        grouped matmul, so the final `c_lora` is written directly in
        planar Q/K/V format without an extra kernel.

        Args:
            idx: 2D index of the epilogue write in the temporary (M, 3N) view,
                where `idx[0] = m` and `idx[1] = j`.
            val: SIMD vector of partial results to be written for columns
                `j .. j+width-1` at row `m`.

        Constraints:
            - `N` is the per-head width; must satisfy `N % width == 0` for aligned
            and in-bounds vector stores.
            - The underlying storage of `c_tensor_lora` aliases the (M, 3N) view.
            - Rank/layout assumptions:
                * Input view is row-major (M, 3N).
                * Output view is row-major (3, M, N) with head-major tiles.
        """
        comptime N = c_lora.static_shape[2]
        var i = idx[0]
        var j = idx[1]
        var new_j, new_k = divmod(j, N)
        comptime assert N % width == 0, "N must be divisible by width"
        # The current index is [i, new_j, new_k] in the M x 3 x N row major
        # tensor.
        # The permdim tensor has the shape 3 x M x N, so the index is then
        # [new_j, i, new_k].
        var off = c_tensor_lora._offset(IndexList[3](new_j, i, new_k))
        c_tensor_lora.ptr.store[width=width, alignment=alignment](
            off, val.cast[c_type]()
        )

    # Run grouped_matmul and apply permute_dim_lora as the elementwise epilogue.
    grouped_matmul[elementwise_lambda_fn=permute_dim_lora_bmn,](
        c,
        a,
        b,
        a_offsets,
        expert_ids,
        max_num_tokens_per_expert,
        num_active_experts,
        ctx,
    )


# ===----------------------------------------------------------------------=== #
# LoRA-B expand: single-launch grouped GEMM via load/store specialization
# ===----------------------------------------------------------------------=== #
#
# Replaces the previous TWO grouped matmuls (one for Q, one for the row-stacked
# K/V) with ONE launch over a single fused LoRA-B weight `[G, q_dim+2*kv_dim, R]`,
# GQA-correct (q_dim != kv_dim). Rather than a bespoke kernel, this reuses the
# tuned persistent warp-specialized grouped matmul (`grouped_matmul` ->
# `grouped_matmul_sm100_persistent` on SM100, naive elsewhere) and specializes
# only its gmem load and store:
#
#   - Load (plane select): the matmul A operand is the planar shrink output
#     `P [3, M, R]` viewed as `[3M, R]`. `a_plane_splits` shifts the activation
#     row by `plane(out_col) * M`, so each Q/K/V output-column region contracts
#     against the matching plane of `P` -- the per-region operand switch, done in
#     the load stage. Region boundaries are tile-aligned, so every output tile is
#     single-region.
#   - Store (route): `elementwise_lambda_fn` routes each output element
#     `(token, out_col)` to `q_out` or the row-stacked `kv_out [2M, kv_dim]`
#     (K -> rows [0, M), V -> rows [M, 2M)), matching `kv_cache_ragged_2m_iadd`.
#
# This is the structure called for on PERF-2688: reuse the kernel tuning;
# specialize loading and storing only, no new kernel.


@always_inline
def expand_qkv_sm100(
    q_out: TileTensor[mut=True, address_space=.GENERIC, ...],
    kv_out: TileTensor[mut=True, address_space=.GENERIC, ...],
    p: TileTensor[mut=False, address_space=.GENERIC, ...],
    b: TileTensor[mut=False, address_space=.GENERIC, ...],
    a_offsets: TileTensor[mut=False, .uint32, address_space=.GENERIC, ...],
    expert_ids: TileTensor[mut=False, .int32, address_space=.GENERIC, ...],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    """Single-launch LoRA-B expand grouped GEMM (load/store specialized).

    Computes the LoRA 'expand' (up-projection by the fused LoRA-B weight) for
    routed tokens in ONE grouped-GEMM launch over a single fused weight,
    GQA-correct (`q_dim != kv_dim`). For token row `m` (in adapter group `g`)
    and output column `j` in `[0, q_dim + 2*kv_dim)`:

      - `j < q_dim`                  -> region Q -> dot(B[g, j, :], P[0, m, :])
                                        written to `q_out[m, j]`.
      - `q_dim <= j < q_dim+kv_dim`  -> region K -> dot(B[g, j, :], P[1, m, :])
                                        written to `kv_out[m, j-q_dim]`.
      - otherwise                    -> region V -> dot(B[g, j, :], P[2, m, :])
                                        written to `kv_out[M+m, j-q_dim-kv_dim]`.

    Reuses the tuned persistent grouped matmul: the fused weight is the matmul B
    operand and the planar shrink output `P [3, M, R]` (viewed `[3M, R]`) is the
    A operand. The per-region operand switch (which plane of `P`) is a load-stage
    specialization (`a_plane_splits`, shifting the activation row by `plane * M`);
    the Q/K/V output routing is an output-epilogue specialization
    (`elementwise_lambda_fn`). K and V are row-stacked in `kv_out` (K into rows
    `[0, M)`, V into rows `[M, 2M)`), matching the layout `kv_cache_ragged_2m_iadd`
    consumes downstream.

    Args:
        q_out:   Q output, shape `(M, q_dim)`, region Q.
        kv_out:  K/V output, shape `(2M, kv_dim)`; region K into rows `[0, M)`,
                 region V into rows `[M, 2M)`.
        p:       Shrink output in planar layout, shape `(3, M, R)`
                 (plane 0 = q-lora, 1 = k-lora, 2 = v-lora). This is exactly the
                 output of `shrink_qkv_permute_3mn_sm100`.
        b:       Fused LoRA-B weight per adapter, shape `(G, q_dim + 2*kv_dim, R)`.
                 Per adapter the `q_dim + 2*kv_dim` rows are the output features,
                 partitioned along that axis into Q, then K, then V; `R` (the LoRA
                 rank, the columns) is shared across all three.
        a_offsets: Inclusive prefix sums of tokens per active adapter, length
                 `(num_experts + 1)`. The SAME grouping used by the shrink -- no
                 separate K/V offset array.
        expert_ids: Adapter indices for the active groups, length
                 `>= num_active_experts`; `-1` marks an inactive group.
        max_num_tokens_per_expert: Upper bound on tokens for any active adapter.
        num_active_experts: Number of adapter groups in this call.
        ctx:     DeviceContext used for the enqueue.

    Constraints:
        - `p` must be rank 3 with static shape `(3, M, R)` (R static).
        - `b` must be rank 3 with static shape `(G, q_dim + 2*kv_dim, R)`.
        - `q_out` and `kv_out` are row-major with the same dtype; `q_out` trailing
          dim is `q_dim`, `kv_out` trailing dim is `kv_dim`.
        - `q_dim` and `kv_dim` must be statically known and `> 0`. For the SM100
          tensor-core path they must be tile-aligned (multiples of the
          grouped-matmul `BM` and the epilogue vector width) so no output tile or
          store vector straddles a Q/K/V region boundary.
        - `a_offsets` is non-decreasing with `a_offsets[0] == 0` and
          `a_offsets[num_active_experts] == M`.
    """
    comptime assert p.rank == 3 and p.flat_rank == 3
    comptime assert b.rank == 3 and b.flat_rank == 3
    comptime assert q_out.rank == 2 and q_out.flat_rank == 2
    comptime assert kv_out.rank == 2 and kv_out.flat_rank == 2
    comptime assert a_offsets.rank == 1 and a_offsets.flat_rank == 1
    comptime assert expert_ids.rank == 1 and expert_ids.flat_rank == 1

    comptime R = p.static_shape[2]
    comptime D_total = b.static_shape[1]
    comptime q_dim = q_out.static_shape[1]
    comptime kv_dim = kv_out.static_shape[1]
    comptime q_type = q_out.dtype
    comptime kv_type = kv_out.dtype
    # The grouped matmul produces a single C dtype; use q_out's. `q_type` and
    # `kv_type` are asserted equal below, but they are distinct comptime params,
    # so each store must cast to its own tensor's dtype.
    comptime c_type = q_type

    comptime assert R != UNKNOWN_VALUE, "R (contraction dim) must be static"
    comptime assert q_dim != UNKNOWN_VALUE, "q_dim must be static"
    comptime assert kv_dim != UNKNOWN_VALUE, "kv_dim must be static"
    comptime assert (
        D_total != UNKNOWN_VALUE
    ), "fused weight D dim must be static"
    comptime assert (
        q_out.dtype == kv_out.dtype
    ), "q_out and kv_out must share a dtype (single grouped-matmul C dtype)"
    comptime assert D_total == q_dim + 2 * kv_dim, String(
        "fused weight D dim (b.static_shape[1]=",
        D_total,
        ") must equal q_dim + 2*kv_dim (",
        q_dim + 2 * kv_dim,
        ")",
    )

    var M = Int(p.dim(1))

    # The grouped matmul's two operands:
    #   - B operand: the fused LoRA-B weight `b` (`[G, D_total, R]`), forwarded as-is.
    #   - A operand: the planar shrink output `P [3, M, R]` reinterpreted as `[3M, R]`,
    #     so plane `t` occupies rows `[t*M, (t+1)*M)`; `a_plane_splits` selects it.
    var a_act = TileTensor(p.ptr, row_major(Coord(3 * M, Idx[R])))

    # Dangling C `[M, D_total]`: all results reach `q_out`/`kv_out` through the
    # `route_qkv` epilogue, so the grouped matmul never stores to C directly (the
    # SM100 tensor-core path skips both the C TMA descriptor encode and the
    # inactive-group direct store when an epilogue is present). `Idx[D_total]`
    # keeps `c.static_shape[1]` static so the dispatcher reaches the SM100
    # tensor-core kernel instead of falling back to naive.
    var c = TileTensor(
        UnsafePointer[Scalar[c_type], MutUntrackedOrigin].unsafe_dangling(),
        row_major(Coord(M, Idx[D_total])),
    )

    var q_tensor = q_out.to_layout_tensor()
    var kv_tensor = kv_out.to_layout_tensor()

    # Plane-select boundaries for the load specialization. For output column `j`
    # the activation plane is:
    #   - plane 0 (Q) when `j < q_dim`
    #   - plane 1 (K) when `j < q_dim + kv_dim`
    #   - plane 2 (V) otherwise
    # Passed as comptime data, not a closure: the warp-specialized kernel's GPU
    # slicer can't handle a capturing closure in the load path. The kernel turns
    # the selected plane into a row offset by multiplying it by the runtime token
    # count (M).
    comptime a_plane_splits = Index(q_dim, q_dim + kv_dim)

    @always_inline
    @__copy_capture(q_tensor, kv_tensor, M)
    @__parameter
    def route_qkv[
        dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width]) -> None:
        """Epilogue: scatter each matmul output element to `q_out` or `kv_out`.

        Routes one grouped-matmul result, produced at logical coordinate
        `(token, out_col)` in the `[M, q_dim + 2*kv_dim]` output, to its physical
        destination based on the Q/K/V region `out_col` falls in.

        This is the store half of the load/store specialization: the matmul's C is
        dangling (never stored to directly), so every result reaches `q_out` /
        `kv_out` only through here.

        Parameters:
            dtype: Element dtype of the matmul result vector `val`.
            width: SIMD vector width (number of contiguous `out_col` values).
            alignment: Byte alignment of the destination store.

        Args:
            idx: `(token, out_col)` -- the row (token) and column (output feature)
                of this write in the logical `[M, q_dim + 2*kv_dim]` output.
            val: The `width`-element vector of results spanning
                `out_col .. out_col + width - 1` at row `token`.

        Constraints:
            - `q_dim` and `kv_dim` must be divisible by `width`, so a stored vector
              never straddles a Q/K/V boundary and lands entirely in one region.
            - `q_tensor` / `kv_tensor` alias `q_out` / `kv_out`; `val` is cast to
              each tensor's own dtype (`q_type` / `kv_type`) on store.
        """
        comptime assert q_dim % width == 0, "q_dim must be divisible by width"
        comptime assert kv_dim % width == 0, "kv_dim must be divisible by width"
        var token = idx[0]
        var j = idx[1]
        if j < q_dim:
            var off = q_tensor._offset(IndexList[2](token, j))
            q_tensor.ptr.store[width=width, alignment=alignment](
                off, val.cast[q_type]()
            )
        elif j < q_dim + kv_dim:
            var off = kv_tensor._offset(IndexList[2](token, j - q_dim))
            kv_tensor.ptr.store[width=width, alignment=alignment](
                off, val.cast[kv_type]()
            )
        else:
            var off = kv_tensor._offset(
                IndexList[2](M + token, j - q_dim - kv_dim)
            )
            kv_tensor.ptr.store[width=width, alignment=alignment](
                off, val.cast[kv_type]()
            )

    # The SM100 tensor-core path selects the P-plane once per output-D tile, so a
    # tile must not straddle a Q/K/V boundary. The dispatcher tiles the output-D
    # (kernel M after swapAB) by `MMA_M = 128 * cta_group`, with `cta_group == 2`
    # exactly when `D_total % 256 == 0` (a cluster of 2 cooperates on a 256-wide
    # tile), else `cta_group == 1` (128-wide). So the boundaries `q_dim` and
    # `q_dim + kv_dim` must be multiples of that tile width -- equivalently
    # `q_dim` and `kv_dim` both multiples of it. Real GQA dims (e.g. 4096 / 1024)
    # satisfy this. Otherwise fall back to the naive grouped matmul, which selects
    # the plane per output element (via the same `a_plane_splits`) and is correct
    # for any shape -- the path the prior bespoke expand kernel always took.
    comptime sm100_output_tile = 256 if D_total % 256 == 0 else 128
    comptime boundaries_tile_aligned = (
        q_dim % sm100_output_tile == 0 and kv_dim % sm100_output_tile == 0
    )

    comptime if boundaries_tile_aligned:
        grouped_matmul[
            elementwise_lambda_fn=route_qkv,
            a_plane_splits=a_plane_splits,
        ](
            c,
            a_act,
            b,
            a_offsets,
            expert_ids,
            max_num_tokens_per_expert,
            num_active_experts,
            ctx,
        )
    else:
        naive_grouped_matmul[
            elementwise_lambda_fn=route_qkv,
            a_plane_splits=a_plane_splits,
        ](
            c,
            a_act,
            b,
            a_offsets,
            expert_ids,
            max_num_tokens_per_expert,
            num_active_experts,
            ctx,
        )
