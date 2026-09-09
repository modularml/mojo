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
"""Per-row phase attribution for the indexer top-k radix select.

Stamps `global_perf_counter_ns` at every phase boundary of the select into a
per-row trace buffer and prints it, so the kernel's time splits into "round `r`
row scan", "round `r` split", "append" and "sort" instead of one opaque total.
Reads the same input as `bench_topk_mt`, so a phase attribution describes the
run that was benchmarked rather than a different one.

Traces whichever kernel the decode dispatch would pick for the shape *and* the
output contract -- `histsel_resident_topk` at whichever payload width holds the
row, `histsel_topk` with prefetch and narrow refine digits otherwise. Tracing a
configuration the dispatch never launches attributes phases that nothing runs,
which is why the contract is a flag here and not a fixed choice.

    trace_topk_bitonic --rows=48 --N=14336 --K=2048 --dist=q17 \
        --unordered=True --deterministic=False
"""

from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, FuncAttribute
from internal_utils import arg_parse
from layout import TileTensor, row_major
from structured_kernels.trace_buf import GmemTrace

from nn.topk_bitonic import (
    HSEL_TRACE_EVENTS,
    _histsel_resident_kernel,
    _histsel_topk_kernel,
    _HSEL_RANK_BITS,
    _HSEL_RES_BLOCK,
    _HSEL_RES_MAX,
    _HSEL_RES_MAX_WIDE,
    _HSEL_RES_VECS,
    _HSEL_RES_VECS_WIDE,
    _HSEL_SEL_CAP,
    _HSEL_SMEM_BYTES,
    _HSEL_TAIL_BITS,
    _PTOPK_BLOCK,
    _PTOPK_TOTAL,
)


@always_inline
def _mix64(x: UInt64) -> UInt64:
    """SplitMix64's finalizer."""
    var z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


@always_inline
def _u01(i: UInt64) -> Float64:
    var h = _mix64((i + 1) * 0x9E3779B97F4A7C15)
    return Float64(h >> 11) * (1.0 / 9007199254740992.0)


@always_inline
def _u_val(r: Int, c: Int, N: Int, k: Int) -> Float64:
    return _u01(((UInt64(r) * UInt64(N) + UInt64(c)) << 2) + UInt64(k))


@always_inline
def _u_len(r: Int) -> Float64:
    return _u01(0x8000000000000000 + UInt64(r))


comptime _IH_TERMS: Int = 6
comptime _IH_SCALE: Float64 = 1.4142135623730951


def _sample(dist: String, r: Int, c: Int, N: Int) -> Float32:
    """One score, generated exactly as `bench_topk_mt` generates it.

    Carried verbatim rather than approximated: the select's cost is set by how
    many columns share the threshold's coarse bin, so a phase attributed on a
    distribution that merely resembles the benchmarked one describes a different
    kernel behaviour. `_u_len` reproduces the same masked-suffix lengths too.
    """
    var u = _u_val(r, c, N, 0)
    if dist == "q17":
        return Float32(Int(u * 17.0)) / 16.0
    if dist == "uniform":
        return Float32(u * 2.0 - 1.0)
    if dist == "narrow":
        return Float32(1.0 + u / 32.0)
    # One index computation for the whole sum: only the term number varies.
    var base = (UInt64(r) * UInt64(N) + UInt64(c)) << 2
    var acc = Float64(0.0)
    for k in range(_IH_TERMS):
        acc += _u01(base + UInt64(k))
    return Float32((acc - Float64(_IH_TERMS) / 2.0) * _IH_SCALE)


def _launch[
    *,
    unordered: Bool = False,
    deterministic: Bool = True,
](
    ctx: DeviceContext,
    scores_t: TileTensor[.float32, ...],
    idxs_t: TileTensor[.int32, ...],
    trace_ptr: MutPointer[UInt64, MutUntrackedOrigin],
    N: Int,
    K: Int,
    rows: Int,
) raises:
    comptime if unordered:

        @__parameter
        @always_inline
        def resident[res_vecs: Int]() raises:
            ctx.enqueue_function[
                _histsel_resident_kernel[
                    GmemTrace,
                    enable_trace=True,
                    sel_cap=_PTOPK_TOTAL,
                    ordered=False,
                    deterministic=deterministic,
                    res_vecs=res_vecs,
                ]
            ](
                rebind[ImmPointer[Float32, ImmutAnyOrigin]](scores_t.ptr),
                rebind[MutPointer[Int32, MutAnyOrigin]](idxs_t.ptr),
                Int32(N),
                Int32(K),
                GmemTrace(trace_ptr),
                grid_dim=rows,
                block_dim=_HSEL_RES_BLOCK,
            )

        if N <= _HSEL_RES_MAX:
            resident[_HSEL_RES_VECS]()
            return
        if N <= _HSEL_RES_MAX_WIDE:
            resident[_HSEL_RES_VECS_WIDE]()
            return

    if N <= _HSEL_RES_MAX and N < K + K // 2:
        ctx.enqueue_function[
            _histsel_resident_kernel[
                GmemTrace, enable_trace=True, bin_digit=True
            ]
        ](
            rebind[ImmPointer[Float32, ImmutAnyOrigin]](scores_t.ptr),
            rebind[MutPointer[Int32, MutAnyOrigin]](idxs_t.ptr),
            Int32(N),
            Int32(K),
            GmemTrace(trace_ptr),
            grid_dim=rows,
            block_dim=_HSEL_RES_BLOCK,
        )
        return

    if N <= _HSEL_RES_MAX:
        ctx.enqueue_function[
            _histsel_resident_kernel[GmemTrace, enable_trace=True]
        ](
            rebind[ImmPointer[Float32, ImmutAnyOrigin]](scores_t.ptr),
            rebind[MutPointer[Int32, MutAnyOrigin]](idxs_t.ptr),
            Int32(N),
            Int32(K),
            GmemTrace(trace_ptr),
            grid_dim=rows,
            block_dim=_HSEL_RES_BLOCK,
        )
        return

    # Past the resident widths the fully relaxed contract streams as well, and it
    # is the only one that skips the rank there. Tracing it with the ordered
    # instantiation would attribute a rank the dispatch never launches, which is
    # the one thing this tool exists not to do -- and every rung of the multi-turn
    # benchmark past the first lands here.
    comptime if unordered and not deterministic:
        ctx.enqueue_function[
            _histsel_topk_kernel[
                GmemTrace,
                enable_trace=True,
                prefetch=True,
                tail_bits=_HSEL_TAIL_BITS,
                ordered=False,
                deterministic=False,
            ]
        ](
            rebind[ImmPointer[Float32, ImmutAnyOrigin]](scores_t.ptr),
            rebind[MutPointer[Int32, MutAnyOrigin]](idxs_t.ptr),
            Int32(N),
            Int32(K),
            GmemTrace(trace_ptr),
            grid_dim=rows,
            block_dim=_PTOPK_BLOCK,
            shared_mem_bytes=_HSEL_SMEM_BYTES,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(_HSEL_SMEM_BYTES)
            ),
        )
        return

    ctx.enqueue_function[
        _histsel_topk_kernel[
            GmemTrace,
            enable_trace=True,
            prefetch=True,
            tail_bits=_HSEL_TAIL_BITS,
            rank_bits=_HSEL_RANK_BITS,
            rank_slots=True,
            sel_cap=_HSEL_SEL_CAP,
        ]
    ](
        rebind[ImmPointer[Float32, ImmutAnyOrigin]](scores_t.ptr),
        rebind[MutPointer[Int32, MutAnyOrigin]](idxs_t.ptr),
        Int32(N),
        Int32(K),
        GmemTrace(trace_ptr),
        grid_dim=rows,
        block_dim=_PTOPK_BLOCK,
        shared_mem_bytes=_HSEL_SMEM_BYTES,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(_HSEL_SMEM_BYTES)
        ),
    )


def main() raises:
    var rows = arg_parse("rows", 48)
    var N = arg_parse("N", 107228)
    var K = arg_parse("K", 2048)
    var dist = arg_parse("dist", String("q17"))
    var unordered = arg_parse("unordered", False)
    var deterministic = arg_parse("deterministic", True)

    with DeviceContext() as ctx:
        var scores_buf = ctx.enqueue_create_buffer[.float32](rows * N)
        var idxs_buf = ctx.enqueue_create_buffer[.int32](rows * K)
        var trace_buf = ctx.enqueue_create_buffer[.uint64](
            rows * HSEL_TRACE_EVENTS
        )

        # `bench_topk_mt`'s own generator, keyed by index so the traced run and
        # the benchmarked run are the same array -- see `_sample`.
        with scores_buf.map_to_host() as h:
            for r in range(rows):
                var nk = N - Int(_u_len(r) * Float64(N) / 8.0)
                for c in range(N):
                    if c < nk:
                        h[r * N + c] = Float32(_sample(dist, r, c, N))
                    else:
                        h[r * N + c] = Float32(-3.0e38)
        ctx.enqueue_memset(trace_buf, 0)

        var scores_t = TileTensor(scores_buf, row_major(rows, N))
        var idxs_t = TileTensor(idxs_buf, row_major(rows, K))
        ctx.synchronize()

        var trace_ptr = rebind[MutPointer[UInt64, MutUntrackedOrigin]](
            trace_buf.unsafe_ptr()
        )

        # Warm up before the traced launch: a cold launch's first row scan
        # measures instruction-cache and page-table misses, not the kernel.
        for _ in range(20):
            if unordered and not deterministic:
                _launch[unordered=True, deterministic=False](
                    ctx, scores_t, idxs_t, trace_ptr, N, K, rows
                )
            elif unordered:
                _launch[unordered=True](
                    ctx, scores_t, idxs_t, trace_ptr, N, K, rows
                )
            else:
                _launch[unordered=False](
                    ctx, scores_t, idxs_t, trace_ptr, N, K, rows
                )
        ctx.synchronize()

        var t0 = perf_counter_ns()
        var iters = arg_parse("iters", 20)
        for _ in range(iters):
            if unordered and not deterministic:
                _launch[unordered=True, deterministic=False](
                    ctx, scores_t, idxs_t, trace_ptr, N, K, rows
                )
            elif unordered:
                _launch[unordered=True](
                    ctx, scores_t, idxs_t, trace_ptr, N, K, rows
                )
            else:
                _launch[unordered=False](
                    ctx, scores_t, idxs_t, trace_ptr, N, K, rows
                )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        print("TIME_MS", Float64(t1 - t0) / Float64(iters) / 1.0e6)

        print(
            "TRACE",
            rows,
            N,
            K,
            dist,
            HSEL_TRACE_EVENTS,
            unordered,
        )
        with trace_buf.map_to_host() as h:
            for r in range(rows):
                var line = String("T ")
                line += String(r)
                for e in range(HSEL_TRACE_EVENTS):
                    line += " "
                    line += String(h[r * HSEL_TRACE_EVENTS + e])
                print(line)
        print("TRACEDONE")

        _ = scores_buf
        _ = idxs_buf
        _ = trace_buf
