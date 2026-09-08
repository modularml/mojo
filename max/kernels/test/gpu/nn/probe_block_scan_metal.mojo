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
# PROBE: isolate block.prefix_sum / block.sum on Metal in the
# >1-warp regime (the topk_fi sampler's geometry: block_size=1024 => 32 warps,
# but only the first 128 threads = 4 warps carry data).
#
# This validates the SHARED primitives in isolation, away from the hang-prone
# topk test. It checks:
#   (1) block.prefix_sum correctness vs host reference (all-ones, 1024 active).
#   (2) block.prefix_sum correctness in the MASKED 4-warp regime (only tx<128
#       carry data) -- the exact topk geometry. Decisive for the cross-warp
#       carry past warp 4.
#   (3) CONSISTENCY: the last active thread's inclusive CDF must equal block.sum
#       total. This is the precise invariant device_sampling_from_prob relies on
#       (topk_fi.mojo:478 gate uses block.sum; :504 boundary uses prefix_sum).
#
# Run: source utils/start-modular.sh; mojo <thisfile>
# ===----------------------------------------------------------------------=== #

from max.gpu import thread_idx
from max.gpu.host import DeviceContext
from max.gpu.primitives import block
from std.testing import assert_equal, assert_almost_equal, TestSuite


def block_scan_kernel[
    block_size: Int
](
    inp: MutPointer[Float32, MutAnyOrigin],
    out_incl: MutPointer[Float32, MutAnyOrigin],
    out_excl: MutPointer[Float32, MutAnyOrigin],
    out_sum: MutPointer[Float32, MutAnyOrigin],
):
    """Per-thread: read input, run block.sum + block.prefix_sum, write results.

    Single block of `block_size` threads. Mirrors device_sampling_from_prob's
    use of block.sum (broadcast=True) and block.prefix_sum (exclusive=True),
    and additionally captures the inclusive scan to compare against the total.
    """
    var tx = thread_idx.x
    var v = inp[tx]

    # block.sum (broadcast=True), mirroring topk_fi.mojo:472.
    var total = block.sum[block_size=block_size, broadcast=True](v)

    # block.prefix_sum exclusive, mirroring topk_fi.mojo:490.
    var excl = block.prefix_sum[
        dtype=DType.float32, block_size=block_size, exclusive=True
    ](v)

    # Inclusive CDF reconstructed the way the kernel does it: exclusive prefix
    # of previous threads + this thread's own value (topk_fi.mojo:497-499 add
    # the thread-local inclusive partial; here each thread carries one scalar so
    # inclusive = exclusive + v).
    var incl = excl + v

    out_excl[tx] = excl
    out_incl[tx] = incl
    out_sum[tx] = total


def run_case[
    block_size: Int
](
    name: String,
    n_active: Int,
    fill_active: Float32,
    ctx: DeviceContext,
) raises:
    """Run one probe case. tx in [0, n_active) get fill_active, rest get 0."""
    var inp = ctx.enqueue_create_buffer[.float32](block_size)
    var out_incl = ctx.enqueue_create_buffer[.float32](block_size)
    var out_excl = ctx.enqueue_create_buffer[.float32](block_size)
    var out_sum = ctx.enqueue_create_buffer[.float32](block_size)

    # Host input + reference.
    var host_in = Array[Float32, block_size](fill=0)
    var ref_incl = Array[Float32, block_size](fill=0)
    var ref_excl = Array[Float32, block_size](fill=0)
    var running = Float32(0)
    for i in range(block_size):
        var val = fill_active if i < n_active else Float32(0)
        host_in[i] = val
        ref_excl[i] = running
        running += val
        ref_incl[i] = running
    var ref_total = running

    inp.enqueue_copy_from(Span(host_in))
    out_incl.enqueue_fill(-1.0e30)
    out_excl.enqueue_fill(-1.0e30)
    out_sum.enqueue_fill(-1.0e30)

    ctx.enqueue_function[block_scan_kernel[block_size]](
        inp,
        out_incl,
        out_excl,
        out_sum,
        grid_dim=1,
        block_dim=block_size,
    )

    var got_incl = Array[Float32, block_size](fill=0)
    var got_excl = Array[Float32, block_size](fill=0)
    var got_sum = Array[Float32, block_size](fill=0)
    out_incl.enqueue_copy_to(Span(got_incl))
    out_excl.enqueue_copy_to(Span(got_excl))
    out_sum.enqueue_copy_to(Span(got_sum))
    ctx.synchronize()

    # --- Diagnostics: print the boundary structure before asserting. ---
    print("=== CASE:", name, "(n_active=", n_active, ") ===")
    print("  ref_total =", ref_total)
    print("  got_sum[0]            =", got_sum[0], "(block.sum, tx=0)")
    print(
        "  got_sum[1023]         =",
        got_sum[block_size - 1],
        "(block.sum, last)",
    )
    print(
        "  got_incl[n_active-1]  =",
        got_incl[n_active - 1],
        "(prefix_sum inclusive at last ACTIVE thread; should == ref_total)",
    )
    print(
        "  got_incl[1023]        =",
        got_incl[block_size - 1],
        "(prefix_sum inclusive at last thread)",
    )
    # Probe each warp boundary in the active region (every 32 threads).
    var w = 0
    while w * 32 < n_active + 64 and w * 32 < block_size:
        var idx = w * 32
        print(
            "    warp",
            w,
            "tx",
            idx,
            ": incl=",
            got_incl[idx],
            " excl=",
            got_excl[idx],
            " (ref incl=",
            ref_incl[idx],
            ")",
        )
        w += 1

    # --- Assertions. ---
    # (1) block.sum broadcast must equal ref_total on EVERY thread.
    for i in range(block_size):
        assert_almost_equal(
            got_sum[i],
            ref_total,
            msg=String("block.sum mismatch at tx=", i),
            rtol=1e-5,
        )
    # (2) prefix_sum exclusive + inclusive must match host reference everywhere.
    for i in range(block_size):
        assert_almost_equal(
            got_excl[i],
            ref_excl[i],
            msg=String("prefix_sum EXCLUSIVE mismatch at tx=", i),
            rtol=1e-5,
        )
        assert_almost_equal(
            got_incl[i],
            ref_incl[i],
            msg=String("prefix_sum INCLUSIVE mismatch at tx=", i),
            rtol=1e-5,
        )
    # (3) CONSISTENCY invariant the topk kernel relies on.
    assert_almost_equal(
        got_incl[block_size - 1],
        got_sum[0],
        msg="CONSISTENCY: last inclusive CDF != block.sum total",
        rtol=1e-5,
    )
    print("  PASS\n")


def test_block_scan() raises:
    comptime BLOCK = 1024
    with DeviceContext() as ctx:
        # Case A: all 1024 threads carry 1.0 -> single big scan, all warps live.
        run_case[BLOCK]("all-ones-1024", 1024, 1.0, ctx)
        # Case B: only first 128 threads (4 warps) carry 1.0 -> EXACT topk
        # geometry. Decisive for the cross-warp carry past warp 4.
        run_case[BLOCK]("masked-128-ones", 128, 1.0, ctx)
        # Case C: masked float distribution (only 4 warps carry data), tests the
        # consistency invariant on non-integer mass.
        run_case[BLOCK]("masked-128-float", 128, 0.013, ctx)
        # Case D: a single warp's worth (32) carries data, others zero -> the
        # N=100 single-warp regime that PASSES in topk, as a control.
        run_case[BLOCK]("masked-32-ones", 32, 1.0, ctx)


def main() raises:
    var suite = TestSuite()
    suite.test[test_block_scan]()
    suite^.run()
