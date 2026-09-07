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
# Run-to-run determinism test for the P2P `allreduce` API (DeepEP-style rerun +
# hash-of-tensor). Arch-portable: the `allreduce` entry point is the same on
# NVIDIA SM90 (H100) / SM100 (B200) and AMD CDNA4 (MI355); the size-keyed
# algorithm selection lives in the shared tuning table (`comm/allreduce.mojo`).
# KERN-3129 (an AMD split-K sync race) is the class of bug this guards against:
# a fixed-order reduction should produce bit-identical output every run.
#
# Method (mirrors DeepEP's 20x rerun-and-hash-compare, and the local idiom in
# `test_allreduce.mojo`):
#   1. Fixed, deterministic per-rank inputs (a closed-form function of
#      (rank, index) -- no RNG state to thread across ranks).
#   2. Run the same `allreduce` N times back-to-back (the realistic pattern:
#      consecutive collectives reuse the same signal buffers).
#   3. Per run, hash every output buffer (bit-reinterpret each element, fold
#      with an order-sensitive FNV-1a) and assert (a) all ranks agree and
#      (b) the hash is identical to run 0's. A divergence is a real race
#      finding (KERN-3129 class), not a tolerance to loosen.
#
# WHY NOT literal "XOR across ranks" (the DeepEP hash-of-tensor): DeepEP hashes
# an all-to-all where each rank holds DIFFERENT data, so XOR-across-tensors is
# non-degenerate. `allreduce` broadcasts the SAME reduced result to every rank,
# so all output buffers are bit-identical -- an XOR across an even number of
# ranks (ngpus in {2,4,8}) cancels to exactly 0 every run, and the run-to-run
# check would pass vacuously. We instead hash each rank independently
# (order-sensitive, non-cancelling), assert the ranks agree, and compare the
# per-run hash across reruns.
#
# HARDWARE: authored arch-portable; validated on B200. A single-GPU box can only
# run the host-side hash positive control (the `allreduce` API requires
# ngpus >= 2 at comptime); the multi-GPU determinism sweep needs a >= 2 GPU
# runner (`--config=remote-b200` / the nightly multi-GPU lane).

from std.sys import size_of

from comm import MAX_GPUS, Signal, group_end, group_start
from comm.allreduce import allreduce, allreduce_tuning_table
from comm.device_query import dispatch_select_comm_config
from comm.sync import enable_p2p, init_signal_buffer
from layout import TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext
from std.testing import assert_equal, assert_true

# DeepEP reruns 20x; reruns inside one process are cheap.
comptime NUM_RERUNS = 20

# 64-bit FNV-1a constants (a standard, order-sensitive rolling hash).
comptime FNV_OFFSET_BASIS = UInt64(0xCBF29CE484222325)
comptime FNV_PRIME = UInt64(0x100000001B3)

# Element counts chosen to straddle the size-keyed algorithm switch. On the
# 8xB200 CI runner (sm_100a) these map to: small -> LAMPORT, mid -> ONE_STAGE,
# large -> TWO_STAGE (bytes = length * size_of[dtype]); the sweep prints the
# selected algorithm per case so the regime coverage is visible in the log.
comptime LEN_SMALL = 16 * 1024
comptime LEN_MID = 256 * 1024
comptime LEN_LARGE = 2 * 1024 * 1024


def _elem_bits_u64[dtype: DType](x: Scalar[dtype]) -> UInt64:
    """Reinterprets one float element's bits as a `UInt64` (zero-extended)."""
    return x.to_bits().cast[.uint64]()


def hash_output[
    dtype: DType
](p: MutPointer[Scalar[dtype], MutUntrackedOrigin], length: Int) -> UInt64:
    """Order-sensitive FNV-1a over a host buffer's element bit patterns.

    Sensitive to any single-bit change and to element order, and (unlike an
    XOR-across-identical-tensors fold) never cancels to a constant.
    """
    var acc = FNV_OFFSET_BASIS
    for j in range(length):
        acc = (acc ^ _elem_bits_u64(p[j])) * FNV_PRIME
    return acc


def hash_positive_control() raises -> None:
    """Host-only positive control: the hash must detect a single-bit flip.

    Proves the hash mechanism has teeth without needing multiple GPUs, so it
    runs (and gives evidence) even on a single-GPU box.
    """
    comptime n = 4096
    var buf = alloc[Float32](n)
    for j in range(n):
        buf[j] = Float32(j) * 0.5 - 1024.0

    var h0 = hash_output(buf, n)
    # Flip the lowest bit of one element through an integer view (a true
    # single-bit change of the stored float bits), re-hash, then restore.
    var ibuf = buf.bitcast[UInt32]()
    var saved_bits = ibuf[n // 2]
    ibuf[n // 2] = saved_bits ^ 1
    var h1 = hash_output(buf, n)
    ibuf[n // 2] = saved_bits
    var h2 = hash_output(buf, n)

    assert_true(h0 != h1, "positive control FAILED: hash blind to a bit flip")
    assert_equal(h0, h2)  # restoring the bit restores the hash
    buf.free()
    print("hash positive control PASS (single-bit flip detected)")


def allreduce_determinism_test[
    dtype: DType, ngpus: Int
](list_of_ctx: List[DeviceContext], length: Int) raises -> None:
    """Runs `allreduce` `NUM_RERUNS` times on fixed inputs; asserts bit-stable.

    Parameters:
        dtype: Element type (bf16 / fp32).
        ngpus: Number of participating GPUs.

    Args:
        list_of_ctx: One `DeviceContext` per GPU.
        length: Element count of the input/output tensors.
    """
    comptime assert ngpus in (2, 4, 8), "ngpus must be 2, 4, or 8"

    var in_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var out_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var host_in = List[MutPointer[Scalar[dtype], MutUntrackedOrigin]](
        capacity=ngpus
    )
    var host_out = List[MutPointer[Scalar[dtype], MutUntrackedOrigin]](
        capacity=ngpus
    )

    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var rank_sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )
    var temp_buffer_num_bytes = ngpus * size_of[dtype]() * length

    for i in range(ngpus):
        in_dev.append(list_of_ctx[i].enqueue_create_buffer[dtype](length))
        out_dev.append(list_of_ctx[i].enqueue_create_buffer[dtype](length))

        var h_in = alloc[Scalar[dtype]](length)
        host_in.append(h_in)
        host_out.append(alloc[Scalar[dtype]](length))

        # Fixed, deterministic per-rank inputs: a closed-form function of
        # (rank, index) mapped to a small bounded float so bf16 sums across up
        # to 8 ranks stay representable. Same values every run.
        for j in range(length):
            var t = (i * 1315423911 + j * 2654435761) % 2039
            h_in[j] = Scalar[dtype](Float64(t - 1019) * 0.001953125)

        signal_buffers.append(
            list_of_ctx[i].create_buffer_sync[.uint8](
                size_of[Signal]() + temp_buffer_num_bytes
            )
        )
        rank_sigs[i] = (
            signal_buffers[i]
            .unsafe_ptr()
            .bitcast[Signal]()
            .as_unsafe_any_origin()
        )
        list_of_ctx[i].enqueue_copy(in_dev[i], host_in[i])

    comptime InTensorType = TileTensor[
        dtype, type_of(row_major(length)), ImmutAnyOrigin
    ]
    var in_tensors = Array[_, ngpus](
        fill_with=lambda (i: Int) -> InTensorType: TileTensor(
            rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                in_dev[i].unsafe_ptr()
            ),
            row_major(length),
        )
    )

    comptime OutTensorType = TileTensor[
        dtype, type_of(row_major(length)), MutAnyOrigin
    ]
    var out_tensors = Array[_, ngpus](
        fill_with=lambda (i: Int) {
            mut out_dev, imm
        } -> OutTensorType: TileTensor(
            out_dev[i], row_major(length)
        ).as_unsafe_any_origin()
    )

    # One-time signal-buffer init (barrier counters + Lamport sentinel), then
    # sync so every rank is initialized before the first push.
    for i in range(ngpus):
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Report which algorithm regime this (ngpus, dtype, length) selects, so the
    # log documents that both the one-stage and two-stage regimes are covered.
    comptime sm_version = DeviceContext.default_device_info.version
    var num_bytes = length * size_of[dtype]()
    var cfg = dispatch_select_comm_config[
        ngpus, sm_version, allreduce_tuning_table
    ](num_bytes)
    print(
        "  ngpus=",
        ngpus,
        " dtype=",
        dtype,
        " length=",
        length,
        " bytes=",
        num_bytes,
        " algo=",
        cfg.algorithm,
        sep="",
    )

    var ref_hash = UInt64(0)
    for run in range(NUM_RERUNS):
        group_start()
        comptime for i in range(ngpus):
            allreduce[ngpus=ngpus, use_multimem=False](
                in_tensors, out_tensors[i], rank_sigs, list_of_ctx[i]
            )
        group_end()
        for i in range(ngpus):
            list_of_ctx[i].synchronize()
        for i in range(ngpus):
            list_of_ctx[i].enqueue_copy(host_out[i], out_dev[i])
        for i in range(ngpus):
            list_of_ctx[i].synchronize()

        # Hash rank 0, and require every other rank to agree bit-for-bit (a
        # correct allreduce broadcasts the same reduction to all ranks).
        var run_hash = hash_output(host_out[0], length)
        for i in range(1, ngpus):
            var rank_hash = hash_output(host_out[i], length)
            if rank_hash != run_hash:
                print(
                    "CROSS-RANK DIVERGENCE at run",
                    run,
                    "rank",
                    i,
                    "hash",
                    rank_hash,
                    "!= rank0 hash",
                    run_hash,
                )
            assert_equal(rank_hash, run_hash)

        if run == 0:
            ref_hash = run_hash
        else:
            if run_hash != ref_hash:
                print(
                    "RUN-TO-RUN DIVERGENCE at run",
                    run,
                    "hash",
                    run_hash,
                    "!= run0 hash",
                    ref_hash,
                )
            assert_equal(run_hash, ref_hash)

    for i in range(ngpus):
        host_in[i].free()
        host_out[i].free()


def run_determinism_sweep() raises -> None:
    """Sweeps (dtype, ngpus, length) covering both algorithm regimes."""
    comptime dtypes = (DType.bfloat16, DType.float32)
    comptime gpu_counts = (2, 4, 8)
    comptime lengths = (LEN_SMALL, LEN_MID, LEN_LARGE)

    comptime for d in range(len(dtypes)):
        comptime for g in range(len(gpu_counts)):
            comptime dtype = rebind[DType](dtypes[d])
            comptime ngpus = rebind[Int](gpu_counts[g])
            if DeviceContext.number_of_devices() < ngpus:
                continue
            var ctx = List[DeviceContext]()
            for i in range(ngpus):
                ctx.append(DeviceContext(device_id=i))
            comptime for li in range(len(lengths)):
                comptime length = rebind[Int](lengths[li])
                allreduce_determinism_test[dtype=dtype, ngpus=ngpus](
                    ctx, length
                )


def main() raises:
    # Positive control runs on any GPU count (host-only): proves the hash
    # detects a flip, so a PASS below is not a placebo.
    hash_positive_control()

    var num_devices = DeviceContext.number_of_devices()
    if num_devices < 2:
        print(
            "SKIP: allreduce determinism sweep needs >= 2 GPUs; found",
            num_devices,
            "- ran the host-only hash positive control only.",
        )
        return

    assert_true(enable_p2p(), "failed to enable P2P access between GPUs")
    run_determinism_sweep()
    print("allreduce determinism sweep PASS")
