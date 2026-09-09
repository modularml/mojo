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
"""AMD split-K matmul run-to-run determinism regression tests (KERN-3129).

Each case re-runs one fixed input set many times across every visible GPU and
asserts the outputs are bit-identical. A correct kernel is deterministic for
fixed inputs, so any difference is a synchronization bug. The shapes are large
and all GPUs launch before the first sync, so the node runs under the
concurrent-kernel, high-occupancy load where the split-K epilogue race surfaces.

Covers amd_4wave_split_k_matmul (the kernel the race lives in; fails without the
num_splits > 1 epilogue drain) and _launch_block_scaled_split_k (a regression guard for
that path).
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory import bitcast
from std.random import random_float64, random_ui64
from std.testing import assert_equal

from layout import TileTensor, row_major
from linalg.fp4_utils import MXFP4_SF_VECTOR_SIZE
from linalg.matmul.gpu.amd.amd_4wave_split_k_matmul import (
    amd_4wave_split_k_matmul,
    SplitKWorkspace,
)
from linalg.matmul.gpu.amd.block_scaled_matmul_amd import (
    _launch_block_scaled_split_k,
)


def _report_divergence[
    out_dtype: DType
](
    outputs: List[DeviceBuffer[out_dtype]],
    num_gpus: Int,
    reps: Int,
    elems: Int,
) raises:
    """Assert every GPU's reps outputs are bit-identical to its first run.

    outputs is laid out flat as gpu * reps + rep; each device runs its own
    inputs, so comparisons stay within a single GPU.
    """
    var total_diverged = 0
    for g in range(num_gpus):
        var ref_vals = List[Scalar[out_dtype]](capacity=elems)
        with outputs[g * reps].map_to_host() as host_ref:
            for i in range(elems):
                ref_vals.append(host_ref[i])

        for r in range(1, reps):
            with outputs[g * reps + r].map_to_host() as host_r:
                for i in range(elems):
                    if host_r[i].to_bits() != ref_vals[i].to_bits():
                        total_diverged += 1
                        break

    if total_diverged != 0:
        print(
            "  ",
            total_diverged,
            " of ",
            num_gpus * (reps - 1),
            " runs diverged",
            sep="",
        )
    assert_equal(total_diverged, 0)


def test_4wave_split_k_determinism[
    M: Int,
    N: Int,
    K: Int,
    num_splits: Int,
    reps: Int,
    enable_swizzle: Bool,
](ctxs: List[DeviceContext]) raises:
    """Re-run one input set reps times on each GPU; assert it never changes."""
    comptime in_dtype = DType.float8_e4m3fn
    comptime out_dtype = DType.float32

    var num_gpus = len(ctxs)

    # Keep all buffers alive until the sync below so every device runs at once.
    var a_bufs = List[DeviceBuffer[in_dtype]](capacity=num_gpus)
    var b_bufs = List[DeviceBuffer[in_dtype]](capacity=num_gpus)
    var workspaces = List[SplitKWorkspace[num_splits]](capacity=num_gpus)
    var outputs = List[DeviceBuffer[out_dtype]](capacity=num_gpus * reps)

    for g in range(num_gpus):
        var ctx = ctxs[g]

        var device_a = ctx.enqueue_create_buffer[in_dtype](M * K)
        var device_b = ctx.enqueue_create_buffer[in_dtype](N * K)

        # Fill A and B once; every repetition reads the same inputs.
        with device_a.map_to_host() as host_a, device_b.map_to_host() as host_b:
            for i in range(M * K):
                host_a[i] = random_float64(-0.5, 0.5).cast[in_dtype]()
            for i in range(K * N):
                host_b[i] = random_float64(-0.5, 0.5).cast[in_dtype]()

        var a_tt = TileTensor(device_a, row_major[M, K]())
        var b_tt = TileTensor(device_b, row_major[N, K]())

        # One workspace per device is safe: same-stream launches run in order,
        # so each reduce reads it before the next matmul overwrites it.
        var workspace = SplitKWorkspace[num_splits](ctx, M * N)
        for _ in range(reps):
            var device_c = ctx.enqueue_create_buffer[out_dtype](M * N)
            ctx.enqueue_memset(device_c, 0)
            var c_tt = TileTensor(device_c, row_major[M, N]())
            amd_4wave_split_k_matmul[
                num_splits=num_splits, enable_swizzle=enable_swizzle
            ](a_tt, b_tt, c_tt, ctx, workspace=workspace)
            outputs.append(device_c)

        workspaces.append(workspace)
        a_bufs.append(device_a)
        b_bufs.append(device_b)

    for g in range(num_gpus):
        ctxs[g].synchronize()

    _report_divergence(outputs, num_gpus, reps, M * N)


def test_mxfp4_split_k_determinism[
    M: Int, N: Int, K: Int, num_splits: Int, reps: Int
](ctxs: List[DeviceContext]) raises:
    """Re-run one input set reps times on each GPU; assert it never changes.

    The launcher allocates its own split-K workspace per call.
    """
    comptime in_dtype = DType.uint8
    comptime sf_dtype = DType.float8_e8m0fnu
    comptime out_dtype = DType.float32

    comptime K_PACKED = K // 2
    comptime K_SCALES = K // MXFP4_SF_VECTOR_SIZE

    var num_gpus = len(ctxs)

    var a_bufs = List[DeviceBuffer[in_dtype]](capacity=num_gpus)
    var b_bufs = List[DeviceBuffer[in_dtype]](capacity=num_gpus)
    var as_bufs = List[DeviceBuffer[sf_dtype]](capacity=num_gpus)
    var bs_bufs = List[DeviceBuffer[sf_dtype]](capacity=num_gpus)
    var outputs = List[DeviceBuffer[out_dtype]](capacity=num_gpus * reps)

    for g in range(num_gpus):
        var ctx = ctxs[g]

        var device_a = ctx.enqueue_create_buffer[in_dtype](M * K_PACKED)
        var device_b = ctx.enqueue_create_buffer[in_dtype](N * K_PACKED)
        var device_as = ctx.enqueue_create_buffer[sf_dtype](M * K_SCALES)
        var device_bs = ctx.enqueue_create_buffer[sf_dtype](N * K_SCALES)

        # Fill inputs once; every repetition reads the same data.
        with device_a.map_to_host() as a:
            for i in range(M * K_PACKED):
                a[i] = UInt8(random_ui64(0, 255))
        with device_b.map_to_host() as b:
            for i in range(N * K_PACKED):
                b[i] = UInt8(random_ui64(0, 255))
        with device_as.map_to_host() as sa:
            for i in range(M * K_SCALES):
                sa[i] = bitcast[sf_dtype](UInt8(random_ui64(125, 129)))
        with device_bs.map_to_host() as sb:
            for i in range(N * K_SCALES):
                sb[i] = bitcast[sf_dtype](UInt8(random_ui64(125, 129)))

        var a_tt = TileTensor(device_a, row_major[M, K_PACKED]()).as_immut()
        var b_tt = TileTensor(device_b, row_major[N, K_PACKED]()).as_immut()
        var as_tt = TileTensor(device_as, row_major[M, K_SCALES]()).as_immut()
        var bs_tt = TileTensor(device_bs, row_major[N, K_SCALES]()).as_immut()

        for _ in range(reps):
            var device_c = ctx.enqueue_create_buffer[out_dtype](M * N)
            ctx.enqueue_memset(device_c, 0)
            var c_tt = TileTensor(device_c, row_major[M, N]())
            _launch_block_scaled_split_k[
                BM=64, BN=128, BK_ELEMS=256, WM=64, WN=32, num_splits=num_splits
            ](c_tt, a_tt, b_tt, as_tt, bs_tt, M, ctx)
            outputs.append(device_c)

        as_bufs.append(device_as)
        bs_bufs.append(device_bs)
        a_bufs.append(device_a)
        b_bufs.append(device_b)

    for g in range(num_gpus):
        ctxs[g].synchronize()

    _report_divergence(outputs, num_gpus, reps, M * N)


def main() raises:
    # Run on every visible GPU at once to keep the node under concurrent-kernel
    # load, the regime where the split-K epilogue race surfaces.
    var num_gpus = min(4, DeviceContext.number_of_devices())
    print("===> AMD split-K determinism (KERN-3129) on", num_gpus, "GPU(s)")

    var ctxs = List[DeviceContext](capacity=num_gpus)
    for i in range(num_gpus):
        ctxs.append(DeviceContext(device_id=i))

    # 4-wave FP8 split-K. Large shape so the split-K launch oversubscribes each
    # device; reps capped so the per-run output buffers fit in memory.
    comptime w_m = 64
    comptime w_n = 8192
    comptime w_k = 8192
    comptime w_splits = 4
    comptime w_reps = 32
    print(
        "  4-wave FP8: M=",
        w_m,
        " N=",
        w_n,
        " K=",
        w_k,
        " num_splits=",
        w_splits,
        " reps=",
        w_reps,
        sep="",
    )
    print("  4-wave / no swizzle...")
    test_4wave_split_k_determinism[
        w_m, w_n, w_k, w_splits, w_reps, enable_swizzle=False
    ](ctxs)
    print("  PASSED: 4-wave, no swizzle")

    print("  4-wave / with swizzle...")
    test_4wave_split_k_determinism[
        w_m, w_n, w_k, w_splits, w_reps, enable_swizzle=True
    ](ctxs)
    print("  PASSED: 4-wave, with swizzle")

    # MXFP4 block-scaled split-K.
    comptime x_m = 64
    comptime x_n = 8192
    comptime x_k = 7168
    comptime x_splits = 14
    comptime x_reps = 32
    print(
        "  MXFP4: M=",
        x_m,
        " N=",
        x_n,
        " K=",
        x_k,
        " num_splits=",
        x_splits,
        " reps=",
        x_reps,
        sep="",
    )
    test_mxfp4_split_k_determinism[x_m, x_n, x_k, x_splits, x_reps](ctxs)
    print("  PASSED: MXFP4")

    print("==== ALL AMD split-K determinism tests passed ====")
