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
# Deterministic memory-safety regression test for modular#7069: the unmasked
# A-tile global->shared copy in multistage_mma_q reads (BM - M) * K elements past
# the [M, K] A buffer when M < BM. The over-read never corrupts output (the
# epilogue stores only rows < M), so it is invisible to a correctness test; the
# only symptom is CUDA_ERROR_ILLEGAL_ADDRESS when the over-read crosses an
# unmapped page. A pooled allocator maps a big arena, so the fault is normally
# allocation-layout dependent and does not reproduce in a plain test.
#
# This test removes that dependence with a CUDA VMM guard page: A's valid rows are
# placed at the very end of a mapped region whose next page is reserved but
# UNMAPPED, so ANY read past row M faults deterministically. On buggy code the
# kernel faults; with the mask_a fix the OOB rows are zero-filled (cp.async
# src-size 0) and never touch the guard page, so the kernel completes.
#
# Runs a Llama-3 down_proj decode shape (N=4096, K=14336) at M=1 with a BM=128
# config, so every A tile has M < BM and drives the masked copy.
from std.ffi import OwnedDLHandle
from std.memory import OpaquePointer

from layout import Coord, Idx, TileTensor, row_major
from linalg.utils_gpu import MatmulKernels
from max.gpu.host import DeviceContext, DeviceBuffer
from quantization.qmatmul_gpu import multistage_gemm_q

comptime N = 4096
comptime K = 14336
comptime GROUP_SIZE = 128
comptime PACK_FACTOR = 8
comptime GROUP_BYTES = GROUP_SIZE // 2 + 2  # 66
comptime A_TYPE = DType.bfloat16
comptime B_TYPE = DType.uint8


def _u32(base: UInt64, byte_off: Int, val: UInt32):
    OpaquePointer[MutUntrackedOrigin](
        unsafe_from_address=Int(base) + byte_off
    ).bitcast[UInt32]()[0] = val


def main() raises:
    with DeviceContext() as ctx:
        print("device:", ctx.name())
        comptime M = 1
        var lib = OwnedDLHandle(path="libcuda.so.1")

        var cuMemGetAllocationGranularity = lib._handle.get_function[
            def(UInt64, UInt64, Int32) thin abi("C") -> Int32
        ]("cuMemGetAllocationGranularity")
        var cuMemCreate = lib._handle.get_function[
            def(UInt64, UInt64, UInt64, UInt64) thin abi("C") -> Int32
        ]("cuMemCreate")
        var cuMemAddressReserve = lib._handle.get_function[
            def(UInt64, UInt64, UInt64, UInt64, UInt64) thin abi("C") -> Int32
        ]("cuMemAddressReserve")
        var cuMemMap = lib._handle.get_function[
            def(UInt64, UInt64, UInt64, UInt64, UInt64) thin abi("C") -> Int32
        ]("cuMemMap")
        var cuMemSetAccess = lib._handle.get_function[
            def(UInt64, UInt64, UInt64, UInt64) thin abi("C") -> Int32
        ]("cuMemSetAccess")
        var cuMemUnmap = lib._handle.get_function[
            def(UInt64, UInt64) thin abi("C") -> Int32
        ]("cuMemUnmap")
        var cuMemRelease = lib._handle.get_function[
            def(UInt64) thin abi("C") -> Int32
        ]("cuMemRelease")
        var cuMemAddressFree = lib._handle.get_function[
            def(UInt64, UInt64) thin abi("C") -> Int32
        ]("cuMemAddressFree")

        # CUmemAllocationProp (32B): type=PINNED(1)@0, loc.type=DEVICE(1)@8, id=0@12
        var prop = ctx.enqueue_create_host_buffer[DType.uint8](32)
        for i in range(32):
            prop.unsafe_ptr()[i] = 0
        var prop_addr = UInt64(Int(prop.unsafe_ptr()))
        _u32(prop_addr, 0, 1)
        _u32(prop_addr, 8, 1)
        _u32(prop_addr, 12, 0)

        # Driver out-params live in real host memory (see feasibility probe notes).
        var outbuf = ctx.enqueue_create_host_buffer[DType.uint64](3)
        var out64 = outbuf.unsafe_ptr()
        out64[0] = 0
        out64[1] = 0
        out64[2] = 0

        _ = cuMemGetAllocationGranularity(UInt64(Int(out64)), prop_addr, 0)
        var chunk = out64[0]
        var reserve = chunk * 2

        _ = cuMemAddressReserve(UInt64(Int(out64) + 8), reserve, 0, 0, 0)
        var base = out64[1]
        _ = cuMemCreate(UInt64(Int(out64) + 16), chunk, prop_addr, 0)
        var handle = out64[2]
        _ = cuMemMap(base, chunk, 0, handle, 0)

        # CUmemAccessDesc (16B): loc.type=DEVICE(1)@0, id=0@4, flags=RW(3)@8
        var desc = ctx.enqueue_create_host_buffer[DType.uint8](16)
        for i in range(16):
            desc.unsafe_ptr()[i] = 0
        var desc_addr = UInt64(Int(desc.unsafe_ptr()))
        _u32(desc_addr, 0, 1)
        _u32(desc_addr, 8, 3)
        _ = cuMemSetAccess(base, chunk, desc_addr, 1)

        # Place A's valid rows [M, K] at the TAIL of the mapped chunk, so the very
        # next byte after row M-1 is the unmapped guard page.
        comptime a_bytes = M * K * 2  # bf16
        var a_addr = base + chunk - a_bytes
        print(
            "guard page: mapped [", base, ",", base + chunk, "), A at", a_addr
        )

        # Wrap the raw VMM device address as a non-owning DeviceBuffer.
        var a_ptr = Pointer[Scalar[A_TYPE], MutUntrackedOrigin](
            unsafe_from_address=Int(a_addr)
        )
        var a_device = DeviceBuffer[A_TYPE](ctx, a_ptr, M * K, owning=False)

        # B (packed weights) and C: ordinary pooled buffers. Values are irrelevant
        # to the memory-safety check; sizes must be correct so B/scales stay in
        # bounds and only the A copy can fault.
        comptime b_cols = (K // GROUP_SIZE) * GROUP_BYTES
        var b_device = ctx.enqueue_create_buffer[B_TYPE](N * b_cols)
        var c_device = ctx.enqueue_create_buffer[A_TYPE](M * N)
        b_device.enqueue_fill(0)
        c_device.enqueue_fill(0)

        var a_tt = TileTensor(a_device, row_major(Coord(M, Idx[K])))
        var b_tt = TileTensor(b_device, row_major(Coord(Idx[N], Idx[b_cols])))
        var c_tt = TileTensor(c_device, row_major(Coord(M, Idx[N])))

        comptime kernels = MatmulKernels[A_TYPE, B_TYPE, A_TYPE, True]()
        comptime config = kernels.ampere_128x128_4  # BM=128 -> M=1 < BM

        print(
            "launching multistage_gemm_q M=",
            M,
            " N=",
            N,
            " K=",
            K,
            " BM=",
            config.block_tile_shape[0],
        )
        multistage_gemm_q[
            group_size=GROUP_SIZE, pack_factor=PACK_FACTOR, config=config
        ](
            c_tt.to_layout_tensor(),
            a_tt.to_layout_tensor(),
            b_tt.to_layout_tensor(),
            config,
            ctx,
        )
        ctx.synchronize()
        print("SURVIVED: kernel completed with no illegal access (fix present)")

        _ = a_device^
        _ = b_device^
        _ = c_device^
        _ = cuMemUnmap(base, chunk)
        _ = cuMemRelease(handle)
        _ = cuMemAddressFree(base, reserve)
