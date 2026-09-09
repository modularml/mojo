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
# Fuzz target: block-scaled MXFP8 SM100 dense matmul.
#
# This is the kernel behind MiniMax-M3-MXFP8 attention (q/k/v/o) and the
# shared-expert MLP: `float8_e4m3fn` operands with one `float8_e8m0fnu` (E8M0)
# scale per 32-element K block, fp32 accumulation, bf16 output. We fuzz M (the
# token dimension) with the boundary-aware generator; N/K/mma_n/cta_group are
# compile-time (`-D N=.. -D K=.. -D mma_n=.. -D cta_group=..`) so the tuned
# tensor-core path engages (the kernel reads N/K from the static tensor shape).
#
# `ref` oracle (--check 1): compares the kernel against `vendor_blas.matmul`
# (cuBLAS) given the SAME E8M0 scales -- the same reference the in-tree MXFP8
# unit test uses. memcheck/redzone also apply (a K-tail or scale-tensor
# over-read across fuzzed M surfaces under the pool-off sanitizer). B200/SM100
# only. Self-contained (the unit test's harness lives in a test file, not an
# importable package), modeled on test_matmul_sm100_block_scaled_mxfp8.mojo.

from std.math import align_up, ceildiv
from std.random import rand, random_ui64, seed
from std.sys import size_of
from std.sys.defines import get_defined_int

import linalg.matmul.vendor.blas as vendor_blas
from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind
from std.utils.index import Index
from std.utils.static_tuple import StaticTuple
from layout import Coord, Idx, TileTensor, row_major
from linalg.matmul.gpu.sm100.block_scaled_matmul import (
    blackwell_block_scaled_matmul_tma_umma_warp_specialized,
)
from linalg.matmul.gpu.sm100.config import BlockScaledMatmulConfig
from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    MXFP8_SF_VECTOR_SIZE,
    SF_ATOM_K,
    SF_ATOM_M,
    SF_MN_GROUP_SIZE,
    set_scale_factor,
)

from _fuzz import boundary_int, collect_args, flag, flag_int, numeric_check

comptime a_type = DType.float8_e4m3fn
comptime out_dtype = DType.bfloat16
comptime scales_dtype = MXFP8_SF_DTYPE
comptime SF_VECTOR_SIZE = MXFP8_SF_VECTOR_SIZE
comptime swizzle = TensorMapSwizzle.SWIZZLE_128B
comptime BK = swizzle.bytes() // size_of[a_type]()
comptime MMA_K = 32
comptime bm = 128

# COMPTIME problem dims (the tuned SM100 kernel reads N/K from the static shape).
comptime N = get_defined_int["N", 1024]()
comptime K = get_defined_int["K", 1024 + 16]()  # K-tail (not a 32 multiple)
comptime mma_n = get_defined_int["mma_n", 128]()
comptime cta_group = get_defined_int["cta_group", 1]()

comptime block_tile_shape = Index(bm, mma_n // cta_group, BK)
comptime umma_shape = Index(cta_group * bm, mma_n, MMA_K)

comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    var m: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "m=",
            self.m,
            " N=",
            N,
            " K=",
            K,
            " mma_n=",
            mma_n,
            " cta_group=",
            cta_group,
        )


def gen_specs(n: Int) -> List[CaseSpec]:
    var specs = List[CaseSpec]()
    for _ in range(n):
        specs.append(CaseSpec(boundary_int(1, 2048, SF_MN_GROUP_SIZE)))
    return specs^


def run_one_case(
    ctx: DeviceContext, spec: CaseSpec, check: Bool = False
) raises:
    var m = spec.m
    comptime transpose_b = True

    var a_shape = row_major(Coord(m, Idx[K]))
    var b_shape = row_major(Coord(Idx[N], Idx[K]))
    var c_shape = row_major(Coord(m, Idx[N]))

    var a_size = m * K
    var b_size = N * K
    var c_size = m * N

    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](a_size)
    var a_host = TileTensor(a_host_ptr, a_shape)
    var b_host_ptr = ctx.enqueue_create_host_buffer[a_type](b_size)
    var b_host = TileTensor(b_host_ptr, b_shape)
    var c_host_ptr = ctx.enqueue_create_host_buffer[out_dtype](c_size)
    var c_host = TileTensor(c_host_ptr, c_shape)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[out_dtype](c_size)

    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var a_tensor = TileTensor(a_device, a_shape)
    var b_device = ctx.enqueue_create_buffer[a_type](b_size)
    var b_tensor = TileTensor(b_device, b_shape)
    var c_device = ctx.enqueue_create_buffer[out_dtype](c_size)
    var c_tensor = TileTensor(c_device, c_shape)
    var c_device_ref = ctx.enqueue_create_buffer[out_dtype](c_size)
    var c_ref_tensor = TileTensor(c_device_ref, c_shape)

    var a_scales_shape = row_major(
        Coord(
            ceildiv(m, SF_MN_GROUP_SIZE),
            Idx[ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K)],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )
    var b_scales_shape = row_major(
        Coord(
            Idx[ceildiv(N, SF_MN_GROUP_SIZE)],
            Idx[ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K)],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )

    var a_scales_total = a_scales_shape.product()
    var b_scales_total = b_scales_shape.product()

    var a_scales_host_ptr = ctx.enqueue_create_host_buffer[scales_dtype](
        a_scales_total
    )
    var a_scales_host = TileTensor(a_scales_host_ptr, a_scales_shape)
    var b_scales_host_ptr = ctx.enqueue_create_host_buffer[scales_dtype](
        b_scales_total
    )
    var b_scales_host = TileTensor(b_scales_host_ptr, b_scales_shape)

    var a_scales_device = ctx.enqueue_create_buffer[scales_dtype](
        a_scales_total
    )
    var a_scales_tensor = TileTensor(a_scales_device, a_scales_shape)
    var b_scales_device = ctx.enqueue_create_buffer[scales_dtype](
        b_scales_total
    )
    var b_scales_tensor = TileTensor(b_scales_device, b_scales_shape)

    # Operands: uniform [0,1) fp8, matching the unit test (keeps cuBLAS ref in
    # the validated tolerance band).
    rand(a_host._storage, a_host.num_elements())
    rand(b_host._storage, b_host.num_elements())

    # E8M0 block scales: random powers of two in-range, 0.0 for padding.
    # NOTE: unused (padding) scales MUST be 0.0 or accuracy breaks (see the
    # unit test's comment).
    for idx0 in range(align_up(m, SF_MN_GROUP_SIZE)):
        for idx1 in range(
            0, align_up(K, SF_VECTOR_SIZE * SF_ATOM_K), SF_VECTOR_SIZE
        ):
            if idx0 < m and idx1 < K:
                var sv = (
                    (1 << random_ui64(0, 3))
                    .cast[.float32]()
                    .cast[scales_dtype]()
                )
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    a_scales_host, idx0, idx1, sv
                )
            else:
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    a_scales_host, idx0, idx1, Scalar[scales_dtype](0.0)
                )

    for idx0 in range(align_up(N, SF_MN_GROUP_SIZE)):
        for idx1 in range(
            0, align_up(K, SF_VECTOR_SIZE * SF_ATOM_K), SF_VECTOR_SIZE
        ):
            if idx0 < N and idx1 < K:
                var sv = (
                    (1 << random_ui64(0, 3))
                    .cast[.float32]()
                    .cast[scales_dtype]()
                )
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    b_scales_host, idx0, idx1, sv
                )
            else:
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    b_scales_host, idx0, idx1, Scalar[scales_dtype](0.0)
                )

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(a_scales_device, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device, b_scales_host_ptr)

    comptime matmul_config = BlockScaledMatmulConfig[
        a_type, a_type, out_dtype, scales_dtype, scales_dtype, transpose_b
    ](
        scaling_kind=UMMAKind.KIND_MXF8F6F4,
        cluster_shape=Index(cta_group, 1, 1),
        mma_shape=umma_shape,
        block_swizzle_size=8,
        cta_group=cta_group,
        num_accum_pipeline_stages=1 if mma_n in (192, 256) else 2,
    )

    blackwell_block_scaled_matmul_tma_umma_warp_specialized[
        transpose_b=transpose_b, K=K, config=matmul_config
    ](c_tensor, a_tensor, b_tensor, a_scales_tensor, b_scales_tensor, ctx)
    ctx.synchronize()

    if check:
        # Higher-precision reference: cuBLAS with the same E8M0 scales.
        var a_lt = a_tensor.to_layout_tensor()
        var b_lt = b_tensor.to_layout_tensor()
        var a_scales_lt = a_scales_tensor.to_layout_tensor()
        var b_scales_lt = b_scales_tensor.to_layout_tensor()
        var c_ref_lt = c_ref_tensor.to_layout_tensor()
        vendor_blas.matmul(
            ctx,
            c_ref_lt.as_unsafe_any_origin(),
            a_lt,
            b_lt,
            a_scales=a_scales_lt.as_imm().as_unsafe_any_origin(),
            b_scales=b_scales_lt.as_imm().as_unsafe_any_origin(),
            transpose_b=transpose_b,
            c_row_major=True,
        )
        ctx.synchronize()
        ctx.enqueue_copy(c_host_ptr, c_device)
        ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
        ctx.synchronize()
        if not numeric_check(
            c_host_ptr.as_span(),
            c_host_ref_ptr.as_span(),
            atol=1e-2,
            rtol=1e-2,
        ):
            raise Error("block-scaled MXFP8 matmul vs cuBLAS mismatch")

    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^
    _ = a_scales_device^
    _ = b_scales_device^


def main() raises:
    var args = collect_args()
    var mode = flag(args, "--mode", "fuzz")
    var the_seed = flag_int(args, "--seed", fuzz_seed)
    var the_budget = flag_int(args, "--budget", budget)
    var check = flag_int(args, "--check", 0) == 1
    seed(the_seed)

    if mode == "list-specs":
        var specs = gen_specs(the_budget)
        for i in range(len(specs)):
            print("FUZZ_SPEC idx=", i, "m=", specs[i].m)
        return

    if mode == "single":
        var m = flag_int(args, "--m", 1000)
        print("FUZZ_SINGLE m=", m, "N=", N, "K=", K, "mma_n=", mma_n)
        with DeviceContext() as ctx:
            run_one_case(ctx, CaseSpec(m), check)
        print("FUZZ_RESULT verdict=PASS")
        return

    print(
        "=== fuzz_block_scaled_mxfp8 seed=",
        the_seed,
        "budget=",
        the_budget,
        "N=",
        N,
        "K=",
        K,
        "mma_n=",
        mma_n,
        "cta_group=",
        cta_group,
        "===",
    )
    var specs = gen_specs(the_budget)
    with DeviceContext() as ctx:
        for i in range(len(specs)):
            print("case", i, ":", specs[i])
            run_one_case(ctx, specs[i], check)
    print("=== done:", len(specs), "cases ===")
