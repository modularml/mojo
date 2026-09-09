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
"""NVIDIA Tensor Cores implementation for matrix multiply-accumulate operations.

This module provides MMA implementations for NVIDIA GPUs with Tensor Cores,
covering architectures from SM70 (Volta) through SM90 (Hopper).

Supported operations:
- FP16 accumulation (SM70+)
- FP32 accumulation with FP16/BF16 inputs (SM80+)
- TF32 operations (SM80+)
- FP8 operations (SM89+)

Reference: https://docs.nvidia.com/cuda/parallel-thread-execution/
"""

from std.sys import _RegisterPackType, llvm_intrinsic
from std.sys._assembly import inlined_assembly
from std.memory import bitcast

# Import helper functions from parent module
from ..mma import _has_type, _has_shape, _unsupported_mma_op


@always_inline
def _mma_nvidia(mut d: SIMD, a: SIMD, b: SIMD, c: SIMD):
    # ===------------------------------------------------------------------===#
    # F16 = F16 * F16 + F16
    # ===------------------------------------------------------------------===#
    comptime if _has_type[DType.float16](
        a.dtype, b.dtype, c.dtype, d.dtype
    ) and _has_shape[(4, 2, 4, 4)](a.length, b.length, c.length, d.length):
        var sa = a.split()
        var sc = c.split()

        var r = llvm_intrinsic[
            "llvm.nvvm.mma.m16n8k8.row.col.f16.f16",
            _RegisterPackType[SIMD[.float16, 2], SIMD[.float16, 2]],
        ](sa[0], sa[1], b, sc[0], sc[1])

        d = rebind[type_of(d)](r[0].join(r[1]))
    elif _has_type[DType.float16](
        a.dtype, b.dtype, c.dtype, d.dtype
    ) and _has_shape[(1, 1, 2, 2)](a.length, b.length, c.length, d.length):
        var r = llvm_intrinsic[
            "llvm.nvvm.mma.m8n8k4.row.col.f16.f16",
            _RegisterPackType[Float16, Float16],
        ](a, b, c)
        d = rebind[type_of(d)](r[0].join(r[1]))

    # ===------------------------------------------------------------------===#
    # F32 = F16 * F16 + F32
    # ===------------------------------------------------------------------===#
    elif _has_type[
        (DType.float16, DType.float16, DType.float32, DType.float32)
    ](a.dtype, b.dtype, c.dtype, d.dtype) and _has_shape[(4, 2, 4, 4)](
        a.length, b.length, c.length, d.length
    ):
        var sa = a.split()
        var c0 = bitcast[.float32, 4](c)

        var r = llvm_intrinsic[
            "llvm.nvvm.mma.m16n8k8.row.col.f32.f32",
            _RegisterPackType[Float32, Float32, Float32, Float32],
        ](
            sa[0],
            sa[1],
            b,
            c0[0],
            c0[1],
            c0[2],
            c0[3],
        )

        d = rebind[type_of(d)](SIMD[.float32, 4](r[0], r[1], r[2], r[3]))
    elif _has_type[
        (DType.float16, DType.float16, DType.float32, DType.float32)
    ](a.dtype, b.dtype, c.dtype, d.dtype) and _has_shape[(1, 1, 2, 2)](
        a.length, b.length, c.length, d.length
    ):
        var r = llvm_intrinsic[
            "llvm.nvvm.mma.m8n8k4.row.col.f32.f32",
            _RegisterPackType[Float32, Float32],
        ](a, b, c)
        d = rebind[type_of(d)](r[0].join(r[1]))

    elif _has_type[
        (DType.float16, DType.float16, DType.float32, DType.float32)
    ](a.dtype, b.dtype, c.dtype, d.dtype) and _has_shape[(8, 4, 4, 4)](
        a.length, b.length, c.length, d.length
    ):
        var sa = a.split()
        var sa1 = sa[0].split()
        var sa2 = sa[1].split()
        var sb = b.split()
        var c0 = bitcast[.float32, 4](c)

        var r = llvm_intrinsic[
            "llvm.nvvm.mma.m16n8k16.row.col.f32.f32",
            _RegisterPackType[Float32, Float32, Float32, Float32],
        ](
            sa1[0],
            sa1[1],
            sa2[0],
            sa2[1],
            sb[0],
            sb[1],
            c0[0],
            c0[1],
            c0[2],
            c0[3],
        )
        d = rebind[type_of(d)](SIMD[.float32, 4](r[0], r[1], r[2], r[3]))

    # ===------------------------------------------------------------------===#
    # F32 = BF16 * BF16 + F32
    # ===------------------------------------------------------------------===#
    elif _has_type[
        (DType.bfloat16, DType.bfloat16, DType.float32, DType.float32)
    ](a.dtype, b.dtype, c.dtype, d.dtype) and _has_shape[(4, 2, 4, 4)](
        a.length, b.length, c.length, d.length
    ):
        var sa = a.split()
        var c0 = bitcast[.float32, 4](c)

        var r = llvm_intrinsic[
            "llvm.nvvm.mma.m16n8k8.row.col.bf16",
            _RegisterPackType[Float32, Float32, Float32, Float32],
        ](
            bitcast[.int32, 1](sa[0]),
            bitcast[.int32, 1](sa[1]),
            bitcast[.int32, 1](b),
            c0[0],
            c0[1],
            c0[2],
            c0[3],
        )
        d = rebind[type_of(d)](SIMD[.float32, 4](r[0], r[1], r[2], r[3]))

    elif _has_type[
        (DType.bfloat16, DType.bfloat16, DType.float32, DType.float32)
    ](a.dtype, b.dtype, c.dtype, d.dtype) and _has_shape[(8, 4, 4, 4)](
        a.length, b.length, c.length, d.length
    ):
        var sa = a.split()
        var sa1 = sa[0].split()
        var sa2 = sa[1].split()
        var sb = b.split()
        var c0 = bitcast[.float32, 4](c)

        var r = llvm_intrinsic[
            "llvm.nvvm.mma.m16n8k16.row.col.bf16",
            _RegisterPackType[Float32, Float32, Float32, Float32],
        ](
            bitcast[.int32, 1](sa1[0]),
            bitcast[.int32, 1](sa1[1]),
            bitcast[.int32, 1](sa2[0]),
            bitcast[.int32, 1](sa2[1]),
            bitcast[.int32, 1](sb[0]),
            bitcast[.int32, 1](sb[1]),
            c0[0],
            c0[1],
            c0[2],
            c0[3],
        )
        d = rebind[type_of(d)](SIMD[.float32, 4](r[0], r[1], r[2], r[3]))

    # ===------------------------------------------------------------------===#
    # F32 = tf32 * tf32 + F32
    # ===------------------------------------------------------------------===#
    elif _has_type[DType.float32](
        a.dtype, b.dtype, c.dtype, d.dtype
    ) and _has_shape[(2, 1, 4, 4)](a.length, b.length, c.length, d.length):
        var a0 = bitcast[.uint32, 2](a)
        var b0 = bitcast[.uint32, 1](b)
        var c0 = bitcast[.float32, 4](c)

        var r = llvm_intrinsic[
            "llvm.nvvm.mma.m16n8k4.row.col.tf32",
            _RegisterPackType[Float32, Float32, Float32, Float32],
        ](
            a0[0],
            a0[1],
            b0,
            c0[0],
            c0[1],
            c0[2],
            c0[3],
        )
        d = rebind[type_of(d)](SIMD[.float32, 4](r[0], r[1], r[2], r[3]))

    elif _has_type[DType.float32](
        a.dtype, b.dtype, c.dtype, d.dtype
    ) and _has_shape[(4, 2, 4, 4)](a.length, b.length, c.length, d.length):
        var a0 = bitcast[.uint32, 4](a)
        var b0 = bitcast[.uint32, 2](b)
        var c0 = bitcast[.float32, 4](c)

        var r = llvm_intrinsic[
            "llvm.nvvm.mma.m16n8k8.row.col.tf32",
            _RegisterPackType[Float32, Float32, Float32, Float32],
        ](
            a0[0],
            a0[1],
            a0[2],
            a0[3],
            b0[0],
            b0[1],
            c0[0],
            c0[1],
            c0[2],
            c0[3],
        )
        d = rebind[type_of(d)](SIMD[.float32, 4](r[0], r[1], r[2], r[3]))

    # ===------------------------------------------------------------------===#
    # F32 = FP8 * FP8 + F32
    # ===------------------------------------------------------------------===#
    elif _has_type[
        (DType.float8_e4m3fn, DType.float8_e4m3fn, DType.float32, DType.float32)
    ](a.dtype, b.dtype, c.dtype, d.dtype) and _has_shape[(16, 8, 4, 4)](
        a.length, b.length, c.length, d.length
    ):
        var a0 = bitcast[.uint32, 4](a)
        var b0 = bitcast[.uint32, 2](b)

        var r = inlined_assembly[
            (
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {$0, $1,"
                " $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};"
            ),
            _RegisterPackType[Float32, Float32, Float32, Float32],
            constraints="=f,=f,=f,=f,r,r,r,r,r,r,r,r,r,r",
        ](
            a0[0],
            a0[1],
            a0[2],
            a0[3],
            b0[0],
            b0[1],
            c[0],
            c[1],
            c[2],
            c[3],
        )
        d = rebind[type_of(d)](SIMD[.float32, 4](r[0], r[1], r[2], r[3]))
    elif _has_type[
        (DType.float8_e5m2, DType.float8_e5m2, DType.float32, DType.float32)
    ](a.dtype, b.dtype, c.dtype, d.dtype) and _has_shape[(16, 8, 4, 4)](
        a.length, b.length, c.length, d.length
    ):
        var a0 = bitcast[.uint32, 4](a)
        var b0 = bitcast[.uint32, 2](b)

        var r = inlined_assembly[
            (
                "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 {$0, $1,"
                " $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};"
            ),
            _RegisterPackType[Float32, Float32, Float32, Float32],
            constraints="=f,=f,=f,=f,r,r,r,r,r,r,r,r,r,r",
        ](
            a0[0],
            a0[1],
            a0[2],
            a0[3],
            b0[0],
            b0[1],
            c[0],
            c[1],
            c[2],
            c[3],
        )
        d = rebind[type_of(d)](SIMD[.float32, 4](r[0], r[1], r[2], r[3]))

    # ===------------------------------------------------------------------===#
    # F64 = F64 * F64 + F64
    # ===------------------------------------------------------------------===#
    elif _has_type[DType.float64](
        a.dtype, b.dtype, c.dtype, d.dtype
    ) and _has_shape[(1, 1, 2, 2)](a.length, b.length, c.length, d.length):
        var r = llvm_intrinsic[
            "llvm.nvvm.mma.m8n8k4.row.col.f64",
            _RegisterPackType[Float64, Float64],
        ](a, b, c[0], c[1])
        d = rebind[type_of(d)](SIMD[.float64, 2](r[0], r[1]))
    elif _has_type[DType.float64](
        a.dtype, b.dtype, c.dtype, d.dtype
    ) and _has_shape[(2, 1, 4, 4)](a.length, b.length, c.length, d.length):
        var r = llvm_intrinsic[
            "llvm.nvvm.mma.m16n8k4.row.col.f64",
            _RegisterPackType[Float64, Float64, Float64, Float64],
        ](a[0], a[1], b, c[0], c[1], c[2], c[3])
        d = rebind[type_of(d)](SIMD[.float64, 4](r[0], r[1], r[2], r[3]))
    elif _has_type[DType.float64](
        a.dtype, b.dtype, c.dtype, d.dtype
    ) and _has_shape[(4, 2, 4, 4)](a.length, b.length, c.length, d.length):
        var r = llvm_intrinsic[
            "llvm.nvvm.mma.m16n8k8.row.col.f64",
            _RegisterPackType[Float64, Float64, Float64, Float64],
        ](a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], c[2], c[3])
        d = rebind[type_of(d)](SIMD[.float64, 4](r[0], r[1], r[2], r[3]))
    elif _has_type[DType.float64](
        a.dtype, b.dtype, c.dtype, d.dtype
    ) and _has_shape[(8, 4, 4, 4)](a.length, b.length, c.length, d.length):
        var r = llvm_intrinsic[
            "llvm.nvvm.mma.m16n8k16.row.col.f64",
            _RegisterPackType[Float64, Float64, Float64, Float64],
        ](
            a[0],
            a[1],
            a[2],
            a[3],
            a[4],
            a[5],
            a[6],
            a[7],
            b[0],
            b[1],
            b[2],
            b[3],
            c[0],
            c[1],
            c[2],
            c[3],
        )
        d = rebind[type_of(d)](SIMD[.float64, 4](r[0], r[1], r[2], r[3]))

    else:
        _unsupported_mma_op(d, a, b, c)
