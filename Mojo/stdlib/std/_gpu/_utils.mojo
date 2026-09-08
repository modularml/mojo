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


from std.collections.string.string_span import (
    _get_kgen_string,
    get_static_string,
)
from std.utils import StaticTuple


# ===-----------------------------------------------------------------------===#
# MLIR type conversion utils
# ===-----------------------------------------------------------------------===#


@always_inline
def to_llvm_shared_cluster_mem_ptr[
    type: AnyType
](
    ptr: Pointer[type, address_space=.SHARED_CLUSTER, ...]
) -> __mlir_type.`!llvm.ptr<7>`:
    """Cast shared cluster memory pointer to LLVMPointer Type.

    Args:
        ptr: Shared cluster memory pointer.
    """
    return __mlir_op.`builtin.unrealized_conversion_cast`[
        _type=__mlir_type.`!llvm.ptr<7>`
    ](ptr)


@always_inline
def to_llvm_global_mem_ptr[
    type: AnyType
](ptr: Pointer[type, address_space=.GLOBAL, ...]) -> __mlir_type.`!llvm.ptr<1>`:
    """Cast global memory pointer to LLVMPointer Type.

    Args:
        ptr: Global memory pointer.

    Returns:
        A pointer of type !llvm.ptr<1>.
    """
    return __mlir_op.`builtin.unrealized_conversion_cast`[
        _type=__mlir_type.`!llvm.ptr<1>`
    ](ptr)


@always_inline
def to_llvm_shared_mem_ptr[
    type: AnyType
](ptr: Pointer[type, address_space=.SHARED, ...]) -> __mlir_type.`!llvm.ptr<3>`:
    """Cast shared memory pointer to LLVMPointer Type.

    Args:
        ptr: Shared memory pointer.

    Returns:
        A pointer of type !llvm.ptr<3>.
    """
    return __mlir_op.`builtin.unrealized_conversion_cast`[
        _type=__mlir_type.`!llvm.ptr<3>`
    ](ptr)


@always_inline
def to_llvm_ptr[
    type: AnyType
](ptr: Pointer[type, ...]) -> __mlir_type.`!llvm.ptr`:
    """Cast a pointer to LLVMPointer Type.

    Args:
        ptr: A pointer.

    Returns:
        A pointer of type !llvm.ptr.
    """
    return __mlir_op.`builtin.unrealized_conversion_cast`[
        _type=__mlir_type.`!llvm.ptr`
    ](ptr)


@always_inline
def to_i32(val: Int32) -> __mlir_type.i32:
    """Cast Scalar I32 value into MLIR i32.

    Args:
        val: Scalar I32 value.

    Returns:
       Input casted to MLIR i32 value.
    """
    return __mlir_op.`pop.cast_to_builtin`[_type=__mlir_type.`i32`](
        val._mlir_value
    )


@always_inline
def to_i16(val: UInt16) -> __mlir_type.i16:
    """Cast a scalar UInt16 value into MLIR i16.

    Args:
        val: Scalar I16 value.

    Returns:
       The input value cast to an MLIR i16.
    """
    return __mlir_op.`pop.cast_to_builtin`[_type=__mlir_type.`i16`](
        val._mlir_value
    )


@always_inline
def to_i64(val: Int64) -> __mlir_type.i64:
    """Cast Scalar I64 value into MLIR i64.

    Args:
        val: Scalar I64 value.

    Returns:
       Input casted to MLIR i64 value.
    """
    return __mlir_op.`pop.cast_to_builtin`[_type=__mlir_type.`i64`](
        val._mlir_value
    )


comptime _dtype_to_llvm_type_f8[dtype: DType] = __mlir_type.`i8` if dtype in (
    DType.float8_e8m0fnu,
    DType.float8_e3m4,
    DType.float8_e4m3fn,
    DType.float8_e4m3fnuz,
    DType.float8_e5m2,
    DType.float8_e5m2fnuz,
) else __mlir_type.`!kgen.none`

comptime _dtype_to_llvm_type_bf16[
    dtype: DType
] = __mlir_type.`bf16` if dtype == .bfloat16 else _dtype_to_llvm_type_f8[dtype]

comptime _dtype_to_llvm_type_f16[
    dtype: DType
] = __mlir_type.`f16` if dtype == .float16 else _dtype_to_llvm_type_bf16[dtype]

comptime _dtype_to_llvm_type_f32[
    dtype: DType
] = __mlir_type.`f32` if dtype == .float32 else _dtype_to_llvm_type_f16[dtype]

comptime _dtype_to_llvm_type_f64[
    dtype: DType
] = __mlir_type.`f64` if dtype == .float64 else _dtype_to_llvm_type_f32[dtype]

comptime _dtype_to_llvm_type_i32[dtype: DType] = __mlir_type.`i32` if dtype in (
    DType.int32,
    DType.uint32,
) else _dtype_to_llvm_type_f64[dtype]

comptime _dtype_to_llvm_type_i64[dtype: DType] = __mlir_type.`i64` if dtype in (
    DType.int64,
    DType.uint64,
) else _dtype_to_llvm_type_i32[dtype]

comptime dtype_to_llvm_type[dtype: DType] = _dtype_to_llvm_type_i64[dtype]


@always_inline("nodebug")
def _dtype_to_llvm_type_str[dtype: DType]() -> StaticString:
    comptime if dtype == .float32:
        return "f32"
    elif dtype == .float16:
        return "f16"
    elif dtype == .bfloat16:
        return "bf16"
    elif dtype == .float64:
        return "f64"
    elif dtype == .int32:
        return "i32"
    elif dtype == .uint32:
        return "i32"
    elif dtype == .int64:
        return "i64"
    elif dtype == .uint64:
        return "i64"
    else:
        return "i8"  # float8 variants


@always_inline("nodebug")
def _get_llvm_struct_fields[n: Int, dtype: DType]() -> StaticString:
    comptime s = _dtype_to_llvm_type_str[dtype]()
    comptime if n == 1:
        return s
    else:
        return get_static_string[
            s, ", ", _get_llvm_struct_fields[n - 1, dtype]()
        ]()


comptime llvm_struct_dtype_splat_type[
    dtype: DType, n: Int
] = __mlir_deferred_type[
    `!llvm.struct<(`,
    +_get_kgen_string[_get_llvm_struct_fields[n, dtype]()](),
    `)>`,
]


@always_inline
def simd_to_llvm_struct[
    dtype: DType, n: Int
](simd: SIMD[dtype, n]) -> llvm_struct_dtype_splat_type[dtype, n]:
    """Repack a SIMD value to a `!llvm.struct`.

    Args:
        simd: A SIMD value.

    Returns:
        A `!llvm.struct` with the same number of fields as the SIMD value.
    """
    var llvmst = __mlir_op.`llvm.mlir.undef`[
        _type=__mlir_deferred_type[
            `!llvm.struct<(`,
            +_get_kgen_string[_get_llvm_struct_fields[n, dtype]()](),
            `)>`,
        ]
    ]()

    var st = __mlir_op.`builtin.unrealized_conversion_cast`[
        _type=__mlir_deferred_type[
            `!kgen.struct<(`,
            +_get_kgen_string[_get_kgen_struct_fields[n, dtype]()](),
            `)>`,
        ]
    ](llvmst)

    comptime for i in range(n):
        var e = simd[i]
        st = __mlir_op.`kgen.struct.replace`[
            _type=__mlir_deferred_type[
                `!kgen.struct<(`,
                +_get_kgen_string[_get_kgen_struct_fields[n, dtype]()](),
                `)>`,
            ],
            index=__mlir_attr[i.__mlir_index__(), `:index`],
        ](e, st)

    return rebind[llvm_struct_dtype_splat_type[dtype, n]](
        __mlir_op.`builtin.unrealized_conversion_cast`[
            _type=__mlir_deferred_type[
                `!llvm.struct<(`,
                +_get_kgen_string[_get_llvm_struct_fields[n, dtype]()](),
                `)>`,
            ]
        ](st)
    )


@always_inline("nodebug")
def _dtype_to_pop_scalar_str[dtype: DType]() -> StaticString:
    comptime if dtype == .bool:
        return "!kgen.scalar<bool>"
    elif dtype == .int8:
        return "!kgen.scalar<si8>"
    elif dtype == .uint8:
        return "!kgen.scalar<ui8>"
    elif dtype == .int16:
        return "!kgen.scalar<si16>"
    elif dtype == .uint16:
        return "!kgen.scalar<ui16>"
    elif dtype == .int32:
        return "!kgen.scalar<si32>"
    elif dtype == .uint32:
        return "!kgen.scalar<ui32>"
    elif dtype == .int64:
        return "!kgen.scalar<si64>"
    elif dtype == .uint64:
        return "!kgen.scalar<ui64>"
    elif dtype == .float16:
        return "!kgen.scalar<f16>"
    elif dtype == .bfloat16:
        return "!kgen.scalar<bf16>"
    elif dtype == .float32:
        return "!kgen.scalar<f32>"
    elif dtype == .float64:
        return "!kgen.scalar<f64>"
    elif dtype == .float8_e5m2:
        return "!kgen.scalar<f8E5M2>"
    elif dtype == .float8_e5m2fnuz:
        return "!kgen.scalar<f8E5M2FNUZ>"
    elif dtype == .float8_e4m3fn:
        return "!kgen.scalar<f8E4M3>"
    elif dtype == .float8_e4m3fnuz:
        return "!kgen.scalar<f8E4M3FNUZ>"
    elif dtype == .float8_e3m4:
        return "!kgen.scalar<f8E3M4>"
    elif dtype == .float8_e8m0fnu:
        return "!kgen.scalar<f8E8M0FNU>"
    else:
        comptime assert False, "unsupported dtype for !kgen.scalar"


@always_inline("nodebug")
def _get_kgen_struct_fields[n: Int, dtype: DType]() -> StaticString:
    comptime s = _dtype_to_pop_scalar_str[dtype]()
    comptime if n == 1:
        return s
    else:
        return get_static_string[
            s, ", ", _get_kgen_struct_fields[n - 1, dtype]()
        ]()


# `!kgen.struct` of N copies of `Scalar[dtype]`, built natively via
# `TypeList.splat`; extracts work through `kgen.pack.extract` without a
# deferred type.
comptime _kgen_pack_splat_type[dtype: DType, n: Int] = __mlir_type[
    `!kgen.struct<`,
    ~TypeList.splat[
        Trait=TrivialRegisterPassable, count=n, type=Scalar[dtype]
    ]().values,
    ` isParamPack>`,
]


@always_inline
def llvm_struct_to_simd[
    dtype: DType, n: Int
](llvmst: llvm_struct_dtype_splat_type[dtype, n]) -> SIMD[dtype, n]:
    """Repack value of a `!llvm.struct` type to SIMD.

    Args:
        llvmst: A `!llvm.struct` value.

    Returns:
        A SIMD value with the same number of elements as the `!llvm.struct`.
    """
    var simd = SIMD[dtype, n]()
    var pack = __mlir_op.`builtin.unrealized_conversion_cast`[
        _type=_kgen_pack_splat_type[dtype, n]
    ](llvmst)

    comptime for i in range(n):
        # `kgen.pack.extract` infers a parametric element type; an
        # unrealized_conversion_cast retypes it to `!kgen.scalar<dtype>`.
        var e = __mlir_op.`builtin.unrealized_conversion_cast`[
            _type=Scalar[dtype]._mlir_type
        ](__mlir_op.`kgen.struct.extract`[index=i.__mlir_index__()](pack))
        simd[i] = Scalar[dtype](mlir_value=e)
    return simd


@always_inline
def array_to_llvm_struct[
    dtype: DType, n: Int
](array: StaticTuple[Scalar[dtype], n]) -> llvm_struct_dtype_splat_type[
    dtype, n
]:
    """Repack a StaticTuple value to a `!llvm.struct`.

    Args:
        array: A array value.

    Returns:
        A `!llvm.struct` with the same number of fields as the array value.
    """
    var llvmst = __mlir_op.`llvm.mlir.undef`[
        _type=__mlir_deferred_type[
            `!llvm.struct<(`,
            +_get_kgen_string[_get_llvm_struct_fields[n, dtype]()](),
            `)>`,
        ]
    ]()

    var st = __mlir_op.`builtin.unrealized_conversion_cast`[
        _type=__mlir_deferred_type[
            `!kgen.struct<(`,
            +_get_kgen_string[_get_kgen_struct_fields[n, dtype]()](),
            `)>`,
        ]
    ](llvmst)

    comptime for i in range(n):
        var e = array[i]
        st = __mlir_op.`kgen.struct.replace`[
            _type=__mlir_deferred_type[
                `!kgen.struct<(`,
                +_get_kgen_string[_get_kgen_struct_fields[n, dtype]()](),
                `)>`,
            ],
            index=__mlir_attr[i.__mlir_index__(), `:index`],
        ](e, st)

    return rebind[llvm_struct_dtype_splat_type[dtype, n]](
        __mlir_op.`builtin.unrealized_conversion_cast`[
            _type=__mlir_deferred_type[
                `!llvm.struct<(`,
                +_get_kgen_string[_get_llvm_struct_fields[n, dtype]()](),
                `)>`,
            ]
        ](st)
    )


@always_inline
def llvm_struct_to_array[
    dtype: DType, n: Int
](llvmst: llvm_struct_dtype_splat_type[dtype, n]) -> StaticTuple[
    Scalar[dtype], n
]:
    """Repack value of a `!llvm.struct` type to StaticTuple.

    Args:
        llvmst: A `!llvm.struct` value.

    Returns:
        A array value with the same number of elements as the `!llvm.struct`.
    """
    var array = StaticTuple[Scalar[dtype], n]()
    var pack = __mlir_op.`builtin.unrealized_conversion_cast`[
        _type=_kgen_pack_splat_type[dtype, n]
    ](llvmst)

    comptime for i in range(n):
        var e = __mlir_op.`builtin.unrealized_conversion_cast`[
            _type=Scalar[dtype]._mlir_type
        ](__mlir_op.`kgen.struct.extract`[index=i.__mlir_index__()](pack))
        array[i] = Scalar[dtype](mlir_value=e)
    return array
