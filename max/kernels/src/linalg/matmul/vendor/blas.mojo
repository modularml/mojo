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

"""Vendor BLAS matmul wrappers for cuBLAS, cuBLASLt, rocBLAS, and hipBLASLt.

Exposes a backend-agnostic `matmul` entry point plus `Backend` and `Handle`
types that abstract over the NVIDIA and AMD vendor libraries so callers can
dispatch GEMM operations without binding to a specific vendor API.
"""

from std.sys import has_amd_gpu_accelerator, size_of
from std.math import ceildiv
from std.ffi import _get_global_or_null, external_call

import _rocblas
from _cublas.cublas import (
    Algorithm,
    ComputeType,
    _convert_to_cublas_datatype,
    _convert_to_cublas_transpose,
    check_cublas_error,
    cublasHandle_t,
    cublasCreate,
    cublasDestroy,
    cublasGemmEx,
    cublasLoggerConfigure,
    cublasMath_t,
    cublasOperation_t,
    cublasSetMathMode,
    cublasSetStream,
    cublasSetWorkspace,
)
from _cublas.cublaslt import (
    Preference,
    cublasLtLoggerSetLevel,
    cublasLtMatmul,
    cublasLtMatmulAlgoGetHeuristic,
    cublasLtMatmulDesc_t,
    cublasLtMatmulDescAttributes_t,
    cublasLtMatmulDescCreate,
    cublasLtMatmulDescDestroy,
    cublasLtMatmulDescSetAttribute,
    cublasLtMatmulHeuristicResult_t,
    cublasLtMatmulPreference_t,
    cublasLtMatmulPreferenceCreate,
    cublasLtMatmulPreferenceDestroy,
    cublasLtMatmulPreferenceSetAttribute,
    cublasLtMatrixLayout_t,
    cublasLtMatrixLayoutCreate,
    cublasLtMatrixLayoutDestroy,
    cublasLtMatmulMatrixScale_t,
)
from _cublas.dtype import DataType
from _rocblas.hipblaslt import (
    _check_hipblas_error,
    _convert_to_hip_datatype,
    hipblasComputeType_t,
    hipblasLtCreate,
    hipblasLtDestroy,
    hipblasLtHandle_t,
    hipblasLtMatmul,
    hipblasLtMatmulAlgoGetHeuristic,
    hipblasLtMatmulDesc_t,
    hipblasLtMatmulDescAttributes_t,
    hipblasLtMatmulDescCreate,
    hipblasLtMatmulDescDestroy,
    hipblasLtMatmulDescSetAttribute,
    hipblasLtMatmulHeuristicResult_t,
    hipblasLtMatmulMatrixScale_t,
    hipblasLtMatmulPreference_t,
    hipblasLtMatmulPreferenceCreate,
    hipblasLtMatmulPreferenceDestroy,
    hipblasLtMatrixLayout_t,
    hipblasLtMatrixLayoutCreate,
    hipblasLtMatrixLayoutDestroy,
    hipblasLtMatrixLayoutSetAttribute,
    hipblasLtMatmulLayoutAttribute_t,
    hipblasOperation_t,
    hipDataType_t,
)
from _rocblas.rocblas import (
    rocblas_set_stream,
    rocblas_gemm_ex,
    rocblas_create_handle,
    rocblas_destroy_handle,
)
from max.gpu.host import DeviceContext
from max.gpu.host._amdgpu_hip import HIP
from max.gpu.host._nvidia_cuda import CUDA
from layout import (
    Coord,
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from layout.tile_tensor import NullableTileTensor
from std.memory.alloc import Layout as AllocLayout
from std.utils import IndexList
from std.utils.variant import Variant
from max.gpu.host.info import B200, _is_sm10x_gpu, _is_sm12x_gpu
from std.collections import OptionalReg

from max.runtime.tracing import Trace, TraceLevel, get_safe_task_id, trace_arg

from linalg.fp4_utils import (
    SF_ATOM_M,
    SF_ATOM_K,
    SF_MN_GROUP_SIZE,
    MXFP4_SF_VECTOR_SIZE,
    MXFP8_SF_VECTOR_SIZE,
    MXFP8_SF_DTYPE,
    NVFP4_SF_VECTOR_SIZE,
    NVFP4_SF_DTYPE,
)

# ===----------------------------------------------------------------------===#
# Backend
# ===----------------------------------------------------------------------===#


struct Backend(Equatable, TrivialRegisterPassable, Writable):
    """Identifies which vendor BLAS library backs a matmul operation.

    Acts as a comptime-selectable tag (`AUTOMATIC`, `CUBLAS`, `CUBLASLT`,
    `ROCBLAS`, `HIPBLASLT`) used to parameterize `Handle` and dispatch to the
    matching vendor implementation.
    """

    var _value: Int32

    comptime AUTOMATIC = Self(0)
    comptime CUBLAS = Self(1)
    comptime CUBLASLT = Self(2)
    comptime ROCBLAS = Self(3)
    comptime HIPBLASLT = Self(4)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __is__(self, other: Self) -> Bool:
        return self == other

    def __isnot__(self, other: Self) -> Bool:
        return self != other

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    def __int__(self) -> Int:
        return Int(self._value)

    def write_to(self, mut writer: Some[Writer]):
        if self is Self.AUTOMATIC:
            return writer.write("AUTOMATIC")
        if self is Self.CUBLAS:
            return writer.write("CUBLAS")
        if self is Self.CUBLASLT:
            return writer.write("CUBLASLT")
        if self is Self.ROCBLAS:
            return writer.write("ROCBLAS")
        writer.write("HIPBLASLT")


def _resolve_backend[
    backend: Backend, dtype: Optional[DType] = None
]() -> Backend:
    comptime if backend is not Backend.AUTOMATIC:
        return backend
    # TODO: Remove this once we have a proper hipBLASLt backend for float32.
    elif dtype == DType.float32 and has_amd_gpu_accelerator():
        return Backend.ROCBLAS
    elif has_amd_gpu_accelerator():
        return Backend.HIPBLASLT
    # TODO (KERN-2238): uint8 is a proxy data type for two Float4-E2M1 values for now.
    # Replace this with float4-e2m1fn when GENAI-337 is fixed.
    elif (
        dtype.value().is_float8() if dtype else False
    ) or dtype == DType.uint8:
        return Backend.CUBLASLT
    return Backend.CUBLAS


# ===----------------------------------------------------------------------===#
# Handle
# ===----------------------------------------------------------------------===#


struct Handle[backend: Backend = _resolve_backend[Backend.AUTOMATIC]()](
    ImplicitlyCopyable
):
    """Owns a vendor BLAS library handle for a resolved `Backend`.

    Parameterized by a `Backend` (defaulting to the automatically resolved
    backend) and stores the underlying cuBLAS, rocBLAS, or hipBLASLt handle in
    a `Variant`. Construction creates the vendor handle and `__exit__` destroys
    it, so instances should be used as context managers.
    """

    comptime resolved_backend = _resolve_backend[Self.backend]()
    comptime _cublas_type = Optional[OpaquePointer[MutAnyOrigin]]
    comptime _rocblas_type = _rocblas.Handle
    comptime _hipblaslt_type = hipblasLtHandle_t
    comptime type = Variant[
        Self._cublas_type,
        Self._rocblas_type,
        Self._hipblaslt_type,
    ]

    @__allow_legacy_any_origin_fields
    var _handle: Self.type

    def __init__(out self) raises:
        comptime if Self.resolved_backend in (Backend.CUBLAS, Backend.CUBLASLT):
            var handle = Self._cublas_type()
            check_cublas_error(cublasCreate(UnsafePointer(to=handle)))
            self._handle = handle
        elif Self.resolved_backend is Backend.ROCBLAS:
            var handle = Self._rocblas_type()
            _rocblas.check_error(
                rocblas_create_handle(UnsafePointer(to=handle))
            )
            self._handle = handle
        elif Self.resolved_backend is Backend.HIPBLASLT:
            var handle = Self._hipblaslt_type()
            _check_hipblas_error(hipblasLtCreate(UnsafePointer(to=handle)))
            self._handle = handle
        else:
            raise Error(
                "the backend '",
                Self.backend,
                "' is not currently supported",
            )

    @always_inline
    def __enter__(self) -> Self:
        return self

    @always_inline
    def __exit__(mut self) raises:
        comptime if Self.resolved_backend in (Backend.CUBLAS, Backend.CUBLASLT):
            check_cublas_error(cublasDestroy(self._get_cublas()))
            self._handle = Self._cublas_type()
            return
        elif Self.resolved_backend is Backend.ROCBLAS:
            _rocblas.check_error(rocblas_destroy_handle(self._get_rocblas()))
            self._handle = Self._rocblas_type()
            return
        elif Self.resolved_backend is Backend.HIPBLASLT:
            _check_hipblas_error(hipblasLtDestroy(self._get_hipblaslt()))
            self._handle = Self._hipblaslt_type()
            return

        raise Error("the backend is not currently supported")

    def _is_null(self) -> Bool:
        comptime if Self.resolved_backend in (Backend.CUBLAS, Backend.CUBLASLT):
            return self._get_cublas() == Self._cublas_type()
        elif Self.resolved_backend is Backend.ROCBLAS:
            return self._get_rocblas() == Self._rocblas_type()
        elif Self.resolved_backend is Backend.HIPBLASLT:
            return self._get_hipblaslt() == Self._hipblaslt_type()

        return False

    def _get_cublas(self) -> Self._cublas_type:
        comptime assert Self.resolved_backend in (
            Backend.CUBLAS,
            Backend.CUBLASLT,
        ), "backend must be CUBLAS/CUBLASLT"
        return self._handle[Self._cublas_type]

    def _get_rocblas(self) -> Self._rocblas_type:
        comptime assert (
            Self.resolved_backend is Backend.ROCBLAS
        ), "backend must be ROCBLAS"
        return self._handle[Self._rocblas_type]

    def _get_hipblaslt(self) -> Self._hipblaslt_type:
        comptime assert (
            Self.resolved_backend is Backend.HIPBLASLT
        ), "backend must be HIPBLASLT"
        return self._handle[Self._hipblaslt_type]

    def __is__(self, other: Backend) -> Bool:
        return Self.resolved_backend is other

    def __isnot__(self, other: Backend) -> Bool:
        return Self.resolved_backend is not other


# ===----------------------------------------------------------------------===#
# Matmul
# ===----------------------------------------------------------------------===#

comptime _DEBUG_VENDOR_BLAS = False


@always_inline
def _ffi_void_ptr[
    T: AnyType, origin: Origin, addr: AddressSpace
](ptr: UnsafePointer[T, origin, address_space=addr]) -> OpaquePointer[
    MutAnyOrigin
]:
    """Cast any pointer to a void pointer for vendor FFI calls."""
    return rebind[OpaquePointer[MutAnyOrigin]](ptr)


@always_inline
def _ffi_void_ptr[
    T: AnyType, origin: Origin, addr: AddressSpace
](
    ptr: Optional[UnsafePointer[T, origin, address_space=addr]]
) -> OptionalPointer[NoneType, MutAnyOrigin]:
    """Cast an optional non-null pointer to a nullable void pointer for vendor
    FFI calls.

    Returns None when the optional is empty.
    """
    if ptr:
        return rebind[OpaquePointer[MutAnyOrigin]](ptr.unsafe_value())
    return None


def _attach_handle_to_stream(ctx: DeviceContext, handle: Handle) raises:
    comptime if handle.resolved_backend in (Backend.CUBLAS, Backend.CUBLASLT):
        check_cublas_error(
            cublasSetStream(handle._get_cublas(), CUDA(ctx.stream()))
        )

        comptime if _DEBUG_VENDOR_BLAS:
            comptime if handle.resolved_backend is Backend.CUBLAS:
                check_cublas_error(
                    cublasLoggerConfigure(
                        1,
                        1,
                        0,
                        OptionalPointer[Int8, MutAnyOrigin](),
                    )
                )
            else:
                check_cublas_error(cublasLtLoggerSetLevel(5))

    elif handle.resolved_backend is Backend.ROCBLAS:
        _rocblas.check_error(
            rocblas_set_stream(handle._get_rocblas(), HIP(ctx.stream()))
        )


def _get_global_handle[
    dtype: DType,
    backend: Backend = _resolve_backend[Backend.AUTOMATIC, dtype=dtype](),
](ctx: DeviceContext) raises -> Handle[backend]:
    var HANDLE_NAME = String(t"LINALG_VENDOR_BLAS_{backend}_{ctx.id()}")
    var global_ptr = _get_global_or_null(HANDLE_NAME)
    if global_ptr:
        var ptr = global_ptr.value().unsafe_bitcast[Handle[backend]]()
        _attach_handle_to_stream(ctx, ptr[])
        return ptr[]

    # Otherwise, we have not initialized the handle yet.
    var handle_ptr = alloc(AllocLayout[Handle[backend]].single()).unsafe_leak()
    handle_ptr.unsafe_write(Handle[backend]())
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(HANDLE_NAME),
        handle_ptr.bitcast[NoneType](),
    )

    _attach_handle_to_stream(ctx, handle_ptr[])

    return handle_ptr[]


def matmul[
    use_tf32: Bool = False,
](
    ctx: DeviceContext,
    c: NullableTileTensor[mut=True, ...],
    a: TileTensor,
    b: TileTensor,
    *,
    c_row_major: Bool = False,
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    alpha: Float32 = 1.0,
    beta: Float32 = 0.0,
    batch_size: Int = 1,
) raises:
    """Matmul using the vendor BLAS library for NullableTileTensor operands.

    Note: This overload does not support a_scales/b_scales. Add scale
    parameters here when a TileTensor caller needs scaled vendor matmul.
    """
    comptime assert c.flat_rank == 2, "c must be of rank 2"
    comptime assert a.flat_rank == 2, "a must be of rank 2"
    comptime assert b.flat_rank == 2, "b must be of rank 2"

    with ctx.push_context() as cur_ctx:
        return matmul[use_tf32=use_tf32](
            cur_ctx,
            _get_global_handle[a.dtype](ctx),
            c,
            a,
            b,
            c_row_major=c_row_major,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            alpha=alpha,
            beta=beta,
            batch_size=batch_size,
        )


def matmul[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    c_layout: Layout,
    a_layout: Layout,
    b_layout: Layout,
    *,
    use_tf32: Bool = False,
    scales_type: DType = a_type,
    a_scales_layout: Layout = Layout.row_major(UNKNOWN_VALUE),
    b_scales_layout: Layout = Layout.row_major(UNKNOWN_VALUE),
](
    ctx: DeviceContext,
    c_tensor: LayoutTensor[mut=True, c_type, c_layout, _],
    a_tensor: LayoutTensor[mut=False, a_type, a_layout, _],
    b_tensor: LayoutTensor[mut=False, b_type, b_layout, _],
    *,
    a_scales: OptionalReg[
        LayoutTensor[scales_type, a_scales_layout, ImmutAnyOrigin]
    ] = None,
    b_scales: OptionalReg[
        LayoutTensor[scales_type, b_scales_layout, ImmutAnyOrigin]
    ] = None,
    c_row_major: Bool = False,
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    alpha: Float32 = 1.0,
    beta: Float32 = 0.0,
    batch_size: Int = 1,
) raises:
    var c_tt = TileTensor(
        rebind[UnsafePointer[Scalar[c_type], MutAnyOrigin]](c_tensor.ptr),
        row_major(Coord(c_tensor.dim(0), c_tensor.dim(1))),
    )
    var a_tt = TileTensor(
        rebind[UnsafePointer[Scalar[a_type], ImmutAnyOrigin]](a_tensor.ptr),
        row_major(Coord(a_tensor.dim(0), a_tensor.dim(1))),
    )
    var b_tt = TileTensor(
        rebind[UnsafePointer[Scalar[b_type], ImmutAnyOrigin]](b_tensor.ptr),
        row_major(Coord(b_tensor.dim(0), b_tensor.dim(1))),
    )
    with ctx.push_context() as cur_ctx:
        return matmul[use_tf32=use_tf32, scales_type=scales_type](
            cur_ctx,
            _get_global_handle[a_type](ctx),
            c_tt,
            a_tt,
            b_tt,
            a_scales=a_scales,
            b_scales=b_scales,
            c_row_major=c_row_major,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            alpha=alpha,
            beta=beta,
            batch_size=batch_size,
        )


def matmul[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    *,
    use_tf32: Bool = False,
    scales_type: DType,
](
    ctx: DeviceContext,
    c_tensor: TileTensor[mut=True, c_type, ...],
    a_tensor: TileTensor[a_type, ...],
    b_tensor: TileTensor[b_type, ...],
    *,
    a_scales: TileTensor[scales_type, ...],
    b_scales: TileTensor[scales_type, ...],
    c_row_major: Bool = False,
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    alpha: Float32 = 1.0,
    beta: Float32 = 0.0,
    batch_size: Int = 1,
) raises:
    """Overload accepting TileTensors for all operands and scale factors.

    Converts TileTensor scale factors to LayoutTensor for the core dispatch
    which passes them through to the cublasLt backend.
    """
    comptime sfa_layout = Layout.row_major(
        a_scales.static_shape[0],
        a_scales.static_shape[1],
        a_scales.static_shape[2],
        a_scales.static_shape[3],
        a_scales.static_shape[4],
    )
    comptime sfb_layout = Layout.row_major(
        b_scales.static_shape[0],
        b_scales.static_shape[1],
        b_scales.static_shape[2],
        b_scales.static_shape[3],
        b_scales.static_shape[4],
    )

    var a_scales_lt = LayoutTensor[scales_type, sfa_layout, ImmutAnyOrigin](
        rebind[UnsafePointer[Scalar[scales_type], ImmutAnyOrigin]](
            a_scales.ptr
        ),
        RuntimeLayout[sfa_layout].row_major(
            IndexList[5](
                Int(a_scales.dim[0]()),
                Int(a_scales.dim[1]()),
                Int(a_scales.dim[2]()),
                Int(a_scales.dim[3]()),
                Int(a_scales.dim[4]()),
            )
        ),
    )
    var b_scales_lt = LayoutTensor[scales_type, sfb_layout, ImmutAnyOrigin](
        rebind[UnsafePointer[Scalar[scales_type], ImmutAnyOrigin]](
            b_scales.ptr
        ),
        RuntimeLayout[sfb_layout].row_major(
            IndexList[5](
                Int(b_scales.dim[0]()),
                Int(b_scales.dim[1]()),
                Int(b_scales.dim[2]()),
                Int(b_scales.dim[3]()),
                Int(b_scales.dim[4]()),
            )
        ),
    )

    with ctx.push_context() as cur_ctx:
        matmul[use_tf32=use_tf32, scales_type=scales_type](
            cur_ctx,
            _get_global_handle[a_type](ctx),
            c_tensor,
            a_tensor,
            b_tensor,
            a_scales=a_scales_lt,
            b_scales=b_scales_lt,
            c_row_major=c_row_major,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            alpha=alpha,
            beta=beta,
            batch_size=batch_size,
        )


def matmul[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    use_tf32: Bool = False,
    scales_type: DType = a_type,
    a_scales_layout: Layout = Layout.row_major(UNKNOWN_VALUE),
    b_scales_layout: Layout = Layout.row_major(UNKNOWN_VALUE),
](
    ctx: DeviceContext,
    handle: Handle,
    c_tensor: NullableTileTensor[mut=True, c_type, ...],
    a_tensor: TileTensor[a_type, ...],
    b_tensor: TileTensor[b_type, ...],
    *,
    a_scales: OptionalReg[
        LayoutTensor[scales_type, a_scales_layout, ImmutAnyOrigin]
    ] = None,
    b_scales: OptionalReg[
        LayoutTensor[scales_type, b_scales_layout, ImmutAnyOrigin]
    ] = None,
    c_row_major: Bool = False,
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    alpha: Float32 = 1.0,
    beta: Float32 = 0.0,
    batch_size: Int = 1,
) raises:
    @always_inline
    def description_fn() {imm} -> String:
        return String(
            trace_arg(
                "A",
                IndexList[2](Int(a_tensor.dim[0]()), Int(a_tensor.dim[1]())),
                a_type,
            ),
            ";",
            trace_arg(
                "B",
                IndexList[2](Int(b_tensor.dim[0]()), Int(b_tensor.dim[1]())),
                b_type,
            ),
            ";",
            trace_arg(
                "C",
                IndexList[2](Int(c_tensor.dim[0]()), Int(c_tensor.dim[1]())),
                c_type,
            ),
            ";transpose_a=",
            transpose_a,
            ";transpose_b=",
            transpose_b,
        )

    with Trace[TraceLevel.OP, target=StaticString("gpu")](
        String(t"{handle.resolved_backend}_matmul"),
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
        task_id=get_safe_task_id(ctx),
    ):
        comptime if handle.resolved_backend is Backend.CUBLAS:
            _cublas_matmul[use_tf32=use_tf32](
                ctx,
                handle._get_cublas(),
                c_tensor,
                a_tensor,
                b_tensor,
                c_row_major=c_row_major,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
                alpha=alpha,
                beta=beta,
            )
        elif handle.resolved_backend is Backend.ROCBLAS:
            _rocblas_matmul[use_tf32=use_tf32](
                ctx,
                handle._get_rocblas(),
                c_tensor,
                a_tensor,
                b_tensor,
                c_row_major=c_row_major,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
                alpha=alpha,
                beta=beta,
            )
        elif handle.resolved_backend is Backend.CUBLASLT:
            _cublasLt_matmul(
                ctx,
                handle._get_cublas(),
                c_tensor,
                a_tensor,
                b_tensor,
                a_scales=a_scales,
                b_scales=b_scales,
                c_row_major=c_row_major,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
                alpha=alpha,
                beta=beta,
            )
        elif handle.resolved_backend is Backend.HIPBLASLT:
            _hipblasLt_matmul(
                ctx,
                handle._get_hipblaslt(),
                c_tensor,
                a_tensor,
                b_tensor,
                a_scales=a_scales,
                b_scales=b_scales,
                c_row_major=c_row_major,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
                alpha=alpha,
                beta=beta,
                batch_size=batch_size,
            )
        else:
            raise Error(
                "the backend '",
                handle.backend,
                "' is not currently supported",
            )


# ===----------------------------------------------------------------------===#
# CUBLAS
# ===----------------------------------------------------------------------===#


def _cublas_matmul[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    use_tf32: Bool = False,
](
    ctx: DeviceContext,
    handle: cublasHandle_t,
    c: NullableTileTensor[mut=True, c_type, ...],
    a: TileTensor[a_type, ...],
    b: TileTensor[b_type, ...],
    *,
    c_row_major: Bool = False,
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    alpha: Float32 = 1.0,
    beta: Float32 = 0.0,
) raises:
    comptime assert a_type == b_type and (
        a_type == .float32 or a_type.is_half_float()
    ), (
        "Only support FP32, FP16 and BF16 for cublas wrapper. Please extend"
        " it if more types are needed."
    )

    var M = Int(c.dim[0]())
    var N = Int(c.dim[1]())
    var K = Int(a.dim[1]()) if not transpose_a else Int(a.dim[0]())

    var c_dynamic_shape = IndexList[2](M, N)
    var a_dynamic_shape = IndexList[2](Int(a.dim[0]()), Int(a.dim[1]()))
    var b_dynamic_shape = IndexList[2](Int(b.dim[0]()), Int(b.dim[1]()))

    var compute_type: ComputeType

    comptime if a_type == .float16:
        compute_type = ComputeType.COMPUTE_32F
    elif a_type == .bfloat16:
        compute_type = ComputeType.COMPUTE_32F
    else:
        compute_type = (
            ComputeType.COMPUTE_32F_FAST_TF32 if use_tf32 else ComputeType.COMPUTE_32F
        )

    # When use_tf32 is True, CUBLAS will use TF32 to speedup the computation.
    # However, the result is not bit-wise identical to the result of FP32.
    comptime if use_tf32:
        check_cublas_error(
            cublasSetMathMode(handle, cublasMath_t.CUBLAS_TF32_TENSOR_OP_MATH)
        )
    else:
        check_cublas_error(
            cublasSetMathMode(handle, cublasMath_t.CUBLAS_DEFAULT_MATH)
        )

    # Rocblas is by default column-major but we like to have the output in row-major
    # to compare with our results. To do this without an explicit transpose, we
    # can swap A, B and output a NxM column-major matrix, which is same as
    # MxN row-major i.e.
    #
    #      C: MxN_row_major = A: MxK_row_major @ B: KxN_row_major
    #   => C: NxM_col_major = B: NxK_col_major @ A: KxM_col_major
    #
    # I haven't seen any significant performance difference before and after this
    # transformation. To be rigorous though, we should set `c_is_row_major = True`
    # for accuracy validations and uses default column-major in benchmark.

    # cuBLAS's default internal workspace pool is uninitialized memory that
    # its split-K reduction kernel reads from, which trips initcheck. Supply
    # our own zeroed workspace instead to pass in.
    var workspace_size = 32 * 1024 * 1024
    var workspace = ctx.enqueue_create_buffer[.uint8](workspace_size)
    ctx.enqueue_memset(workspace, UInt8(0))
    check_cublas_error(
        cublasSetWorkspace(
            handle,
            workspace.unsafe_ptr().bitcast[NoneType]().as_unsafe_any_origin(),
            workspace_size,
        )
    )

    if c_row_major:
        # Forward each operand's row stride as the leading dimension; a
        # non-unit inner stride can't be expressed, so fall back to naive.
        if (
            a.dynamic_stride(1) != 1
            or b.dynamic_stride(1) != 1
            or c.layout.stride[1]().value() != 1
        ):
            raise Error("vendor BLAS matmul requires a unit inner stride")
        var a_lead = Int32(a.dynamic_stride(0))
        var b_lead = Int32(b.dynamic_stride(0))
        var c_lead = Int32(c.layout.stride[0]().value())
        check_cublas_error(
            cublasGemmEx(
                handle,
                _convert_to_cublas_transpose(transpose_b),
                _convert_to_cublas_transpose(transpose_a),
                Int32(N),
                Int32(M),
                Int32(K),
                UnsafePointer(to=alpha)
                .bitcast[NoneType]()
                .as_imm()
                .as_unsafe_any_origin(),
                _ffi_void_ptr(b.ptr),
                _convert_to_cublas_datatype[b_type](),
                b_lead,
                _ffi_void_ptr(a.ptr),
                _convert_to_cublas_datatype[a_type](),
                a_lead,
                UnsafePointer(to=beta)
                .bitcast[NoneType]()
                .as_imm()
                .as_unsafe_any_origin(),
                _ffi_void_ptr(c.ptr),
                _convert_to_cublas_datatype[c_type](),
                c_lead,
                compute_type,
                Algorithm.DEFAULT,
            ),
            msg=String(
                "failed to operate on cublas on the shape C=",
                c_dynamic_shape,
                "x",
                c_type,
                ", A=",
                a_dynamic_shape,
                "x",
                a_type,
                ", B=",
                b_dynamic_shape,
                "x",
                b_type,
            ),
        )
        _ = workspace^
        return
    # Default column-major.
    check_cublas_error(
        cublasGemmEx(
            handle,
            _convert_to_cublas_transpose(transpose_a),
            _convert_to_cublas_transpose(transpose_b),
            Int32(M),
            Int32(N),
            Int32(K),
            UnsafePointer(to=alpha)
            .bitcast[NoneType]()
            .as_imm()
            .as_unsafe_any_origin(),
            _ffi_void_ptr(a.ptr),
            _convert_to_cublas_datatype[a_type](),
            Int32(M),
            _ffi_void_ptr(b.ptr),
            _convert_to_cublas_datatype[b_type](),
            Int32(N) if transpose_b else Int32(K),
            UnsafePointer(to=beta)
            .bitcast[NoneType]()
            .as_imm()
            .as_unsafe_any_origin(),
            _ffi_void_ptr(c.ptr),
            _convert_to_cublas_datatype[c_type](),
            Int32(M),
            compute_type,
            Algorithm.DEFAULT,
        ),
        msg=String(
            "failed to operate on cublas on the shape C=",
            c_dynamic_shape,
            "x",
            c_type,
            ", A=",
            a_dynamic_shape,
            "x",
            a_type,
            ", B=",
            b_dynamic_shape,
            "x",
            b_type,
        ),
    )
    _ = workspace^


# ===----------------------------------------------------------------------===#
# ROCBLAS
# ===----------------------------------------------------------------------===#


def _rocblas_matmul[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    use_tf32: Bool = False,
](
    ctx: DeviceContext,
    handle: _rocblas.Handle,
    c: NullableTileTensor[mut=True, c_type, ...],
    a: TileTensor[a_type, ...],
    b: TileTensor[b_type, ...],
    *,
    c_row_major: Bool = False,
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    alpha: Float32 = 1.0,
    beta: Float32 = 0.0,
) raises:
    comptime assert a_type == b_type and (
        a_type == .float32 or a_type.is_half_float()
    ), (
        "Only support FP32, FP16 and BF16 for cublas wrapper. Please extend"
        " it if more types are needed."
    )

    var M = Int(c.dim[0]())
    var N = Int(c.dim[1]())
    var K = Int(a.dim[1]()) if not transpose_a else Int(a.dim[0]())

    var compute_type = _rocblas.DataType(DType.float32)

    # Cublas is by default column-major but we like to have the output in row-major
    # to compare with our results. To do this without an explicit transpose, we
    # can swap A, B and output a NxM column-major matrix, which is same as
    # MxN row-major i.e.
    #
    #      C: MxN_row_major = A: MxK_row_major @ B: KxN_row_major
    #   => C: NxM_col_major = B: NxK_col_major @ A: KxM_col_major
    #
    # I haven't seen any significant performance difference before and after this
    # transformation. To be rigorous though, we should set `c_is_row_major = True`
    # for accuracy validations and uses default column-major in benchmark.

    def _convert_to_rocblas_transpose(tr: Bool) -> _rocblas.Operation:
        if tr:
            return _rocblas.Operation.TRANSPOSE
        return _rocblas.Operation.NONE

    if c_row_major:
        # Forward each operand's row stride as the leading dimension; a
        # non-unit inner stride can't be expressed, so fall back to naive.
        if (
            a.dynamic_stride(1) != 1
            or b.dynamic_stride(1) != 1
            or c.layout.stride[1]().value() != 1
        ):
            raise Error("vendor BLAS matmul requires a unit inner stride")
        var a_lead = Int32(a.dynamic_stride(0))
        var b_lead = Int32(b.dynamic_stride(0))
        var c_lead = Int32(c.layout.stride[0]().value())
        return _rocblas.check_error(
            rocblas_gemm_ex(
                handle,
                _convert_to_rocblas_transpose(transpose_b),
                _convert_to_rocblas_transpose(transpose_a),
                Int32(N),
                Int32(M),
                Int32(K),
                UnsafePointer(to=alpha).bitcast[NoneType](),
                _ffi_void_ptr(b.ptr),
                _rocblas.DataType(b_type),
                b_lead,
                _ffi_void_ptr(a.ptr),
                _rocblas.DataType(a_type),
                a_lead,
                UnsafePointer(to=beta).bitcast[NoneType](),
                _ffi_void_ptr(c.ptr),
                _rocblas.DataType(c_type),
                c_lead,
                _ffi_void_ptr(c.ptr),
                _rocblas.DataType(c_type),
                c_lead,
                compute_type,
                _rocblas.Algorithm.STANDARD,
                0,
                0,
            )
        )
    # Default column-major.
    _rocblas.check_error(
        rocblas_gemm_ex(
            handle,
            _convert_to_rocblas_transpose(transpose_a),
            _convert_to_rocblas_transpose(transpose_b),
            Int32(M),
            Int32(N),
            Int32(K),
            UnsafePointer(to=alpha).bitcast[NoneType](),
            _ffi_void_ptr(a.ptr),
            _rocblas.DataType(a_type),
            Int32(M),
            _ffi_void_ptr(b.ptr),
            _rocblas.DataType(b_type),
            Int32(N) if transpose_b else Int32(K),
            UnsafePointer(to=beta).bitcast[NoneType](),
            _ffi_void_ptr(c.ptr),
            _rocblas.DataType(c_type),
            Int32(M),
            _ffi_void_ptr(c.ptr),
            _rocblas.DataType(c_type),
            Int32(M),
            compute_type,
            _rocblas.Algorithm.STANDARD,
            0,
            0,
        )
    )


# ===----------------------------------------------------------------------===#
# CUBLASLT
# ===----------------------------------------------------------------------===#


def _cublasLt_matmul[
    d_type: DType,
    a_type: DType,
    b_type: DType,
    scales_type: DType = a_type,
    a_scales_layout: Layout = Layout.row_major(UNKNOWN_VALUE),
    b_scales_layout: Layout = Layout.row_major(UNKNOWN_VALUE),
](
    ctx: DeviceContext,
    handle: cublasHandle_t[_],
    d: NullableTileTensor[mut=True, d_type, ...],
    a: TileTensor[a_type, ...],
    b: TileTensor[b_type, ...],
    *,
    a_scales: OptionalReg[
        LayoutTensor[scales_type, a_scales_layout, ImmutAnyOrigin]
    ] = None,
    b_scales: OptionalReg[
        LayoutTensor[scales_type, b_scales_layout, ImmutAnyOrigin]
    ] = None,
    c_row_major: Bool = True,
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    alpha: Float32 = 1.0,
    beta: Float32 = 0.0,
) raises:
    var M = Int(d.dim[0]())
    var N = Int(d.dim[1]())
    var K = Int(a.dim[1]())

    comptime assert a_type in (
        DType.float8_e4m3fn,
        DType.float8_e5m2,
        DType.bfloat16,
        DType.float16,
        DType.uint8,
    ), (
        "Only E4M3, E5M2, bfloat16, float16, and E2M1x2 (UInt8) input data"
        " types are supported. Please extend it if you need more data"
        " types."
    )

    comptime assert a_type == b_type, "A and B must have the same type"

    comptime if a_type.is_float8():
        comptime assert not (a_type == b_type == .float8_e5m2), (
            "E5M2xE5m2 is not supported! Please refer to"
            " `https://docs.nvidia.com/cuda/cublas/#id105`"
        )

    if transpose_a or not transpose_b:
        raise Error(
            "the cuBLASLT backend currently only is implemented for"
            " transpose_a=False and transpose_b=True"
        )

    var cuda_stream = CUDA(ctx.stream())

    # CublasLt is by default column-major but we like to have the output in row-major
    # to compare with our results. Use `c_row_major` to determine the output layout.

    # To use FP8 kernels, the following set of requirements must be satisfied:
    # 1) All matrix dimensions must meet the optimal requirements listed in Tensor Core Usage (See Below)
    # 2) A must be transposed and B non-transposed (The “TN” format).
    # 3) The compute type must be CUBLAS_COMPUTE_32F.
    # 4) The scale type must be CUDA_R_32F.

    # A verity of A, B, and D data types are supported by this API. For more
    # information please refer to `https://docs.nvidia.com/cuda/cublas/#id105`

    # The best performance when using Tensor Cores can be achieved when the matrix dimensions and
    # pointers meet certain memory alignment requirements.
    # Specifically, all of the following conditions must be satisfied to get the most performance out of Tensor Cores:
    # 1) ((op_A == CUBLAS_OP_N ? m : k) * AtypeSize) % 16 == 0
    # 2) ((op_B == CUBLAS_OP_N ? k : n) * BtypeSize) % 16 == 0
    # 3) (m * CtypeSize) % 16 == 0
    # 4) (lda * AtypeSize) % 16 == 0
    # 5) (ldb * BtypeSize) % 16 == 0
    # 6) (ldc * CtypeSize) % 16 == 0
    # 7) intptr_t(A) % 16 == 0
    # 8) intptr_t(B) % 16 == 0
    # 9) intptr_t(C) % 16 == 0

    # TN format required for FP8
    var transa = cublasOperation_t.CUBLAS_OP_T
    var transb = cublasOperation_t.CUBLAS_OP_N

    # create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults;
    var compute_desc = cublasLtMatmulDesc_t()
    check_cublas_error(
        cublasLtMatmulDescCreate(
            UnsafePointer(to=compute_desc).as_unsafe_any_origin(),
            ComputeType.COMPUTE_32F,
            DataType.R_32F,
        ),
        msg="failed to create cublasLtMatmulDesc",
    )

    check_cublas_error(
        cublasLtMatmulDescSetAttribute(
            compute_desc,
            cublasLtMatmulDescAttributes_t.CUBLASLT_MATMUL_DESC_TRANSA,
            UnsafePointer(to=transa)
            .bitcast[NoneType]()
            .as_imm()
            .as_unsafe_any_origin(),
            size_of[cublasOperation_t](),
        ),
        msg="failed to set cublasLtMatmulDescAttribute for transa",
    )
    check_cublas_error(
        cublasLtMatmulDescSetAttribute(
            compute_desc,
            cublasLtMatmulDescAttributes_t.CUBLASLT_MATMUL_DESC_TRANSB,
            UnsafePointer(to=transb)
            .bitcast[NoneType]()
            .as_imm()
            .as_unsafe_any_origin(),
            size_of[cublasOperation_t](),
        ),
        msg="failed to set cublasLtMatmulDescAttribute for transb",
    )

    comptime if (
        _is_sm10x_gpu(ctx.default_device_info)
        or _is_sm12x_gpu(ctx.default_device_info)
    ):
        if a_scales or b_scales:
            if not (a_scales and b_scales):
                raise Error("a_scales and b_scales must be provided together")
            var a_scale_tensor = a_scales.value()
            var b_scale_tensor = b_scales.value()

            comptime SF_VECTOR_SIZE = NVFP4_SF_VECTOR_SIZE if scales_type == NVFP4_SF_DTYPE else MXFP8_SF_VECTOR_SIZE

            if scales_type not in (MXFP8_SF_DTYPE, NVFP4_SF_DTYPE):
                raise Error(
                    "Only float8_e8m0fnu(scaling type: MXFP8) and"
                    " float8_e4m3fn(scaling type: MXFP4) are supported for B200"
                )
            if not (
                a_type == b_type
                and (
                    (a_type == .float8_e4m3fn and scales_type == MXFP8_SF_DTYPE)
                    or (a_type == .uint8 and scales_type == NVFP4_SF_DTYPE)
                )
            ):
                raise Error(
                    "Only E4M3 input with MXFP8 scales or E2M1x2(i.e,"
                    " UINT8) input with NVFP4 scales are supported for block"
                    " scaled matmul"
                )

            # TODO (KERN-2238): uint8 is a proxy data type for two Float4-E2M1 values for now.
            # We need to double the K dimension as we are allocating for uint8 input data type.
            # Remove this when GENAI-337 is fixed.
            if a_type == .uint8 and scales_type == NVFP4_SF_DTYPE:
                K = K * 2

            if not (
                (a_type == .uint8 and K % 32 == 0)
                or (a_type == .float8_e4m3fn and K % 16 == 0)
            ):
                raise Error(
                    "Due to TMA 16B alignment requirement, K must be divisible"
                    " by 16/32 for MXFP8/NVFP4 input data type, respectively"
                )

            if comptime (
                a_scales_layout.rank() != 5 or b_scales_layout.rank() != 5
            ):
                raise Error(
                    "Invalid A/B scales dimensions. Expected 5D tensors."
                )

            if (
                a_scale_tensor.dim(0) != ceildiv(M, SF_MN_GROUP_SIZE)
                or a_scale_tensor.dim(1)
                != ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K)
                or b_scale_tensor.dim(0) != ceildiv(N, SF_MN_GROUP_SIZE)
                or b_scale_tensor.dim(1)
                != ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K)
                or a_scale_tensor.dim(2)
                != b_scale_tensor.dim(2)
                != SF_ATOM_M[0]
                or a_scale_tensor.dim(3)
                != b_scale_tensor.dim(3)
                != SF_ATOM_M[1]
                or a_scale_tensor.dim(4) != b_scale_tensor.dim(4) != SF_ATOM_K
            ):
                raise Error("Invalid A/B scales dimensions.")

            var a_scale_mode: cublasLtMatmulMatrixScale_t
            var b_scale_mode: cublasLtMatmulMatrixScale_t

            a_scale_mode = b_scale_mode = (
                cublasLtMatmulMatrixScale_t.MATRIX_SCALE_VEC16_UE4M3 if scales_type
                == NVFP4_SF_DTYPE else cublasLtMatmulMatrixScale_t.MATRIX_SCALE_VEC32_UE8M0
            )

            var a_scale_ptr = b_scale_tensor.ptr.bitcast[
                NoneType
            ]() if c_row_major else a_scale_tensor.ptr.bitcast[NoneType]()
            var b_scale_ptr = a_scale_tensor.ptr.bitcast[
                NoneType
            ]() if c_row_major else b_scale_tensor.ptr.bitcast[NoneType]()

            check_cublas_error(
                cublasLtMatmulDescSetAttribute(
                    compute_desc,
                    cublasLtMatmulDescAttributes_t.CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                    UnsafePointer(to=a_scale_mode)
                    .bitcast[NoneType]()
                    .as_imm()
                    .as_unsafe_any_origin(),
                    size_of[Int32](),
                ),
                msg=(
                    "failed to set cublasLtMatmulDescAttribute for Matrix A"
                    " scale mode"
                ),
            )
            check_cublas_error(
                cublasLtMatmulDescSetAttribute(
                    compute_desc,
                    cublasLtMatmulDescAttributes_t.CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                    UnsafePointer(to=b_scale_mode)
                    .bitcast[NoneType]()
                    .as_imm()
                    .as_unsafe_any_origin(),
                    size_of[Int32](),
                ),
                msg=(
                    "failed to set cublasLtMatmulDescAttribute for Matrix B"
                    " scale mode"
                ),
            )

            check_cublas_error(
                cublasLtMatmulDescSetAttribute(
                    compute_desc,
                    cublasLtMatmulDescAttributes_t.CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                    UnsafePointer(to=a_scale_ptr)
                    .bitcast[NoneType]()
                    .as_imm()
                    .as_unsafe_any_origin(),
                    size_of[OpaquePointer[UntrackedOrigin[mut=True]]](),
                ),
                msg=(
                    "failed to set cublasLtMatmulDescAttribute for Matrix A"
                    " scale factor"
                ),
            )
            check_cublas_error(
                cublasLtMatmulDescSetAttribute(
                    compute_desc,
                    cublasLtMatmulDescAttributes_t.CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                    UnsafePointer(to=b_scale_ptr)
                    .bitcast[NoneType]()
                    .as_imm()
                    .as_unsafe_any_origin(),
                    size_of[OpaquePointer[UntrackedOrigin[mut=True]]](),
                ),
                msg=(
                    "failed to set cublasLtMatmulDescAttribute for Matrix B"
                    " scale factor"
                ),
            )
    else:
        if a_scales or b_scales:
            raise Error("block scaling is only supported on B200 devices")

    # create matrix descriptors, we are good with the details here so no need to set any extra attributes
    # table of supported type combinations can be found in the documentation: https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul
    var _adesc = cublasLtMatrixLayout_t()
    check_cublas_error(
        cublasLtMatrixLayoutCreate(
            UnsafePointer(to=_adesc).as_unsafe_any_origin(),
            _convert_to_cublas_datatype[a_type](),
            UInt64(K),
            UInt64(N) if c_row_major else UInt64(M),
            Int64(K),
        ),
        msg="failed to create cublasLtMatrixLayout for adesc",
    )

    var _bdesc = cublasLtMatrixLayout_t()
    check_cublas_error(
        cublasLtMatrixLayoutCreate(
            UnsafePointer(to=_bdesc).as_unsafe_any_origin(),
            _convert_to_cublas_datatype[b_type](),
            UInt64(K),
            UInt64(M) if c_row_major else UInt64(N),
            Int64(K),
        ),
        msg="failed to create cublasLtMatrixLayout for bdesc",
    )

    var _ddesc = cublasLtMatrixLayout_t()
    check_cublas_error(
        cublasLtMatrixLayoutCreate(
            UnsafePointer(to=_ddesc).as_unsafe_any_origin(),
            _convert_to_cublas_datatype[d_type](),
            UInt64(N) if c_row_major else UInt64(M),
            UInt64(M) if c_row_major else UInt64(N),
            Int64(N) if c_row_major else Int64(M),
        ),
        msg="failed to create cublasLtMatrixLayout for ddesc",
    )

    var _cdesc = cublasLtMatrixLayout_t()
    check_cublas_error(
        cublasLtMatrixLayoutCreate(
            UnsafePointer(to=_cdesc).as_unsafe_any_origin(),
            _convert_to_cublas_datatype[d_type](),
            UInt64(N) if c_row_major else UInt64(M),
            UInt64(M) if c_row_major else UInt64(N),
            Int64(N) if c_row_major else Int64(M),
        ),
        msg="failed to create cublasLtMatrixLayout for cdesc",
    )

    var preference = cublasLtMatmulPreference_t()
    check_cublas_error(
        cublasLtMatmulPreferenceCreate(
            UnsafePointer(to=preference).as_unsafe_any_origin()
        ),
        msg="failed to create cublasLtMatmulPreference",
    )

    var workspace_size = 32 * 1024 * 1024
    check_cublas_error(
        cublasLtMatmulPreferenceSetAttribute(
            preference,
            Preference.MAX_WORKSPACE_BYTES,
            UnsafePointer(to=workspace_size)
            .bitcast[NoneType]()
            .as_imm()
            .as_unsafe_any_origin(),
            size_of[Int64](),
        ),
        msg=(
            "failed to set cublasLtMatmulPreferenceAttribute for"
            " max_workspace_bytes"
        ),
    )

    var heuristic_result = cublasLtMatmulHeuristicResult_t()
    var algorithm_count = 0
    check_cublas_error(
        cublasLtMatmulAlgoGetHeuristic(
            handle,
            compute_desc,
            _adesc,
            _bdesc,
            _cdesc,
            _ddesc,
            preference,
            1,
            UnsafePointer(to=heuristic_result).as_unsafe_any_origin(),
            UnsafePointer(to=algorithm_count).as_unsafe_any_origin(),
        ),
        msg="failed to get cublasLtMatmulAlgoGetHeuristic",
    )

    if algorithm_count == 0:
        raise Error("No algorithm was found!")

    var matmul_workspace = ctx.enqueue_create_buffer[.uint8](workspace_size)

    var d_void = _ffi_void_ptr(d.ptr)
    if c_row_major:
        check_cublas_error(
            cublasLtMatmul(
                handle,  # light_handle
                compute_desc,  # compute_desc
                UnsafePointer(to=alpha)
                .bitcast[NoneType]()
                .as_imm()
                .as_unsafe_any_origin(),
                _ffi_void_ptr(b.ptr),
                _adesc,  # _adesc
                _ffi_void_ptr(a.ptr).as_imm(),  # _b
                _bdesc,  # _bdesc
                UnsafePointer(to=beta)
                .bitcast[NoneType]()
                .as_imm()
                .as_unsafe_any_origin(),  # beta
                None,  # _c
                _cdesc,  # _cdesc
                d_void,  # _d
                _ddesc,  # _ddesc
                UnsafePointer(to=heuristic_result.algo)
                .as_imm()
                .as_unsafe_any_origin(),  # algo
                matmul_workspace.unsafe_ptr()
                .bitcast[NoneType]()
                .as_unsafe_any_origin(),  # workspace
                workspace_size,  # workspace_size_in_bytes
                cuda_stream.value()[],  # stream
            ),
            msg="failed to cublasLtMatmul for c_row_major=True",
        )
    else:
        check_cublas_error(
            cublasLtMatmul(
                handle,  # light_handle
                compute_desc,  # compute_desc
                UnsafePointer(to=alpha)
                .bitcast[NoneType]()
                .as_imm()
                .as_unsafe_any_origin(),  # alpha
                _ffi_void_ptr(a.ptr),  # _a
                _adesc,  # _adesc
                _ffi_void_ptr(b.ptr).as_imm(),  # _b
                _bdesc,  # _bdesc
                UnsafePointer(to=beta)
                .bitcast[NoneType]()
                .as_imm()
                .as_unsafe_any_origin(),  # beta
                None,  # _c
                _cdesc,  # _cdesc
                d_void,  # _d
                _ddesc,  # _ddesc
                UnsafePointer(to=heuristic_result.algo)
                .as_imm()
                .as_unsafe_any_origin(),  # algo
                matmul_workspace.unsafe_ptr()
                .bitcast[NoneType]()
                .as_unsafe_any_origin(),  # workspace
                workspace_size,  # workspace_size_in_bytes
                cuda_stream.value()[],  # stream
            ),
            msg="failed to cublasLtMatmul for c_row_major=False",
        )

    check_cublas_error(
        cublasLtMatmulDescDestroy(compute_desc),
        msg="failed to destroy cublasLtMatmulDesc",
    )
    check_cublas_error(
        cublasLtMatrixLayoutDestroy(_adesc),
        msg="failed to destroy cublasLtMatrixLayout for adesc",
    )
    check_cublas_error(
        cublasLtMatrixLayoutDestroy(_bdesc),
        msg="failed to destroy cublasLtMatrixLayout for bdesc",
    )
    check_cublas_error(
        cublasLtMatrixLayoutDestroy(_cdesc),
        msg="failed to destroy cublasLtMatrixLayout for cdesc",
    )
    check_cublas_error(
        cublasLtMatrixLayoutDestroy(_ddesc),
        msg="failed to destroy cublasLtMatrixLayout for ddesc",
    )
    check_cublas_error(
        cublasLtMatmulPreferenceDestroy(preference),
        msg="failed to destroy cublasLtMatmulPreference",
    )

    _ = matmul_workspace^


# ===----------------------------------------------------------------------===#
# HIPBLASLT
# ===----------------------------------------------------------------------===#


def _hipblasLt_matmul[
    d_type: DType,
    a_type: DType,
    b_type: DType,
    scales_type: DType = a_type,
    a_scales_layout: Layout = Layout.row_major(UNKNOWN_VALUE),
    b_scales_layout: Layout = Layout.row_major(UNKNOWN_VALUE),
](
    ctx: DeviceContext,
    handle: hipblasLtHandle_t,
    d: NullableTileTensor[mut=True, d_type, ...],
    a: TileTensor[a_type, ...],
    b: TileTensor[b_type, ...],
    *,
    a_scales: OptionalReg[
        LayoutTensor[scales_type, a_scales_layout, ImmutAnyOrigin]
    ] = None,
    b_scales: OptionalReg[
        LayoutTensor[scales_type, b_scales_layout, ImmutAnyOrigin]
    ] = None,
    c_row_major: Bool = True,
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    alpha: Float32 = 1.0,
    beta: Float32 = 0.0,
    batch_size: Int = 1,
) raises:
    comptime assert a_type in (
        DType.float32,
        DType.float16,
        DType.bfloat16,
        DType.float8_e4m3fn,
        DType.float8_e5m2,
        DType.float8_e4m3fnuz,
        DType.float8_e5m2fnuz,
        DType.uint8,
    ), "Unsupported data type. Please extend it if you need more data types."

    comptime assert a_type == b_type, "A and B must have the same type"

    @always_inline
    def create_hipblas_matrix_layout[
        buf_type: DType,
    ](rows: Int, cols: Int) raises -> hipblasLtMatrixLayout_t:
        var _desc = hipblasLtMatrixLayout_t()
        _check_hipblas_error(
            hipblasLtMatrixLayoutCreate(
                UnsafePointer(to=_desc),
                _convert_to_hip_datatype[buf_type](),
                UInt64(cols),
                UInt64(rows),
                Int64(cols),
            )
        )
        return _desc

    @always_inline
    def set_matrix_layout_batch_size(
        mat_layout: hipblasLtMatrixLayout_t,
        batch_size: Int,
        batch_stride: Int64,
    ) raises:
        _check_hipblas_error(
            hipblasLtMatrixLayoutSetAttribute(
                mat_layout,
                hipblasLtMatmulLayoutAttribute_t.BATCH_COUNT,
                UnsafePointer(to=batch_size).bitcast[NoneType](),
                size_of[Int](),
            )
        )
        _check_hipblas_error(
            hipblasLtMatrixLayoutSetAttribute(
                mat_layout,
                hipblasLtMatmulLayoutAttribute_t.STRIDED_BATCH_OFFSET,
                UnsafePointer(to=batch_stride).bitcast[NoneType](),
                size_of[Int64](),
            )
        )

    var a_rows = Int(a.dim[0]())
    var a_cols = Int(a.dim[1]())
    var b_rows = Int(b.dim[0]())
    var b_cols = Int(b.dim[1]())
    var d_rows = Int(d.dim[0]())
    var d_cols = Int(d.dim[1]())

    var operationDesc = hipblasLtMatmulDesc_t()
    _check_hipblas_error(
        hipblasLtMatmulDescCreate(
            UnsafePointer(to=operationDesc),
            hipblasComputeType_t.COMPUTE_32F,
            hipDataType_t.R_32F,
        )
    )

    if a_scales or b_scales:
        if not (a_scales and b_scales):
            raise Error("a_scales and b_scales must be provided together")
        var a_scale_tensor = a_scales.value()
        var b_scale_tensor = b_scales.value()

        if comptime (scales_type != MXFP8_SF_DTYPE):
            raise Error("Only float8_e8m0fnu(scale type: MXFP8) supported")
        if comptime (a_type != .uint8):
            raise Error("Only float4_e2m1fnx2 input type supported")
        if not c_row_major or transpose_a or not transpose_b:
            raise Error("Unexpected transpose flags")

        # Convert to the logical number of columns (K) for FP4.
        a_cols *= 2
        b_cols *= 2

        if comptime (
            a_scales_layout.rank() != 2 or b_scales_layout.rank() != 2
        ):
            raise Error("Invalid A/B scales dimensions. Expected 2D tensors.")

        if (
            a_scale_tensor.dim(0) != a_rows
            or a_scale_tensor.dim(1) != ceildiv(a_cols, MXFP4_SF_VECTOR_SIZE)
            or b_scale_tensor.dim(0) != b_rows
            or b_scale_tensor.dim(1) != ceildiv(b_cols, MXFP4_SF_VECTOR_SIZE)
        ):
            raise Error("Invalid A/B scales dimensions.")

        var scale_mode = hipblasLtMatmulMatrixScale_t.VEC32_UE8M0

        _check_hipblas_error(
            hipblasLtMatmulDescSetAttribute(
                operationDesc,
                hipblasLtMatmulDescAttributes_t.A_SCALE_MODE,
                UnsafePointer(to=scale_mode).bitcast[NoneType](),
                size_of[hipblasLtMatmulMatrixScale_t](),
            )
        )
        _check_hipblas_error(
            hipblasLtMatmulDescSetAttribute(
                operationDesc,
                hipblasLtMatmulDescAttributes_t.B_SCALE_MODE,
                UnsafePointer(to=scale_mode).bitcast[NoneType](),
                size_of[hipblasLtMatmulMatrixScale_t](),
            )
        )

        var a_scale_ptr = a_scale_tensor.ptr.bitcast[NoneType]()
        var b_scale_ptr = b_scale_tensor.ptr.bitcast[NoneType]()

        if c_row_major:
            swap(a_scale_ptr, b_scale_ptr)

        _check_hipblas_error(
            hipblasLtMatmulDescSetAttribute(
                operationDesc,
                hipblasLtMatmulDescAttributes_t.A_SCALE_POINTER,
                UnsafePointer(to=a_scale_ptr).bitcast[NoneType](),
                size_of[OpaquePointer[UntrackedOrigin[mut=True]]](),
            )
        )
        _check_hipblas_error(
            hipblasLtMatmulDescSetAttribute(
                operationDesc,
                hipblasLtMatmulDescAttributes_t.B_SCALE_POINTER,
                UnsafePointer(to=b_scale_ptr).bitcast[NoneType](),
                size_of[OpaquePointer[UntrackedOrigin[mut=True]]](),
            )
        )

    var _adesc = create_hipblas_matrix_layout[a_type](a_rows, a_cols)
    var _bdesc = create_hipblas_matrix_layout[b_type](b_rows, b_cols)
    var _ddesc = create_hipblas_matrix_layout[d_type](d_rows, d_cols)

    # set batch size accordingly
    if batch_size > 1:
        set_matrix_layout_batch_size(_adesc, batch_size, Int64(a_rows * a_cols))
        set_matrix_layout_batch_size(_bdesc, batch_size, Int64(b_rows * b_cols))
        set_matrix_layout_batch_size(_ddesc, batch_size, Int64(d_rows * d_cols))

    var transa = (
        hipblasOperation_t.OP_T if transpose_a else hipblasOperation_t.OP_N
    )
    var transb = (
        hipblasOperation_t.OP_T if transpose_b else hipblasOperation_t.OP_N
    )

    # hipblasLt is by default column-major but we like to have the output in row-major
    # to compare with our results. Use `c_row_major` to determine the output layout.
    if c_row_major:
        swap(_adesc, _bdesc)
        swap(transa, transb)

    _check_hipblas_error(
        hipblasLtMatmulDescSetAttribute(
            operationDesc,
            hipblasLtMatmulDescAttributes_t.TRANSA,
            UnsafePointer(to=transa).bitcast[NoneType](),
            size_of[hipblasOperation_t](),
        )
    )
    _check_hipblas_error(
        hipblasLtMatmulDescSetAttribute(
            operationDesc,
            hipblasLtMatmulDescAttributes_t.TRANSB,
            UnsafePointer(to=transb).bitcast[NoneType](),
            size_of[hipblasOperation_t](),
        )
    )

    var preference = hipblasLtMatmulPreference_t()
    _check_hipblas_error(
        hipblasLtMatmulPreferenceCreate(UnsafePointer(to=preference))
    )

    var heuristicResult = hipblasLtMatmulHeuristicResult_t()
    var returnedResults = 0
    _check_hipblas_error(
        hipblasLtMatmulAlgoGetHeuristic(
            handle,
            operationDesc,
            _adesc,
            _bdesc,
            _ddesc,
            _ddesc,
            preference,
            1,
            UnsafePointer(to=heuristicResult),
            UnsafePointer(to=returnedResults),
        )
    )

    if returnedResults == 0:
        raise Error("No algorithm was found!")

    var workspace_size = heuristicResult.workspaceSize
    var workspace = ctx.enqueue_create_buffer[.uint8](workspace_size)

    var d_void = _ffi_void_ptr(d.ptr)
    if c_row_major:
        _check_hipblas_error(
            hipblasLtMatmul(
                handle,
                operationDesc,
                UnsafePointer(to=alpha).bitcast[NoneType](),
                _ffi_void_ptr(b.ptr),
                _adesc,
                _ffi_void_ptr(a.ptr),
                _bdesc,
                UnsafePointer(to=beta).bitcast[NoneType](),
                d_void,
                _ddesc,
                d_void,
                _ddesc,
                UnsafePointer(to=heuristicResult.algo),
                workspace.unsafe_ptr().bitcast[NoneType](),
                workspace_size,
                HIP(ctx.stream()),
            )
        )
    else:
        _check_hipblas_error(
            hipblasLtMatmul(
                handle,
                operationDesc,
                UnsafePointer(to=alpha).bitcast[NoneType](),
                _ffi_void_ptr(a.ptr),
                _adesc,
                _ffi_void_ptr(b.ptr),
                _bdesc,
                UnsafePointer(to=beta).bitcast[NoneType](),
                d_void,
                _ddesc,
                d_void,
                _ddesc,
                UnsafePointer(to=heuristicResult.algo),
                workspace.unsafe_ptr().bitcast[NoneType](),
                workspace_size,
                HIP(ctx.stream()),
            )
        )

    _check_hipblas_error(hipblasLtMatmulPreferenceDestroy(preference))
    _check_hipblas_error(hipblasLtMatmulDescDestroy(operationDesc))
    _check_hipblas_error(hipblasLtMatrixLayoutDestroy(_adesc))
    _check_hipblas_error(hipblasLtMatrixLayoutDestroy(_bdesc))
    _check_hipblas_error(hipblasLtMatrixLayoutDestroy(_ddesc))
