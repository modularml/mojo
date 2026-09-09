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


import std.ffi as ffi

from std.complex import ComplexFloat32, ComplexFloat64
from max.gpu.host._nvidia_cuda import CUstream

from .types import LibraryProperty, Property, Status, Type
from .utils import _get_dylib_function

comptime cufftHandle = ffi.c_uint


def cufftCreate(handle: UnsafePointer[cufftHandle, _]) raises -> Status:
    return _get_dylib_function[
        "cufftCreate", def(type_of(handle)) thin -> Status
    ]()(handle)


def cufftGetVersion(version: UnsafePointer[ffi.c_int, _]) raises -> Status:
    return _get_dylib_function[
        "cufftGetVersion", def(type_of(version)) thin -> Status
    ]()(version)


def cufftExecZ2Z(
    plan: cufftHandle,
    idata: UnsafePointer[ComplexFloat64, _],
    odata: UnsafePointer[ComplexFloat64, _],
    direction: ffi.c_int,
) raises -> Status:
    return _get_dylib_function[
        "cufftExecZ2Z",
        def(
            cufftHandle,
            type_of(idata),
            type_of(odata),
            ffi.c_int,
        ) thin -> Status,
    ]()(plan, idata, odata, direction)


def cufftExecC2C(
    plan: cufftHandle,
    idata: UnsafePointer[ComplexFloat32, _],
    odata: UnsafePointer[ComplexFloat32, _],
    direction: ffi.c_int,
) raises -> Status:
    return _get_dylib_function[
        "cufftExecC2C",
        def(
            cufftHandle,
            type_of(idata),
            type_of(odata),
            ffi.c_int,
        ) thin -> Status,
    ]()(plan, idata, odata, direction)


def cufftExecR2C(
    plan: cufftHandle,
    idata: UnsafePointer[ffi.c_float, _],
    odata: UnsafePointer[ComplexFloat32, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftExecR2C",
        def(
            cufftHandle,
            type_of(idata),
            type_of(odata),
        ) thin -> Status,
    ]()(plan, idata, odata)


def cufftSetWorkArea(
    plan: cufftHandle, work_area: OpaquePointer[_]
) raises -> Status:
    return _get_dylib_function[
        "cufftSetWorkArea", def(cufftHandle, type_of(work_area)) thin -> Status
    ]()(plan, work_area)


def cufftPlan1d(
    plan: UnsafePointer[cufftHandle, _],
    nx: ffi.c_int,
    type: Type,
    batch: ffi.c_int,
) raises -> Status:
    return _get_dylib_function[
        "cufftPlan1d",
        def(type_of(plan), ffi.c_int, Type, ffi.c_int) thin -> Status,
    ]()(plan, nx, type, batch)


def cufftMakePlan2d(
    plan: cufftHandle,
    nx: ffi.c_int,
    ny: ffi.c_int,
    type: Type,
    work_size: UnsafePointer[Int, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftMakePlan2d",
        def(
            cufftHandle, ffi.c_int, ffi.c_int, Type, type_of(work_size)
        ) thin -> Status,
    ]()(plan, nx, ny, type, work_size)


def cufftSetPlanPropertyInt64(
    plan: cufftHandle, property: Property, input_value_int: ffi.c_long_long
) raises -> Status:
    return _get_dylib_function[
        "cufftSetPlanPropertyInt64",
        def(cufftHandle, Property, ffi.c_long_long) thin -> Status,
    ]()(plan, property, input_value_int)


def cufftPlan2d(
    plan: UnsafePointer[cufftHandle, _],
    nx: ffi.c_int,
    ny: ffi.c_int,
    type: Type,
) raises -> Status:
    return _get_dylib_function[
        "cufftPlan2d",
        def(type_of(plan), ffi.c_int, ffi.c_int, Type) thin -> Status,
    ]()(plan, nx, ny, type)


def cufftMakePlan1d(
    plan: cufftHandle,
    nx: ffi.c_int,
    type: Type,
    batch: ffi.c_int,
    work_size: UnsafePointer[Int, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftMakePlan1d",
        def(
            cufftHandle, ffi.c_int, Type, ffi.c_int, type_of(work_size)
        ) thin -> Status,
    ]()(plan, nx, type, batch, work_size)


def cufftExecC2R(
    plan: cufftHandle,
    idata: UnsafePointer[ComplexFloat32, _],
    odata: UnsafePointer[ffi.c_float, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftExecC2R",
        def(
            cufftHandle,
            type_of(idata),
            type_of(odata),
        ) thin -> Status,
    ]()(plan, idata, odata)


def cufftMakePlanMany(
    plan: cufftHandle,
    rank: ffi.c_int,
    n: UnsafePointer[ffi.c_int, _],
    inembed: UnsafePointer[ffi.c_int, _],
    istride: ffi.c_int,
    idist: ffi.c_int,
    onembed: UnsafePointer[ffi.c_int, _],
    ostride: ffi.c_int,
    odist: ffi.c_int,
    type: Type,
    batch: ffi.c_int,
    work_size: UnsafePointer[Int, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftMakePlanMany",
        def(
            cufftHandle,
            ffi.c_int,
            type_of(n),
            type_of(inembed),
            ffi.c_int,
            ffi.c_int,
            type_of(onembed),
            ffi.c_int,
            ffi.c_int,
            Type,
            ffi.c_int,
            type_of(work_size),
        ) thin -> Status,
    ]()(
        plan,
        rank,
        n,
        inembed,
        istride,
        idist,
        onembed,
        ostride,
        odist,
        type,
        batch,
        work_size,
    )


def cufftSetAutoAllocation(
    plan: cufftHandle, auto_allocate: ffi.c_int
) raises -> Status:
    return _get_dylib_function[
        "cufftSetAutoAllocation", def(cufftHandle, ffi.c_int) thin -> Status
    ]()(plan, auto_allocate)


def cufftEstimate1d(
    nx: ffi.c_int,
    type: Type,
    batch: ffi.c_int,
    work_size: UnsafePointer[Int, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftEstimate1d",
        def(ffi.c_int, Type, ffi.c_int, type_of(work_size)) thin -> Status,
    ]()(nx, type, batch, work_size)


def cufftGetSize(
    handle: cufftHandle, work_size: UnsafePointer[Int, _]
) raises -> Status:
    return _get_dylib_function[
        "cufftGetSize", def(cufftHandle, type_of(work_size)) thin -> Status
    ]()(handle, work_size)


def cufftExecZ2D(
    plan: cufftHandle,
    idata: UnsafePointer[ComplexFloat64, _],
    odata: UnsafePointer[ffi.c_double, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftExecZ2D",
        def(
            cufftHandle,
            type_of(idata),
            type_of(odata),
        ) thin -> Status,
    ]()(plan, idata, odata)


def cufftEstimate2d(
    nx: ffi.c_int, ny: ffi.c_int, type: Type, work_size: UnsafePointer[Int, _]
) raises -> Status:
    return _get_dylib_function[
        "cufftEstimate2d",
        def(ffi.c_int, ffi.c_int, Type, type_of(work_size)) thin -> Status,
    ]()(nx, ny, type, work_size)


def cufftSetStream(plan: cufftHandle, stream: CUstream) raises -> Status:
    return _get_dylib_function[
        "cufftSetStream", def(cufftHandle, CUstream) thin -> Status
    ]()(plan, stream)


def cufftMakePlanMany64(
    plan: cufftHandle,
    rank: ffi.c_int,
    n: UnsafePointer[ffi.c_long_long, _],
    inembed: UnsafePointer[ffi.c_long_long, _],
    istride: ffi.c_long_long,
    idist: ffi.c_long_long,
    onembed: UnsafePointer[ffi.c_long_long, _],
    ostride: ffi.c_long_long,
    odist: ffi.c_long_long,
    type: Type,
    batch: ffi.c_long_long,
    work_size: UnsafePointer[Int, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftMakePlanMany64",
        def(
            cufftHandle,
            ffi.c_int,
            type_of(n),
            type_of(inembed),
            ffi.c_long_long,
            ffi.c_long_long,
            type_of(onembed),
            ffi.c_long_long,
            ffi.c_long_long,
            Type,
            ffi.c_long_long,
            type_of(work_size),
        ) thin -> Status,
    ]()(
        plan,
        rank,
        n,
        inembed,
        istride,
        idist,
        onembed,
        ostride,
        odist,
        type,
        batch,
        work_size,
    )


def cufftGetSize1d(
    handle: cufftHandle,
    nx: ffi.c_int,
    type: Type,
    batch: ffi.c_int,
    work_size: UnsafePointer[Int, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftGetSize1d",
        def(
            cufftHandle, ffi.c_int, Type, ffi.c_int, type_of(work_size)
        ) thin -> Status,
    ]()(handle, nx, type, batch, work_size)


def cufftMakePlan3d(
    plan: cufftHandle,
    nx: ffi.c_int,
    ny: ffi.c_int,
    nz: ffi.c_int,
    type: Type,
    work_size: UnsafePointer[Int, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftMakePlan3d",
        def(
            cufftHandle,
            ffi.c_int,
            ffi.c_int,
            ffi.c_int,
            Type,
            type_of(work_size),
        ) thin -> Status,
    ]()(plan, nx, ny, nz, type, work_size)


def cufftGetSizeMany(
    handle: cufftHandle,
    rank: ffi.c_int,
    n: UnsafePointer[ffi.c_int, _],
    inembed: UnsafePointer[ffi.c_int, _],
    istride: ffi.c_int,
    idist: ffi.c_int,
    onembed: UnsafePointer[ffi.c_int, _],
    ostride: ffi.c_int,
    odist: ffi.c_int,
    type: Type,
    batch: ffi.c_int,
    work_area: UnsafePointer[Int, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftGetSizeMany",
        def(
            cufftHandle,
            ffi.c_int,
            type_of(n),
            type_of(inembed),
            ffi.c_int,
            ffi.c_int,
            type_of(onembed),
            ffi.c_int,
            ffi.c_int,
            Type,
            ffi.c_int,
            type_of(work_area),
        ) thin -> Status,
    ]()(
        handle,
        rank,
        n,
        inembed,
        istride,
        idist,
        onembed,
        ostride,
        odist,
        type,
        batch,
        work_area,
    )


def cufftPlan3d(
    plan: UnsafePointer[cufftHandle, _],
    nx: ffi.c_int,
    ny: ffi.c_int,
    nz: ffi.c_int,
    type: Type,
) raises -> Status:
    return _get_dylib_function[
        "cufftPlan3d",
        def(
            type_of(plan), ffi.c_int, ffi.c_int, ffi.c_int, Type
        ) thin -> Status,
    ]()(plan, nx, ny, nz, type)


def cufftPlanMany(
    plan: UnsafePointer[cufftHandle, _],
    rank: ffi.c_int,
    n: UnsafePointer[ffi.c_int, _],
    inembed: UnsafePointer[ffi.c_int, _],
    istride: ffi.c_int,
    idist: ffi.c_int,
    onembed: UnsafePointer[ffi.c_int, _],
    ostride: ffi.c_int,
    odist: ffi.c_int,
    type: Type,
    batch: ffi.c_int,
) raises -> Status:
    return _get_dylib_function[
        "cufftPlanMany",
        def(
            type_of(plan),
            ffi.c_int,
            type_of(n),
            type_of(inembed),
            ffi.c_int,
            ffi.c_int,
            type_of(onembed),
            ffi.c_int,
            ffi.c_int,
            Type,
            ffi.c_int,
        ) thin -> Status,
    ]()(
        plan,
        rank,
        n,
        inembed,
        istride,
        idist,
        onembed,
        ostride,
        odist,
        type,
        batch,
    )


def cufftResetPlanProperty(
    plan: cufftHandle, property: Property
) raises -> Status:
    return _get_dylib_function[
        "cufftResetPlanProperty", def(cufftHandle, Property) thin -> Status
    ]()(plan, property)


def cufftGetSize3d(
    handle: cufftHandle,
    nx: ffi.c_int,
    ny: ffi.c_int,
    nz: ffi.c_int,
    type: Type,
    work_size: UnsafePointer[Int, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftGetSize3d",
        def(
            cufftHandle,
            ffi.c_int,
            ffi.c_int,
            ffi.c_int,
            Type,
            type_of(work_size),
        ) thin -> Status,
    ]()(handle, nx, ny, nz, type, work_size)


def cufftGetProperty(
    type: LibraryProperty, value: UnsafePointer[ffi.c_int, _]
) raises -> Status:
    return _get_dylib_function[
        "cufftGetProperty",
        def(LibraryProperty, type_of(value)) thin -> Status,
    ]()(type, value)


def cufftGetSizeMany64(
    plan: cufftHandle,
    rank: ffi.c_int,
    n: UnsafePointer[ffi.c_long_long, _],
    inembed: UnsafePointer[ffi.c_long_long, _],
    istride: ffi.c_long_long,
    idist: ffi.c_long_long,
    onembed: UnsafePointer[ffi.c_long_long, _],
    ostride: ffi.c_long_long,
    odist: ffi.c_long_long,
    type: Type,
    batch: ffi.c_long_long,
    work_size: UnsafePointer[Int, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftGetSizeMany64",
        def(
            cufftHandle,
            ffi.c_int,
            type_of(n),
            type_of(inembed),
            ffi.c_long_long,
            ffi.c_long_long,
            type_of(onembed),
            ffi.c_long_long,
            ffi.c_long_long,
            Type,
            ffi.c_long_long,
            type_of(work_size),
        ) thin -> Status,
    ]()(
        plan,
        rank,
        n,
        inembed,
        istride,
        idist,
        onembed,
        ostride,
        odist,
        type,
        batch,
        work_size,
    )


def cufftDestroy(plan: cufftHandle) raises -> Status:
    return _get_dylib_function[
        "cufftDestroy", def(cufftHandle) thin -> Status
    ]()(plan)


def cufftEstimateMany(
    rank: ffi.c_int,
    n: UnsafePointer[ffi.c_int, _],
    inembed: UnsafePointer[ffi.c_int, _],
    istride: ffi.c_int,
    idist: ffi.c_int,
    onembed: UnsafePointer[ffi.c_int, _],
    ostride: ffi.c_int,
    odist: ffi.c_int,
    type: Type,
    batch: ffi.c_int,
    work_size: UnsafePointer[Int, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftEstimateMany",
        def(
            ffi.c_int,
            type_of(n),
            type_of(inembed),
            ffi.c_int,
            ffi.c_int,
            type_of(onembed),
            ffi.c_int,
            ffi.c_int,
            Type,
            ffi.c_int,
            type_of(work_size),
        ) thin -> Status,
    ]()(
        rank,
        n,
        inembed,
        istride,
        idist,
        onembed,
        ostride,
        odist,
        type,
        batch,
        work_size,
    )


def cufftExecD2Z(
    plan: cufftHandle,
    idata: UnsafePointer[ffi.c_double, _],
    odata: UnsafePointer[ComplexFloat64, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftExecD2Z",
        def(
            cufftHandle,
            type_of(idata),
            type_of(odata),
        ) thin -> Status,
    ]()(plan, idata, odata)


def cufftEstimate3d(
    nx: ffi.c_int,
    ny: ffi.c_int,
    nz: ffi.c_int,
    type: Type,
    work_size: UnsafePointer[Int, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftEstimate3d",
        def(
            ffi.c_int, ffi.c_int, ffi.c_int, Type, type_of(work_size)
        ) thin -> Status,
    ]()(nx, ny, nz, type, work_size)


def cufftGetSize2d(
    handle: cufftHandle,
    nx: ffi.c_int,
    ny: ffi.c_int,
    type: Type,
    work_size: UnsafePointer[Int, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftGetSize2d",
        def(
            cufftHandle, ffi.c_int, ffi.c_int, Type, type_of(work_size)
        ) thin -> Status,
    ]()(handle, nx, ny, type, work_size)


def cufftGetPlanPropertyInt64(
    plan: cufftHandle,
    property: Property,
    return_ptr_value: UnsafePointer[ffi.c_long_long, _],
) raises -> Status:
    return _get_dylib_function[
        "cufftGetPlanPropertyInt64",
        def(cufftHandle, Property, type_of(return_ptr_value)) thin -> Status,
    ]()(plan, property, return_ptr_value)
