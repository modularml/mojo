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

from max.gpu.host._amdgpu_hip import hipStream_t

from .types import (
    Handle,
    TensorDescriptor,
    ConvolutionDescriptor,
    Status,
    DataType,
    TensorLayout,
    ConvolutionMode,
    ConvFwdAlgorithm,
    Problem,
    Solution,
    FindOptions,
    ProblemDirection,
    TensorArgumentId,
    ConvAlgoPerf,
)
from .utils import _get_dylib_function

# ===-----------------------------------------------------------------------===#
# Handle Management
# ===-----------------------------------------------------------------------===#


def miopenCreate(handle: OpaquePointer) raises -> Status:
    return _get_dylib_function[
        "miopenCreate",
        def(type_of(handle)) thin -> Status,
    ]()(handle)


def miopenDestroy(handle: Handle) raises -> Status:
    return _get_dylib_function["miopenDestroy", def(Handle) thin -> Status]()(
        handle
    )


def miopenSetStream(handle: Handle, stream: hipStream_t) raises -> Status:
    return _get_dylib_function[
        "miopenSetStream",
        def(Handle, hipStream_t) thin -> Status,
    ]()(handle, stream)


# ===-----------------------------------------------------------------------===#
# Tensor Descriptor Management
# ===-----------------------------------------------------------------------===#


def miopenCreateTensorDescriptor(
    desc: OpaquePointer,
) raises -> Status:
    return _get_dylib_function[
        "miopenCreateTensorDescriptor",
        def(type_of(desc)) thin -> Status,
    ]()(desc)


def miopenDestroyTensorDescriptor(
    desc: TensorDescriptor,
) raises -> Status:
    return _get_dylib_function[
        "miopenDestroyTensorDescriptor",
        def(TensorDescriptor) thin -> Status,
    ]()(desc)


def miopenSetTensorDescriptor(
    desc: TensorDescriptor,
    dtype: DataType,
    num_dims: Int32,
    dims: UnsafePointer[Int32, ImmutAnyOrigin],
    strides: UnsafePointer[Int32, ImmutAnyOrigin],
) raises -> Status:
    """Set an N-D tensor descriptor with explicit dimensions and strides.
    Dims are always in NCHW order; strides describe the physical layout."""
    return _get_dylib_function[
        "miopenSetTensorDescriptor",
        def(
            TensorDescriptor,
            DataType,
            Int32,
            type_of(dims),
            type_of(strides),
        ) thin -> Status,
    ]()(desc, dtype, num_dims, dims, strides)


def miopenSetTensorDescriptorV2(
    desc: TensorDescriptor,
    dtype: DataType,
    num_dims: Int32,
    dims: UnsafePointer[UInt64, ImmutAnyOrigin],
    strides: UnsafePointer[UInt64, ImmutAnyOrigin],
) raises -> Status:
    """Set an N-D tensor descriptor with explicit dimensions and strides.
    Dims are always in NCHW order; strides describe the physical layout."""
    return _get_dylib_function[
        "miopenSetTensorDescriptorV2",
        def(
            TensorDescriptor,
            DataType,
            Int32,
            type_of(dims),
            type_of(strides),
        ) thin -> Status,
    ]()(desc, dtype, num_dims, dims, strides)


def miopenSetNdTensorDescriptorWithLayout(
    desc: TensorDescriptor,
    layout: TensorLayout,
    dtype: DataType,
    num_dims: Int32,
    dims: OpaquePointer,
) raises -> Status:
    return _get_dylib_function[
        "miopenSetNdTensorDescriptorWithLayout",
        def(
            TensorDescriptor,
            TensorLayout,
            DataType,
            Int32,
            type_of(dims),
        ) thin -> Status,
    ]()(desc, layout, dtype, num_dims, dims)


def miopenSet4dTensorDescriptorEx(
    desc: TensorDescriptor,
    dtype: DataType,
    n: Int32,
    c: Int32,
    h: Int32,
    w: Int32,
    n_stride: Int32,
    c_stride: Int32,
    h_stride: Int32,
    w_stride: Int32,
) raises -> Status:
    """Set a 4D tensor descriptor with explicit strides."""
    return _get_dylib_function[
        "miopenSet4dTensorDescriptorEx",
        def(
            TensorDescriptor,
            DataType,
            Int32,
            Int32,
            Int32,
            Int32,
            Int32,
            Int32,
            Int32,
            Int32,
        ) thin -> Status,
    ]()(desc, dtype, n, c, h, w, n_stride, c_stride, h_stride, w_stride)


# ===-----------------------------------------------------------------------===#
# Convolution Descriptor Management
# ===-----------------------------------------------------------------------===#


def miopenCreateConvolutionDescriptor(
    desc: OpaquePointer,
) raises -> Status:
    return _get_dylib_function[
        "miopenCreateConvolutionDescriptor",
        def(type_of(desc)) thin -> Status,
    ]()(desc)


def miopenDestroyConvolutionDescriptor(
    desc: ConvolutionDescriptor,
) raises -> Status:
    return _get_dylib_function[
        "miopenDestroyConvolutionDescriptor",
        def(ConvolutionDescriptor) thin -> Status,
    ]()(desc)


def miopenInitConvolutionNdDescriptor(
    desc: ConvolutionDescriptor,
    spatial_dim: Int32,
    pad: UnsafePointer[Int32, ImmutAnyOrigin],
    stride: UnsafePointer[Int32, ImmutAnyOrigin],
    dilation: UnsafePointer[Int32, ImmutAnyOrigin],
    mode: ConvolutionMode,
) raises -> Status:
    return _get_dylib_function[
        "miopenInitConvolutionNdDescriptor",
        def(
            ConvolutionDescriptor,
            Int32,
            type_of(pad),
            type_of(stride),
            type_of(dilation),
            ConvolutionMode,
        ) thin -> Status,
    ]()(desc, spatial_dim, pad, stride, dilation, mode)


def miopenSetConvolutionGroupCount(
    desc: ConvolutionDescriptor,
    group_count: Int32,
) raises -> Status:
    return _get_dylib_function[
        "miopenSetConvolutionGroupCount",
        def(ConvolutionDescriptor, Int32) thin -> Status,
    ]()(desc, group_count)


# ===-----------------------------------------------------------------------===#
# Convolution Forward Operations
# ===-----------------------------------------------------------------------===#


def miopenConvolutionForwardGetWorkSpaceSize(
    handle: Handle,
    w_desc: TensorDescriptor,
    x_desc: TensorDescriptor,
    conv_desc: ConvolutionDescriptor,
    y_desc: TensorDescriptor,
    workspace_size: UnsafePointer[UInt64, MutAnyOrigin],
) raises -> Status:
    return _get_dylib_function[
        "miopenConvolutionForwardGetWorkSpaceSize",
        def(
            Handle,
            TensorDescriptor,
            TensorDescriptor,
            ConvolutionDescriptor,
            TensorDescriptor,
            type_of(workspace_size),
        ) thin -> Status,
    ]()(handle, w_desc, x_desc, conv_desc, y_desc, workspace_size)


def miopenFindConvolutionForwardAlgorithm(
    handle: Handle,
    x_desc: TensorDescriptor,
    x: OpaquePointer,
    w_desc: TensorDescriptor,
    w: OpaquePointer,
    conv_desc: ConvolutionDescriptor,
    y_desc: TensorDescriptor,
    y: OpaquePointer,
    requested_algo_count: Int32,
    returned_algo_count: UnsafePointer[Int32, MutAnyOrigin],
    perf_results: UnsafePointer[ConvAlgoPerf, MutAnyOrigin],
    workspace: OpaquePointer,
    workspace_size: UInt64,
    exhaustive_search: Bool,
) raises -> Status:
    return _get_dylib_function[
        "miopenFindConvolutionForwardAlgorithm",
        def(
            Handle,
            TensorDescriptor,
            type_of(x),
            TensorDescriptor,
            type_of(w),
            ConvolutionDescriptor,
            TensorDescriptor,
            type_of(y),
            Int32,
            type_of(returned_algo_count),
            type_of(perf_results),
            type_of(workspace),
            UInt64,
            Bool,
        ) thin -> Status,
    ]()(
        handle,
        x_desc,
        x,
        w_desc,
        w,
        conv_desc,
        y_desc,
        y,
        requested_algo_count,
        returned_algo_count,
        perf_results,
        workspace,
        workspace_size,
        exhaustive_search,
    )


def miopenConvolutionForward(
    handle: Handle,
    alpha: UnsafePointer[Float32, ImmutAnyOrigin],
    x_desc: TensorDescriptor,
    x: OpaquePointer,
    w_desc: TensorDescriptor,
    w: OpaquePointer,
    conv_desc: ConvolutionDescriptor,
    algo: ConvFwdAlgorithm,
    beta: UnsafePointer[Float32, ImmutAnyOrigin],
    y_desc: TensorDescriptor,
    y: OpaquePointer,
    workspace: OpaquePointer,
    workspace_size: UInt64,
) raises -> Status:
    return _get_dylib_function[
        "miopenConvolutionForward",
        def(
            Handle,
            type_of(alpha),
            TensorDescriptor,
            type_of(x),
            TensorDescriptor,
            type_of(w),
            ConvolutionDescriptor,
            ConvFwdAlgorithm,
            type_of(beta),
            TensorDescriptor,
            type_of(y),
            type_of(workspace),
            UInt64,
        ) thin -> Status,
    ]()(
        handle,
        alpha,
        x_desc,
        x,
        w_desc,
        w,
        conv_desc,
        algo,
        beta,
        y_desc,
        y,
        workspace,
        workspace_size,
    )


# ===-----------------------------------------------------------------------===#
# Immediate API (Forward)
# ===-----------------------------------------------------------------------===#


def miopenConvolutionForwardGetSolutionCount(
    handle: Handle,
    w_desc: TensorDescriptor,
    x_desc: TensorDescriptor,
    conv_desc: ConvolutionDescriptor,
    y_desc: TensorDescriptor,
    solution_count: OpaquePointer,
) raises -> Status:
    return _get_dylib_function[
        "miopenConvolutionForwardGetSolutionCount",
        def(
            Handle,
            TensorDescriptor,
            TensorDescriptor,
            ConvolutionDescriptor,
            TensorDescriptor,
            type_of(solution_count),
        ) thin -> Status,
    ]()(handle, w_desc, x_desc, conv_desc, y_desc, solution_count)


def miopenConvolutionForwardGetSolution(
    handle: Handle,
    w_desc: TensorDescriptor,
    x_desc: TensorDescriptor,
    conv_desc: ConvolutionDescriptor,
    y_desc: TensorDescriptor,
    max_solution_count: UInt64,
    solution_count: OpaquePointer,
    solutions: OpaquePointer,
) raises -> Status:
    return _get_dylib_function[
        "miopenConvolutionForwardGetSolution",
        def(
            Handle,
            TensorDescriptor,
            TensorDescriptor,
            ConvolutionDescriptor,
            TensorDescriptor,
            UInt64,
            type_of(solution_count),
            type_of(solutions),
        ) thin -> Status,
    ]()(
        handle,
        w_desc,
        x_desc,
        conv_desc,
        y_desc,
        max_solution_count,
        solution_count,
        solutions,
    )


def miopenConvolutionForwardCompileSolution(
    handle: Handle,
    w_desc: TensorDescriptor,
    x_desc: TensorDescriptor,
    conv_desc: ConvolutionDescriptor,
    y_desc: TensorDescriptor,
    solution_id: UInt64,
) raises -> Status:
    return _get_dylib_function[
        "miopenConvolutionForwardCompileSolution",
        def(
            Handle,
            TensorDescriptor,
            TensorDescriptor,
            ConvolutionDescriptor,
            TensorDescriptor,
            UInt64,
        ) thin -> Status,
    ]()(handle, w_desc, x_desc, conv_desc, y_desc, solution_id)


def miopenConvolutionForwardImmediate(
    handle: Handle,
    w_desc: TensorDescriptor,
    w: OpaquePointer,
    x_desc: TensorDescriptor,
    x: OpaquePointer,
    conv_desc: ConvolutionDescriptor,
    y_desc: TensorDescriptor,
    y: OpaquePointer,
    workspace: OpaquePointer,
    workspace_size: UInt64,
    solution_id: UInt64,
) raises -> Status:
    return _get_dylib_function[
        "miopenConvolutionForwardImmediate",
        def(
            Handle,
            TensorDescriptor,
            type_of(w),
            TensorDescriptor,
            type_of(x),
            ConvolutionDescriptor,
            TensorDescriptor,
            type_of(y),
            type_of(workspace),
            UInt64,
            UInt64,
        ) thin -> Status,
    ]()(
        handle,
        w_desc,
        w,
        x_desc,
        x,
        conv_desc,
        y_desc,
        y,
        workspace,
        workspace_size,
        solution_id,
    )


# ===-----------------------------------------------------------------------===#
# Find 2.0 / Problem API
# ===-----------------------------------------------------------------------===#


def miopenCreateConvProblem(
    problem: OpaquePointer,
    conv_desc: ConvolutionDescriptor,
    direction: ProblemDirection,
) raises -> Status:
    return _get_dylib_function[
        "miopenCreateConvProblem",
        def(
            type_of(problem),
            ConvolutionDescriptor,
            ProblemDirection,
        ) thin -> Status,
    ]()(problem, conv_desc, direction)


def miopenDestroyProblem(
    problem: Problem,
) raises -> Status:
    return _get_dylib_function[
        "miopenDestroyProblem",
        def(Problem) thin -> Status,
    ]()(problem)


def miopenSetProblemTensorDescriptor(
    problem: Problem,
    id: TensorArgumentId,
    desc: TensorDescriptor,
) raises -> Status:
    return _get_dylib_function[
        "miopenSetProblemTensorDescriptor",
        def(
            Problem,
            TensorArgumentId,
            TensorDescriptor,
        ) thin -> Status,
    ]()(problem, id, desc)


def miopenCreateFindOptions(
    options: OpaquePointer,
) raises -> Status:
    return _get_dylib_function[
        "miopenCreateFindOptions",
        def(type_of(options)) thin -> Status,
    ]()(options)


def miopenDestroyFindOptions(
    options: FindOptions,
) raises -> Status:
    return _get_dylib_function[
        "miopenDestroyFindOptions",
        def(FindOptions) thin -> Status,
    ]()(options)


def miopenSetFindOptionPreallocatedWorkspace(
    options: FindOptions,
    workspace: OpaquePointer,
    workspace_size: UInt64,
) raises -> Status:
    return _get_dylib_function[
        "miopenSetFindOptionPreallocatedWorkspace",
        def(
            FindOptions,
            type_of(workspace),
            UInt64,
        ) thin -> Status,
    ]()(options, workspace, workspace_size)


def miopenFindSolutions(
    handle: Handle,
    problem: Problem,
    options: FindOptions,
    solutions: OpaquePointer,
    num_solutions: OpaquePointer,
    max_solutions: UInt64,
) raises -> Status:
    return _get_dylib_function[
        "miopenFindSolutions",
        def(
            Handle,
            Problem,
            FindOptions,
            type_of(solutions),
            type_of(num_solutions),
            UInt64,
        ) thin -> Status,
    ]()(handle, problem, options, solutions, num_solutions, max_solutions)


def miopenGetSolutionWorkspaceSize(
    solution: Solution,
    workspace_size: OpaquePointer,
) raises -> Status:
    return _get_dylib_function[
        "miopenGetSolutionWorkspaceSize",
        def(Solution, type_of(workspace_size)) thin -> Status,
    ]()(solution, workspace_size)


def miopenRunSolution(
    handle: Handle,
    solution: Solution,
    num_inputs: UInt64,
    tensor_args: OpaquePointer,
    workspace: OpaquePointer,
    workspace_size: UInt64,
) raises -> Status:
    return _get_dylib_function[
        "miopenRunSolution",
        def(
            Handle,
            Solution,
            UInt64,
            type_of(tensor_args),
            type_of(workspace),
            UInt64,
        ) thin -> Status,
    ]()(handle, solution, num_inputs, tensor_args, workspace, workspace_size)


def miopenDestroySolution(
    solution: Solution,
) raises -> Status:
    return _get_dylib_function[
        "miopenDestroySolution",
        def(Solution) thin -> Status,
    ]()(solution)
