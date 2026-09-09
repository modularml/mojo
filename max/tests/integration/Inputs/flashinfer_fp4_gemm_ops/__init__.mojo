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
"""FlashInfer FP4 GEMM custom op for loading TVM FFI modules."""

import extensibility
import std.format
from std.memory import stack_allocation
from std.collections import Span
from std.os import abort

from std.ffi import OwnedDLHandle
from extensibility import InputTensor, OutputTensor, ManagedTensorSlice
from std.utils import IndexList

from max.gpu.host import DeviceContext
from max.gpu.host._nvidia_cuda import CUstream

from .dlpack import DLTensor
from .tvm_ffi import SafeFunction, TVMFFIAny, take_latest_error


@fieldwise_init
struct Module:
    var lib: OwnedDLHandle

    def fp4_gemm(
        self,
        mat1: DLTensor[dtype=DType.uint8, rank=2],
        mat2: DLTensor[dtype=DType.uint8, rank=2],
        mat1_scale: DLTensor[dtype=DType.uint8, rank=1],
        mat2_scale: DLTensor[dtype=DType.uint8, rank=1],
        global_scale: DLTensor[dtype=DType.float32, rank=1],
        out_tensor: DLTensor[dtype=DType.bfloat16, rank=2],
        workspace: DLTensor[dtype=DType.int8, rank=1],
        tactic: Int = 0,  # auto
    ) raises -> None:
        # `borrow()` hands back the underlying handle so we call
        # `get_function` directly. That's safe here because `self.lib`
        # is a long-lived member that outlives the call, so the looked-up
        # symbol can't be `dlclose`d out from under us.
        var safe_call = self.lib.borrow().get_function[SafeFunction](
            "__tvm_ffi_fp4_gemm"
        )

        # `def` params are already mutable local copies, and
        # `DLTensor.__copyinit__` (used at the call site) already fixed
        # up self-referential pointers + nulled strides for contiguous
        # tensors.  So we can take their addresses directly.
        var args: Array[TVMFFIAny, 8] = [
            TVMFFIAny(Pointer(to=mat1)),
            TVMFFIAny(Pointer(to=mat2)),
            TVMFFIAny(Pointer(to=mat1_scale)),
            TVMFFIAny(Pointer(to=mat2_scale)),
            TVMFFIAny(Pointer(to=global_scale)),
            TVMFFIAny(Pointer(to=out_tensor)),
            TVMFFIAny(Pointer(to=workspace)),
            TVMFFIAny(tactic),
        ]

        var result = TVMFFIAny(0)

        var errno = safe_call(
            Optional[Pointer[NoneType, MutAnyOrigin]](),  # null unused module
            Pointer[TVMFFIAny, MutAnyOrigin](to=args[0]),
            8,  # num_args
            Pointer[TVMFFIAny, MutAnyOrigin](to=result),
        )

        if errno != 0:
            var error = take_latest_error()
            raise Error("FlashInfer fp4_gemm failed: {}".format(error))


@extensibility.register("flashinfer_fp4_gemm")
struct FlashInferFP4Gemm[lib_path: StaticString]:
    """Custom op that calls FlashInfer FP4 GEMM via TVM FFI.

    Parameters:
        lib_path: Path to the FlashInfer .so file built by flashinfer.aot.

    Inputs:
        - mat1: [M, K/2] uint8 (packed FP4)
        - mat2: [N, K/2] uint8 (packed FP4)
        - mat1_scale: scale factors for mat1
        - mat2_scale: scale factors for mat2
        - global_scale: [1] float32
        - workspace: workspace buffer

    Output:
        - out: [M, N] bfloat16

    Note: TVM FFI must be loaded by Python (`import tvm_ffi` and
    `tvm_ffi.module.load_module`) before this op runs.
    """

    @staticmethod
    def execute[
        target: StaticString
    ](
        out_tensor: OutputTensor[dtype=.bfloat16, rank=2, ...],
        mat1: InputTensor[dtype=.uint8, rank=2, ...],
        mat2: InputTensor[dtype=.uint8, rank=2, ...],
        mat1_scale: InputTensor[dtype=.uint8, rank=1, ...],
        mat2_scale: InputTensor[dtype=.uint8, rank=1, ...],
        global_scale: InputTensor[dtype=.float32, rank=1, ...],
        workspace: InputTensor[dtype=.int8, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        """Execute the FP4 GEMM operation by calling FlashInfer."""
        comptime assert target == "gpu"

        var mod = Module(OwnedDLHandle(path=Self.lib_path))

        mod.fp4_gemm(
            DLTensor(mat1),
            DLTensor(mat2),
            DLTensor(mat1_scale),
            DLTensor(mat2_scale),
            DLTensor(global_scale),
            DLTensor(out_tensor),
            DLTensor(workspace),
        )
