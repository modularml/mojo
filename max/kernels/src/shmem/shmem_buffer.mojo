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

"""Symmetric heap buffer type for OpenSHMEM-backed multi-GPU memory.

Provides `SHMEMBuffer`, a typed buffer allocated from the SHMEM symmetric heap
so that it is directly addressable by all GPUs in the job.
"""

from std.sys import (
    CompilationTarget,
    has_nvidia_gpu_accelerator,
    has_amd_gpu_accelerator,
    size_of,
)
from std.ffi import external_call

from max.gpu.host import DeviceContext, HostBuffer
from max.gpu.host.device_context import _checked, _CString, _DeviceContextPtr

from .shmem_api import shmem_free, shmem_malloc
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder


struct SHMEMBuffer[dtype: DType](DevicePassable, Sized):
    """A typed buffer allocated from the OpenSHMEM symmetric heap.

    Provides a `DevicePassable` typed buffer backed by `shmem_malloc` so that
    every GPU in the SHMEM job can directly address the allocation. Use
    `SHMEMBuffer` as the storage backing for cross-node collective operations
    where `DeviceBuffer` cannot be used because the memory must be symmetric.

    Parameters:
        dtype: The element data type of the buffer.
    """

    var _data: UnsafePointer[Scalar[Self.dtype], MutUntrackedOrigin]
    var _ctx_ptr: _DeviceContextPtr[mut=True]
    var _size: Int

    comptime device_type: AnyType = UnsafePointer[
        Scalar[Self.dtype], MutAnyOrigin
    ]

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self._data, target)

    @staticmethod
    def get_type_name() -> String:
        return String(t"SHMEMBuffer[{Self.dtype}]")

    @doc_hidden
    @always_inline
    def __init__(
        out self,
        ctx: DeviceContext,
        size: Int,
    ) raises:
        comptime if has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator():
            self._data = shmem_malloc[Self.dtype](size)
            self._ctx_ptr = ctx._handle
            self._size = size
        else:
            CompilationTarget.unsupported_target_error[
                operation="SHMEMBuffer.__init__",
            ]()

    @doc_hidden
    @always_inline
    def __init__(
        out self,
        ctx: DeviceContext,
        data: UnsafePointer[Scalar[Self.dtype], MutUntrackedOrigin],
        size: Int,
    ):
        self._data = data
        self._ctx_ptr = ctx._handle
        self._size = size

    def __deinit__(deinit self):
        shmem_free(self._data)

    def __len__(self) -> Int:
        return self._size

    def unsafe_ptr(self) -> UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]:
        return self._data.as_unsafe_any_origin()

    def enqueue_copy_to(
        self, dst_ptr: UnsafePointer[mut=True, Scalar[Self.dtype], _]
    ) raises:
        """Enqueues an asynchronous copy from this buffer to host memory.

        This method schedules a memory copy operation from this device buffer to the
        specified host memory location. The operation is asynchronous and will be
        executed in the stream associated with this buffer's context.

        Args:
            dst_ptr: Pointer to the destination host memory location.
        """
        _checked(
            external_call[
                "AsyncRT_DeviceContext_DtoH_async_sized",
                _CString[],
                _DeviceContextPtr[mut=True],
                type_of(dst_ptr),
                UnsafePointer[Scalar[Self.dtype], MutUntrackedOrigin],
                Int,
            ](
                self._ctx_ptr,
                dst_ptr,
                self._data,
                self._size * size_of[Self.dtype](),
            )
        )

    def enqueue_copy_to(self, dst: HostBuffer[Self.dtype]) raises:
        """Enqueues an asynchronous copy from this buffer to host memory.

        This method schedules a memory copy operation from this device buffer to the
        specified host memory location. The operation is asynchronous and will be
        executed in the stream associated with this buffer's context.

        Args:
            dst: Host buffer to copy to.

        Raises:
            If the copy operation fails.
        """
        _checked(
            external_call[
                "AsyncRT_DeviceContext_DtoH_async_sized",
                _CString[],
            ](
                self._ctx_ptr,
                dst.unsafe_ptr(),
                self._data,
                Int(self._size * size_of[Self.dtype]()),
            )
        )

    def enqueue_copy_from(
        self, src_ptr: UnsafePointer[mut=True, Scalar[Self.dtype], _]
    ) raises:
        """Enqueues an asynchronous copy from host memory to this buffer.

        This method schedules a memory copy operation from the specified host memory
        location to this device buffer. The operation is asynchronous and will be
        executed in the stream associated with this buffer's context.

        Args:
            src_ptr: Pointer to the source host memory location.

        Raises:
            If the copy operation fails.
        """
        _checked(
            external_call[
                "AsyncRT_DeviceContext_HtoD_async_sized",
                _CString[],
                _DeviceContextPtr[mut=True],
                UnsafePointer[Scalar[Self.dtype], MutUntrackedOrigin],
                type_of(src_ptr),
                Int,
            ](
                self._ctx_ptr,
                self._data,
                src_ptr,
                self._size * size_of[Self.dtype](),
            )
        )

    def enqueue_copy_from(self, src: HostBuffer[Self.dtype]) raises:
        """Enqueues an asynchronous copy from host memory to this buffer.

        This method schedules a memory copy operation from the specified host memory
        location to this device buffer. The operation is asynchronous and will be
        executed in the stream associated with this buffer's context.

        Args:
            src: Host buffer to copy from.

        Raises:
            If the copy operation fails.
        """
        _checked(
            external_call[
                "AsyncRT_DeviceContext_HtoD_async_sized",
                _CString[],
            ](
                self._ctx_ptr,
                self._data,
                src.unsafe_ptr(),
                Int(self._size * size_of[Self.dtype]()),
            )
        )
