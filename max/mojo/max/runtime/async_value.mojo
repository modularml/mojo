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

"""Provides a limited Mojo handle to the AsyncRT `AsyncValue` type.

`AnyAsyncValueRef` is an owning, reference-counted handle to a C++ `AsyncValue`.
It is used as the storage handle that keeps a device buffer's or tensor's
backing memory alive (for example, inside the `OwnedByteBuffer` and
`OwnedTensor` composites), mirroring the C++ `AnyAsyncValueRef` /
`RCRef[AsyncValue]`.
"""

from std.ffi import external_call

from max.gpu.host import DeviceBuffer


# Opaque stand-in for the C++ `AsyncValue` type -- the reference-counted payload
# behind an `AnyAsyncValueRef`.
struct _AsyncValueCpp:
    pass


# Typed, nullable C pointer to the C++ `AsyncValue`. A null handle is the empty /
# non-tracked reference, so `OptionalPointer` (nullable) is required rather than a bare
# `UnsafePointer`; the default `UntrackedOrigin` marks the pointee as living
# outside the Mojo program, mirroring `_DeviceBufferPtr`.
comptime _AsyncValuePtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = OptionalPointer[_AsyncValueCpp, origin]


struct AnyAsyncValueRef(ImplicitlyCopyable, Movable):
    """Owning, reference-counted handle to a C++ `AsyncValue` -- a limited Mojo
    counterpart of the C++ `AnyAsyncValueRef` / `RCRef[AsyncValue]`.

    Holds a pointer to the `AsyncValue`; copying retains (adds one reference) and
    destruction releases one. A null handle is the empty / non-tracked reference.
    Used as the storage handle of an `OwnedByteBuffer`, mirroring the `storageRef`
    field of the C++ `TensorBufferRef`.
    """

    var _handle: _AsyncValuePtr[mut=True]

    def __init__(out self):
        """Creates an empty (null) AsyncValue reference."""
        self._handle = {}

    def __init__(out self, handle: _AsyncValuePtr[mut=True]):
        """Adopts an already-owning handle net-zero (does not retain).

        Args:
            handle: An `AsyncValue*` whose single live reference is transferred
                to this wrapper.
        """
        self._handle = handle

    def __init__(out self, *, copy: Self):
        """Creates a new owning reference by retaining the shared `AsyncValue`.

        Args:
            copy: The reference to copy.
        """
        # void AsyncRT_AsyncValue_retain(AsyncValue *value)
        external_call["AsyncRT_AsyncValue_retain", NoneType](copy._handle)
        self._handle = copy._handle

    def __init__(out self, *, retained_storage_of: OpaquePointer[MutAnyOrigin]):
        """Retains the backing storage of a packed async slot into a new ref.

        Reads the `AsyncValue*` behind the slot's `TensorBufferRef` storage
        handle and adds one reference, so the returned handle independently keeps
        the backing memory alive (e.g. when an unpacked value is re-packed).

        Args:
            retained_storage_of: A pointer to the packed async value (slot) whose
                backing storage should be retained.
        """
        # AsyncValue *AsyncRT_AsyncValue_retainBufferStorage(
        #     AnyAsyncValueRef *async)
        self._handle = external_call[
            "AsyncRT_AsyncValue_retainBufferStorage", _AsyncValuePtr[mut=True]
        ](retained_storage_of)

    def __init__(out self, *, retain_handle: OpaquePointer[MutAnyOrigin]):
        """Retains an existing `AnyAsyncValueRef` storage handle into a new ref.

        Reads the `AsyncValue*` behind the given C++ `AnyAsyncValueRef` handle
        (e.g. a cached buffer's memory handle) and adds one reference. A null
        handle yields the empty (non-tracked) reference.

        Args:
            retain_handle: A pointer to a C++ `AnyAsyncValueRef` storage handle.
        """
        # AsyncValue *AsyncRT_AsyncValue_retainHandle(AnyAsyncValueRef *handle)
        self._handle = external_call[
            "AsyncRT_AsyncValue_retainHandle", _AsyncValuePtr[mut=True]
        ](retain_handle)

    def __init__(out self, *, var storage_buf: DeviceBuffer):
        """Wraps a live owning `DeviceBuffer` in an `AsyncValue[DeviceBufferRef]`.

        The buffer's handle is surrendered net-zero (`take_handle`) and adopted
        by the runtime, so no extra reference is created.

        Args:
            storage_buf: The owning device buffer to wrap.
        """
        # AsyncValue *AsyncRT_AsyncValue_createFromDeviceBuffer(
        #     DeviceBuffer *handle)
        var handle = external_call[
            "AsyncRT_AsyncValue_createFromDeviceBuffer",
            _AsyncValuePtr[mut=True],
        ](storage_buf^.take_handle())
        self._handle = handle

    def __deinit__(deinit self):
        """Releases this reference to the underlying `AsyncValue`."""
        # void AsyncRT_AsyncValue_release(AsyncValue *value)
        external_call["AsyncRT_AsyncValue_release", NoneType](self._handle)

    def take_handle(deinit self) -> _AsyncValuePtr[mut=True]:
        """Surrenders the owning handle net-zero, suppressing the destructor.

        Returns:
            The owning `AsyncValue*`; the caller must hand it to a runtime owner
            that adopts it without an extra reference.
        """
        return self._handle
