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

from std.pathlib import Path
from std.os import getenv, abort
from std.ffi import (
    _get_dylib_function,
    _Global,
    OwnedDLHandle,
    c_int,
    RTLD,
)
from std.sys.info import has_nvidia_gpu_accelerator, has_amd_gpu_accelerator

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#

comptime MPI_LIBRARY = _Global["MPI_LIBRARY", _init_mpi_dylib]


def mpi_lib_name() -> String:
    comptime if has_nvidia_gpu_accelerator():
        return "nvshmem_bootstrap_mpi.so.3"
    else:
        return "libmpi.so"


def _init_mpi_dylib() -> OwnedDLHandle:
    var lib = mpi_lib_name()
    # If provided, allow an override directory for nvshmem bootstrap libs.
    # Example:
    #   export MODULAR_SHMEM_LIB_DIR="/path/to/venv/lib"
    # will dlopen the library from:
    #   /path/to/venv/lib/libmpi.so
    var dir_name = getenv("MODULAR_SHMEM_LIB_DIR")
    if dir_name:
        lib = String(Path(dir_name) / lib)

    var flags = RTLD.NOW | RTLD.GLOBAL

    # AMD interaction with libmpi needs the handle to stay alive after MPI_Finalize
    comptime if has_amd_gpu_accelerator():
        flags = flags | RTLD.NODELETE

    try:
        return OwnedDLHandle(path=lib, flags=flags)
    except e:
        abort(t"failed to load MPI library: {e}")


@always_inline
def _get_mpi_function[
    func_name: StaticString, result_type: TrivialRegisterPassable
]() raises -> result_type:
    return _get_dylib_function[
        MPI_LIBRARY(),
        func_name,
        result_type,
    ]()


# ===-----------------------------------------------------------------------===#
# Types and constants
# ===-----------------------------------------------------------------------===#

comptime MPIComm = UnsafePointer[
    OpaquePointer[MutUntrackedOrigin], MutUntrackedOrigin
]

comptime MPI_THREAD_SINGLE = 0
comptime MPI_THREAD_FUNNELED = 1
comptime MPI_THREAD_SERIALIZED = 2
comptime MPI_THREAD_MULTIPLE = 3

# ===-----------------------------------------------------------------------===#
# Function bindings
# ===-----------------------------------------------------------------------===#


def MPI_Init(
    mut argc: Int, mut argv: Span[StaticString, ImmStaticOrigin]
) raises:
    """Initialize MPI."""
    var result = _get_mpi_function[
        "MPI_Init",
        def(
            UnsafePointer[Int, origin_of(argc)],
            UnsafePointer[Span[StaticString, ImmStaticOrigin], origin_of(argv)],
        ) thin -> c_int,
    ]()(UnsafePointer(to=argc), UnsafePointer(to=argv))
    if result != 0:
        raise Error("failed to MPI_Init with error code:", result)


def MPI_Init_thread(
    mut argc: Int,
    mut argv: Span[StaticString, ImmStaticOrigin],
    required: c_int,
    provided: UnsafePointer[c_int, MutAnyOrigin],
) raises:
    """Initialize MPI."""
    var result = _get_mpi_function[
        "MPI_Init_thread",
        def(
            UnsafePointer[Int, origin_of(argc)],
            UnsafePointer[Span[StaticString, ImmStaticOrigin], origin_of(argv)],
            c_int,
            UnsafePointer[c_int, MutAnyOrigin],
        ) thin -> c_int,
    ]()(UnsafePointer(to=argc), UnsafePointer(to=argv), required, provided)
    if result != 0:
        raise Error("failed to MPI_Init_thread with error code:", result)


def MPI_Initialized(flag: UnsafePointer[c_int, MutUntrackedOrigin]) raises:
    """Check if MPI has been initialized.

    Raises:
        If the MPI operation fails.
    """
    var result = _get_mpi_function[
        "MPI_Initialized",
        def(UnsafePointer[c_int, MutUntrackedOrigin]) thin -> c_int,
    ]()(flag)
    if result != 0:
        raise Error("failed to check MPI_Initialized with error code:", result)


def MPI_Finalize() raises:
    """Finalize MPI.

    Raises:
        If the MPI operation fails.
    """
    var result = _get_mpi_function[
        "MPI_Finalize",
        def() thin -> c_int,
    ]()()
    if result != 0:
        raise Error("failed to finalize MPI with error code:", result)


def MPI_Comm_split(
    comm: MPIComm,
    color: c_int,
    key: c_int,
    newcomm: UnsafePointer[MPIComm, MutAnyOrigin],
) raises:
    """Split a communicator into multiple subcommunicators."""
    var result = _get_mpi_function[
        "MPI_Comm_split",
        def(
            MPIComm, c_int, c_int, UnsafePointer[MPIComm, MutAnyOrigin]
        ) thin -> c_int,
    ]()(comm, color, key, newcomm)
    if result != 0:
        raise Error("failed to MPI_Comm_split with error code:", result)


def MPI_Comm_rank(
    comm: MPIComm, rank: UnsafePointer[c_int, MutAnyOrigin]
) raises:
    """Get the rank of the current process in the communicator."""
    var result = _get_mpi_function[
        "MPI_Comm_rank",
        def(MPIComm, UnsafePointer[c_int, MutAnyOrigin]) thin -> c_int,
    ]()(comm, rank)
    if result != 0:
        raise Error("failed to get MPI_Comm_rank with error code:", result)


def MPI_Comm_size(
    comm: MPIComm, size: UnsafePointer[c_int, MutAnyOrigin]
) raises:
    """Get the size of the communicator."""
    var result = _get_mpi_function[
        "MPI_Comm_size",
        def(MPIComm, UnsafePointer[c_int, MutAnyOrigin]) thin -> c_int,
    ]()(comm, size)
    if result != 0:
        raise Error("failed to get MPI_Comm_size with error code:", result)


def get_mpi_comm_world() raises -> MPIComm:
    """Get the MPI_COMM_WORLD communicator.

    Raises:
        If the MPI library is not available or the symbol cannot be found.
    """
    var handle = MPI_LIBRARY.get_or_create_ptr()[].borrow()
    var comm_world_ptr = handle.get_symbol[OpaquePointer[MutUntrackedOrigin]](
        cstr_name="ompi_mpi_comm_world".as_c_string_slice()
    )
    if not comm_world_ptr:
        raise Error("symbol ompi_mpi_comm_world not found in MPI library")
    return comm_world_ptr.value().unsafe_origin_cast[MutUntrackedOrigin]()
