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
from std.collections.string.string_span import get_static_string
from std.os import abort, getenv
from std.pathlib import Path
from std.sys import argv, size_of
from std.ffi import (
    _get_dylib_function,
    _Global,
    OwnedDLHandle,
    c_int,
    c_size_t,
    external_call,
)
from std.sys.info import CompilationTarget, is_nvidia_gpu

from max.gpu.host import DeviceContext
from max.gpu.host._nvidia_cuda import CUmodule, CUstream

from ._mpi import MPI_Comm_rank, MPI_Init, MPIComm, get_mpi_comm_world
from .shmem_api import SHMEMScope

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#


struct NVSHMEMIVersion(RegisterPassable):
    var major: c_int
    var minor: c_int
    var patch: c_int

    def __init__(out self):
        self.major = 3
        self.minor = 4
        self.patch = 5


comptime NVSHMEM_LIBRARY = _Global["NVSHMEM_LIBRARY", _init_nvshmem_dylib]


def _init_nvshmem_dylib() -> OwnedDLHandle:
    var lib = "libnvshmem_host.so.3"
    # If provided, allow an override directory for nvshmem bootstrap libs.
    # Example:
    #   export MODULAR_SHMEM_LIB_DIR="/path/to/venv/lib"
    # will dlopen the library from:
    #   /path/to/venv/lib/libnvshmem_host.so.3
    var dir_name = getenv("MODULAR_SHMEM_LIB_DIR")
    if dir_name:
        lib = String(Path(dir_name) / lib)
    try:
        return OwnedDLHandle(path=lib)
    except e:
        abort(t"failed to load NVSHMEM library: {e}")


@always_inline
def _get_nvshmem_function[
    func_name: StaticString, result_type: TrivialRegisterPassable
]() -> result_type:
    try:
        return _get_dylib_function[
            NVSHMEM_LIBRARY(),
            func_name,
            result_type,
        ]()
    except e:
        abort(String(e))


# ===-----------------------------------------------------------------------===#
# Types
# ===-----------------------------------------------------------------------===#

comptime nvshmem_team_id_t = Int32

# ===-----------------------------------------------------------------------===#
# Constants
# ===-----------------------------------------------------------------------===#

comptime NVSHMEM_SUCCESS = 0

comptime NVSHMEMX_INIT_WITH_MPI_COMM = 1 << 1

comptime CHANNEL_BUF_SIZE: c_int = 1 << 22
comptime CHANNEL_BUF_SIZE_LOG: c_int = 22
comptime CHANNEL_ENTRY_BYTES: c_int = 8

comptime NVSHMEMX_ERROR_INTERNAL = 1
comptime NVSHMEM_MAX_NAME_LEN: c_int = 256

comptime NVSHMEM_THREAD_SINGLE: c_int = 0
comptime NVSHMEM_THREAD_FUNNELED: c_int = 1
comptime NVSHMEM_THREAD_SERIALIZED: c_int = 2
comptime NVSHMEM_THREAD_MULTIPLE: c_int = 3
comptime NVSHMEM_THREAD_TYPE_SENTINEL: c_int = c_int.MAX

comptime NVSHMEM_CMP_EQ: c_int = 0
comptime NVSHMEM_CMP_NE: c_int = 1
comptime NVSHMEM_CMP_GT: c_int = 2
comptime NVSHMEM_CMP_LE: c_int = 3
comptime NVSHMEM_CMP_LT: c_int = 4
comptime NVSHMEM_CMP_GE: c_int = 5
comptime NVSHMEM_CMP_SENTINEL: c_int = c_int.MAX

comptime PROXY_GLOBAL_EXIT_INIT: c_int = 1
comptime PROXY_GLOBAL_EXIT_REQUESTED: c_int = 2
comptime PROXY_GLOBAL_EXIT_FINISHED: c_int = 3
comptime PROXY_GLOBAL_EXIT_MAX_STATE: c_int = c_int.MAX

comptime PROXY_DMA_REQ_BYTES: c_int = 32
comptime PROXY_AMO_REQ_BYTES: c_int = 40
comptime PROXY_INLINE_REQ_BYTES: c_int = 24

comptime NVSHMEM_STATUS_NOT_INITIALIZED: c_int = 0
comptime NVSHMEM_STATUS_IS_BOOTSTRAPPED: c_int = 1
comptime NVSHMEM_STATUS_IS_INITIALIZED: c_int = 2
comptime NVSHMEM_STATUS_LIMITED_MPG: c_int = 4
comptime NVSHMEM_STATUS_FULL_MPG: c_int = 5
comptime NVSHMEM_STATUS_INVALID: c_int = c_int.MAX

comptime NVSHMEM_SIGNAL_SET: c_int = 9
comptime NVSHMEM_SIGNAL_ADD: c_int = 10

comptime NVSHMEM_TEAM_INVALID: nvshmem_team_id_t = -1
comptime NVSHMEM_TEAM_WORLD: nvshmem_team_id_t = 0
comptime NVSHMEM_TEAM_WORLD_INDEX: nvshmem_team_id_t = 0
comptime NVSHMEM_TEAM_SHARED: nvshmem_team_id_t = 1
comptime NVSHMEM_TEAM_SHARED_INDEX: nvshmem_team_id_t = 1
comptime NVSHMEMX_TEAM_NODE: nvshmem_team_id_t = 2
comptime NVSHMEM_TEAM_NODE_INDEX: nvshmem_team_id_t = 2
comptime NVSHMEMX_TEAM_SAME_MYPE_NODE: nvshmem_team_id_t = 3
comptime NVSHMEM_TEAM_SAME_MYPE_NODE_INDEX: nvshmem_team_id_t = 3
comptime NVSHMEMI_TEAM_SAME_GPU: nvshmem_team_id_t = 4
comptime NVSHMEM_TEAM_SAME_GPU_INDEX: nvshmem_team_id_t = 4
comptime NVSHMEMI_TEAM_GPU_LEADERS: nvshmem_team_id_t = 5
comptime NVSHMEM_TEAM_GPU_LEADERS_INDEX: nvshmem_team_id_t = 5
comptime NVSHMEM_TEAMS_MIN: nvshmem_team_id_t = 6
comptime NVSHMEM_TEAM_INDEX_MAX: nvshmem_team_id_t = nvshmem_team_id_t.MAX


# Structs
struct NVSHMEMXInitAttr[origin: MutOrigin]:
    var version: c_int

    var mpi_comm: UnsafePointer[MPIComm, Self.origin]
    var args: NVSHMEMXInitArgs

    def __init__(out self, mpi_comm: UnsafePointer[MPIComm, Self.origin]):
        comptime assert (
            size_of[Self]() == 144
        ), "NVSHMEMXInitAttr must be 144 bytes"
        self.version = c_int((1 << 16) + size_of[Self]())
        self.mpi_comm = mpi_comm
        self.args = NVSHMEMXInitArgs()


struct NVSHMEMXInitArgs:
    var version: c_int
    var uid_args: NVSHMEMXUniqueIDArgs
    var content: Array[Byte, 96]

    def __init__(out self):
        comptime assert (
            size_of[Self]() == 128
        ), "NVSHMEMXInitArgs must be 128 bytes"
        self.version = c_int((1 << 16) + size_of[NVSHMEMXInitArgs]())
        self.uid_args = NVSHMEMXUniqueIDArgs()
        self.content = Array[Byte, 96](fill=0)


struct NVSHMEMXUniqueIDArgs:
    var version: c_int

    var id: Optional[UnsafePointer[NVSHMEMXUniqueID, MutUntrackedOrigin]]
    var myrank: c_int
    var nranks: c_int

    def __init__(out self):
        comptime assert (
            size_of[Self]() == 24
        ), "NVSHMEMXUniqueIDArgs must be 24 bytes"
        self.version = c_int((1 << 16) + size_of[NVSHMEMXUniqueIDArgs]())
        self.id = None
        self.myrank = 0
        self.nranks = 0


struct NVSHMEMXUniqueID:
    var version: c_int
    var internal: Array[Byte, 124]

    def __init__(out self):
        comptime assert (
            size_of[Self]() == 128
        ), "nvshmemx_uniqueid_t must be 128 bytes"
        self.version = c_int((1 << 16) + size_of[NVSHMEMXUniqueID]())
        self.internal = Array[Byte, 124](fill=0)


def _get_prefix[scope: SHMEMScope]() -> StaticString:
    comptime if scope == SHMEMScope.default:
        return "nvshmem_"
    else:
        return "nvshmemx_"


def _dtype_to_nvshmem_type[
    prefix: StaticString,
    dtype: DType,
    suffix: StaticString,
    scope: StaticString = "",
]() -> StaticString:
    """
    Returns the NVSHMEM name for the given dtype surrounded by the given prefix
    and suffix, for calling the correct symbol on the device-side bitcode.

    The below mappings come from
    https://docs.nvidia.com/nvshmem/api/gen/api/rma.html#remote-memory-access as
    of nvshmem 3.3, and the bitwidths were generated from CUDA 12.6 on x86-64
    Ubuntu 22.04.

    c_name               nvshmem_name  bitwidth
    -------------------------------------------
    float                float         32
    double               double        64
    __nv_bfloat16        bfloat16      16
    half                 half          16
    char                 char          8
    signed char          schar         8
    short                short         16
    int                  int           32
    long                 long          64
    long long            longlong      64
    unsigned char        uchar         8
    unsigned short       ushort        16
    unsigned int         uint          32
    unsigned long        ulong         64
    unsigned long long   ulonglong     64
    int8_t               int8          8
    int16_t              int16         16
    int32_t              int32         32
    int64_t              int64         64
    uint8_t              uint8         8
    uint16_t             uint16        16
    uint32_t             uint32        32
    uint64_t             uint64        64
    size_t               size          64
    ptrdiff_t            ptrdiff       64
    """

    comptime if dtype == .float16:
        return get_static_string[prefix, "half", suffix, scope]()
    elif dtype == .bfloat16:
        return get_static_string[prefix, "bfloat16", suffix, scope]()
    elif dtype == .float32:
        return get_static_string[prefix, "float", suffix, scope]()
    elif dtype == .float64:
        return get_static_string[prefix, "double", suffix, scope]()
    elif dtype == .int8:
        return get_static_string[prefix, "int8", suffix, scope]()
    elif dtype == .uint8:
        return get_static_string[prefix, "uint8", suffix, scope]()
    elif dtype == .int16:
        return get_static_string[prefix, "int16", suffix, scope]()
    elif dtype == .uint16:
        return get_static_string[prefix, "uint16", suffix, scope]()
    elif dtype == .int32:
        return get_static_string[prefix, "int32", suffix, scope]()
    elif dtype == .uint32:
        return get_static_string[prefix, "uint32", suffix, scope]()
    elif dtype == .int64:
        return get_static_string[prefix, "int64", suffix, scope]()
    elif dtype == .uint64:
        return get_static_string[prefix, "uint64", suffix, scope]()
    elif dtype == .int:
        return get_static_string[prefix, "size", suffix, scope]()
    else:
        CompilationTarget.unsupported_target_error[
            operation=__get_current_function_name()
        ]()


# ===-----------------------------------------------------------------------===#
# 1: Library Setup, Exit, and Query
# https://docs.nvidia.com/nvshmem/api/gen/api/setup.html
# ===-----------------------------------------------------------------------===#


# Run one GPU per process
def nvshmemx_init() raises:
    var _argv = argv()
    var argc = len(_argv)
    MPI_Init(argc, _argv)

    # Get MPI rank and size
    var rank = c_int(0)
    var mpi_comm = get_mpi_comm_world()

    _ = MPI_Comm_rank(mpi_comm, UnsafePointer(to=rank).as_unsafe_any_origin())
    # Set CUDA device early - needed for CUDA-related NVSHMEM initialization
    var ctx = DeviceContext(device_id=Int(rank))
    ctx.set_as_current()

    # Initialize NVSHMEM with MPI
    var attr = NVSHMEMXInitAttr(UnsafePointer(to=mpi_comm))
    # For single process per GPU, fallback to one device per process per node.
    attr.args.uid_args.myrank = 0
    attr.args.uid_args.nranks = 1

    _ = nvshmemx_hostlib_init_attr(
        NVSHMEMX_INIT_WITH_MPI_COMM,
        UnsafePointer(to=attr).as_unsafe_any_origin(),
    )

    # Check initialization status
    if nvshmemx_init_status() != 2:
        raise Error("failed to initialize NVSHMEM")


# Modular specific, initialize a DeviceContext on this thread to be SHMEM
# enabled.
def nvshmemx_init_thread(ctx: DeviceContext, gpus_per_node: Int = -1) raises:
    # Must set the associated CUcontext on this thread prior to init
    ctx.set_as_current()
    var nranks = gpus_per_node if gpus_per_node > 0 else ctx.number_of_devices()

    # Initialize NVSHMEM with MPI
    var mpi_comm = get_mpi_comm_world()
    var attr = NVSHMEMXInitAttr(UnsafePointer(to=mpi_comm))
    attr.args.uid_args.myrank = Int32(ctx.id())
    attr.args.uid_args.nranks = c_int(nranks)

    var status = nvshmemx_hostlib_init_attr(
        NVSHMEMX_INIT_WITH_MPI_COMM,
        UnsafePointer(to=attr).as_unsafe_any_origin(),
    )
    if status:
        raise Error("failed to initialize NVSHMEM with status:", status)
    # Check initialization status
    status = nvshmemx_init_status()
    if status != 2:
        raise Error("failed to initialize NVSHMEM with status:", status)


def nvshmemx_hostlib_init_attr[
    origin: MutOrigin, //
](
    flags: UInt32,
    attr: UnsafePointer[NVSHMEMXInitAttr[origin], MutAnyOrigin],
) -> c_int:
    return _get_nvshmem_function[
        "nvshmemx_hostlib_init_attr",
        def(UInt32, type_of(attr)) thin -> c_int,
    ]()(flags, attr)


def nvshmemx_hostlib_finalize():
    _get_nvshmem_function[
        "nvshmemx_hostlib_finalize",
        def() thin -> NoneType,
    ]()()


def nvshmemx_cumodule_init(module: CUmodule) -> c_int:
    return _get_nvshmem_function[
        "nvshmemx_cumodule_init",
        def(CUmodule) thin -> c_int,
    ]()(module)


def nvshmemx_cumodule_finalize(module: CUmodule) -> c_int:
    return _get_nvshmem_function[
        "nvshmemx_cumodule_finalize",
        def(CUmodule) thin -> c_int,
    ]()(module)


def nvshmemx_init_status() -> c_int:
    return _get_nvshmem_function[
        "nvshmemx_init_status",
        def() thin -> c_int,
    ]()()


def nvshmem_my_pe() -> c_int:
    comptime if is_nvidia_gpu():
        return external_call["nvshmem_my_pe", c_int]()
    else:
        return _get_nvshmem_function[
            "nvshmem_my_pe",
            def() thin -> c_int,
        ]()()


def nvshmem_n_pes() -> c_int:
    comptime if is_nvidia_gpu():
        return external_call["nvshmem_n_pes", c_int]()
    else:
        return _get_nvshmem_function[
            "nvshmem_n_pes",
            def() thin -> c_int,
        ]()()


# ===----------------------------------------------------------------------=== #
# 3: Memory Management
# https://docs.nvidia.com/nvshmem/api/gen/api/memory.html
# ===----------------------------------------------------------------------=== #


def nvshmem_malloc[
    dtype: DType
](size: c_size_t) -> UnsafePointer[Scalar[dtype], MutUntrackedOrigin]:
    return _get_nvshmem_function[
        "nvshmem_malloc",
        def(c_size_t) thin -> UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    ]()(size)


def nvshmem_calloc[
    dtype: DType
](count: c_size_t, size: c_size_t) -> UnsafePointer[
    Scalar[dtype], MutUntrackedOrigin
]:
    return _get_nvshmem_function[
        "nvshmem_calloc",
        def(
            c_size_t, c_size_t
        ) thin -> UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    ]()(count, size)


def nvshmem_free[
    dtype: DType, //
](ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin]):
    _get_nvshmem_function[
        "nvshmem_free",
        def(type_of(ptr)) thin -> NoneType,
    ]()(ptr)


# ===----------------------------------------------------------------------=== #
# 4: Team Management
# https://docs.nvidia.com/nvshmem/api/gen/api/symmetric.html
# ===----------------------------------------------------------------------=== #


def nvshmem_team_my_pe(team: c_int) -> c_int:
    return _get_nvshmem_function[
        "nvshmem_team_my_pe",
        def(c_int) thin -> c_int,
    ]()(team)


def nvshmemx_team_init() -> c_int:
    return _get_nvshmem_function[
        "nvshmemx_team_init",
        def() thin -> c_int,
    ]()()


# ===----------------------------------------------------------------------=== #
# 6: Remote Memory Access (RMA)
# https://docs.nvidia.com/nvshmem/api/gen/api/rma.html
# ===----------------------------------------------------------------------=== #


def nvshmem_put[
    dtype: DType,
    //,
    scope: SHMEMScope,
](
    dest: UnsafePointer[Scalar[dtype], _],
    source: UnsafePointer[Scalar[dtype], _],
    nelems: c_size_t,
    pe: c_int,
):
    comptime symbol = _dtype_to_nvshmem_type[
        _get_prefix[scope](), dtype, "_put", scope.value
    ]()
    external_call[symbol, NoneType](dest, source, nelems, pe)


def nvshmem_put_nbi[
    dtype: DType,
    //,
    scope: SHMEMScope,
](
    dest: UnsafePointer[Scalar[dtype], _],
    source: UnsafePointer[Scalar[dtype], _],
    nelems: c_size_t,
    pe: c_int,
):
    comptime symbol = _dtype_to_nvshmem_type[
        _get_prefix[scope](), dtype, "_put_nbi", scope.value
    ]()
    external_call[symbol, NoneType](dest, source, nelems, pe)


def nvshmem_p[
    dtype: DType
](dest: UnsafePointer[Scalar[dtype], _], value: Scalar[dtype], pe: c_int,):
    comptime symbol = _dtype_to_nvshmem_type["nvshmem_", dtype, "_p"]()
    external_call[symbol, NoneType](dest, value, pe)


def nvshmem_get[
    dtype: DType,
    //,
    scope: SHMEMScope,
](
    dest: UnsafePointer[Scalar[dtype], _],
    source: UnsafePointer[Scalar[dtype], _],
    nelems: c_size_t,
    pe: c_int,
):
    comptime symbol = _dtype_to_nvshmem_type[
        _get_prefix[scope](), dtype, "_get", scope.value
    ]()
    external_call[symbol, NoneType](dest, source, nelems, pe)


def nvshmem_get_nbi[
    dtype: DType,
    //,
    scope: SHMEMScope,
](
    dest: UnsafePointer[Scalar[dtype], _],
    source: UnsafePointer[Scalar[dtype], _],
    nelems: c_size_t,
    pe: c_int,
):
    comptime symbol = _dtype_to_nvshmem_type[
        _get_prefix[scope](), dtype, "_get_nbi", scope.value
    ]()
    external_call[symbol, NoneType](dest, source, nelems, pe)


def nvshmem_g[
    dtype: DType
](source: UnsafePointer[Scalar[dtype], _], pe: c_int) -> Scalar[dtype]:
    comptime symbol = _dtype_to_nvshmem_type["nvshmem_", dtype, "_g"]()
    return external_call[symbol, Scalar[dtype]](source, pe)


# ===----------------------------------------------------------------------=== #
# 8: Signaling Operations
# https://docs.nvidia.com/nvshmem/api/gen/api/collectives.html
# ===----------------------------------------------------------------------=== #


@extern("nvshmemx_signal_op")
def nvshmemx_signal_op(
    sig_addr: UnsafePointer[UInt64, MutAnyOrigin],
    signal: UInt64,
    sig_op: c_int,
    pe: c_int,
) abi("C"):
    ...


def nvshmem_put_signal_nbi[
    dtype: DType
](
    dest: UnsafePointer[Scalar[dtype], _],
    source: UnsafePointer[Scalar[dtype], _],
    nelems: Int,
    sig_addr: UnsafePointer[UInt64, _],
    signal: UInt64,
    sig_op: c_int,
    pe: c_int,
):
    comptime symbol = _dtype_to_nvshmem_type[
        "nvshmem_", dtype, "_put_signal_nbi"
    ]()
    external_call[symbol, NoneType](
        dest, source, nelems, sig_addr, signal, sig_op, pe
    )


# ===----------------------------------------------------------------------=== #
# 10: Collective Communication
# https://docs.nvidia.com/nvshmem/api/gen/api/collective.html
# ===----------------------------------------------------------------------=== #


def nvshmem_sync_all():
    _get_nvshmem_function[
        "nvshmem_sync_all",
        def() thin -> NoneType,
    ]()()


def nvshmem_barrier_all():
    comptime if is_nvidia_gpu():
        external_call["nvshmem_barrier_all", NoneType]()
    else:
        _get_nvshmem_function[
            "nvshmem_barrier_all",
            def() thin -> NoneType,
        ]()()


def nvshmemx_barrier_all_on_stream(stream: CUstream):
    _get_nvshmem_function[
        "nvshmemx_barrier_all_on_stream",
        def(CUstream) thin -> NoneType,
    ]()(stream)


# ===----------------------------------------------------------------------=== #
# 11: Point-To-Point Synchronization
# https://docs.nvidia.com/nvshmem/api/gen/api/sync.html
# ===----------------------------------------------------------------------=== #


@extern("nvshmem_signal_wait_until")
def nvshmem_signal_wait_until(
    sig_addr: UnsafePointer[UInt64, MutAnyOrigin], cmp: c_int, cmp_value: UInt64
) abi("C"):
    ...


# ===----------------------------------------------------------------------=== #
# 12: Memory Ordering
# https://docs.nvidia.com/nvshmem/api/gen/api/ordering.html
# ===----------------------------------------------------------------------=== #


@extern("nvshmem_fence")
def nvshmem_fence() abi("C"):
    ...
