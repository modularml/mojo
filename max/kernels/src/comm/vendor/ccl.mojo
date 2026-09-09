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
"""Provides Mojo FFI bindings for NCCL (NVIDIA) and RCCL (AMD) collective operations.

Selects and loads the correct vendor library at runtime: `librccl.so` on AMD
systems and `libnccl.so` on NVIDIA systems. Exposes allreduce, allgather, and
broadcast collectives, along with communicator initialization helpers and
availability probes.
"""

from std.sys import has_amd_gpu_accelerator, simd_width_of, size_of
from std.pathlib import Path
from max.algorithm import elementwise
from std.utils import IndexList
from std.ffi import _get_global_or_null, external_call
from std.ffi import _find_dylib
from std.ffi import _get_dylib_function as _ffi_get_dylib_function
from std.ffi import OwnedDLHandle, _Global
from std.collections.optional import Optional
from layout import TensorLayout, TileTensor
from std.memory.unsafe_pointer import unsafe_cast
from std.memory.alloc import Layout as AllocLayout
from max.gpu.host import DeviceContext, DeviceBuffer, get_gpu_target
from max.gpu.host._amdgpu_hip import HIP
from max.gpu.host._nvidia_cuda import CUDA
from comm import MAX_GPUS, Signal
from comm.allreduce import elementwise_epilogue_type
from max.gpu.primitives.grid_controls import PDLLevel
from std.utils.coord import Coord

comptime ncclComm_t = OptionalPointer[NoneType, MutUntrackedOrigin]


@fieldwise_init
struct ncclResult_t(Equatable, TrivialRegisterPassable, Writable):
    """Status code returned by NCCL/RCCL collective operations.

    Wraps the integer error code from the NCCL/RCCL C API. Use
    `ncclResult_t.ncclSuccess` (value 0) to check for a successful call.
    """

    var _value: Int32
    comptime ncclSuccess = Self(0)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ncclResult_t(", Int(self._value), ")")


@fieldwise_init
struct ncclRedOp_t(TrivialRegisterPassable):
    """Reduction operation selector for NCCL/RCCL collective calls.

    Wraps the `ncclRedOp_t` C enum. Only `ncclSum` (value 0) is currently
    used; other operations from the NCCL API may be added in the future.
    """

    var _value: Int32
    comptime ncclSum = Self(0)

    def __init__(out self, value: Int):
        self._value = Int32(value)


@fieldwise_init
struct ncclDataType_t(TrivialRegisterPassable):
    """Data-type selector for NCCL/RCCL collective calls.

    Wraps the `ncclDataType_t` C enum for the floating-point types supported by
    the collective bridge. Supported aliases: `ncclFloat16`, `ncclFloat32`, and
    `ncclBfloat16`.
    """

    var _value: Int32
    comptime ncclFloat16 = Self(6)
    comptime ncclFloat32 = Self(7)
    comptime ncclBfloat16 = Self(9)

    def __init__(out self, value: Int):
        self._value = Int32(value)


comptime RCCL_LIBRARY_PATHS: List[Path] = [
    "librccl.so",
    "librccl.so.1",
    "/opt/rocm/lib/librccl.so",
    "/opt/rocm/lib/librccl.so.1",
]


comptime NCCL_LIBRARY_PATHS: List[Path] = [
    "libnccl.so",
    "libnccl.so.2",
    "/usr/lib/x86_64-linux-gnu/libnccl.so",
    "/usr/lib/x86_64-linux-gnu/libnccl.so.2",
]


# Unified CCL loader (selects RCCL/NCCL at compile time)
def _init_ccl_dylib() -> OwnedDLHandle:
    comptime if has_amd_gpu_accelerator():
        return _find_dylib["RCCL"](materialize[RCCL_LIBRARY_PATHS]())
    else:
        return _find_dylib["NCCL"](materialize[NCCL_LIBRARY_PATHS]())


comptime CCL_LIBRARY = _Global["CCL_LIBRARY", _init_ccl_dylib]


@always_inline
def _get_ccl_function[
    func_name: StaticString, result_type: TrivialRegisterPassable
]() raises -> result_type:
    return _ffi_get_dylib_function[CCL_LIBRARY(), func_name, result_type]()


# Paired wrappers grouped RCCl/NCCL for comparison
struct _Group:
    def __init__(out self):
        pass

    def __enter__(self) raises:
        _check_ccl_ok(
            _get_ccl_function["ncclGroupStart", def() thin -> ncclResult_t]()()
        )

    def __exit__(self) raises:
        _check_ccl_ok(
            _get_ccl_function["ncclGroupEnd", def() thin -> ncclResult_t]()()
        )


def group() -> _Group:
    """Returns a context manager that groups NCCL/RCCL collective calls.

    Use as a `with` statement to bracket a series of collective API calls
    between `ncclGroupStart` and `ncclGroupEnd`, enabling the NCCL/RCCL
    library to fuse or pipeline them for better performance.

    Returns:
        A `_Group` context manager that calls `ncclGroupStart` on entry and
        `ncclGroupEnd` on exit.
    """
    return _Group()


def ncclCommInitAll(
    comms: UnsafePointer[ncclComm_t, _],
    ndev: Int,
    devlist: UnsafePointer[Int32, _],
) raises -> ncclResult_t:
    """Initializes NCCL/RCCL communicators for a set of GPUs.

    Thin FFI wrapper around `ncclCommInitAll`. Allocates one communicator per
    device in `devlist` and stores the handles in `comms`. Must be called from
    a single thread; concurrent calls for the same device set cause undefined
    behavior in the NCCL library.

    Args:
        comms: Output array of communicator handles; must have room for `ndev`
            entries.
        ndev: Number of GPUs to include in the communicator group.
        devlist: Array of CUDA/ROCm device IDs to include.

    Returns:
        `ncclResult_t.ncclSuccess` on success; a non-zero status on failure.

    Raises:
        If the CCL function symbol cannot be resolved from the vendor library.
    """
    return _get_ccl_function[
        "ncclCommInitAll",
        def(type_of(comms), Int, type_of(devlist)) thin -> ncclResult_t,
    ]()(comms, ndev, devlist)


@always_inline
def _ccl_allreduce(
    sendbuff: OpaquePointer,
    recvbuff: OpaquePointer,
    count: Int,
    datatype: ncclDataType_t,
    op: ncclRedOp_t,
    comm: ncclComm_t,
    ctx: DeviceContext,
) raises -> ncclResult_t:
    var stream_ptr = _ccl_stream_ptr(ctx)
    return _get_ccl_function[
        "ncclAllReduce",
        def(
            type_of(sendbuff),
            type_of(recvbuff),
            Int,
            ncclDataType_t,
            ncclRedOp_t,
            ncclComm_t,
            type_of(stream_ptr),
        ) thin -> ncclResult_t,
    ]()(sendbuff, recvbuff, count, datatype, op, comm, stream_ptr)


# === AllGather binding (unified) ===
@always_inline
def _ccl_allgather(
    sendbuff: OpaquePointer,
    recvbuff: OpaquePointer,
    count: Int,
    datatype: ncclDataType_t,
    comm: ncclComm_t,
    ctx: DeviceContext,
) raises -> ncclResult_t:
    var stream_ptr = _ccl_stream_ptr(ctx)
    return _get_ccl_function[
        "ncclAllGather",
        def(
            type_of(sendbuff),
            type_of(recvbuff),
            Int,
            ncclDataType_t,
            ncclComm_t,
            type_of(stream_ptr),
        ) thin -> ncclResult_t,
    ]()(sendbuff, recvbuff, count, datatype, comm, stream_ptr)


# === Broadcast binding (unified) ===
@always_inline
def _ccl_broadcast(
    sendbuff: OpaquePointer,
    recvbuff: OpaquePointer,
    count: Int,
    datatype: ncclDataType_t,
    root: Int,
    comm: ncclComm_t,
    ctx: DeviceContext,
) raises -> ncclResult_t:
    var stream_ptr = _ccl_stream_ptr(ctx)
    return _get_ccl_function[
        "ncclBroadcast",
        def(
            type_of(sendbuff),
            type_of(recvbuff),
            Int,
            ncclDataType_t,
            Int,
            ncclComm_t,
            type_of(stream_ptr),
        ) thin -> ncclResult_t,
    ]()(
        sendbuff,
        recvbuff,
        count,
        datatype,
        root,
        comm,
        stream_ptr,
    )


@always_inline
def _ccl_stream_ptr(
    ctx: DeviceContext,
) raises -> OptionalPointer[NoneType, UntrackedOrigin[mut=True]]:
    comptime if has_amd_gpu_accelerator():
        return unsafe_cast[Type=NoneType](HIP(ctx.stream()))
    else:
        return unsafe_cast[Type=NoneType](CUDA(ctx.stream()))


@fieldwise_init
struct Communicators(ImplicitlyCopyable):
    """Holds NCCL/RCCL communicator handles for a fixed set of GPUs.

    Stores one `ncclComm_t` handle per GPU (up to `MAX_GPUS`). Instances are
    initialized lazily by `_get_global_comms` and cached process-wide; call
    `init_comms()` from a single thread before using multi-threaded collectives
    to avoid the check-then-create race in `_get_global_comms`.
    """

    var ngpus: Int
    """The number of GPUs participating in the communicator group."""

    var comms: Array[ncclComm_t, MAX_GPUS]
    """Per-GPU communicator handles, valid for indices `0..ngpus-1`."""

    def __init__(out self, *, copy: Self):
        self.ngpus = copy.ngpus
        self.comms = copy.comms.copy()


def _dtype_to_ccl[dtype: DType]() raises -> ncclDataType_t:
    comptime if dtype == .float32:
        return ncclDataType_t.ncclFloat32
    elif dtype == .bfloat16:
        return ncclDataType_t.ncclBfloat16
    elif dtype == .float16:
        return ncclDataType_t.ncclFloat16

    raise Error("vendor_ccl: dtype not supported: ", dtype)


@always_inline
def _check_ccl_ok(status: ncclResult_t) raises:
    if status != ncclResult_t.ncclSuccess:
        raise Error("CCL call failed with status ", Int(status._value))


def _get_global_comms(ngpus: Int) raises -> Communicators:
    var NAME = String(t"COMM_VENDOR_CCL_{ngpus}")
    var global_ptr = _get_global_or_null(NAME)
    if global_ptr:
        return global_ptr.value().unsafe_bitcast[Communicators]()[]

    if ngpus > MAX_GPUS:
        raise Error("too many GPUs for CCL")

    var comms = Array[ncclComm_t, MAX_GPUS](fill={})
    var devlist = Array[Int32, MAX_GPUS](fill={})
    for i in range(ngpus):
        devlist[i] = Int32(i)

    _check_ccl_ok(
        ncclCommInitAll(comms.unsafe_ptr(), ngpus, devlist.unsafe_ptr())
    )

    var c = Communicators(ngpus=ngpus, comms=comms.copy())

    var ptr = alloc(AllocLayout[Communicators].single()).unsafe_leak()
    ptr.unsafe_write(c)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(NAME), ptr.bitcast[NoneType]()
    )
    return ptr[]


def init_comms(ngpus: Int) raises:
    """Pre-initialize NCCL/RCCL communicators.

    Must be called from a single thread before using allreduce
    from multiple threads. _get_global_comms has a check-then-create
    race: two threads seeing null simultaneously would both call
    ncclCommInitAll and one would leak its communicators.

    Raises:
        If the NCCL/RCCL communicator initialization fails.
    """
    _ = _get_global_comms(ngpus)


def wait_for_comms(ngpus: Int):
    """Spin-wait until communicators for ngpus have been initialized.

    Use from non-zero device threads while device 0 calls init_comms.
    """
    var NAME = String(t"COMM_VENDOR_CCL_{ngpus}")
    while not _get_global_or_null(NAME):
        pass


def allreduce[
    dtype: DType,
    in_layout: TensorLayout,
    in_origin: ImmOrigin,
    rank_sigs_origin: Origin[mut=True],
    out_layout: TensorLayout,
    out_origin: MutOrigin,
    //,
    ngpus: Int,
    output_lambda: Optional[elementwise_epilogue_type] = None,
    pdl_level: PDLLevel = PDLLevel(),
    *,
    use_multimem: Bool = False,
](
    input_tensors: Array[
        TileTensor[dtype, in_layout, in_origin], 1 if use_multimem else ngpus
    ],
    output_tensor: TileTensor[mut=True, dtype, out_layout, out_origin],
    rank_sigs: Array[MutPointer[Signal, rank_sigs_origin], MAX_GPUS],
    ctx: DeviceContext,
    _max_num_blocks: Optional[Int] = None,
) raises:
    """Per-GPU allreduce for use in multi-threaded contexts.

    Currently requires prior single-threaded call to init_comms, as thread-safe
    version not yet implemented.
    """
    comptime assert (
        not use_multimem
    ), "vendor_ccl allreduce does not support multimem path"
    # Determine this device's rank from its context id.
    var device_rank = Int(ctx.id())
    var count = input_tensors[0].num_elements()
    var dtype_ccl = _dtype_to_ccl[dtype]()
    var op = ncclRedOp_t.ncclSum
    var comms = _get_global_comms(ngpus)

    var input_tensor = input_tensors[0] if use_multimem else input_tensors[
        device_rank
    ]

    _check_ccl_ok(
        _ccl_allreduce(
            input_tensor._storage.bitcast[NoneType](),
            output_tensor._storage.bitcast[NoneType](),
            count,
            dtype_ccl,
            op,
            comms.comms[device_rank],
            ctx,
        )
    )

    comptime if output_lambda:
        comptime epilogue = output_lambda.value()
        comptime simd_size = simd_width_of[dtype, target=get_gpu_target()]()

        def epilogue_wrapper[
            simd_width: Int, alignment: Int = 1
        ](idx: Coord) {var}:
            var flat_idx = idx[0].value()
            var val = output_tensor.raw_load[
                width=simd_width,
                alignment=alignment * size_of[dtype](),
            ](flat_idx)
            epilogue[dtype, simd_width, alignment=alignment](
                output_tensor.layout.idx2crd(Int(flat_idx)),
                val,
            )

        elementwise[
            simd_size,
            target="gpu",
            _trace_description="ccl_epilogue",
        ](epilogue_wrapper, Coord(output_tensor.num_elements()), ctx)


def _is_ccl_symbol_available[name: StaticString]() -> Bool:
    # Resolve a CCL symbol by name from the appropriate vendor DSO.
    # We intentionally cast to a trivial signature and do not call it.
    try:
        _ = _get_ccl_function[name, def() thin -> ncclResult_t]()
        return True
    except:
        return False


def is_allreduce_available() -> Bool:
    """Reports whether the vendor CCL allreduce symbol is loadable at runtime.

    Probes the NCCL/RCCL shared library for the `ncclAllReduce` symbol without
    calling it. Returns `False` if the library is absent or the symbol cannot
    be resolved.

    Returns:
        `True` if `ncclAllReduce` is available, `False` otherwise.
    """
    return _is_ccl_symbol_available["ncclAllReduce"]()


def is_allgather_available() -> Bool:
    """Reports whether the vendor CCL allgather symbol is loadable at runtime.

    Probes the NCCL/RCCL shared library for the `ncclAllGather` symbol without
    calling it. Returns `False` if the library is absent or the symbol cannot
    be resolved.

    Returns:
        `True` if `ncclAllGather` is available, `False` otherwise.
    """
    return _is_ccl_symbol_available["ncclAllGather"]()


def is_broadcast_available() -> Bool:
    """Reports whether the vendor CCL broadcast symbol is loadable at runtime.

    Probes the NCCL/RCCL shared library for the `ncclBroadcast` symbol without
    calling it. Returns `False` if the library is absent or the symbol cannot
    be resolved.

    Returns:
        `True` if `ncclBroadcast` is available, `False` otherwise.
    """
    return _is_ccl_symbol_available["ncclBroadcast"]()


def allgather[
    dtype: DType,
    in_layout: TensorLayout,
    in_origin: ImmOrigin,
    out_layout: TensorLayout,
    out_origin: MutOrigin,
    //,
    ngpus: Int,
](
    inputs: Array[TileTensor[dtype, in_layout, in_origin], ngpus],
    outputs: Array[
        TileTensor[mut=True, dtype, out_layout, out_origin], ngpus * ngpus
    ],
    list_of_ctx: List[DeviceContext],
) raises:
    """Performs an allgather across all GPUs via the vendor CCL library.

    Each GPU contributes its local `inputs[i]` chunk; after the call every
    output slot `outputs[dev * ngpus + src]` on device `dev` holds a copy of
    the input from device `src`. Uses `ncclAllGather` (NVIDIA) or
    `rcclAllGather` (AMD) internally, wrapped inside an NCCL group for
    correct pipelining.

    Parameters:
        dtype: Element data type of all input and output tensors.
        in_layout: `TensorLayout` of each per-GPU input.
        in_origin: Origin tag for the input tensors.
        out_layout: `TensorLayout` of each per-GPU output slot.
        out_origin: Mutable origin tag for the output tensors.
        ngpus: Number of participating GPUs.

    Args:
        inputs: Per-GPU input `TileTensor`s; must all have the same element
            count.
        outputs: Flat array of `ngpus * ngpus` output `TileTensor`s. Slot
            `i * ngpus + j` on device `i` receives device `j`'s input.
        list_of_ctx: Device context for each GPU; length must equal `ngpus`.

    Raises:
        If `ngpus < 1` or `ngpus > MAX_GPUS`.
        If `len(list_of_ctx) != ngpus`.
        If any input element count differs from `inputs[0]`'s count.
        If the CCL collective call fails.
    """
    if ngpus < 1:
        raise Error("ngpus must be >= 1")
    if ngpus > MAX_GPUS:
        raise Error("too many GPUs")
    if len(list_of_ctx) != ngpus:
        raise Error("ctx count must match ngpus")

    var count = inputs[0].num_elements()
    for i in range(ngpus):
        if inputs[i].num_elements() != count:
            raise Error("vendor_ccl allgather requires equal per-rank counts")

    var dtype_nccl = _dtype_to_ccl[dtype]()
    var comms = _get_global_comms(ngpus)

    var recv_tmp = List[DeviceBuffer[dtype]](capacity=ngpus)
    for i in range(ngpus):
        recv_tmp.append(
            list_of_ctx[i].enqueue_create_buffer[dtype](ngpus * count)
        )

    with group():
        for i in range(ngpus):
            with list_of_ctx[i].push_context():
                _check_ccl_ok(
                    _ccl_allgather(
                        inputs[i]._storage.bitcast[NoneType](),
                        recv_tmp[i].unsafe_ptr().bitcast[NoneType](),
                        count,
                        dtype_nccl,
                        comms.comms[i],
                        list_of_ctx[i],
                    )
                )

    for dev in range(ngpus):
        var ctx = list_of_ctx[dev]
        for src in range(ngpus):
            var src_off = src * count
            var out_idx = dev * ngpus + src
            var dest_db = DeviceBuffer[dtype](
                ctx, outputs[out_idx]._storage, count, owning=False
            )
            var src_db = DeviceBuffer[dtype](
                ctx, recv_tmp[dev].unsafe_ptr() + src_off, count, owning=False
            )
            # API takes (dst, src)
            ctx.enqueue_copy(dest_db, src_db)


def broadcast[
    dtype: DType,
    in_layout: TensorLayout,
    in_origin: Origin,
    out_layout: TensorLayout,
    out_origin: MutOrigin,
    //,
    ngpus: Int,
    pdl_level: PDLLevel = PDLLevel(),
    use_multimem: Bool = False,
](
    input_tensor: TileTensor[dtype, in_layout, in_origin],
    output_tensor: TileTensor[mut=True, dtype, out_layout, out_origin],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    ctx: DeviceContext,
    root: Int,
    _max_num_blocks: Optional[Int] = None,
) raises:
    """Per-GPU broadcast for use in multi-threaded contexts.

    Currently requires prior single-threaded call to init_comms, as thread-safe
    version not yet implemented.
    """
    comptime assert (
        not use_multimem
    ), "vendor_ccl broadcast does not support multimem path"
    # Determine this device's rank from its context id.
    var device_rank = Int(ctx.id())
    var count = output_tensor.num_elements()
    var dtype_ccl = _dtype_to_ccl[dtype]()
    var comms = _get_global_comms(ngpus)

    _check_ccl_ok(
        _ccl_broadcast(
            input_tensor._storage.bitcast[NoneType](),
            output_tensor._storage.bitcast[NoneType](),
            count,
            dtype_ccl,
            root,
            comms.comms[device_rank],
            ctx,
        )
    )
