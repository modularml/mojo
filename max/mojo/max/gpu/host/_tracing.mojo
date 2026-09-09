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

from std.os import abort
from std.pathlib import Path
from std.sys import (
    has_accelerator,
    has_amd_gpu_accelerator,
    has_nvidia_gpu_accelerator,
    size_of,
)
from std.ffi import _get_dylib_function as _ffi_get_dylib_function
from std.ffi import CStringSlice, _Global, OwnedDLHandle, _try_find_dylib
from std.sys.defines import get_defined_int

from std.utils.variant import Variant

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#

comptime CUDA_NVTX_LIBRARY_PATHS: List[Path] = [
    "libnvToolsExt.so",
    "/usr/local/cuda/lib64/libnvToolsExt.so",
    "/usr/lib/x86_64-linux-gnu/libnvToolsExt.so.1",
]
comptime ROCTX_LIBRARY_PATHS: List[Path] = [
    "librocprofiler-sdk-roctx.so",
    "/opt/rocm/lib/librocprofiler-sdk-roctx.so",
]

comptime LIBRARY_PATHS = CUDA_NVTX_LIBRARY_PATHS if has_nvidia_gpu_accelerator() else ROCTX_LIBRARY_PATHS


comptime _TraceType_OTHER = 0
comptime _TraceType_ASYNCRT = 1
comptime _TraceType_MEM = 2
comptime _TraceType_KERNEL = 3
comptime _TraceType_MAX = 4


@always_inline
def _setup_category(
    name_category: def(UInt32, CStringSlice[_]) thin abi("C") -> NoneType,
    value: Int,
    name: StaticString,
):
    name_category(UInt32(value), name.as_c_string_slice())


def _setup_categories(
    name_category: def(UInt32, CStringSlice[_]) thin abi("C") -> NoneType
):
    _setup_category(name_category, _TraceType_OTHER, "Other")
    _setup_category(name_category, _TraceType_ASYNCRT, "AsyncRT")
    _setup_category(name_category, _TraceType_MEM, "Memory")
    _setup_category(name_category, _TraceType_KERNEL, "Kernel")
    _setup_category(name_category, _TraceType_MAX, "Max")


def _on_error_msg() -> Error:
    return Error(
        (
            "Cannot find the GPU Tracing libraries. Please make sure that "
            "the library path is correctly set in one of the following paths ["
        ),
        ", ".join(materialize[LIBRARY_PATHS]()),
        (
            "]. You may need to make sure that you are using the non-slim"
            " version of the MAX container."
        ),
    )


comptime GPU_TRACING_LIBRARY = _Global[
    "GPU_TRACING_LIBRARY", _init_dylib, on_error_msg=_on_error_msg
]()


def _init_dylib() -> OwnedDLHandle:
    comptime if _is_disabled():
        abort("cannot load dylib when disabled")

    try:
        var dylib = _try_find_dylib["GPU tracing library"](
            materialize[LIBRARY_PATHS]()
        )

        comptime if has_nvidia_gpu_accelerator():
            _setup_categories(
                dylib._handle.get_function[
                    def(UInt32, CStringSlice[_]) thin abi("C") -> NoneType
                ]("nvtxNameCategoryA")
            )

        return dylib^
    except e:
        return OwnedDLHandle(unsafe_uninitialized=True)


@always_inline
def _get_dylib_function[
    func_name: StaticString, result_type: TrivialRegisterPassable
]() raises -> result_type:
    return _ffi_get_dylib_function[
        GPU_TRACING_LIBRARY,
        func_name,
        result_type,
    ]()


# ===-----------------------------------------------------------------------===#
# Types
# ===-----------------------------------------------------------------------===#


comptime RangeID = UInt64
comptime EventPayload = UInt64
comptime NVTXVersion = 2


@fieldwise_init
struct Color(Intable, TrivialRegisterPassable):
    var _value: Int

    comptime FORMAT = 1  # ARGB
    comptime MODULAR_PURPLE = Self(0xB5BAF5)
    comptime BLUE = Self(0x0000FF)
    comptime GREEN = Self(0x008000)
    comptime ORANGE = Self(0xFFA500)
    comptime PURPLE = Self(0x800080)
    comptime RED = Self(0xFF0000)
    comptime WHITE = Self(0xFFFFFF)
    comptime YELLOW = Self(0xFFFF00)

    def __init__(out self, colorname: StaticString):
        """Initialize Color from a StaticString color name.

        Args:
            colorname: The name of the color to use.
        """
        if colorname == "modular_purple":
            self = Color.MODULAR_PURPLE
        elif colorname == "blue":
            self = Color.BLUE
        elif colorname == "green":
            self = Color.GREEN
        elif colorname == "orange":
            self = Color.ORANGE
        elif colorname == "purple":
            self = Color.PURPLE
        elif colorname == "red":
            self = Color.RED
        elif colorname == "white":
            self = Color.WHITE
        elif colorname == "yellow":
            self = Color.YELLOW
        else:
            # Default to MODULAR_PURPLE for unknown color names
            self = Color.MODULAR_PURPLE

    def __int__(self) -> Int:
        return self._value


@fieldwise_init
struct _C_EventAttributes[message_origin: ImmOrigin](TrivialRegisterPassable):
    var version: UInt16
    """Version flag of the structure."""

    var size: UInt16
    """Size of the structure."""

    var category: UInt32
    """ID of the category the event is assigned to."""

    var color_type: Int32
    """Color type specified in this attribute structure."""

    var color: UInt32
    """Color assigned to this event."""

    var payload_type: Int32
    """Payload type specified in this attribute structure."""

    var _reserved: Int32
    """Reserved."""

    var event_payload: EventPayload
    """Payload assigned to this event."""

    var message_type: Int32
    """Message type specified in this attribute structure."""

    var message: CStringSlice[Self.message_origin]
    """Message assigned to this attribute structure."""


def c_event_attrs_ffi(
    attrs: _C_EventAttributes[_],
) -> _C_EventAttributes[ImmUntrackedOrigin]:
    return rebind[_C_EventAttributes[ImmUntrackedOrigin]](attrs)


@always_inline
def color_from_category(category: Int) -> Color:
    if category == _TraceType_MAX:
        return Color.MODULAR_PURPLE
    if category == _TraceType_KERNEL:
        return Color.GREEN
    if category == _TraceType_ASYNCRT:
        return Color.ORANGE
    if category == _TraceType_MEM:
        return Color.RED
    return Color.PURPLE


struct EventAttributes[message_origin: ImmOrigin](TrivialRegisterPassable):
    var _value: _C_EventAttributes[Self.message_origin]

    @always_inline
    def __init__(
        out self,
        *,
        message: CStringSlice[Self.message_origin],
        category: Int = _TraceType_MAX,
        color: Optional[Color] = None,
    ):
        comptime ASCII = 1
        var resolved_color: Color
        if color:
            resolved_color = color.value()
        else:
            resolved_color = color_from_category(category)
        self._value = _C_EventAttributes(
            version=NVTXVersion,
            size=UInt16(size_of[_C_EventAttributes[ImmUntrackedOrigin]]()),
            category=UInt32(category),
            color_type=Color.FORMAT,
            color=UInt32(Int(resolved_color)),
            payload_type=0,
            _reserved=0,
            event_payload=0,
            message_type=ASCII,
            message=message,
        )


struct _dylib_function[fn_name: StaticString, fn_type: TrivialRegisterPassable](
    TrivialRegisterPassable
):
    @staticmethod
    def load() raises -> Self.fn_type:
        return _get_dylib_function[Self.fn_name, Self.fn_type]()


# ===-----------------------------------------------------------------------===#
# NVTX Bindings
# ===-----------------------------------------------------------------------===#

# NVTX_DECLSPEC void NVTX_API nvtxMarkEx(const nvtxEventAttributes_t* eventAttrib);
comptime _nvtxMarkEx = _dylib_function[
    "nvtxMarkEx",
    def(
        Pointer[_C_EventAttributes[ImmUntrackedOrigin], ImmutAnyOrigin]
    ) thin -> NoneType,
]

# NVTX_DECLSPEC nvtxRangeId_t NVTX_API nvtxRangeStartEx(const nvtxEventAttributes_t* eventAttrib);
comptime _nvtxRangeStartEx = _dylib_function[
    "nvtxRangeStartEx",
    def(
        Pointer[_C_EventAttributes[ImmUntrackedOrigin], ImmutAnyOrigin]
    ) thin -> RangeID,
]

# NVTX_DECLSPEC void NVTX_API nvtxRangeEnd(nvtxRangeId_t id);
comptime _nvtxRangeEnd = _dylib_function[
    "nvtxRangeEnd", def(RangeID) thin -> NoneType
]

# NVTX_DECLSPEC int NVTX_API nvtxRangePushEx(const nvtxEventAttributes_t* eventAttrib);
comptime _nvtxRangePushEx = _dylib_function[
    "nvtxRangePushEx",
    def(
        Pointer[_C_EventAttributes[ImmUntrackedOrigin], ImmutAnyOrigin]
    ) thin -> Int32,
]

# NVTX_DECLSPEC int NVTX_API nvtxRangePop(void);
comptime _nvtxRangePop = _dylib_function["nvtxRangePop", def() thin -> Int32]


# ===-----------------------------------------------------------------------===#
# ROCTX Bindings
# ===-----------------------------------------------------------------------===#

# ROCTX_API void roctxMarkA(const char* message) ROCTX_VERSION_4_1;
comptime _roctxMarkA = _dylib_function[
    "roctxMarkA", def(CStringSlice[ImmutAnyOrigin]) thin -> NoneType
]

# ROCTX_API int roctxRangePushA(const char* message) ROCTX_VERSION_4_1;
comptime _roctxRangePushA = _dylib_function[
    "roctxRangePushA", def(CStringSlice[ImmutAnyOrigin]) thin -> Int32
]

# ROCTX_API int roctxRangePop() ROCTX_VERSION_4_1;
comptime _roctxRangePop = _dylib_function["roctxRangePop", def() thin -> Int32]
# ROCTX_API roctx_range_id_t roctxRangeStartA(const char* message)
comptime _roctxRangeStartA = _dylib_function[
    "roctxRangeStartA",
    def(CStringSlice[ImmutAnyOrigin]) thin -> RangeID,
]

# ROCTX_API void roctxRangeStop(roctx_range_id_t id) ROCTX_VERSION_4_1;
comptime _roctxRangeStop = _dylib_function[
    "roctxRangeStop", def(RangeID) thin -> NoneType
]

# ===-----------------------------------------------------------------------===#
# Bindings
# ===-----------------------------------------------------------------------===#


struct _Mark:
    var _fn: Variant[_nvtxMarkEx.fn_type, _roctxMarkA.fn_type]

    def __init__(out self) raises:
        comptime if has_nvidia_gpu_accelerator():
            self._fn = _nvtxMarkEx.load()
        else:
            self._fn = _roctxMarkA.load()

    def __call__(self, val: _C_EventAttributes[_]):
        comptime assert has_nvidia_gpu_accelerator()
        var attrs = c_event_attrs_ffi(val)
        self._fn[_nvtxMarkEx.fn_type](Pointer(to=attrs).as_unsafe_any_origin())

    def __call__(self, val: CStringSlice[_]):
        comptime assert has_amd_gpu_accelerator()
        self._fn[_roctxMarkA.fn_type](val.as_unsafe_any_origin())


struct _RangeStart:
    var _fn: Variant[_nvtxRangeStartEx.fn_type, _roctxRangeStartA.fn_type]

    def __init__(out self) raises:
        comptime if has_nvidia_gpu_accelerator():
            self._fn = _nvtxRangeStartEx.load()
        else:
            self._fn = _roctxRangeStartA.load()

    def __call__(
        self,
        val: _C_EventAttributes[_],
    ) -> RangeID:
        comptime assert has_nvidia_gpu_accelerator()
        var attrs = c_event_attrs_ffi(val)
        return self._fn[_nvtxRangeStartEx.fn_type](
            Pointer(to=attrs).as_unsafe_any_origin()
        )

    def __call__(self, val: CStringSlice[_]) -> RangeID:
        comptime assert has_amd_gpu_accelerator()
        return self._fn[_roctxRangeStartA.fn_type](val.as_unsafe_any_origin())


struct _RangeEnd:
    var _fn: def(RangeID) thin -> NoneType

    def __init__(out self) raises:
        comptime if has_nvidia_gpu_accelerator():
            self._fn = _nvtxRangeEnd.load()
        else:
            self._fn = _roctxRangeStop.load()

    def __call__(self, val: RangeID):
        self._fn(val)


struct _RangePush:
    var _fn: Variant[_nvtxRangePushEx.fn_type, _roctxRangePushA.fn_type]

    def __init__(out self) raises:
        comptime if has_nvidia_gpu_accelerator():
            self._fn = _nvtxRangePushEx.load()
        else:
            self._fn = _roctxRangePushA.load()

    def __call__(self, val: _C_EventAttributes[_]) -> Int32:
        comptime assert has_nvidia_gpu_accelerator()
        var attrs = c_event_attrs_ffi(val)
        return self._fn[_nvtxRangePushEx.fn_type](
            Pointer(to=attrs).as_unsafe_any_origin()
        )

    def __call__(self, val: CStringSlice[_]) -> Int32:
        comptime assert has_amd_gpu_accelerator()
        return self._fn[_roctxRangePushA.fn_type](val.as_unsafe_any_origin())


struct _RangePop:
    var _fn: _nvtxRangePop.fn_type

    def __init__(out self) raises:
        comptime if has_nvidia_gpu_accelerator():
            self._fn = _nvtxRangePop.load()
        else:
            self._fn = _roctxRangePop.load()

    def __call__(self) -> Int32:
        return self._fn()


# ===-----------------------------------------------------------------------===#
# Functions
# ===-----------------------------------------------------------------------===#


def _is_enabled_details() -> Bool:
    return (
        has_accelerator()
        and get_defined_int["MODULAR_ENABLE_GPU_PROFILING_DETAILED", 0]() == 1
    )


def _is_enabled() -> Bool:
    return has_accelerator() and (
        get_defined_int["MODULAR_ENABLE_GPU_PROFILING", 0]() == 1
        or _is_enabled_details()
    )


def _is_disabled() -> Bool:
    return not _is_enabled()


@always_inline
def _start_range(
    *,
    var message: String = "",
    category: Int = _TraceType_MAX,
    color: Optional[Color] = None,
) raises -> RangeID:
    comptime if _is_disabled():
        return 0

    comptime if has_nvidia_gpu_accelerator():
        var info = EventAttributes(
            message=message.as_c_string_slice(), color=color, category=category
        )
        var result = _RangeStart()(info._value)
        return result
    else:
        return _RangeStart()(message.as_c_string_slice())


@always_inline
def _end_range(id: RangeID) raises:
    comptime if _is_disabled():
        return
    _RangeEnd()(id)


@always_inline
def _mark(
    *,
    var message: String = "",
    color: Optional[Color] = None,
    category: Int = _TraceType_MAX,
) raises:
    comptime if _is_disabled():
        return

    comptime if has_nvidia_gpu_accelerator():
        var info = EventAttributes(
            message=message.as_c_string_slice(), color=color, category=category
        )
        _Mark()(info._value)
    else:
        _Mark()(message.as_c_string_slice())
