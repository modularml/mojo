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

from std.collections import OptionalReg
from std.os import abort
from std.sys.info import size_of

# ===-----------------------------------------------------------------------===#
# Opaque handle types
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct Handle(Defaultable, Equatable, TrivialRegisterPassable):
    var _value: OptionalReg[OpaquePointer[MutUntrackedOrigin]]

    def __init__(out self):
        self._value = None


@fieldwise_init
struct TensorDescriptor(Defaultable, Equatable, TrivialRegisterPassable):
    var _value: OptionalReg[OpaquePointer[MutUntrackedOrigin]]

    def __init__(out self):
        self._value = None


@fieldwise_init
struct ConvolutionDescriptor(Defaultable, Equatable, TrivialRegisterPassable):
    var _value: OptionalReg[OpaquePointer[MutUntrackedOrigin]]

    def __init__(out self):
        self._value = None


# ===-----------------------------------------------------------------------===#
# Enum-like types
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct Status(Equatable, TrivialRegisterPassable, Writable):
    var _value: Int32
    comptime SUCCESS = Self(0)
    comptime NOT_INITIALIZED = Self(1)
    comptime INVALID_VALUE = Self(2)
    comptime BAD_PARAM = Self(3)
    comptime ALLOC_FAILED = Self(4)
    comptime INTERNAL_ERROR = Self(5)
    comptime NOT_SUPPORTED = Self(6)
    comptime UNKNOWN_ERROR = Self(7)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __int__(self) -> Int:
        return Int(self._value)

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.SUCCESS:
            return writer.write_string("SUCCESS")
        if self == Self.NOT_INITIALIZED:
            return writer.write_string("NOT_INITIALIZED")
        if self == Self.INVALID_VALUE:
            return writer.write_string("INVALID_VALUE")
        if self == Self.BAD_PARAM:
            return writer.write_string("BAD_PARAM")
        if self == Self.ALLOC_FAILED:
            return writer.write_string("ALLOC_FAILED")
        if self == Self.INTERNAL_ERROR:
            return writer.write_string("INTERNAL_ERROR")
        if self == Self.NOT_SUPPORTED:
            return writer.write_string("NOT_SUPPORTED")
        if self == Self.UNKNOWN_ERROR:
            return writer.write_string("UNKNOWN_ERROR")

        abort("unreachable: invalid Status entry")


@fieldwise_init
struct DataType(Equatable, TrivialRegisterPassable):
    var _value: Int32
    comptime HALF = Self(0)
    comptime FLOAT = Self(1)
    comptime INT32 = Self(2)
    comptime INT8 = Self(3)
    comptime BFLOAT16 = Self(5)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __init__(out self, dtype: DType) raises:
        if dtype == .float32:
            self = Self.FLOAT
        elif dtype == .float16:
            self = Self.HALF
        elif dtype == .bfloat16:
            self = Self.BFLOAT16
        else:
            raise Error(
                "the dtype '", dtype, "' is not currently handled by MIOpen"
            )

    def __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct TensorLayout(Equatable, TrivialRegisterPassable):
    var _value: Int32
    comptime NCHW = Self(0)
    comptime NHWC = Self(1)
    comptime NCDHW = Self(7)
    comptime NDHWC = Self(8)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct ConvolutionMode(Equatable, TrivialRegisterPassable):
    """MIOpen convolution modes.

    Note: MIOpen's miopenConvolution (0) is cross-correlation (no filter
    flip), which is the standard for DNN frameworks. There is NO separate
    cross-correlation mode like cuDNN has. TRANSPOSE (1) is deconvolution.
    """

    var _value: Int32
    comptime CONVOLUTION = Self(0)  # Cross-correlation (standard DNN conv)
    comptime TRANSPOSE = Self(1)  # Transpose conv (deconvolution)
    comptime GROUP_CONV = Self(2)  # Deprecated
    comptime DEPTHWISE = Self(3)  # Deprecated

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct ConvFwdAlgorithm(Equatable, TrivialRegisterPassable):
    var _value: Int32
    comptime GEMM = Self(0)
    comptime DIRECT = Self(1)
    comptime FFT = Self(2)
    comptime WINOGRAD = Self(3)
    comptime IMPLICIT_GEMM = Self(5)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __int__(self) -> Int:
        return Int(self._value)


# ===-----------------------------------------------------------------------===#
# Data structs
# ===-----------------------------------------------------------------------===#


struct ConvAlgoPerf(RegisterPassable):
    """Matches the C ABI layout of miopenConvAlgoPerf_t.
    C layout: union{fwd_algo/bwd_algo} (Int32), time (float), memory (size_t).
    Total size: 16 bytes with natural alignment padding.
    """

    var fwd_algo: Int32
    var time: Float32
    var memory: UInt64

    def __init__(out self):
        comptime assert size_of[Self]() == 16, "ConvAlgoPerf ABI size mismatch"
        self.fwd_algo = 0
        self.time = 0.0
        self.memory = 0


struct ConvSolution(RegisterPassable):
    """Matches the C ABI layout of miopenConvSolution_t.
    C layout: { float time, size_t workspace_size, uint64_t solution_id,
                miopenConvAlgorithm_t algorithm }.
    Total size: 32 bytes on 64-bit (4 + 4 pad + 8 + 8 + 4 + 4 pad = 32).
    """

    var time: Float32
    var _pad0: Int32
    var workspace_size: UInt64
    var solution_id: UInt64
    var algorithm: Int32
    var _pad1: Int32

    def __init__(out self):
        comptime assert size_of[Self]() == 32, "ConvSolution ABI size mismatch"
        self.time = 0.0
        self._pad0 = 0
        self.workspace_size = 0
        self.solution_id = 0
        self.algorithm = 0
        self._pad1 = 0


# ===-----------------------------------------------------------------------===#
# Find 2.0 / Problem API types
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct Problem(Defaultable, Equatable, TrivialRegisterPassable):
    var _value: OptionalReg[OpaquePointer[MutUntrackedOrigin]]

    def __init__(out self):
        self._value = None


@fieldwise_init
struct Solution(Defaultable, Equatable, TrivialRegisterPassable):
    var _value: OptionalReg[OpaquePointer[MutUntrackedOrigin]]

    def __init__(out self):
        self._value = None


@fieldwise_init
struct FindOptions(Defaultable, Equatable, TrivialRegisterPassable):
    var _value: OptionalReg[OpaquePointer[MutUntrackedOrigin]]

    def __init__(out self):
        self._value = None


@fieldwise_init
struct ProblemDirection(Equatable, TrivialRegisterPassable):
    var _value: Int32
    comptime FORWARD = Self(0)
    comptime BACKWARD = Self(1)
    comptime BACKWARD_WEIGHTS = Self(2)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct TensorArgumentId(Equatable, TrivialRegisterPassable):
    """Named tensor argument IDs for the Problem API."""

    var _value: Int32
    comptime CONV_X = Self(1)
    comptime CONV_W = Self(2)
    comptime CONV_Y = Self(3)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __int__(self) -> Int:
        return Int(self._value)


struct TensorArgument(RegisterPassable):
    """Tensor argument for miopenRunSolution.

    C layout: { miopenTensorArgumentId_t id, miopenTensorDescriptor_t* descriptor, void* buffer }
    Total: 24 bytes (4 + 4 pad + 8 + 8) on 64-bit.
    """

    var id: TensorArgumentId
    var _pad0: Int32
    var descriptor: Int  # miopenTensorDescriptor_t* (null = use problem desc)
    var buffer: Int  # void* (device pointer to tensor data)

    def __init__(out self, id: TensorArgumentId, buffer: OpaquePointer):
        """Create a tensor argument with null descriptor (uses problem desc)."""
        comptime assert (
            size_of[Self]() == 24
        ), "TensorArgument ABI size mismatch"
        self.id = id
        self._pad0 = 0
        self.descriptor = 0
        self.buffer = Int(buffer)

    def __init__(
        out self,
        id: TensorArgumentId,
        desc_ptr: Int,
        buffer: OpaquePointer,
    ):
        """Create a tensor argument with explicit descriptor override."""
        comptime assert (
            size_of[Self]() == 24
        ), "TensorArgument ABI size mismatch"
        self.id = id
        self._pad0 = 0
        self.descriptor = desc_ptr
        self.buffer = Int(buffer)
