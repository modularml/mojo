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
"""Implements methods for querying the host target info.

You can import these APIs from the `sys` package. For example:

```mojo
from std.sys import CompilationTarget

print(CompilationTarget.is_x86())
```
"""

from std.collections.string.string_span import _get_kgen_string
from std.ffi import _external_call_const, external_call

comptime _TargetType = __mlir_type.`!kgen.target`


@always_inline("nodebug")
def _current_target() -> _TargetType:
    return __mlir_attr.`#kgen.param.expr<current_target> : !kgen.target`


struct CompilationTarget[value: _TargetType = _current_target()](
    TrivialRegisterPassable
):
    """A struct that provides information about a target architecture.

    This struct encapsulates various methods to query target-specific
    information such as architecture features, OS details, endianness, and
    memory characteristics.

    Parameters:
        value: The target architecture to query. Defaults to the current target.
    """

    def __init__(out self):
        """Initialize a `CompilationTarget` with the default target."""
        pass

    @always_inline("nodebug")
    @staticmethod
    def unsupported_target_error[
        *,
        operation: Optional[String] = None,
        note: Optional[String] = None,
    ]() -> Never:
        """Produces a constraint failure when called indicating that some
        operation is not supported by the current compilation target.

        Parameters:
            operation: Optional name of the operation that is not supported.
                Should be a function name or short description.
            note: Optional additional note to print.
        """

        comptime note_text = String(" Note: ", note.value() if note else "")
        comptime msg = "Current compilation target does not support"
        comptime op_text = String(
            " operation: ", operation.value(), "."
        ) if operation else " this operation."
        comptime assert False, String(msg, op_text, note_text)

    @always_inline("nodebug")
    @staticmethod
    def _has_feature[name: StaticString]() -> Bool:
        """Checks if the target has a specific feature.

        Parameters:
            name: The name of the feature to check.

        Returns:
            True if the target has the specified feature, False otherwise.
        """
        return __mlir_attr[
            `#kgen.param.expr<target_has_feature,`,
            Self.value,
            `,`,
            _get_kgen_string[name](),
            `> : i1`,
        ]

    @always_inline("nodebug")
    @staticmethod
    def _arch() -> StaticString:
        return StaticString(Self.__arch())

    @always_inline("nodebug")
    @staticmethod
    def __arch() -> __mlir_type.`!kgen.string`:
        """Get the target processor string from the compilation target.

        Returns:
            The CPU model for a host target (e.g., "znver3", "neoverse-n1") or
            the GPU architecture for an accelerator target (e.g., "sm_100a",
            "gfx942"). For the host architecture, see `__triple_arch()`.
        """
        return __mlir_attr[
            `#kgen.param.expr<target_get_field,`,
            Self.value,
            `, "arch" : !kgen.string`,
            `> : !kgen.string`,
        ]

    @staticmethod
    def _is_arch[name: StaticString]() -> Bool:
        """Helper function to check if the target processor is the same as given
        by the name.

        NOTE: This function is needed so that we don't compare the strings at
        compile time using `==`, which would lead to a recursions due to SIMD
        (and potentially many other things) depending on architecture checks.

        Parameters:
            name: The CPU model or GPU architecture to check against.

        Returns:
            True if the target processor is the same as the given name, False
            otherwise.
        """
        return __mlir_attr[
            `#kgen.param.identical<`,
            Self.__arch(),
            `, `,
            _get_kgen_string[name](),
            `> : !kgen.scalar<bool>`,
        ]

    @always_inline("nodebug")
    @staticmethod
    def __triple_arch() -> __mlir_type.`!kgen.string`:
        """Get the architecture of the compilation target's triple.

        Returns:
            The canonical LLVM architecture name (e.g., "x86_64", "aarch64").
            Equivalent spellings are canonicalized, so an `arm64-apple-darwin`
            triple reports "aarch64" and an `i686-*` triple reports "i386".
        """
        return __mlir_attr[
            `#kgen.param.expr<target_get_field,`,
            Self.value,
            `, "triple_arch" : !kgen.string`,
            `> : !kgen.string`,
        ]

    @staticmethod
    def _is_triple_arch[name: StaticString]() -> Bool:
        """Helper function to check if the target's architecture is the same as
        given by the name.

        NOTE: As with `_is_arch`, the comparison happens in the parameter
        expression rather than via `==` on strings, which would recurse through
        SIMD.

        Parameters:
            name: The canonical LLVM architecture name to check against.

        Returns:
            True if the target's architecture is the same as the given name,
            False otherwise.
        """
        return __mlir_attr[
            `#kgen.param.identical<`,
            Self.__triple_arch(),
            `, `,
            _get_kgen_string[name](),
            `> : !kgen.scalar<bool>`,
        ]

    @always_inline("nodebug")
    @staticmethod
    def _os() -> StaticString:
        var res = __mlir_attr[
            `#kgen.param.expr<target_get_field,`,
            Self.value,
            `, "os" : !kgen.string`,
            `> : !kgen.string`,
        ]
        return StaticString(res)

    @always_inline("nodebug")
    @staticmethod
    def default_compile_options() -> StaticString:
        """Returns the default compile options for the compilation target.

        Returns:
            The string of default compile options for the compilation target.
        """

        comptime if is_triple["nvptx64-nvidia-cuda", Self.value]():
            # TODO: use `is_nvidia_gpu` when moved to into this struct.
            return "target-abi=shortptr"
        else:
            return ""

    # Features

    @staticmethod
    def has_sse4() -> Bool:
        """Checks if the target supports SSE4 instructions.

        Returns:
            True if the target supports SSE4, False otherwise.
        """
        return Self._has_feature["sse4.1"]()

    @staticmethod
    def has_avx() -> Bool:
        """Returns True if the host system has AVX, otherwise returns False.

        Returns:
            True if the host system has AVX, otherwise returns False.
        """
        return Self._has_feature["avx"]()

    @staticmethod
    def has_avx2() -> Bool:
        """Returns True if the host system has AVX2, otherwise returns False.

        Returns:
            True if the host system has AVX2, otherwise returns False.
        """
        return Self._has_feature["avx2"]()

    @staticmethod
    def has_avx512f() -> Bool:
        """Returns True if the host system has AVX512, otherwise returns False.

        Returns:
            True if the host system has AVX512, otherwise returns False.
        """
        return Self._has_feature["avx512f"]()

    @staticmethod
    def has_intel_amx() -> Bool:
        """Returns True if the host system has Intel AMX support, otherwise returns
        False.

        Returns:
            True if the host system has Intel AMX and False otherwise.
        """
        return Self._has_feature["amx-tile"]()

    @staticmethod
    def has_fma() -> Bool:
        """Returns True if the target has FMA (Fused Multiply-Add) support,
        otherwise returns False.

        Returns:
            True if the target has FMA support, otherwise returns False.
        """
        return Self._has_feature["fma"]()

    @staticmethod
    def has_vnni() -> Bool:
        """Returns True if the target has avx512_vnni, otherwise returns False.

        Returns:
            True if the target has avx512_vnni, otherwise returns False.
        """
        return (
            Self._has_feature["avx512vnni"]() or Self._has_feature["avxvnni"]()
        )

    @staticmethod
    def has_neon() -> Bool:
        """Returns True if the target has Neon support, otherwise returns
        False.

        Returns:
            True if the target support the Neon instruction set.
        """
        return Self._has_feature["neon"]() or Self.is_apple_silicon()

    @staticmethod
    def has_neon_int8_dotprod() -> Bool:
        """Returns True if the target has the Neon int8 dot product extension,
        otherwise returns False.

        Returns:
            True if the target support the Neon int8 dot product extension and
            False otherwise.
        """
        return Self.has_neon() and Self._has_feature["dotprod"]()

    @staticmethod
    def has_neon_int8_matmul() -> Bool:
        """Returns True if the target has the Neon int8 matrix multiplication
        extension (I8MM), otherwise returns False.

        Returns:
            True if the target support the Neon int8 matrix multiplication
            extension (I8MM) and False otherwise.
        """
        return Self.has_neon() and Self._has_feature["i8mm"]()

    @staticmethod
    def has_riscv_extension[name: StaticString]() -> Bool:
        """Returns True if the target implements a RISC-V ISA extension.

        RISC-V has too many extensions to give each one a predicate, so this
        takes the extension name directly. Use the lowercase name LLVM gives the
        extension, which is the ISA-string spelling without the `rv32`/`rv64`
        prefix and without any version: `"m"`, `"a"`, `"f"`, `"d"`, `"c"`,
        `"v"`, `"zba"`, `"zicsr"`, and so on. `"g"` is not an extension; query
        its members individually.

        An extension implied by another one counts as present, so a target built
        with `d` also reports `f`, and one built with `a` also reports `zaamo`
        and `zalrsc`.

        Parameters:
            name: The extension to check for.

        Returns:
            True if the target is RISC-V and implements the extension, False
            otherwise. Always False on a non-RISC-V target.

        Example:

        ```mojo
        from std.sys import CompilationTarget

        def triple(x: Int32) -> Int32:
            comptime if CompilationTarget.has_riscv_extension["m"]():
                return x * 3
            return (x << 1) + x
        ```
        """
        comptime assert name.islower(), String(
            "RISC-V extension names are lowercase, got: ", name
        )
        return Self.is_riscv() and Self._has_feature[name]()

    # Platforms

    @staticmethod
    def is_x86() -> Bool:
        """Checks if the target is an x86 architecture.

        This is an architecture check, not an instruction-set check: use
        `has_sse4()`, `has_avx2()`, and friends to gate code on a specific
        extension.

        Returns:
            True if the target is 32- or 64-bit x86, False otherwise.
        """
        return (
            Self._is_triple_arch["x86_64"]() or Self._is_triple_arch["i386"]()
        )

    @staticmethod
    def is_arm() -> Bool:
        """Checks if the target is an ARM architecture.

        This is an architecture check, not an instruction-set check: use
        `has_neon()` and friends to gate code on a specific extension.

        Returns:
            True if the target is 32- or 64-bit ARM, False otherwise.
        """
        return (
            Self._is_triple_arch["aarch64"]()
            or Self._is_triple_arch["aarch64_be"]()
            or Self._is_triple_arch["aarch64_32"]()
            or Self._is_triple_arch["arm"]()
            or Self._is_triple_arch["armeb"]()
            or Self._is_triple_arch["thumb"]()
            or Self._is_triple_arch["thumbeb"]()
        )

    @staticmethod
    def is_riscv() -> Bool:
        """Checks if the target is a RISC-V architecture.

        This is an architecture check, not an instruction-set check: use
        `has_riscv_extension()` to gate code on a specific extension.

        Returns:
            True if the target is 32- or 64-bit RISC-V, False otherwise.
        """
        return Self.is_rv32() or Self.is_rv64()

    @staticmethod
    def is_rv32() -> Bool:
        """Checks if the target is a 32-bit RISC-V architecture.

        Returns:
            True if the target is 32-bit RISC-V, False otherwise.
        """
        return (
            Self._is_triple_arch["riscv32"]()
            or Self._is_triple_arch["riscv32be"]()
        )

    @staticmethod
    def is_rv64() -> Bool:
        """Checks if the target is a 64-bit RISC-V architecture.

        Returns:
            True if the target is 64-bit RISC-V, False otherwise.
        """
        return (
            Self._is_triple_arch["riscv64"]()
            or Self._is_triple_arch["riscv64be"]()
        )

    @staticmethod
    def is_apple_m1() -> Bool:
        """Check if the target is an Apple M1 system.

        Returns:
            True if the host system is an Apple M1, False otherwise.
        """
        return Self._is_arch["apple-m1"]()

    @staticmethod
    def is_apple_m2() -> Bool:
        """Check if the target is an Apple M2 system.

        Returns:
            True if the host system is an Apple M2, False otherwise.
        """
        return Self._is_arch["apple-m2"]()

    @staticmethod
    def is_apple_m3() -> Bool:
        """Check if the target is an Apple M3 system.

        Returns:
            True if the host system is an Apple M3, False otherwise.
        """
        return Self._is_arch["apple-m3"]()

    @staticmethod
    def is_apple_m4() -> Bool:
        """Check if the target is an Apple M4 system.

        Returns:
            True if the host system is an Apple M4, False otherwise.
        """
        return Self._is_arch["apple-m4"]()

    @staticmethod
    def is_apple_m5() -> Bool:
        """Check if the target is an Apple M5 system.

        Returns:
            True if the host system is an Apple M5, False otherwise.
        """
        return Self._is_arch["apple-m5"]()

    @staticmethod
    def is_apple_silicon() -> Bool:
        """Check if the host system is an Apple Silicon with AMX support.

        Returns:
            True if the host system is an Apple Silicon with AMX support, and
            False otherwise.
        """
        return (
            Self.is_apple_m1()
            or Self.is_apple_m2()
            or Self.is_apple_m3()
            or Self.is_apple_m4()
            or Self.is_apple_m5()
        )

    @staticmethod
    def is_neoverse_n1() -> Bool:
        """Returns True if the host system is a Neoverse N1 system, otherwise
        returns False.

        Returns:
            True if the host system is a Neoverse N1 system and False otherwise.
        """
        return Self._is_arch["neoverse-n1"]()

    # OS

    @staticmethod
    def is_linux() -> Bool:
        """Returns True if the host operating system is Linux.

        Returns:
            True if the host operating system is Linux and False otherwise.
        """
        return Self._os() == "linux"

    @staticmethod
    def is_macos() -> Bool:
        """Returns True if the host operating system is macOS.

        Returns:
            True if the host operating system is macOS and False otherwise.
        """
        return Self._os() in ["darwin", "macosx"]


def platform_map[
    T: Copyable,
    //,
    operation: Optional[String] = None,
    *,
    linux: Optional[T] = None,
    macos: Optional[T] = None,
]() -> T:
    """Helper for defining a compile time value depending
    on the current compilation target, raising a compilation
    error if trying to access the value on an unsupported target.

    Parameters:
        T: The type of the platform-specific value.
        operation: Optional operation name for error messages.
        linux: The value to use on Linux platforms.
        macos: The value to use on macOS platforms.

    Returns:
        The platform-specific value for the current target.

    Example:

    ```mojo
    from std.sys.info import platform_map

    comptime EDEADLK = platform_map["EDEADLK", linux=35, macos=11]()
    ```
    """

    comptime if CompilationTarget.is_macos() and macos:
        return materialize[macos.value()]()
    elif CompilationTarget.is_linux() and linux:
        return materialize[linux.value()]()
    else:
        CompilationTarget.unsupported_target_error[operation=operation]()


@always_inline("nodebug")
def _accelerator_arch() -> StaticString:
    """Returns the accelerator architecture string for the current target
    accelerator.

    If there is no accelerator on the system, this function returns an empty
    string.

    Returns:
        The accelerator architecture string for the current target accelerator.
    """
    return StaticString(
        __mlir_attr.`#kgen.param.expr<accelerator_arch> : !kgen.string`
    )


# ===-----------------------------------------------------------------------===#
# Vendor
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct Vendor(Equatable, TrivialRegisterPassable, Writable):
    """Represents GPU vendors.

    This struct provides identifiers for different GPU vendors and utility
    methods for comparison and string representation.

    The struct defines a constant for each vendor the stdlib knows about and a
    `NO_GPU` option for systems without an accelerator. It provides comparison
    operators and string conversion methods for vendor identification.

    It classifies the accelerator the compiler is targeting, which is what
    `has_amd_gpu_accelerator()`, `has_nvidia_gpu_accelerator()` and
    `has_apple_gpu_accelerator()` report. To identify a specific device, use
    `GPUInfo.api`.
    """

    var _value: Int8
    """The underlying integer value representing the vendor."""

    comptime NO_GPU = Self(0)
    """Represents no GPU or CPU-only execution."""

    comptime AMD_GPU = Self(1)
    """Represents AMD GPU vendor."""

    comptime NVIDIA_GPU = Self(2)
    """Represents NVIDIA GPU vendor."""

    comptime APPLE_GPU = Self(3)
    """Represents Apple GPU vendor."""

    def __eq__(self, other: Self) -> Bool:
        """Checks if two `Vendor` instances are equal.

        Args:
            other: The `Vendor` to compare with.

        Returns:
            True if vendors are equal, False otherwise.
        """
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        """Checks if two `Vendor` instances are not equal.

        Args:
            other: The `Vendor` to compare with.

        Returns:
            True if vendors are not equal, False otherwise.
        """
        return not (self == other)

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        """Writes vendor information to a writer.

        Args:
            writer: The writer to output vendor information to.
        """
        if self == Vendor.NO_GPU:
            writer.write("no_gpu")
            return
        if self == Vendor.AMD_GPU:
            writer.write("amd_gpu")
            return
        if self == Vendor.APPLE_GPU:
            writer.write("apple_gpu")
            return
        if self == Vendor.NVIDIA_GPU:
            writer.write("nvidia_gpu")
            return

        # Unreachable. Can't use `os.abort` here (`std.os` imports `std.sys`,
        # so it would cycle) nor `assert False` (elided under `-D ASSERT=none`);
        # trap directly, as `os.abort` itself does.
        __mlir_op.`llvm.intr.trap`()


@always_inline("nodebug")
def _vendor_from_arch[arch: StaticString]() -> Vendor:
    """Classifies an accelerator architecture string to its GPU `Vendor`.

    This is the single source of truth for arch-string -> vendor mapping. It
    recognizes bare architectures ("gfx950", "sm_90", "apple-m4"),
    vendor-prefixed forms ("amdgpu:gfx950", "amd:gfx950", "nvidia:sm_90",
    "metal:4"), and the generic "cuda" target, mapping each to the same vendor.

    Only vendor-relevant substrings are matched. Unknown or empty arch strings
    classify to `Vendor.NO_GPU`.

    Parameters:
        arch: The raw accelerator architecture string (e.g. from
            `_accelerator_arch()`).

    Returns:
        The `Vendor` the architecture belongs to, or `Vendor.NO_GPU` if it is
        empty or unrecognized.
    """
    # NOTE: use `in`-substring matching only (never `StaticString.startswith`,
    # which miscompiles in deep comptime instantiation contexts; see MOCO-4328).
    comptime if "amd" in arch or "gfx" in arch or "mi" in arch:
        return Vendor.AMD_GPU
    elif "nvidia" in arch or "sm" in arch or arch == "cuda":
        return Vendor.NVIDIA_GPU
    elif "metal" in arch or "apple" in arch:
        return Vendor.APPLE_GPU
    else:
        return Vendor.NO_GPU


@always_inline("nodebug")
def _triple_attr[
    target: _TargetType = _current_target()
]() -> __mlir_type.`!kgen.string`:
    return __mlir_attr[
        `#kgen.param.expr<target_get_field,`,
        target,
        `, "triple" : !kgen.string`,
        `> : !kgen.string`,
    ]


@always_inline("nodebug")
def is_triple[
    name: StringLiteral, target: _TargetType = _current_target()
]() -> Bool:
    """Returns True if the target triple of the compiler matches the input and
    False otherwise.

    Parameters:
      name: The name of the triple value.
      target: The triple value to be checked against.

    Returns:
        True if the triple matches and False otherwise.
    """
    return __mlir_attr[
        `#kgen.param.identical<`,
        _triple_attr[target](),
        `, `,
        name.value,
        `> : !kgen.scalar<bool>`,
    ]


@always_inline("nodebug")
def _is_nvidia_gpu_any[archs: List[StaticString]]() -> Bool:
    comptime for arch in archs:
        comptime if is_nvidia_gpu[arch]():
            return True
    return False


@always_inline("nodebug")
def _has_nvidia_gpu_any[archs: List[StaticString]]() -> Bool:
    comptime if not has_nvidia_gpu_accelerator():
        return False

    comptime for arch in archs:
        comptime if arch.removeprefix("sm_") in _accelerator_arch():
            return True
    return False


comptime _SM_80X_ARCHS: List[StaticString] = ["sm_80", "sm_86", "sm_89"]


@always_inline("nodebug")
def _is_sm_8x() -> Bool:
    return _is_nvidia_gpu_any[_SM_80X_ARCHS]()


@always_inline("nodebug")
def _has_sm_8x() -> Bool:
    return _has_nvidia_gpu_any[_SM_80X_ARCHS]()


comptime _SM_90X_ARCHS: List[StaticString] = ["sm_90", "sm_90a"]


@always_inline("nodebug")
def _is_sm_9x() -> Bool:
    return _is_nvidia_gpu_any[_SM_90X_ARCHS]()


@always_inline("nodebug")
def _has_sm_9x() -> Bool:
    return _has_nvidia_gpu_any[_SM_90X_ARCHS]()


comptime _SM_100X_ARCHS: List[StaticString] = ["sm_100", "sm_100a"]


@always_inline("nodebug")
def _is_sm_100x() -> Bool:
    return _is_nvidia_gpu_any[_SM_100X_ARCHS]()


@always_inline("nodebug")
def _has_sm_100x() -> Bool:
    return _has_nvidia_gpu_any[_SM_100X_ARCHS]()


comptime _SM_101X_ARCHS: List[StaticString] = ["sm_101", "sm_101a"]


@always_inline("nodebug")
def _is_sm_101x() -> Bool:
    return _is_nvidia_gpu_any[_SM_101X_ARCHS]()


@always_inline("nodebug")
def _has_sm_101x() -> Bool:
    return _has_nvidia_gpu_any[_SM_101X_ARCHS]()


comptime _SM_103X_ARCHS: List[StaticString] = ["sm_103", "sm_103a"]


@always_inline("nodebug")
def _is_sm_103x() -> Bool:
    return _is_nvidia_gpu_any[_SM_103X_ARCHS]()


@always_inline("nodebug")
def _has_sm_103x() -> Bool:
    return _has_nvidia_gpu_any[_SM_103X_ARCHS]()


comptime _SM_110X_ARCHS: List[StaticString] = ["sm_110", "sm_110a"]


@always_inline("nodebug")
def _is_sm_110x() -> Bool:
    return _is_nvidia_gpu_any[_SM_110X_ARCHS]()


@always_inline("nodebug")
def _has_sm_110x() -> Bool:
    return _has_nvidia_gpu_any[_SM_110X_ARCHS]()


comptime _SM_120X_ARCHS: List[StaticString] = ["sm_120", "sm_120a"]


@always_inline("nodebug")
def _is_sm_120x() -> Bool:
    return _is_nvidia_gpu_any[_SM_120X_ARCHS]()


@always_inline("nodebug")
def _has_sm_120x() -> Bool:
    return _has_nvidia_gpu_any[_SM_120X_ARCHS]()


comptime _SM_121X_ARCHS: List[StaticString] = ["sm_121", "sm_121a"]


@always_inline("nodebug")
def _is_sm_121x() -> Bool:
    return _is_nvidia_gpu_any[_SM_121X_ARCHS]()


@always_inline("nodebug")
def _has_sm_121x() -> Bool:
    return _has_nvidia_gpu_any[_SM_121X_ARCHS]()


@always_inline("nodebug")
def _has_blackwell_tcgen05() -> Bool:
    return _has_nvidia_gpu_any[
        _SM_100X_ARCHS + _SM_101X_ARCHS + _SM_103X_ARCHS
    ]()


@always_inline("nodebug")
def _is_sm_8x_or_newer() -> Bool:
    return _is_sm_8x() or _is_sm_9x_or_newer()


@always_inline("nodebug")
def _has_sm_8x_or_newer() -> Bool:
    return _has_sm_8x() or _has_sm_9x_or_newer()


@always_inline("nodebug")
def _is_sm_9x_or_newer() -> Bool:
    return _is_sm_9x() or _is_sm_100x_or_newer()


@always_inline("nodebug")
def _has_sm_9x_or_newer() -> Bool:
    return _has_sm_9x() or _has_sm_100x_or_newer()


@always_inline("nodebug")
def _is_sm_100x_or_newer() -> Bool:
    return _is_sm_100x() or _is_sm_103x() or _is_sm_110x_or_newer()


@always_inline("nodebug")
def _has_sm_100x_or_newer() -> Bool:
    return (
        _has_blackwell_tcgen05()
        or _has_sm_100x()
        or _has_sm_103x()
        or _has_sm_110x_or_newer()
    )


@always_inline("nodebug")
def _is_sm_110x_or_newer() -> Bool:
    return _is_sm_110x() or _is_sm_120x_or_newer()


@always_inline("nodebug")
def _has_sm_110x_or_newer() -> Bool:
    return _has_sm_110x() or _has_sm_120x_or_newer()


@always_inline("nodebug")
def _is_sm_120x_or_newer() -> Bool:
    return _is_sm_120x() or _is_sm_121x()


@always_inline("nodebug")
def _has_sm_120x_or_newer() -> Bool:
    return _has_sm_120x() or _has_sm_121x()


@always_inline("nodebug")
def is_apple_m5() -> Bool:
    """Returns True if the target is an Apple M5 GPU and False otherwise.

    Returns:
        True if the target is Apple M5 and False otherwise.
    """
    return is_apple_gpu() and CompilationTarget.is_apple_m5()


@always_inline("nodebug")
def is_apple_gpu() -> Bool:
    """Returns True if the target triple is for Apple GPU (Metal) and False otherwise.

    Returns:
        True if the triple target is Apple GPU and False otherwise.
    """
    return is_triple["air64-apple-macosx"]()


@always_inline("nodebug")
def is_apple_gpu[subarch: StaticString]() -> Bool:
    """Returns True if the target triple of the compiler is `air64-apple-macosx`
    and we are compiling for the specified sub-architecture and False otherwise.

    Parameters:
        subarch: The subarchitecture (e.g. sm_80).

    Returns:
        True if the triple target is cuda and False otherwise.
    """
    return is_apple_gpu() and CompilationTarget._is_arch[subarch]()


@always_inline("nodebug")
def is_nvidia_gpu() -> Bool:
    """Returns True if the target triple of the compiler is `nvptx64-nvidia-cuda`
    False otherwise.

    Returns:
        True if the triple target is cuda and False otherwise.
    """
    return is_triple["nvptx64-nvidia-cuda"]()


@always_inline("nodebug")
def is_nvidia_gpu[subarch: StaticString]() -> Bool:
    """Returns True if the target triple of the compiler is `nvptx64-nvidia-cuda`
    and we are compiling for the specified sub-architecture and False otherwise.

    Parameters:
        subarch: The subarchitecture (e.g. sm_80).

    Returns:
        True if the triple target is cuda and False otherwise.
    """
    return is_nvidia_gpu() and CompilationTarget._is_arch[subarch]()


comptime _AMD_GCN_ARCHS: List[StaticString] = [
    "gfx600",
    "gfx601",
    "gfx602",
    "gfx700",
    "gfx701",
    "gfx702",
    "gfx703",
    "gfx704",
    "gfx705",
    "gfx801",
    "gfx802",
    "gfx803",
    "gfx805",
    "gfx810",
    "gfx900",
    "gfx902",
    "gfx904",
    "gfx906",
    "gfx909",
]


@always_inline("nodebug")
def _is_amd_gcn() -> Bool:
    """Returns True if the target triple of the compiler is `amdgcn-amd-amdhsa`
    and we are compiling for any of the GCN (Graphics Core Next) architectures.

    Returns:
        True if GCN and False otherwise.
    """
    comptime for arch in _AMD_GCN_ARCHS:
        comptime if is_amd_gpu[arch]():
            return True
    return False


comptime _AMD_RDNA1_ARCHS: List[StaticString] = [
    "gfx1010",  # Navi 10 (RX 5700 XT/5700)
    "gfx1011",  # Navi 12
    "gfx1012",  # Navi 14 (RX 5500 XT/5500)
    "gfx1013",  # Navi 14
]


@always_inline("nodebug")
def _is_amd_rdna1() -> Bool:
    """Returns True if the target triple of the compiler is `amdgcn-amd-amdhsa`
    and we are compiling for the any of the Radeon RX 5000 series
    sub-architectures:

        gfx1010: Navi 10 (RX 5700 XT/5700)
        gfx1011: Navi 12
        gfx1012: Navi 14 (RX 5500 XT/5500)
        gfx1013: Navi 14

    Returns:
        True if the RDNA1 and False otherwise.
    """
    comptime for arch in _AMD_RDNA1_ARCHS:
        comptime if is_amd_gpu[arch]():
            return True
    return False


comptime _AMD_RDNA2_ARCHS: List[StaticString] = [
    "gfx1030",  # Navi 21 (RX 6900/6800)
    "gfx1031",  # Navi 22 (RX 6700)
    "gfx1032",  # Navi 23 (RX 6600)
    "gfx1033",  # Navi 24 (Van Gogh)
    "gfx1034",  # Navi 24
    "gfx1035",  # Rembrandt APU
    "gfx1036",  # Raphael APU
]


@always_inline("nodebug")
def _is_amd_rdna2() -> Bool:
    """Returns True if the target triple of the compiler is `amdgcn-amd-amdhsa`
    and we are compiling for the any of the Radeon RX 6000 series
    sub-architectures:

        gfx1030: Navi 21 (RX 6900/6800)
        gfx1031: Navi 22 (RX 6700)
        gfx1032: Navi 23 (RX 6600)
        gfx1033: Navi 24 (Van Gogh)
        gfx1034: Navi 24
        gfx1035: Rembrandt APU
        gfx1036: Raphael APU

    Returns:
        True if the RDNA2 and False otherwise.
    """
    comptime for arch in _AMD_RDNA2_ARCHS:
        comptime if is_amd_gpu[arch]():
            return True
    return False


comptime _AMD_RDNA3_ARCHS: List[StaticString] = [
    "gfx1100",  # Navi 31
    "gfx1101",  # Navi 32
    "gfx1102",  # Navi 33
    "gfx1103",  # Navi 34
    "gfx1150",  # Navi 41
    "gfx1151",  # Navi 42
    "gfx1152",  # Navi 43
    "gfx1153",  # Navi 44
]


@always_inline("nodebug")
def _is_amd_rdna3() -> Bool:
    """Returns True if the target triple of the compiler is `amdgcn-amd-amdhsa`
    and we are compiling for the any of the Radeon RX 7000 series
    sub-architectures:

        gfx1100: Navi 31
        gfx1101: Navi 32
        gfx1102: Navi 33
        gfx1103: Navi 34
        gfx1150: Navi 41
        gfx1151: Navi 42
        gfx1152: Navi 43
        gfx1153: Navi 44

    Returns:
        True if the RDNA3 and False otherwise.
    """
    comptime for arch in _AMD_RDNA3_ARCHS:
        comptime if is_amd_gpu[arch]():
            return True
    return False


@always_inline("nodebug")
def _is_amd_rdna4() -> Bool:
    return is_amd_gpu["gfx1200"]() or is_amd_gpu["gfx1201"]()


@always_inline("nodebug")
def _is_amd_rdna() -> Bool:
    return (
        _is_amd_rdna1() or _is_amd_rdna2() or _is_amd_rdna3() or _is_amd_rdna4()
    )


@always_inline("nodebug")
def _is_amd_rdna2_or_earlier() -> Bool:
    """Returns True if the target is GCN, RDNA1, or RDNA2.

    Returns:
        True if the GPU is GCN, RDNA1, or RDNA2, False otherwise.
    """
    return _is_amd_gcn() or _is_amd_rdna1() or _is_amd_rdna2()


@always_inline("nodebug")
def _is_amd_mi250x() -> Bool:
    return is_amd_gpu["gfx90a"]()


@always_inline("nodebug")
def _is_amd_mi300x() -> Bool:
    return is_amd_gpu["gfx942"]()


@always_inline("nodebug")
def _is_amd_mi355x() -> Bool:
    return is_amd_gpu["gfx950"]()


@always_inline("nodebug")
def _cdna_version() -> Int:
    comptime assert (
        _is_amd_mi250x() or _is_amd_mi300x() or _is_amd_mi355x()
    ), "querying the cdna version is only supported on AMD hardware"

    comptime if _is_amd_mi250x():
        return 2
    elif _is_amd_mi300x():
        return 3
    else:
        return 4


@always_inline("nodebug")
def _cdna_3_or_newer() -> Bool:
    comptime if _is_amd_cdna():
        return _cdna_version() >= 3
    return False


@always_inline("nodebug")
def _cdna_4_or_newer() -> Bool:
    comptime if _is_amd_cdna():
        return _cdna_version() >= 4
    return False


@always_inline("nodebug")
def _is_amd_cdna() -> Bool:
    return _is_amd_mi250x() or _is_amd_mi300x() or _is_amd_mi355x()


@always_inline("nodebug")
def is_amd_gpu() -> Bool:
    """Returns True if the target triple of the compiler is `amdgcn-amd-amdhsa`
    False otherwise.

    Returns:
        True if the triple target is amdgpu and False otherwise.
    """
    return is_triple["amdgcn-amd-amdhsa"]()


@always_inline("nodebug")
def is_amd_gpu[subarch: StaticString]() -> Bool:
    """Returns True if the target triple of the compiler is `amdgcn-amd-amdhsa`
    and we are compiling for the specified sub-architecture, False otherwise.

    Parameters:
        subarch: The AMD GPU sub-architecture to check for (e.g., "gfx90a").

    Returns:
        True if the triple target is amdgpu and False otherwise.
    """
    return is_amd_gpu() and CompilationTarget._is_arch[subarch]()


@always_inline("nodebug")
def is_gpu() -> Bool:
    """Returns True if the target triple is GPU and False otherwise.

    Returns:
        True if the triple target is GPU and False otherwise.
    """
    return is_nvidia_gpu() or is_amd_gpu() or is_apple_gpu()


@always_inline("nodebug")
def is_little_endian[target: _TargetType = _current_target()]() -> Bool:
    """Returns True if the target's endianness is little and False otherwise.

    Parameters:
        target: The target architecture.

    Returns:
        True if the target is little endian and False otherwise.
    """
    return __mlir_attr[
        `#kgen.param.identical<`,
        __mlir_attr[
            `#kgen.param.expr<target_get_field,`,
            target,
            `, "endianness" : !kgen.string`,
            `> : !kgen.string`,
        ],
        `,`,
        `"little" : !kgen.string`,
        `> : !kgen.scalar<bool>`,
    ]


@always_inline("nodebug")
def is_big_endian[target: _TargetType = _current_target()]() -> Bool:
    """Returns True if the target's endianness is big and False otherwise.

    Parameters:
        target: The target architecture.

    Returns:
        True if the target is big endian and False otherwise.
    """
    return __mlir_attr[
        `#kgen.param.identical<`,
        __mlir_attr[
            `#kgen.param.expr<target_get_field,`,
            target,
            `, "endianness" : !kgen.string`,
            `> : !kgen.string`,
        ],
        `,`,
        `"big" : !kgen.string`,
        `> : !kgen.scalar<bool>`,
    ]


@always_inline("nodebug")
def is_32bit[target: _TargetType = _current_target()]() -> Bool:
    """Returns True if the maximum integral value is 32 bit.

    Parameters:
        target: The target architecture.

    Returns:
        True if the maximum integral value is 32 bit, False otherwise.
    """
    return size_of[DType.int, target]() == size_of[DType.int32, target]()


@always_inline("nodebug")
def is_64bit[target: _TargetType = _current_target()]() -> Bool:
    """Returns True if the maximum integral value is 64 bit.

    Parameters:
        target: The target architecture.

    Returns:
        True if the maximum integral value is 64 bit, False otherwise.
    """
    return size_of[DType.int, target]() == size_of[DType.int64, target]()


@always_inline("nodebug")
def simd_bit_width[target: _TargetType = _current_target()]() -> Int:
    """Returns the vector size (in bits) of the specified target.

    Parameters:
        target: The target architecture.

    Returns:
        The vector size (in bits) of the specified target.
    """
    return Int(
        mlir_value=__mlir_attr[
            `#kgen.param.expr<target_get_field,`,
            target,
            `, "simd_bit_width" : !kgen.string`,
            `> : index`,
        ]
    )


@always_inline("nodebug")
def simd_byte_width[target: _TargetType = _current_target()]() -> Int:
    """Returns the vector size (in bytes) of the specified target.

    Parameters:
        target: The target architecture.

    Returns:
        The vector size (in bytes) of the host system.
    """
    comptime CHAR_BIT = 8
    return simd_bit_width[target]() // CHAR_BIT


@always_inline("nodebug")
def stdlib_plugin[target: _TargetType = _current_target()]() -> StaticString:
    """Returns the stdlib plugin name for the specified target.

    Parameters:
        target: The target architecture.

    Returns:
        The stdlib plugin name for the specified target.
    """
    return StaticString(
        __mlir_attr[
            `#kgen.param.expr<target_get_field,`,
            target,
            `, "stdlib_plugin" : !kgen.string`,
            `> : !kgen.string`,
        ]
    )


@always_inline("nodebug")
def size_of[type: AnyType, target: _TargetType = _current_target()]() -> Int:
    """Returns the size of (in bytes) of the type.

    The size includes any padding required by the type's alignment, so it is
    always a multiple of `align_of[type]()` and always matches the stride
    between adjacent elements of an array of the type.

    Parameters:
        type: The type in question.
        target: The target architecture.

    Returns:
        The size of the type in bytes.

    Example:
    ```mojo
    from std.sys.info import size_of
    def main() raises:
        print(
            size_of[UInt8]() == 1,
            size_of[UInt16]() == 2,
            size_of[Int32]() == 4,
            size_of[Float64]() == 8,
            size_of[
                SIMD[.uint8, 4]
            ]() == 4,
        )
    ```
    Note: `align_of` is in same module.
    """
    comptime mlir_type = __mlir_attr[
        `#kgen.param.expr<rebind, #kgen.type<!kgen.param<`,
        type,
        `>> : `,
        AnyType,
        `> : !kgen.type`,
    ]
    return Int(
        SIMDLength(
            mlir_value=__mlir_attr[
                `#kgen.param.expr<get_sizeof, #kgen.type<`,
                mlir_type,
                `> : !kgen.type,`,
                target,
                `> : index`,
            ]
        )
    )


@always_inline("nodebug")
def size_of[dtype: DType, target: _TargetType = _current_target()]() -> Int:
    """Returns the size of (in bytes) of the dtype.

    Parameters:
        dtype: The DType in question.
        target: The target architecture.

    Returns:
        The size of the dtype in bytes.
    """
    return Int(
        mlir_value=__mlir_attr[
            `#kgen.param.expr<get_sizeof, #kgen.type<`,
            Scalar[dtype]._mlir_type,
            `> : !kgen.type,`,
            target,
            `> : index`,
        ]
    )


@always_inline("builtin")
def align_of[type: AnyType, target: _TargetType = _current_target()]() -> Int:
    """Returns the align of (in bytes) of the type.

    Parameters:
        type: The type in question.
        target: The target architecture.

    Returns:
        The alignment of the type in bytes.
    """
    comptime mlir_type = __mlir_attr[
        `#kgen.param.expr<rebind, #kgen.type<!kgen.param<`,
        type,
        `>> : `,
        AnyType,
        `> : !kgen.type`,
    ]
    return Int(
        SIMDLength(
            mlir_value=__mlir_attr[
                `#kgen.param.expr<get_alignof, #kgen.type<`,
                +mlir_type,
                `> : !kgen.type,`,
                target,
                `> : index`,
            ]
        )
    )


@always_inline("nodebug")
def align_of[dtype: DType, target: _TargetType = _current_target()]() -> Int:
    """Returns the align of (in bytes) of the dtype.

    Parameters:
        dtype: The DType in question.
        target: The target architecture.

    Returns:
        The alignment of the dtype in bytes.
    """
    return Int(
        SIMDLength(
            mlir_value=__mlir_attr[
                `#kgen.param.expr<get_alignof, #kgen.type<`,
                Scalar[dtype]._mlir_type,
                `> : !kgen.type,`,
                target,
                `> : index`,
            ]
        )
    )


@always_inline("nodebug")
def bit_width_of[
    type: RegisterPassable, target: _TargetType = _current_target()
]() -> Int:
    """Returns the size of (in bits) of the type.

    Parameters:
        type: The type in question.
        target: The target architecture.

    Returns:
        The size of the type in bits.
    """
    comptime CHAR_BIT = 8
    return CHAR_BIT * size_of[type, target=target]()


@always_inline("nodebug")
def bit_width_of[
    dtype: DType, target: _TargetType = _current_target()
]() -> Int:
    """Returns the size of (in bits) of the dtype.

    Parameters:
        dtype: The type in question.
        target: The target architecture.

    Returns:
        The size of the dtype in bits.
    """
    return bit_width_of[Scalar[dtype]._mlir_type, target=target]()


@always_inline("nodebug")
def simd_width_of[
    type: RegisterPassable, target: _TargetType = _current_target()
]() -> Int:
    """Returns the vector size of the type on the host system.

    Parameters:
        type: The type in question.
        target: The target architecture.

    Returns:
        The vector size of the type on the host system.
    """
    return simd_bit_width[target]() // bit_width_of[type, target]()


@always_inline("nodebug")
def simd_width_of[
    dtype: DType, target: _TargetType = _current_target()
]() -> Int:
    """Returns the vector size of the type on the host system.

    Parameters:
        dtype: The DType in question.
        target: The target architecture.

    Returns:
        The vector size of the dtype on the host system.
    """
    return simd_width_of[Scalar[dtype]._mlir_type, target]()


@always_inline("nodebug")
def num_physical_cores() -> Int:
    """Returns the number of physical cores across all CPU sockets.


    Returns:
        Int: The number of physical cores on the system.
    """
    return _external_call_const["KGEN_CompilerRT_NumPhysicalCores", Int]()


@always_inline
def num_logical_cores() -> Int:
    """Returns the number of hardware threads, including hyperthreads across all
    CPU sockets.

    Returns:
        Int: The number of threads on the system.
    """
    return _external_call_const["KGEN_CompilerRT_NumLogicalCores", Int]()


@always_inline
def num_performance_cores() -> Int:
    """Returns the number of physical performance cores across all CPU sockets.
    If not known, returns the total number of physical cores.

    Returns:
        Int: The number of physical performance cores on the system.
    """
    return _external_call_const["KGEN_CompilerRT_NumPerformanceCores", Int]()


@always_inline
def _macos_version() raises -> Tuple[Int, Int, Int]:
    """Gets the macOS version.

    Returns:
        The version triple of macOS.
    """

    comptime assert (
        CompilationTarget.is_macos()
    ), "the operating system must be macOS"

    comptime INITIAL_CAPACITY = 32

    # Overallocate the string.
    var buf_len = Int(INITIAL_CAPACITY)
    var osver = String(unsafe_uninit_length=buf_len)

    var err = external_call["sysctlbyname", Int32](
        "kern.osproductversion".as_c_string_slice(),
        osver.unsafe_as_bytes_mut().unsafe_ptr(),
        Pointer(to=buf_len),
        Optional[Pointer[NoneType, MutAnyOrigin]](),
        Int(0),
    )
    if err:
        raise "Unable to query macOS version"

    # Truncate the string down to the actual length.
    osver.resize(buf_len)

    var major = 0
    var minor = 0
    var patch = 0

    if "." in osver:
        major = Int(osver[byte = : osver.find(".")])
        # TODO(MOCO-4373): split into two statements so the interior-origin
        # slice of `osver` is copied into a new String before `osver` is
        # reassigned; the single-statement form false-positives on
        # interior-origin invalidation.
        var rest_major = String(osver[byte = osver.find(".") + 1 :])
        osver = rest_major^

    if "." in osver:
        minor = Int(osver[byte = : osver.find(".")])
        var rest_minor = String(osver[byte = osver.find(".") + 1 :])
        osver = rest_minor^

    if "." in osver:
        patch = Int(osver[byte = : osver.find(".")])

    return (major, minor, patch)


# ===-----------------------------------------------------------------------===#
# Detect GPU on host side
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
def has_accelerator() -> Bool:
    """Returns True if the host system has an accelerator and False otherwise.

    Returns:
        True if the host system has an accelerator.
    """
    return is_gpu() or _accelerator_arch() != ""


@always_inline("nodebug")
def has_amd_gpu_accelerator() -> Bool:
    """Returns True if the host system has an AMD GPU and False otherwise.

    Returns:
        True if the host system has an AMD GPU.
    """
    # `_vendor_from_arch` is the single source of truth for classifying the raw
    # `--target-accelerator` value, whether it is a bare architecture
    # ("gfx950", "mi300x") or a vendor-prefixed form ("amdgpu:gfx950",
    # "amd:gfx950"), so a bare target behaves identically to a prefixed one.
    return (
        is_amd_gpu()
        or _vendor_from_arch[_accelerator_arch()]() == Vendor.AMD_GPU
    )


@always_inline("nodebug")
def has_amd_rdna_gpu_accelerator() -> Bool:
    """Returns True if the host system has an AMD RDNA GPU and False otherwise.

    Returns:
        True if the host system has an AMD RDNA GPU.
    """
    var arch = _accelerator_arch()
    return (
        _is_amd_rdna() or "gfx10" in arch or "gfx11" in arch or "gfx12" in arch
    )


@always_inline("nodebug")
def has_nvidia_gpu_accelerator() -> Bool:
    """Returns True if the host system has an NVIDIA GPU and False otherwise.

    Returns:
        True if the host system has an NVIDIA GPU.
    """
    # `_vendor_from_arch` is the single source of truth for classifying the raw
    # `--target-accelerator` value, whether it is a bare architecture ("sm_90"),
    # a vendor-prefixed form ("nvidia:sm_90"), or the generic "cuda", so a bare
    # target behaves identically to a prefixed one.
    return (
        is_nvidia_gpu()
        or _vendor_from_arch[_accelerator_arch()]() == Vendor.NVIDIA_GPU
    )


@always_inline("nodebug")
def has_nvidia_gpu_accelerator[subarch: String]() -> Bool:
    """Returns True if the host system has an NVIDIA GPU of the specified
    sub-architecture and False otherwise.

    Parameters:
        subarch: The NVIDIA GPU sub-architecture to check for (e.g., "sm_80").

    Returns:
        True if the host system has an NVIDIA GPU of the specified
        sub-architecture and False otherwise.
    """
    return is_nvidia_gpu[subarch]() or subarch in _accelerator_arch()


@always_inline("nodebug")
def has_nvidia_gpu_accelerator[subarchs: List[StaticString]]() -> Bool:
    """Returns True if the host system has an NVIDIA GPU of the specified
    sub-architecture and False otherwise.

    Parameters:
        subarchs: The NVIDIA GPU sub-architectures to check for (e.g., ["sm_80", "sm_86"]).

    Returns:
        True if the host system has an NVIDIA GPU of the specified
        sub-architecture and False otherwise.
    """
    return _is_nvidia_gpu_any[subarchs]() or _has_nvidia_gpu_any[subarchs]()


@always_inline("nodebug")
def has_apple_gpu_accelerator() -> Bool:
    """Returns True if the host system has a Metal GPU and False otherwise.

    Returns:
        True if the host system has a Metal GPU.
    """
    # `_vendor_from_arch` is the single source of truth for classifying the raw
    # `--target-accelerator` value, whether it is a bare architecture
    # ("apple-m4") or a vendor-prefixed form ("metal:4"), so a bare target
    # behaves identically to a prefixed one.
    return (
        is_apple_gpu()
        or _vendor_from_arch[_accelerator_arch()]() == Vendor.APPLE_GPU
    )
