"""C POSIX types."""

from sys.info import is_64bit, os_is_windows
from sys.ffi import external_call
from os import abort
from utils import StaticTuple, StringSlice
from memory import memcpy, UnsafePointer

# ===----------------------------------------------------------------------=== #
# Base Types
# ===----------------------------------------------------------------------=== #


struct C:
    """C types. This assumes that the platform is 32 or 64 bit, and char is
    always 8 bit (POSIX standard).
    """

    alias char = Int8
    """Type: `char`. The signedness of `char` is platform specific. Most
    systems, including x86 GNU/Linux and Windows, use `signed char`, but those
    based on PowerPC and ARM processors typically use `unsigned char`."""
    alias s_char = Int8
    """Type: `signed char`."""
    alias u_char = UInt8
    """Type: `unsigned char`."""
    alias short = Int16
    """Type: `short`."""
    alias u_short = UInt16
    """Type: `unsigned short`."""
    alias int = Int32
    """Type: `int`."""
    alias u_int = UInt32
    """Type: `unsigned int`."""
    alias long = Scalar[_c_long_dtype()]
    """Type: `long`."""
    alias u_long = Scalar[_c_u_long_dtype()]
    """Type: `unsigned long`."""
    alias long_long = Int64
    """Type: `long long`."""
    alias u_long_long = UInt64
    """Type: `unsigned long long`."""
    alias float = Float32
    """Type: `float`."""
    alias double = Float64
    """Type: `double`."""
    alias void = Int8
    """Type: `void`."""
    alias NULL = UnsafePointer[Self.void]()
    """Constant: NULL pointer."""
    alias ptr_addr = Int
    """Type: A Pointer Address."""


alias size_t = UInt
"""Type: `size_t`."""
alias ssize_t = Int
"""Type: `ssize_t`."""


# ===----------------------------------------------------------------------=== #
# Utils
# ===----------------------------------------------------------------------=== #


fn _c_long_dtype() -> DType:
    # https://en.wikipedia.org/wiki/64-bit_computing#64-bit_data_models

    @parameter
    if is_64bit() and os_is_windows():
        return DType.int32  # LLP64
    elif is_64bit():
        return DType.int64  # LP64
    else:
        return DType.int32  # ILP32


fn _c_u_long_dtype() -> DType:
    # https://en.wikipedia.org/wiki/64-bit_computing#64-bit_data_models

    @parameter
    if is_64bit() and os_is_windows():
        return DType.uint32  # LLP64
    elif is_64bit():
        return DType.uint64  # LP64
    else:
        return DType.uint32  # ILP32


trait _UnsafePtrU8:
    fn unsafe_ptr(self) -> UnsafePointer[UInt8]:
        ...


@always_inline
fn char_ptr[T: _UnsafePtrU8](item: T) -> UnsafePointer[C.char]:
    """Get the `C.char` pointer.

    Parameters:
        T: The type.

    Args:
        item: The item.

    Returns:
        The pointer.
    """
    return item.unsafe_ptr().bitcast[C.char]()


@always_inline
fn char_ptr[T: AnyType](ptr: UnsafePointer[T]) -> UnsafePointer[C.char]:
    """Get the `C.char` pointer.

    Parameters:
        T: The type.

    Args:
        ptr: The pointer.

    Returns:
        The pointer.
    """
    return ptr.bitcast[C.char]()


fn char_ptr_to_string(s: UnsafePointer[C.char]) -> String:
    """Create a String **copying** a **null terminated** char pointer.

    Args:
        s: A pointer to a C string.

    Returns:
        The String.
    """
    idx = 0
    while s[idx] != 0:
        idx += 1
    idx += 1
    buf = UnsafePointer[UInt8].alloc(idx)
    memcpy(buf.bitcast[C.char](), s, idx)
    return String(ptr=buf, length=idx)


# ===----------------------------------------------------------------------=== #
# Networking Types
# ===----------------------------------------------------------------------=== #

alias sa_family_t = C.u_short
"""Type: `sa_family_t`."""
alias socklen_t = C.u_int
"""Type: `socklen_t`."""
alias in_addr_t = C.u_int
"""Type: `in_addr_t`."""
alias in_port_t = C.u_short
"""Type: `in_port_t`."""


@value
@register_passable("trivial")
struct in_addr:
    """Incoming IPv4 Socket Address."""

    var s_addr: in_addr_t
    """Source Address."""


@value
@register_passable("trivial")
struct in6_addr:
    """Incoming IPv6 Socket Address."""

    var s6_addr: StaticTuple[C.char, 16]
    """Source IPv6 Address."""


@value
@register_passable("trivial")
struct sockaddr:
    """Socket Address."""

    var sa_family: sa_family_t
    """Socket Address Family."""
    var sa_data: StaticTuple[C.char, 14]
    """Socket Address."""


@value
@register_passable("trivial")
struct sockaddr_in:
    """Incoming Socket Address."""

    var sin_family: sa_family_t
    """Socket Address Family."""
    var sin_port: in_port_t
    """Socket Address Port."""
    var sin_addr: in_addr
    """Socket Address."""
    var sin_zero: StaticTuple[C.char, 8]
    """Socket zero padding."""


@value
@register_passable("trivial")
struct sockaddr_in6:
    """Incoming IPv6 Socket Address."""

    var sin6_family: sa_family_t
    """Socket Address Family."""
    var sin6_port: in_port_t
    """Socket Address Port."""
    var sin6_flowinfo: C.u_int
    """Flow Information."""
    var sin6_addr: in6_addr
    """Socket Address."""
    var sin6_scope_id: C.u_int
    """Scope ID."""


@value
@register_passable("trivial")
struct addrinfo:
    """Address Information."""

    var ai_flags: C.int
    """Address Information Flags."""
    var ai_family: C.int
    """Address Family."""
    var ai_socktype: C.int
    """Socket Type."""
    var ai_protocol: C.int
    """Socket Protocol."""
    var ai_addrlen: socklen_t
    """Socket Address Length."""
    var ai_addr: UnsafePointer[sockaddr]
    """Address Information."""
    var ai_canonname: UnsafePointer[C.char]
    """Canon Name."""
    # FIXME: This should be UnsafePointer[addrinfo]
    var ai_next: UnsafePointer[C.void]
    """Next Address Information struct."""

    fn __init__(inout self):
        """Construct an empty addrinfo struct."""
        p0 = UnsafePointer[sockaddr]()
        self = Self(0, 0, 0, 0, 0, p0, UnsafePointer[C.char](), C.NULL)


# ===----------------------------------------------------------------------=== #
# File Types
# ===----------------------------------------------------------------------=== #

alias off_t = Int64
"""Type: `off_t`."""
alias mode_t = UInt32
"""Type: `mode_t`."""


@register_passable("trivial")
struct FILE:
    """Type: `FILE`."""

    pass
