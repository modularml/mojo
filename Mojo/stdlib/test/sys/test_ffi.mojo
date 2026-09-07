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

from std.os.path import realpath
from std.ffi import ErrNo, RTLD, get_errno, set_errno
from std.sys.info import CompilationTarget

from std.testing import assert_equal, assert_raises
from std.testing import TestSuite

comptime error_message_linux: List[Tuple[ErrNo, String]] = [
    (ErrNo.SUCCESS, "Success"),
    (ErrNo.EPERM, "Operation not permitted"),
    (ErrNo.ENOENT, "No such file or directory"),
    (ErrNo.ESRCH, "No such process"),
    (ErrNo.EINTR, "Interrupted system call"),
    (ErrNo.EIO, "Input/output error"),
    (ErrNo.ENXIO, "No such device or address"),
    (ErrNo.E2BIG, "Argument list too long"),
    (ErrNo.ENOEXEC, "Exec format error"),
    (ErrNo.EBADF, "Bad file descriptor"),
    (ErrNo.ECHILD, "No child processes"),
    (ErrNo.EAGAIN, "Resource temporarily unavailable"),
    (ErrNo.EWOULDBLOCK, "Resource temporarily unavailable"),
    (ErrNo.ENOMEM, "Cannot allocate memory"),
    (ErrNo.EACCES, "Permission denied"),
    (ErrNo.EFAULT, "Bad address"),
    (ErrNo.ENOTBLK, "Block device required"),
    (ErrNo.EBUSY, "Device or resource busy"),
    (ErrNo.EEXIST, "File exists"),
    (ErrNo.EXDEV, "Invalid cross-device link"),
    (ErrNo.ENODEV, "No such device"),
    (ErrNo.ENOTDIR, "Not a directory"),
    (ErrNo.EISDIR, "Is a directory"),
    (ErrNo.EINVAL, "Invalid argument"),
    (ErrNo.ENFILE, "Too many open files in system"),
    (ErrNo.EMFILE, "Too many open files"),
    (ErrNo.ENOTTY, "Inappropriate ioctl for device"),
    (ErrNo.ETXTBSY, "Text file busy"),
    (ErrNo.EFBIG, "File too large"),
    (ErrNo.ENOSPC, "No space left on device"),
    (ErrNo.ESPIPE, "Illegal seek"),
    (ErrNo.EROFS, "Read-only file system"),
    (ErrNo.EMLINK, "Too many links"),
    (ErrNo.EPIPE, "Broken pipe"),
    (ErrNo.EDOM, "Numerical argument out of domain"),
    (ErrNo.ERANGE, "Numerical result out of range"),
    (ErrNo.EDEADLK, "Resource deadlock avoided"),
    (ErrNo.ENAMETOOLONG, "File name too long"),
    (ErrNo.ENOLCK, "No locks available"),
    (ErrNo.ENOSYS, "Function not implemented"),
    (ErrNo.ENOTEMPTY, "Directory not empty"),
    (ErrNo.ELOOP, "Too many levels of symbolic links"),
    (ErrNo.ENOMSG, "No message of desired type"),
    (ErrNo.EIDRM, "Identifier removed"),
    (ErrNo.ECHRNG, "Channel number out of range"),
    (ErrNo.EL2NSYNC, "Level 2 not synchronized"),
    (ErrNo.EL3HLT, "Level 3 halted"),
    (ErrNo.EL3RST, "Level 3 reset"),
    (ErrNo.ELNRNG, "Link number out of range"),
    (ErrNo.EUNATCH, "Protocol driver not attached"),
    (ErrNo.ENOCSI, "No CSI structure available"),
    (ErrNo.EL2HLT, "Level 2 halted"),
    (ErrNo.EBADE, "Invalid exchange"),
    (ErrNo.EBADR, "Invalid request descriptor"),
    (ErrNo.EXFULL, "Exchange full"),
    (ErrNo.ENOANO, "No anode"),
    (ErrNo.EBADRQC, "Invalid request code"),
    (ErrNo.EBADSLT, "Invalid slot"),
    (ErrNo.EBFONT, "Bad font file format"),
    (ErrNo.ENOSTR, "Device not a stream"),
    (ErrNo.ENODATA, "No data available"),
    (ErrNo.ETIME, "Timer expired"),
    (ErrNo.ENOSR, "Out of streams resources"),
    (ErrNo.ENONET, "Machine is not on the network"),
    (ErrNo.ENOPKG, "Package not installed"),
    (ErrNo.EREMOTE, "Object is remote"),
    (ErrNo.ENOLINK, "Link has been severed"),
    (ErrNo.EADV, "Advertise error"),
    (ErrNo.ESRMNT, "Srmount error"),
    (ErrNo.ECOMM, "Communication error on send"),
    (ErrNo.EPROTO, "Protocol error"),
    (ErrNo.EMULTIHOP, "Multihop attempted"),
    (ErrNo.EDOTDOT, "RFS specific error"),
    (ErrNo.EBADMSG, "Bad message"),
    (ErrNo.EOVERFLOW, "Value too large for defined data type"),
    (ErrNo.ENOTUNIQ, "Name not unique on network"),
    (ErrNo.EBADFD, "File descriptor in bad state"),
    (ErrNo.EREMCHG, "Remote address changed"),
    (ErrNo.ELIBACC, "Can not access a needed shared library"),
    (ErrNo.ELIBBAD, "Accessing a corrupted shared library"),
    (ErrNo.ELIBSCN, ".lib section in a.out corrupted"),
    (ErrNo.ELIBMAX, "Attempting to link in too many shared libraries"),
    (ErrNo.ELIBEXEC, "Cannot exec a shared library directly"),
    (ErrNo.EILSEQ, "Invalid or incomplete multibyte or wide character"),
    (ErrNo.ERESTART, "Interrupted system call should be restarted"),
    (ErrNo.ESTRPIPE, "Streams pipe error"),
    (ErrNo.EUSERS, "Too many users"),
    (ErrNo.ENOTSOCK, "Socket operation on non-socket"),
    (ErrNo.EDESTADDRREQ, "Destination address required"),
    (ErrNo.EMSGSIZE, "Message too long"),
    (ErrNo.EPROTOTYPE, "Protocol wrong type for socket"),
    (ErrNo.ENOPROTOOPT, "Protocol not available"),
    (ErrNo.EPROTONOSUPPORT, "Protocol not supported"),
    (ErrNo.ESOCKTNOSUPPORT, "Socket type not supported"),
    (ErrNo.EOPNOTSUPP, "Operation not supported"),
    (ErrNo.EPFNOSUPPORT, "Protocol family not supported"),
    (ErrNo.EAFNOSUPPORT, "Address family not supported by protocol"),
    (ErrNo.EADDRINUSE, "Address already in use"),
    (ErrNo.EADDRNOTAVAIL, "Cannot assign requested address"),
    (ErrNo.ENETDOWN, "Network is down"),
    (ErrNo.ENETUNREACH, "Network is unreachable"),
    (ErrNo.ENETRESET, "Network dropped connection on reset"),
    (ErrNo.ECONNABORTED, "Software caused connection abort"),
    (ErrNo.ECONNRESET, "Connection reset by peer"),
    (ErrNo.ENOBUFS, "No buffer space available"),
    (ErrNo.EISCONN, "Transport endpoint is already connected"),
    (ErrNo.ENOTCONN, "Transport endpoint is not connected"),
    (ErrNo.ESHUTDOWN, "Cannot send after transport endpoint shutdown"),
    (ErrNo.ETOOMANYREFS, "Too many references: cannot splice"),
    (ErrNo.ETIMEDOUT, "Connection timed out"),
    (ErrNo.ECONNREFUSED, "Connection refused"),
    (ErrNo.EHOSTDOWN, "Host is down"),
    (ErrNo.EHOSTUNREACH, "No route to host"),
    (ErrNo.EALREADY, "Operation already in progress"),
    (ErrNo.EINPROGRESS, "Operation now in progress"),
    (ErrNo.ESTALE, "Stale file handle"),
    (ErrNo.EUCLEAN, "Structure needs cleaning"),
    (ErrNo.ENOTNAM, "Not a XENIX named type file"),
    (ErrNo.ENAVAIL, "No XENIX semaphores available"),
    (ErrNo.EISNAM, "Is a named type file"),
    (ErrNo.EREMOTEIO, "Remote I/O error"),
    (ErrNo.EDQUOT, "Disk quota exceeded"),
    (ErrNo.ENOMEDIUM, "No medium found"),
    (ErrNo.EMEDIUMTYPE, "Wrong medium type"),
    (ErrNo.ECANCELED, "Operation canceled"),
    (ErrNo.ENOKEY, "Required key not available"),
    (ErrNo.EKEYEXPIRED, "Key has expired"),
    (ErrNo.EKEYREVOKED, "Key has been revoked"),
    (ErrNo.EKEYREJECTED, "Key was rejected by service"),
    (ErrNo.EOWNERDEAD, "Owner died"),
    (ErrNo.ENOTRECOVERABLE, "State not recoverable"),
    (ErrNo.ERFKILL, "Operation not possible due to RF-kill"),
    (ErrNo.EHWPOISON, "Memory page has hardware error"),
]


comptime error_message_macos: List[Tuple[ErrNo, String]] = [
    (ErrNo.EPERM, "Operation not permitted"),
    (ErrNo.ENOENT, "No such file or directory"),
    (ErrNo.ESRCH, "No such process"),
    (ErrNo.EINTR, "Interrupted system call"),
    (ErrNo.EIO, "Input/output error"),
    (ErrNo.ENXIO, "Device not configured"),
    (ErrNo.E2BIG, "Argument list too long"),
    (ErrNo.ENOEXEC, "Exec format error"),
    (ErrNo.EBADF, "Bad file descriptor"),
    (ErrNo.ECHILD, "No child processes"),
    (ErrNo.EAGAIN, "Resource temporarily unavailable"),
    (ErrNo.EWOULDBLOCK, "Resource temporarily unavailable"),
    (ErrNo.ENOMEM, "Cannot allocate memory"),
    (ErrNo.EACCES, "Permission denied"),
    (ErrNo.EFAULT, "Bad address"),
    (ErrNo.ENOTBLK, "Block device required"),
    (ErrNo.EBUSY, "Resource busy"),
    (ErrNo.EEXIST, "File exists"),
    (ErrNo.EXDEV, "Cross-device link"),
    (ErrNo.ENODEV, "Operation not supported by device"),
    (ErrNo.ENOTDIR, "Not a directory"),
    (ErrNo.EISDIR, "Is a directory"),
    (ErrNo.EINVAL, "Invalid argument"),
    (ErrNo.ENFILE, "Too many open files in system"),
    (ErrNo.EMFILE, "Too many open files"),
    (ErrNo.ENOTTY, "Inappropriate ioctl for device"),
    (ErrNo.ETXTBSY, "Text file busy"),
    (ErrNo.EFBIG, "File too large"),
    (ErrNo.ENOSPC, "No space left on device"),
    (ErrNo.ESPIPE, "Illegal seek"),
    (ErrNo.EROFS, "Read-only file system"),
    (ErrNo.EMLINK, "Too many links"),
    (ErrNo.EPIPE, "Broken pipe"),
    (ErrNo.EDOM, "Numerical argument out of domain"),
    (ErrNo.ERANGE, "Result too large"),
    (ErrNo.EDEADLK, "Resource deadlock avoided"),
    (ErrNo.ENAMETOOLONG, "File name too long"),
    (ErrNo.ENOLCK, "No locks available"),
    (ErrNo.ENOSYS, "Function not implemented"),
    (ErrNo.ENOTEMPTY, "Directory not empty"),
    (ErrNo.ELOOP, "Too many levels of symbolic links"),
    (ErrNo.ENOMSG, "No message of desired type"),
    (ErrNo.EIDRM, "Identifier removed"),
    (ErrNo.ENOSTR, "Not a STREAM"),
    (ErrNo.ENODATA, "No message available on STREAM"),
    (ErrNo.ETIME, "STREAM ioctl timeout"),
    (ErrNo.ENOSR, "No STREAM resources"),
    (ErrNo.EREMOTE, "Too many levels of remote in path"),
    (ErrNo.ENOLINK, "ENOLINK (Reserved)"),
    (ErrNo.EPROTO, "Protocol error"),
    (ErrNo.EMULTIHOP, "EMULTIHOP (Reserved)"),
    (ErrNo.EBADMSG, "Bad message"),
    (ErrNo.EOVERFLOW, "Value too large to be stored in data type"),
    (ErrNo.EILSEQ, "Illegal byte sequence"),
    (ErrNo.EUSERS, "Too many users"),
    (ErrNo.ENOTSOCK, "Socket operation on non-socket"),
    (ErrNo.EDESTADDRREQ, "Destination address required"),
    (ErrNo.EMSGSIZE, "Message too long"),
    (ErrNo.EPROTOTYPE, "Protocol wrong type for socket"),
    (ErrNo.ENOPROTOOPT, "Protocol not available"),
    (ErrNo.EPROTONOSUPPORT, "Protocol not supported"),
    (ErrNo.ESOCKTNOSUPPORT, "Socket type not supported"),
    (ErrNo.EOPNOTSUPP, "Operation not supported on socket"),
    (ErrNo.EPFNOSUPPORT, "Protocol family not supported"),
    (ErrNo.EAFNOSUPPORT, "Address family not supported by protocol family"),
    (ErrNo.EADDRINUSE, "Address already in use"),
    (ErrNo.EADDRNOTAVAIL, "Can't assign requested address"),
    (ErrNo.ENETDOWN, "Network is down"),
    (ErrNo.ENETUNREACH, "Network is unreachable"),
    (ErrNo.ENETRESET, "Network dropped connection on reset"),
    (ErrNo.ECONNABORTED, "Software caused connection abort"),
    (ErrNo.ECONNRESET, "Connection reset by peer"),
    (ErrNo.ENOBUFS, "No buffer space available"),
    (ErrNo.EISCONN, "Socket is already connected"),
    (ErrNo.ENOTCONN, "Socket is not connected"),
    (ErrNo.ESHUTDOWN, "Can't send after socket shutdown"),
    (ErrNo.ETOOMANYREFS, "Too many references: can't splice"),
    (ErrNo.ETIMEDOUT, "Operation timed out"),
    (ErrNo.ECONNREFUSED, "Connection refused"),
    (ErrNo.EHOSTDOWN, "Host is down"),
    (ErrNo.EHOSTUNREACH, "No route to host"),
    (ErrNo.EALREADY, "Operation already in progress"),
    (ErrNo.EINPROGRESS, "Operation now in progress"),
    (ErrNo.ESTALE, "Stale NFS file handle"),
    (ErrNo.EDQUOT, "Disc quota exceeded"),
    (ErrNo.ECANCELED, "Operation canceled"),
    (ErrNo.EOWNERDEAD, "Previous owner died"),
    (ErrNo.ENOTRECOVERABLE, "State not recoverable"),
    (ErrNo.ENOTSUP, "Operation not supported"),
    (ErrNo.EPROCLIM, "Too many processes"),
    (ErrNo.EBADRPC, "RPC struct is bad"),
    (ErrNo.ERPCMISMATCH, "RPC version wrong"),
    (ErrNo.EPROGUNAVAIL, "RPC prog. not avail"),
    (ErrNo.EPROGMISMATCH, "Program version wrong"),
    (ErrNo.EPROCUNAVAIL, "Bad procedure for program"),
    (ErrNo.EFTYPE, "Inappropriate file type or format"),
    (ErrNo.EAUTH, "Authentication error"),
    (ErrNo.ENEEDAUTH, "Need authenticator"),
    (ErrNo.EPWROFF, "Device power is off"),
    (ErrNo.EDEVERR, "Device error"),
    (ErrNo.EBADEXEC, "Bad executable (or shared library)"),
    (ErrNo.EBADARCH, "Bad CPU type in executable"),
    (ErrNo.ESHLIBVERS, "Shared library version mismatch"),
    (ErrNo.EBADMACHO, "Malformed Mach-o file"),
    (ErrNo.ENOATTR, "Attribute not found"),
    (ErrNo.ENOPOLICY, "Policy not found"),
    (ErrNo.EQFULL, "Interface output queue is full"),
]


def _test_errno_message[error_message: List[Tuple[ErrNo, String]]]() raises:
    comptime for i in range(len(error_message)):
        var errno, msg = materialize[error_message[i]]()
        set_errno(errno)
        assert_equal(get_errno(), errno)
        assert_equal(String(errno), msg)
    set_errno(ErrNo.SUCCESS)


def test_errno_message() raises:
    comptime if CompilationTarget.is_linux():
        _test_errno_message[error_message_linux]()
    elif CompilationTarget.is_macos():
        _test_errno_message[error_message_macos]()
    else:
        comptime assert False, "test not implemented for the platform"


def test_errno() raises:
    # test it raises the correct libc error
    with assert_raises(contains=String(ErrNo.ENOENT)):
        _ = realpath("does/not/exist")

    # Test that it sets errno correctly
    assert_equal(get_errno(), ErrNo.ENOENT)
    assert_equal(String(ErrNo.ENOENT), "No such file or directory")

    # test that errno can be reset to success
    set_errno(ErrNo.SUCCESS)
    assert_equal(get_errno(), ErrNo.SUCCESS)

    # Make sure we can set errno to a different value
    set_errno(ErrNo.EPERM)
    if get_errno() != ErrNo.EPERM:
        raise Error("Failed to set errno to EPERM")


def test_rtld_flags() raises:
    assert_equal(RTLD.LAZY, 1)
    assert_equal(RTLD.NOW, 2)
    comptime if CompilationTarget.is_linux():
        assert_equal(RTLD.LOCAL, 0)
        assert_equal(RTLD.GLOBAL, 256)
        assert_equal(RTLD.NODELETE, 4096)
    elif CompilationTarget.is_macos():
        assert_equal(RTLD.LOCAL, 4)
        assert_equal(RTLD.GLOBAL, 8)
        assert_equal(RTLD.NODELETE, 128)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
