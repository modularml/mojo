"""Libc POSIX file syscalls."""

from collections import Optional
from memory import UnsafePointer, stack_allocation
from sys.ffi import external_call, DLHandle
from sys.info import os_is_windows, triple_is_nvidia_cuda

from .types import C


@value
struct TryLibc[static: Bool]:
    """Try to execute Libc code, append the Libc Error, and rethrow.

    Parameters:
        static: Whether the library is statically linked.
    """

    var _lib: Libc[static=static]

    fn __enter__(self):
        """Enter a context."""
        pass

    fn __exit__(self):
        """Exit a context with no error."""
        pass

    fn __exit__(self, error: Error) raises -> Bool:
        """Exit a context with an error.

        Arguments:
            error: The error.
        """
        raise Error(
            str(error)
            + "\nLibc Error: "
            + char_ptr_to_string(self._lib.strerror(self._lib.get_errno()))
        )


@value
struct Libc[*, static: Bool]:
    """An implementation of Libc. Can be dynamically or statically linked.

    Parameters:
        static: Whether the library is statically linked.

    Notes:

        - Some exceptions are made for Microsoft Windows. Pull requests to extend
            support are welcome.
        - All reference links point to the POSIX section of the linux manual
            pages, to read the linux documentation which is often more thorough
            in explaining caveats (applicable to Linux, but similar in other
            implementations), replace the end of the link `3p.html` with
            `3.html`.
    """

    var _lib: Optional[DLHandle]

    fn __init__(inout self: Libc[static=True]):
        self._lib = None

    fn __init__(
        inout self: Libc[static=False], path: StringLiteral = "libc.so.6"
    ):
        self._lib = DLHandle(path)

    # ===------------------------------------------------------------------=== #
    # Logging
    # ===------------------------------------------------------------------=== #

    fn get_errno(self) -> C.int:
        """Get a copy of the current value of the `errno` global variable for
        the current thread.

        Returns:
            A copy of the current value of `errno` for the current thread.
        """

        @parameter
        if os_is_windows():
            errno = stack_allocation[1, C.int]()

            @parameter
            if static:
                _ = external_call["_get_errno", C.void](errno)
            else:
                _ = self._lib.value().call["_get_errno", C.void](errno)
            return errno[]
        else:

            @parameter
            if static:
                return external_call[
                    "__errno_location", UnsafePointer[C.int]
                ]()[]
            else:
                return self._lib.value().call[
                    "__errno_location", UnsafePointer[C.int]
                ]()[]

    fn set_errno(self, errno: C.int):
        """Set the `errno` global variable for the current thread.

        Args:
            errno: The value to set `errno` to.
        """

        @parameter
        if os_is_windows():

            @parameter
            if static:
                _ = external_call["_set_errno", C.int](errno)
            else:
                _ = self._lib.value().call["_set_errno", C.int](errno)
        else:

            @parameter
            if static:
                _ = external_call["__errno_location", UnsafePointer[C.int]]()[
                    0
                ] = errno
            else:
                _ = self._lib.value().call[
                    "__errno_location", UnsafePointer[C.int]
                ]()[0] = errno

    fn strerror(self, errnum: C.int) -> UnsafePointer[C.char]:
        """Libc POSIX `strerror` function.

        Args:
            errnum: The number of the error.

        Returns:
            A Pointer to the error message.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/strerror.3p.html).
            Fn signature: `char *strerror(int errnum)`.
        """

        @parameter
        if static:
            return external_call["strerror", UnsafePointer[C.char]](errnum)
        else:
            return self._lib.value().call["strerror", UnsafePointer[C.char]](
                errnum
            )

    fn perror(self, s: UnsafePointer[C.char] = UnsafePointer[C.char]()):
        """Libc POSIX `perror` function.

        Args:
            s: The string to print in front of the error message.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/perror.3p.html).
            Fn signature: `void perror(const char *s)`.
        """

        @parameter
        if static:
            _ = external_call["perror", C.void](s)
        else:
            _ = self._lib.value().call["perror", C.void](s)

    fn openlog(
        self, ident: UnsafePointer[C.char], logopt: C.int, facility: C.int
    ):
        """Libc POSIX `openlog` function.

        Args:
            ident: A File Descriptor to open the file with.
            logopt: Flags which control the operation of openlog.
            facility: Establishes a default to be used if none is specified in
                subsequent calls to syslog.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/openlog.3p.html).
            Fn signature: `void openlog(const char *ident, int logopt,
                int facility)`.
        """

        @parameter
        if static:
            _ = external_call["openlog", C.void](ident, logopt, facility)
        else:
            _ = self._lib.value().call["openlog", C.void](
                ident, logopt, facility
            )

    fn syslog[
        *T: AnyType
    ](self, priority: C.int, format: UnsafePointer[C.char], *args: *T):
        """Libc POSIX `syslog` function.

        Args:
            priority: A File Descriptor to open the file with.
            format: A format string to print.
            args: The extra arguments.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/syslog.3p.html).
            Fn signature: `void syslog(int priority, const char *format, ...)`.
        """

        @parameter
        if static:
            # FIXME: externall_call should handle this
            _ = __mlir_op.`pop.external_call`[
                func = "syslog".value,
                variadicType = __mlir_attr[
                    `(`,
                    `!pop.scalar<si32>,`,
                    `!kgen.pointer<scalar<si8>>`,
                    `) -> !pop.scalar<si8>`,
                ],
                _type = C.void,
            ](priority, format, args.get_loaded_kgen_pack())
        else:
            _ = self._lib.value().call["syslog", C.void](
                priority, format, args.get_loaded_kgen_pack()
            )

    fn setlogmask(self, maskpri: C.int) -> C.int:
        """Libc POSIX `setlogmask` function.

        Args:
            maskpri: A new priority mask.

        Returns:
            The previous log priority mask.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/setlogmask.3p.html).
            Fn signature: `int setlogmask(int maskpri)`.
            A process has a log priority mask that determines which calls to
            `syslog(3)` may be logged.  All other calls will be ignored.
            Logging is enabled for the priorities that have the corresponding
            bit set in mask.  The initial mask is such that logging is
            enabled for all priorities.
        """

        @parameter
        if static:
            return external_call["setlogmask", C.int](maskpri)
        else:
            return self._lib.value().call["setlogmask", C.int](maskpri)

    fn closelog(self):
        """Libc POSIX `closelog` function.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/closelog.3p.html).
            Fn signature: `void closelog(void)`.
        """

        @parameter
        if static:
            _ = external_call["closelog", C.void]()
        else:
            _ = self._lib.value().call["closelog", C.void]()

    # ===------------------------------------------------------------------=== #
    # Files
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn close(self, fildes: C.int) -> C.int:
        """Libc POSIX `open` function. The argument flags must include one of
        the following access modes: O_RDONLY, O_WRONLY, or O_RDWR.

        Args:
            fildes: A File Descriptor to close.

        Returns:
            Value `0` on success, `-1` on error and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/close.3p.html).
            Fn signature: `int close(int fildes)`.
        """

        @parameter
        if static:
            return external_call["close", C.int](fildes)
        else:
            return self._lib.value().call["close", C.int](fildes)

    @always_inline
    fn open(
        self, path: UnsafePointer[C.char], oflag: C.int, mode: mode_t = 0o666
    ) -> C.int:
        """Libc POSIX `open` function. The argument flags must include one of
        the following access modes: `O_RDONLY`, `O_WRONLY`, or `O_RDWR`.

        Args:
            path: A path to a file.
            oflag: A flag to open the file with.
            mode: The permission mode to open the file with.

        Returns:
            A File Descriptor. Otherwise `-1` and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/open.3p.html).
            Fn signature: `int open(const char *path, int oflag, ...)`.
        """

        @parameter
        if static:
            return external_call["open", C.int](path, oflag, mode)
        else:
            return self._lib.value().call["open", C.int](path, oflag, mode)

    @always_inline
    fn remove[*T: AnyType](self, pathname: UnsafePointer[C.char]) -> C.int:
        """Libc POSIX `open` function.

        Parameters:
            T: The type of the arguments.

        Args:
            pathname: A path to a file.

        Returns:
            Value `0` on success, otherwise `-1` and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/remove.3p.html).
            Fn signature: `int remove(const char *pathname)`.

            If the removed name was the last link to a file and no processes
            have the file open, the file is deleted and the space it was
            using is made available for reuse.

            If the name was the last link to a file, but any processes still
            have the file open, the file will remain in existence until the
            last file descriptor referring to it is closed.

            If the name referred to a symbolic link, the link is removed.

            If the name referred to a socket, FIFO, or device, the name is
            removed, but processes which have the object open may continue to
            use it.
        """

        @parameter
        if static:
            return external_call["remove", C.int](pathname)
        else:
            return self._lib.value().call["remove", C.int](pathname)

    @always_inline
    fn openat(
        self,
        fd: C.int,
        path: UnsafePointer[C.char],
        oflag: C.int,
        mode: mode_t = 0o666,
    ) -> C.int:
        """Libc POSIX `openat` function.

        Args:
            fd: A File Descriptor to open the file with.
            path: A path to a file.
            oflag: A flag to open the file with.
            mode: The permission mode to open the file with.

        Returns:
            A File Descriptor. Otherwise `-1` and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/openat.3p.html).
            Fn signature: `int openat(int fd, const char *path, int oflag, ...
            )`.
        """

        @parameter
        if static:
            return external_call["openat", C.int](fd, path, oflag, mode)
        else:
            return self._lib.value().call["openat", C.int](
                fd, path, oflag, mode
            )

    @always_inline
    fn fopen(
        self, pathname: UnsafePointer[C.char], mode: UnsafePointer[C.char]
    ) -> UnsafePointer[FILE]:
        """Libc POSIX `fopen` function.

        Args:
            pathname: A path to a file.
            mode: A mode to open the file with.

        Returns:
            A pointer to a File. Otherwise `NULL` and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/fopen.3p.html).
            Fn signature: `FILE *fopen(const char *restrict pathname,
                const char *restrict mode)`.
        """

        @parameter
        if static:
            return external_call["fopen", UnsafePointer[FILE]](pathname, mode)
        else:
            return self._lib.value().call["fopen", UnsafePointer[FILE]](
                pathname, mode
            )

    @always_inline
    fn fdopen(
        self, fildes: C.int, mode: UnsafePointer[C.char]
    ) -> UnsafePointer[FILE]:
        """Libc POSIX `fdopen` function.

        Args:
            fildes: A File Descriptor to open the file with.
            mode: A mode to open the file with.

        Returns:
            A pointer to a File. Otherwise `NULL` and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/fdopen.3p.html).
            Fn signature: `FILE *fdopen(int fildes, const char *mode)`.
        """
        alias name = "_fdopen" if os_is_windows() else "fdopen"

        @parameter
        if static:
            return external_call[name, UnsafePointer[FILE]](fildes, mode)
        else:
            return self._lib.value().call[name, UnsafePointer[FILE]](
                fildes, mode
            )

    @always_inline
    fn fclose(self, stream: UnsafePointer[FILE]) -> C.int:
        """Libc POSIX `fclose` function.

        Args:
            stream: A pointer to a stream.

        Returns:
            Value 0 on success, otherwise `EOF` (usually -1) and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/fclose.3p.html).
            Fn signature: `int fclose(FILE *stream)`.
        """

        @parameter
        if static:
            return external_call["fclose", C.int](stream)
        else:
            return self._lib.value().call["fclose", C.int](stream)

    @always_inline
    fn freopen(
        self,
        pathname: UnsafePointer[C.char],
        mode: UnsafePointer[C.char],
        stream: UnsafePointer[FILE],
    ) -> UnsafePointer[FILE]:
        """Libc POSIX `freopen` function.

        Args:
            pathname: A path to a file.
            mode: A mode to open the file with.
            stream: A pointer to a stream.

        Returns:
            A pointer to a File. Otherwise `NULL` and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/freopen.3p.html).
            Fn signature: `FILE *freopen(const char *restrict pathname,
                const char *restrict mode, FILE *restrict stream)`.
        """

        @parameter
        if static:
            return external_call["freopen", UnsafePointer[FILE]](
                pathname, mode, stream
            )
        else:
            return self._lib.value().call["freopen", UnsafePointer[FILE]](
                pathname, mode, stream
            )

    @always_inline
    fn fmemopen(
        self,
        buf: UnsafePointer[C.void],
        size: size_t,
        mode: UnsafePointer[C.char],
    ) -> UnsafePointer[FILE]:
        """Libc POSIX `fmemopen` function.

        Args:
            buf: A pointer to a buffer.
            size: The size of the buffer.
            mode: A mode to open the file with.

        Returns:
            A pointer to a File. Otherwise `NULL` and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/fmemopen.3p.html).
            Fn signature: `FILE *fmemopen(void *restrict buf, size_t size,
                const char *restrict mode)`.
        """

        @parameter
        if static:
            return external_call["fmemopen", UnsafePointer[FILE]](
                buf, size, mode
            )
        else:
            return self._lib.value().call["fmemopen", UnsafePointer[FILE]](
                buf, size, mode
            )

    @always_inline
    fn creat(self, path: UnsafePointer[C.char], mode: mode_t) -> C.int:
        """Libc POSIX `creat` function.

        Args:
            path: A path to a file.
            mode: A mode to open the file with.

        Returns:
            A File Descriptor. Otherwise `-1` and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/creat.3p.html).
            Fn signature: `int creat(const char *path, mode_t mode)`.
        """

        @parameter
        if static:
            return external_call["creat", C.int](path, mode)
        else:
            return self._lib.value().call["creat", C.int](path, mode)

    @always_inline
    fn fseek(
        self,
        stream: UnsafePointer[FILE],
        offset: C.long,
        whence: C.int = SEEK_SET,
    ) -> C.int:
        """Libc POSIX `fseek` function.

        Args:
            stream: A pointer to a stream.
            offset: An offset to seek to.
            whence: From whence to start seeking from (`SEEK_SET`, `SEEK_CUR`,
                `SEEK_END`).

        Returns:
            Value `0` on success, `-1` on error and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/fseek.3p.html).
            Fn signature: `int fseek(FILE *stream, long offset, int whence)`.
        """

        @parameter
        if static:
            return external_call["fseek", C.int](stream, offset, whence)
        else:
            return self._lib.value().call["fseek", C.int](
                stream, offset, whence
            )

    @always_inline
    fn fseeko(
        self, stream: UnsafePointer[FILE], offset: off_t, whence: C.int
    ) -> C.int:
        """Libc POSIX `fseeko` function.

        Args:
            stream: A pointer to a stream.
            offset: An offset to seek to.
            whence: From whence to start seeking from (`SEEK_SET`, `SEEK_CUR`,
                `SEEK_END`).

        Returns:
            Value `0` on success, `-1` on error and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/fseek.3p.html).
            Fn signature: `int fseeko(FILE *stream, off_t offset, int whence)`.
        """

        @parameter
        if static:
            return external_call["fseeko", C.int](stream, offset, whence)
        else:
            return self._lib.value().call["fseeko", C.int](
                stream, offset, whence
            )

    @always_inline
    fn lseek(self, fildes: C.int, offset: off_t, whence: C.int) -> off_t:
        """Libc POSIX `lseek` function.

        Args:
            fildes: A File Descriptor to open the file with.
            offset: An offset to seek to.
            whence: A pointer to a buffer to store the length of the address of
                the accepted socket.

        Returns:
            The resulting offset, as measured in bytes from the beginning of the
            file on success, -1 on error and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/lseek.3p.html).
            Fn signature: `off_t lseek(int fildes, off_t offset, int whence)`.
        """

        @parameter
        if static:
            return external_call["lseek", off_t](fildes, offset, whence)
        else:
            return self._lib.value().call["lseek", off_t](
                fildes, offset, whence
            )

    @always_inline
    fn fputc(self, c: C.int, stream: UnsafePointer[FILE]) -> C.int:
        """Libc POSIX `fputc` function.

        Args:
            c: A character to write.
            stream: A pointer to a stream.

        Returns:
            The value it has written. Otherwise `EOF` (usually -1) and `errno`
            is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/fputc.3p.html).
            Fn signature: `int fputc(int c, FILE *stream)`.
        """

        @parameter
        if static:
            return external_call["fputc", C.int](c, stream)
        else:
            return self._lib.value().call["fputc", C.int](c, stream)

    @always_inline
    fn fputs(
        self, s: UnsafePointer[C.char], stream: UnsafePointer[FILE]
    ) -> C.int:
        """Libc POSIX `fputs` function.

        Args:
            s: A string to write.
            stream: A pointer to a stream.

        Returns:
            Positive number. Otherwise `EOF` (usually -1) and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/fputs.3p.html).
            Fn signature: `int fputs(const char *restrict s, FILE *restrict
            stream)`.
        """

        @parameter
        if static:
            return external_call["fputs", C.int](s, stream)
        else:
            return self._lib.value().call["fputs", C.int](s, stream)

    @always_inline
    fn fgetc(self, stream: UnsafePointer[FILE]) -> C.int:
        """Libc POSIX `fgetc` function.

        Args:
            stream: A pointer to a stream.

        Returns:
            The next byte from the input stream pointed to by stream. If the
            end-of-file indicator for the stream is set, or if the stream is at
            end-of-file, the end-of-file indicator for the stream shall be set
            and `fgetc()` shall return EOF. If a read error occurs, the error
            indicator for the stream shall be set, fgetc() shall return `EOF`,
            and shall set `errno` to indicate the error.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/fgetc.3p.html).
            Fn signature: `int fgetc(FILE *stream)`.
        """

        @parameter
        if static:
            return external_call["fgetc", C.int](stream)
        else:
            return self._lib.value().call["fgetc", C.int](stream)

    @always_inline
    fn fgets(
        self, s: UnsafePointer[C.char], n: C.int, stream: UnsafePointer[FILE]
    ) -> UnsafePointer[C.char]:
        """Libc POSIX `fgets` function.

        Args:
            s: A pointer to a buffer to store the read string.
            n: The maximum number of characters to read.
            stream: A pointer to a stream.

        Returns:
            Upon successful completion, fgets() shall return s. If the stream is
            at end-of-file, the end-of-file indicator for the stream shall be
            set and `fgets()` shall return a null pointer. If a read error
            occurs, the error indicator for the stream shall be set, `fgets()`
            shall return a null pointer, and shall set `errno` to indicate the
            error.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/fgets.3p.html).
            Fn signature: `char *fgets(char *restrict s, int n,
                FILE *restrict stream)`.
        """

        @parameter
        if static:
            return external_call["fgets", UnsafePointer[C.char]](s, n, stream)
        else:
            return self._lib.value().call["fgets", UnsafePointer[C.char]](
                s, n, stream
            )

    @always_inline
    fn printf(
        self,
        format: UnsafePointer[C.char],
        args: VariadicPack[element_trait=AnyType],
    ) -> C.int:
        """Libc POSIX `printf` function.

        Args:
            format: The format string.
            args: The arguments to be added into the format string.

        Returns:
            The number of bytes transmitted excluding the terminating null byte.
            Otherwise a negative value and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/printf.3p.html).
        """

        @parameter
        if static:
            # FIXME: externall_call should handle this
            num = __mlir_op.`pop.external_call`[
                func = "printf".value,
                variadicType = __mlir_attr[
                    `(`,
                    `!kgen.pointer<scalar<si8>>`,
                    `) -> !pop.scalar<si32>`,
                ],
                _type = C.int,
            ](format, args.get_loaded_kgen_pack())
            return int(num)
        else:
            num = self._lib.value().call["printf", C.int](
                format, args.get_loaded_kgen_pack()
            )
            return int(num)

    @always_inline
    fn printf[
        *T: AnyType
    ](self, format: UnsafePointer[C.char], *args: *T) -> C.int:
        """Libc POSIX `printf` function.

        Parameters:
            T: The type of the arguments.

        Args:
            format: The format string.
            args: The arguments to be added into the format string.

        Returns:
            The number of bytes transmitted excluding the terminating null byte.
            Otherwise a negative value and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/printf.3p.html).
        """
        return self.printf(format, args)

    @always_inline
    fn fprintf(
        self,
        stream: UnsafePointer[FILE],
        format: UnsafePointer[C.char],
        args: VariadicPack[element_trait=AnyType],
    ) -> C.int:
        """Libc POSIX `fprintf` function.

        Args:
            stream: A pointer to a stream.
            format: A format string.
            args: The arguments to be added into the format string.

        Returns:
            The number of bytes transmitted excluding the terminating null byte.
            Otherwise a negative value and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/fprintf.3p.html).
            Fn signature: `int fprintf(FILE *restrict stream,
                const char *restrict format, ...)`.
        """

        @parameter
        if static:
            # FIXME: externall_call should handle this
            return __mlir_op.`pop.external_call`[
                func = "fprintf".value,
                variadicType = __mlir_attr[
                    `(`,
                    `!kgen.pointer<none>,`,
                    `!kgen.pointer<scalar<si8>>`,
                    `) -> !pop.scalar<si32>`,
                ],
                _type = C.int,
            ](stream, format, args.get_loaded_kgen_pack())
        else:
            return self._lib.value().call["fprintf", C.int](
                stream, format, args.get_loaded_kgen_pack()
            )

    @always_inline
    fn fprintf[
        *T: AnyType
    ](
        self,
        stream: UnsafePointer[FILE],
        format: UnsafePointer[C.char],
        *args: *T,
    ) -> C.int:
        """Libc POSIX `fprintf` function.

        Parameters:
            T: The type of the arguments.

        Args:
            stream: A pointer to a stream.
            format: A format string.
            args: The arguments to be added into the format string.

        Returns:
            The number of bytes transmitted excluding the terminating null byte.
            Otherwise a negative value and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/fprintf.3p.html).
            Fn signature: `int fprintf(FILE *restrict stream,
                const char *restrict format, ...)`.
        """
        return self.fprintf(stream, format, args)

    @always_inline
    fn fprintf(
        self, stream: UnsafePointer[FILE], format: UnsafePointer[C.char]
    ) -> C.int:
        """Libc POSIX `fprintf` function.

        Args:
            stream: A pointer to a stream.
            format: A format string.

        Returns:
            The number of bytes transmitted excluding the terminating null byte.
            Otherwise a negative value and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/fprintf.3p.html).
            Fn signature: `int fprintf(FILE *restrict stream,
                const char *restrict format, ...)`.
        """
        return self.fprintf(stream, format, None)

    @always_inline
    fn dprintf(
        self,
        fd: C.int,
        format: UnsafePointer[C.char],
        args: VariadicPack[element_trait=AnyType],
    ) -> C.int:
        """Libc POSIX `dprintf` function.

        Args:
            fd: A File Descriptor to open the file with.
            format: A format string.
            args: The arguments to be added into the format string.

        Returns:
            The number of bytes transmitted excluding the terminating null byte.
            Otherwise a negative value and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/dprintf.3p.html).
            Fn signature: `int dprintf(int fd, const char *restrict format,
                ...)`.
        """

        @parameter
        if static:
            # FIXME: externall_call should handle this
            return __mlir_op.`pop.external_call`[
                func = "dprintf".value,
                variadicType = __mlir_attr[
                    `(`,
                    `!pop.scalar<si32>,`,
                    `!kgen.pointer<scalar<si8>>`,
                    `) -> !pop.scalar<si32>`,
                ],
                _type = C.int,
            ](fd, format, args.get_loaded_kgen_pack())
        else:
            return self._lib.value().call["dprintf", C.int](
                fd, format, args.get_loaded_kgen_pack()
            )

    @always_inline
    fn dprintf[
        *T: AnyType
    ](self, fd: C.int, format: UnsafePointer[C.char], *args: *T) -> C.int:
        """Libc POSIX `dprintf` function.

        Parameters:
            T: The type of the arguments.

        Args:
            fd: A File Descriptor to open the file with.
            format: A format string.
            args: The arguments to be added into the format string.

        Returns:
            The number of bytes transmitted. Otherwise a negative value and
            `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/dprintf.3p.html).
            Fn signature: `int dprintf(int fd, const char *restrict format,
                ...)`.
        """
        return self.dprintf(fd, format, args)

    @always_inline
    fn sprintf[
        *T: AnyType
    ](
        self,
        str: UnsafePointer[C.char],
        format: UnsafePointer[C.char],
        *args: *T,
    ) -> C.int:
        """Libc POSIX `sprintf` function.

        Parameters:
            T: The type of the arguments.

        Args:
            str: A pointer to a buffer to store the read string.
            format: A format string.
            args: The arguments to be added into the format string.

        Returns:
            The number of bytes written, excluding the terminating null byte.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/sprintf.3p.html).
            Fn signature: `int sprintf(char *restrict str,
                const char *restrict format, ...)`.
        """

        @parameter
        if static:
            # FIXME: externall_call should handle this
            num = __mlir_op.`pop.external_call`[
                func = "sprintf".value,
                variadicType = __mlir_attr[
                    `(`,
                    `!kgen.pointer<scalar<si8>>,`,
                    `!kgen.pointer<scalar<si8>>`,
                    `) -> !pop.scalar<si32>`,
                ],
                _type = C.int,
            ](str, format, args.get_loaded_kgen_pack())
            return int(num)
        else:
            num = self._lib.value().call["sprintf", C.int](
                str, format, args.get_loaded_kgen_pack()
            )
            return int(num)

    @always_inline
    fn snprintf[
        *T: AnyType
    ](
        self,
        s: UnsafePointer[C.char],
        n: size_t,
        format: UnsafePointer[C.char],
        *args: *T,
    ) -> C.int:
        """Libc POSIX `snprintf` function.

        Parameters:
            T: The type of the arguments.

        Args:
            s: A pointer to a buffer to store the read string.
            n: The maximum number of characters to read.
            format: A format string.
            args: The arguments to be added into the format string.

        Returns:
            The number of bytes that would be written to s had n been
            sufficiently large excluding the terminating null byte.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/snprintf.3p.html).
            Fn signature: `int snprintf(char *restrict s, size_t n,
                const char *restrict format, ...)`.
        """

        @parameter
        if static:
            # FIXME: externall_call should handle this
            num = __mlir_op.`pop.external_call`[
                func = "snprintf".value,
                variadicType = __mlir_attr[
                    `(`,
                    `!kgen.pointer<scalar<si8>>,`,
                    `!pop.scalar<index>, `,
                    `!kgen.pointer<scalar<si8>>`,
                    `) -> !pop.scalar<si32>`,
                ],
                _type = C.int,
            ](s, n, format, args.get_loaded_kgen_pack())
            return int(num)
        else:
            num = self._lib.value().call["snprintf", C.int](
                s, n, format, args.get_loaded_kgen_pack()
            )
            return int(num)

    # TODO: add va_list builder and test
    # @always_inline
    # fn vprintf(
    #     self,
    #     format: UnsafePointer[C.char],
    #     ap: VariadicPack[element_trait=AnyType],
    # ) -> C.int:
    #     """Libc POSIX `vprintf` function.

    #     Args:
    #         format: A format string.
    #         ap: The arguments to be added into the format string.

    #     Returns:
    #         The number of bytes transmitted excluding the terminating null byte.
    #         Otherwise a negative value and `errno` is set.

    #     Notes:
    #         [Reference](https://man7.org/linux/man-pages/man3/vprintf.3p.html).
    #         Fn signature: `int vprintf(const char *restrict format,
    #             va_list ap)`.
    #     """

    #     a = ap.get_loaded_kgen_pack()
    #     p = UnsafePointer.address_of(a)

    #     @parameter
    #     if static:
    #         return int(external_call["vprintf", C.int](format, p))
    #     else:
    #         return int(self._lib.value().call["vprintf", C.int](format, p))

    # TODO: add va_list builder and test
    # @always_inline
    # fn vfprintf(
    #     self,
    #     stream: UnsafePointer[FILE],
    #     format: UnsafePointer[C.char],
    #     ap: VariadicPack[element_trait=AnyType],
    # ) -> C.int:
    #     """Libc POSIX `vfprintf` function.

    #     Args:
    #         stream: A pointer to a stream.
    #         format: A format string.
    #         ap: The arguments to be added into the format string.

    #     Returns:
    #         The number of bytes transmitted excluding the terminating null byte.
    #         Otherwise a negative value and `errno` is set.

    #     Notes:
    #         [Reference](https://man7.org/linux/man-pages/man3/vfprintf.3p.html).
    #         Fn signature: `int vfprintf(FILE *restrict stream,
    #             const char *restrict format, va_list ap)`.
    #     """

    #     a = ap.get_loaded_kgen_pack()
    #     p = UnsafePointer.address_of(a)

    #     @parameter
    #     if static:
    #         return int(external_call["vfprintf", C.int](stream, format, p))
    #     else:
    #         return int(
    #             self._lib.value().call["vfprintf", C.int](stream, format, p)
    #         )

    # TODO: add va_list builder and test
    # @always_inline
    # fn vdprintf(
    #     self,
    #     fd: C.int,
    #     format: UnsafePointer[C.char],
    #     ap: VariadicPack[element_trait=AnyType],
    # ) -> C.int:
    #     """Libc POSIX `vdprintf` function.

    #     Args:
    #         fd: A file descriptor.
    #         format: A format string.
    #         ap: The arguments to be added into the format string.

    #     Returns:
    #         The number of bytes transmitted excluding the terminating null byte.
    #         Otherwise a negative value and `errno` is set.

    #     Notes:
    #         [Reference](https://man7.org/linux/man-pages/man3/vdprintf.3p.html).
    #         Fn signature: `int vdprintf(int fd, const char *restrict format,
    #             va_list ap)`.
    #     """

    #     a = ap.get_loaded_kgen_pack()
    #     p = UnsafePointer.address_of(a)

    #     @parameter
    #     if static:
    #         return int(external_call["vdprintf", C.int](fd, format, p))
    #     else:
    #         return int(self._lib.value().call["vdprintf", C.int](fd, format, p))

    # TODO: add va_list builder and test
    # @always_inline
    # fn vsprintf(
    #     self,
    #     str: UnsafePointer[C.char],
    #     format: UnsafePointer[C.char],
    #     ap: VariadicPack[element_trait=AnyType],
    # ) -> C.int:
    #     """Libc POSIX `vsprintf` function.

    #     Args:
    #         str: A pointer to a buffer to store the read string.
    #         format: A format string.
    #         ap: The arguments to be added into the format string.

    #     Returns:
    #         The number of bytes transmitted excluding the terminating null byte.
    #         Otherwise a negative value and `errno` is set.

    #     Notes:
    #         [Reference](https://man7.org/linux/man-pages/man3/vsprintf.3p.html).
    #         Fn signature: `int vsprintf(char *restrict str,
    #             const char *restrict format, va_list ap)`.
    #     """

    #     a = ap.get_loaded_kgen_pack()
    #     p = UnsafePointer.address_of(a)

    #     @parameter
    #     if static:
    #         return int(external_call["vsprintf", C.int](str, format, p))
    #     else:
    #         return int(
    #             self._lib.value().call["vsprintf", C.int](str, format, p)
    #         )

    # TODO: add va_list builder and test
    # @always_inline
    # fn vsnprintf(
    #     self,
    #     s: UnsafePointer[C.char],
    #     n: size_t,
    #     format: UnsafePointer[C.char],
    #     ap: VariadicPack[element_trait=AnyType],
    # ) -> C.int:
    #     """Libc POSIX `vsnprintf` function.

    #     Args:
    #         s: A pointer to a buffer to store the read string.
    #         n: The maximum number of characters to read.
    #         format: A format string.
    #         ap: The arguments to be added into the format string.

    #     Returns:
    #         The number of bytes that would be written to s had n been
    #         sufficiently large excluding the terminating null byte.

    #     Notes:
    #         [Reference](https://man7.org/linux/man-pages/man3/vsnprintf.3p.html).
    #         Fn signature: `int vsnprintf(char *restrict s, size_t n,
    #             const char *restrict format, va_list ap)`.
    #     """

    #     a = ap.get_loaded_kgen_pack()
    #     p = UnsafePointer.address_of(a)

    #     @parameter
    #     if static:
    #         return int(external_call["vsnprintf", C.int](s, n, format, p))
    #     else:
    #         return int(
    #             self._lib.value().call["vsnprintf", C.int](s, n, format, p)
    #         )

    @always_inline
    fn fscanf[
        *T: AnyType
    ](
        self,
        stream: UnsafePointer[FILE],
        format: UnsafePointer[C.char],
        *args: *T,
    ) -> C.int:
        """Libc POSIX `fscanf` function.

        Parameters:
            T: The type of the arguments.

        Args:
            stream: A pointer to a stream.
            format: A format string.
            args: The set of pointer arguments indicating where the converted
                input should be stored.

        Returns:
            The number of successfully matched and assigned input items; this
            number can be zero in the event of an early matching failure. If the
            input ends before the first conversion (if any) has completed, and
            without a matching failure having occurred, `EOF` shall be returned.
            If an error occurs before the first conversion (if any) has
            completed, and without a matching failure having occurred, `EOF`
            shall be returned and `errno` shall be set to indicate the error.
            If a read error occurs, the error indicator for the stream shall be
            set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/fscanf.3p.html).
            Fn signature: `int fscanf(FILE *restrict stream,
                const char *restrict format, ...)`.
        """

        @parameter
        if static:
            # FIXME: externall_call should handle this
            return __mlir_op.`pop.external_call`[
                func = "fscanf".value,
                variadicType = __mlir_attr[
                    `(`,
                    `!kgen.pointer<none>,`,
                    `!kgen.pointer<scalar<si8>>`,
                    `) -> !pop.scalar<si32>`,
                ],
                _type = C.int,
            ](stream, format, args.get_loaded_kgen_pack())
        else:
            return self._lib.value().call["fscanf", C.int](
                stream, format, args.get_loaded_kgen_pack()
            )

    @always_inline
    fn scanf[
        *T: AnyType
    ](self, format: UnsafePointer[C.char], *args: *T) -> C.int:
        """Libc POSIX `scanf` function.

        Parameters:
            T: The type of the arguments.

        Args:
            format: A format string.
            args: The set of pointer arguments indicating where the converted
                input should be stored.

        Returns:
            The number of successfully matched and assigned input items; this
            number can be zero in the event of an early matching failure. If the
            input ends before the first conversion (if any) has completed, and
            without a matching failure having occurred, `EOF` shall be returned.
            If an error occurs before the first conversion (if any) has
            completed, and without a matching failure having occurred, `EOF`
            shall be returned and `errno` shall be set to indicate the error. If
            a read error occurs, the error indicator for the stream shall be
            set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/fscanf.3p.html).
            Fn signature: `int scanf(const char *restrict format, ...)`.`.
        """

        @parameter
        if static:
            # FIXME: externall_call should handle this
            return __mlir_op.`pop.external_call`[
                func = "scanf".value,
                variadicType = __mlir_attr[
                    `(`,
                    `!kgen.pointer<scalar<si8>>`,
                    `) -> !pop.scalar<si32>`,
                ],
                _type = C.int,
            ](format, args.get_loaded_kgen_pack())
        else:
            return self._lib.value().call["scanf", C.int](
                format, args.get_loaded_kgen_pack()
            )

    @always_inline
    fn sscanf(
        self, s: UnsafePointer[C.char], format: UnsafePointer[C.char]
    ) -> C.int:
        """Libc POSIX `sscanf` function.

        Args:
            s: A pointer to a buffer to store the read string.
            format: A format string.

        Returns:
            The number of successfully matched and assigned input items; this
            number can be zero in the event of an early matching failure. If the
            input ends before the first conversion (if any) has completed, and
            without a matching failure having occurred, `EOF` shall be returned.
            If an error occurs before the first conversion (if any) has
            completed, and without a matching failure having occurred, `EOF`
            shall be returned and `errno` shall be set to indicate the error.
            If a read error occurs, the error indicator for the stream shall be
            set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/sscanf.3p.html).
            Fn signature: `int sscanf(const char *restrict s,
                const char *restrict format, ...)`.
        """

        @parameter
        if static:
            return external_call["sscanf", C.int](s, format)
        else:
            return self._lib.value().call["sscanf", C.int](s, format)

    @always_inline
    fn fread(
        self,
        ptr: UnsafePointer[C.void],
        size: size_t,
        nitems: size_t,
        stream: UnsafePointer[FILE],
    ) -> size_t:
        """Libc POSIX `fread` function.

        Args:
            ptr: A pointer to a buffer to store the read string.
            size: The size of the buffer.
            nitems: The number of items to read.
            stream: A pointer to a stream.

        Returns:
            The number of elements successfully read which is less than nitems
            only if a read error or end-of-file is encountered. If size or
            nitems is 0, `fread()` shall return 0 and the contents of the array
            and the state of the stream remain unchanged. Otherwise, if a read
            error occurs, the error indicator for the stream shall be set, and
            `errno` shall be set to indicate the error.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/fread.3p.html).
            Fn signature: `size_t fread(void *restrict ptr, size_t size,
                size_t nitems, FILE *restrict stream)`.
        """

        @parameter
        if static:
            return external_call["fread", size_t](ptr, size, nitems, stream)
        else:
            return self._lib.value().call["fread", size_t](
                ptr, size, nitems, stream
            )

    @always_inline
    fn rewind(self, stream: UnsafePointer[FILE]):
        """Libc POSIX `rewind` function.

        Args:
            stream: A pointer to a stream.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/rewind.3p.html).
            Fn signature: `void rewind(FILE *stream)`.
        """

        @parameter
        if static:
            _ = external_call["rewind", C.void](stream)
        else:
            _ = self._lib.value().call["rewind", C.void](stream)

    # FIXME: stream should be UnsafePointer[UnsafePointer[FILE]]
    fn getline(
        self,
        lineptr: UnsafePointer[C.ptr_addr],
        n: UnsafePointer[C.u_int],
        stream: UnsafePointer[FILE],
    ) -> C.u_int:
        """Libc POSIX `getline` function.

        Args:
            lineptr: A pointer to a pointer to a buffer to store the read string.
            n: The length in bytes of the buffer.
            stream: A pointer to a stream.

        Returns:
            The number of bytes written into the buffer, including the delimiter
            character if one was encountered before EOF, but excluding the
            terminating NUL character. If the end-of-file indicator for the
            stream is set, or if no characters were read and the stream is at
            end-of-file, the end-of-file indicator for the stream shall be set
            and the function shall return -1.  If an error occurs, the error
            indicator for the stream shall be set, and the function shall return
            -1 and set `errno` to indicate the error.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/getline.3p.html).
            Fn signature: `ssize_t getline(char **restrict lineptr,
                size_t *restrict n, FILE *restrict stream);`.
        """

        @parameter
        if static:
            return external_call["getline", C.u_int](lineptr, n, stream)
        else:
            return self._lib.value().call["getline", C.u_int](
                lineptr, n, stream
            )

    # FIXME: lineptr should be UnsafePointer[addrinfo]
    fn getdelim(
        self,
        lineptr: UnsafePointer[C.ptr_addr],
        n: UnsafePointer[C.u_int],
        stream: UnsafePointer[FILE],
    ) -> C.u_int:
        """Libc POSIX `getdelim` function.

        Args:
            lineptr: A pointer to a pointer to a buffer to store the read string.
            n: The length in bytes of the buffer.
            stream: A pointer to a stream.

        Returns:
            The number of bytes written into the buffer, including the delimiter
            character if one was encountered before EOF, but excluding the
            terminating NUL character. If the end-of-file indicator for the
            stream is set, or if no characters were read and the stream is at
            end-of-file, the end-of-file indicator for the stream shall be set
            and the function shall return -1.  If an error occurs, the error
            indicator for the stream shall be set, and the function shall return
            -1 and set `errno` to indicate the error.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/getdelim.3p.html).
            Fn signature: `ssize_t getdelim(char **restrict lineptr,
                size_t *restrict n, FILE *restrict stream);`.
        """

        @parameter
        if static:
            return external_call["getdelim", C.u_int](lineptr, n, stream)
        else:
            return self._lib.value().call["getdelim", C.u_int](
                lineptr, n, stream
            )

    @always_inline
    fn pread(
        self,
        fildes: C.int,
        buf: UnsafePointer[C.void],
        nbyte: C.u_int,
        offset: off_t,
    ) -> C.u_int:
        """Libc POSIX `pread` function.

        Args:
            fildes: A File Descriptor to open the file with.
            buf: A pointer to a buffer to store the read string.
            nbyte: The maximum number of characters to read.
            offset: An offset to seek to.

        Returns:
            The number of bytes read. Otherwise -1 and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/read.3p.html).
            Fn signature: `ssize_t pread(int fildes, void *buf, size_t nbyte,
                off_t offset)`.
        """

        @parameter
        if static:
            return external_call["pread", C.u_int](fildes, buf, nbyte, offset)
        else:
            return self._lib.value().call["pread", C.u_int](
                fildes, buf, nbyte, offset
            )

    @always_inline
    fn read(
        self, fildes: C.int, buf: UnsafePointer[C.void], nbyte: size_t
    ) -> ssize_t:
        """Libc POSIX `read` function.

        Args:
            fildes: A File Descriptor to open the file with.
            buf: A pointer to a buffer to store the read string.
            nbyte: The maximum number of characters to read.

        Returns:
            The number of bytes read. Otherwise -1 and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/read.3p.html).
            Fn signature: `size_t read(int fildes, void *buf, size_t nbyte)`.
        """

        @parameter
        if static:
            return external_call["read", ssize_t](fildes, buf, nbyte)
        else:
            return self._lib.value().call["read", ssize_t](fildes, buf, nbyte)

    @always_inline
    fn pwrite(
        self,
        fildes: C.int,
        buf: UnsafePointer[C.void],
        nbyte: size_t,
        offset: off_t,
    ) -> ssize_t:
        """Libc POSIX `pwrite` function.

        Args:
            fildes: A File Descriptor to open the file with.
            buf: A pointer to a buffer to store.
            nbyte: The maximum number of characters to write.
            offset: An offset to seek to.

        Returns:
            The number of bytes written. Otherwise -1 and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/pwrite.3p.html).
            Fn signature: `ssize_t pwrite(int fildes, const void *buf,
                size_t nbyte, off_t offset)`.
        """

        @parameter
        if static:
            return external_call["pwrite", ssize_t](fildes, buf, nbyte, offset)
        else:
            return self._lib.value().call["pwrite", ssize_t](
                fildes, buf, nbyte, offset
            )

    @always_inline
    fn write(
        self, fildes: C.int, buf: UnsafePointer[C.void], nbyte: size_t
    ) -> ssize_t:
        """Libc POSIX `write` function.

        Args:
            fildes: A File Descriptor to open the file with.
            buf: A pointer to a buffer to store.
            nbyte: The maximum number of characters to write.

        Returns:
            The number of bytes written. Otherwise -1 and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/write.3p.html).
            Fn signature: `ssize_t write(int fildes, const void *buf,
                size_t nbyte)`.
        """

        @parameter
        if static:
            return external_call["write", ssize_t](fildes, buf, nbyte)
        else:
            return self._lib.value().call["write", ssize_t](fildes, buf, nbyte)

    @always_inline
    fn ftell(self, stream: UnsafePointer[FILE]) -> C.long:
        """Libc POSIX `ftell` function.

        Args:
            stream: A pointer to a stream.

        Returns:
            The byte offset form the start. Otherwise -1 and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/ftell.3p.html).
            Fn signature: `long ftell(FILE *stream)`.
        """

        @parameter
        if static:
            return external_call["ftell", C.long](stream)
        else:
            return self._lib.value().call["ftell", C.long](stream)

    @always_inline
    fn ftello(self, stream: UnsafePointer[FILE]) -> off_t:
        """Libc POSIX `ftello` function.

        Args:
            stream: A pointer to a stream.

        Returns:
            The byte offset form the start. Otherwise -1 and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/ftell.3p.html).
            Fn signature: `off_t ftello(FILE *stream)`.
        """

        @parameter
        if static:
            return external_call["ftello", off_t](stream)
        else:
            return self._lib.value().call["ftello", off_t](stream)

    @always_inline
    fn fflush(self, stream: UnsafePointer[FILE]) -> C.int:
        """Libc POSIX `fflush` function.

        Args:
            stream: The stream.

        Returns:
            Value 0 on success, otherwise `EOF` (usually -1) and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/fflush.3p.html).
            Fn signature: `int fflush(FILE *stream)`.
        """

        @parameter
        if static:
            return external_call["fflush", C.int](stream)
        else:
            return self._lib.value().call["fflush", C.int](stream)

    @always_inline
    fn clearerr(self, stream: UnsafePointer[FILE]):
        """Libc POSIX `clearerr` function.

        Args:
            stream: A pointer to a stream.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/clearerr.3p.html).
            Fn signature: `void clearerr(FILE *stream)`.
        """

        @parameter
        if static:
            _ = external_call["clearerr", C.void](stream)
        else:
            _ = self._lib.value().call["clearerr", C.void](stream)

    @always_inline
    fn feof(self, stream: UnsafePointer[FILE]) -> C.int:
        """Libc POSIX `feof` function.

        Args:
            stream: A pointer to a stream.

        Returns:
            A non-zero value if the end-of-file indicator associated with the
            stream is set, else 0.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/feof.3p.html).
            Fn signature: `int feof(FILE *stream)`.
        """

        @parameter
        if static:
            return external_call["feof", C.int](stream)
        else:
            return self._lib.value().call["feof", C.int](stream)

    @always_inline
    fn ferror(self, stream: UnsafePointer[FILE]) -> C.int:
        """Libc POSIX `ferror` function.

        Args:
            stream: A pointer to a stream.

        Returns:
            A non-zero value if the error indicator associated with the stream
            is set, else 0.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/ferror.3p.html).
            Fn signature: `int ferror(FILE *stream)`.
        """

        @parameter
        if static:
            return external_call["ferror", C.int](stream)
        else:
            return self._lib.value().call["ferror", C.int](stream)

    @always_inline
    fn fcntl[*T: AnyType](self, fildes: C.int, cmd: C.int, *args: *T) -> C.int:
        """Libc POSIX `fcntl` function.

        Parameters:
            T: The types of the arguments.

        Args:
            fildes: A File Descriptor to close.
            cmd: A command to execute.
            args: The extra args.

        Returns:
            Value depending on cmd on success, `-1` on error and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/fcntl.3p.html).
            Fn signature: `int fcntl(int fildes, int cmd, ...)`.
        """

        @parameter
        if static:
            # FIXME: externall_call should handle this
            return __mlir_op.`pop.external_call`[
                func = "fcntl".value,
                variadicType = __mlir_attr[
                    `(`,
                    `!pop.scalar<si32>,`,
                    `!pop.scalar<si32>`,
                    `) -> !pop.scalar<si32>`,
                ],
                _type = C.int,
            ](fildes, cmd, args.get_loaded_kgen_pack())
        else:
            return self._lib.value().call["fcntl", C.int](fildes, cmd, args)

    # TODO: this needs to be tested thoroughly
    # @always_inline
    # fn ioctl[
    #     *T: AnyType
    # ](self, fildes: C.int, request: C.int, *args: *T) -> C.int:
    #     """Libc POSIX `ioctl` function.

    #     Parameters:
    #         T: The types of the arguments.

    #     Args:
    #         fildes: A File Descriptor to open the file with.
    #         request: An offset to seek to.
    #         args: The extra args.

    #     Returns:
    #         Upon successful completion, `ioctl()` shall return a value other
    #         than -1 that depends upon the STREAMS device control function.
    #         Otherwise, it shall return -1 and set `errno` to indicate the error.

    #     Notes:
    #         [Reference](https://man7.org/linux/man-pages/man3/ioctl.3p.html).
    #         Fn signature: `int ioctl(int fildes, int request, ...)`.
    #     """

    #     @parameter
    #     if static:
    #         # FIXME: externall_call should handle this
    #         return __mlir_op.`pop.external_call`[
    #             func = "ioctl".value,
    #             variadicType = __mlir_attr[
    #                 `(`,
    #                 `!pop.scalar<si32>,`,
    #                 `!pop.scalar<si32>`,
    #                 `) -> !pop.scalar<si32>`,
    #             ],
    #             _type = C.int,
    #         ](fildes, request, args.get_loaded_kgen_pack())
    #     else:
    #         return self._lib.value().call["ioctl", C.int](
    #             fildes, request, args.get_loaded_kgen_pack()
    #         )

    # ===------------------------------------------------------------------=== #
    # Networking
    # ===------------------------------------------------------------------=== #

    fn htonl(self, hostlong: C.u_int) -> C.u_int:
        """Libc POSIX `htonl` function.

        Args:
            hostlong: A 32-bit unsigned integer in host byte order.

        Returns:
            A 32-bit unsigned integer in network byte order.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/htonl.3p.html).
            Fn signature: `uint32_t htonl(uint32_t hostlong)`.
        """

        @parameter
        if static:
            return external_call["htonl", C.u_int](hostlong)
        else:
            return self._lib.value().call["htonl", C.u_int](hostlong)

    fn htons(self, hostshort: C.u_short) -> C.u_short:
        """Libc POSIX `htons` function.

        Args:
            hostshort: A 16-bit unsigned integer in host byte order.

        Returns:
            A 16-bit unsigned integer in network byte order.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/htonl.3p.html).
            Fn signature: `uint16_t htons(uint16_t hostshort)`.
        """

        @parameter
        if static:
            return external_call["htons", C.u_short](hostshort)
        else:
            return self._lib.value().call["htons", C.u_short](hostshort)

    fn ntohl(self, netlong: C.u_int) -> C.u_int:
        """Libc POSIX `ntohl` function.

        Args:
            netlong: A 32-bit unsigned integer in network byte order.

        Returns:
            A 32-bit unsigned integer in host byte order.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/htonl.3p.html).
            Fn signature: `uint32_t ntohl(uint32_t netlong)`.
        """

        @parameter
        if static:
            return external_call["ntohl", C.u_int](netlong)
        else:
            return self._lib.value().call["ntohl", C.u_int](netlong)

    fn ntohs(self, netshort: C.u_short) -> C.u_short:
        """Libc POSIX `ntohs` function.

        Args:
            netshort: A 16-bit unsigned integer in network byte order.

        Returns:
            A 16-bit unsigned integer in host byte order.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/htonl.3p.html).
            Fn signature: `uint16_t ntohs(uint16_t netshort)`.
        """

        @parameter
        if static:
            return external_call["ntohs", C.u_short](netshort)
        else:
            return self._lib.value().call["ntohs", C.u_short](netshort)

    fn inet_ntop(
        self,
        af: C.int,
        src: UnsafePointer[C.void],
        dst: UnsafePointer[C.char],
        size: socklen_t,
    ) -> UnsafePointer[C.char]:
        """Libc POSIX `inet_ntop` function.

        Args:
            af: Address Family see AF_ alises.
            src: A pointer to a binary address.
            dst: A pointer to a buffer to store the string representation of the
                address.
            size: The size of the buffer pointed by dst.

        Returns:
            A pointer to the buffer containing the text string if the conversion
            succeeds, and `NULL` otherwise, and set `errno` to indicate the
            error.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/inet_ntop.3p.html).
            Fn signature: `const char *inet_ntop(int af,
                const void *restrict src, char *restrict dst, socklen_t size)`.
        """

        @parameter
        if static:
            return external_call["inet_ntop", UnsafePointer[C.char]](
                af, src, dst, size
            )
        else:
            return self._lib.value().call["inet_ntop", UnsafePointer[C.char]](
                af, src, dst, size
            )

    fn inet_pton(
        self, af: C.int, src: UnsafePointer[C.char], dst: UnsafePointer[C.void]
    ) -> C.int:
        """Libc POSIX `inet_pton` function.

        Args:
            af: Address Family see AF_ alises.
            src: A pointer to a string representation of an address.
            dst: A pointer to a buffer to store the binary address.

        Returns:
            Returns 1 on success (network address was successfully converted). 0
            is returned if src does not contain a character string representing
            a valid network address in the specified address family.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/inet_ntop.3p.html).
            Fn signature: `int inet_pton(int af, const char *restrict src,
                void *restrict dst)`.
        """

        @parameter
        if static:
            return external_call["inet_pton", C.int](af, src, dst)
        else:
            return self._lib.value().call["inet_pton", C.int](af, src, dst)

    fn inet_addr(self, cp: UnsafePointer[C.char]) -> in_addr_t:
        """Libc POSIX `inet_addr` function.

        Args:
            cp: A pointer to a string representation of an address.

        Returns:
            If the input is invalid, INADDR_NONE (usually -1) is returned. Use
            of this function is problematic because -1 is a valid address
            `(255.255.255.255)`. Avoid its use in favor of inet_aton(),
            inet_pton(3), or getaddrinfo(3), which provide a cleaner way to
            indicate error return.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/inet_addr.3p.html).
            Fn signature: `in_addr_t inet_addr(const char *cp)`.
        """

        @parameter
        if static:
            return external_call["inet_addr", in_addr_t](cp)
        else:
            return self._lib.value().call["inet_addr", in_addr_t](cp)

    fn inet_aton(
        self, cp: UnsafePointer[C.char], addr: UnsafePointer[in_addr]
    ) -> C.int:
        """Libc POSIX `inet_aton` function.

        Args:
            cp: A pointer to a string representation of an address.
            addr: A pointer to a binary address.

        Returns:
            Value 1 if successful, 0 if the string is invalid.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/inet_aton.3p.html).
            Fn signature: `int inet_aton(const char *cp, struct in_addr *inp)`.
        """

        @parameter
        if static:
            return external_call["inet_aton", C.int](cp, addr)
        else:
            return self._lib.value().call["inet_aton", C.int](cp, addr)

    fn inet_ntoa(self, addr: in_addr) -> UnsafePointer[C.char]:
        """Libc POSIX `inet_ntoa` function.

        Args:
            addr: A pointer to a binary address.

        Returns:
            A pointer to the string in IPv4 dotted-decimal notation.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/inet_addr.3p.html).
            Fn signature: `char *inet_ntoa(struct in_addr in)`.
            Allocated buffer is 16-18 bytes depending on implementation.
        """

        @parameter
        if static:
            return external_call["inet_ntoa", UnsafePointer[C.char]](addr)
        else:
            return self._lib.value().call["inet_ntoa", UnsafePointer[C.char]](
                addr
            )

    fn socket(self, domain: C.int, type: C.int, protocol: C.int) -> C.int:
        """Libc POSIX `socket` function.

        Args:
            domain: Address Family see AF_ alises.
            type: Socket Type see SOCK_ alises.
            protocol: Protocol see IPPROTO_ alises.

        Returns:
            A file descriptor for the socket.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/socket.3p.html).
            Fn signature: `int socket(int domain, int type, int protocol)`.
        """

        @parameter
        if static:
            return external_call["socket", C.int](domain, type, protocol)
        else:
            return self._lib.value().call["socket", C.int](
                domain, type, protocol
            )

    fn socketpair(
        self,
        domain: C.int,
        type: C.int,
        protocol: C.int,
        socket_vector: UnsafePointer[C.int],
    ) -> C.int:
        """Libc POSIX `socketpair` function.

        Args:
            domain: Address Family see AF_ alises.
            type: Socket Type see SOCK_ alises.
            protocol: Protocol see IPPROTO_ alises.
            socket_vector: A pointer of `C.int` of length 2 to store the file
                descriptors.

        Returns:
            Value `0` on success, `-1` on error and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/socketpair.3p.html).
            Fn signature: `int socketpair(int domain, int type, int protocol,
            int socket_vector[2])`.
        """

        @parameter
        if static:
            return external_call["socketpair", C.int](
                domain, type, protocol, socket_vector
            )
        else:
            return self._lib.value().call["socketpair", C.int](
                domain, type, protocol, socket_vector
            )

    fn setsockopt(
        self,
        socket: C.int,
        level: C.int,
        option_name: C.int,
        option_value: UnsafePointer[C.void],
        option_len: socklen_t,
    ) -> C.int:
        """Libc POSIX `setsockopt` function.

        Args:
            socket: The socket's file descriptor.
            level: Protocol Level see SOL_ alises.
            option_name: Option name see SO_ alises.
            option_value: A pointer to a buffer containing the option value.
            option_len: The **byte size** of the buffer pointed by option_value.

        Returns:
            Value `0` on success, `-1` on error and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/setsockopt.3p.html).
            Fn signature: `int setsockopt(int socket, int level,
                int option_name, const void *option_value, socklen_t option_len
                )`.
        """

        @parameter
        if static:
            return external_call["setsockopt", C.int](
                socket, level, option_name, option_value, option_len
            )
        else:
            return self._lib.value().call["setsockopt", C.int](
                socket, level, option_name, option_value, option_len
            )

    fn bind(
        self,
        socket: C.int,
        address: UnsafePointer[sockaddr],
        address_len: socklen_t,
    ) -> C.int:
        """Libc POSIX `bind` function.

        Args:
            socket: The socket's file descriptor.
            address: A pointer to the address.
            address_len: The length of the pointer.

        Returns:
            Value `0` on success, `-1` on error and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/bind.3p.html).
            Fn signature: `int bind(int socket, const struct sockaddr *address,
                socklen_t address_len)`.
        """

        @parameter
        if static:
            return external_call["bind", C.int](socket, address, address_len)
        else:
            return self._lib.value().call["bind", C.int](
                socket, address, address_len
            )

    fn listen(self, socket: C.int, backlog: C.int) -> C.int:
        """Libc POSIX `listen` function.

        Args:
            socket: The socket's file descriptor.
            backlog: The maximum length to which the queue of pending
                connections for socket may grow.

        Returns:
            Value `0` on success, `-1` on error and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/listen.3p.html).
            Fn signature: `int listen(int socket, int backlog)`.
        """

        @parameter
        if static:
            return external_call["listen", C.int, C.int, C.int](socket, backlog)
        else:
            return self._lib.value().call["listen", C.int, C.int, C.int](
                socket, backlog
            )

    fn accept(
        self,
        socket: C.int,
        address: UnsafePointer[sockaddr],
        address_len: UnsafePointer[socklen_t],
    ) -> C.int:
        """Libc POSIX `accept` function.

        Args:
            socket: The socket's file descriptor.
            address: A pointer to a buffer to store the address of the accepted
                socket.
            address_len: A pointer to a buffer to store the length of the
                address of the accepted socket.

        Returns:
            Value `0` on success, `-1` on error and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/accept.3p.html).
            Fn signature: `int accept(int socket,
                struct sockaddr *restrict address,
                socklen_t *restrict address_len);`.
        """

        @parameter
        if static:
            return external_call["accept", C.int](socket, address, address_len)
        else:
            return self._lib.value().call["accept", C.int](
                socket, address, address_len
            )

    fn connect(
        self,
        socket: C.int,
        address: UnsafePointer[sockaddr],
        address_len: socklen_t,
    ) -> C.int:
        """Libc POSIX `connect` function.

        Args:
            socket: The socket's file descriptor.
            address: A pointer of the address to connect to.
            address_len: The length of the address.

        Returns:
            Value `0` on success, `-1` on error and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/connect.3p.html).
            Fn signature: `int connect(int socket,
                const struct sockaddr *address,
                socklen_t address_len)`.
        """

        @parameter
        if static:
            return external_call["connect", C.int](socket, address, address_len)
        else:
            return self._lib.value().call["connect", C.int](
                socket, address, address_len
            )

    fn recv(
        self,
        socket: C.int,
        buffer: UnsafePointer[C.void],
        length: size_t,
        flags: C.int,
    ) -> ssize_t:
        """Libc POSIX `recv` function.

        Args:
            socket: The socket's file descriptor.
            buffer: A pointer to a buffer to store the recieved bytes.
            length: The amount of bytes to store in the buffer.
            flags: Specifies the type of message reception.

        Returns:
            The amount of bytes recieved. Value -1 on error and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/recv.3p.html).
            Fn signature: `ssize_t recv(int socket, void *buffer, size_t length,
                int flags)`.
        """

        @parameter
        if static:
            return external_call["recv", ssize_t](socket, buffer, length, flags)
        else:
            return self._lib.value().call["recv", ssize_t](
                socket, buffer, length, flags
            )

    fn recvfrom(
        self,
        socket: C.int,
        buffer: UnsafePointer[C.void],
        length: size_t,
        flags: C.int,
        address: UnsafePointer[sockaddr],
        address_len: UnsafePointer[socklen_t],
    ) -> ssize_t:
        """Libc POSIX `recvfrom` function.

        Args:
            socket: The socket's file descriptor.
            buffer: A pointer to a buffer to store the recieved bytes.
            length: The amount of bytes to store in the buffer.
            flags: Specifies the type of message reception.
            address: A pointer to a sockaddr to store the address of the sending
                socket.
            address_len: A pointer to a buffer to store the length of the
                address of the sending socket.

        Returns:
            The amount of bytes recieved. Value -1 on error and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/recvfrom.3p.html).
            Fn signature: `ssize_t recvfrom(int socket, void *restrict buffer,
                size_t length, int flags, struct sockaddr *restrict address,
                socklen_t *restrict address_len)`.
        """

        @parameter
        if static:
            return external_call["recvfrom", ssize_t](
                socket, buffer, length, flags, address, address_len
            )
        else:
            return self._lib.value().call["recvfrom", ssize_t](
                socket, buffer, length, flags, address, address_len
            )

    fn send(
        self,
        socket: C.int,
        buffer: UnsafePointer[C.void],
        length: size_t,
        flags: C.int,
    ) -> ssize_t:
        """Libc POSIX `send` function.

        Args:
            socket: The socket's file descriptor.
            buffer: Points to the buffer containing the message to send.
            length: Specifies the length of the message in bytes.
            flags: Specifies the type of message transmission.

        Returns:
            The number of bytes sent. Value -1 on error and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/send.3p.html).
            Fn signature: `ssize_t send(int socket, const void *buffer,
                size_t length, int flags)`.
        """

        @parameter
        if static:
            return external_call["send", ssize_t](socket, buffer, length, flags)
        else:
            return self._lib.value().call["send", ssize_t](
                socket, buffer, length, flags
            )

    fn sendto(
        self,
        socket: C.int,
        message: UnsafePointer[C.void],
        length: size_t,
        flags: C.int,
        dest_addr: UnsafePointer[sockaddr],
        dest_len: socklen_t,
    ) -> ssize_t:
        """Libc POSIX `sendto` function.

        Args:
            socket: The socket's file descriptor.
            message: A pointer to a buffer to store the address of the accepted
                socket.
            length: A pointer to a buffer to store the length of the address of
                the accepted socket.
            flags: A pointer to a buffer to store the length of the address of
                the accepted socket.
            dest_addr: A pointer to a buffer to store the length of the address
                of the accepted socket.
            dest_len: A pointer to a buffer to store the length of the address
                of the accepted socket.

        Returns:
            The number of bytes sent. Value -1 on error and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/sendto.3p.html).
            Fn signature: `ssize_t sendto(int socket, const void *message,
                size_t length, int flags, const struct sockaddr *dest_addr,
                socklen_t dest_len)`.
        """

        @parameter
        if static:
            return external_call["sendto", ssize_t](
                socket, message, length, flags, dest_addr, dest_len
            )
        else:
            return self._lib.value().call["sendto", ssize_t](
                socket, message, length, flags, dest_addr, dest_len
            )

    fn shutdown(self, socket: C.int, how: C.int = SHUT_RDWR) -> C.int:
        """Libc POSIX `shutdown` function.

        Args:
            socket: The socket's file descriptor.
            how: Specifies the type of shutdown: {`SHUT_RD`, `SHUT_WR`,
                `SHUT_RDWR`}.

        Returns:
            Value `0` on success, `-1` on error and `errno` is set.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/shutdown.3p.html).
            Fn signature: `int shutdown(int socket, int how)`.
        """

        @parameter
        if static:
            return external_call["shutdown", C.int](socket, how)
        else:
            return self._lib.value().call["shutdown", C.int](socket, how)

    # FIXME: res should be res: UnsafePointer[UnsafePointer[addrinfo]]
    fn getaddrinfo(
        self,
        nodename: UnsafePointer[C.char],
        servname: UnsafePointer[C.char],
        hints: UnsafePointer[addrinfo],
        res: UnsafePointer[C.ptr_addr],
    ) -> C.int:
        """Libc POSIX `getaddrinfo` function.

        Args:
            nodename: The node name.
            servname: The service name.
            hints: The hints.
            res: The Pointer to the Pointer to store the result.

        Returns:
            Value 0 on success, one of several errors otherwise.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/freeaddrinfo.3p.html).
            Fn signature: `int getaddrinfo(const char *restrict nodename,
                const char *restrict servname,
                const struct addrinfo *restrict hints,
                struct addrinfo **restrict res)`.
        """

        @parameter
        if static:
            return external_call["getaddrinfo", C.int](
                nodename, servname, hints, res
            )
        else:
            return self._lib.value().call["getaddrinfo", C.int](
                nodename, servname, hints, res
            )

    fn gai_strerror(self, ecode: C.int) -> UnsafePointer[C.char]:
        """Libc POSIX `gai_strerror` function.

        Args:
            ecode: An error code.

        Returns:
            A pointer to a text string describing an error value for the
            `getaddrinfo()` and `getnameinfo()` functions.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/gai_strerror.3p.html).
            Fn signature: `const char *gai_strerror(int ecode)`.
        """

        @parameter
        if static:
            return external_call["gai_strerror", UnsafePointer[C.char]](ecode)
        else:
            return self._lib.value().call[
                "gai_strerror", UnsafePointer[C.char]
            ](ecode)

    # ===------------------------------------------------------------------=== #
    # Utils
    # ===------------------------------------------------------------------=== #

    fn strlen(self, s: UnsafePointer[C.char]) -> size_t:
        """Libc POSIX `strlen` function.

        Args:
            s: A pointer to a C string.

        Returns:
            The length of the string.

        Notes:
            [Reference](https://man7.org/linux/man-pages/man3/strlen.3p.html).
            Fn signature: `size_t strlen(const char *s)`.
        """

        @parameter
        if static:
            return external_call["strlen", size_t](s)
        else:
            return self._lib.value().call["strlen", size_t](s)
