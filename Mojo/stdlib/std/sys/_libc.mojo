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
"""Implements low-level bindings to functions from the C standard library.

The functions in this module are intended to be thin wrappers around their
C standard library counterparts. These are used to implement higher level
functionality in the rest of the Mojo standard library.
"""

from std.ffi import (
    c_char,
    c_int,
    c_size_t,
    c_pid_t,
    external_call,
    get_errno,
    CStringSlice,
)
from std.sys import CompilationTarget

# ===-----------------------------------------------------------------------===#
# stdlib.h — core C standard library operations
# ===-----------------------------------------------------------------------===#


@always_inline
def free(ptr: MutPointer[NoneType, ...]):
    # manually construct the call to free and attach the
    # correct attributes
    __mlir_op.`pop.external_call`[
        func=__mlir_attr[`"free" : !kgen.string`],
        _type=None,
        argAttrs=__mlir_attr.`[{llvm.allocptr}]`,
        funcAttrs=__mlir_attr.`[["allockind", "4"], ["alloc-family", "malloc"]]`,
    ](ptr)


@always_inline
def free(ptr: OptionalPointer[mut=True, NoneType, MutAnyOrigin]):
    """Frees memory previously allocated by `malloc`, `calloc`, or `realloc`.

    This overload accepts an `Optional[Pointer]` because it is valid in
    C to call `free` on a null pointer (it is a no-op).

    Args:
        ptr: A pointer to the memory to free.
    """
    free(
        Pointer(to=ptr).unsafe_bitcast[
            Pointer[NoneType, UntrackedOrigin[mut=True]]
        ]()[]
    )


@always_inline
def exit(status: c_int):
    external_call["exit", NoneType](status)


# ===-----------------------------------------------------------------------===#
# stdio.h — input/output operations
# ===-----------------------------------------------------------------------===#

comptime FILE_ptr = OptionalPointer[NoneType, UntrackedOrigin[mut=True]]


@always_inline
def fdopen(fd: c_int, mode: CStringSlice[_]) -> FILE_ptr:
    return external_call["fdopen", FILE_ptr](fd, mode)


@always_inline
def fclose(stream: FILE_ptr) -> c_int:
    return external_call["fclose", c_int](stream)


@always_inline
def fflush(stream: FILE_ptr) -> c_int:
    return external_call["fflush", c_int](stream)


@always_inline
def popen(
    command: CStringSlice[_],
    type: CStringSlice[_],
) -> FILE_ptr:
    return external_call["popen", FILE_ptr](command, type)


@always_inline
def pclose(stream: FILE_ptr) -> c_int:
    return external_call["pclose", c_int](stream)


@always_inline
def setvbuf(
    stream: FILE_ptr,
    buffer: MutPointer[c_char, _],
    mode: c_int,
    size: c_size_t,
) -> c_int:
    return external_call["setvbuf", c_int](stream, buffer)


struct BufferMode:
    """
    Modes for use in `setvbuf` to control buffer output.
    """

    comptime buffered = 0
    """Equivalent to `_IOFBF`."""
    comptime line_buffered = 1
    """Equivalent to `_IOLBF`."""
    comptime unbuffered = 2
    """Equivalent to `_IONBF`."""


# ===-----------------------------------------------------------------------===#
# spawn.h - Spawn process
# ===-----------------------------------------------------------------------===#


@always_inline
def posix_spawnp[
    argv_origin: ImmOrigin,
    //,
](
    pid: MutPointer[c_pid_t, _],
    file: CStringSlice[_],
    argv: Pointer[Optional[CStringSlice[argv_origin]], _],
    envp: OptionalPointer[
        Optional[CStringSlice[ImmutAnyOrigin]], ImmutAnyOrigin
    ],
) -> c_int:
    """[`posix_spawn`](https://pubs.opengroup.org/onlinepubs/007904975/functions/posix_spawn.html)
    — function creates a new process (child process) from the specified process image.

    Args:
        pid: `Pointer` destination for the process id if spawned successfully.
        file: NUL-terminated C string (`CStringSlice`) with the path to the executable.
        argv: The argument array; must be terminated with a NULL (`None`) entry.
        envp: The environment array; must be terminated with a NULL (`None`) entry.
    """
    # TODO: Implement `const posix_spawn_file_actions_t`, `*file_actions, const posix_spawnattr_t *restrict attrp,`
    # to allow full control of how process is spawned
    return external_call["posix_spawnp", c_int](
        pid,
        file,
        OptionalPointer[NoneType, ImmUntrackedOrigin](),
        OptionalPointer[NoneType, ImmUntrackedOrigin](),
        argv,
        envp,
    )


@always_inline
def _get_environ() -> (
    OptionalPointer[Optional[CStringSlice[ImmutAnyOrigin]], ImmutAnyOrigin]
):
    """Returns the process environment pointer (POSIX `environ`).

    Returns:
        A pointer to the null-terminated array of environment strings,
        suitable for passing as the `envp` argument to `posix_spawnp`.
    """
    comptime _EnvpType = OptionalPointer[
        Optional[CStringSlice[ImmutAnyOrigin]], ImmutAnyOrigin
    ]
    comptime if CompilationTarget.is_macos():
        # _NSGetEnviron() from <crt_externs.h> returns char ***,
        # a pointer to the `environ` variable.
        return external_call[
            "_NSGetEnviron",
            OptionalPointer[_EnvpType, ImmUntrackedOrigin],
        ]().value()[]
    elif CompilationTarget.is_linux():
        # On Linux, look up `environ` via dlsym(RTLD_DEFAULT, "environ").
        # RTLD_DEFAULT is ((void *)0) on Linux.
        return dlsym[_EnvpType](
            OptionalPointer[NoneType, MutUntrackedOrigin](),
            "environ".as_c_string_slice().ptr(),
        ).value()[]
    else:
        CompilationTarget.unsupported_target_error[operation="_get_environ"]()


# ===-----------------------------------------------------------------------===#
# unistd.h
# ===-----------------------------------------------------------------------===#


@always_inline
def dup(oldfd: c_int) -> c_int:
    return external_call["dup", c_int](oldfd)


@always_inline
def dup2(oldfd: c_int, newfd: c_int) -> c_int:
    return external_call["dup2", c_int](oldfd, newfd)


@always_inline
def execvp[
    origin: ImmOrigin,
    //,
](
    file: ImmPointer[c_char, _],
    argv: ImmPointer[OptionalPointer[mut=False, c_char, origin], _],
) -> c_int:
    """[`execvp`](https://pubs.opengroup.org/onlinepubs/9799919799/functions/exec.html)
    — execute a file.

    Args:
        file: NUL-terminated C string (`Pointer` to `c_char`) with the path to the executable.
        argv: The argument array; must be terminated with a NULL pointer.
    """
    return external_call["execvp", c_int](file, argv)


@always_inline
def vfork() -> c_int:
    """[`vfork()`](https://pubs.opengroup.org/onlinepubs/009696799/functions/vfork.html).
    """
    return external_call["vfork", c_int]()


struct SignalCodes:
    comptime HUP = 1  # (hang up)
    comptime INT = 2  # (interrupt)
    comptime QUIT = 3  # (quit)
    comptime ABRT = 6  # (abort)
    comptime KILL = 9  # (non-catchable, non-ignorable kill)
    comptime ALRM = 14  # (alarm clock)
    comptime TERM = 15  # (software termination signal)


@always_inline
def kill(pid: c_int, sig: c_int) -> c_int:
    """[`kill()`](https://pubs.opengroup.org/onlinepubs/9799919799/functions/kill.html)
    — send a signal to a process or group of processes."""
    return external_call["kill", c_int](pid, sig)


@always_inline
def pipe(fildes: MutPointer[c_int, _]) -> c_int:
    """[`pipe()`](https://pubs.opengroup.org/onlinepubs/9799919799/functions/pipe.html) — create an interprocess channel.
    """
    return external_call["pipe", c_int](fildes)


@always_inline
def close(fd: c_int) -> c_int:
    """[`close()`](https://pubs.opengroup.org/onlinepubs/9799919799/functions/close.html)
    — close a file descriptor.
    """
    return external_call["close", c_int](fd)


@always_inline
def write(fd: c_int, buf: ImmOpaquePointer[_], nbyte: c_size_t) -> c_int:
    """[`write()`](https://pubs.opengroup.org/onlinepubs/9799919799/functions/write.html)
    — write to a file descriptor.
    """
    return external_call["write", c_int](fd, buf, nbyte)


# ===-----------------------------------------------------------------------===#
# sys/wait.h - Control over file descriptors
# ===-----------------------------------------------------------------------===#


struct WaitFlags:
    """Flags for `waitpid`."""

    comptime WNOHANG: c_int = 1


# pid_t waitpid(pid_t pid, int *wstatus, int options);
@always_inline
def waitpid(
    pid: c_pid_t,
    status: MutPointer[c_int, _],
    options: c_int,
) -> c_pid_t:
    """[`waitpid()`](https://pubs.opengroup.org/onlinepubs/9799919799/functions/waitpid.html)
    — Wait on child process to finish executing.
    """
    return external_call["waitpid", c_pid_t](pid, status, options)


# ===-----------------------------------------------------------------------===#
# fcntl.h - Control over file descriptors
# ===-----------------------------------------------------------------------===#


struct FcntlCommands:
    comptime F_GETFD: c_int = 1
    comptime F_SETFD: c_int = 2


struct FcntlFDFlags:
    comptime FD_CLOEXEC: c_int = 1


@always_inline
def fcntl[*types: Intable](fd: c_int, cmd: c_int, *args: *types) -> c_int:
    """[`fcntl()`](https://pubs.opengroup.org/onlinepubs/9799919799/functions/fcntl.html)
    — file control.
    """
    # The pack is loaded so the variadic arguments are the values themselves
    # rather than references to them.
    return external_call["fcntl", c_int, num_fixed_args=2](
        fd, cmd, args.get_loaded_kgen_pack()
    )


# ===-----------------------------------------------------------------------===#
# dlfcn.h — dynamic library operations
# ===-----------------------------------------------------------------------===#


@always_inline
def dlerror(out result: OptionalPointer[c_char, MutUntrackedOrigin]):
    result = external_call["dlerror", type_of(result)]()


@always_inline
def dlopen(
    filename: OptionalPointer[mut=False, c_char, ImmUntrackedOrigin],
    flags: c_int,
) -> OptionalPointer[NoneType, MutUntrackedOrigin]:
    return external_call[
        "dlopen", OptionalPointer[NoneType, MutUntrackedOrigin]
    ](filename, flags)


@always_inline
def dlclose(handle: OptionalPointer[mut=True, NoneType, _]) -> c_int:
    return external_call["dlclose", c_int](handle)


@always_inline
def dlsym[
    # Default `dlsym` result is an OpaquePointer.
    result_type: AnyType = NoneType
](
    handle: OptionalPointer[NoneType, _],
    name: ImmPointer[c_char, _],
    out result: OptionalPointer[result_type, MutUntrackedOrigin],
):
    result = external_call["dlsym", type_of(result)](handle, name)


def realpath(
    path: CStringSlice[_],
    resolved_path: MutPointer[c_char, _],
    out result: OptionalPointer[c_char, MutUntrackedOrigin],
):
    """Expands all symbolic links and resolves references to /./, /../ and extra
    '/' characters in the null-terminated string named by path to produce a
    canonicalized absolute pathname.  The resulting pathname is stored as a
    null-terminated string, up to a maximum of PATH_MAX bytes, in the buffer
    pointed to by resolved_path.  The resulting path will have no symbolic link,
    /./ or /../ components.

    libc also accepts a NULL `resolved_path`, in which case it allocates the
    result buffer itself. This binding does not expose that mode, so the caller
    always owns the destination buffer.

    Args:
        path: The path to resolve.
        resolved_path: The buffer to store the resolved path. It must be at
            least `PATH_MAX` bytes long.

    Returns:
        A pointer to the resolved path.
    """
    return external_call["realpath", type_of(result)](path, resolved_path)
