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
"""Implements os methods for dealing with processes.

Example:

```mojo
from std.os import Process
from std.collections import List
_ = Process.run("echo", ["== TEST_ECHO"])
```
"""
from std.collections import List, Optional
from std.collections.string import StringSlice
from std.sys import CompilationTarget
from std.sys._libc import (
    waitpid,
    posix_spawnp,
    _get_environ,
    kill,
    SignalCodes,
    pipe,
    fcntl,
    FcntlCommands,
    FcntlFDFlags,
    close,
    WaitFlags,
)
from std.ffi import c_char, c_int, c_pid_t, get_errno, CStringSlice
from .os import abort, sep


# ===----------------------------------------------------------------------=== #
# Process comm.
# ===----------------------------------------------------------------------=== #


struct ProcessStatus(Copyable, ImplicitlyCopyable, Movable):
    """Represents the termination status of a process.

    This struct is returned by `poll()` and `wait()`.
    """

    var exit_code: Optional[Int]
    """The exit code if the process terminated normally."""

    var term_signal: Optional[Int]
    """The signal number that terminated the process."""

    def __init__(
        out self,
        exit_code: Optional[Int] = None,
        term_signal: Optional[Int] = None,
    ):
        """Initializes a new `ProcessStatus`.

        Args:
            exit_code: The exit code if the process terminated normally.
            term_signal: The signal number that terminated the process.
        """
        self.exit_code = exit_code
        self.term_signal = term_signal

    @staticmethod
    def running() -> Self:
        """Creates a status for a running process.

        Returns:
            A `ProcessStatus` for a running process.
        """
        return Self()

    def has_exited(self) -> Bool:
        """Checks if the process has terminated.

        Returns:
            True if the process has terminated, either normally or by a signal.
        """
        return Bool(self.exit_code) or Bool(self.term_signal)


struct Pipe:
    """Create a pipe for interprocess communication.

    Example usage:
    ```mojo
    from std.os.process import Pipe

    def main() raises:
        var pipe = Pipe()
        pipe.write_bytes("TEST".as_bytes())
    ```
    """

    var fd_in: Optional[FileDescriptor]
    """File descriptor for pipe input."""
    var fd_out: Optional[FileDescriptor]
    """File descriptor for pipe output."""

    def __init__(
        out self,
        in_close_on_exec: Bool = True,
        out_close_on_exec: Bool = True,
    ) raises:
        """Initializes a new `Pipe`.

        Args:
            in_close_on_exec: Close the read side of pipe if an `exec` syscall
              is issued in the process.
            out_close_on_exec: Close the write side of pipe if an `exec`
              syscall is issued in the process.

        Raises:
            Error: If the pipe could not be created or configured.
        """
        var pipe_fds = Array[c_int, 2](fill=0)
        if pipe(pipe_fds.unsafe_ptr()) < 0:
            raise Error("Failed to create pipe")

        if in_close_on_exec:
            if not self._set_close_on_exec(pipe_fds[0]):
                _ = close(pipe_fds[0])
                _ = close(pipe_fds[1])
                raise Error("Failed to configure input pipe close on exec")

        if out_close_on_exec:
            if not self._set_close_on_exec(pipe_fds[1]):
                _ = close(pipe_fds[0])
                _ = close(pipe_fds[1])
                raise Error("Failed to configure output pipe close on exec")

        self.fd_in = FileDescriptor(Int(pipe_fds[0]))
        self.fd_out = FileDescriptor(Int(pipe_fds[1]))

    def __deinit__(deinit self):
        """Ensures pipes input and output file descriptors are closed, when the object is destroyed.
        """
        self.set_input_only()
        self.set_output_only()

    @staticmethod
    def _set_close_on_exec(fd: c_int) -> Bool:
        return (
            fcntl(
                fd,
                FcntlCommands.F_SETFD,
                fcntl(fd, FcntlCommands.F_GETFD, 0) | FcntlFDFlags.FD_CLOEXEC,
            )
            == 0
        )

    @always_inline
    def set_input_only(mut self):
        """Close the output descriptor/ channel for this side of the pipe."""
        if self.fd_out:
            _ = close(Int32(rebind[Int](self.fd_out.value())))
            self.fd_out = None

    @always_inline
    def set_output_only(mut self):
        """Close the input descriptor/ channel for this side of the pipe."""
        if self.fd_in:
            _ = close(Int32(rebind[Int](self.fd_in.value())))
            self.fd_in = None

    @always_inline
    def write_bytes(mut self, bytes: Span[Byte, _]) raises:
        """Writes a span of bytes to the pipe.

        Args:
            bytes: The byte span to write to this pipe.

        Raises:
            Error: If called on a read-only pipe.
        """
        if self.fd_out:
            self.fd_out.value().write_bytes(bytes)
        else:
            raise Error("Can not write from read only side of pipe")

    @always_inline
    def read_bytes(mut self, buffer: MutSpan[Byte, _]) raises -> Int:
        """Read a number of bytes from this pipe.

        Args:
            buffer: Span[Byte] of length n where to store read bytes. n = number of bytes to read.

        Returns:
            Actual number of bytes read.

        Raises:
            Error: If the pipe is in write-only mode.
        """
        if self.fd_in:
            return self.fd_in.value().read_bytes(buffer)

        raise Error("Can not read from write only side of pipe")


# ===----------------------------------------------------------------------=== #
# Process execution
# ===----------------------------------------------------------------------=== #
struct Process:
    """Create and manage child processes from file executables.

    Example usage:
    ```mojo
    from std.os.process import Process

    def main() raises:
        var child_process = Process.run("ls", ["-lha"])
        if child_process.interrupt():
            print("Successfully interrupted.")
    ```
    """

    var child_pid: c_pid_t
    """Child process id."""

    var status: Optional[ProcessStatus]
    """Cached status of the process. `None` if the process has not been waited on yet."""

    def __init__(out self, child_pid: c_pid_t):
        """Struct to manage metadata about child process.
        Use the `run` static method to create new process.

        Args:
          child_pid: The pid of child process returned by `posix_spawnp` that the struct will manage.
        """

        self.child_pid = child_pid
        self.status = None

    def __deinit__(deinit self):
        """Waits for the process to exit when the `Process` object is destroyed.
        """
        try:
            _ = self.wait()
        except:
            # Errors in __deinit__ should be suppressed.
            pass

    def _kill(mut self, signal: Int) -> Bool:
        # We need to check the "cached" status to avoid trying to
        # kill a process after already having waited upon its exit.
        # Such a process no longer exists and its pid could be reused
        if self.status and self.status.value().has_exited():
            return False

        # `kill` returns 0 on success and -1 on failure
        return kill(Int32(self.child_pid), Int32(signal)) > -1

    def _check_status(
        self, pid: c_pid_t, status: c_int
    ) raises -> ProcessStatus:
        """Helper to decode the result of a waitpid call.

        The decoding logic is a direct implementation of the standard C macros
        used to interpret the `status` integer returned by `waitpid`. These
        macros are defined in `<sys/wait.h>` on POSIX systems.

        This implementation is based on the definitions found in `musl` libc.:
        https://git.musl-libc.org/cgit/musl/tree/include/sys/wait.h

        The core logic relies on the following macro definitions:
        - `#define WEXITSTATUS(s) (((s) & 0xff00) >> 8)`
        - `#define WTERMSIG(s)    ((s) & 0x7f)`
        - `#define WIFEXITED(s)   (WTERMSIG(s) == 0)`

        Note on Endianness:
        This logic is endianness-independent. The `waitpid` status is an integer
        value provided by the kernel. All bitwise operations (`&`, `>>`) are
        performed on this integer's numerical value, not its byte representation
        in memory. The result is therefore consistent across architectures.
        """
        if pid == self.child_pid:
            # Process has terminated. Decode the status.
            if (status & 0x7F) == 0:
                # Process exited normally. Extract the exit code.
                var code = (status & 0xFF00) >> 8
                return ProcessStatus(exit_code=Optional(Int(code)))
            else:
                # Process was terminated by a signal. Extract the signal number.
                var signal = status & 0x7F
                return ProcessStatus(term_signal=Optional(Int(signal)))
        elif pid == 0:
            # Process is still running (only for non-blocking calls).
            return ProcessStatus.running()
        else:
            # An error occurred.
            var err = get_errno()
            raise Error("waitpid failed with errno " + String(err))

    def hangup(mut self) -> Bool:
        """Send the Hang up signal to the managed child process.

        Returns:
          Upon successful completion, True is returned else False.
        """
        return self._kill(SignalCodes.HUP)

    def interrupt(mut self) -> Bool:
        """Send the Interrupt signal to the managed child process.

        Returns:
          Upon successful completion, True is returned else False.
        """
        return self._kill(SignalCodes.INT)

    def kill(mut self) -> Bool:
        """Send the Kill signal to the managed child process.

        Returns:
          Upon successful completion, True is returned else False.
        """
        return self._kill(SignalCodes.KILL)

    def poll(mut self) raises -> ProcessStatus:
        """Check if the child process has terminated in a non-blocking way.

        This method updates the internal state of the `Process` object.
        If the process has terminated, the status is cached.

        Returns:
            A `ProcessStatus` indicating the status of the process.

        Raises:
            Error: If `waitpid` fails.
        """
        if self.status:
            return self.status.value()

        var status: c_int = 0
        var pid = waitpid(self.child_pid, Pointer(to=status), WaitFlags.WNOHANG)
        var result = self._check_status(pid, status)
        if result.has_exited():
            self.status = result
        return result

    def wait(mut self) raises -> ProcessStatus:
        """Wait for the child process to terminate (blocking).

        This method updates the internal state of the `Process` object.
        If the process has terminated, the status is cached.

        Returns:
          A `ProcessStatus` indicating the process has exited and its status.

        Raises:
            Error: If `waitpid` fails or the process does not exit.
        """
        if self.status:
            return self.status.value()

        var status: c_int = 0
        var pid = waitpid(self.child_pid, Pointer(to=status), 0)
        var result = self._check_status(pid, status)
        if result.has_exited():
            self.status = result
        else:
            # This should not be reachable with a blocking waitpid call.
            raise Error("Blocking waitpid returned without process exiting.")
        return result

    @staticmethod
    def run(var path: String, argv: List[String]) raises -> Process:
        """Spawn new process from file executable.

        Args:
          path: The path to the file.
          argv: A list of string arguments to be passed to executable.

        Returns:
          An instance of `Process` struct.

        Raises:
            Error: If the process fails to spawn.
        """

        comptime assert (
            CompilationTarget.is_linux() or CompilationTarget.is_macos()
        ), "Unknown platform process execution not implemented"
        var parts = path.split(sep)
        var file_name = String(parts[len(parts) - 1])

        var arg_count = len(argv)
        var argv_array_ptr_cstr_ptr = List[
            Optional[CStringSlice[ImmutAnyOrigin]]
        ](
            length=arg_count + 2,
            fill={},
        )
        var offset = 0
        # Arg 0 in `argv` ptr array should be the file name
        argv_array_ptr_cstr_ptr[offset] = rebind[CStringSlice[ImmutAnyOrigin]](
            file_name.as_c_string_slice()
        )
        offset += 1

        for var arg in argv:
            argv_array_ptr_cstr_ptr[offset] = rebind[
                CStringSlice[ImmutAnyOrigin]
            ](arg.as_c_string_slice())
            offset += 1

        # `argv` ptr array terminates with NULL PTR
        argv_array_ptr_cstr_ptr[offset] = {}

        var pid: c_pid_t = 0

        var has_error_code = posix_spawnp(
            Pointer(to=pid),
            path.as_c_string_slice(),
            # Safety: `argv_array_ptr_cstr_ptr` has at least 2 elements so is non-null
            argv_array_ptr_cstr_ptr.unsafe_ptr(),
            _get_environ(),  # inherit parent's environment
        )

        if has_error_code > 0:
            raise Error(
                t"Failed to execute {path}, EINT error code: {has_error_code}"
            )

        return Process(child_pid=pid)
