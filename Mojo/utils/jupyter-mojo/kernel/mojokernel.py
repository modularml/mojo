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
#
# This file contains an implementation of a Jupyter kernel for Mojo. It
# communicates to Mojo using the MojoJupyter API library.
#
# ===----------------------------------------------------------------------=== #


import argparse
import ctypes
import json
import os
import time
import traceback
from configparser import ConfigParser
from enum import IntEnum
from pathlib import Path
from typing import Any

from ipykernel.kernelapp import IPKernelApp
from ipykernel.kernelbase import Kernel

# ===----------------------------------------------------------------------=== #
# OutputProcessor
# ===----------------------------------------------------------------------=== #

# Special start and end output markers that denote a display message.
display_start = "%%%%%%%DISPLAY_START"
display_end = "%%%%%%%DISPLAY_END"


class ExecutionFinishedState(IntEnum):
    """
    Copied from MojoJupyter/Kernel.cpp - models the possible states of a kernel execution.
    """

    NotFinished = 0
    FinishedSuccess = 1
    FinishedError = 2


class OutputProcessor:
    """
    Process the output coming from stdout/stderr to find special markers that
    indicate specific messages need to be sent.
    """

    def __init__(self, kernel: Kernel) -> None:
        # The kernel object that we are processing output for.
        self.kernel = kernel

        # The current contents of a pending display message.
        self.pending_display_message = None

    def send_message(self, name: str, text: str) -> None:
        """
        Send a stream message to the client. `name` should be stderr or stderr.
        """
        stream_content = {
            "name": name,
            "text": text,
        }
        self.kernel.log.info(stream_content)
        self.kernel.send_response(
            self.kernel.iopub_socket, "stream", stream_content
        )

    # Create the output callback function. This is called by the MojoJupyter
    # library to send output back to the Jupyter client.
    def process_output(self, name: bytes, msg: bytes) -> None:
        """Process output from the Mojo kernel."""
        msgstr = msg.decode()

        def send_stream(text: str) -> None:
            self.send_message(name.decode(), text)

        # Short-circuit stderr right away - we don't report images or anything
        # through stderr, and we should always report it immediately.
        if name.decode() == "stderr":
            send_stream(msgstr)
            return

        # If we don't have a pending display message, and we don't have a
        # display start marker, then just send the output as a normal stream message.
        display_marker_loc = msgstr.find(display_start)
        if self.pending_display_message is None and display_marker_loc == -1:
            send_stream(msgstr)
            return

        while len(msgstr) > 0:
            # Buffer the display message, sending the correct bits to the stdout.
            if self.pending_display_message is None:
                # We need this for the case where we've just come back around from sending a display message.
                if display_marker_loc == -1:
                    send_stream(msgstr)
                    msgstr = ""
                    continue

                # Send the beginning of the message to the stream output.
                if display_marker_loc > 0:
                    send_stream(msgstr[:display_marker_loc])
                # Meanwhile, set up for the display message.
                self.pending_display_message = ""
                msgstr = msgstr[display_marker_loc + len(display_start) :]
                continue
            # endif self.pending_display_message is None

            # Try to find the end marker. If we did find it, add it to the
            # pending display message and send it out.
            display_marker_loc = msgstr.find(display_end)
            if display_marker_loc != -1:
                self.pending_display_message += msgstr[:display_marker_loc]
                self._send_display_message()
                # Reset display_marker_loc to a potential display_start index
                # and reset msgstr so things get sent properly.
                msgstr = msgstr[display_marker_loc + len(display_end) :]
                display_marker_loc = msgstr.find(display_start)
                continue
            # endif display_marker_loc != -1

            self.pending_display_message += msgstr
            # If we just completed the display end message, send it out.
            display_marker_loc = self.pending_display_message.find(display_end)
            if display_marker_loc == -1:
                msgstr = ""
                continue
            self.pending_display_message = self.pending_display_message[
                :display_marker_loc
            ]
            # Discard the display end message.
            msgstr = self.pending_display_message[
                display_marker_loc + len(display_end) :
            ].rstrip("\r\n")
            self._send_display_message()

            # Finally, flush the stream from the message.
            send_stream(msgstr)
        # endwhile len(msgstr) > 0

    def _send_display_message(self) -> None:
        """Send the current pending display message to the client."""

        display_message = json.loads(self.pending_display_message)
        self.kernel.send_response(
            self.kernel.iopub_socket,
            "display_data",
            {
                "data": display_message[0],
                "metadata": display_message[1],
            },
        )
        self.pending_display_message = None


# ===----------------------------------------------------------------------=== #
# MojoKernel
# ===----------------------------------------------------------------------=== #


class MojoKernel(Kernel):
    """A Jupyter kernel for Mojo."""

    def __init__(self, **kwargs) -> None:
        """Initialize the Mojo kernel.

        This loads the MojoJupyter library and starts a kernel repl session.
        """
        # Kernel Metadata.
        self.implementation = "MojoKernel"
        self.implementation_version = "0.1"
        self.language = "mojo"
        self.language_version = "0.1"
        self.language_info = {
            "name": "mojo",
            "mimetype": "text/x-mojo",
            "file_extension": ".mojo",
            "codemirror_mode": {"name": "mojo"},
        }
        self.banner = ""
        self.auto_gen_cell_id_count = 0
        super().__init__(**kwargs)

        # Load the MojoJupyter library, and initialize the result types of the
        # functions we use.
        self.lib_mojo_jupyter: ctypes.CDLL = self.load_mojo_lib()
        self.lib_mojo_jupyter.initMojoKernel.restype = ctypes.c_void_p
        self.lib_mojo_jupyter.checkMojoExecutionFinished.restype = ctypes.c_int

        # The type of the output callback function. It takes a name and a
        # message.
        self.output_callback_type: ctypes.CFUNCTYPE = ctypes.CFUNCTYPE(
            None,
            ctypes.c_char_p,
            ctypes.c_char_p,
        )
        self.output_processor = OutputProcessor(self)
        self.output_callback = self.output_callback_type(
            lambda name, msg: self.output_processor.process_output(name, msg)
        )

        self.mojo_kernel: ctypes.c_void_p = (
            self.lib_mojo_jupyter.initMojoKernel(
                self.output_callback,
                ctypes.c_char_p(self.mojoReplExe.encode("utf-8")),
                ctypes.c_char_p(os.getcwd().encode("utf-8")),
                ctypes.c_char_p(None),  # lldbInitFile
            )
        )
        if not self.mojo_kernel:
            raise RuntimeError("Unable to initialize Mojo kernel.")

    def _send_internal_error_message(self) -> None:
        self.output_processor.send_message(
            "stderr",
            (
                "Internal Mojo Kernel Error\n\nThe Jupyter Notebook encountered"
                " an internal error and was unable to evaluate the provided"
                " expression. Please report this issue.\n\nMore information"
                " about this error can be found in the server error log."
            ),
        )

    def __del__(self) -> None:
        """Destroy the Mojo kernel."""
        self.lib_mojo_jupyter.destroyMojoKernel(self.mojo_kernel)

    def load_mojo_lib(self) -> ctypes.CDLL:
        """Load the libMojoJupyter library.

        On success, this initializes `mojoReplExe` returns the loaded library.
        """

        # Grab the mojo repl executable from the config.
        config = ConfigParser()
        config.read(Path(os.environ["MODULAR_HOME"]) / "modular.cfg")
        mojoReplExePath = Path(
            config.get("mojo-max", "repl_entry_point").rstrip(";")
        )
        if not mojoReplExePath.exists():
            raise RuntimeError(
                "Unable to locate `mojo-repl-entry-point` executable."
            )
        self.mojoReplExe = str(mojoReplExePath)

        # Make sure the lib directory is in the path.
        libDir = mojoReplExePath.parent
        os.environ["PATH"] += os.pathsep + str(libDir)

        # Load the MojoJupyter library. This library provides the internal
        # implementation of the kernel.
        mojoJupyterPath = Path(
            config.get("mojo-max", "jupyter_path").rstrip(";")
        )
        if not mojoJupyterPath.exists():
            raise RuntimeError("Unable to locate `MojoJupyter` library.")
        return ctypes.cdll.LoadLibrary(str(mojoJupyterPath))

    def do_execute(  # noqa: ANN201
        self,
        code: str,
        silent: bool = False,
        store_history: bool = True,
        user_expressions: dict[str, Any] | None = None,
        allow_stdin: bool = False,
        *,
        cell_id: str | None = None,
    ):
        """Execute a code cell."""
        # TODO: Better propagate errors from the kernel execution, process
        # provided arguments, etc.

        # Wait for the currently running execution to finish.
        def wait_for_execution() -> ExecutionFinishedState:
            # Wait for the execution to finish.
            while True:
                # Sleep for a bit to avoid busy spinning while waiting for the
                # execution to finish.
                time.sleep(0.05)

                # Poll the kernel to see if the execution has finished.
                result: int = self.lib_mojo_jupyter.checkMojoExecutionFinished(
                    ctypes.c_void_p(self.mojo_kernel),
                )
                if result != ExecutionFinishedState.NotFinished:
                    return ExecutionFinishedState(result)

        try:
            # jupyter on the cli doesn't provide a cell id, so we need to
            # autogenerate one.
            if cell_id is None:
                cell_id = f"__autogen_cell_id_{self.auto_gen_cell_id_count}"
                self.auto_gen_cell_id_count += 1

            # Start execution of the expression.
            finish_state = self.lib_mojo_jupyter.startMojoExecution(
                ctypes.c_void_p(self.mojo_kernel),
                ctypes.c_char_p(cell_id.encode("utf-8")),
                ctypes.c_char_p(code.encode("utf-8")),
                ctypes.c_int(store_history),
            )
            if finish_state == ExecutionFinishedState.NotFinished:
                finish_state = wait_for_execution()
            if finish_state == ExecutionFinishedState.FinishedError:
                return {"status": "error"}

            return {
                "status": "ok",
                "execution_count": self.execution_count,
                "payload": [],
                "user_expressions": {},
            }
        except KeyboardInterrupt:
            # Interrupt the current kernel execution.
            self.lib_mojo_jupyter.interruptMojoExecution(
                ctypes.c_void_p(self.mojo_kernel)
            )
            wait_for_execution()

            # TODO: When Mojo actually has debug info again, we should emit the
            # current stack frame here.
            return {
                "status": "error",
                "execution_count": self.execution_count,
            }

        except:
            traceback.print_exc()
            self._send_internal_error_message()

            return {
                "status": "error",
                "execution_count": self.execution_count,
            }

    def do_complete(self, code: str, cursor_pos: int):  # noqa: ANN201
        """Find code completions for the given code and cursor position."""

        # The type of the completion function, it takes a completion label.
        completion_callback_type: ctypes.CFUNCTYPE = ctypes.CFUNCTYPE(
            None, ctypes.c_char_p
        )

        # Build the callback handler used to process completion results.
        results = []
        completion_callback = completion_callback_type(
            lambda result: results.append(result.decode())
        )

        self.lib_mojo_jupyter.checkMojoCodeComplete(
            ctypes.c_void_p(self.mojo_kernel),
            ctypes.c_char_p(code.encode()),
            ctypes.c_int(cursor_pos),
            completion_callback,
        )

        return {
            "matches": results,
            "cursor_end": cursor_pos,
            "cursor_start": cursor_pos,
            "metadata": {},
            "status": "ok",
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--modular-home",
        required=True,
        help="The value of the env var MODULAR_HOME.",
    )
    args, jupyter_args = parser.parse_known_args()

    os.environ["MODULAR_HOME"] = args.modular_home

    # We pass the kernel name as a command-line arg, since Jupyter gives those
    # highest priority (in particular overriding any system-wide config).
    IPKernelApp.launch_instance(
        argv=jupyter_args + ["--IPKernelApp.kernel_class=__main__.MojoKernel"]
    )
