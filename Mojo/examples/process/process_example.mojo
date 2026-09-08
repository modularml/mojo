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

from std.collections import List, Optional
from std.os import Process
from std.sys._libc import SignalCodes
from std.time import sleep


def test_long_running_process_wait() raises:
    print("== Test: Wait for a long-running process ==")

    var command = "sleep"
    var arguments = List[String]()
    arguments.append("1")

    print("Running 'sleep 1'...")
    var process = Process.run(command, arguments)

    print("Waiting for process to finish...")
    var status = process.wait()

    if status.exit_code:
        print("Process finished with exit code:", status.exit_code.value())
    elif status.term_signal:
        print("Process terminated by signal:", status.term_signal.value())


def test_long_running_process_poll() raises:
    print("== Test: Poll a long-running process ==")

    var command = "sleep"
    var arguments = List[String]()
    arguments.append("2")

    print("Running 'sleep 2'...")
    var process = Process.run(command, arguments)

    print("Polling process status...")
    while not process.poll().has_exited():
        print("Process is still running...")
        sleep(0.5)

    var status = process.status.value()
    print("Process has exited.")
    if status.exit_code:
        print("Process finished with exit code:", status.exit_code.value())
    elif status.term_signal:
        print("Process terminated by signal:", status.term_signal.value())


def test_interrupt_process() raises:
    print("== Test: Interrupt a process (SIGINT) ==")

    var command = "sleep"
    var arguments = List[String]()
    arguments.append("5")

    print("Running 'sleep 5'...")
    var process = Process.run(command, arguments)

    print("Sleeping for 1 second before sending SIGINT...")
    sleep(1.0)

    if process.interrupt():
        print("Successfully sent SIGINT.")
    else:
        print("Failed to send SIGINT.")

    var status = process.wait()

    if status.term_signal:
        print("Process terminated by signal:", status.term_signal.value())
        if status.term_signal.value() == SignalCodes.INT:
            print("Termination signal was SIGINT, as expected.")
    elif status.exit_code:
        print("Process finished with exit code:", status.exit_code.value())


def test_kill_process() raises:
    print("== Test: Kill a process (SIGKILL) ==")

    var command = "sleep"
    var arguments = List[String]()
    arguments.append("5")

    print("Running 'sleep 5'...")
    var process = Process.run(command, arguments)

    print("Sleeping for 1 second before sending SIGKILL...")
    sleep(1.0)

    if process.kill():
        print("Successfully sent SIGKILL.")
    else:
        print("Failed to send SIGKILL.")

    var status = process.wait()

    if status.term_signal:
        print("Process terminated by signal:", status.term_signal.value())
        if status.term_signal.value() == SignalCodes.KILL:
            print("Termination signal was SIGKILL, as expected.")
    elif status.exit_code:
        print("Process finished with exit code:", status.exit_code.value())


def main() raises:
    test_long_running_process_wait()
    print("\n--------------------\n")
    test_long_running_process_poll()
    print("\n--------------------\n")
    test_interrupt_process()
    print("\n--------------------\n")
    test_kill_process()
