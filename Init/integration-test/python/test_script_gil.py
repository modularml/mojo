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
import faulthandler
import signal
import sys

import init_bindings


def deep_function_3() -> None:
    """Third level function that triggers the segfault with GIL held."""
    print(
        "About to trigger SIGSEGV from C++ with GIL explicitly held",
        file=sys.stderr,
    )
    # Trigger a segfault from C++ while the GIL is explicitly held
    # This tests the most dangerous scenario for signal handlers
    init_bindings.trigger_segfault_with_gil_held()


def deep_function_2() -> None:
    """Second level function in the call stack."""
    deep_function_3()


def deep_function_1() -> None:
    """First level function in the call stack."""
    deep_function_2()


def main() -> None:
    """Main function that sets up the signal handler and triggers the crash."""
    print("Starting GIL-held signal handler test", file=sys.stderr)

    # Register faulthandler for SIGUSR2 to enable async-safe Python stack traces
    # Note: chain=False so it doesn't interfere with our signal handling flow
    faulthandler.register(
        signal.SIGUSR2, file=sys.stderr, all_threads=True, chain=False
    )
    print("Python faulthandler registered for SIGUSR2", file=sys.stderr)

    # Initialize the C++ signal handler
    init_bindings.initialize_signal_handler("test_gil_signal_handler")
    print("Signal handler initialized", file=sys.stderr)

    # Call the function chain that will trigger the signal while GIL is held
    deep_function_1()


if __name__ == "__main__":
    main()
