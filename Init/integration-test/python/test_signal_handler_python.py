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
import os
import signal
import subprocess
import sys


def test_python_stack_trace_in_signal_handler() -> None:
    """Test that Python stack traces are captured in signal handler output."""

    # Use the test script file in the same directory
    current_dir = os.path.dirname(__file__)
    test_file = os.path.join(current_dir, "test_script.py")

    if not os.path.exists(test_file):
        raise AssertionError(f"Test script not found: {test_file}")

    print(f"Running test script: {test_file}")

    # Run the test script and capture output
    result = subprocess.run(
        [sys.executable, test_file],
        capture_output=True,
        text=True,
        timeout=10,
    )

    print(f"Process exit code: {result.returncode}")
    print(f"Expected exit code: {128 + signal.SIGSEGV}")
    print("=== STDOUT ===")
    print(result.stdout)
    print("=== STDERR ===")
    print(result.stderr)
    print("==============")

    # The process should exit with signal code
    # Note: Python reports negative signal codes, so -11 == -(SIGSEGV)
    expected_exit_code = 128 + signal.SIGSEGV  # 139
    actual_exit_code = result.returncode

    # Handle both positive (139) and negative (-11) representations
    if actual_exit_code < 0:
        actual_signal = -actual_exit_code
        is_correct_signal = actual_signal == signal.SIGSEGV
    else:
        is_correct_signal = actual_exit_code == expected_exit_code

    assert is_correct_signal, (
        f"Expected signal {signal.SIGSEGV} (exit code {expected_exit_code} or {-signal.SIGSEGV}), got {actual_exit_code}"
    )

    # Check that the signal handler output is present
    stderr_output = result.stderr

    # Check for Python stack trace patterns
    python_trace_patterns = [
        ("Python stack trace:", "Should include Python stack trace section"),
        ("deep_function_1", "Should show deep_function_1 in Python stack"),
        ("deep_function_2", "Should show deep_function_2 in Python stack"),
        ("deep_function_3", "Should show deep_function_3 in Python stack"),
    ]

    # Check Python stack trace functionality
    missing_python_trace = []
    for pattern, description in python_trace_patterns:
        if pattern not in stderr_output:
            missing_python_trace.append(f"Missing: {pattern} ({description})")

    if missing_python_trace:
        print("Missing Python stack trace patterns:")
        for missing in missing_python_trace:
            print(f"  - {missing}")

        raise AssertionError(
            f"Python stack trace feature not implemented. Missing: {missing_python_trace}"
        )
    else:
        print(
            "✓ All patterns found - Python stack trace feature appears to be working!"
        )


if __name__ == "__main__":
    test_python_stack_trace_in_signal_handler()
