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
"""Test that async-safe Python stack traces work when the GIL is held in C++."""

import os
import subprocess
import sys


def test_signal_handler_with_gil_held() -> None:
    """Test that signal handler works when the GIL is explicitly held in C++."""

    # Get the directory containing this test file
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(test_dir, "test_script_gil.py")

    print(f"Running test script: {test_file}")

    # Run the test script and capture output
    # PYTHONPATH is set in BUILD.bazel to include the nanobind extension
    result = subprocess.run(
        [sys.executable, test_file],
        capture_output=True,
        text=True,
        timeout=10,
    )

    print("Return code:", result.returncode)
    print("\n--- STDOUT ---")
    print(result.stdout)
    print("\n--- STDERR ---")
    print(result.stderr)

    # Check that the process crashed with SIGSEGV (exit code 139 = 128 + 11)
    assert result.returncode == 139, (
        f"Expected exit code 139, got {result.returncode}"
    )

    # Check that we got the expected output components
    stderr_output = result.stderr

    # Verify key components are present
    assert "Starting GIL-held signal handler test" in stderr_output
    assert "Signal handler initialized" in stderr_output
    assert (
        "About to trigger SIGSEGV from C++ with GIL explicitly held"
        in stderr_output
    )

    # Check for signal handler output
    assert "test_gil_signal_handler crashed!" in stderr_output
    assert "Signal Information:" in stderr_output
    assert "SIGSEGV" in stderr_output
    assert "C++ stack trace:" in stderr_output

    # Most importantly, check for Python stack trace
    # This is the critical test - we should get Python stack traces
    # even when the GIL is held by C++ code
    assert "Python stack trace:" in stderr_output
    assert "deep_function_3" in stderr_output
    assert "deep_function_2" in stderr_output
    assert "deep_function_1" in stderr_output

    print("✅ All assertions passed!")
    print("✅ Python stack traces work correctly even with GIL held in C++")


def main() -> int | None:
    """Run the GIL signal handler test."""
    print("Testing signal handler with GIL held in C++...")

    try:
        test_signal_handler_with_gil_held()
        print("🎉 GIL signal handler test PASSED!")
        return 0
    except Exception as e:
        print(f"❌ GIL signal handler test FAILED: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
