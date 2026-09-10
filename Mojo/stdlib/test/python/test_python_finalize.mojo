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
"""Tests that the embedded CPython interpreter is properly finalized at exit.

Specifically tests that:
- Python print() output is produced when stdout is piped (not a TTY)
- Python atexit handlers run during interpreter finalization
"""

from std.python import Python, PythonObject
from std.testing import assert_equal, assert_true, TestSuite
from std.sys.arg import argv


def test_python_print_captured() raises:
    """Python print() works and produces output via StringIO capture."""
    var captured = String(
        Python.evaluate(
            "("
            "  lambda: ("
            "    __import__('sys').__dict__.__setitem__('stdout',"
            " __import__('io').StringIO()) or"
            "    print('hello from Python') or"
            "    __import__('sys').stdout.getvalue()"
            "  )"
            ")()"
        )
    )
    assert_equal(captured, "hello from Python\n")


def test_python_atexit_via_subprocess() raises:
    """Python atexit handlers fire during interpreter finalization.

    Runs a Mojo subprocess that registers an atexit handler writing
    a marker file. Verifies the file exists after the subprocess exits,
    confirming atexit handlers execute during Py_FinalizeEx inside Mojo.
    """
    var os = Python.import_module("os")
    var tempfile = Python.import_module("tempfile")
    var subprocess = Python.import_module("subprocess")

    var tmp = String(tempfile.mktemp(suffix=".txt"))
    var test_mojo_script = String(tempfile.mktemp(suffix=".mojo"))

    var mojo_code = (
        "from std.python import Python\n"
        "def main() raises:\n"
        '    _ = Python.evaluate(\\"\\"\\"\n'
        "import atexit; f=r'"
        + tmp
        + "'; atexit.register(lambda: open(f,'w').write('OK'))\n\\\"\\\"\\\")"
    )

    _ = Python.evaluate(
        "open(r'" + test_mojo_script + "','w').write(r'''" + mojo_code + "''')"
    )

    # Use the mojo binary used to run this test
    var mojo_bin = argv()[0]
    _ = subprocess.run(
        Python.list(mojo_bin, "run", test_mojo_script),
        check=PythonObject(True),
    )

    assert_true(Bool(os.path.exists(tmp)))
    assert_equal(String(Python.evaluate("open(r'" + tmp + "').read()")), "OK")
    _ = os.unlink(tmp)
    _ = os.unlink(test_mojo_script)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
