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

# start-python-to-mojo-conversions
from std.python import Python, PythonObject


def main() raises:
    var py_string = PythonObject("Hello, Mojo!")
    var py_bool = PythonObject(True)
    var py_int = PythonObject(123)
    var py_float = PythonObject(3.14)

    var mojo_string = String(py=py_string)
    var mojo_bool = Bool(py=py_bool)
    var mojo_int = Int(py=py_int)
    var mojo_float = Float64(py=py_float)
    # end-python-to-mojo-conversions
    _ = mojo_string^
    _ = mojo_bool
    _ = mojo_int
    _ = mojo_float
