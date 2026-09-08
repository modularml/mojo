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

# start-mojo-to-python-conversions
from std.python import Python


def main() raises:
    var py_module = """
def type_printer(value):
    print(type(value))
"""
    var py_utils = Python.evaluate(py_module, file=True, name="py_utils")

    py_utils.type_printer(4)
    py_utils.type_printer(3.14)
    py_utils.type_printer(True)
    py_utils.type_printer("Mojo")
    # end-mojo-to-python-conversions
