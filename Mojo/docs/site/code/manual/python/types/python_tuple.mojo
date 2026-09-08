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

# start-python-tuple-example
from std.python import Python


def main() raises:
    var py_tuple = Python.tuple("cat", 2, 3.1415, "cat")
    var n = py_tuple[2]
    print("n =", n)
    print("Number of cats:", py_tuple.count("cat"))
    # end-python-tuple-example
