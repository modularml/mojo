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

from std.python import Python


def share_array() raises:
    var np = Python.import_module("numpy")
    var arr = np.array(Python.list(1, 2, 3, 4, 5, 6, 7, 8, 9))
    var ptr = arr.ctypes.data.unsafe_get_as_pointer[.int64]()
    for i in range(9):
        print(ptr[unsafe_offset=i], end=", ")
    print()


def main() raises:
    share_array()
