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

# simple examples from the intro section
comptime NUM_TILES = 1024 // 32

comptime SIZE = 3


def get_array_size() -> Int:
    return 32


def main():
    comptime for i in range(4):
        print(i)

    var array = Array[Int, get_array_size()](fill=0)
    _ = array
