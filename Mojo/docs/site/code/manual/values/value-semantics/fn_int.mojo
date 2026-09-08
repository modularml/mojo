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


def add_two(y: Int):
    # y += 2  # This would cause a compiler error because `y` is immutable
    # We can instead make an explicit copy:
    var z = y
    z += 2
    print("z:", z)


def main():
    var x = 1
    add_two(x)
    print("x:", x)
