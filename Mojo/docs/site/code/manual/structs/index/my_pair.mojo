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


@fieldwise_init
struct MyPair(ImplicitlyCopyable):
    var first: Int
    var second: Int

    def get_sum(self) -> Int:
        return self.first + self.second


def main():
    var a = MyPair(1, 2)

    # copyable, movable
    var original_pair = MyPair(2, 6)
    var copied_pair = original_pair  # copy
    var moved_pair = original_pair^  # move

    # methods
    var mine = MyPair(6, 8)
    print(mine.get_sum())

    # Suppress compiler warnings
    _ = a^
    _ = copied_pair^
    _ = moved_pair^
