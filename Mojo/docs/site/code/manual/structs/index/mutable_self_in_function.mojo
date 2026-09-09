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


struct MyStruct:
    var value: Int

    def increment(mut self):
        self.value += 1  # Works. Mutable `self` allows assignment
        # But without `mut`:
        # ERROR: expression must be mutable in assignment

    def __init__(out self, value: Int):
        self.value = value


def main():
    var my_struct = MyStruct(1)
    my_struct.increment()
    print(my_struct.value)
