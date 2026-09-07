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
struct MyStruct:
    var name: String

    def move_field(mut self, var new_name: String):
        var name = self.name^  # Moves field; self.name is now uninitialized
        print("Name:", name)  # Prints: "Name: Ken"
        self.name = new_name^  # reinitialize the field


def main():
    var instance = MyStruct("Ken")
    instance.move_field("Scott")
    print("Name:", instance.name)  # Prints: "Name: Scott"
