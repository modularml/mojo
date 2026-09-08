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

from std.collections import List


struct NameList:
    var names: List[String]

    def __init__(out self, *names: String):
        self.names = List[String]()
        for name in names:
            self.names.append(name)

    def __getitem__(ref self, index: Int) raises -> ref[self.names[0]] String:
        if index >= 0 and index < len(self.names):
            return self.names[index]
        else:
            raise Error("index out of bounds")


def main() raises:
    var list = NameList("Thor", "Athena", "Dana", "Vrinda")
    ref name = list[2]
    print(name)
    name += "?"
    print(list[2])
