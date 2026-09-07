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

from std.memory import ArcPointer


struct SharedDict(ImplicitlyCopyable):
    var attributes: ArcPointer[Dict[String, String]]

    def __init__(out self):
        var attributesDict: Dict[String, String] = {}
        self.attributes = ArcPointer(attributesDict^)

    def __init__(out self, *, copy: Self):
        self.attributes = copy.attributes

    def __setitem__(mut self, key: String, value: String):
        self.attributes[][key] = value

    def __getitem__(self, key: String) -> String:
        return self.attributes[].get(key, default="")


def main():
    var thing1 = SharedDict()
    var thing2 = thing1
    thing1["Flip"] = "Flop"
    print(thing2["Flip"])
