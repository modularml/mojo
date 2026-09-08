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


def main():
    var d = Dict[String, Int]()
    d["one"] = 1
    d["two"] = 2
    d["three"] = 3
    print(len(d))  # bp1: d has 3 live entries

    var d_empty = Dict[String, Int]()
    print(len(d_empty))  # bp2: empty dict
