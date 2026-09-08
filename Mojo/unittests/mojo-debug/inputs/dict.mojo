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

from debug_test_utils import keep_alive


def main():
    var d = Dict[String, Int]()
    d["one"] = 1
    d["two"] = 2
    d["three"] = 3
    keep_alive(d)  # breakpoint

    var d2 = Dict[String, Int]()
    d2["x"] = 10
    keep_alive(d2)  # breakpoint

    var d3 = Dict[String, Int]()
    keep_alive(d3)  # breakpoint
