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

# Test input for verifying that ASAP-destroyed non-trivial variables (String)
# show as "not available" in the debugger after their last use, rather than
# showing wrong/garbage values (MOTO-1424).


def get_string() -> String:
    var s = "hel"
    s += "lo"  # defeat the string literal optimization.
    return s


def main():
    var text = get_string()
    print(text)  # breakpoint
    print("done")  # breakpoint
