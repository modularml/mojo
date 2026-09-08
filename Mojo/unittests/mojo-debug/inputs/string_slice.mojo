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
    # StaticString is StringSlice[False, ImmStaticOrigin] -- exercises the
    # formatter for a non-empty and an empty slice.
    var s1: StaticString = "static_string"
    var s2: StaticString = ""
    keep_alive(s1, s2)  # breakpoint
