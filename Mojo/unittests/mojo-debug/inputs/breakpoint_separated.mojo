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

# Test that three breakpoint() calls with statements between them each
# produce a separate stop and can each be resumed past.


def main():
    var x = 0
    # fmt: off
    breakpoint(); x += 1  # stop_1
    breakpoint(); x += 1  # stop_2
    breakpoint(); x += 1  # stop_3
    # fmt: on
    print(x)
