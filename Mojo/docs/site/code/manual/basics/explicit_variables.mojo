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


def explicit_variable():
    # start-explicit-variable
    var x = 10
    # end-explicit-variable
    _ = x


# start-typed-variables
def main():
    var x: Int = 10
    var sum: Int
    sum = x + x
    # end-typed-variables
    _ = sum
    explicit_variable()
