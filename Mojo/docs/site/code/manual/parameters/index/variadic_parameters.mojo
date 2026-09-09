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


struct MyTensor[*dimensions: Int]:
    pass


def sum_params[*values: Int]() -> Int:
    var sum = 0
    for v in values:
        sum += v
    return sum


comptime sum = sum_params[1, 2, 3, 4, 5]()


def main():
    print(sum)
