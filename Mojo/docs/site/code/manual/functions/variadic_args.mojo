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


def count_many_things[*ArgTypes: Intable](*args: *ArgTypes) -> Int:
    var total = 0

    comptime for i in range(args.__len__()):
        total += Int(args[i])

    return total


def main():
    print(count_many_things(5, 11.7, 12))
