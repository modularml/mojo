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
# RUN: %mojo %s | FileCheck %s

# COM: See MOCO-756

from std.collections import Optional


def print_second_string(first: String, second: String) -> None:
    print("Received", second)


def main():
    var optional_func: Optional[
        def(flags: String, args: String) thin -> None
    ] = print_second_string
    # CHECK: Received second
    optional_func.value()("first", "second")
