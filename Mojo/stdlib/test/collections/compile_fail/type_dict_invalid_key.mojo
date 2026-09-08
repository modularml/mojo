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

# RUN: not %mojo %s 2>&1 | FileCheck %s

from std.collections import TypeDict
from std.reflection import reflect


def main():
    comptime td = TypeDict[
        T=Int,
        Trait=AnyType,
        [1, 2, 3],
        Int,
        String,
        Float64,
    ]

    # CHECK: Key is not present in TypeDict
    comptime t = td.get[4]

    print(reflect[t].name())
