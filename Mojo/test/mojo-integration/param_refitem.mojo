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

# RUN: kgen %s -elaborate -S -o - | FileCheck %s

from std.utils import Variant


# CHECK-LABEL: func export @param_refitem
@export
def param_refitem() abi("Mojo") -> Int:
    comptime vec = Variant[Int](42)
    comptime value = vec[Int]
    # CHECK-NEXT: constant: scalar<index> = <42>
    return value
