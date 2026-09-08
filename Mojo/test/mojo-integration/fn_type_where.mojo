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

# A trailing `where` clause on a `thin` function type constrains the type's own
# parameters, so a generic algorithm can state what it promises the function it
# is handed.

comptime Kernel = def[w: Int](Int) thin -> None where w > 0


def scale[w: Int](x: Int):
    print(w * x)


def apply[F: Kernel](x: Int):
    F[4](x)


def main():
    # CHECK: 12
    apply[scale](3)
