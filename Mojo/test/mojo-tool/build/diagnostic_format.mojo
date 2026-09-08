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

# RUN: not %mojo-build --diagnostic-format json /does/not.exist 2>&1 | FileCheck %s
# CHECK: {"kind":"error","message":"cannot open '/does/not.exist'{{.*}}"}


# RUN: not %mojo-build --diagnostic-format json %s 2>&1 | FileCheck %s --check-prefix=CHECK-DIAG
# CHECK-DIAG: "line":[[@LINE+3]]{{.*}}"message":"expression must be mutable in assignment{{.*}}"
# CHECK-DIAG-NEXT: {"kind":"error","message":"failed to parse{{.*}}"}
def main():
    4 = "hello"
