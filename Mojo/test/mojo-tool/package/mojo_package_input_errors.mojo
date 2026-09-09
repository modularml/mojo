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

# RUN: not mojo precompile %S/test_package -o test.mojonot 2>&1 | FileCheck %s
# RUN: not mojo precompile %S/test_package -o not-a-directory/ 2>&1 | FileCheck %s
# RUN: not mojo precompile --diagnostic-format json %S/test_package \
# RUN:   -o test.mojonot 2>&1 | FileCheck %s --check-prefix=CHECK-DIAG
# CHECK: output path must have a '.mojoc' extension
# CHECK-DIAG: "kind":"error","message":"output path must have a '.mojoc' extension"}
