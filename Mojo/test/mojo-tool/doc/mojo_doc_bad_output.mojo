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

# When the output file cannot be created or opened, we print a nice error.
# RUN: not mojo doc %s -o no/such/directory.mojo 2>&1 | FileCheck %s
# CHECK: mojo{{.*}}: error: cannot open output file 'no/such/directory.mojo': {{N|n}}o such file or directory

# RUN: not mojo doc --diagnostic-format json %s -o no/such/directory.mojo 2>&1 | FileCheck %s --check-prefix CHECK-DIAG
# CHECK-DIAG: {"kind":"error","message":"cannot open output file{{.*}}"}
