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

# More than one input is not allowed.
# RUN: not mojo doc %t.1.mojo %t.2.mojo 2>&1 | FileCheck %s --check-prefix CHECK-TOO-MANY-INPUT
# CHECK-TOO-MANY-INPUT: mojo{{.*}}: error: too many input files, cannot process both '{{.*}}.1.mojo' and '{{.*}}.2.mojo'
