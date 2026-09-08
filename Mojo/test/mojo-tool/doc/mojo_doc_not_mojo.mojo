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

# We don't allow input files that don't end in '.mojo'.
# RUN: touch %t.foo
# RUN: not mojo doc %t.foo 2>&1 | FileCheck %s --check-prefix CHECK-NOT-MOJO
# CHECK-NOT-MOJO: mojo{{.*}}: error: cannot open '{{.*}}.foo', since it does not appear to be a Mojo file (it does not end in '.mojo' or '.mojoc') or a Mojo source package
