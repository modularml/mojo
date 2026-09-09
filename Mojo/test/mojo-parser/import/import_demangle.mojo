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
# RUN: %parse-mojo-isolated -I %S/inputs %s | FileCheck %s

# COM: Run it twice to ensure it works on a cache hit.

from test_package.module import `use()weird[]`

# CHECK: lit.package @test_package
# CHECK-NEXT: lit.file_module @module
# CHECK: lit.struct.decl @"weird()struct[]"
# CHECK: lit.fn @"use()weird[]()"


def weird_struct():
    _ = `use()weird[]`()
