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

# RUN: mkdir -p %t.lazy-test
# RUN: mojo precompile %S/inputs/struct_and_extension_lazy -o %t.lazy-test/lazy_test.mojoc
# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo -I %t.lazy-test %s | FileCheck %s

# This imports a precompiled file's sub-package's extension.
# That extension's file loads a struct from yet another package.
# This will load the extension before its target struct.
from lazy_test import simple_extension
import lazy_test.simple_struct


def main():
    # CHECK: lit.call @lazy_test::@simple_struct::@BaseType::@"__init__()"
    var x = lazy_test.simple_struct.BaseType()
