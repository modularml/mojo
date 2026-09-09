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

# LLDB fails with asan because it's built by default with python support in the
# CI, and python fails asan.
# TODO(MOTO-1007): Remove this once we have a fix for asan.
# UNSUPPORTED: asan

# COM: Using the default language Mojo
# RUN: %mojo-build --debug-level=full -O0 %s -o %t
# RUN: mojo debug -X -o -X 'image lookup -vs build_mojo_lang_dwarf::foo()' -X -b %t | FileCheck %s --check-prefix CHECK-MOJO
# CHECK-MOJO: language = "mojo"

# COM: Setting explicitly the language C
# RUN: %mojo-build --debug-level full -O0 --debug-info-language C %s -o %t
# RUN: mojo debug -X -o -X 'image lookup -vs build_mojo_lang_dwarf::foo()' -X -b %t | FileCheck %s --check-prefix CHECK-C
# CHECK-C: language = "c"


def foo():
    pass


def main():
    foo()
