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
# RUN: mkdir -p %t.closure-dir
# RUN: mojo precompile %S/inputs/closure -o %t.closure-dir/closure.mojoc
# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo -I %t.closure-dir %s | FileCheck %s


# CHECK-LABEL: lit.struct.decl @Box
# CHECK: lit.fn @"__init__{{.*}}"{{.*}}*, %move:

from closure import Box


def main() raises:
    var box = Box()

    def use_box() raises {var box^}:
        pass
