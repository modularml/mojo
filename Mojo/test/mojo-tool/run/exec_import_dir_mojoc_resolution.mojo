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

# A `.mojoc` inside a plain (non-package) directory still wins the usual
# `.mojoc`-over-`.mojo` resolution precedence, but importing it through the
# directory is rejected: its recorded symbol roots resolve only for a
# top-level binding, so a nested mount would leave every escaping type
# unresolvable. TODO(MOCO-4487): once loading re-anchors recorded roots to
# the mount point, these imports resolve instead (restore the `mojo run`
# form of this test). Inside a source package the `.mojoc` remains
# unresolvable: a package can't legitimately nest a precompiled copy of its
# own submodules.

# RUN: rm -rf %t.dir && mkdir -p %t.dir/only/tmp %t.dir/mixed/tmp %t.dir/pkg/tmp
# RUN: mojo precompile %S/inputs/dir_precompiled/helper -o %t.dir/only/tmp/helper.mojoc

## A `.mojoc`-only child of a plain directory resolves, then is rejected as a
## nested mount.
# RUN: not mojo run -I %t.dir/only %s 2>&1 | FileCheck %s --check-prefix=NESTED

## A `.mojoc`/`.mojo` collision in a plain directory: the `.mojoc` still wins
## (the diagnostic names the precompiled file, not the source module).
# RUN: cp %t.dir/only/tmp/helper.mojoc %t.dir/mixed/tmp/helper.mojoc
# RUN: cp %S/inputs/dir_precompiled/helper_module.mojo %t.dir/mixed/tmp/helper.mojo
# RUN: not mojo run -I %t.dir/mixed %s 2>&1 | FileCheck %s --check-prefix=NESTED

## The same `.mojoc` inside a source package is not importable.
# RUN: cp %t.dir/only/tmp/helper.mojoc %t.dir/pkg/tmp/helper.mojoc
# RUN: cp %S/inputs/dir_precompiled/empty_init.mojo %t.dir/pkg/tmp/__init__.mojo
# RUN: not mojo run -I %t.dir/pkg %s 2>&1 | FileCheck %s --check-prefix=PKG

# NESTED: error: precompiled package '{{.*}}helper.mojoc' must be imported directly from an import root, not as 'tmp.helper'
# PKG: error: unable to locate module 'helper'

from tmp.helper import foo


def main():
    foo()
