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

# Regression test (MOCO-2845): a package's own precompiled whole-package
# `.mojoc` staged alongside a same-named source submodule (mirroring
# `layout.mojoc` inside `layout/`) must not shadow that source submodule.
# When no source sibling exists at all, resolution must fail outright --
# MOCO-2845's fix has no precompiled-only fallback.

# RUN: rm -rf %t.dir && mkdir -p %t.dir/present %t.dir/absent
# RUN: cp -r %S/inputs/self_named_pkg %t.dir/present/self_named_pkg
# RUN: cp -r %S/inputs/self_named_pkg %t.dir/absent/self_named_pkg
# RUN: mojo precompile %t.dir/present/self_named_pkg -o %t.dir/present/self_named_pkg/self_named_pkg.mojoc
# RUN: mojo precompile %t.dir/absent/self_named_pkg -o %t.dir/absent/self_named_pkg/self_named_pkg.mojoc
# RUN: rm %t.dir/absent/self_named_pkg/self_named_pkg.mojo
# RUN: mojo run -I %t.dir/present %s | FileCheck %s --check-prefix=PRESENT
# RUN: not mojo run -I %t.dir/absent %s 2>&1 | FileCheck %s --check-prefix=ABSENT

# PRESENT: source-submodule
# ABSENT: error: unable to locate module 'self_named_pkg'

from self_named_pkg import run


def main():
    run()
