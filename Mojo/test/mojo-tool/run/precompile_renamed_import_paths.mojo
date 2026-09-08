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

# A renamed precompiled package's recorded import paths root at the output
# name across every import form kept in the artifact: resolved imports
# retained inside file modules (including their nested submodule imports)
# and wildcard imports. Relative imports stay relative: they anchor at the
# importing module's position, so the rename must leave them alone.

# RUN: rm -rf %t.dir && mkdir -p %t.dir/pkg/sub
# RUN: echo "# pkg" > %t.dir/pkg/__init__.mojo
# RUN: echo "def afn():" > %t.dir/pkg/a.mojo
# RUN: echo "    pass" >> %t.dir/pkg/a.mojo
# RUN: echo "import pkg.a" > %t.dir/pkg/b.mojo
# RUN: echo "def bfn():" >> %t.dir/pkg/b.mojo
# RUN: echo "    pkg.a.afn()" >> %t.dir/pkg/b.mojo
# RUN: echo "from pkg.a import *" > %t.dir/pkg/c.mojo
# RUN: echo "def cfn():" >> %t.dir/pkg/c.mojo
# RUN: echo "    afn()" >> %t.dir/pkg/c.mojo
# RUN: echo "# sub" > %t.dir/pkg/sub/__init__.mojo
# RUN: echo "from ..a import afn" > %t.dir/pkg/sub/m.mojo
# RUN: echo "def mfn():" >> %t.dir/pkg/sub/m.mojo
# RUN: echo "    afn()" >> %t.dir/pkg/sub/m.mojo
# RUN: mojo precompile %t.dir/pkg -o %t.dir/renamed_pkg.mojoc
# RUN: kgen-opt %t.dir/renamed_pkg.mojoc \
# RUN:   | FileCheck %s --implicit-check-not '["pkg"' --implicit-check-not '@pkg::'

# CHECK: lit.import @pkg path <0, ["renamed_pkg"]>
# CHECK: lit.import @a path <0, ["renamed_pkg", "a"]>
# CHECK: lit.unresolved_wildcard_import from <0, ["renamed_pkg", "a"]>
# CHECK: lit.unresolved_import <2, ["a"]>
