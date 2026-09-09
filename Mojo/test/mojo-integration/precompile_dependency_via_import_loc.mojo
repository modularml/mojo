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
#
# Regression test: when a precompiled (.mojoc) package's bytecode records a
# top-level `LinkDependency` on another package, resolving that dependency must
# consult the importing file's location (so its main CWD is walked), not just
# the bytecode's synthetic location.
#
# Concretely: `importer_src/__init__.mojo` does `from simple import ...`. After
# precompile, `importer.mojoc` carries a top-level `LinkDependency` for
# `simple`. `simple/` lives as a source directory adjacent to the user
# file's directory, with the user file run via `-I %t.dir/deps`. The only
# path that can locate `simple/` is the user file's main-CWD walk.
#
# ===----------------------------------------------------------------------=== #

# RUN: rm -rf %t.dir && mkdir -p %t.dir/simple %t.dir/importer_src %t.dir/deps

# A top-level "dependency" package, located as a source directory inside the
# test's working directory. It must remain reachable *only* via the user
# file's main-directory walk -- it is intentionally not placed on `-I`.
# RUN: echo 'def answer() -> Int:' > %t.dir/simple/__init__.mojo
# RUN: echo '    return 7' >> %t.dir/simple/__init__.mojo

# An "importer" package whose `__init__.mojo` depends top-level on `simple`.
# After `precompile`, the dependency ends up baked into the bytecode as a
# `LinkDependency`.
# RUN: echo 'from simple import answer' > %t.dir/importer_src/__init__.mojo
# RUN: echo 'def ask() -> Int:' >> %t.dir/importer_src/__init__.mojo
# RUN: echo '    return answer()' >> %t.dir/importer_src/__init__.mojo

# `-I %t.dir` makes `simple/` discoverable during the precompile-stage
# import resolution, so the dep is recorded. The produced mojoc is placed
# into a separate subdirectory that is *not* recursive-walked.
# RUN: mojo precompile -I %t.dir %t.dir/importer_src -o %t.dir/deps/importer.mojoc

# A small user program in the same directory as `simple/`, importing the
# precompiled `importer`. It is the only thing that needs to resolve
# `importer`'s recorded dep on `simple`.
# RUN: echo 'from importer import ask' > %t.dir/main.mojo
# RUN: echo 'def main():' >> %t.dir/main.mojo
# RUN: echo '    print(ask())' >> %t.dir/main.mojo

# Run from %t.dir so its main CWD is %t.dir/ -- `simple/` is reachable only
# via this walk (not via `-I %t.dir/deps`, which holds only the mojoc).
# RUN: cd %t.dir && %mojo -I %t.dir/deps main.mojo | FileCheck %s

# CHECK: 7
