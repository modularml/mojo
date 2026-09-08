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

# Regression test: the MOCO-2845 self-named-submodule fix must not leak into
# top-level `-I` resolution, where a same-named `.mojoc`/`.mojo` pair is an
# ordinary collision and `.mojoc` must still win.

# RUN: rm -rf %t.dir && mkdir -p %t.dir
# RUN: cp -r %S/inputs/top_level_name_collision/foo %t.dir/foo
# RUN: mojo precompile %S/inputs/top_level_name_collision/foo_pkg_src -o %t.dir/foo/foo.mojoc
# RUN: mojo run -I %t.dir/foo %s | FileCheck %s

from foo import describe


def main():
    # CHECK: precompiled
    describe()
