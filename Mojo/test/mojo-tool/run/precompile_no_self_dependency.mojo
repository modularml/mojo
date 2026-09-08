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

# A package never records itself as a link dependency, even when a module
# absolutely imports its own package (which materializes a duplicate
# package op during precompilation).

# RUN: rm -rf %t.dir && mkdir -p %t.dir/pkg
# RUN: echo "# pkg" > %t.dir/pkg/__init__.mojo
# RUN: echo "def afn():" > %t.dir/pkg/a.mojo
# RUN: echo "    pass" >> %t.dir/pkg/a.mojo
# RUN: echo "from pkg.a import afn" > %t.dir/pkg/b.mojo
# RUN: mojo precompile %t.dir/pkg -o %t.dir/renamed_pkg.mojoc
# RUN: kgen-opt %t.dir/renamed_pkg.mojoc | FileCheck %s --implicit-check-not link.dependencies

# CHECK: lit.package @renamed_pkg
