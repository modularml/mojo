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

# The name a package parses and packages under comes from its directory, not
# from the spelling of the input path: a trailing slash and `.` for the
# current directory both canonicalize to the directory's name, so absolute
# self-imports keep resolving to the package under compilation (no duplicate
# package, no self link-dependency) and the default output file keeps the
# directory's name.

# RUN: rm -rf %t.dir && mkdir -p %t.dir/pkg
# RUN: echo "# pkg" > %t.dir/pkg/__init__.mojo
# RUN: echo "def afn():" > %t.dir/pkg/a.mojo
# RUN: echo "    pass" >> %t.dir/pkg/a.mojo
# RUN: echo "from pkg.a import afn" > %t.dir/pkg/b.mojo

# RUN: cd %t.dir && mojo precompile pkg/ -o slash_renamed.mojoc
# RUN: kgen-opt %t.dir/slash_renamed.mojoc \
# RUN:   | FileCheck %s --check-prefix=SLASH \
# RUN:       --implicit-check-not link.dependencies --implicit-check-not '["pkg"'

# SLASH: lit.package @slash_renamed

# RUN: cd %t.dir/pkg && mojo precompile . -o %t.dir/dot_renamed.mojoc
# RUN: kgen-opt %t.dir/dot_renamed.mojoc \
# RUN:   | FileCheck %s --check-prefix=DOT \
# RUN:       --implicit-check-not link.dependencies --implicit-check-not '["pkg"'

# DOT: lit.package @dot_renamed

# RUN: cd %t.dir && mojo precompile pkg/
# RUN: kgen-opt %t.dir/pkg.mojoc \
# RUN:   | FileCheck %s --check-prefix=DEFAULT --implicit-check-not link.dependencies

# DEFAULT: lit.package @pkg

# RUN: mkdir %t.dir/out
# RUN: cd %t.dir && mojo precompile pkg/ -o out
# RUN: kgen-opt %t.dir/out/pkg.mojoc \
# RUN:   | FileCheck %s --check-prefix=DIROUT --implicit-check-not link.dependencies

# DIROUT: lit.package @pkg

# RUN: cd %t.dir && mojo precompile pkg/ -o - > stdout.mojoc
# RUN: kgen-opt %t.dir/stdout.mojoc \
# RUN:   | FileCheck %s --check-prefix=STDOUT --implicit-check-not link.dependencies

# STDOUT: lit.package @pkg
