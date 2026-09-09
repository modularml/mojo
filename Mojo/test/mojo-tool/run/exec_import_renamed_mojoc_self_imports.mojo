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

# A renamed precompiled package is self-contained for every self-import form:
# a resolved submodule import (dotted access), a wildcard import, and a
# relative import climbing out of a nested package all re-resolve to the
# package under its output name once the source tree is gone.

# RUN: rm -rf %t.dir && mkdir -p %t.dir/src/pkg/sub %t.dir/lib
# RUN: echo "# pkg" > %t.dir/src/pkg/__init__.mojo
# RUN: echo "def afn():" > %t.dir/src/pkg/a.mojo
# RUN: echo "    print(42)" >> %t.dir/src/pkg/a.mojo
# RUN: echo "import pkg.a" > %t.dir/src/pkg/b.mojo
# RUN: echo "def bfn():" >> %t.dir/src/pkg/b.mojo
# RUN: echo "    pkg.a.afn()" >> %t.dir/src/pkg/b.mojo
# RUN: echo "from pkg.a import *" > %t.dir/src/pkg/c.mojo
# RUN: echo "def cfn():" >> %t.dir/src/pkg/c.mojo
# RUN: echo "    afn()" >> %t.dir/src/pkg/c.mojo
# RUN: echo "# sub" > %t.dir/src/pkg/sub/__init__.mojo
# RUN: echo "from ..a import afn" > %t.dir/src/pkg/sub/m.mojo
# RUN: echo "def mfn():" >> %t.dir/src/pkg/sub/m.mojo
# RUN: echo "    afn()" >> %t.dir/src/pkg/sub/m.mojo
# RUN: mojo precompile %t.dir/src/pkg -o %t.dir/lib/renamed_pkg.mojoc
# RUN: rm -rf %t.dir/src
# RUN: mojo run -I %t.dir/lib %s | FileCheck %s

# CHECK: 42
# CHECK-NEXT: 42
# CHECK-NEXT: 42

from renamed_pkg.b import bfn
from renamed_pkg.c import cfn
from renamed_pkg.sub.m import mfn


def main():
    bfn()
    cfn()
    mfn()
