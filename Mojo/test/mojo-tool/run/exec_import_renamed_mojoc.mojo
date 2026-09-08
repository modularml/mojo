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

# A renamed precompiled package is self-contained: it loads with the source
# tree gone, including when its modules use absolute self-imports (which
# once recorded a dangling self-dependency that forced a source re-parse).

# RUN: rm -rf %t.dir && mkdir -p %t.dir/src/pkg %t.dir/lib
# RUN: echo "# pkg" > %t.dir/src/pkg/__init__.mojo
# RUN: echo "def afn():" > %t.dir/src/pkg/a.mojo
# RUN: echo "    print(42)" >> %t.dir/src/pkg/a.mojo
# RUN: echo "from pkg.a import afn" > %t.dir/src/pkg/b.mojo
# RUN: echo "def bfn():" >> %t.dir/src/pkg/b.mojo
# RUN: echo "    afn()" >> %t.dir/src/pkg/b.mojo
# RUN: mojo precompile %t.dir/src/pkg -o %t.dir/lib/renamed_pkg.mojoc
# RUN: rm -rf %t.dir/src
# RUN: mojo run -I %t.dir/lib %s | FileCheck %s

# CHECK: 42

from renamed_pkg.b import bfn


def main():
    bfn()
