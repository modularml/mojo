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

# Regression test: a package whose module re-exports a submodule under an alias
# (`from . import _impl as impl`) must survive being precompiled and reloaded by
# a consumer. Resolving that from-import binds a gated ImportOp over the
# submodule; the now-superseded `unresolved_import` placeholder must be dropped
# before serialization, otherwise reloading the package lists both under the
# alias and fails with "invalid redefinition of 'impl'".

# RUN: mkdir -p %t.dir
# RUN: mojo precompile %S/inputs/submodule_alias_pkg -o %t.dir/submodule_alias_pkg.mojoc
# RUN: mojo run -I %t.dir %s | FileCheck %s

from submodule_alias_pkg.api import api_value


def main():
    # CHECK: 7
    print(api_value())
