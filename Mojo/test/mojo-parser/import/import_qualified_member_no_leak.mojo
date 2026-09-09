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

# A *qualified* component access `mod.x` must resolve `x` only against `mod`.
# If `x` is not a member of `mod`, it is a hard error - even when `x` happens
# to be a sibling module or an __init__ re-export of the *accessing* file's own
# package. The deprecated intra-package fallback is for bare unqualified
# references only; it must not hijack a qualified access whose base is a
# different module. See inputs/qualified_member_leak/.

# (Note we can't use -verify-diagnostics as the errors appear in other packages)
# RUN: not %parse-mojo-isolated -split-input-file -I=%S/inputs %s 2>&1 | FileCheck %s


# `other.shared_fn`: `shared_fn` is an __init__ re-export, not a member of
# `other`. Must NOT resolve via the enclosing package's __init__.
from qualified_member_leak.consumer_reexport import consume

def main():
    _ = consume()

# CHECK: consumer_reexport.mojo:{{[0-9]+}}:{{[0-9]+}}: error: module 'other' has no declaration 'shared_fn'
# CHECK-NOT: from the enclosing package's '__init__' is deprecated

# // -----

# `other.helpers`: `helpers` is a sibling module, not a member of `other`.
# Must NOT resolve via the enclosing package's sibling-module fallback.
from qualified_member_leak.consumer_sibling import consume

def main():
    _ = consume()

# CHECK: consumer_sibling.mojo:{{[0-9]+}}:{{[0-9]+}}: error: module 'other' has no declaration 'helpers'
# CHECK-NOT: implicit reference to sibling module 'helpers'
