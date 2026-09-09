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

# RUN: %parse-mojo-isolated -I %S/inputs -verify-diagnostics %s

# Extensions apply only to the struct decl they were declared on, not to any
# struct that shares its name. `ext_ident_x` and `ext_ident_y` each define
# their own `Foo` plus an extension of it; the explicit import below makes
# `ext_ident_y`'s `Foo` the visible one, while both extensions sit in scope
# under the name key "extension:Foo". Member lookup must only apply the
# extension whose resolved target is the visible `Foo`. Previously the other
# module's static resolved silently to a function of the wrong struct, and its
# instance method died in self-unification with "cannot be converted from
# 'Foo' to 'Foo'" instead of a missing-attribute error.
#
# See MOCO-4406.

from ext_ident_x import *
from ext_ident_y import Foo


def main():
    var f = Foo()
    _ = f.y_inst()
    _ = Foo.y_static()
    # expected-error @+1 {{'Foo' value has no attribute 'x_inst'}}
    _ = f.x_inst()
    # expected-error @+1 {{'Foo' value has no attribute 'x_static'}}
    _ = Foo.x_static()
