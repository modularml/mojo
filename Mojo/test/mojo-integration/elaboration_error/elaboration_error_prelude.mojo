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

# RUN: kgen -elaborate %s --verify-diagnostics
# RUN: not kgen -elaborate %s --elaboration-error-include-prelude 2>&1 | FileCheck %s --check-prefix=CHECK-PRELUDE
# RUN: not mojo --elaboration-error-include-prelude  %s 2>&1 | FileCheck %s --check-prefix=CHECK-PRELUDE

from std.collections.string.string_span import _get_kgen_string

# CHECK-PRELUDE: {{.*}}std/builtin/_startup.mojo
# CHECK-PRELUDE-SAME: error: function instantiation failed
# CHECK-PRELUDE: {{.*}}std/builtin/_startup.mojo
# CHECK-PRELUDE-SAME: note: call expansion failed
# CHECK-PRELUDE: {{.*}}std/builtin/_startup.mojo
# CHECK-PRELUDE-SAME: note: function instantiation failed
# CHECK-PRELUDE: {{.*}}std/builtin/_startup.mojo
# CHECK-PRELUDE-SAME: note: call expansion failed


@always_inline("nodebug")
def my_constrained[cond: Bool, msg: StaticString, *extra: StaticString]():
    __mlir_op.`kgen.param.assert`[
        cond=cond.__mlir_bool__(),
        message=_get_kgen_string[msg, *extra](),
    ]()  # expected-note {{constraint failed}}


# expected-note @below {{function instantiation failed}}
def my_func():
    # expected-note @below {{call expansion failed}}
    my_constrained[False, "foo"]()


# expected-error @below {{function instantiation failed}}
def main():
    # expected-note @below {{call expansion failed}}
    my_func()
