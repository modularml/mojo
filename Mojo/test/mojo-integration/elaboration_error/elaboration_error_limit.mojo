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

# RUN: kgen %s -elaborate -elaboration-error-limit=2 --num-threads=1 -verify-diagnostics
# RUN: not kgen %s -elaborate=use-parametric-interpreter -elaboration-error-limit=2 --num-threads=1 2>&1 | FileCheck %s --check-prefix=CHECK-PARAM

# COM: -elaborate=use-parametric-interpreter slight difference from -elaborate for error messages.
#      Using FileCheck instead to check those with CHECK-PRAMA prefix.
#      (TODO) error message for parametric interpreter is non-deterministic with limit=2, i.e.
#      either combination of 2 out of foo, bar and baz can show up, hence not checking with a label


# expected-error @+2{{function instantiation failed}}
@export
def entry_method0() abi("Mojo"):
    foo()  # expected-note {{call expansion failed}}


# expected-error @+3{{function instantiation failed}}
# expected-note @+2{{too many errors emitted, stopping now}}
@export
def entry_method1() abi("Mojo"):
    bar()  # expected-note {{call expansion failed}}


@export
def entry_method2() abi("Mojo"):
    baz()


# expected-note @+2{{function instantiation failed}}
@no_inline
def foo():
    __mlir_op.`kgen.param.assert`[
        cond=__mlir_attr.`#kgen.simd<false> : !kgen.scalar<bool>`,
        message="oops".value,
    ]()  # expected-note {{constraint failed}}


# CHECK-PARAM: function instantiation failed
# CHECK-PARAM: constraint failed
# expected-note @+2 {{function instantiation failed}}
@no_inline
def bar():
    __mlir_op.`kgen.param.assert`[
        cond=__mlir_attr.`#kgen.simd<false> : !kgen.scalar<bool>`,
        message="oops".value,
    ]()  # expected-note {{constraint failed}}


# CHECK-PARAM: function instantiation failed
# CHECK-PARAM: constraint failed
@no_inline
def baz():
    __mlir_op.`kgen.param.assert`[
        cond=__mlir_attr.`#kgen.simd<false> : !kgen.scalar<bool>`,
        message="oops".value,
    ]()


# CHECK-PARAM: too many errors emitted, stopping now
