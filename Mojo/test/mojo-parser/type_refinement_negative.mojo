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

# Negative tests for type refinement — verify refinement does NOT incorrectly
# propagate.
#
# RUN: %parse-mojo-isolated -verify-diagnostics %s


trait ProcessTrait:
    def process(self) -> Int:
        ...


trait LeakTestTrait:
    def leak_method(self) -> Int:
        ...


comptime AllCopyableAttr[*Ts: AnyType]: Bool = conforms_to(Ts.values, Copyable)


# --- Refinement should NOT leak to a different parameter ---


# expected-note @below {{function declared here}}
def process_element_1[T: ProcessTrait](elem: T) -> Int:
    return elem.process()


def test_wrong_param[
    T: ImplicitlyCopyable, U: ImplicitlyCopyable
](t_val: T, u_val: U) -> Int where conforms_to(T, ProcessTrait):
    return process_element_1(u_val)  # expected-error {{cannot be converted}}


# --- No refinement without where clause ---


# expected-note @below {{function declared here}}
def process_element_2[T: ProcessTrait](elem: T) -> Int:
    return elem.process()


def test_no_where_clause[T: ImplicitlyCopyable](val: T) -> Int:
    return process_element_2(val)  # expected-error {{cannot be converted}}


# --- comptime assert on unrelated condition should NOT refine ---


# expected-note @below {{function declared here}}
def process_element_3[T: ProcessTrait](elem: T) -> Int:
    return elem.process()


def test_unrelated_assert[T: ImplicitlyCopyable](val: T) -> Int:
    comptime assert True, "always true"
    return process_element_3(val)  # expected-error {{cannot be converted}}


# --- Refinement inside comptime if must NOT leak outside the branch ---


# expected-note @below {{function declared here}}
def needs_leak_trait_1[T: LeakTestTrait](x: T) -> Int:
    return x.leak_method()


def test_refinement_does_not_leak_past_branch[
    T: ImplicitlyCopyable
](val: T) -> Int:
    comptime if conforms_to(T, LeakTestTrait):
        pass
    return needs_leak_trait_1(val)  # expected-error {{cannot be converted}}


# expected-note @below {{function declared here}}
def needs_leak_trait_2[T: LeakTestTrait](x: T) -> Int:
    return x.leak_method()


def test_refinement_does_not_leak_to_else_branch[
    T: ImplicitlyCopyable
](val: T) -> Int:
    comptime if conforms_to(T, LeakTestTrait):
        return needs_leak_trait_2(val)
    else:
        return needs_leak_trait_2(val)  # expected-error {{cannot be converted}}


# --- Variable refinement must NOT persist past comptime if ---


# expected-note @below {{function declared here}}
def needs_leak_trait_3[T: LeakTestTrait](x: T) -> Int:
    return x.leak_method()


def test_var_refinement_does_not_persist_past_comptime_if[
    T: ImplicitlyCopyable
](val: T) -> Int:
    var x = val
    comptime if conforms_to(T, LeakTestTrait):
        _ = needs_leak_trait_3(x)
    return needs_leak_trait_3(x)  # expected-error {{cannot be converted}}


# --- Refinement from nested comptime assert must NOT leak past branch ---


# expected-note @below {{function declared here}}
def needs_leak_trait_4[T: LeakTestTrait](x: T) -> Int:
    return x.leak_method()


def test_nested_assert_refinement_does_not_leak[
    T: ImplicitlyCopyable
](val: T) -> Int:
    comptime if conforms_to(T, LeakTestTrait):
        comptime assert conforms_to(T, LeakTestTrait)
        _ = needs_leak_trait_4(val)
    return needs_leak_trait_4(val)  # expected-error {{cannot be converted}}


# --- Variadic helper refinement must NOT persist past comptime if ---


# expected-note @below {{function declared here}}
def needs_copyable_trait[T: Copyable](x: T):
    pass


def test_variadic_refinement_does_not_leak_past_branch[
    *Ts: Deinitable & Movable
](*args: *Ts):
    comptime if AllCopyableAttr[*Ts]:
        needs_copyable_trait(args[0])
    needs_copyable_trait(args[0])  # expected-error {{cannot be converted}}


# --- Refined-but-still-failing type-value parameter binding mentions the
# refined trait composition in the diagnostic ---
#
# When `T: AnyType` is bound to a parameter expecting an unrelated trait, the
# compiler also tries the refined value `downcast(T, ...)`. If that still
# doesn't satisfy the parameter, the diagnostic should report the type the
# compiler actually considered (the merged trait composition), not the
# original bound — otherwise the user can't tell that the assumption was
# taken into account at all.


trait NeededTrait:
    pass


trait UnrelatedTrait:
    pass


# expected-note @below {{function declared here}}
def needs_needed_trait[T: NeededTrait]():
    pass


def test_diag_shows_refined_type[T: AnyType]():
    comptime if conforms_to(T, UnrelatedTrait):
        # expected-error @+1 {{value has type 'UnrelatedTrait'}}
        needs_needed_trait[T]()


# --- Refinement fallback for type-base attribute lookup falls through
# cleanly when the refined trait also lacks the member ---
#
# `AttributeRefNode::emitLCVIR` refines the type-valued base for this lookup.
# If the refined trait doesn't expose the member, we fall through to the
# standard "no such attribute" diagnostic, which should report the refined
# merged trait so the user knows the assumption was applied.


trait HasKnownStaticMethod:
    @staticmethod
    def known_method() -> Int:
        ...


def test_refined_type_base_unknown_member[T: AnyType]():
    comptime if conforms_to(T, HasKnownStaticMethod):
        # expected-error @+1 {{'HasKnownStaticMethod' value has no attribute 'unknown_method'}}
        _ = T.unknown_method()
