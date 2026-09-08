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

# Surfacing conditional-conformance `where` clauses as notes on
# conformance-failure diagnostics (generic trait bounds, ancestors, packs).
# Notes say "failed" or "unproven"; a user message is an optional suffix.
#
# Instance-to-trait bindings are category mismatches, not covered here.
# Each scenario uses its own struct so notes don't bleed across sites.

# RUN: %parse-mojo-isolated -verify-diagnostics %s


trait Marker:
    pass


struct Yes(Marker):
    pass


struct No:
    pass


##===----------------------------------------------------------------------===##
# Satisfied conditional conformance - positive case (no error, no note).
##===----------------------------------------------------------------------===##


struct OkBox[T: Deinitable](
    Marker where (conforms_to(T, Marker), "OkBox[T] is a Marker only when T is")
):
    pass


def wants_marker_ok[U: Marker & Deinitable](x: U):
    pass


def use_ok(b: OkBox[Yes]):
    wants_marker_ok(b)


##===----------------------------------------------------------------------===##
# Violated via a generic trait-bounded parameter.
##===----------------------------------------------------------------------===##


struct GenBox[T: Deinitable](
    # expected-note @below {{failed constraint: GenBox[T] requires T to be a Marker}}
    Marker where (conforms_to(T, Marker), "GenBox[T] requires T to be a Marker")
):
    pass


# expected-note @below {{function declared here}}
def wants_marker_gen[U: Marker & Deinitable](x: U):
    pass


def use_gen(b: GenBox[No]):
    # expected-error @below {{does not conform to trait 'Deinitable & Marker'}}
    wants_marker_gen(b)


##===----------------------------------------------------------------------===##
# Same path without a user message: the note is still emitted at the `where`.
##===----------------------------------------------------------------------===##


struct PlainBox[T: Deinitable](
    # expected-note @below {{failed constraint}}
    Marker where conforms_to(T, Marker)
):
    pass


# expected-note @below {{function declared here}}
def wants_marker_plain[U: Marker & Deinitable](x: U):
    pass


def use_plain(b: PlainBox[No]):
    # expected-error @below {{does not conform to trait 'Deinitable & Marker'}}
    wants_marker_plain(b)


##===----------------------------------------------------------------------===##
# Requiring a propagated ancestor trait: the message is written on the derived
# conformance but is carried down to the propagated ancestor constraint, so
# requiring the bare ancestor still surfaces it (see the design doc).
##===----------------------------------------------------------------------===##


trait Base:
    pass


trait Refined(Base):
    pass


struct RefinedBox[T: Deinitable](
    # expected-note @below {{failed constraint: RefinedBox[T] requires T to be a Marker}}
    Refined where (conforms_to(T, Marker), "RefinedBox[T] requires T to be a Marker")
):
    pass


# expected-note @below {{function declared here}}
def wants_base[U: Base & Deinitable](x: U):
    pass


def use_ancestor(b: RefinedBox[No]):
    # expected-error @below {{does not conform to trait 'Deinitable & Base'}}
    wants_base(b)


##===----------------------------------------------------------------------===##
# A conformance satisfied by a caller `where` assumption is NOT reported: only
# the still-unproven conformance's message is surfaced. (The absence of a note
# for the CommonA message is enforced by -verify-diagnostics: an unexpected
# note fails the test.)
##===----------------------------------------------------------------------===##


trait MarkerA:
    pass


trait MarkerB:
    pass


trait CommonA:
    pass


trait CommonB:
    pass


struct TwoBox[T: Deinitable](
    CommonA where (conforms_to(T, MarkerA), "TwoBox needs MarkerA for CommonA"),
    # expected-note @below {{unproven constraint: TwoBox needs MarkerB for CommonB}}
    CommonB where (conforms_to(T, MarkerB), "TwoBox needs MarkerB for CommonB"),
):
    pass


# expected-note @below {{function declared here}}
def wants_both[U: CommonA & CommonB & Deinitable](x: U):
    pass


# V is assumed to be MarkerA (so the CommonA conformance holds) but nothing is
# assumed about MarkerB, so only CommonB remains unproven.
def use_two[V: Deinitable](b: TwoBox[V]) where (
    conforms_to(V, MarkerA), "V is a MarkerA"
):
    # expected-error @below {{does not conform to trait}}
    wants_both(b)


##===----------------------------------------------------------------------===##
# When neither marker is available, BOTH conditional conformances are unproven
# and both messages surface as separate notes at the one call site.
##===----------------------------------------------------------------------===##


struct BothBox[T: Deinitable](
    # expected-note @below {{unproven constraint: BothBox needs MarkerA for CommonA}}
    CommonA where (conforms_to(T, MarkerA), "BothBox needs MarkerA for CommonA"),
    # expected-note @below {{unproven constraint: BothBox needs MarkerB for CommonB}}
    CommonB where (conforms_to(T, MarkerB), "BothBox needs MarkerB for CommonB"),
):
    pass


# expected-note @below {{function declared here}}
def wants_both_nm[U: CommonA & CommonB & Deinitable](x: U):
    pass


def use_both_none[V: Deinitable](b: BothBox[V]):
    # expected-error @below {{does not conform to trait}}
    wants_both_nm(b)


##===----------------------------------------------------------------------===##
# Violated via a variadic trait-bounded pack element.
##===----------------------------------------------------------------------===##


struct PackBox[T: Deinitable](
    # expected-note @below {{failed constraint: PackBox[T] requires T to be a Marker}}
    Marker where (conforms_to(T, Marker), "PackBox[T] requires T to be a Marker")
):
    pass


# expected-note @below {{function declared here}}
def wants_markers[*Ts: Marker & Deinitable](*args: *Ts):
    pass


def use_pack(b: PackBox[No]):
    # expected-error @below {{does not conform to trait}}
    wants_markers(b)


##===----------------------------------------------------------------------===##
# Refuted and unproven in the same query: `U` is concretely not a MarkerA, and
# nothing is known about `T`. The verdict is `no`, so only the refutation is
# reported -- proving MarkerB would not make the conformance hold. The absent
# note is the point of this case, and -verify-diagnostics enforces it.
##===----------------------------------------------------------------------===##


struct MixedBox[T: Deinitable, U: Deinitable](
    # expected-note @below {{failed constraint: MixedBox needs MarkerA for CommonA}}
    CommonA where (conforms_to(U, MarkerA), "MixedBox needs MarkerA for CommonA"),
    CommonB where (conforms_to(T, MarkerB), "MixedBox needs MarkerB for CommonB"),
):
    pass


# expected-note @below {{function declared here}}
def wants_both_mixed[W: CommonA & CommonB & Deinitable](x: W):
    pass


def use_mixed[V: Deinitable](b: MixedBox[V, No]):
    # expected-error @below {{does not conform to trait}}
    wants_both_mixed(b)


##===----------------------------------------------------------------------===##
# Same, with the refutation arriving after the unproven one rather than before.
# The verdict is still `no` and still reports only the refutation, so the
# unproven constraint gathered first has to be dropped once it shows up.
##===----------------------------------------------------------------------===##


struct LateMixedBox[T: Deinitable, U: Deinitable](
    CommonA where (conforms_to(T, MarkerA), "LateMixedBox needs MarkerA for CommonA"),
    # expected-note @below {{failed constraint: LateMixedBox needs MarkerB for CommonB}}
    CommonB where (conforms_to(U, MarkerB), "LateMixedBox needs MarkerB for CommonB"),
):
    pass


# expected-note @below {{function declared here}}
def wants_both_late[W: CommonA & CommonB & Deinitable](x: W):
    pass


def use_late_mixed[V: Deinitable](b: LateMixedBox[V, No]):
    # expected-error @below {{does not conform to trait}}
    wants_both_late(b)
