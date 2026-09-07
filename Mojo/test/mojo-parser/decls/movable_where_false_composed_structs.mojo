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

# Regression test for a compiler bug where a struct combining its own
# `where <predicate>` clause with `Trait where False` -- especially when
# composed with another such struct as a field -- spuriously failed
# conformance/witness verification ("cannot synthesize move constructor" /
# "does not implement all requirements for 'Movable'"), even though both
# sides had already declared they don't need the trait. See MOCO-4135.
#
# Root cause was two distinct bugs, both in
# Mojo/lib/LITDialect/LITUtils.cpp:
#   1. The single-assumption `LIT::isPropositionImplied` overload was
#      missing an "assumption trivially false implies anything" case -- this
#      alone fixes the self-conformance case below.
#   2. The multi-assumption overload had a separate bug: an early `no` verdict
#      from one assumption discarded an already-found `yes` from another --
#      this is what actually blocked the field-composition case below.
#
# This is deliberately a standalone, minimal test rather than folded into an
# existing large file: this area has a documented history of subtle
# regressions -- a structurally similar (but not identical) fix landed in
# PR #90795 and was reverted 3 days later in PR #90810 (ref MOCO-3942) for
# silently causing CheckLifetimes.cpp to skip a needed destructor call. If
# this test starts failing, read the commits that introduced it (and the
# corresponding KGEN/internal/claude_kb/entries/known-limitations/
# where-false-constraints.md entry) before touching
# the `isPropositionImplied` overload family again.

# RUN: %parse-mojo-isolated %s -mlir-print-debuginfo | FileCheck %s

# `Movable where False` is an opt-out: no move constructor is synthesized, so
# each struct below carries only the unconditionally-injected `Deinitable`
# `__del__`. The per-struct `CHECK-NOT`s (bounded by the next `CHECK-LABEL`)
# pin that absence; without the opt-out honored, a move ctor or a "does not
# implement all requirements for 'Movable'" error would appear here.

# Self-conformance case: a struct's own arithmetic where-clause combined with
# `Movable where False` on itself, no field composition needed.
# CHECK-LABEL: lit.struct.decl @SelfConstrained
# CHECK-NOT: __init__(move:
struct SelfConstrained[value: Int](Movable where False) where value >= 0:
    pass


# Field-composition case: Outer has its own where-clause AND `Movable where
# False`; Inner (used as Outer's field) is also `Movable where False`.
# CHECK-LABEL: lit.struct.decl @ComposedInner
# CHECK-NOT: __init__(move:
struct ComposedInner[value: Int](Movable where False):
    pass

# CHECK-LABEL: lit.struct.decl @ComposedOuter
# CHECK-NOT: __init__(move:
struct ComposedOuter[value: Int](Movable where False) where value >= 0:
    var field: ComposedInner[Self.value]


# Mixed-conformance case: Inner additionally has a genuinely conditional
# (non-False) conformance to a different trait, matching
# explicit_destroy.mojo's PredicateOnStructInner shape. The conditional
# `Deinitable` slot survives (`!constrained_..._Marker`) while the
# `Movable where False` slot is dropped.
# CHECK-LABEL: lit.struct.decl @MixedConformanceInner
# CHECK-SAME: (!constrained_AnyType_Deinitable)
# CHECK-NOT: __init__(move:
struct MixedConformanceInner[value: Int](
    Deinitable where value >= 0, Movable where False,
):
    pass

# CHECK-LABEL: lit.struct.decl @MixedConformanceOuter
# CHECK-NOT: __init__(move:
struct MixedConformanceOuter[value: Int](Movable where False) where value >= 0:
    var field: MixedConformanceInner[Self.value]


# Sentinel: bounds the `CHECK-NOT` block above so it stops before the stdlib
# decls (which legitimately synthesize move ctors) appear in the dump.
# CHECK-LABEL: lit.struct.decl @WhereFalseSentinel
struct WhereFalseSentinel:
    pass
