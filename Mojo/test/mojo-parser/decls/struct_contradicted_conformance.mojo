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

# A conformance condition disproven by the struct's own `where` clause leaves the
# facts in scope unsatisfiable, so it must reach the same verdict as the literal
# `where False` spelling -- otherwise the behavior depends on whether the
# condition happens to fold to a literal at parse time. These shapes reproduce
# with no fields at all; the field-carrying ones are covered by
# `movable_where_false_semantic_contradiction.mojo`. See MOCO-4135.

# RUN: %parse-mojo-isolated %s | FileCheck %s

# The conformance is suppressed exactly like the literal `where False` spelling:
# no move constructor is synthesized, only the unconditional `Deinitable`
# `__del__` remains. The `CHECK-NOT`s (each bounded by the next `CHECK-LABEL`)
# pin that absence; a leaked move ctor or a "does not implement all requirements
# for 'Movable'" error would fail here.

# CHECK-LABEL: lit.struct.decl @ContradictedByOwnClause
# CHECK-NOT: __init__(move:
struct ContradictedByOwnClause[n: Int](Movable where not (n > 0)) where n > 0:
    pass


# The `conforms_to`-negation spelling, which also pins the order of the relation
# rules: `Movable` canonicalizes to a multi-symbol trait, so the goal decomposes
# into a conjunction that the negated assumption contradicts only as a whole.
# CHECK-LABEL: lit.struct.decl @ContradictedByOwnConformsToClause
# CHECK-NOT: __init__(move:
struct ContradictedByOwnConformsToClause[T: AnyType](
    Movable where conforms_to(T, Movable)
) where not conforms_to(T, Movable):
    pass


# Sentinel: bounds the final `CHECK-NOT` block before the stdlib decls (which
# legitimately synthesize move ctors) appear in the dump.
# CHECK-LABEL: lit.struct.decl @ContradictedSentinel
struct ContradictedSentinel:
    pass
