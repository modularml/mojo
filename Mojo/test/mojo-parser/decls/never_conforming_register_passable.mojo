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

# `RegisterPassable`/`TrivialRegisterPassable where False` under the
# never-conforming semantics. Both conventions are driven by
# getConformanceCondition, so an unsatisfiable slot (literal `where False` or a
# body-clause contradiction) reports the conformance as absent: the struct keeps
# the default `MemoryOnly` convention, silently, exactly as if the trait were
# omitted. This is invariant 4 (`Trait where False` is silently legal and
# equivalent to omitting the trait) applied to the register-passable traits.
#
# A genuinely conditional `TrivialRegisterPassable` (a satisfiable where-clause)
# is still rejected with "conditional conformance ... is not supported"; that
# route is pinned in struct_conditional_trait_conformance_errors.mojo. The
# contrast matters: opt-out (nullopt) is silent, a real conditional is an error.

# RUN: %parse-mojo-isolated %s | FileCheck %s
# The opt-out spellings emit no diagnostic at all -- no "conditional conformance
# not supported", no convention change. An empty expected-diagnostics set makes
# any stray diagnostic a failure.
# RUN: %parse-mojo-isolated %s -verify-diagnostics


# --- Positive controls: the conventions still apply unconditionally ---------

# CHECK: lit.struct.decl @RPUncond(!AnyType_Deinitable_Movable_RegisterPassable) register_passable attributes
struct RPUncond(RegisterPassable):
    pass

# CHECK: lit.struct.decl @TRPUncond({{.*}}_TrivialRegisterPassable) register_passable_trivial attributes
struct TRPUncond(TrivialRegisterPassable):
    pass


# --- where False: silent opt-out, MemoryOnly (no convention keyword) --------

# The false slot is erased from the canonical trait, so neither the marker nor a
# convention keyword mentions RegisterPassable.
# CHECK: lit.struct.decl @RPFalse(!AnyType_Deinitable_Movable) attributes
struct RPFalse(RegisterPassable where False):
    pass

# CHECK: lit.struct.decl @TRPFalse(!AnyType_Deinitable_Movable) attributes
struct TRPFalse(TrivialRegisterPassable where False):
    pass


# --- Contradicted conditional: same verdict as literal `where False` --------

# `not (n > 0)` under `where n > 0` is unsatisfiable but not *literally* false,
# so the slot survives in the canonical-trait marker (cosmetic, like the Movable
# contradicted case). The observable verdict is what matters and it matches
# TRPFalse: no `register_passable_trivial` convention keyword and no error.
# CHECK: lit.struct.decl @TRPContradicted
# CHECK-SAME: _TrivialRegisterPassable) attributes
struct TRPContradicted[n: Int](
    TrivialRegisterPassable where not (n > 0)
) where n > 0:
    pass
