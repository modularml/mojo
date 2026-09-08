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
# RUN: %parse-mojo-isolated %s -split-input-file | FileCheck %s

# COM: Parameter inference from equality (`where a == b`) body constraints, as
# COM: introduced by `ParamInf::inferFromBodyConstraints`. `R3` is inferred from
# COM: the closure's typed-raises, then the fixpoint scan chains `R2` from
# COM: `R2 == R3` and `R1` from `R1 == R2`, leaving the call fully bound.


# COM: Two equalities written as *separate* `where` clauses become two distinct
# COM: body constraints.

def split_chain[
    R1: AnyType,
    R2: AnyType,
    R3: AnyType,
    F: def() raises R3,
    //,
](f: F) where R1 == R2 where R2 == R3:
    pass


# CHECK-LABEL: lit.fn @"use_split
def use_split():
    def closure() raises Int:
        pass

    # COM: All three type parameters are inferred to `Int`.
    # CHECK: lit.call @{{.*}}::@"split_chain{{.*}}"[{{.*}}]<:!AnyType !Int, :!AnyType !Int, :!AnyType !Int
    split_chain(closure)


# // -----


# COM: The *same* two equalities written as a single conjunction (`a and b`)
# COM: form one `and`-proposition constraint. `inferFromBodyConstraints`
# COM: flattens the conjunction so each `==` conjunct still drives inference,
# COM: matching the separate-clause behavior above.

def conj_chain[
    R1: AnyType,
    R2: AnyType,
    R3: AnyType,
    F: def() raises R3,
    //,
](f: F) where R1 == R2 and R2 == R3:
    pass


# CHECK-LABEL: lit.fn @"use_conj
def use_conj():
    def closure() raises Int:
        pass

    # COM: The conjunction is decomposed, so the call is fully bound too.
    # CHECK: lit.call @{{.*}}::@"conj_chain{{.*}}"[{{.*}}]<:!AnyType !Int, :!AnyType !Int, :!AnyType !Int
    conj_chain(closure)


# // -----


# COM: An equality whose open side is a *parametric type* binds more than one
# COM: parameter from a single constraint. `T` is inferred from the argument,
# COM: then `T == Pair[A, B]` binds both `A` and `B` at once by structurally
# COM: matching against `Pair[Int, Bool]`.

@fieldwise_init
struct Pair[A: AnyType, B: AnyType](Copyable, Movable):
    pass


def unpack[
    A: AnyType,
    B: AnyType,
    T: AnyType,
    //,
](x: T) where T == Pair[A, B]:
    pass


# CHECK-LABEL: lit.fn @"use_unpack
def use_unpack():
    # CHECK: lit.call @{{.*}}::@"unpack{{.*}}"[{{.*}}]<:!AnyType !Int, :!AnyType !Bool, :!AnyType {{.*}}::@Pair<:!AnyType !Int, :!AnyType !Bool>>
    unpack(Pair[Int, Bool]())


# // -----


# COM: Default values are bound last

@fieldwise_init
struct Wrapper[A: AnyType](Copyable, Movable):
    var x: Int


def constrained_default[
    A: AnyType,
    B: AnyType = Int,
    //,
](w: Wrapper[A]) where A == B:
    pass


# CHECK-LABEL: lit.fn @"use_constrained_default
def use_constrained_default():
    # COM: `B` is `Bool` (from `A == B`), not its `Int` default.
    # CHECK: lit.call @{{.*}}::@"constrained_default{{.*}}"[{{.*}}]<:!AnyType !Bool, :!AnyType !Bool>
    constrained_default(Wrapper[Bool](0))
