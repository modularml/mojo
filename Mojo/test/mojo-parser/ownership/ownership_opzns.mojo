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

# RUN: %parse-mojo-isolated %s --mlir-print-debuginfo -o %t.mlir
# RUN: kgen-opt %t.mlir -lower-semantic-cf -check-lifetimes -verify-parameters -verify-diagnostics | FileCheck %s
# RUN: %parse-mojo-isolated %s --mlir-print-debuginfo --debug-level full -o /dev/null

# Test for CheckLifetimes optimizations.


# CHECK-LABEL: lit.struct.decl @MemExample
struct MemExample(ImplicitlyCopyable):
    var x: Int

    def __init__(out self):
        self.x = 42
        pass

    def noop(self):
        pass

    def __init__(out self, *, deinit move: Self):
        self.x = move.x

    def __init__(out self, *, copy: Self):
        self.x = copy.x

    def __bool__(self) -> Bool:
        return True

    def __deinit__(deinit self):
        pass


# CHECK-LABEL: lit.struct.decl @RegExample
struct RegExample(ImplicitlyCopyable, RegisterPassable):
    def __init__(out self):
        return

    def __init__(out self, *, copy: Self):
        return

    def noop(self):
        pass

    def __deinit__(deinit self):
        pass

    def mutate(mut self):
        pass


# This type is a unique value that cannot be moved without ending lifetime.
# CHECK-LABEL: lit.struct.decl @MemoryUniqueMovable
struct MemoryUniqueMovable(Movable):
    var state: MemExample

    def __init__(out self):
        self.state = MemExample()

    # CHECK: lit.fn @"__init__{{.*}}(*, %move:
    def __init__(out self, *, deinit move: Self):
        # Mercilessly steal 'move's state which could be interesting.

        # CHECK-NEXT: %0 = lit.ref.struct.ger %self[state]
        # CHECK-NEXT: %1 = lit.ref.struct.ger %move[state]
        # CHECK-NEXT: lit.ownership.use %1
        # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}*, "move":
        self.state = move.state^

        # CHECK-NEXT: kgen.param.constant: none
        # CHECK-NEXT: lit.ownership.mark_destroyed %move
        # CHECK-NEXT: kgen.return


# This type is copyable/moveable.
# CHECK-LABEL: lit.struct.decl @MemoryMovableCopyable
struct MemoryMovableCopyable(ImplicitlyCopyable):
    var state: MemExample

    def __init__(out self):
        self.state = MemExample()

    def __init__(out self, *, deinit move: Self):
        # Mercilessly steal 'move's state which could be interesting.
        self.state = move.state^

    def __init__(out self, *, copy: Self):
        self.state = copy.state

    def __deinit__(deinit self):
        pass


# CHECK-LABEL: lit.fn @"result_mem1
def result_mem1(var a: MemoryUniqueMovable) -> MemoryUniqueMovable:
    # CHECK-NEXT: lit.ownership.use %a
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}move"
    # CHECK-NEXT: kgen.param.constant: none
    # CHECK-NEXT: kgen.return
    return a^


# CHECK-LABEL: lit.fn @"result_mem3
def result_mem3(var a: MemoryMovableCopyable) -> MemoryMovableCopyable:
    # CHECK-NEXT: lit.ownership.use %a
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}move"{{.*}} deinit_mem{{.*}}byref_result
    # CHECK-NEXT: kgen.param.constant: none
    # CHECK-NEXT: kgen.return
    return a^


# CHECK-LABEL: lit.fn @"self_copy
def self_copy(mut x: MemoryMovableCopyable):
    # Mojo introduces a temporary to avoid exclusivity error.
    # CHECK: %__call_result_tmp__ = lit.var.decl
    # CHECK: lit.call {{.*}}__init__{{.*}}move"
    # CHECK: lit.call {{.*}}__init__{{.*}}move"
    x = x


struct RegUniqueMovable(RegisterPassable):
    def __init__(out self):
        return

    def __deinit__(deinit self):
        pass


struct RegMovableCopyable(ImplicitlyCopyable, RegisterPassable):
    def __init__(out self):
        return

    def __init__(out self, *, copy: Self):
        return

    def __deinit__(deinit self):
        pass


# CHECK-LABEL: lit.fn @"result_reg1
def result_reg1(var a: RegUniqueMovable) -> RegUniqueMovable:
    # CHECK-NEXT: lit.ownership.use %a
    # CHECK-NEXT: [[AVAL:%.*]] = lit.load.consume %a
    # CHECK-NEXT: kgen.return [[AVAL]]
    return a^


# CHECK-LABEL: lit.fn @"result_reg2
def result_reg2(var a: RegMovableCopyable) -> RegMovableCopyable:
    # CHECK-NEXT: [[A:%.*]] = lit.load.consume %a
    # CHECK-NEXT: kgen.return [[A]]
    return a


# CHECK-LABEL: lit.fn @"result_reg3
def result_reg3(var a: RegMovableCopyable) -> RegMovableCopyable:
    # CHECK-NEXT: lit.ownership.use %a
    # CHECK-NEXT: [[A:%.*]] = lit.load.consume %a
    # CHECK-NEXT: kgen.return [[A]]
    return a^


# CHECK-LABEL: lit.fn @"result_reg4
def result_reg4(var a: RegMovableCopyable) -> RegMovableCopyable:
    # CHECK-NEXT: lit.ownership.use %a
    # CHECK-NEXT: %x = lit.var.decl "x"
    # CHECK-NEXT: [[A:%.*]] = lit.load.consume %a
    # CHECK-NEXT: lifetime.start %x
    # CHECK-NEXT: lit.ref.store [[A]], %x
    var x = a^

    # CHECK-NEXT: lit.ownership.use %x
    # CHECK-NEXT: [[RES:%.*]] = lit.load.consume %x
    # CHECK-NEXT: lifetime.end %x
    # CHECK-NEXT: kgen.return [[RES]]
    return x^


def takeOwnedInt(var x: Int):
    pass


# CHECK-LABEL: lit.fn @"passFieldToOwnedInt
def passFieldToOwnedInt(var a: MemExample):
    # CHECK-NEXT: %0 = lit.ref.struct.ger %a[x]
    # CHECK-NEXT: %1 = lit.ref.load %0
    # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%a)
    # CHECK-NEXT: [[RB:%.*]] = kgen.rebind %1
    # CHECK-NEXT: [[ANON:%.*]] = lit.var.decl "anonymous*
    # CHECK-NEXT: lit.var.lifetime.start [[ANON]]
    # CHECK-NEXT: lit.ref.store [[RB]], [[ANON]]
    # CHECK-NEXT: lit.call {{.*}}takeOwnedInt{{.*}}([[ANON]])
    # CHECK-NEXT: lit.var.lifetime.end [[ANON]]
    takeOwnedInt(a.x)

    # CHECK-NEXT: kgen.param.constant: none


# Generic type: Issue #14018
struct MyGenericType[Type: TrivialRegisterPassable](Movable where False):
    var value: Self.Type

    @implicit
    def __init__(out self, v: Self.Type):
        self.value = v


def takeTwo(var x: RegExample, var y: RegExample):
    pass


def takeTwo(var x: MemExample, var y: MemExample):
    pass


# Check that copies that are immediately destroyed are elided.
# CHECK-LABEL: lit.fn @"optimizeCopyElision
def optimizeCopyElision():
    # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}()
    # CHECK: %a = lit.var.decl "a"
    # CHECK-NEXT: lifetime.start %a
    # CHECK-NEXT: lit.ref.store [[TMP]], %a
    var a = RegExample()

    # We need one copy of 'a' here, not two + dtor.
    # CHECK-NEXT: [[A:%.*]] = lit.ref.immut %a
    # CHECK-NEXT: [[COPY1:%.*]] = lit.call {{.*}}__init__{{.*}}copy"
    # CHECK-NEXT: [[COPY2:%.*]] = lit.load.consume %a
    # CHECK-NEXT: lifetime.end %a

    # CHECK-NEXT: [[ANON:%.*]] = lit.var.decl
    # CHECK-NEXT: lifetime.start [[ANON]]
    # CHECK-NEXT: lit.ref.store [[COPY1]], [[ANON]]

    # CHECK-NEXT: [[ANON2:%.*]] = lit.var.decl
    # CHECK-NEXT: lifetime.start [[ANON2]]
    # CHECK-NEXT: lit.ref.store [[COPY2]], [[ANON2]]

    # CHECK-NEXT: lit.call {{.*}}takeTwo{{.*}}([[ANON]], [[ANON2]])
    takeTwo(a, a)
    # CHECK-NEXT: lifetime.end [[ANON2]]
    # CHECK-NEXT: lifetime.end [[ANON]]

    # CHECK-NEXT: %x = lit.var.decl "x"
    # CHECK-NEXT: lifetime.start %x
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%x)
    var x = MemExample()

    # We need one copy of 'x' here, not two + dtor.

    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %x
    # CHECK-NEXT: [[ANON:%.*]] = lit.var.decl "__call_result_tmp__"
    # CHECK-NEXT: lifetime.start [[ANON]]
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}copy"
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %x
    # CHECK-NEXT: kgen.param.declare
    # CHECK-NEXT: [[PTR:%.*]] = kgen.rebind %x
    # CHECK-NEXT: lit.call {{.*}}takeTwo{{.*}}([[ANON]], [[PTR]])
    # CHECK-NEXT: lifetime.end %x
    # CHECK-NEXT: lifetime.end [[ANON]]
    takeTwo(x, x)

    # CHECK-NEXT: kgen.param.constant: none


def consume(var value: MemExample):
    pass


# CHECK-LABEL: lit.fn @"copyElisionArgument
def copyElisionArgument(var value: MemExample):
    # CHECK-NEXT: %0 = lit.ref.immut %value
    # CHECK-NEXT: kgen.param.declare
    # CHECK-NEXT: %1 = kgen.rebind %value
    # CHECK-NEXT: call {{.*}}consume{{.*}}(%1)
    # CHECK-NEXT: %none =
    consume(value)


# CHECK-LABEL: lit.fn @"optimizeCopyToMove
def optimizeCopyToMove():
    # All the copy ctors should be eliminated in favor of moves.

    # CHECK: %m1 = lit.var.decl
    # CHECK-NEXT: lifetime.start %m1
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%m1)
    var m1 = MemExample()
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %m1
    # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[IMMREF]])
    m1.noop()

    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %m1
    # CHECK-NEXT: kgen.param.declare *"m2`
    # CHECK-NEXT: [[M2:%.*]] = kgen.rebind %m1
    var m2 = m1
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut [[M2]]
    # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[IMMREF]])
    m2.noop()

    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut [[M2]]
    # CHECK-NEXT: kgen.param.declare *"m3`
    # CHECK-NEXT: [[M3:%.*]] = kgen.rebind [[M2]]
    var m3 = m2

    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut [[M3]]
    # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[IMMREF]])
    m3.noop()
    # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}([[M3]])
    # CHECK-NEXT: lit.var.lifetime.end %m1

    # All the copyinit's should be removed.

    # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}()
    # CHECK-NEXT: %r1 = lit.var.decl "r1"
    # CHECK-NEXT: lifetime.start %r1
    # CHECK-NEXT: lit.ref.store [[TMP]], %r1
    var r1 = RegExample()
    # CHECK-NEXT: [[R1:%.*]] = lit.ref.immut %r1
    # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[R1]])
    r1.noop()

    # CHECK-NEXT: %r2 = lit.var.decl "r2"
    # CHECK-NEXT: [[TMP:%.*]] = lit.load.consume %r1
    # CHECK-NEXT: lit.var.lifetime.end %r1
    # CHECK-NEXT: lit.var.lifetime.start %r2
    # CHECK-NEXT: lit.ref.store [[TMP]], %r2
    var r2 = r1
    # CHECK-NEXT: [[R2I:%.*]] = lit.ref.immut %r2
    # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[R2I]])
    r2.noop()

    # CHECK-NEXT: %r3 = lit.var.decl "r3"
    # CHECK-NEXT: [[TMP:%.*]] = lit.load.consume %r2
    # CHECK-NEXT: lit.var.lifetime.end %r2
    # CHECK-NEXT: lit.var.lifetime.start %r3
    # CHECK-NEXT: lit.ref.store [[TMP]], %r3
    var r3 = r2
    # CHECK-NEXT: [[R3I:%.*]] = lit.ref.immut %r3
    # CHECK-NEXT: lit.call {{.*}}noop{{.*}}([[R3I]])
    r3.noop()
    # CHECK-NEXT: lit.call {{.*}}__deinit__{{.*}}(%r3)
    # CHECK-NEXT: lifetime.end %r3


# This is an integration test for elideCopyDestroyPair
# CHECK-LABEL: lit.fn @"optimize_copies
def optimize_copies() -> MemExample:
    # CHECK: lit.call {{.*}}__init__{{.*}}(%x
    var x = MemExample()

    # Optimized away, so the vardecl is gone, but the lifetime still gets
    # declared.
    # CHECK-NOT: lit.var.decl
    # CHECK: kgen.param.declare *"y`2":
    var y = x
    # CHECK-NOT: lit.var.decl
    # CHECK: kgen.param.declare *"z`3":
    # CHECK-NOT: lit.var.decl
    var z = y
    # CHECK: lit.call {{.*}}__init__{{.*}}move"
    return z


# This is not optimized, because there are no destructors for CheckLifetimes
# to insert, so it is a different optimization.


# CHECK-LABEL: lit.fn @"optimize_transfers
# Issue #34138
def optimize_transfers() -> MemExample:
    # CHECK: lit.call {{.*}}__init__{{.*}}(%x
    var x = MemExample()

    # CHECK: lit.call {{.*}}__init__{{.*}}move"
    var y = x^
    # CHECK: lit.call {{.*}}__init__{{.*}}move"
    var z = y^
    # CHECK: lit.call {{.*}}__init__{{.*}}move"
    return z^
