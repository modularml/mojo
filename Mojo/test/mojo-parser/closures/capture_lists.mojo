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
# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values -split-input-file| FileCheck %s

def takeIt[T: def (prefix: String) -> String, //](state: T, prefix:String):
    _ = state(prefix)



struct MoveMe(Movable):
    var x:Int

def use(a:String, d:MoveMe):
    pass

# CHECK:  lit.fn @"moveMeUser
def moveMeUser(byCopy:String, prefix:String, var byMove: MoveMe):
    # COM: `var byCopy` copies the String into a temporary via the copy ctor.
    # CHECK: [[V0:%.*]] = lit.var.decl "anonymous*"
    # CHECK-NEXT: lit.call {{.*}}::@String::@"__init__{{.*}}(%byCopy, [[V0]]){{.*}}(*, "copy":
    # COM: The closure storage initializer receives byCopy by copy (imm) and
    # COM: byMove by move (owned_in_mem).
    # CHECK: lit.call {{.*}}myclosure::__storage"::@"__init__
    # CHECK-SAME: %byMove,
    # CHECK-SAME: "byMove": !lit.ref<!MoveMe, {{[^>]*}}> owned_in_mem
    def myclosure(prefix: String) {var byCopy, var byMove^} -> String:
        use(byCopy, byMove)
        return prefix

    takeIt(myclosure, prefix)

# // -----

# COM: Trivial Capture

# CHECK:  lit.fn @"make_closure
def make_closure(x: Int):
    # COM: `var x` on a trivial Int becomes a trivial copy threaded through the
    # COM: closure storage initializer.
    # CHECK: lit.call {{.*}}my_closure::__storage"::@"__init__
    # CHECK-SAME: "x": !lit.ref<!Int, {{[^>]*}}> imm_mem
    def my_closure(y: Int) {var x} -> Int:
        return x + y

# // -----

# COM: Nested Captures

def use(y:String):
    pass

def make_closure(x: Int, str:String):
    # COM: The outer closure copies its captures into its storage struct.
    # CHECK-DAG: lit.var.decl "anonymous*"
    # CHECK-DAG: lit.call {{.*}}my_closure::__storage"::@"__init__
    # COM: The nested closure copies the outer copies again into its storage.
    # CHECK-DAG: lit.var.decl "my_nested_closure"
    # CHECK-DAG: lit.call {{.*}}my_nested_closure::__storage"::@"__init__
    def my_closure(y: Int) {var x, var str} -> Int:
        def my_nested_closure(z: Int) {var x, var str} -> Int:
            use(str)
            return x

        return x + y

# // -----

# COM: Nested unified closure capturing locals declared in a nested lexical
# COM: scope of the enclosing closure (MOCO-4652).

def takeInt[T: def (x: Int) -> Int, //](state: T, x: Int):
    _ = state(x)


def capture_nested_scope_local(t: Int):
    def task(x: Int) {imm}:
        if x:
            var kg = x
            # CHECK: lit.call {{.*}}use_imm::__storage"::@"__init__
            # CHECK-SAME: "kg":
            def use_imm(i: Int) {imm kg} -> Int:
                return kg + i

            var acc = x
            # CHECK: lit.call {{.*}}use_mut::__storage"::@"__init__
            # CHECK-SAME: "acc":
            def use_mut(i: Int) {mut acc, imm} -> Int:
                acc += i
                return acc

            _ = takeInt(use_imm, t)
            _ = takeInt(use_mut, t)
        while x:
            var kg = x
            # CHECK: lit.call {{.*}}use_while::__storage"::@"__init__
            # CHECK-SAME: "kg":
            def use_while(i: Int) {imm kg} -> Int:
                return kg + i

            _ = takeInt(use_while, t)
            break
    task(t)

# // -----

# COM: Verify mutability casts are inserted

def takeIt[T: def () -> None, //](state: T):
    state()

def takesImmut(str: String):
    pass

def takesMut(mut str: String):
    pass

# CHECK: lit.fn @"no_castsImmut({{.*}})"[imm *"byRef`"
def no_castsImmut(byRef:String):
    # COM: An imm capture of an immutable ref needs no mutability cast.
    # CHECK: "byRef": !lit.ref<!String, imm *"byRef`"> ref
    def myclosure() {imm byRef}:
        takesImmut(byRef)

    takeIt(myclosure)

# CHECK: lit.fn @"no_castsMut({{.*}})"[mut *"byRefMut`"
def no_castsMut(mut byRefMut: String):
    # COM: A mut capture of a mutable ref needs no mutability cast.
    # CHECK: "byRefMut": !lit.ref<!String, mut *"byRefMut`"> ref
    def myclosure() {mut byRefMut}:
        takesImmut(byRefMut)

    takeIt(myclosure)

# CHECK: lit.fn @"casts({{.*}})"[mut *"byRefMut`"
def casts(mut byRefMut: String):
    # COM: An imm capture of a mutable ref inserts a mut-to-immutable cast.
    # CHECK: [[V0:%.*]] = lit.ref.immut %byRefMut : <!String, mut *"byRefMut`">
    # CHECK: lit.call {{.*}}myclosure::__storage"::@"__init__
    # CHECK-SAME: "byRefMut": !lit.ref<!String, muttoimm *"byRefMut`"> ref
    def myclosure() {imm byRefMut}:
        takesImmut(byRefMut)

    takeIt(myclosure)

# // -----

# COM: Verify ref capture preserves original mutability

def use(ref a: String, ref b: String, ref c: String):
    pass


# COM: Capture-all-by-ref preserves each value's original mutability
# CHECK-LABEL: lit.fn @"captureAllByRef
def captureAllByRef(A: String, mut B: String, ref C: String):
    # CHECK-NOT: lit.ref.immut
    # CHECK: lit.call {{.*}}refAll::__storage"::@"__init__
    # CHECK-SAME: "A": !lit.ref<!String, imm *"A`"> ref
    # CHECK-SAME: "B": !lit.ref<!String, mut *"B`1"> ref
    # CHECK-SAME: "C": !lit.ref<!String, mut=*"C_is_mut`2", *"C_is_origin`3"> ref
    def refAll() {ref}:
        use(A, B, C)
        pass


# // -----

# COM: Ensure "capture all by" emits the correct IR.

def takeIt[T: def () -> String, //](state: T):
    _ = state()


def use(a: String, d: String):
    pass


# CHECK-LABEL:  lit.fn @"toy
def toy(A: String, B: String, mut C: String, mut D: String):
    # COM: `imm` capture-all threads A and B into storage by immutable ref.
    # CHECK: lit.call {{.*}}immAll::__storage"::@"__init__
    # CHECK-SAME: "A": !lit.ref<!String, imm *"A`"> ref
    # CHECK-SAME: "B": !lit.ref<!String, imm *"B`1"> ref
    def immAll() {imm} -> String:
        use(A, B)
        return A
    takeIt(immAll)

    # COM: `var` capture-all copies each value before storing it.
    # CHECK: @String::@"__init__{{.*}}"{{.*}}*, "copy"
    # CHECK: @String::@"__init__{{.*}}"{{.*}}*, "copy"
    # CHECK: lit.call {{.*}}copyAll::__storage"::@"__init__
    def copyAll() {var} -> String:
        use(A, B)
        return C

    takeIt(copyAll)

    # COM: `var^` capture-all moves each value into storage (owned_in_mem).
    # CHECK: lit.call {{.*}}moveAll::__storage"::@"__init__
    # CHECK-SAME: "C": !lit.ref<!String, {{[^>]*}}> owned_in_mem
    # CHECK-SAME: "D": !lit.ref<!String, {{[^>]*}}> owned_in_mem
    def moveAll() {var^} -> String:
        use(C, D)
        return D
    takeIt(moveAll)


# COM: Ensure multiple references to the same capture result in a single copy

struct MyCopyableType(ImplicitlyCopyable):
    def __init__(out self, *, copy: Self):
        pass

def use(y: MyCopyableType, wy:MyCopyableType):
    pass

# CHECK: lit.fn @"testOnce
def testOnce(x: MyCopyableType):
    # CHECK-COUNT: 1 @MyCopyableType::@"__init__{{.*}}"{{.*}}*, "copy"
    def myclosure() {var}:
        use(x, x)

# // -----

# COM: Trailing commas are supported

def callIt(x: String, x1: String, x2: String, x3: String, x4: String) raises:
    pass


@no_inline
def takeIt[T: def () raises -> None](impl: T) raises:
    impl()


# CHECK-LABEL: lit.fn @"longCaptureLists
def longCaptureLists(
    mut something: String,
    mut something1: String,
    mut something2: String,
    mut something3: String,
    mut something4: String,
    mut something5: String,
) raises:
    # COM: The mixed capture list threads each value through the storage
    # COM: initializer with its own convention.
    # CHECK: lit.call {{.*}}closure::__storage"::@"__init__
    # CHECK-SAME: "something": !lit.ref<!String, {{[^>]*}}> imm_mem
    # CHECK-SAME: "something2": !lit.ref<!String, mut {{[^>]*}}> ref
    # CHECK-SAME: "something3": !lit.ref<!String, muttoimm {{[^>]*}}> ref
    def closure() raises {
        var something,
        mut something2,
        imm something3,
        mut something4,
        imm something5,
    }:
        callIt(something, something2, something3, something4, something5)

    takeIt(closure)

# // -----

# COM: `imm` marks a capture as an immutable reference - no mutability cast
# COM: should be inserted, for a named capture or a capture-all-by-convention.

def takeIt[T: def () -> None, //](state: T):
    state()

def takesImmut(str: String):
    pass

# CHECK: lit.fn @"no_castsImm({{.*}})"[imm *"byRef`"
def no_castsImm(byRef:String):
    # COM: `imm byRef` captures by immutable ref with no mutability cast.
    # CHECK: "byRef": !lit.ref<!String, imm *"byRef`"> ref
    def myclosure() {imm byRef}:
        takesImmut(byRef)

    takeIt(myclosure)

# // -----

def use(a: String, b: String):
    pass

def takeIt[T: def () -> String, //](state: T):
    _ = state()

# CHECK-LABEL:  lit.fn @"toyImm
def toyImm(A: String, B: String):
    # COM: `imm` capture-all threads A and B into storage by immutable ref.
    # CHECK: lit.call {{.*}}immAll::__storage"::@"__init__
    # CHECK-SAME: "A": !lit.ref<!String, imm *"A`"> ref
    # CHECK-SAME: "B": !lit.ref<!String, imm *"B`1"> ref
    def immAll() {imm} -> String:
        use(A, B)
        return A
    takeIt(immAll)

# // -----

# Make sure sugar is preserved correctly for argument type.

comptime MyKey = Movable

struct Inner[T: MyKey](Movable where False):
    var _x: Int

    def __init__(out self):
        self._x = 0

    def deinit_with(
        deinit self, deinit_func: Some[def(var Self.T, var NoneType)], /
    ):
        pass


struct Outer[T: MyKey](Movable where False):
    var _inner: Inner[Self.T]

    def __init__(out self):
        self._inner = Inner[Self.T]()

    def deinit_with(deinit self, deinit_func: Some[def(var Self.T)], /):
        # CHECK: lit.fn @"forward
        # CHECK-SAME: %key: !lit.ref<:!alias_MyKey
        def forward(var key: Self.T, var value: NoneType) {imm deinit_func}:
            deinit_func(key^)

        self._inner^.deinit_with(forward)

# // -----

# COM: Copy-capturing method `self`. Storage `__init__` already has a trailing
# COM: byref-result named `self`; the capture argument must use a distinct
# COM: spelling so the two origins do not alias.

struct CaptureSelf(ImplicitlyCopyable, Movable):
    var x: Int

    def __init__(out self, x: Int):
        self.x = x

    # CHECK: lit.fn @"method
    def method(self):
        # CHECK: lit.call {{.*}}inner::__storage"::@"__init__
    # CHECK-SAME: "__capture_self"
        def inner() {var self}:
            _ = self.x
