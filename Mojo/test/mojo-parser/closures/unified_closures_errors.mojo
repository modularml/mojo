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
# RUN: %parse-mojo-isolated -verify-diagnostics %s

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder

trait MyInterface:
    def thing(self):
        ...


def make_closure(x: Int) -> Int:
    def parametric[T: MyInterface](a: T):
        # `X` is never referenced, so it's dead code the same as any other
        # unreferenced comptime alias: never resolved, no diagnostic.
        comptime X = A
        pass

    return x

struct Mem(ImplicitlyCopyable):
    pass

def use(a:Mem):
    pass

# COM: ambiguous captures

def aThing(x: Int) -> Int:
    return x


def aThing() -> Int:
    return 5


def definesClosure():
    # expected-error @below {{ambiguous captured value: 'aThing'}}
    def aClosure() {var aThing}:
        pass


struct Bar(ImplicitlyCopyable, RegisterPassable):
    var x: Int
    var y: Int

    def __init__(out self, *, copy: Self):
        pass

# expected-note @+1 {{function declared here}}
def takeDevicePassable[T: DevicePassable](impl: T):
    pass


def foo(bar: Bar) raises:
    # COM: This should fail because Bar is not trivial.

    def closure(number: Int) {var bar} -> Int:
        return bar.x

    # TODO: Rename Wrappers (MOCO-2541)
    # expected-error @below {{'takeDevicePassable' parameter 'T' has 'DevicePassable' type, but value has type 'def(number: Int) -> Int'}}
    takeDevicePassable[type_of(closure)](closure)


# COM: Test that a  closure capturing a non trivial
# COM:  type does NOT conform to TrivialRegisterPassable.
# expected-note @below {{function declared here}}
def takeTrivialRegisterPassable[T: TrivialRegisterPassable](impl: T):
    pass


def testNonTrivialClosureNotTrivialRegisterPassable(bar: Bar) raises:
    def closure() {var bar} -> Int:
        return bar.x

    # expected-error-re @below {{'{{.*}}' does not conform to trait 'TrivialRegisterPassable'}}
    takeTrivialRegisterPassable(closure)


# COM: Test that a closure capturing a memory-only type does not conform to
# COM: RegisterPassable.
# expected-note @below {{function declared here}}
def takeRegisterPassable[T: RegisterPassable](impl: T):
    pass


def testMemoryOnlyClosureNotRegisterPassable(mem: Mem):
    def closure(number: Int) {var mem} -> Int:
        _ = mem
        return number

    # expected-error-re @below {{'{{.*}}' does not conform to trait 'RegisterPassable'}}
    takeRegisterPassable(closure)


# COM: Test that using deprecated `register_passable` effect emits a warning.
def testRegisterPassableEffectDeprecated():
    # expected-warning @below {{the 'register_passable' function effect is no longer supported; use trait constraints like 'RegisterPassable & def(...) -> ...' instead}}
    def closure(x: Int) register_passable -> Int:
        return x


# expected-note @below {{function declared here}}
def changeIt(mut aString: String):
    pass


def nestedCaptureAll(mut aString: String) raises:
    def aFinalThing(x:Int) {imm}:
        # expected-error @below {{invalid call to 'changeIt': value passed to mutable argument 'aString' must be mutable}}
        changeIt(aString)

        def aChildThing(x:Int) {var}:
            changeIt(aString)



def topLevel(x: String) -> String:
    return x

# expected-note @+1 {{function declared here}}
def takesClosure[T: def(Int) -> Int](cb: T, x: Int) -> Int:
    return cb(x)


def useTopLevelClosure():
    # expected-error @below {{invalid call to 'takesClosure': 'takesClosure' parameter 'T' has 'def(Int) -> Int' type, but value has type 'def topLevel(x: String) thin -> String'}}
    # expected-note @below {{a thin function cannot bind to a closure trait; use 'type_of(topLevel)' to pass its type instead}}
    takesClosure[topLevel](topLevel, 1)


# ===----------------------------------------------------------------------=== #
# Closure type mismatch errors
# ===----------------------------------------------------------------------=== #

trait Animal:
    def speak(self):
        ...


trait Mammal(Animal):
    pass

struct Dog(Mammal, Movable where False):
    def speak(self):
        pass

# expected-note @below {{function declared here}}
def takeClosureMammalParam[W: Mammal, C: def (x: W) -> None](impl: C):
    pass


def traitConstraintMismatch[Q: Animal]():
    def closure(x: Q) {var}:
        x.speak()

    # expected-error @below {{does not conform to trait 'def(x: W) -> None'}}
    takeClosureMammalParam(closure)

    def closureWrongConvention(mut x: Dog) {var}:
        x.speak()

    # expected-error @below {{'takeClosureMammalParam' parameter 'C' has 'def(x: W) -> None' type, but value has type 'def(mut x: Dog) -> None'}}
    takeClosureMammalParam[Dog, type_of(closureWrongConvention)](closureWrongConvention)

# ===----------------------------------------------------------------------=== #
# Enforce Parameter Capture
# ===----------------------------------------------------------------------=== #

trait Coord(ImplicitlyCopyable):
  pass

struct Cartesian(Coord):
   var x: Int
   var y: Int
   var z: Int

struct Sphere(Coord):
   var theta: Int
   var phi: Int


# expected-note @below {{constraint declared here evaluated to False}}
# expected-note @below {{function declared here}}
def takeClosure[T: Coord, C:def() -> T](impl: C) -> T:
   _ = impl()


def makeClosure[B:Int](something: Cartesian):
   def closureImpl() {var} -> Cartesian:
      return something
   # expected-error @below {{invalid call to 'takeClosure': violated constraint}}
   takeClosure[Sphere, type_of(closureImpl)](closureImpl)


# ===----------------------------------------------------------------------=== #
# Non-compatible parameter signatures disqualify conformance
# ===----------------------------------------------------------------------=== #

def _print(x: Int):
    pass

# expected-note @below {{function declared here}}
def callee_no_params[
    func: def() -> None,
    //,
](closure: func):
    closure()


def incompatible_param_signature() raises:
    var x = 42

    @always_inline
    def my_func[param_only: Int]() {imm x}:
        _print(x)

    # expected-error @below {{does not conform to trait 'def() -> None'}}
    callee_no_params(my_func)

# ===----------------------------------------------------------------------=== #
# Multiple default capture conventions specified
# ===----------------------------------------------------------------------=== #


def multiple_default_capture_conventions(x: Int):
    # expected-error @below {{default capture convention was already specified; remove the duplicate}}
    # expected-note @below {{a capture convention (like 'mut' or 'var') before the capture list sets the default for all captured variables}}
    def my_closure(y: Int) {var, ref} -> Int:
        return y

# ===----------------------------------------------------------------------=== #
# Incompatible capture conventions
# ===----------------------------------------------------------------------=== #


def incompatible_capture_conventions(x: Int):
    # expected-error @below {{'^' requires 'var' convention; write 'var x^' to move a capture}}
    def my_closure(y: Int) {ref x^} -> Int:
        return y


# ===----------------------------------------------------------------------=== #
# Default capture convention violation
# ===----------------------------------------------------------------------=== #

def default_capture_convention_violation():
    var y = 20
    var x = 10

    def my_fn() {imm, mut y}:
        # Assigning to `y` work
        y = 20
        # expected-error @below {{expression must be mutable in assignment}}
        x = 10

# ===----------------------------------------------------------------------=== #
# Capture RTP with no-read convention
# ===----------------------------------------------------------------------=== #

def capture_RTP(x : Int) :
    # expected-error @below{{register passible value 'x' can not be captured by 'mut'. Do you mean 'imm'?}}
    def my_func() {mut x}:
        pass

# ===----------------------------------------------------------------------=== #
# By-copy / by-move unified closure captures still require a conforming type
# when no scope assumption refines the generic bound (MOCO-4229 regression
# guard: refinement must not swallow this diagnostic for a truly unbound T).
# ===----------------------------------------------------------------------=== #

trait NotImplicitlyCopyable:
    pass


# expected-error @below {{value of type 'T' cannot be implicitly copied, it does not conform to 'ImplicitlyCopyable'}}
def capture_by_copy_no_refinement[T: NotImplicitlyCopyable](z: T):
    # expected-error @below {{cannot capture z by copy because it is not copyable.}}
    def f() {var z}:
        _ = z

    f()


trait NotMovable:
    pass


def capture_by_move_no_refinement[T: NotMovable](var z: T):
    # expected-error @below {{Cannot capture z by move because the type is not movable}}
    def h() {var z^}:
        _ = z

    h()


# ===----------------------------------------------------------------------=== #
# A closure capturing a DevicePassable value by reference is not DevicePassable
# ===----------------------------------------------------------------------=== #
# COM: A by-reference capture stores a host pointer (LIT::RefType) as its
# COM: storage field, so the closure cannot be device-encoded even though the
# COM: capture's type is itself DevicePassable (MOCO-4045). .


@fieldwise_init
struct RegPassableDevice(
    DevicePassable, ImplicitlyCopyable, TrivialRegisterPassable
):
    comptime device_type: AnyType = Int
    var value: Int

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self.value, target)

    @staticmethod
    def get_type_name() -> String:
        return "RegPassableDevice"


# expected-note @below {{function declared here}}
def takeDevicePassableByRef[T: DevicePassable](impl: T):
    pass


def capture_device_passable_by_reference(value: RegPassableDevice) raises:
    def closure(argument: Int) {imm value} -> Int:
        return argument + value.value

    # expected-error-re @below {{'{{.*}}' does not conform to trait 'DevicePassable'}}
    takeDevicePassableByRef(closure)


@fieldwise_init
struct MemType:
    pass


# expected-note @below {{function declared here}}
def apply(f: Some[RegisterPassable & def() -> None]):
    pass


def main():
    var t = MemType()

    def func() {var t^}:
        pass
    # expected-error @below {{invalid call to 'apply': value passed to 'f' cannot be converted from 'def() -> None' to 'T', argument type 'def() -> None' does not conform to trait 'def() -> None & RegisterPassable'}}
    apply(func)
