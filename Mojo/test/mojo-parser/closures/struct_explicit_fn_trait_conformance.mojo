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

# Regression test: declaration-time validation of explicit closure trait
# conformance (e.g. `struct Foo(def(Int) -> Int):`).

# -----
# COM: Explicit conformance fully satisfied — no diagnostics.

struct GoodCallable(def(Int) -> Int):
    def __call__(self, x: Int) capturing -> Int:
        return x


# -----
# COM: Missing __call__ is caught at struct declaration time.

# expected-error @below {{'MissingCall' does not implement all requirements for 'def(Bool) -> Int'}}
# expected-note @below {{required function '__call__' is not implemented}}
# expected-note @below {{def __call__(self, : Bool, /) -> Int}}
# expected-note @below {{trait 'def(Bool) -> Int' declared here}}
struct MissingCall(def(Bool) -> Int):
    pass


# -----
# COM: Wrong return type is caught at struct declaration time.

# expected-error @below {{'WrongReturnType' does not implement all requirements for 'def(String) -> Int'}}
# expected-note @below {{no '__call__' candidates have type 'def(WrongReturnType, String) capturing thin -> Int'}}
# expected-note @below {{trait 'def(String) -> Int' declared here}}
struct WrongReturnType(def(String) -> Int):
    # expected-note @below {{candidate declared here with type 'def(self: WrongReturnType, x: String) capturing thin -> Bool'}}
    def __call__(self, x: String) capturing -> Bool:
        return True


# -----
# COM: Struct satisfying two closure traits — no diagnostics.

struct TwoTraitsGood(def(Int) -> Int, def(String) -> String):
    def __call__(self, x: Int) capturing -> Int:
        return x

    def __call__(self, x: String) capturing -> String:
        return x


# -----
# COM: Second closure trait conformance fails — error names the failing trait.

# expected-error @below {{'TwoTraitsBadSecond' does not implement all requirements for 'def(Float64) -> Float64'}}
# expected-note @below {{no '__call__' candidates have type 'def(TwoTraitsBadSecond, Float64) capturing thin -> Float64'}}
# expected-note @below {{trait 'def(Float64) -> Float64' declared here}}
struct TwoTraitsBadSecond(def(Int) -> Int, def(Float64) -> Float64):
    # expected-note @below {{candidate declared here with type 'def(self: TwoTraitsBadSecond, x: Int) capturing thin -> Int'}}
    # expected-note @below {{.x.dtype of the first value is 'DType.float64' but the second value is 'DType.int'}}
    def __call__(self, x: Int) capturing -> Int:
        return x

    # expected-note @below {{candidate declared here with type 'def(self: TwoTraitsBadSecond, x: Float64) capturing thin -> String'}}
    def __call__(self, x: Float64) capturing -> String:
        return ""


# -----
# COM: A struct that explicitly declares conformance can be passed to a
# COM: `def(Int)` parameter and called (positive, end-to-end). `capturing` is
# COM: not required on `__call__` to conform.

# expected-note @below {{function declared here}}
def call_it[F: def(Int)](func: F):
    func(5)


@fieldwise_init
struct ExplicitCallable(def(Int)):
    var value: Int

    def __call__(self, arg: Int):
        _ = arg + self.value


def test_explicit_conformance_end_to_end():
    var c = ExplicitCallable(10)
    call_it(c)


# -----
# COM: A struct with a compatible __call__ but no explicit `(def(Int))`
# COM: declaration is rejected — implicit conformance is not allowed.

@fieldwise_init
struct NoExplicitConformance(Movable where False):
    var value: Int

    def __call__(self, arg: Int):
        _ = arg + self.value


def test_implicit_conformance_rejected():
    var c = NoExplicitConformance(10)
    # expected-error @below {{invalid call to 'call_it': value passed to 'func' cannot be converted from 'NoExplicitConformance' to 'F', argument type 'NoExplicitConformance' does not conform to trait 'def(Int) -> None'}}
    call_it(c)
