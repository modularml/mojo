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


def fn_missing_constraint[n: Int]() where n > 0:
    pass


def overloaded_function(n: Int):
    pass


def overloaded_function(n: Float64):
    pass


def overloaded_function(n: Int, m: Float64):
    pass


@fieldwise_init
struct PosOnlyStruct[a: Int, b: Int, /, c: Int = 9]:
    pass


struct DeprecatedImplicitConversion:
    @implicit(deprecated=True)
    def __init__(out self, value: Int):
        pass


@always_inline("builtin")
def foldable_predicate(y: Int) -> Bool:
    return y > 10


def constraint_fn[a: Int, b: Int]() where a > 0:
    pass


def unfoldable_predicate(y: Int) -> Bool:
    return y > 2


def unprovable_constraints[x: Int]() where unfoldable_predicate(x):
    pass


struct Conflict:
    pass


trait ConflictTraitName:
    def test[a: Int](self):
        pass


trait NoDefaultFunc:
    def doSomething(self):
        ...


trait StillNoDefaultFunc(NoDefaultFunc):
    def doEverything(self):
        ...


trait UnprovableCandidateTrait:
    # expected-note @below {{required by trait method here}}
    def handle(self):
        ...


trait ConflictTraitMethod:
    def test(self) -> Bool:
        return True


trait OtherConflictTraitMethod(ConflictTraitMethod):
    def test(self) -> Bool:
        return False


trait TraitWithMember:
    comptime N: Int
