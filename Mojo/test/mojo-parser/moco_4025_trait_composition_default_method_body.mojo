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

# Regression test for MOCO-4025: trait-composition default method body
# resolution.
#
# The generic function refines `T: AnyType` with a local equality trait. This
# creates an `AnyType & LocalEquatable` trait composition for method lookup. The
# inherited default method body must not be resolved while building that
# composition; it should remain tied to the declaring trait/concrete witness.

# RUN: %parse-mojo-isolated %s | FileCheck %s


trait LocalEquatable:
    def __eq__(self, other: Self) -> Bool:
        comptime assert conforms_to(
            type_of(__struct_field_ref(0, self)), LocalEquatable
        )
        if __struct_field_ref(0, self) != __struct_field_ref(0, other):
            return False
        return True

    def __ne__(self, other: Self) -> Bool:
        return not self == other


struct FieldValue(LocalEquatable, Movable where False):
    def __init__(out self):
        pass

    def __eq__(self, other: Self) -> Bool:
        return True


struct LocalValue(LocalEquatable, Movable where False):
    var field: FieldValue

    def __init__(out self):
        self.field = FieldValue()


# CHECK-LABEL: lit.fn @"eq_on_refined
def eq_on_refined[
    T: AnyType
](x: T, y: T) -> Bool where conforms_to(T, LocalEquatable):
    return x == y


def test_eq_on_refined():
    var x = LocalValue()
    var y = LocalValue()
    _ = eq_on_refined(x, y)
