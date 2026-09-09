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
# RUN: %parse-mojo-isolated %s | kgen-opt -lower-semantic-cf -check-lifetimes | FileCheck %s

# A closure that captures an interior-origin reference into its enclosing
# function makes CheckLifetimes analyze a nested function reaching into its
# parent's IR. Verify it lowers through the pass cleanly (the RUN succeeding
# means CheckLifetimes neither errors nor crashes) and that the closure
# captures the interior origin.
#
# This guards the pattern behind the CheckLifetimes data race. To test the data
# race, this test needs to be executed using ThreadSanitizer.

# A minimal interior-origin-producing container (no stdlib at the parser level).
struct MyList[T: AnyType](Movable where False):
    var data: UnsafePointer[Self.T, UntrackedOrigin[mut=True]]

    def __init__(out self):
        self.data = UnsafePointer[
            Self.T, UntrackedOrigin[mut=True]].unsafe_dangling()

    def __deinit__(deinit self):
        pass

    @__unsafe_nested_origins_read_only
    def __getitem__(
        ref self,
    ) -> ref[self.data._get_ref_with_unsafe_interior_origin["element"](self)] Self.T:
        return self.data._get_ref_with_unsafe_interior_origin["element"](self)


def use_any[*Ts: AnyType](*args: *Ts):
    pass


# CHECK-LABEL: lit.fn @"outer()"
def outer():
    var list = MyList[Int]()
    # The interior-origin reference; its origin carries the ["element"] tag.
    # CHECK: lit.var.decl "r" ref {{.*}}["element"]
    ref r = list[]
    use_any(r)

    # A nested closure capturing that interior origin, then called indirectly.
    # CHECK: lit.fn {{.*}}emit(){{.*}}["element"]{{.*}}capturing
    # CHECK: lit.call{{.*}}emit()
    @__parameter
    def emit():
        use_any(r)

    emit()
