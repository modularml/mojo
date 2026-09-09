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

# A closure cannot derive an interior origin from a captured container while
# an interior reference to the same container is also captured.
# TODO: lift this limitation

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


struct MyPtr[origin: Origin[mut=False]]:
    var p: UnsafePointer[Int, Self.origin]

    def __init__(out self, *, ref [Self.origin] to: Int):
        self.p = UnsafePointer[Int, Self.origin].address_of(to)

    def get(self) -> Int:
        return self.p[]


def test_derive_and_capture_interior[O: Origin[mut=True]](ref [O] list: MyList[Int]):
    var p = MyPtr(to=list[])

    @always_inline
    def inner() {imm}:
        # expected-error @below {{cannot derive an interior origin from a captured container while an interior reference to the container is also captured}}
        _ = list[]
        _ = p.get()

    inner()


def main():
    var list = MyList[Int]()
    test_derive_and_capture_interior(list)
