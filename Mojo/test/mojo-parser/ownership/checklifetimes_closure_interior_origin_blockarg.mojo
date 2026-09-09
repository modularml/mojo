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
# RUN: %parse-mojo-isolated %s -mlir-print-debuginfo | kgen-opt -lower-semantic-cf -check-lifetimes | FileCheck %s

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder

# A unified closure capturing a value whose type embeds an interior origin
# outlines that interior origin into a fresh parameter on the closure's
# storage struct, binding the interior origin at the instantiation site.

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


# Register-passable wrapper so a `{var}` capture is DevicePassable and the
# outlined interior origin is also rewritten into `__device_type` field types.
@fieldwise_init
struct DevicePtr[
    T: AnyType,
    origin: Origin,
](DevicePassable, ImplicitlyCopyable, TrivialRegisterPassable):
    comptime device_type: AnyType = Self
    var p: Pointer[Self.T, Self.origin]

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "DevicePtr"


# Storage structs are emitted before the parent functions. Quoted names use
# `*"?__interior_origin_N"` because `?` is not a legal bare identifier.

# `{var}` capture of a DevicePassable wrapper is device-encodable, so the
# outlined interior appears on both `__device_type` and host storage.
# CHECK-LABEL: lit.struct.decl @"device_interior()::closure::__storage::__device_type"<
# CHECK-SAME: *"?__interior_origin_0": origin<true>
# CHECK: lit.struct.field wrapped : {{.*}}*"?__interior_origin_0"
# CHECK-LABEL: lit.struct.decl @"device_interior()::closure::__storage"<
# CHECK-SAME: *"?__interior_origin_0": origin<true>
# CHECK: lit.struct.field wrapped : {{.*}}*"?__interior_origin_0"

# Two captures whose types embed distinct interiors outline two parameters.
# CHECK-LABEL: lit.struct.decl @"two_interiors()::closure::__storage"<
# CHECK-SAME: *"?__interior_origin_0": origin<true>
# CHECK-SAME: *"?__interior_origin_1": origin<true>
# CHECK: lit.struct.field p : {{.*}}*"?__interior_origin_0"
# CHECK: lit.struct.field q : {{.*}}*"?__interior_origin_1"

# Capturing both the container and an element keeps the base origin as a
# storage parameter; the interior is still outlined separately.
# `p`'s own reference is captured by `imm` (promoted to origin<false>),
# but its pointee origin remains mutable (origin<true>).
# CHECK-LABEL: lit.struct.decl @"keep_base{{.*}}::__storage"<
# CHECK-SAME: O._mlir_origin
# CHECK-SAME: origin<false>
# CHECK-SAME: *"?__interior_origin_0": origin<true>
# CHECK: lit.struct.field list : {{.*}}*"O._mlir_origin
# CHECK: lit.struct.field p : !lit.ref<!lit.struct<#Pointer {{.*}}*"?__interior_origin_0"{{.*}}>, imm *"p

# Capturing the interior-origin *element* by `imm` mutcasts the outlined
# parameter to origin<false>.
# CHECK-LABEL: lit.struct.decl @"promote_interior()::closure::__storage"<
# CHECK-SAME: *"?__interior_origin_0": origin<false>
# CHECK: lit.struct.field r : !lit.ref<!Int, imm *"?__interior_origin_0">

# CHECK-LABEL: lit.struct.decl @"outer()::closure::__storage"<
# CHECK-SAME: *"?__interior_origin_0": origin<true>
# CHECK: lit.struct.field p : !lit.ref<!lit.struct<#Pointer {{.*}}*"?__interior_origin_0"
# CHECK-LABEL: lit.fn @"outer()"
def outer():
    var list = MyList[Int]()

    # `p`'s type embeds the interior origin `list["element"]`, so capturing it
    # outlines that origin into `?__interior_origin_0` on the closure storage.
    var p = Pointer(to=list[])

    @always_inline
    def closure() {imm}:
        _ = p[]

    closure()


# CHECK-LABEL: lit.fn @"promote_interior()"
# CHECK: lit.call {{.*}}promote_interior()::closure::__storage"::@"__init__
# CHECK-SAME: (mutcast mut {{.*}}["element"]
def promote_interior():
    var list = MyList[Int]()
    ref r = list[]

    @always_inline
    def closure() {imm r}:
        _ = r

    closure()


def keep_base[O: Origin[mut=True]](ref [O] list: MyList[Int]):
    var p = Pointer(to=list[])

    @always_inline
    def closure() {imm}:
        _ = list
        _ = p[]

    closure()


def two_interiors():
    var list_a = MyList[Int]()
    var list_b = MyList[Int]()
    var p = Pointer(to=list_a[])
    var q = Pointer(to=list_b[])

    @always_inline
    def closure() {imm}:
        _ = p[]
        _ = q[]

    closure()


def device_interior():
    var list = MyList[Int]()
    var wrapped = DevicePtr(Pointer(to=list[]))

    @always_inline
    def closure() {var}:
        _ = wrapped.p[]

    closure()
