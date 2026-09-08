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
# RUN: kgen -elaborate -O0 %s -S | FileCheck %s


@fieldwise_init
struct RegPassable(ImplicitlyCopyable, RegisterPassable):
    var v: Float32
    var w: Float32


@fieldwise_init
struct MemOnly(ImplicitlyCopyable):
    var a: Int
    var b: Int


def owned_generic[T: Deinitable](var x: T):
    pass


def borrowed_generic[T: AnyType](x: T):
    pass


# CHECK-LABEL: kgen.func export @test_owned(
# CHECK-SAME: %arg0: !kgen.struct<(scalar<f32>, scalar<f32>)> owned,
# CHECK-SAME: %arg1: !kgen.pointer<struct<(scalar<index>, scalar<index>) memoryOnly>> owned_in_mem)
@export
def test_owned(var x: RegPassable, var y: MemOnly) abi("Mojo"):
    # CHECK: [[LEN:%.*]] = kgen.param.constant = <16>
    # CHECK: kgen.call {{.*}}borrowed_generic{{.*}}"(%arg0)
    borrowed_generic(x)

    # CHECK: kgen.call {{.*}}borrowed_generic{{.*}}"(%arg1)
    borrowed_generic(y)

    # COM: check callsite to inlined RegPassable copy ctor
    # NOTE: The copy is optimized away since it is trivial.
    # CHECK: kgen.call {{.*}}owned_generic{{.*}}"(%arg0)
    owned_generic(x)

    # CHECK: [[XPTR3:%.*]] = pop.stack_allocation 1 x struct<(scalar<index>, scalar<index>) memoryOnly>
    # CHECK: pop.memcpy [[XPTR3]], %arg1, [[LEN]] : !kgen.pointer<struct<(scalar<index>, scalar<index>) memoryOnly>> -> !kgen.pointer<struct<(scalar<index>, scalar<index>) memoryOnly>>
    # CHECK: kgen.call {{.*}}owned_generic{{.*}}"([[XPTR3]])
    owned_generic(y)

    # CHECK: kgen.call {{.*}}owned_generic{{.*}}"(%arg0)
    owned_generic(x^)
    # CHECK: kgen.call {{.*}}owned_generic{{.*}}"(%arg1)
    owned_generic(y^)


# CHECK: kgen.func export @test_borrowed(
# CHECK-SAME: %arg0: !kgen.struct<(scalar<f32>, scalar<f32>)>,
# CHECK-SAME: %arg1: !kgen.pointer<struct<(scalar<index>, scalar<index>) memoryOnly>> imm_mem)
@export
def test_borrowed(x: RegPassable, y: MemOnly) abi("Mojo"):
    # CHECK: [[LEN:%.*]] = kgen.param.constant = <16>
    # CHECK: kgen.call {{.*}}borrowed_generic{{.*}}"(%arg0)
    borrowed_generic(x)

    # CHECK: kgen.call {{.*}}borrowed_generic{{.*}}"(%arg1)
    borrowed_generic(y)

    # COM: check callsite to inlined RegPassable copy ctor
    # CHECK: kgen.call {{.*}}owned_generic{{.*}}"(%arg0)
    owned_generic(x)

    # CHECK: [[XPTR3:%.*]] = pop.stack_allocation 1 x struct<(scalar<index>, scalar<index>) memoryOnly>
    # CHECK: pop.memcpy [[XPTR3]], %arg1, [[LEN]] : !kgen.pointer<struct<(scalar<index>, scalar<index>) memoryOnly>> -> !kgen.pointer<struct<(scalar<index>, scalar<index>) memoryOnly>>
    # CHECK: kgen.call {{.*}}owned_generic{{.*}}"([[XPTR3]])
    owned_generic(y)
