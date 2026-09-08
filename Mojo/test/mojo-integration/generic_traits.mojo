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

comptime Index = __mlir_type.index


trait SimpleTrait:
    @staticmethod
    def bar() -> Index:
        ...


struct MemType(SimpleTrait):
    @staticmethod
    @always_inline
    def bar() -> Index:
        return __mlir_attr.`1:index`


struct RegType(RegisterPassable, SimpleTrait):
    @staticmethod
    @always_inline
    def bar() -> Index:
        return __mlir_attr.`2:index`


struct RegTypeTrivial(SimpleTrait, TrivialRegisterPassable):
    var x: Index

    @staticmethod
    @always_inline
    def bar() -> Index:
        return __mlir_attr.`3:index`


def generic_arg[T: SimpleTrait](x: T) -> Index:
    return T.bar()


# CHECK: kgen.func @"{{.*}}generic_arg
# CHECK-SAME: %arg0: !kgen.pointer<struct<() memoryOnly>>
# CHECK-NEXT: <1>

# CHECK: kgen.func @"{{.*}}generic_arg
# CHECK-SAME: %arg0: !kgen.struct<()>
# CHECK-NEXT: <2>

# CHECK: kgen.func @"{{.*}}generic_arg
# CHECK-SAME: %arg0: index
# CHECK-NEXT: <3>


@export
def top(a: MemType, b: RegType, c: RegTypeTrivial) abi("Mojo"):
    _ = generic_arg(a)
    _ = generic_arg(b)
    _ = generic_arg(c)


trait GrandFather:
    def bar(self):
        ...


trait Father(GrandFather):
    def baz(self):
        ...


struct Son(Father):
    def bar(self):
        pass

    def baz(self):
        pass


# CHECK: kgen.func [[TAKE_GRAND_FATHER:@.*take_grand_father.*]](%arg0
def take_grand_father[T: GrandFather](value: T):
    # CHECK: kgen.call {{.*}}Son::bar
    value.bar()


# CHECK: kgen.func [[TAKE_FATHER:@.*take_father.*]](%arg0
def take_father[T: Father](value: T):
    # CHECK: kgen.call {{.*}}Son::baz
    value.baz()
    # CHECK: kgen.call tail [[TAKE_GRAND_FATHER]]
    take_grand_father(value)


# CHECK: kgen.func export @like_father_like
@export
def like_father_like(value: Son) abi("Mojo"):
    # CHECK: kgen.call tail [[TAKE_FATHER]]
    take_father(value)


struct SomeType(ImplicitlyCopyable, RegisterPassable):
    def __deinit__(deinit self):
        pass


# CHECK-LABEL: kgen.func {{.*}}drop_copy
def drop_copy[T: ImplicitlyCopyable & Deinitable](value: T):
    # CHECK: [[V0:%.*]] = kgen.param.constant: struct<()> = <{ }>
    # CHECK: kgen.call {{.*}}SomeType::__deinit__{{.*}}([[V0]])
    var _unused = value


@export
def copy_destroy(x: SomeType) abi("Mojo"):
    drop_copy(x)
