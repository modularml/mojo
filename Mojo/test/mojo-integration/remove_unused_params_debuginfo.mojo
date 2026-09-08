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
# RUN: kgen -emit=llvm --debug-level line-tables %s | FileCheck %s --check-prefix=CHECK-LT
# RUN: kgen -emit=llvm --debug-level full %s | FileCheck %s --check-prefix=CHECK-FULL


# CHECK-LT: define {{.*}}agnostic_user{{.*}} !dbg ![[SP:[0-9]+]]
# CHECK-FULL: define {{.*}}agnostic_user{{.*}} !dbg ![[SP:[0-9]+]]
@no_inline
def agnostic_user[
    T: AnyType, dt: DType
](b: Pointer[T, _], dp: Pointer[Scalar[dt], _]):
    print(b.unsafe_bitcast[UInt32]())
    print(dp.unsafe_bitcast[UInt32]())


# In line-tables mode, type stripping makes all specializations identical,
# allowing RemoveUnusedParams to merge them into a single shared function.
# CHECK-LT-NOT: define {{.*}}agnostic_user


def main():
    var x: Int = 8
    var y: Float64 = 42.5
    var d: UInt8 = 9
    agnostic_user[Int, DType.uint8](
        Pointer(to=x),
        Pointer(to=d),
    )
    agnostic_user[Float64, DType.uint8](
        Pointer(to=y),
        Pointer(to=d),
    )


# In line-tables mode, subroutine arg/result types are stripped to reduce
# debug info size. The types list should contain only null (the return type
# placeholder); no argument type entries should appear.
# CHECK-LT-DAG: ![[SP]] = distinct !DISubprogram({{.*}}name:{{.*}}agnostic_user{{.*}}, type: ![[SP_TYPE:[0-9]+]],
# CHECK-LT-DAG: ![[SP_TYPE]] = !DISubroutineType({{.*}}types: ![[SP_MEMBER_TYPES:[0-9]+]]
# CHECK-LT-DAG: ![[SP_MEMBER_TYPES]] = !{null}

# In full mode, types are not stripped; the subprogram retains its argument
# types. Unlike line-tables mode (which produces a single merged function with
# an empty types list), full mode keeps one specialization per distinct type
# signature, each with its concrete argument types intact.
# CHECK-FULL-DAG: ![[SP]] = distinct !DISubprogram({{.*}}name:{{.*}}agnostic_user{{.*}}, type: ![[SP_TYPE:[0-9]+]],
# CHECK-FULL-DAG: ![[SP_TYPE]] = !DISubroutineType({{.*}}types: ![[SP_MEMBER_TYPES:[0-9]+]]
# CHECK-FULL-DAG: ![[SP_MEMBER_TYPES]] = !{null, !{{[0-9]+}}, !{{[0-9]+}}}
