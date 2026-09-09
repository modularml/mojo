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


# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo %s | kgen-opt --lower-semantic-cf --lower-lit | FileCheck %s


trait TensorStorage:
    comptime StorageType[
        mut: Bool,
        //,
        dtype: DType,
        origin: Origin[mut=mut],
        address_space: AddressSpace,
    ]: TrivialRegisterPassable


@fieldwise_init
struct Tile[
    mut: Bool,
    //,
    dtype: DType,
    origin: Origin[mut=mut],
    *,
    Storage: TensorStorage = PointerStorage,
](ImplicitlyCopyable, TrivialRegisterPassable):
    var _storage: Self.Storage.StorageType[Self.dtype, Self.origin, .GENERIC]


struct PointerStorage(TensorStorage):
    comptime StorageType[
        mut: Bool,
        //,
        dtype: DType,
        origin: Origin[mut=mut],
        address_space: AddressSpace,
    ]: TrivialRegisterPassable = Pointer[
        SIMD[dtype, 1], origin, address_space=address_space
    ]


trait Kernel(Deinitable, ImplicitlyCopyable):
    def run(self, c: Tile[mut=True, ...]):
        ...


@fieldwise_init
struct MyKernel(Kernel):
    def run(self, c: Tile[mut=True, ...]):
        pass


# CHECK-LABEL: kgen.generator @"moco_4146::go
def go[K: Kernel, o: MutOrigin](alg: K, c: Tile[.float32, o]):
    # Make sure that we fold the type `Tile[DType.float32, o]` used in the indirect call to `!kgen.pointer<none>` correctly.
    # CHECK: kgen.call_param tail[(!kgen.pointer<K> imm_mem, !kgen.pointer<none>)
    alg.run(c)


def main():
    var x = Float32(0)
    go(MyKernel(), Tile(Pointer(to=x)))
