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

# RUN: %mojo %s | FileCheck %s


trait Store:
    comptime StorageType[dtype: DType]: TrivialRegisterPassable

    @staticmethod
    def add[
        LT: ImplicitlyCopyable & Deinitable, //, dtype: DType
    ](storage: Tuple[Self.StorageType[dtype], LT]):
        ...


struct PS(Store):
    comptime StorageType[dtype: DType]: TrivialRegisterPassable = SIMD[dtype, 1]

    @staticmethod
    def add[
        LT: ImplicitlyCopyable & Deinitable, //, dtype: DType
    ](storage: Tuple[Self.StorageType[dtype], LT]):
        # CHECK: worked
        print("worked")


# Dispatch `add` through the generic `S: Store` bound.
def go[
    S: Store, dt: DType, LT: ImplicitlyCopyable & Deinitable
](p: S.StorageType[dt], layout: LT):
    # Here, in order to match the type between trait and witness table, we need
    # to be able to fold `conforms_to(SIMD[*(0, 1)], xxx)` to true. (Note that
    # SIMD[*(0, 1)] is not yet concretized).
    S.add((p, layout))


def main():
    go[PS](Float32(1.0), 0)
