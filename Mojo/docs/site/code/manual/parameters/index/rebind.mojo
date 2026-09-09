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


def take_simd8(x: SIMD[DType.float32, 8]):
    pass


def generic_simd[nelts: SIMDLength](x: SIMD[DType.float32, nelts]):
    comptime if nelts == 8:
        take_simd8(rebind[SIMD[DType.float32, 8]](x))


def main():
    generic_simd(SIMD[DType.float32, 8](1))
