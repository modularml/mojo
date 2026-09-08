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

comptime AnyCoroutine = __mlir_type.`!co.routine`


struct Coroutine[T: AnyType, origins: __mlir_type.`!lit.origin.set`](
    Deinitable where False, RegisterPassable
):
    var value: __mlir_type.`!co.routine`

    @implicit
    def __init__(out self, handle: AnyCoroutine):
        self.value = handle

    def __await__(deinit self) -> Self.T:
        while True:
            pass


struct RaisingCoroutine[T: AnyType, origins: __mlir_type.`!lit.origin.set`](
    Deinitable where False, RegisterPassable
):
    var value: __mlir_type.`!co.routine`

    @implicit
    def __init__(out self, handle: AnyCoroutine):
        self.value = handle

    def __await__(deinit self) raises -> Self.T:
        while True:
            pass
