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


def __closure_wrapper_noop_dtor(self: __mlir_type.`!kgen.pointer<none>`, /):
    pass


def __closure_wrapper_noop_copy(
    *, copy: __mlir_type.`!kgen.pointer<none>`
) -> __mlir_type.`!kgen.pointer<none>`:
    return copy


def __ownership_keepalive[*Ts: AnyType](*args: *Ts):
    pass
