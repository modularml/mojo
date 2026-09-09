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

from debug_test_utils import keep_alive
from std.memory import dealloc

# The invalid dtype, built directly from the underlying MLIR attribute so this
# debugger test does not depend on any public `DType` alias. This exercises the
# type system's rendering of `scalar<invalid>` (see PrimitiveTypesTest.cpp).
comptime _invalid_dtype = DType(
    mlir_value=__mlir_attr.`#kgen.dtype.constant<invalid> : !kgen.dtype`
)


def main():
    var base_alloc = alloc[Float32]({count = 1})
    var base: Pointer[
        Float32, origin_of(base_alloc._alloc)
    ] = base_alloc.unsafe_ptr()
    var ptr = base.unsafe_bitcast[Scalar[_invalid_dtype]]()
    keep_alive(ptr)  # breakpoint
    dealloc(base_alloc^)
