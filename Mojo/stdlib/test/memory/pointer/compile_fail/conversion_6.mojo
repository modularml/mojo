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

# RUN: not %mojo %s 2>&1 | FileCheck %s


def test_cannot_cast_from_immutable_any_to_named[
    T: AnyType, mut: Bool, //, origin: Origin[mut=mut]
](p: Pointer[T, origin]):
    pass


def main() raises:
    var x = 42

    var p = Pointer(to=x).as_imm().as_unsafe_any_origin()
    # CHECK: value passed to 'p' cannot be converted
    test_cannot_cast_from_immutable_any_to_named[ImmOrigin(origin_of(x))](p)
