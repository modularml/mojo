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


def test_cannot_cast_between_different_named_origins[
    T: AnyType, mut: Bool, //, origin: Origin[mut=mut]
](p: Pointer[T, origin]):
    pass


def main() raises:
    var x = 42
    var y = 55

    var p = Pointer(to=x)
    # CHECK: value passed to 'p' cannot be converted from 'Pointer[Int, origin_of(x)]' to 'Pointer[T, origin_of(y)]'
    test_cannot_cast_between_different_named_origins[origin_of(y)](p)
