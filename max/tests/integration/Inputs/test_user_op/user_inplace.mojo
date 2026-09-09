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

import extensibility
from extensibility import (
    _MutableInputTensor as MutableInputTensor,
)


@extensibility.register("mutable_test_op")
struct MutableTestOp:
    @staticmethod
    def execute(in_place_tensor: MutableInputTensor) raises:
        var x = in_place_tensor._ptr.unsafe_load(0)
        x += 1
        in_place_tensor._ptr.unsafe_store(0, x)


@extensibility.register("foo")
struct FooKernel:
    @staticmethod
    def execute(in_place_tensor: MutableInputTensor) raises:
        in_place_tensor._ptr.unsafe_store(0, 0)


@extensibility.register("bar")
struct BarKernel:
    @staticmethod
    def execute(in_place_tensor: MutableInputTensor) raises:
        in_place_tensor._ptr.unsafe_store(0, 0)


@extensibility.register("baz")
struct BazKernel:
    @staticmethod
    def execute(in_place_tensor: MutableInputTensor) raises:
        in_place_tensor._ptr.unsafe_store(0, 0)
