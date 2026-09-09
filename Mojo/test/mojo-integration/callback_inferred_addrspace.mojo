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

# RUN: kgen -elaborate -O0 %s -S | FileCheck %s

# Regression: a `def(...)` callback whose argument is a deferred `memref` type
# with an inferred memory space leaves that parameter abstract in the callback's
# own scope until the callee is instantiated. The materialization
# "invalid MLIR type in deferred_type" error must not fire inside that nested
# parameter scope -- the memory space resolves to 1 once `apply` binds `cb`.


# Minimal memref-backed type whose memory space is the parametric
# `address_space`, so its stored type is a deferred MLIR type.
@fieldwise_init
struct MR[address_space: Int](TrivialRegisterPassable):
    var value: __mlir_deferred_type[
        `memref<4xf32, `, +Self.address_space.__mlir_index__(), `>`
    ]


# Polymorphic over the inferred `address_space`: the callback's `MR` memory
# space stays abstract in the `def(...)` signature.
def apply(x: MR, func: Some[def(MR)]):
    func(x)


# CHECK: kgen.func export @top(%{{.*}}: memref<4xf32, 1>)
@export
def top(output: MR[1]):
    def cb(t: MR):
        pass

    apply(output, cb)
