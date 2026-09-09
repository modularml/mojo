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

# RUN: kgen %s -elaborate -verify-diagnostics

# Counterpart to callback_inferred_addrspace.mojo: there the deferred type stays
# abstract (an escaping parameter reference) so the materialization error is
# suppressed. Here the parameter `n` is fully bound at materialization, leaving
# no escaping reference, but the resulting concrete type string is unparseable.
# The "invalid MLIR type in deferred_type" error must fire.


@no_inline
def use[
    n: Int
]() -> __mlir_deferred_type[`!llvm.array<`, +n.__mlir_index__(), ` bogus f32>`]:
    return __mlir_op.`llvm.mlir.undef`[
        _type=__mlir_deferred_type[
            `!llvm.array<`, +n.__mlir_index__(), ` bogus f32>`
        ]
    ]()


# expected-error @+1 {{function instantiation failed}}
def main():
    comptime sz: Int = 4
    # expected-note @+1 {{invalid MLIR type in deferred_type: !llvm.array<4 bogus f32>}}
    _ = use[sz]()
    print("ok")
