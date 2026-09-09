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

# RUN: %mojo %s --debug-level full 2>&1 | FileCheck %s


@no_inline
def _trait_is_eq[t1: type_of(AnyType), t2: type_of(AnyType)]() -> Bool:
    return __mlir_attr[
        `#kgen.param.identical<`,
        `#kgen.type<`,
        +t1,
        `> : !kgen.type`,
        `,`,
        `#kgen.type<`,
        +t2,
        `> : !kgen.type`,
        `> : !kgen.scalar<bool>`,
    ]


def main():
    # CHECK: True
    print(_trait_is_eq[AnyType, AnyType]())
    # CHECK-NEXT: False
    print(_trait_is_eq[Copyable, AnyType]())
