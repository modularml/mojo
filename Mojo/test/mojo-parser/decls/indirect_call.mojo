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
# RUN: %parse-mojo-isolated %s | FileCheck %s


def return_function[T: AnyType]() -> T:
    pass


# CHECK-LABEL: lit.fn @"mrvalue_indirect_callee
def mrvalue_indirect_callee():
    # CHECK-NEXT: [[RESULT:%.*]] = lit.var.decl
    # CHECK-NEXT: call {{.*}}return_function{{.*}}([[RESULT]])
    # CHECK-NEXT: [[CALLEE:%.*]] = lit.load.consume [[RESULT]]
    # CHECK-NEXT: lit.call_indirect tail [[CALLEE]]()
    return_function[def() thin -> None]()()


def indirect_callee() raises -> def() thin -> None:
    pass


# CHECK-LABEL: lit.fn @"call_it
def call_it() raises:
    # CHECK-NEXT: [[RESULT:%.*]] = lit.var.decl
    # CHECK-NEXT: call {{.*}}indirect_callee{{.*}}(%__error__, [[RESULT]])
    # CHECK-NEXT: [[CALLEE:%.*]] = lit.load.consume [[RESULT]]
    # CHECK-NEXT: lit.call_indirect tail [[CALLEE]]()
    indirect_callee()()
