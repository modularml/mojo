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


trait SelfMethod:
    def foo(self):
        pass


struct SelfStruct(SelfMethod):
    def foo(self):
        pass


# CHECK-LABEL: kgen.func @"{{.*}}call_it{{.*}}T=
def call_it[T: SelfMethod](x: T):
    # CHECK: call {{.*}}SelfStruct::foo{{.*}}(%arg0)
    x.foo()


@export
def pass_it(x: SelfStruct) abi("Mojo"):
    call_it(x)
