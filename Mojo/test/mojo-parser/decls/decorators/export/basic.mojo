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


@export("my_named_export", ABI="C")
# CHECK: lit.fn export @"export_me()"{{.*}}cabi
# CHECK-SAME: linkageName = #kgen.linkage_name<"my_named_export" : !kgen.string, false>
def export_me() -> None:
    ...


@export
# CHECK: lit.fn export @"not_c_exported()"
def not_c_exported() abi("Mojo"):
    pass


struct Thing(Movable where False):
    # CHECK: lit.fn export @"member
    @export
    def member(self) abi("Mojo"):
        pass
