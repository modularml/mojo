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
# RUN: not kgen -emit=llvm %s 2>&1 | FileCheck %s


# CHECK: invalid 'llvm.frame_pointer' value 'bogus'
# CHECK-SAME: expected "none", "non-leaf", "all", or "reserved"
@export
@__llvm_metadata(`llvm.frame_pointer`=__mlir_attr.`"bogus"`)
def fn_bad_frame_pointer() abi("Mojo"):
    pass


@export
def use() abi("Mojo"):
    fn_bad_frame_pointer()
