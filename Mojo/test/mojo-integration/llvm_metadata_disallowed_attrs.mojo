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

# `target_cpu`, `tune_cpu`, and `target_features` are all rejected by
# convertLLVMMetadata; one representative case is checked here. Lowering
# aborts at the first failure, so the other two names cannot be exercised
# in the same file.


# CHECK: 'llvm.target_cpu' is not allowed to be set via @__llvm_metadata
@export
@__llvm_metadata(`llvm.target_cpu`=__mlir_attr.`"znver4"`)
def fn_disallowed_target_cpu() abi("Mojo"):
    pass


@export
def use() abi("Mojo"):
    fn_disallowed_target_cpu()
