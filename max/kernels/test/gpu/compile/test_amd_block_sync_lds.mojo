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
# RUN: %mojo-no-debug %s | FileCheck %s
# RUN: %mojo-no-debug -D USE_EXPERIMENTAL_AMD_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM=True %s | FileCheck --check-prefix=CHECK-BLOCK-SYNC %s

from max.gpu.sync import barrier
from max.gpu.host.compile import _compile_code
from max.gpu.host.info import MI355X

comptime MI355X_TARGET = MI355X.target()


# CHECK-LABEL: test_barrier
def test_barrier() raises:
    print("== test_barrier")

    def barrier_kernel():
        barrier()

    # CHECK: fence syncscope("workgroup") release
    # CHECK: tail call void @llvm.amdgcn.s.barrier()
    # CHECK: fence syncscope("workgroup") acquire

    # CHECK-BLOCK-SYNC: tail call void @llvm.amdgcn.s.waitcnt(i32 49279)
    # CHECK-BLOCK-SYNC: tail call void @llvm.amdgcn.s.barrier()
    print(
        _compile_code[
            barrier_kernel, target=MI355X_TARGET, emission_kind="llvm-opt"
        ]()
    )


def main() raises:
    test_barrier()
