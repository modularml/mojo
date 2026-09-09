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

from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code
from layout.tma_async import SharedMemBarrier
from std.memory import unsafe_stack_allocation


# CHECK-LABEL: test_shared_mem_barrier
# CHECK-NOT: ld.local
# CHECK-NOT: st.local
def test_shared_mem_barrier():
    var mbar = unsafe_stack_allocation[
        10,
        SharedMemBarrier,
        address_space=.SHARED,
        alignment=8,
    ]()

    comptime for i in range(10):
        mbar[i].init()


def main() raises:
    print("== test_shared_mem_barrier")
    comptime kernel = test_shared_mem_barrier
    print(_compile_code[kernel, target=get_gpu_target["sm_90a"]()]())
