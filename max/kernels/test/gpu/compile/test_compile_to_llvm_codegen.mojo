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

from max.gpu import thread_idx
from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code
from max.gpu.memory import external_memory


# CHECK-LABEL: test_array_offset
def test_array_offset():
    print("== test_array_offset")

    def kernel(
        output: MutPointer[Float32, MutAnyOrigin],
        p: ImmPointer[Float32, ImmutAnyOrigin, address_space=.SHARED],
        idx: Int,
    ):
        output[] = p[idx]

    # CHECK: getelementptr inbounds float, ptr addrspace(3) %1, i{{[0-9]+}} %{{.*}}
    print(_compile_code[kernel, emission_kind="llvm"]())


# CHECK-LABEL: test_case_thread_id_nvidia
def test_case_thread_id_nvidia():
    print("== test_case_thread_id_nvidia")

    def kernel(output: MutPointer[Int32, MutAnyOrigin]):
        output[] = Int32(thread_idx.x + thread_idx.x + thread_idx.x)

    # CHECK-COUNT-1: call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
    print(
        _compile_code[
            kernel, emission_kind="llvm", target=get_gpu_target["sm_80"]()
        ]()
    )


# CHECK-LABEL: test_case_thread_id_mi355x
def test_case_thread_id_mi355x():
    print("== test_case_thread_id_mi355x")

    def kernel(output: MutPointer[Int32, MutAnyOrigin]):
        output[] = Int32(thread_idx.x + thread_idx.x + thread_idx.x)

    # CHECK-COUNT-1: call i32 @llvm.amdgcn.workitem.id.x()
    print(
        _compile_code[
            kernel, emission_kind="llvm", target=get_gpu_target["mi355x"]()
        ]()
    )


# CHECK-LABEL: test_dynamic_shared_mem
def test_dynamic_shared_mem():
    print("== test_dynamic_shared_mem")

    # CHECK: @extern_ptr_syml = external dso_local addrspace(3) global [0 x float], align 4
    # CHECK: @extern_ptr_syml_0 = external dso_local addrspace(3) global [0 x float], align 4
    def kernel(output: MutPointer[Float32, MutAnyOrigin]):
        # CHECK: %2 = load float, ptr addrspace(3) @extern_ptr_syml, align 4
        # CHECK: %3 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @extern_ptr_syml_0, i{{[0-9]+}}  4), align 4
        # CHECK: fadd contract float %2, %3
        var dynamic_sram_ptr_1 = external_memory[
            Float32, address_space=.SHARED, alignment=4
        ]()
        var dynamic_sram_ptr_2 = external_memory[
            Float32, address_space=.SHARED, alignment=4
        ]()
        output[] = dynamic_sram_ptr_1[0] + dynamic_sram_ptr_2[1]

    print(_compile_code[kernel, emission_kind="llvm"]())


def main():
    test_array_offset()
    test_case_thread_id_nvidia()
    test_case_thread_id_mi355x()
    test_dynamic_shared_mem()
