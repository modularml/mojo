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
from max.gpu.memory import CacheEviction, async_copy
from max.gpu.sync import async_copy_arrive, mbarrier_init, mbarrier_test_wait
from std.memory import unsafe_stack_allocation
from std.testing import assert_true


def test_mbarrier(
    addr0: MutPointer[Int8, MutAnyOrigin],
    addr1: MutPointer[UInt8, MutAnyOrigin],
    addr2: MutPointer[Float32, MutAnyOrigin, address_space=.GLOBAL],
    addr3: MutPointer[Float32, MutAnyOrigin, address_space=.SHARED],
    addr4: MutPointer[Float64, MutAnyOrigin, address_space=.GLOBAL],
    addr5: MutPointer[Float64, MutAnyOrigin, address_space=.SHARED],
):
    async_copy_arrive(addr0)
    async_copy_arrive(addr1)
    async_copy_arrive(addr2)
    async_copy_arrive(addr3)
    async_copy_arrive(addr4)
    async_copy_arrive(addr5)


def _verify_mbarrier(asm: StringSlice) raises -> None:
    assert_true("cp.async.mbarrier.arrive.b64" in asm)
    assert_true("cp.async.mbarrier.arrive.shared.b64" in asm)


def test_mbarrier_sm80() raises:
    print("test_mbarrier_sm80")
    var asm = _compile_code[test_mbarrier, target=get_gpu_target()]().asm
    _verify_mbarrier(asm)


def test_mbarrier_sm90() raises:
    print("test_mbarrier_sm90")
    var asm = _compile_code[
        test_mbarrier, target=get_gpu_target["sm_90"]()
    ]().asm
    _verify_mbarrier(asm)


def test_mbarrier_init(
    shared_mem: MutPointer[Int32, MutAnyOrigin, address_space=.SHARED],
):
    mbarrier_init(shared_mem, 4)


def _verify_mbarrier_init(asm: StringSlice) raises -> None:
    assert_true("ld.param::entry.b32" in asm)
    assert_true("mov.b32" in asm)
    assert_true("mbarrier.init.shared.b64" in asm)


def test_mbarrier_init_sm80() raises:
    print("test_mbarrier_init_sm80")
    var asm = _compile_code[test_mbarrier_init, target=get_gpu_target()]().asm

    _verify_mbarrier_init(asm)


def test_mbarrier_init_sm90() raises:
    print("test_mbarrier_init_sm90")
    var asm = _compile_code[
        test_mbarrier_init, target=get_gpu_target["sm_90"]()
    ]().asm
    _verify_mbarrier_init(asm)


def test_mbarrier_test_wait(
    shared_mem: MutPointer[Int32, MutAnyOrigin, address_space=.SHARED],
    state: Int,
):
    var done = False
    while not done:
        done = mbarrier_test_wait(shared_mem, state)


def _verify_mbarrier_test_wait(asm: StringSlice) raises -> None:
    assert_true("mbarrier.test_wait.shared.b64" in asm)


def test_mbarrier_test_wait_sm80() raises:
    print("test_mbarrier_test_wait_sm80")
    var asm = _compile_code[
        test_mbarrier_test_wait, target=get_gpu_target()
    ]().asm
    _verify_mbarrier_test_wait(asm)


def test_mbarrier_test_wait_sm90() raises:
    print("test_mbarrier_test_wait_sm90")
    var asm = _compile_code[
        test_mbarrier_test_wait, target=get_gpu_target["sm_90"]()
    ]().asm
    assert_true("mbarrier.test_wait.shared.b64" in asm)


def test_async_copy(
    src: ImmPointer[Float32, ImmutAnyOrigin, address_space=.GLOBAL]
):
    var shared_mem = unsafe_stack_allocation[
        4, DType.float32, address_space=.SHARED
    ]()
    async_copy[4](src, shared_mem)
    async_copy[16](src, shared_mem)


def _verify_async_copy(asm: StringSlice) raises -> None:
    assert_true("cp.async.ca.shared.global" in asm)
    assert_true("cp.async.cg.shared.global" in asm)


def test_async_copy_sm80() raises:
    print("test_async_copy_sm80")
    var asm = _compile_code[test_async_copy, target=get_gpu_target()]().asm
    _verify_async_copy(asm)


def test_async_copy_sm90() raises:
    print("test_async_copy_sm90")
    var asm = _compile_code[
        test_async_copy, target=get_gpu_target["sm_90"]()
    ]().asm
    _verify_async_copy(asm)


def test_async_copy_l2_prefetch(
    src: ImmPointer[Float32, ImmutAnyOrigin, address_space=.GLOBAL]
):
    var shared_mem = unsafe_stack_allocation[
        4, DType.float32, address_space=.SHARED
    ]()
    async_copy[4, bypass_L1_16B=False, l2_prefetch=128](src, shared_mem)
    async_copy[16, bypass_L1_16B=False, l2_prefetch=64](src, shared_mem)


def _verify_async_copy_l2_prefetch(asm: StringSlice) raises -> None:
    assert_true("cp.async.ca.shared.global.L2::128B" in asm)
    assert_true("cp.async.ca.shared.global.L2::64B" in asm)


def test_async_copy_l2_prefetch_sm80() raises:
    print("test_async_l2_prefetch_sm80")
    var asm = _compile_code[
        test_async_copy_l2_prefetch, target=get_gpu_target()
    ]().asm
    _verify_async_copy_l2_prefetch(asm)


def test_async_copy_l2_prefetch_sm90() raises:
    print("test_async_l2_prefetch_sm90")
    var asm = _compile_code[
        test_async_copy_l2_prefetch, target=get_gpu_target["sm_90"]()
    ]().asm
    _verify_async_copy_l2_prefetch(asm)


def test_async_copy_with_zero_fill_kernel(
    src: ImmPointer[Float32, ImmutAnyOrigin, address_space=.GLOBAL]
):
    var shared_mem = unsafe_stack_allocation[
        4, DType.float32, address_space=.SHARED
    ]()
    async_copy[4, bypass_L1_16B=False, l2_prefetch=128, fill=Float32(0)](
        src, shared_mem
    )
    async_copy[16, bypass_L1_16B=False, l2_prefetch=64, fill=Float32(0)](
        src, shared_mem
    )


def _verify_test_async_copy_with_zero_fill(asm: StringSlice) raises -> None:
    # Should contain something like:
    #     cp.async.ca.shared.global.L2::128B [%r1], [%rd2], 4, %r2;
    # Regex would be nice here, but alas...
    var cp128_begin = asm.find("cp.async.ca.shared.global.L2::128B")
    assert_true(cp128_begin >= 0)
    var cp128_end = asm.find(";", cp128_begin)
    assert_true(cp128_end >= 0)
    var cp128_str = asm[byte=cp128_begin:cp128_end]
    # Find various parts of the string we expect to be there
    var cp128_first_reg32_pos = cp128_str.find("[%r")
    assert_true(cp128_first_reg32_pos >= 0)
    var cp128_reg64_pos = cp128_str.find("[%rd")
    assert_true(cp128_reg64_pos >= 0)
    var cp128_bytes_pos = cp128_str.find(", 4,")
    assert_true(cp128_bytes_pos >= 0)
    var cp128_last_reg32_pos = cp128_str.rfind(", %r")
    assert_true(cp128_last_reg32_pos >= 0)
    # Assert they're in the right order
    assert_true(cp128_first_reg32_pos < cp128_reg64_pos)
    assert_true(cp128_reg64_pos < cp128_bytes_pos)
    assert_true(cp128_bytes_pos < cp128_last_reg32_pos)

    # Should contain something like:
    #     cp.async.ca.shared.global.L2::64B [%r1], [%rd1], 16, %r4;
    # Regex would be nice here, but alas...
    var cp64_begin = asm.find("cp.async.ca.shared.global.L2::64B")
    assert_true(cp64_begin >= 0)
    var cp64_end = asm.find(";", cp64_begin)
    assert_true(cp64_end >= 0)
    var cp64_str = asm[byte=cp64_begin:cp64_end]
    # Find various parts of the string we expect to be there
    var cp64_first_reg32_pos = cp64_str.find("[%r")
    assert_true(cp64_first_reg32_pos >= 0)
    var cp64_reg64_pos = cp64_str.find("[%rd")
    assert_true(cp64_reg64_pos >= 0)
    var cp64_bytes_pos = cp64_str.find(", 16,")
    assert_true(cp64_bytes_pos >= 0)
    var cp64_last_reg32_pos = cp64_str.rfind(", %r")
    assert_true(cp64_last_reg32_pos >= 0)
    # Assert they're in the right order
    assert_true(cp64_first_reg32_pos < cp64_reg64_pos)
    assert_true(cp64_reg64_pos < cp64_bytes_pos)
    assert_true(cp64_bytes_pos < cp64_last_reg32_pos)


def test_async_copy_with_zero_fill() raises:
    print("test_async_copy_zero_fill")
    var asm = _compile_code[
        test_async_copy_with_zero_fill_kernel, target=get_gpu_target()
    ]().asm
    _verify_test_async_copy_with_zero_fill(asm)


def test_async_copy_with_eviction(
    src: ImmPointer[Float32, ImmutAnyOrigin, address_space=.GLOBAL]
):
    print("test_async_copy_with_eviction")
    var shared_mem = unsafe_stack_allocation[
        4, DType.float32, address_space=.SHARED
    ]()
    async_copy[4, eviction_policy=CacheEviction.EVICT_FIRST](src, shared_mem)
    async_copy[16, eviction_policy=CacheEviction.EVICT_FIRST](src, shared_mem)
    async_copy[16, eviction_policy=CacheEviction.EVICT_LAST](src, shared_mem)


def async_copy_with_non_zero_fill_kernel(
    src: ImmPointer[Int32, ImmutAnyOrigin, address_space=.GLOBAL]
):
    var shared_mem = unsafe_stack_allocation[
        4, DType.int32, address_space=.SHARED
    ]()
    async_copy[16, bypass_L1_16B=False, l2_prefetch=128, fill=Int32(32)](
        src, shared_mem, predicate=True
    )
    async_copy[16, bypass_L1_16B=False, l2_prefetch=64, fill=Int32(32)](
        src, shared_mem, predicate=False
    )


def _verify_async_copy_with_non_zero_fill(asm: StringSlice) raises -> None:
    assert_true("mov.b32 	%r3, 32;" in asm)
    assert_true("@p cp.async.ca.shared.global.L2::128B" in asm and "16" in asm)
    assert_true("@p cp.async.ca.shared.global.L2::64B" in asm and "16" in asm)
    assert_true("@!p st.shared.v4.b32" in asm)


def test_async_copy_with_non_zero_fill() raises:
    print("test_async_copy_with_non_zero_fill")
    var asm = _compile_code[
        async_copy_with_non_zero_fill_kernel, target=get_gpu_target()
    ]().asm
    _verify_async_copy_with_non_zero_fill(asm)


def _verify_async_copy_with_eviction(asm: StringSlice) raises -> None:
    assert_true("createpolicy.fractional.L2::evict_first.b64" in asm)
    assert_true("createpolicy.fractional.L2::evict_last.b64" in asm)
    assert_true("cp.async.ca.shared.global" in asm)


def test_async_copy_with_eviction_sm80() raises:
    print("test_async_copy_with_eviction_sm80")
    var asm = _compile_code[
        test_async_copy_with_eviction, target=get_gpu_target["sm_80"]()
    ]().asm
    _verify_async_copy_with_eviction(asm)


def test_async_copy_with_eviction_sm90() raises:
    print("test_async_copy_with_eviction_sm90")
    var asm = _compile_code[
        test_async_copy_with_eviction, target=get_gpu_target["sm_90"]()
    ]().asm
    _verify_async_copy_with_eviction(asm)


def main() raises:
    test_mbarrier_sm80()
    test_mbarrier_sm90()
    test_mbarrier_init_sm80()
    test_mbarrier_init_sm90()
    test_mbarrier_test_wait_sm80()
    test_mbarrier_test_wait_sm90()
    test_async_copy_sm80()
    test_async_copy_sm90()
    test_async_copy_l2_prefetch_sm80()
    test_async_copy_l2_prefetch_sm90()
    test_async_copy_with_zero_fill()
    test_async_copy_with_non_zero_fill()
    test_async_copy_with_eviction_sm80()
    test_async_copy_with_eviction_sm90()
