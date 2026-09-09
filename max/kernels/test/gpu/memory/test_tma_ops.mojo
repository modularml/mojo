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

from max.gpu.primitives.cluster import elect_one_sync
from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code
from max.gpu.memory import (
    CacheEviction,
    ReduceOp,
    cp_async_bulk_tensor_global_shared_cta,
    cp_async_bulk_tensor_reduce_global_shared_cta,
    cp_async_bulk_tensor_shared_cluster_global,
    fence_proxy_tensormap_generic_sys_acquire,
    fence_proxy_tensormap_generic_sys_release,
)

comptime OpaquePointer = ImmPointer[NoneType, ImmutAnyOrigin]
from std.utils.index import Index


# CHECK-LABEL: test_async_copy_asm
def test_async_copy_asm():
    print("== test_async_copy_asm")

    def test_async_copy_kernel(
        dst_mem: MutPointer[Float32, MutAnyOrigin, address_space=.SHARED],
        tma_descriptor: OpaquePointer,
        mem_bar: MutPointer[Float32, MutAnyOrigin, address_space=.SHARED],
        *coords: Int32,
    ):
        # CHECK: cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
        cp_async_bulk_tensor_shared_cluster_global(
            dst_mem, tma_descriptor, mem_bar, Index(coords[0], coords[1])
        )
        # CHECK: cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
        cp_async_bulk_tensor_shared_cluster_global(
            dst_mem, tma_descriptor, mem_bar, Index(coords[0])
        )

    print(
        _compile_code[
            test_async_copy_kernel,
            target=get_gpu_target["sm_90"](),
        ]()
    )


# CHECK-LABEL: test_async_store_asm
def test_async_store_asm():
    print("== test_async_store_asm")

    def test_async_store_kernel(
        src_mem: ImmPointer[Float32, ImmutAnyOrigin, address_space=.SHARED],
        tma_descriptor: OpaquePointer,
        *coords: Int32,
    ):
        # CHECK: cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%rd1, {%r2, %r3}], [%r1];
        cp_async_bulk_tensor_global_shared_cta(
            src_mem, tma_descriptor, Index(coords[0], coords[1])
        )
        # CHECK: cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint [%rd1, {%r4, %r5}], [%r1], %rd9;
        cp_async_bulk_tensor_global_shared_cta[
            eviction_policy=CacheEviction.EVICT_FIRST
        ](src_mem, tma_descriptor, Index(coords[0], coords[1]))
        # CHECK: cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group [%rd1, {%r6}], [%r1];
        cp_async_bulk_tensor_global_shared_cta(
            src_mem, tma_descriptor, Index(coords[0])
        )
        # CHECK: cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group.L2::cache_hint [%rd1, {%r7}], [%r1], %rd12;
        cp_async_bulk_tensor_global_shared_cta[
            eviction_policy=CacheEviction.EVICT_LAST
        ](src_mem, tma_descriptor, Index(coords[0]))

    print(
        _compile_code[
            test_async_store_kernel,
            target=get_gpu_target["sm_90"](),
        ]()
    )


# CHECK-LABEL: test_async_bulk_tensor_reduce_asm
def test_async_bulk_tensor_reduce_asm():
    print("== test_async_bulk_tensor_reduce_asm")

    def test_async_bulk_tensor_reduce_asm(
        src_mem: ImmPointer[Float32, ImmutAnyOrigin, address_space=.SHARED],
        tma_descriptor: OpaquePointer,
        *coords: Int32,
    ):
        cp_async_bulk_tensor_reduce_global_shared_cta[
            reduction_kind=ReduceOp.ADD
        ](src_mem, tma_descriptor, Index(coords[0], coords[1]))
        cp_async_bulk_tensor_reduce_global_shared_cta[
            reduction_kind=ReduceOp.ADD,
            eviction_policy=CacheEviction.EVICT_FIRST,
        ](src_mem, tma_descriptor, Index(coords[0], coords[1]))
        cp_async_bulk_tensor_reduce_global_shared_cta[
            reduction_kind=ReduceOp.ADD
        ](src_mem, tma_descriptor, Index(coords[0]))
        cp_async_bulk_tensor_reduce_global_shared_cta[
            reduction_kind=ReduceOp.ADD,
            eviction_policy=CacheEviction.EVICT_LAST,
        ](src_mem, tma_descriptor, Index(coords[0]))

    print(
        _compile_code[
            test_async_bulk_tensor_reduce_asm,
            target=get_gpu_target["sm_90"](),
        ]()
    )


# CHECK-LABEL: test_tma_fence_proxy
def test_tma_fence_proxy():
    print("== test_tma_fence_proxy")

    def test_tma_fence_proxy_kernel(
        descriptor_ptr: MutPointer[Int32, MutAnyOrigin]
    ):
        # CHECK: fence.proxy.tensormap::generic.acquire.sys [%rd1], 128;
        fence_proxy_tensormap_generic_sys_acquire(descriptor_ptr, 128)
        # CHECK: fence.proxy.tensormap::generic.release.sys;
        fence_proxy_tensormap_generic_sys_release()

    print(
        _compile_code[
            test_tma_fence_proxy_kernel,
            target=get_gpu_target["sm_90"](),
        ]()
    )


# CHECK-LABEL: test_elect_one_sync
def test_elect_one_sync():
    print("== test_elect_one_sync")

    def test_elect_one_sync_kernel():
        # CHECK: elect.sync      %r1|%p1, -1;
        var _lane_predicate: Bool = elect_one_sync()

    print(
        _compile_code[
            test_elect_one_sync_kernel,
            target=get_gpu_target["sm_90"](),
        ]()
    )


def main():
    test_async_copy_asm()
    test_async_store_asm()
    test_async_bulk_tensor_reduce_asm()
    test_tma_fence_proxy()
    test_elect_one_sync()
