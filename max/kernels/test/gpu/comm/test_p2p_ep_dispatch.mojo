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

from std.random import randint, randn, seed
from std.sys import has_nvidia_gpu_accelerator, has_amd_gpu_accelerator

from max.algorithm import sync_parallelize
from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchmarkInfo,
    BenchId,
    BenchMetric,
    Report,
    ThroughputMeasure,
)
from comm.sync import enable_p2p
from max.gpu.host import DeviceBuffer, DeviceContext
from layout import TileTensor, Idx, row_major
from std.math import ceildiv
from shmem.ep import (
    ep_dispatch_async_kernel_api,
    ep_dispatch_wait_kernel_api,
)
from shmem.ep_comm import (
    BF16TokenFormat,
    BlockwiseFP8TokenFormat,
    EP_DATA_READY_FLAG,
    EPLocalSyncCounters,
    MXTokenFormat,
    NVBlockScaledTokenFormat,
    TokenFormat,
)
from std.testing import assert_almost_equal, assert_equal

from linalg.fp4_utils import (
    E2M1_TO_FLOAT32,
    MXFP4_SF_VECTOR_SIZE,
    NVFP4_SF_VECTOR_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    SF_MN_GROUP_SIZE,
    get_scale_factor,
)
from max.gpu.host.info import _is_sm10x_gpu, MI355X


def legalize_topk_ids[
    n_experts: Int, top_k: Int
](topk_ids: MutPointer[Int32, MutUntrackedOrigin], n_tokens: Int):
    for tok_id in range(n_tokens):
        var topk_ids_for_token = topk_ids + tok_id * top_k

        # The top-k ids for a token should be unique. If not, we will assign a
        # random id to the duplicate id.
        def is_duplicate() {imm} -> Int:
            for i in range(top_k):
                for j in range(i + 1, top_k):
                    if topk_ids_for_token[i] == topk_ids_for_token[j]:
                        return i
            return -1

        var duplicate_idx = is_duplicate()
        while duplicate_idx != -1:
            randint(topk_ids_for_token + duplicate_idx, 1, 0, n_experts - 1)
            duplicate_idx = is_duplicate()


trait DispatchTestT(Deinitable):
    """Trait to unify the test dispatch logic for different token formats."""

    comptime hidden_size: Int
    comptime top_k: Int
    comptime n_experts: Int
    comptime n_ranks: Int
    comptime n_slots: Int
    comptime n_tokens_per_rank: Int
    comptime TokenFormatType: TokenFormat

    def __init__(out self, list_of_ctx: List[DeviceContext]) raises:
        ...

    @always_inline
    def get_token_handler(
        self,
        dev_idx: Int,
        slot_idx: Int,
        ctx: DeviceContext,
        out result: Self.TokenFormatType,
    ):
        ...

    @always_inline
    def save_outputs_to_host(
        self, list_of_ctx: List[DeviceContext]
    ) raises -> None:
        ...

    @always_inline
    def check_output_val(
        self,
        dev_idx: Int,
        slot_idx: Int,
        expert_idx: Int,
        expert_token_idx: Int,
        token_idx: Int,
        hid_dim_idx: Int,
        expected_val: BFloat16,
    ) raises -> None:
        ...


struct BF16DispatchTest[
    _hidden_size: Int,
    _top_k: Int,
    _n_experts: Int,
    _n_ranks: Int,
    _n_slots: Int,
    _n_tokens_per_rank: Int,
](DispatchTestT):
    comptime hidden_size = Self._hidden_size
    comptime top_k = Self._top_k
    comptime n_experts = Self._n_experts
    comptime n_ranks = Self._n_ranks
    comptime n_slots = Self._n_slots
    comptime n_tokens_per_rank = Self._n_tokens_per_rank
    comptime max_recv_num_tokens = min(
        Self.n_experts, Self.n_ranks * Self.top_k
    ) * Self.n_tokens_per_rank

    comptime output_layout = row_major(
        (Self.max_recv_num_tokens, Idx[Self.hidden_size])
    )
    comptime TokenFormatType = BF16TokenFormat[
        output_layout=type_of(Self.output_layout), Self.hidden_size, Self.top_k
    ]

    var device_output_bufs_list: List[DeviceBuffer[.bfloat16]]
    var host_output_bufs_list: List[MutPointer[BFloat16, MutUntrackedOrigin]]

    def __init__(out self, list_of_ctx: List[DeviceContext]) raises:
        self.device_output_bufs_list = List[DeviceBuffer[.bfloat16]](
            capacity=Self.n_ranks
        )
        self.host_output_bufs_list = List[
            MutPointer[BFloat16, MutUntrackedOrigin]
        ](capacity=Self.n_ranks)
        for i in range(Self.n_ranks):
            self.device_output_bufs_list.append(
                list_of_ctx[i].enqueue_create_buffer[.bfloat16](
                    Self.n_slots * Self.max_recv_num_tokens * Self.hidden_size
                )
            )
            self.host_output_bufs_list.append(
                alloc[BFloat16](
                    Self.n_slots * Self.max_recv_num_tokens * Self.hidden_size
                )
            )

    def __deinit__(deinit self):
        for i in range(Self.n_ranks):
            self.host_output_bufs_list[i].free()

    @always_inline
    def get_token_handler(
        self,
        dev_idx: Int,
        slot_idx: Int,
        ctx: DeviceContext,
        out result: Self.TokenFormatType,
    ):
        var output_tensor = TileTensor(
            ptr=self.device_output_bufs_list[dev_idx].unsafe_ptr()
            + slot_idx * Self.max_recv_num_tokens * Self.hidden_size,
            layout=Self.output_layout,
        )
        result = Self.TokenFormatType(output_tensor)

    @always_inline
    def save_outputs_to_host(
        self, list_of_ctx: List[DeviceContext]
    ) raises -> None:
        for i in range(Self.n_ranks):
            list_of_ctx[i].enqueue_copy(
                self.host_output_bufs_list[i], self.device_output_bufs_list[i]
            )
            list_of_ctx[i].synchronize()

    @always_inline
    def check_output_val(
        self,
        dev_idx: Int,
        slot_idx: Int,
        expert_idx: Int,
        expert_token_idx: Int,
        token_idx: Int,
        hid_dim_idx: Int,
        expected_val: BFloat16,
    ) raises -> None:
        var device_val = self.host_output_bufs_list[dev_idx][
            slot_idx * Self.max_recv_num_tokens * Self.hidden_size
            + token_idx * Self.hidden_size
            + hid_dim_idx
        ]

        assert_equal(
            device_val,
            expected_val,
            "Output value mismatch for dev "
            + String(dev_idx)
            + " slot "
            + String(slot_idx)
            + " token "
            + String(token_idx)
            + " hid_dim "
            + String(hid_dim_idx),
        )


struct BlockwiseFP8DispatchTest[
    fp8_dtype: DType,
    scales_dtype: DType,
    _hidden_size: Int,
    _top_k: Int,
    _n_experts: Int,
    _n_ranks: Int,
    _n_slots: Int,
    _n_tokens_per_rank: Int,
](DispatchTestT):
    comptime hidden_size = Self._hidden_size
    comptime top_k = Self._top_k
    comptime n_experts = Self._n_experts
    comptime n_ranks = Self._n_ranks
    comptime n_slots = Self._n_slots
    comptime n_tokens_per_rank = Self._n_tokens_per_rank
    comptime group_size = 128
    comptime max_recv_num_tokens = min(
        Self.n_experts, Self.n_ranks * Self.top_k
    ) * Self.n_tokens_per_rank

    comptime output_layout = row_major(
        (Self.max_recv_num_tokens, Idx[Self.hidden_size])
    )
    comptime output_scales_layout = row_major(
        (
            Self.hidden_size // Self.group_size,
            Idx[Self.max_recv_num_tokens],
        )
    )
    comptime TokenFormatType = BlockwiseFP8TokenFormat[
        fp8_dtype=Self.fp8_dtype,
        scales_dtype=Self.scales_dtype,
        output_layout=type_of(Self.output_layout),
        scales_layout=type_of(Self.output_scales_layout),
        Self.hidden_size,
        Self.top_k,
    ]

    var device_output_bufs_list: List[DeviceBuffer[Self.fp8_dtype]]
    var device_output_scales_bufs_list: List[DeviceBuffer[Self.scales_dtype]]
    var host_output_bufs_list: List[
        MutPointer[Scalar[Self.fp8_dtype], MutUntrackedOrigin]
    ]
    var host_output_scales_bufs_list: List[
        MutPointer[Scalar[Self.scales_dtype], MutUntrackedOrigin]
    ]

    def __init__(out self, list_of_ctx: List[DeviceContext]) raises:
        self.device_output_bufs_list = List[DeviceBuffer[Self.fp8_dtype]](
            capacity=Self.n_ranks
        )
        self.device_output_scales_bufs_list = List[
            DeviceBuffer[Self.scales_dtype]
        ](capacity=Self.n_ranks)
        self.host_output_bufs_list = List[
            MutPointer[Scalar[Self.fp8_dtype], MutUntrackedOrigin]
        ](capacity=Self.n_ranks)
        self.host_output_scales_bufs_list = List[
            MutPointer[Scalar[Self.scales_dtype], MutUntrackedOrigin]
        ](capacity=Self.n_ranks)
        for i in range(Self.n_ranks):
            self.device_output_bufs_list.append(
                list_of_ctx[i].enqueue_create_buffer[Self.fp8_dtype](
                    Self.n_slots * Self.max_recv_num_tokens * Self.hidden_size
                )
            )
            self.device_output_scales_bufs_list.append(
                list_of_ctx[i].enqueue_create_buffer[Self.scales_dtype](
                    Self.n_slots
                    * Self.max_recv_num_tokens
                    * Self.hidden_size
                    // Self.group_size
                )
            )
            self.host_output_bufs_list.append(
                alloc[Scalar[Self.fp8_dtype]](
                    Self.n_slots * Self.max_recv_num_tokens * Self.hidden_size
                )
            )
            self.host_output_scales_bufs_list.append(
                alloc[Scalar[Self.scales_dtype]](
                    Self.n_slots
                    * Self.max_recv_num_tokens
                    * Self.hidden_size
                    // Self.group_size
                )
            )

    def __deinit__(deinit self):
        for i in range(Self.n_ranks):
            self.host_output_bufs_list[i].free()
            self.host_output_scales_bufs_list[i].free()

    @always_inline
    def get_token_handler(
        self,
        dev_idx: Int,
        slot_idx: Int,
        ctx: DeviceContext,
        out result: Self.TokenFormatType,
    ):
        var output_tensor = TileTensor(
            ptr=self.device_output_bufs_list[dev_idx].unsafe_ptr()
            + slot_idx * Self.max_recv_num_tokens * Self.hidden_size,
            layout=Self.output_layout,
        )
        var output_scales_tensor = TileTensor(
            ptr=self.device_output_scales_bufs_list[dev_idx].unsafe_ptr()
            + slot_idx
            * Self.max_recv_num_tokens
            * Self.hidden_size
            // Self.group_size,
            layout=Self.output_scales_layout,
        )
        result = Self.TokenFormatType(output_tensor, output_scales_tensor)

    @always_inline
    def save_outputs_to_host(
        self, list_of_ctx: List[DeviceContext]
    ) raises -> None:
        for i in range(Self.n_ranks):
            list_of_ctx[i].enqueue_copy(
                self.host_output_bufs_list[i], self.device_output_bufs_list[i]
            )
            list_of_ctx[i].enqueue_copy(
                self.host_output_scales_bufs_list[i],
                self.device_output_scales_bufs_list[i],
            )
            list_of_ctx[i].synchronize()

    @always_inline
    def check_output_val(
        self,
        dev_idx: Int,
        slot_idx: Int,
        expert_idx: Int,
        expert_token_idx: Int,
        token_idx: Int,
        hid_dim_idx: Int,
        expected_val: BFloat16,
    ) raises -> None:
        var output_offset = (
            slot_idx * Self.max_recv_num_tokens * Self.hidden_size
            + token_idx * Self.hidden_size
            + hid_dim_idx
        )
        var scale_offset = (
            slot_idx
            * Self.max_recv_num_tokens
            * Self.hidden_size
            // Self.group_size
            + (hid_dim_idx // Self.group_size) * Self.max_recv_num_tokens
            + token_idx
        )
        var fp8_val = self.host_output_bufs_list[dev_idx][output_offset]
        var token_scale = self.host_output_scales_bufs_list[dev_idx][
            scale_offset
        ]
        var token_val = fp8_val.cast[Self.scales_dtype]() * token_scale

        assert_almost_equal(
            expected_val,
            token_val.cast[.bfloat16](),
            "Output value mismatch for dev "
            + String(dev_idx)
            + " slot "
            + String(slot_idx)
            + " token "
            + String(token_idx)
            + " hid_dim "
            + String(hid_dim_idx),
            rtol=1e-1,
            atol=1e-1,
        )


struct NVFP4DispatchTest[
    fp4_dtype: DType,
    scales_dtype: DType,
    _hidden_size: Int,
    _top_k: Int,
    _n_experts: Int,
    _n_ranks: Int,
    _n_slots: Int,
    _n_tokens_per_rank: Int,
](DispatchTestT):
    comptime hidden_size = Self._hidden_size
    comptime top_k = Self._top_k
    comptime n_experts = Self._n_experts
    comptime n_ranks = Self._n_ranks
    comptime n_slots = Self._n_slots
    comptime n_tokens_per_rank = Self._n_tokens_per_rank
    comptime max_recv_num_tokens = min(
        Self.n_experts, Self.n_ranks * Self.top_k
    ) * Self.n_tokens_per_rank
    comptime n_local_experts = Self.n_experts // Self.n_ranks

    comptime scales_padded_size = Self.max_recv_num_tokens + Self.n_local_experts * SF_MN_GROUP_SIZE

    comptime uint8_last_dim = Self.hidden_size // 2

    comptime output_layout = row_major(
        (Self.max_recv_num_tokens, Idx[Self.uint8_last_dim])
    )
    comptime output_scales_layout = row_major(
        (
            Self.scales_padded_size // SF_MN_GROUP_SIZE,
            Idx[ceildiv(Self.hidden_size, SF_ATOM_K * NVFP4_SF_VECTOR_SIZE)],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )
    comptime output_scales_offset_layout = row_major[
        Self.n_experts // Self.n_ranks
    ]()
    comptime TokenFormatType = NVBlockScaledTokenFormat[
        quant_dtype=Self.fp4_dtype,
        scales_dtype=Self.scales_dtype,
        output_layout=type_of(Self.output_layout),
        scales_offset_layout=type_of(Self.output_scales_offset_layout),
        Self.hidden_size,
        Self.top_k,
    ]

    var device_output_bufs_list: List[DeviceBuffer[Self.fp4_dtype]]
    var device_output_scales_bufs_list: List[DeviceBuffer[Self.scales_dtype]]
    var device_output_scales_offset_bufs_list: List[DeviceBuffer[.uint32]]
    var host_output_bufs_list: List[
        MutPointer[Scalar[Self.fp4_dtype], MutUntrackedOrigin]
    ]
    var host_output_scales_bufs_list: List[
        MutPointer[Scalar[Self.scales_dtype], MutUntrackedOrigin]
    ]
    var host_output_scales_offset_bufs_list: List[
        MutPointer[UInt32, MutUntrackedOrigin]
    ]

    def __init__(out self, list_of_ctx: List[DeviceContext]) raises:
        self.device_output_bufs_list = List[DeviceBuffer[Self.fp4_dtype]](
            capacity=Self.n_ranks
        )
        self.device_output_scales_bufs_list = List[
            DeviceBuffer[Self.scales_dtype]
        ](capacity=Self.n_ranks)
        self.device_output_scales_offset_bufs_list = List[
            DeviceBuffer[.uint32]
        ](capacity=Self.n_ranks)
        self.host_output_bufs_list = List[
            MutPointer[Scalar[Self.fp4_dtype], MutUntrackedOrigin]
        ](capacity=Self.n_ranks)
        self.host_output_scales_bufs_list = List[
            MutPointer[Scalar[Self.scales_dtype], MutUntrackedOrigin]
        ](capacity=Self.n_ranks)
        self.host_output_scales_offset_bufs_list = List[
            MutPointer[UInt32, MutUntrackedOrigin]
        ](capacity=Self.n_ranks)
        for i in range(Self.n_ranks):
            self.device_output_bufs_list.append(
                list_of_ctx[i].enqueue_create_buffer[Self.fp4_dtype](
                    Self.n_slots
                    * Self.max_recv_num_tokens
                    * Self.uint8_last_dim
                )
            )
            self.device_output_scales_bufs_list.append(
                list_of_ctx[i].enqueue_create_buffer[Self.scales_dtype](
                    Self.n_slots
                    * Self.scales_padded_size
                    * Self.hidden_size
                    // NVFP4_SF_VECTOR_SIZE
                )
            )
            self.device_output_scales_offset_bufs_list.append(
                list_of_ctx[i].enqueue_create_buffer[.uint32](
                    Self.n_slots * (Self.n_experts // Self.n_ranks)
                )
            )
            self.host_output_bufs_list.append(
                alloc[Scalar[Self.fp4_dtype]](
                    Self.n_slots
                    * Self.max_recv_num_tokens
                    * Self.uint8_last_dim
                )
            )
            self.host_output_scales_bufs_list.append(
                alloc[Scalar[Self.scales_dtype]](
                    Self.n_slots
                    * Self.scales_padded_size
                    * Self.hidden_size
                    // NVFP4_SF_VECTOR_SIZE
                )
            )
            self.host_output_scales_offset_bufs_list.append(
                alloc[UInt32](Self.n_slots * (Self.n_experts // Self.n_ranks))
            )

    def __deinit__(deinit self):
        for i in range(Self.n_ranks):
            self.host_output_bufs_list[i].free()
            self.host_output_scales_bufs_list[i].free()
            self.host_output_scales_offset_bufs_list[i].free()

    @always_inline
    def get_token_handler(
        self,
        dev_idx: Int,
        slot_idx: Int,
        ctx: DeviceContext,
        out result: Self.TokenFormatType,
    ):
        var output_tensor = TileTensor(
            ptr=self.device_output_bufs_list[dev_idx].unsafe_ptr()
            + slot_idx * Self.max_recv_num_tokens * Self.uint8_last_dim,
            layout=Self.output_layout,
        )
        var output_scales_tensor = TileTensor(
            ptr=self.device_output_scales_bufs_list[dev_idx].unsafe_ptr()
            + slot_idx
            * Self.scales_padded_size
            * Self.hidden_size
            // NVFP4_SF_VECTOR_SIZE,
            layout=Self.output_scales_layout,
        )
        var output_scales_offset_tensor = TileTensor(
            ptr=self.device_output_scales_offset_bufs_list[dev_idx].unsafe_ptr()
            + slot_idx * (Self.n_experts // Self.n_ranks),
            layout=Self.output_scales_offset_layout,
        )

        result = Self.TokenFormatType(
            output_tensor,
            output_scales_tensor,
            output_scales_offset_tensor,
            ctx,
        )

    @always_inline
    def save_outputs_to_host(
        self, list_of_ctx: List[DeviceContext]
    ) raises -> None:
        for i in range(Self.n_ranks):
            list_of_ctx[i].enqueue_copy(
                self.host_output_bufs_list[i], self.device_output_bufs_list[i]
            )
            list_of_ctx[i].enqueue_copy(
                self.host_output_scales_bufs_list[i],
                self.device_output_scales_bufs_list[i],
            )
            list_of_ctx[i].enqueue_copy(
                self.host_output_scales_offset_bufs_list[i],
                self.device_output_scales_offset_bufs_list[i],
            )
            list_of_ctx[i].synchronize()

    @always_inline
    def check_output_val(
        self,
        dev_idx: Int,
        slot_idx: Int,
        expert_idx: Int,
        expert_token_idx: Int,
        token_idx: Int,
        hid_dim_idx: Int,
        expected_val: BFloat16,
    ) raises -> None:
        var output_offset = (
            slot_idx * Self.max_recv_num_tokens * Self.uint8_last_dim
            + token_idx * Self.uint8_last_dim
            + (hid_dim_idx // 2)
        )

        var host_scales_tensor = TileTensor(
            ptr=self.host_output_scales_bufs_list[dev_idx]
            + slot_idx
            * Self.scales_padded_size
            * Self.hidden_size
            // NVFP4_SF_VECTOR_SIZE,
            layout=Self.output_scales_layout,
        )

        var expert_start_index = token_idx - expert_token_idx
        var scales_block_id = (
            UInt32(expert_start_index // SF_MN_GROUP_SIZE)
            + self.host_output_scales_offset_bufs_list[dev_idx][
                slot_idx * (Self.n_experts // Self.n_ranks) + expert_idx
            ]
        )
        var _scales_tensor = TileTensor(
            ptr=host_scales_tensor.ptr_at_offset(
                (scales_block_id, Idx[0], Idx[0], Idx[0], Idx[0])
            ),
            layout=Self.output_scales_layout,
        )

        var uint8_val = self.host_output_bufs_list[dev_idx][output_offset]

        var token_scale = get_scale_factor[SF_VECTOR_SIZE=NVFP4_SF_VECTOR_SIZE](
            _scales_tensor, expert_token_idx, hid_dim_idx
        )
        var token_val = (
            E2M1_TO_FLOAT32[
                Int(
                    (
                        uint8_val
                        >> Scalar[Self.fp4_dtype](((hid_dim_idx % 2) * 4))
                    )
                    & 0x0F
                )
            ]
            * token_scale.cast[.float32]()
        )

        assert_almost_equal(
            expected_val,
            token_val.cast[.bfloat16](),
            "Scaled by :"
            + String(token_scale)
            + "Output value mismatch for dev "
            + String(dev_idx)
            + " slot "
            + String(slot_idx)
            + " token "
            + String(token_idx)
            + " hid_dim "
            + String(hid_dim_idx),
            rtol=2.5e-1,
            atol=2.5e-1,
        )


struct MXFP4DispatchTest[
    fp4_dtype: DType,
    scales_dtype: DType,
    _hidden_size: Int,
    _top_k: Int,
    _n_experts: Int,
    _n_ranks: Int,
    _n_slots: Int,
    _n_tokens_per_rank: Int,
](DispatchTestT):
    comptime hidden_size = Self._hidden_size
    comptime top_k = Self._top_k
    comptime n_experts = Self._n_experts
    comptime n_ranks = Self._n_ranks
    comptime n_slots = Self._n_slots
    comptime n_tokens_per_rank = Self._n_tokens_per_rank
    comptime max_recv_num_tokens = min(
        Self.n_experts, Self.n_ranks * Self.top_k
    ) * Self.n_tokens_per_rank
    comptime n_local_experts = Self.n_experts // Self.n_ranks

    comptime scales_padded_size = Self.max_recv_num_tokens + Self.n_local_experts * SF_MN_GROUP_SIZE

    comptime uint8_last_dim = Self.hidden_size // 2

    comptime output_layout = row_major(
        (Self.max_recv_num_tokens, Idx[Self.uint8_last_dim])
    )
    comptime output_scales_layout = row_major(
        (
            Idx[Self.max_recv_num_tokens],
            Self.hidden_size // MXFP4_SF_VECTOR_SIZE,
        )
    )
    comptime TokenFormatType = MXTokenFormat[
        quant_dtype=Self.fp4_dtype,
        scales_dtype=Self.scales_dtype,
        output_layout=type_of(Self.output_layout),
        scales_layout=type_of(Self.output_scales_layout),
        Self.hidden_size,
        Self.top_k,
    ]

    var device_output_bufs_list: List[DeviceBuffer[Self.fp4_dtype]]
    var device_output_scales_bufs_list: List[DeviceBuffer[Self.scales_dtype]]
    var host_output_bufs_list: List[
        MutPointer[Scalar[Self.fp4_dtype], MutUntrackedOrigin]
    ]
    var host_output_scales_bufs_list: List[
        MutPointer[Scalar[Self.scales_dtype], MutUntrackedOrigin]
    ]

    def __init__(out self, list_of_ctx: List[DeviceContext]) raises:
        self.device_output_bufs_list = List[DeviceBuffer[Self.fp4_dtype]](
            capacity=Self.n_ranks
        )
        self.device_output_scales_bufs_list = List[
            DeviceBuffer[Self.scales_dtype]
        ](capacity=Self.n_ranks)
        self.host_output_bufs_list = List[
            MutPointer[Scalar[Self.fp4_dtype], MutUntrackedOrigin]
        ](capacity=Self.n_ranks)
        self.host_output_scales_bufs_list = List[
            MutPointer[Scalar[Self.scales_dtype], MutUntrackedOrigin]
        ](capacity=Self.n_ranks)
        for i in range(Self.n_ranks):
            self.device_output_bufs_list.append(
                list_of_ctx[i].enqueue_create_buffer[Self.fp4_dtype](
                    Self.n_slots
                    * Self.max_recv_num_tokens
                    * Self.uint8_last_dim
                )
            )
            self.device_output_scales_bufs_list.append(
                list_of_ctx[i].enqueue_create_buffer[Self.scales_dtype](
                    Self.n_slots
                    * Self.scales_padded_size
                    * Self.hidden_size
                    // MXFP4_SF_VECTOR_SIZE
                )
            )
            self.host_output_bufs_list.append(
                alloc[Scalar[Self.fp4_dtype]](
                    Self.n_slots
                    * Self.max_recv_num_tokens
                    * Self.uint8_last_dim
                )
            )
            self.host_output_scales_bufs_list.append(
                alloc[Scalar[Self.scales_dtype]](
                    Self.n_slots
                    * Self.scales_padded_size
                    * Self.hidden_size
                    // MXFP4_SF_VECTOR_SIZE
                )
            )

    def __deinit__(deinit self):
        for i in range(Self.n_ranks):
            self.host_output_bufs_list[i].free()
            self.host_output_scales_bufs_list[i].free()

    @always_inline
    def get_token_handler(
        self,
        dev_idx: Int,
        slot_idx: Int,
        ctx: DeviceContext,
        out result: Self.TokenFormatType,
    ):
        var output_tensor = TileTensor(
            ptr=self.device_output_bufs_list[dev_idx].unsafe_ptr()
            + slot_idx * Self.max_recv_num_tokens * Self.uint8_last_dim,
            layout=Self.output_layout,
        )
        var output_scales_tensor = TileTensor(
            ptr=self.device_output_scales_bufs_list[dev_idx].unsafe_ptr()
            + slot_idx
            * Self.scales_padded_size
            * Self.hidden_size
            // MXFP4_SF_VECTOR_SIZE,
            layout=Self.output_scales_layout,
        )

        result = Self.TokenFormatType(output_tensor, output_scales_tensor)

    @always_inline
    def save_outputs_to_host(
        self, list_of_ctx: List[DeviceContext]
    ) raises -> None:
        for i in range(Self.n_ranks):
            list_of_ctx[i].enqueue_copy(
                self.host_output_bufs_list[i], self.device_output_bufs_list[i]
            )
            list_of_ctx[i].enqueue_copy(
                self.host_output_scales_bufs_list[i],
                self.device_output_scales_bufs_list[i],
            )
            list_of_ctx[i].synchronize()

    @always_inline
    def check_output_val(
        self,
        dev_idx: Int,
        slot_idx: Int,
        expert_idx: Int,
        expert_token_idx: Int,
        token_idx: Int,
        hid_dim_idx: Int,
        expected_val: BFloat16,
    ) raises -> None:
        var output_offset = (
            slot_idx * Self.max_recv_num_tokens * Self.uint8_last_dim
            + token_idx * Self.uint8_last_dim
            + (hid_dim_idx // 2)
        )

        var uint8_val = self.host_output_bufs_list[dev_idx][output_offset]

        var scale_offset = (
            slot_idx
            * Self.max_recv_num_tokens
            * Self.hidden_size
            // MXFP4_SF_VECTOR_SIZE
            + token_idx * (Self.hidden_size // MXFP4_SF_VECTOR_SIZE)
            + (hid_dim_idx // MXFP4_SF_VECTOR_SIZE)
        )
        var token_scale = self.host_output_scales_bufs_list[dev_idx][
            scale_offset
        ]

        var token_val = (
            E2M1_TO_FLOAT32[
                Int(
                    (
                        uint8_val
                        >> Scalar[Self.fp4_dtype](((hid_dim_idx % 2) * 4))
                    )
                    & 0x0F
                )
            ]
            * token_scale.cast[.float32]()
        )

        assert_almost_equal(
            expected_val,
            token_val.cast[.bfloat16](),
            "Scaled by :"
            + String(token_scale)
            + "Output value mismatch for dev "
            + String(dev_idx)
            + " slot "
            + String(slot_idx)
            + " token "
            + String(token_idx)
            + " hid_dim "
            + String(hid_dim_idx),
            rtol=2.5e-1,
            atol=2.5e-1,
        )


def test_dispatch_common[
    DispatchTestType: DispatchTestT,
    bench_e2e: Bool = False,
](list_of_ctx: List[DeviceContext]) raises:
    comptime input_type = DType.bfloat16
    comptime hidden_size = DispatchTestType.hidden_size
    comptime top_k = DispatchTestType.top_k
    comptime n_experts = DispatchTestType.n_experts
    comptime n_ranks = DispatchTestType.n_ranks
    comptime n_slots = DispatchTestType.n_slots
    comptime n_tokens_per_rank = DispatchTestType.n_tokens_per_rank
    comptime token_fmt_type = DispatchTestType.TokenFormatType

    comptime msg_bytes = token_fmt_type.msg_size()
    comptime n_local_experts = n_experts // n_ranks
    comptime max_recv_num_tokens = n_experts * n_tokens_per_rank

    comptime num_bytes = msg_bytes * top_k * n_tokens_per_rank

    var dispatch_test = DispatchTestType(list_of_ctx)

    print(
        "Running ep_dispatch test:",
        token_fmt_type.get_type_name(),
        "hidden_size:",
        hidden_size,
        "top_k:",
        top_k,
        "n_experts:",
        n_experts,
        "n_ranks:",
        n_ranks,
        "n_tokens_per_rank:",
        n_tokens_per_rank,
    )

    # fmt: off
    var send_bufs_list = List[DeviceBuffer[.uint8]](capacity=n_ranks)
    var recv_bufs_list = List[DeviceBuffer[.uint8]](capacity=n_ranks)
    var recv_count_bufs_list = List[DeviceBuffer[.uint64]](capacity=n_ranks)
    var atomic_counters_list = List[DeviceBuffer[.int32]](capacity=n_ranks)

    var host_topk_ids_list = Array[MutPointer[Int32, MutUntrackedOrigin], n_ranks](uninitialized=True)
    var host_input_tokens_list = Array[MutPointer[Scalar[input_type], MutUntrackedOrigin], n_ranks](uninitialized=True)

    var device_topk_bufs_list = List[DeviceBuffer[.int32]](capacity=n_ranks)
    var device_input_bufs_list = List[DeviceBuffer[input_type]](capacity=n_ranks)
    var device_row_offsets_bufs_list = List[DeviceBuffer[.uint32]](capacity=n_ranks)
    var device_expert_ids_bufs_list = List[DeviceBuffer[.int32]](capacity=n_ranks)
    var device_src_token_info_bufs_list = List[DeviceBuffer[.int32]](capacity=n_ranks)


    for i in range(n_ranks):
        var ctx = list_of_ctx[i]
        send_bufs_list.append(list_of_ctx[i].enqueue_create_buffer[.uint8](n_slots * n_tokens_per_rank * msg_bytes))
        recv_bufs_list.append(ctx.enqueue_create_buffer[.uint8](n_slots * max_recv_num_tokens * msg_bytes))
        recv_count_bufs_list.append(ctx.enqueue_create_buffer[.uint64](n_slots * n_experts))
        atomic_counters_list.append(ctx.enqueue_create_buffer[.int32](
            n_slots * EPLocalSyncCounters[n_experts].total_size()
        ))
        ctx.enqueue_memset(atomic_counters_list[i], Int32(0))
        ctx.enqueue_memset(recv_count_bufs_list[i], UInt64.MAX_FINITE)

        host_topk_ids_list[i] = alloc[Int32](n_slots * n_tokens_per_rank * top_k)
        host_input_tokens_list[i] = alloc[Scalar[input_type]](n_slots * n_tokens_per_rank * hidden_size)

        device_topk_bufs_list.append(ctx.enqueue_create_buffer[.int32](n_slots * n_tokens_per_rank * top_k))
        device_input_bufs_list.append(ctx.enqueue_create_buffer[input_type](n_slots * n_tokens_per_rank * hidden_size))
        device_row_offsets_bufs_list.append(ctx.enqueue_create_buffer[.uint32](n_slots * (n_local_experts + 1)))
        device_expert_ids_bufs_list.append(ctx.enqueue_create_buffer[.int32](n_slots * n_local_experts))
        device_src_token_info_bufs_list.append(ctx.enqueue_create_buffer[.int32](n_slots * max_recv_num_tokens * 2))
    # fmt: on

    var topk_ids_layout = row_major(n_tokens_per_rank, Idx[top_k])
    var input_tokens_layout = row_major((n_tokens_per_rank, Idx[hidden_size]))
    var row_offsets_layout = row_major[n_local_experts + 1]()
    var expert_ids_layout = row_major[n_local_experts]()
    var src_token_info_layout = row_major((Idx[max_recv_num_tokens], Idx[2]))
    var ptrs_layout = row_major[n_ranks]()
    comptime counters_size = EPLocalSyncCounters[n_experts].total_size()
    var counters_layout = row_major[counters_size]()

    # Initialize the inputs
    for dev_idx in range(n_ranks):
        var ctx = list_of_ctx[dev_idx]
        # Initialize the topk ids and input tokens using fixed seed,
        seed(dev_idx)
        randint(
            host_topk_ids_list[dev_idx],
            n_slots * n_tokens_per_rank * top_k,
            0,
            n_experts - 1,
        )

        # The topk ids for a token is the expert id it needs to be sent to.
        # Since a token won't be sent to the same expert multiple times, we
        # need to legalize the topk ids to make sure they are unique for
        # each token.
        legalize_topk_ids[n_experts, top_k](
            host_topk_ids_list[dev_idx], n_slots * n_tokens_per_rank
        )

        randn(
            host_input_tokens_list[dev_idx],
            n_slots * n_tokens_per_rank * hidden_size,
        )

        ctx.enqueue_copy(
            device_topk_bufs_list[dev_idx], host_topk_ids_list[dev_idx]
        )
        ctx.enqueue_copy(
            device_input_bufs_list[dev_idx], host_input_tokens_list[dev_idx]
        )

    # fmt: off
    var send_ptrs_inputs = alloc[UInt64](n_slots * n_ranks)
    var recv_ptrs_inputs = alloc[UInt64](n_slots * n_ranks)
    var recv_count_ptrs_inputs = alloc[UInt64](n_slots * n_ranks)

    for slot_idx in range(n_slots):
        for dev_idx in range(n_ranks):
            var ptr_idx = slot_idx * n_ranks + dev_idx
            send_ptrs_inputs[ptr_idx] = UInt64(
                Int(send_bufs_list[dev_idx].unsafe_ptr() + slot_idx * n_tokens_per_rank * msg_bytes)
            )
            recv_ptrs_inputs[ptr_idx] = UInt64(
                Int(recv_bufs_list[dev_idx].unsafe_ptr() + slot_idx * max_recv_num_tokens * msg_bytes)
            )
            recv_count_ptrs_inputs[ptr_idx] = UInt64(
                Int(recv_count_bufs_list[dev_idx].unsafe_ptr() + slot_idx * n_experts)
            )

    @always_inline
    @__parameter
    def get_send_ptrs_tensor(slot_idx: Int, out result: TileTensor[.uint64, type_of(ptrs_layout), ImmutAnyOrigin]) raises:
        return type_of(result)(ptr=(send_ptrs_inputs + slot_idx * n_ranks).as_unsafe_any_origin(), layout=ptrs_layout)

    @always_inline
    @__parameter
    def get_recv_ptrs_tensor(slot_idx: Int, out result: TileTensor[.uint64, type_of(ptrs_layout), ImmutAnyOrigin]) raises:
        return type_of(result)( ptr=(recv_ptrs_inputs + slot_idx * n_ranks).as_unsafe_any_origin(), layout=ptrs_layout)

    @always_inline
    @__parameter
    def get_recv_count_ptrs_tensor(slot_idx: Int, out result: TileTensor[.uint64, type_of(ptrs_layout), ImmutAnyOrigin]) raises:
        return type_of(result)(ptr=(recv_count_ptrs_inputs + slot_idx * n_ranks).as_unsafe_any_origin(), layout=ptrs_layout)

    @always_inline
    @__parameter
    def get_atomic_counters_tensor( dev_idx: Int, slot_idx: Int, out result: TileTensor[.int32, type_of(counters_layout), MutAnyOrigin]) raises:
        return type_of(result)(
            ptr=(atomic_counters_list[dev_idx].unsafe_ptr() + slot_idx * counters_size).as_unsafe_any_origin(), layout=counters_layout
        )

    @always_inline
    @__parameter
    def get_topk_ids_tensor(dev_idx: Int, slot_idx: Int, out result: TileTensor[.int32, type_of(topk_ids_layout), ImmutAnyOrigin]) raises:
        return type_of(result)(ptr=(device_topk_bufs_list[dev_idx].unsafe_ptr() + slot_idx * n_tokens_per_rank * top_k).as_unsafe_any_origin(), layout=topk_ids_layout)

    @always_inline
    @__parameter
    def get_input_tokens_tensor(dev_idx: Int, slot_idx: Int, out result: TileTensor[input_type, type_of(input_tokens_layout), ImmutAnyOrigin]) raises:
        return type_of(result)(ptr=(device_input_bufs_list[dev_idx].unsafe_ptr() + slot_idx * n_tokens_per_rank * hidden_size).as_unsafe_any_origin(), layout=input_tokens_layout)

    @always_inline
    @__parameter
    def get_row_offsets_tensor(dev_idx: Int, slot_idx: Int, out result: TileTensor[.uint32, type_of(row_offsets_layout), MutAnyOrigin]) raises:
        return type_of(result)(ptr=(device_row_offsets_bufs_list[dev_idx].unsafe_ptr() + slot_idx * (n_local_experts + 1)).as_unsafe_any_origin(), layout=row_offsets_layout)

    @always_inline
    @__parameter
    def get_expert_ids_tensor(dev_idx: Int, slot_idx: Int, out result: TileTensor[.int32, type_of(expert_ids_layout), MutAnyOrigin]) raises:
        return type_of(result)(ptr=(device_expert_ids_bufs_list[dev_idx].unsafe_ptr() + slot_idx * n_local_experts).as_unsafe_any_origin(), layout=expert_ids_layout)

    @always_inline
    @__parameter
    def get_src_token_info_tensor(dev_idx: Int, slot_idx: Int, out result: TileTensor[.int32, type_of(src_token_info_layout), MutAnyOrigin]) raises:
        return type_of(result)(ptr=(device_src_token_info_bufs_list[dev_idx].unsafe_ptr() + slot_idx * max_recv_num_tokens * 2).as_unsafe_any_origin(), layout=src_token_info_layout)
    # fmt: on

    @always_inline
    @__parameter
    def run_dispatch_async(dev_idx: Int, slot_idx: Int) raises:
        var ctx = list_of_ctx[dev_idx]
        ep_dispatch_async_kernel_api[
            token_fmt_type,
            n_experts,
            n_tokens_per_rank,
            n_ranks,
            1,
            "gpu",
            use_shmem=False,
        ](
            get_atomic_counters_tensor(dev_idx, slot_idx),
            get_input_tokens_tensor(dev_idx, slot_idx),
            get_topk_ids_tensor(dev_idx, slot_idx),
            get_send_ptrs_tensor(slot_idx),
            get_recv_ptrs_tensor(slot_idx),
            get_recv_count_ptrs_tensor(slot_idx),
            ctx,
        )

    @always_inline
    @__parameter
    def run_dispatch_async_wait(dev_idx: Int, slot_idx: Int) raises:
        var ctx = list_of_ctx[dev_idx]
        var format_handler = dispatch_test.get_token_handler(
            dev_idx, slot_idx, ctx
        )
        ep_dispatch_wait_kernel_api[
            n_experts,
            n_tokens_per_rank,
            n_ranks,
            1,
            "gpu",
        ](
            format_handler,
            get_row_offsets_tensor(dev_idx, slot_idx),
            get_expert_ids_tensor(dev_idx, slot_idx),
            get_src_token_info_tensor(dev_idx, slot_idx),
            get_recv_ptrs_tensor(slot_idx),
            get_recv_count_ptrs_tensor(slot_idx),
            get_atomic_counters_tensor(dev_idx, slot_idx),
            ctx,
        )

    @always_inline
    @__parameter
    def run_e2e(dev_idx: Int, slot_idx: Int) raises:
        run_dispatch_async(dev_idx, slot_idx)
        run_dispatch_async_wait(dev_idx, slot_idx)

    @always_inline
    @__parameter
    def clean_up(dev_idx: Int) raises:
        var ctx = list_of_ctx[dev_idx]
        ctx.enqueue_memset(atomic_counters_list[dev_idx], Int32(0))
        ctx.enqueue_memset(recv_count_bufs_list[dev_idx], UInt64.MAX_FINITE)

    # warm up by running once
    for dev_i in range(n_ranks):
        run_e2e(dev_i, 0)

    for dev_i in range(n_ranks):
        clean_up(dev_i)
        list_of_ctx[dev_i].synchronize()

    # Necessary to fill this Array w/ default BenchmarkInfo
    # otherwise each thread attempts to free uninitialized BenchmarkInfo
    # when copying below
    var default_info = BenchmarkInfo(
        name="",
        result=Report(),
        measures=List[ThroughputMeasure](),
    )
    var results_b = Array[BenchmarkInfo, n_ranks](fill=default_info)

    # First, bench the dispatch kernel overhead

    @always_inline
    def call_fn_dispatch(ctx: DeviceContext, cache_iter: Int) raises {}:
        var dev_id = Int(ctx.id())
        run_dispatch_async(dev_id, cache_iter)

    def per_gpu_dispatch(i: Int) raises {mut results_b, imm}:
        @always_inline
        def bench_iter(mut b: Bencher) raises {imm}:
            bencher_iter_custom(b, call_fn_dispatch, list_of_ctx[i])

        var bench_config = BenchConfig()
        bench_config.show_progress = False
        var b = Bench(bench_config^)
        b.bench_function(
            bench_iter,
            BenchId("bench dispatch"),
            [ThroughputMeasure(BenchMetric.bytes, 0)],
            fixed_iterations=n_slots,
        )
        results_b[i] = b.info_vec[0].copy()

    sync_parallelize(per_gpu_dispatch, n_ranks)

    var max_time = 0.0
    var max_loc = 0

    for i in range(n_ranks):
        var val = results_b[i].result.mean(unit="ms")
        if val > max_time:
            max_time = val
            max_loc = i

    var b_final = Bench()
    b_final.info_vec.append(results_b[max_loc].copy())
    b_final.dump_report()

    # Then, bench the dispatch_wait kernel overhead
    for dev_i in range(n_ranks):
        list_of_ctx[dev_i].synchronize()

    @always_inline
    def call_fn_dispatch_wait(ctx: DeviceContext, cache_iter: Int) raises {}:
        var dev_id = Int(ctx.id())
        run_dispatch_async_wait(dev_id, cache_iter)

    def per_gpu_dispatch_wait(i: Int) raises {mut results_b, imm}:
        @always_inline
        def bench_iter(mut b: Bencher) raises {imm}:
            bencher_iter_custom(b, call_fn_dispatch_wait, list_of_ctx[i])

        var bench_config = BenchConfig()
        bench_config.show_progress = False
        var b = Bench(bench_config^)
        b.bench_function(
            bench_iter,
            BenchId("bench dispatch_wait"),
            [ThroughputMeasure(BenchMetric.bytes, 0)],
            fixed_iterations=n_slots,
        )
        results_b[i] = b.info_vec[0].copy()

    sync_parallelize(per_gpu_dispatch_wait, n_ranks)

    max_time = 0.0
    max_loc = 0

    for i in range(n_ranks):
        var val = results_b[i].result.mean(unit="ms")
        if val > max_time:
            max_time = val
            max_loc = i

    b_final = Bench()
    b_final.info_vec.append(results_b[max_loc].copy())
    b_final.dump_report()

    # We don't enable e2e benchmarking by default because it would hang
    # if AsyncRT has less than n_ranks worker threads.
    comptime if bench_e2e:
        for dev_i in range(n_ranks):
            clean_up(dev_i)
            list_of_ctx[dev_i].synchronize()

        @always_inline
        def call_fn_e2e(ctx: DeviceContext, cache_iter: Int) raises {}:
            var dev_id = Int(ctx.id())
            run_dispatch_async(dev_id, cache_iter + 1)
            run_dispatch_async_wait(dev_id, cache_iter + 1)

        def per_gpu_e2e(i: Int) raises {mut results_b, imm}:
            @always_inline
            def bench_iter(mut b: Bencher) raises {imm}:
                bencher_iter_custom(b, call_fn_e2e, list_of_ctx[i])

            run_e2e(i, 0)
            list_of_ctx[i].synchronize()

            var bench_config = BenchConfig()
            bench_config.show_progress = False
            var b = Bench(bench_config^)
            b.bench_function(
                bench_iter,
                BenchId("bench dispatch e2e"),
                [ThroughputMeasure(BenchMetric.bytes, num_bytes)],
                fixed_iterations=n_slots - 1,
            )
            results_b[i] = b.info_vec[0].copy()

        sync_parallelize(per_gpu_e2e, n_ranks)

        max_time = 0.0
        max_loc = 0

        for i in range(n_ranks):
            var val = results_b[i].result.mean(unit="ms")
            if val > max_time:
                max_time = val
                max_loc = i

        b_final = Bench()
        b_final.info_vec.append(results_b[max_loc].copy())
        b_final.dump_report()

    # Verify the results for each device and each slot
    print("Verifying results...")

    @always_inline
    def verify_results(dev_idx: Int) raises {imm}:
        var ctx = list_of_ctx[dev_idx]

        # Allocate host buffers for copying device outputs
        var host_output = alloc[Scalar[input_type]](
            n_slots * max_recv_num_tokens * hidden_size
        )
        var host_row_offsets = alloc[UInt32](n_slots * (n_local_experts + 1))
        var host_expert_ids = alloc[Int32](n_slots * n_local_experts)
        var host_src_token_info = alloc[Int32](
            n_slots * max_recv_num_tokens * 2
        )
        var host_atomic_counter = alloc[Int32](
            n_slots * EPLocalSyncCounters[n_experts].total_size()
        )

        # Copy device outputs to host
        ctx.enqueue_copy(
            host_row_offsets, device_row_offsets_bufs_list[dev_idx]
        )
        ctx.enqueue_copy(host_expert_ids, device_expert_ids_bufs_list[dev_idx])
        ctx.enqueue_copy(
            host_src_token_info, device_src_token_info_bufs_list[dev_idx]
        )
        ctx.enqueue_copy(host_atomic_counter, atomic_counters_list[dev_idx])
        ctx.synchronize()

        # Check results for each slot
        for slot_idx in range(n_slots):
            # Get pointers to this slot's data
            var slot_expert_ids = host_expert_ids + slot_idx * n_local_experts
            var slot_src_token_info = (
                host_src_token_info + slot_idx * max_recv_num_tokens * 2
            )
            var slot_dispatch_wait_counter = EPLocalSyncCounters[n_experts](
                host_atomic_counter
                + slot_idx * EPLocalSyncCounters[n_experts].total_size()
            ).get_dispatch_wait_ptr()

            # Check if we have received the correct number of tokens
            var expert_start_idx = n_local_experts * dev_idx
            var expert_end_idx = expert_start_idx + n_local_experts
            var expected_tokens = 0
            var received_tokens = 0

            # Count expected tokens from all ranks for this slot
            for rank in range(n_ranks):
                var rank_topk_ids = (
                    host_topk_ids_list[rank]
                    + slot_idx * n_tokens_per_rank * top_k
                )
                for tok_idx in range(n_tokens_per_rank * top_k):
                    if (
                        expert_start_idx
                        <= Int(rank_topk_ids[tok_idx])
                        < expert_end_idx
                    ):
                        expected_tokens += 1

            # Check the output tokens
            for expert_idx in range(n_local_experts):
                var curr_local_expert = slot_expert_ids[expert_idx]
                var curr_expert = (
                    Int32(n_local_experts * dev_idx) + curr_local_expert
                )
                var expert_token_idx = 0

                for remote_rank in range(n_ranks):
                    var expert_rank_offset = curr_local_expert * Int32(
                        n_ranks
                    ) + Int32(remote_rank)
                    var token_end = (
                        slot_dispatch_wait_counter[2 * expert_rank_offset]
                        - EP_DATA_READY_FLAG
                    )
                    var num_tokens = slot_dispatch_wait_counter[
                        2 * expert_rank_offset + 1
                    ]
                    var token_start = token_end - num_tokens
                    received_tokens += Int(num_tokens)

                    for token_idx in range(token_start, token_end):
                        var remote_loc = slot_src_token_info[2 * token_idx]
                        var remote_topk_id = slot_src_token_info[
                            2 * token_idx + 1
                        ]

                        # Get the remote rank's topk_ids for this slot
                        var remote_rank_topk_ids = (
                            host_topk_ids_list[remote_rank]
                            + slot_idx * n_tokens_per_rank * top_k
                        )

                        # Check if curr_expert is in remote rank's topk_ids
                        assert_equal(
                            remote_rank_topk_ids[
                                remote_loc * Int32(top_k) + remote_topk_id
                            ],
                            curr_expert,
                            "Expert mismatch for dev "
                            + String(dev_idx)
                            + " slot "
                            + String(slot_idx)
                            + " token "
                            + String(token_idx),
                        )

                        # Get the remote rank's input tokens for this slot
                        var remote_rank_input_tokens = (
                            host_input_tokens_list[remote_rank]
                            + slot_idx * n_tokens_per_rank * hidden_size
                        )

                        # Check if the received token matches the remote rank's token
                        for i in range(hidden_size):
                            var remote_token_val = remote_rank_input_tokens[
                                remote_loc * Int32(hidden_size) + Int32(i)
                            ]
                            dispatch_test.check_output_val(
                                dev_idx,
                                slot_idx,
                                expert_idx,
                                expert_token_idx,
                                Int(token_idx),
                                i,
                                remote_token_val,
                            )

                        expert_token_idx += 1

            assert_equal(
                received_tokens,
                expected_tokens,
                "Received tokens mismatch for dev "
                + String(dev_idx)
                + " slot "
                + String(slot_idx),
            )

        # Free host buffers
        host_output.free()
        host_row_offsets.free()
        host_expert_ids.free()
        host_src_token_info.free()
        host_atomic_counter.free()

    dispatch_test.save_outputs_to_host(list_of_ctx)
    sync_parallelize(verify_results, n_ranks)
    print("All results verified successfully!")

    for dev_idx in range(n_ranks):
        host_topk_ids_list[dev_idx].free()
        host_input_tokens_list[dev_idx].free()


def test_dispatch_bf16[
    hidden_size: Int,
    top_k: Int,
    n_experts: Int,
    n_ranks: Int,
    n_slots: Int,
    n_tokens_per_rank: Int,
    bench_e2e: Bool = False,
](list_of_ctx: List[DeviceContext]) raises:
    comptime dispatch_test_type = BF16DispatchTest[
        hidden_size, top_k, n_experts, n_ranks, n_slots, n_tokens_per_rank
    ]
    test_dispatch_common[
        DispatchTestType=dispatch_test_type, bench_e2e=bench_e2e
    ](list_of_ctx)


def test_dispatch_blockwise_fp8[
    hidden_size: Int,
    top_k: Int,
    n_experts: Int,
    n_ranks: Int,
    n_slots: Int,
    n_tokens_per_rank: Int,
    bench_e2e: Bool = False,
](list_of_ctx: List[DeviceContext]) raises:
    comptime dispatch_test_type = BlockwiseFP8DispatchTest[
        fp8_dtype=DType.float8_e4m3fn,
        scales_dtype=DType.float32,
        _hidden_size=hidden_size,
        _top_k=top_k,
        _n_experts=n_experts,
        _n_ranks=n_ranks,
        _n_slots=n_slots,
        _n_tokens_per_rank=n_tokens_per_rank,
    ]
    test_dispatch_common[
        DispatchTestType=dispatch_test_type, bench_e2e=bench_e2e
    ](list_of_ctx)


def test_dispatch_block_scaled_nv[
    hidden_size: Int,
    top_k: Int,
    n_experts: Int,
    n_ranks: Int,
    n_slots: Int,
    n_tokens_per_rank: Int,
    bench_e2e: Bool = False,
](list_of_ctx: List[DeviceContext]) raises:
    comptime dispatch_test_type = NVFP4DispatchTest[
        fp4_dtype=DType.uint8,
        scales_dtype=DType.float8_e4m3fn,
        _hidden_size=hidden_size,
        _top_k=top_k,
        _n_experts=n_experts,
        _n_ranks=n_ranks,
        _n_slots=n_slots,
        _n_tokens_per_rank=n_tokens_per_rank,
    ]
    test_dispatch_common[
        DispatchTestType=dispatch_test_type, bench_e2e=bench_e2e
    ](list_of_ctx)


def test_dispatch_mxfp4[
    hidden_size: Int,
    top_k: Int,
    n_experts: Int,
    n_ranks: Int,
    n_slots: Int,
    n_tokens_per_rank: Int,
    bench_e2e: Bool = False,
](list_of_ctx: List[DeviceContext]) raises:
    comptime dispatch_test_type = MXFP4DispatchTest[
        fp4_dtype=DType.uint8,
        scales_dtype=DType.float8_e8m0fnu,
        _hidden_size=hidden_size,
        _top_k=top_k,
        _n_experts=n_experts,
        _n_ranks=n_ranks,
        _n_slots=n_slots,
        _n_tokens_per_rank=n_tokens_per_rank,
    ]
    test_dispatch_common[
        DispatchTestType=dispatch_test_type, bench_e2e=bench_e2e
    ](list_of_ctx)


def main() raises:
    comptime test_gpu_counts = (2, 4, 8)

    if enable_p2p():
        print("Enabled P2P Mem Access on all GPUs.")
    else:
        raise Error("Cannot enable P2P Mem Access!")

    comptime assert (
        has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator()
    ), "Only NVIDIA and AMD GPUs are supported"

    comptime for gpu_idx in range(len(test_gpu_counts)):
        comptime num_gpus = rebind[Int](test_gpu_counts[gpu_idx])
        if DeviceContext.number_of_devices() != num_gpus:
            continue

        # Create GPU context.
        var ctx = List[DeviceContext]()
        for i in range(num_gpus):
            ctx.append(DeviceContext(device_id=i))

        comptime for n_local_experts in [32, 64]:
            comptime if num_gpus * n_local_experts > 256:
                continue

            test_dispatch_bf16[
                hidden_size=3584,  # equivalent to send 7168 FP8s.
                top_k=8,
                n_experts=num_gpus * n_local_experts,
                n_ranks=num_gpus,
                n_slots=1,
                n_tokens_per_rank=64,
                bench_e2e=False,
            ](ctx)

            test_dispatch_blockwise_fp8[
                hidden_size=7168,
                top_k=8,
                n_experts=num_gpus * n_local_experts,
                n_ranks=num_gpus,
                n_slots=1,
                n_tokens_per_rank=64,
                bench_e2e=False,
            ](ctx)

            comptime device_info = DeviceContext.default_device_info

            comptime if has_nvidia_gpu_accelerator() and _is_sm10x_gpu(
                device_info
            ):
                test_dispatch_block_scaled_nv[
                    hidden_size=7168,
                    top_k=8,
                    n_experts=num_gpus * n_local_experts,
                    n_ranks=num_gpus,
                    n_slots=1,
                    n_tokens_per_rank=64,
                    bench_e2e=False,
                ](ctx)

            comptime if device_info == MI355X:
                test_dispatch_mxfp4[
                    hidden_size=7168,
                    top_k=8,
                    n_experts=num_gpus * n_local_experts,
                    n_ranks=num_gpus,
                    n_slots=1,
                    n_tokens_per_rank=64,
                    bench_e2e=False,
                ](ctx)

        # More local experts than half the comm-SM count. `dispatch_wait`
        # maps SMs to experts with `umod(sm_id, n_local_experts)`, so past
        # that point an expert gets a single SM -- and the block-scaled
        # format splits each token tile into two claims, one per K tile. An
        # SM that stopped on the first claim left the second K half of the
        # final token tile uncopied (every token at batch 1), which reached
        # the expert matmuls as NaN. 112 experts per device is the first
        # production shape that hits it (896 / EP8). Kept outside the sweep
        # above because its `> 256` cap would skip this count on 4+ GPUs.
        comptime if has_nvidia_gpu_accelerator() and _is_sm10x_gpu(
            DeviceContext.default_device_info
        ):
            test_dispatch_block_scaled_nv[
                hidden_size=3584,
                top_k=16,
                n_experts=num_gpus * 112,
                n_ranks=num_gpus,
                n_slots=1,
                n_tokens_per_rank=64,
                bench_e2e=False,
            ](ctx)
