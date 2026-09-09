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


import os

import numpy as np
import pytest
from _transfer_engine_helpers import kv_memory
from max.driver import CPU, Accelerator
from max.driver.buffer import Buffer
from max.dtype import DType
from max.pipelines.kv_cache import KVTransferEngine


def test_constructor() -> None:
    tensor = Buffer(DType.int8, (10, 10), device=CPU())

    # ok - DP=1, TP=1
    _ = KVTransferEngine(
        "abc",
        [[kv_memory(tensor, 2)]],
    )
    _ = KVTransferEngine(
        "abc",
        [[kv_memory(tensor.to(Accelerator()), 2)]],
    )


def test_initiate_send_transfer() -> None:
    device = CPU()
    total_num_pages = 3
    elts_per_page = 3
    num_elts = total_num_pages * elts_per_page

    blocks_1 = Buffer.from_numpy(np.arange(num_elts, dtype=np.int16) + 10).to(
        device
    )
    blocks_2 = Buffer.from_numpy(np.arange(num_elts, dtype=np.int16) + 80).to(
        device
    )

    # DP=1, TP=1
    engine_1 = KVTransferEngine(
        "engine_1",
        [[kv_memory(blocks_1, total_num_pages)]],
    )
    engine_2 = KVTransferEngine(
        "engine_2",
        [[kv_memory(blocks_2, total_num_pages)]],
    )

    engine_1.connect(engine_2.metadata)
    engine_2.connect(engine_1.metadata)

    # ok
    _ = engine_1.initiate_send_transfer(
        engine_2.metadata,
        src_idxs=[2, 1],
        dst_idxs=[1, 0],
        src_replica_idx=0,
        dst_replica_idx=0,
    )

    # oob src_idx
    with pytest.raises(ValueError):
        _ = engine_1.initiate_send_transfer(
            engine_2.metadata,
            src_idxs=[100],
            dst_idxs=[1],
            src_replica_idx=0,
            dst_replica_idx=0,
        )

    # oob dst_idx
    with pytest.raises(ValueError):
        _ = engine_1.initiate_send_transfer(
            engine_2.metadata,
            src_idxs=[2, 0],
            dst_idxs=[100, 0],
            src_replica_idx=0,
            dst_replica_idx=0,
        )

    # oob dst_idx
    with pytest.raises(ValueError):
        _ = engine_1.initiate_send_transfer(
            engine_2.metadata,
            src_idxs=[2],
            dst_idxs=[-1],
            src_replica_idx=0,
            dst_replica_idx=0,
        )

    # mismatch lengths
    with pytest.raises(ValueError):
        _ = engine_1.initiate_send_transfer(
            engine_2.metadata,
            src_idxs=[2],
            dst_idxs=[0, 1],
            src_replica_idx=0,
            dst_replica_idx=0,
        )

    # write to same dst page
    with pytest.raises(ValueError):
        _ = engine_1.initiate_send_transfer(
            engine_2.metadata,
            src_idxs=[2, 1],
            dst_idxs=[0, 0],
            src_replica_idx=0,
            dst_replica_idx=0,
        )

    engine_1.cleanup()
    engine_2.cleanup()


def test_ensure_we_use_memory_manager() -> None:
    cpu_device = CPU()
    cpu_tensor = Buffer.from_numpy(np.arange(10, dtype=np.int16)).to(cpu_device)

    acc_device = Accelerator()
    acc_tensor = cpu_tensor.to(acc_device)

    # unset BAZEL_TEST to pretend that we are not in a bazel test
    os.environ.pop("BAZEL_TEST", None)

    # fails
    with pytest.raises(
        ValueError,
        match="MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT must be set when using TransferEngine with GPU memory",
    ):
        engine = KVTransferEngine("engine", [[kv_memory(acc_tensor, 1)]])

    # ok
    engine = KVTransferEngine("engine", [[kv_memory(cpu_tensor, 1)]])
    engine.cleanup()

    # ok
    os.environ["MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT"] = "99"
    engine = KVTransferEngine("engine", [[kv_memory(acc_tensor, 1)]])
    engine.cleanup()


def test_dp_structure_validation() -> None:
    """Test that DP structure (2D list) is properly validated."""
    tensor = Buffer(DType.int8, (10, 10), device=CPU())

    # Empty outer list should fail
    with pytest.raises(ValueError, match="must contain at least one replica"):
        _ = KVTransferEngine("engine", [])

    # Empty inner list should fail
    with pytest.raises(ValueError, match="must contain at least one tensor"):
        _ = KVTransferEngine("engine", [[]])

    # Valid DP=1, TP=1
    engine = KVTransferEngine("engine", [[kv_memory(tensor, 2)]])
    engine.cleanup()


def test_multi_replica_construction() -> None:
    """Test constructing engine with multiple DP replicas."""
    tensor1 = Buffer(DType.int8, (10, 10), device=CPU())
    tensor2 = Buffer(DType.int8, (10, 10), device=CPU())

    # DP=2, TP=1
    engine = KVTransferEngine(
        "engine",
        [[kv_memory(tensor1, 2)], [kv_memory(tensor2, 2)]],
    )

    engine.cleanup()


def test_replicas_must_have_same_tp_degree() -> None:
    """Test that all replicas must have the same bytes_per_page.

    Note: This test validates the error message but uses TP=1 for both replicas
    since CPU only supports TP=1. The validation would work the same for GPU.
    """
    t1 = Buffer(DType.int8, (10, 10), device=CPU())
    # Different size to test validation
    t2 = Buffer(DType.int8, (20, 20), device=CPU())

    # Note: We can't actually test TP>1 mismatch with CPU since CPU requires TP=1.
    # Instead, test that different tensor shapes are caught by the bytes_per_page validation.
    # For TP>1 validation, the same check would apply with GPUs.
    with pytest.raises(
        ValueError,
        match=r"All replicas must have the same bytes_per_page",
    ):
        _ = KVTransferEngine(
            "engine",
            [
                [kv_memory(t1, 2)],
                [kv_memory(t2, 2)],
            ],  # Different shapes lead to different bytes_per_page
        )


def test_replicas_must_have_same_total_num_pages() -> None:
    """All groups must agree on total_num_pages, read off the buffers."""
    t2 = Buffer(DType.int8, (20,), device=CPU())  # 2 pages of 10 bytes
    t3 = Buffer(DType.int8, (30,), device=CPU())  # 3 pages of 10 bytes

    with pytest.raises(ValueError, match=r"total_num_pages"):
        _ = KVTransferEngine(
            "engine",
            [[kv_memory(t2, 2)], [kv_memory(t3, 3)]],
        )


def test_replica_idx_validation() -> None:
    """Test that replica_idx is validated in transfer methods."""
    device = CPU()
    tensor1 = Buffer.from_numpy(np.arange(9, dtype=np.int16)).to(device)
    tensor2 = Buffer.from_numpy(np.arange(9, dtype=np.int16) + 10).to(device)

    # DP=2, TP=1
    engine_1 = KVTransferEngine(
        "engine_1",
        [[kv_memory(tensor1, 3)], [kv_memory(tensor2, 3)]],
    )
    engine_2 = KVTransferEngine(
        "engine_2",
        [[kv_memory(tensor1, 3)], [kv_memory(tensor2, 3)]],
    )

    engine_1.connect(engine_2.metadata)
    engine_2.connect(engine_1.metadata)

    # Valid src_replica_idx=0, dst_replica_idx=0
    _ = engine_1.initiate_send_transfer(
        engine_2.metadata,
        src_idxs=[0],
        dst_idxs=[1],
        src_replica_idx=0,
        dst_replica_idx=0,
    )

    # Valid src_replica_idx=1, dst_replica_idx=1
    _ = engine_1.initiate_send_transfer(
        engine_2.metadata,
        src_idxs=[0],
        dst_idxs=[2],
        src_replica_idx=1,
        dst_replica_idx=1,
    )

    # Invalid src_replica_idx=-1, dst_replica_idx=0
    with pytest.raises(ValueError, match=r"src_replica_idx .* must be between"):
        _ = engine_1.initiate_send_transfer(
            engine_2.metadata,
            src_idxs=[0],
            dst_idxs=[1],
            src_replica_idx=-1,
            dst_replica_idx=0,
        )

    # Invalid src_replica_idx=2, dst_replica_idx=0 (only have 2 replicas: 0 and 1)
    with pytest.raises(ValueError, match=r"src_replica_idx .* must be between"):
        _ = engine_1.initiate_send_transfer(
            engine_2.metadata,
            src_idxs=[0],
            dst_idxs=[1],
            src_replica_idx=2,
            dst_replica_idx=0,
        )

    engine_1.cleanup()
    engine_2.cleanup()
