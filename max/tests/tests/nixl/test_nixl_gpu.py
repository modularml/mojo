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
"""Tests basic NIXL functionality"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from max._core.nixl import (
    Agent,
    AgentConfig,
    MemoryType,
    RegistrationDescriptorList,
    Status,
    TransferDescriptorList,
    TransferOpType,
)
from max.driver import CPU, Accelerator, Buffer, Device
from numpy.typing import NDArray


def create_agent(
    name: str = "test_agent",
    listen_port: int = 8047,
    gpu_device_id: int | None = None,
) -> tuple[Agent, int]:
    # Stand up Agent
    agent = Agent(
        name,
        AgentConfig(
            use_prog_thread=False,
            use_listen_thread=True,
            listen_port=listen_port,
        ),
    )

    # Upstream NIXL plugin names are uppercase (UCX, LIBFABRIC).
    assert "UCX" in agent.get_available_plugins()

    # Create UCX backend.
    ucx_params = agent.get_plugin_params("UCX")
    # Add GPU device ID to parameters if specified.
    if gpu_device_id is not None:
        ucx_params[0]["gpu_device_id"] = str(gpu_device_id)
    ucx_backend = agent.create_backend(type="UCX", init_params=ucx_params[0])

    return agent, ucx_backend


def test_agent_registration() -> None:
    _ = create_agent()


def test_remote_agent_registration() -> None:
    agent_1, _ = create_agent("agent_1", 8047)
    agent_2, _ = create_agent("agent_2", 8057)

    # Get Agent Metadata from Agents
    agent_1_md = agent_1.get_local_metadata()
    agent_2_md = agent_2.get_local_metadata()

    # Register Agent with Pair
    remote_bytes_1 = agent_2.load_remote_metadata(agent_1_md)
    assert remote_bytes_1.decode() == "agent_1"
    remote_bytes_2 = agent_1.load_remote_metadata(agent_2_md)
    assert remote_bytes_2.decode() == "agent_2"


@pytest.mark.parametrize("device", [CPU(), Accelerator()])
def test_memory_registration(device: Device) -> None:
    agent, ucx_backend = create_agent(
        gpu_device_id=0 if not device.is_host else None
    )

    buffer = Buffer.from_numpy(np.ones((100, 100))).to(device)

    # if the descriptors are sorted for descs and device ID.
    # sort criteria has the comparison order of dev_id, then addr, then len.
    memory_type = MemoryType.DRAM if device.is_host else MemoryType.VRAM

    registration_descriptor = RegistrationDescriptorList(
        type=memory_type,
        # This cast should not be necessary.
        # descs should accept any object that implements __dlpack__ instead.
        descs=[cast(NDArray[Any], buffer)],
    )

    # Test append()
    buffer2 = Buffer.from_numpy(np.ones((50, 50))).to(device)
    registration_descriptor.append(cast(NDArray[Any], buffer2))

    # Register Memory
    status = agent.register_memory(
        descs=registration_descriptor,
        backends=[ucx_backend],
    )
    assert status == Status.SUCCESS


@pytest.mark.parametrize("device", [CPU()])
def test_memory_transfer(device: Device) -> None:
    buffer_size_1 = 1024
    buffer_magic_val_1 = 99
    buffer_size_2 = 2048
    buffer_magic_val_2 = 42
    agent_1_name = "agent_1"
    agent_2_name = "agent_2"

    buffer_1 = Buffer.from_numpy(
        np.full((buffer_size_1,), buffer_magic_val_1, dtype=np.int8)
    ).to(device)
    buffer_2 = Buffer.from_numpy(
        np.full((buffer_size_2,), buffer_magic_val_2, dtype=np.int8)
    ).to(device)

    memory_type = MemoryType.DRAM if device.is_host else MemoryType.VRAM

    # Create Agents
    agent_1, ucx_backend_1 = create_agent(agent_1_name, listen_port=8037)
    agent_2, ucx_backend_2 = create_agent(agent_2_name, listen_port=8042)

    # Register Memory
    reg_dlist_1 = RegistrationDescriptorList(
        type=memory_type, descs=[cast(NDArray[Any], buffer_1)]
    )
    agent_1.register_memory(descs=reg_dlist_1, backends=[ucx_backend_1])

    reg_dlist_2 = RegistrationDescriptorList(
        type=memory_type, descs=[cast(NDArray[Any], buffer_2)]
    )
    agent_2.register_memory(descs=reg_dlist_2, backends=[ucx_backend_2])

    # Get Agent Metadata from Agents
    agent_1_md = agent_1.get_local_metadata()
    agent_2_md = agent_2.get_local_metadata()

    # Register Agent with Pair
    remote_bytes_2 = agent_2.load_remote_metadata(agent_1_md)
    assert remote_bytes_2.decode() == agent_1_name
    remote_bytes_1 = agent_1.load_remote_metadata(agent_2_md)
    assert remote_bytes_1.decode() == agent_2_name

    # Compute buffer regions to transfer
    # The offsets and transfer sizes are arbitrarily chosen for this test.
    dst_offset = buffer_size_1 // 4
    src_offset = 0
    bytes_to_copy = buffer_size_1 // 2
    src_base = buffer_1._data_ptr() + src_offset
    dst_base = buffer_2._data_ptr() + dst_offset

    # Create xfer request
    xfer_dlist_src = TransferDescriptorList(
        type=memory_type,
        descs=[(src_base, bytes_to_copy, memory_type.value)],
    )
    xfer_dlist_dst = TransferDescriptorList(
        type=memory_type,
        descs=[(dst_base, bytes_to_copy, memory_type.value)],
    )
    secret_message = "mojo is so fast 🔥"
    xfer_req = agent_1.create_transfer_request(
        operation=TransferOpType.WRITE,
        local_descs=xfer_dlist_src,
        remote_descs=xfer_dlist_dst,
        remote_agent=agent_2_name,
        notif_msg=secret_message,
    )

    status = agent_1.post_transfer_request(xfer_req)
    assert status in [Status.SUCCESS, Status.IN_PROG], (
        "Failed to post transfer request"
    )

    check_status = Status.IN_PROG
    notif_received = False
    start_time = time.time()
    timeout = 5.0
    notif_map: dict[str, list[bytes]] = {}

    while (check_status != Status.SUCCESS or not notif_received) and (
        time.time() - start_time
    ) < timeout:
        if check_status != Status.SUCCESS:
            check_status = agent_1.get_transfer_status(xfer_req)
            assert check_status in [Status.SUCCESS, Status.IN_PROG], (
                "Received unexpected transfer status"
            )

        if not notif_received:
            notif_map = agent_2.get_notifs()
            if agent_1_name in notif_map and len(notif_map[agent_1_name]) > 0:
                notif_received = True

        # Yield/Sleep briefly to avoid busy-waiting
        time.sleep(0.001)

    assert check_status == Status.SUCCESS, (
        f"Transfer did not complete in {timeout} seconds"
    )
    assert agent_1_name in notif_map, "Notification map missing agent 1"
    assert len(notif_map[agent_1_name]) == 1, (
        "Incorrect number of notifications received"
    )
    received_message = notif_map[agent_1_name][0].decode()
    assert received_message == secret_message, "Notification message mismatch"

    def s2ms(s: float) -> float:
        return s * 1000

    # Should take only a couple ms
    print(f"Transfer completed in {s2ms(time.time() - start_time):.2f} ms")
    print(f"Agent 2 received notification from Agent 1: '{received_message}'")

    # Verify transferred data

    # Buffer 1 should be unchanged
    buffer_1_expected = np.full(
        (buffer_size_1,), buffer_magic_val_1, dtype=np.int8
    )
    assert np.array_equal(buffer_1.to_numpy(), buffer_1_expected), (
        "Transferred data does not match expected data"
    )

    # Buffer 2 should be mostly 42's except for some region of 99's
    buffer_2_expected = np.full(
        (buffer_size_2,), buffer_magic_val_2, dtype=np.int8
    )
    buffer_2_expected[dst_offset : dst_offset + bytes_to_copy] = (
        buffer_magic_val_1
    )
    assert np.array_equal(buffer_2.to_numpy(), buffer_2_expected), (
        "Transferred data does not match expected data"
    )

    # Cleanup
    status = agent_1.release_transfer_request(xfer_req)
    assert status == Status.SUCCESS, "Failed to release transfer request"

    status = agent_2.deregister_memory(reg_dlist_2)
    assert status == Status.SUCCESS, "Failed to deregister memory"

    status = agent_1.invalidate_remote_metadata(agent_2_name)
    assert status == Status.SUCCESS, "Failed to invalidate remote metadata"

    status = agent_2.invalidate_remote_metadata(agent_1_name)
    assert status == Status.SUCCESS, "Failed to invalidate remote metadata"

    status = agent_1.deregister_memory(reg_dlist_1)
    assert status == Status.SUCCESS, "Failed to deregister memory"


def test_get_transfer_telemetry_enabled_by_env() -> None:
    # Upstream NIXL reads NIXL_TELEMETRY_ENABLE / NIXL_TELEMETRY_DIR /
    # NIXL_TELEMETRY_RUN_INTERVAL (not the MODULAR_-prefixed fork names).
    # The binding's get_transfer_telemetry() calls upstream getXferTelemetry()
    # which returns NIXL_ERR_NO_TELEMETRY if telemetry was not enabled at
    # nixlAgent construction time.
    os.environ["NIXL_TELEMETRY_ENABLE"] = "y"
    telemetry_dir = Path("/tmp") / "nixl_py_telem_env"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NIXL_TELEMETRY_DIR"] = str(telemetry_dir)
    os.environ["NIXL_TELEMETRY_RUN_INTERVAL"] = "1"
    try:
        buffer_size = 1024
        agent_1_name = "agent_telemetry_1"
        agent_2_name = "agent_telemetry_2"

        buffer_1 = Buffer.from_numpy(
            np.full((buffer_size,), 99, dtype=np.int8)
        ).to(CPU())
        buffer_2 = Buffer.from_numpy(
            np.full((buffer_size,), 42, dtype=np.int8)
        ).to(CPU())

        agent_1, ucx_backend_1 = create_agent(agent_1_name, listen_port=8051)
        agent_2, ucx_backend_2 = create_agent(agent_2_name, listen_port=8052)

        reg_dlist_1 = RegistrationDescriptorList(
            type=MemoryType.DRAM, descs=[cast(NDArray[Any], buffer_1)]
        )
        reg_dlist_2 = RegistrationDescriptorList(
            type=MemoryType.DRAM, descs=[cast(NDArray[Any], buffer_2)]
        )
        agent_1.register_memory(descs=reg_dlist_1, backends=[ucx_backend_1])
        agent_2.register_memory(descs=reg_dlist_2, backends=[ucx_backend_2])

        agent_1_md = agent_1.get_local_metadata()
        agent_2_md = agent_2.get_local_metadata()
        assert agent_2.load_remote_metadata(agent_1_md).decode() == agent_1_name
        assert agent_1.load_remote_metadata(agent_2_md).decode() == agent_2_name

        xfer_dlist_src = TransferDescriptorList(
            type=MemoryType.DRAM,
            descs=[(buffer_1._data_ptr(), buffer_size, MemoryType.DRAM.value)],
        )
        xfer_dlist_dst = TransferDescriptorList(
            type=MemoryType.DRAM,
            descs=[(buffer_2._data_ptr(), buffer_size, MemoryType.DRAM.value)],
        )
        xfer_req = agent_1.create_transfer_request(
            operation=TransferOpType.WRITE,
            local_descs=xfer_dlist_src,
            remote_descs=xfer_dlist_dst,
            remote_agent=agent_2_name,
            notif_msg="telemetry",
        )

        status = agent_1.post_transfer_request(xfer_req)
        assert status in [Status.SUCCESS, Status.IN_PROG]

        notif_received = False
        start_time = time.time()
        timeout = 5.0
        notif_map: dict[str, list[bytes]] = {}

        while (status != Status.SUCCESS or not notif_received) and (
            time.time() - start_time
        ) < timeout:
            if status != Status.SUCCESS:
                status = agent_1.get_transfer_status(xfer_req)
                assert status in [Status.SUCCESS, Status.IN_PROG]

            if not notif_received:
                notif_map = agent_2.get_notifs()
                if (
                    agent_1_name in notif_map
                    and len(notif_map[agent_1_name]) > 0
                ):
                    notif_received = True

            time.sleep(0.001)

        assert status == Status.SUCCESS
        assert notif_received

        telemetry = agent_1.get_transfer_telemetry(xfer_req)
        assert telemetry.descCount == 1
        assert telemetry.totalBytes == buffer_size
        assert telemetry.postDuration > 0
        assert telemetry.xferDuration >= telemetry.postDuration

        agent_1.release_transfer_request(xfer_req)
        agent_1.invalidate_remote_metadata(agent_2_name)
        agent_2.invalidate_remote_metadata(agent_1_name)
        assert agent_1.deregister_memory(reg_dlist_1) == Status.SUCCESS
        assert agent_2.deregister_memory(reg_dlist_2) == Status.SUCCESS
    finally:
        os.environ.pop("NIXL_TELEMETRY_ENABLE", None)
        os.environ.pop("NIXL_TELEMETRY_DIR", None)
        os.environ.pop("NIXL_TELEMETRY_RUN_INTERVAL", None)
