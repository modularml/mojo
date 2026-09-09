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

"""End-to-end tests for the inlined DispatcherClient / DispatcherServer.

Exercises the ZMQ round-trip between ``DispatcherClient`` (dealer) and
``DispatcherServer`` (router) at the scheduler boundary, parameterized with
plain ``int`` request/reply types so the tests are decoupled from the
disaggregated-inference scheduler payloads.
"""

from __future__ import annotations

import queue
import time
from collections.abc import Callable
from typing import TypeVar

from max.serve.scheduler.di_dispatchers import (
    DispatcherClient,
    DispatcherServer,
)
from max.serve.worker_interface._zmq_queue import (
    ClientIdentity,
    generate_zmq_ipc_path,
)

T = TypeVar("T")

TIMEOUT = 1.0


def blocking_recv(fn: Callable[[], T], timeout: float = TIMEOUT) -> T:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            return fn()
        except queue.Empty:
            time.sleep(0.001)
    raise queue.Empty()


class BasicDispatcherServer(DispatcherServer[int, int]):
    def __init__(self, bind_addr: str):
        self.bind_addr = bind_addr
        super().__init__(endpoint=bind_addr, request_type=int, reply_type=int)

    def recv_request_blocking(self) -> tuple[int, ClientIdentity]:
        return blocking_recv(self.recv_request_nowait)


class BasicDispatcherClient(DispatcherClient[int, int]):
    def __init__(self, bind_addr: str):
        self.bind_addr = bind_addr
        super().__init__(
            bind_addr=bind_addr,
            request_type=int,
            reply_type=int,
        )

    def recv_reply_blocking(self) -> int:
        return blocking_recv(self.recv_reply_nowait)


def make_servers_and_clients(
    num_servers: int, num_clients: int
) -> tuple[list[BasicDispatcherServer], list[BasicDispatcherClient]]:
    server_addrs = [generate_zmq_ipc_path() for _ in range(num_servers)]
    client_addrs = [generate_zmq_ipc_path() for _ in range(num_clients)]
    servers = [
        BasicDispatcherServer(bind_addr=server_addr)
        for server_addr in server_addrs
    ]
    clients = [
        BasicDispatcherClient(bind_addr=client_addr)
        for client_addr in client_addrs
    ]
    return servers, clients


def test_server_client() -> None:
    servers, clients = make_servers_and_clients(1, 1)
    server = servers[0]
    client = clients[0]

    for _ in range(100):
        client.send_request_nowait(42, server.bind_addr)
        request, identity = server.recv_request_blocking()
        assert request == 42
        server.send_reply_nowait(99, identity)
        reply = client.recv_reply_blocking()
        assert reply == 99


def test_many_clients_one_server() -> None:
    servers, clients = make_servers_and_clients(1, 10)
    server = servers[0]

    for i, client in enumerate(clients):
        client.send_request_nowait(i, server.bind_addr)

    num_requests_received = 0
    while True:
        try:
            request, identity = server.recv_request_blocking()
            num_requests_received += 1
            server.send_reply_nowait(request + 100, identity)
        except queue.Empty:
            break
    assert num_requests_received == len(clients)

    for i, client in enumerate(clients):
        reply = client.recv_reply_blocking()
        assert reply == i + 100


def test_many_servers_one_client() -> None:
    servers, clients = make_servers_and_clients(10, 1)
    client = clients[0]

    for server in servers:
        client.send_request_nowait(-1, server.bind_addr)

    for i, server in enumerate(servers):
        request, identity = server.recv_request_blocking()
        server.send_reply_nowait(request * i * 100, identity)

    replies = []
    while True:
        try:
            reply = client.recv_reply_blocking()
            replies.append(reply)
        except queue.Empty:
            break

    assert len(replies) == len(servers)
    assert set(replies) == {-1 * i * 100 for i in range(len(servers))}


def test_spam_server() -> None:
    servers, clients = make_servers_and_clients(1, 1)
    server = servers[0]
    client = clients[0]

    for i in range(100):
        client.send_request_nowait(i, server.bind_addr)

    while True:
        try:
            request, identity = server.recv_request_blocking()
            server.send_reply_nowait(request + 100, identity)
        except queue.Empty:
            break

    for i in range(100):
        reply = client.recv_reply_blocking()
        assert reply == i + 100
