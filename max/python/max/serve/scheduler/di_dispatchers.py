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

from __future__ import annotations

import logging
import queue
from typing import Any, Generic, TypeVar, cast

from max.pipelines.context import TextContext
from max.pipelines.kv_cache import KVTransferEngineMetadata
from max.serve.scheduler.base import (
    CancelRequest,
    PrefillFailure,
    PrefillProgressPing,
    PrefillRequest,
    PrefillResponse,
)
from max.serve.worker_interface._zmq_queue import (
    ZmqDealerSocket,
    ZmqRouterSocket,
)

logger = logging.getLogger("max.serve")

Request = TypeVar("Request")
Reply = TypeVar("Reply")

DispatcherServer = ZmqRouterSocket[Request, Reply]


class DispatcherClient(Generic[Request, Reply]):
    def __init__(
        self,
        *,
        bind_addr: str,
        request_type: Any,
        reply_type: Any,
    ):
        self._request_type = request_type
        self._reply_type = reply_type

        self._dealers: dict[str, ZmqDealerSocket[Request, Reply]] = {}

    def send_request_nowait(self, request: Request, dest_addr: str) -> None:
        if dest_addr not in self._dealers:
            self._dealers[dest_addr] = ZmqDealerSocket[Request, Reply](
                endpoint=dest_addr,
                request_type=self._request_type,
                reply_type=self._reply_type,
            )
        dealer = self._dealers[dest_addr]
        dealer.send_request_nowait(request)

    def recv_reply_nowait(self) -> Reply:
        for dealer in self._dealers.values():
            try:
                reply = dealer.recv_reply_nowait()
            except queue.Empty:
                continue
            return reply
        raise queue.Empty()


RequestType = (
    PrefillRequest[TextContext] | KVTransferEngineMetadata | CancelRequest
)
ReplyType = (
    PrefillResponse
    | PrefillFailure
    | KVTransferEngineMetadata
    | PrefillProgressPing
)


class PrefillDispatcherServer(DispatcherServer[RequestType, ReplyType]):
    def __init__(
        self,
        bind_addr: str,
        context_type: type[TextContext] = TextContext,
    ):
        """Initializes the prefill-side dispatcher server.

        Args:
            bind_addr: ZMQ endpoint to bind the router socket to.
            context_type: The architecture's concrete context class. msgspec
                decodes ``PrefillRequest.context`` at this declared type, so
                VLM architectures must pass their ``TextAndVisionContext``
                subclass or vision fields are silently dropped on the wire.
        """
        logger.info(f"Starting Prefill Dispatcher Server on {bind_addr}")
        super().__init__(
            endpoint=bind_addr,
            # context_type is chosen at runtime per architecture, so mypy
            # can't validate it as a type argument here — erase to Any
            # before subscripting to avoid a static "not valid as a type"
            # error on what is otherwise a normal runtime parametrization.
            request_type=cast(Any, PrefillRequest)[context_type]
            | KVTransferEngineMetadata
            | CancelRequest,
            reply_type=ReplyType,
        )


class DecodeDispatcherClient(DispatcherClient[RequestType, ReplyType]):
    def __init__(self, bind_addr: str):
        logger.info(f"Starting Decode Dispatcher Client on {bind_addr}")
        super().__init__(
            bind_addr=bind_addr,
            request_type=RequestType,
            reply_type=ReplyType,
        )
