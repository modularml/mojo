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

"""Reset prefix cache functionality for MAX. This simply allows the API Worker to
communicate with the model worker about when to reset the prefix cache.

The actual resetting of prefix cache is handled by caller to `should_reset_prefix_cache()`
in model_worker.py.

We utilize a `zmq_endpoint_base` in order to unique the ZMQ endpoints in case there
are multiple MAX Serve instances running on the same host.
"""

import queue

from max.serve.queue import get_blocking
from max.serve.worker_interface._zmq_queue import ZmqPullSocket, ZmqPushSocket

ZMQ_RESET_PREFIX_CACHE_ENDPOINT = "reset_prefix_cache"


class ResetPrefixCacheFrontend:
    """Frontend for resetting the prefix cache.

    This frontend is constructed by the API worker and exposes a API to enqueue
    a request to reset the prefix cache. This request is forwarded to the backend.
    """

    def __init__(self, zmq_endpoint_base: str):
        """Initialize the frontend.

        Args:
            zmq_endpoint_base: The base endpoint for the ZMQ socket, this should
                               be the same as the one used for the backend.
        """
        self.socket = ZmqPushSocket[None](
            endpoint=f"{zmq_endpoint_base}-{ZMQ_RESET_PREFIX_CACHE_ENDPOINT}",
            payload_type=None,
        )

    def enqueue_reset_prefix_cache(self) -> None:
        self.socket.put(None)


class ResetPrefixCacheBackend:
    """Backend for resetting the prefix cache.

    This backend is constructed by the model worker and can poll for request to
    reset the prefix cache.
    """

    def __init__(self, zmq_endpoint_base: str):
        """Initialize the backend.

        Args:
            zmq_endpoint_base: The base endpoint for the ZMQ socket, this should
                               be the same as the one used for the frontend.
        """
        self.socket = ZmqPullSocket[None](
            endpoint=f"{zmq_endpoint_base}-{ZMQ_RESET_PREFIX_CACHE_ENDPOINT}",
            payload_type=None,
        )

    def should_reset_prefix_cache(self, blocking: bool = False) -> bool:
        """Check if there is a request to reset the prefix cache.

        Args:
            blocking: Whether to block until a request is received. This is primarily
                      used for testing purposes.

        Returns:
            True if there is a request to reset the prefix cache, False otherwise.
        """
        # If blocking is True, we do not return until we receive a message from
        # the frontend to reset the prefix cache. Hence, it will always return True.
        if blocking:
            get_blocking(self.socket)
            return True

        # If non-blocking, we return True if there is a message in the queue.
        try:
            self.socket.get_nowait()
            return True
        except queue.Empty:
            return False
