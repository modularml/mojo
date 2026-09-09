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
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

"""
nanobind NIXL plugin: Implements NIXL descriptors and lists, as well as bindings of NIXL CPP APIs.
"""

import enum
from collections.abc import Mapping, Sequence
from typing import overload

from numpy.typing import NDArray

INIT_AGENT: str = ""

DEFAULT_COMM_PORT: int = 8888

class ThreadSyncMode(enum.Enum):
    NONE = 0

    STRICT = 1

class MemoryType(enum.Enum):
    DRAM = 0

    VRAM = 1

    BLK = 2

    OBJ = 3

    FILE = 4

class TransferOpType(enum.Enum):
    READ = 0

    WRITE = 1

class Status(enum.Enum):
    """
    Status code for NIXL operations.

    !! NOTE !! This differs from the C++ API. When an operation would
    return an error in the C++ API, here it would instead throw a similarly
    named exception. If a function can only one of these two statuses (usually
    SUCCESS) aside from throwing an error, the return code is omitted and is
    assumed to be that status code (again, usually SUCCESS).
    """

    IN_PROG = 1

    SUCCESS = 0

class NotPostedError(Exception): ...
class InvalidParamError(Exception): ...
class BackendError(Exception): ...
class NotFoundError(Exception): ...
class MismatchError(Exception): ...
class NotAllowedError(Exception): ...
class RepostActiveError(Exception): ...
class UnknownError(Exception): ...
class NotSupportedError(Exception): ...
class RemoteDisconnectError(Exception): ...
class CanceledError(Exception): ...
class NoTelemetryError(Exception): ...

class TransferTelemetry:
    @property
    def startTime(self) -> int: ...
    @property
    def postDuration(self) -> int: ...
    @property
    def xferDuration(self) -> int: ...
    @property
    def totalBytes(self) -> int: ...
    @property
    def descCount(self) -> int: ...

class TransferDescriptorList:
    @overload
    def __init__(self, type: MemoryType, init_size: int = 0) -> None:
        """
        Constructs an empty descriptor list.

        Args:
          type: The type of memory each element describes.
          init_size: Initial capacity of the list.
        """

    @overload
    def __init__(
        self, type: MemoryType, descs: Sequence[NDArray | tuple[int, int, int]]
    ) -> None:
        """
        Constructs a descriptor list with given values.

        Args:
          type: The type of memory each element describes.
          descs: A list of descriptors, each describing a section of memory.
                 Each element is either a tuple or a dlpack object.
        """

    def append(self, desc: NDArray | tuple[int, int, int]) -> None:
        """
        Adds a descriptor to the list.

        Args:
          desc: A descriptor describing a section of memory.
                Either a tuple or a dlpack object.
        """

    @property
    def type(self) -> MemoryType: ...
    @property
    def descriptor_count(self) -> int: ...
    def is_empty(self) -> bool: ...
    def __eq__(self, arg: object, /) -> bool: ...
    def __getitem__(self, idx: int) -> tuple[int, int, int]: ...
    def __setitem__(self, idx: int, desc: tuple[int, int, int]) -> None: ...
    def index(self, desc: tuple) -> int: ...
    def remove(self, idx: int) -> None: ...
    def clear(self) -> None: ...
    def print(self) -> None: ...

class RegistrationDescriptorList:
    @overload
    def __init__(self, type: MemoryType, init_size: int = 0) -> None:
        """
        Constructs an empty descriptor list.

        Args:
          type: The type of memory each element describes.
          init_size: Initial capacity of the list.
        """

    @overload
    def __init__(
        self,
        type: MemoryType,
        descs: Sequence[NDArray | tuple[int, int, int, str]],
    ) -> None:
        """
        Constructs a descriptor list with given values.

        Args:
          type: The type of memory each element describes.
          descs: A list of descriptors, each describing a section of memory.
                 Each element is either a tuple or a dlpack object.
        """

    def append(self, desc: NDArray | tuple[int, int, int, str]) -> None:
        """
        Adds a descriptor to the list.

        Args:
          desc: A descriptor describing a section of memory.
                Either a tuple or a dlpack object.
        """

    @property
    def type(self) -> MemoryType: ...
    @property
    def descriptor_count(self) -> int: ...
    def is_empty(self) -> bool: ...
    def __eq__(self, arg: object, /) -> bool: ...
    def __getitem__(self, idx: int) -> tuple[int, int, int, str]: ...
    def __setitem__(
        self, idx: int, desc: tuple[int, int, int, str]
    ) -> None: ...
    def index(self, desc: tuple) -> int: ...
    def trim(self) -> TransferDescriptorList: ...
    def remove(self, idx: int) -> None: ...
    def clear(self) -> None: ...
    def print(self) -> None: ...

class AgentConfig:
    def __init__(
        self,
        use_prog_thread: bool,
        use_listen_thread: bool = False,
        listen_port: int = 8888,
        sync_mode: ThreadSyncMode = ThreadSyncMode.NONE,
        num_workers: int = 1,
        pthr_delay_us: int = 0,
        lthr_delay_us: int = 100000,
    ) -> None: ...

class Agent:
    def __init__(self, name: str, config: AgentConfig) -> None: ...
    def get_available_plugins(self) -> list[str]: ...
    def get_plugin_params(
        self, type: str
    ) -> tuple[dict[str, str], list[str]]: ...
    def get_backend_params(
        self, backend: int
    ) -> tuple[dict[str, str], list[str]]: ...
    def create_backend(
        self, type: str, init_params: Mapping[str, str]
    ) -> int: ...
    def register_memory(
        self, descs: RegistrationDescriptorList, backends: Sequence[int] = []
    ) -> Status: ...
    def deregister_memory(
        self, descs: RegistrationDescriptorList, backends: Sequence[int] = []
    ) -> Status: ...
    def query_memory(
        self, descs: RegistrationDescriptorList, backend: int
    ) -> list[dict[str, str] | None]:
        """
        Queries a registered descriptor list against a backend.

        Args:
            descs: The registered descriptors to query.
            backend: The backend handle to query against.

        Returns:
            A list of backend-specific responses for the descriptors.
        """

    def make_connection(
        self, remote_agent: str, backends: Sequence[int]
    ) -> Status: ...
    def prep_transfer_descriptor_list(
        self,
        agent_name: str,
        descs: TransferDescriptorList,
        backend: Sequence[int] = [],
    ) -> int: ...
    def make_transfer_request(
        self,
        operation: TransferOpType,
        local_side: int,
        local_indices: Sequence[int],
        remote_side: int,
        remote_indices: Sequence[int],
        notif_msg: str = "",
        backend: Sequence[int] = [],
        skip_desc_merg: bool = False,
    ) -> int: ...
    def create_transfer_request(
        self,
        operation: TransferOpType,
        local_descs: TransferDescriptorList,
        remote_descs: TransferDescriptorList,
        remote_agent: str,
        notif_msg: str = "",
        backend: Sequence[int] = [],
    ) -> int: ...
    def post_transfer_request(
        self, request_handle: int, notif_msg: str = ""
    ) -> Status: ...
    def get_transfer_status(self, request_handle: int) -> Status: ...
    def get_transfer_telemetry(
        self, request_handle: int
    ) -> TransferTelemetry: ...
    def query_transfer_backend(self, request_handle: int) -> int: ...
    def release_transfer_request(self, request_handle: int) -> Status: ...
    def release_descriptor_list_handle(self, handle: int) -> Status: ...
    def get_notifs(
        self, backends: Sequence[int] = []
    ) -> dict[str, list[bytes]]:
        """
        Returns a map from agent name to a list of notifications received from that agent.

        Optionally, a list of backends can be mentioned to only get those backends' notifications.
        """

    def gen_notif(
        self, remote_agent: str, msg: str, backends: Sequence[int] = []
    ) -> Status: ...
    def get_local_metadata(self) -> bytes: ...
    def get_local_partial_metadata(
        self,
        descs: RegistrationDescriptorList,
        inc_conn_info: bool = False,
        backends: Sequence[int] = [],
    ) -> bytes: ...
    def load_remote_metadata(self, agent_metadata: bytes) -> bytes: ...
    def invalidate_remote_metadata(self, remote_name: str) -> Status: ...
    def send_local_metadata(self, ip_addr: str = "", port: int = 0) -> None: ...
    def send_local_partial_metadata(
        self,
        descs: RegistrationDescriptorList,
        inc_conn_info: bool = False,
        backends: Sequence[int] = [],
        ip_addr: str = "",
        port: int = 0,
    ) -> None: ...
    def fetch_remote_metadata(
        self, remote_name: str, ip_addr: str = "", port: int = 0
    ) -> None: ...
    def invalidate_local_metadata(
        self, ip_addr: str = "", port: int = 0
    ) -> None: ...
    def check_remote_metadata(
        self, remote_name: str, descs: TransferDescriptorList
    ) -> Status: ...
