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

from collections.abc import Sequence

import max._core.dtype

def make_dlpack_capsule(
    owner: object,
    data: int,
    device_type: int,
    device_id: int,
    dtype: max._core.dtype.DType,
    shape: Sequence[int],
    strides: Sequence[int] | None,
    max_version: tuple[int, int] | None,
) -> object: ...

class DLPackImport:
    @property
    def data(self) -> int: ...
    @property
    def byte_offset(self) -> int: ...
    @property
    def dtype(self) -> max._core.dtype.DType: ...
    @property
    def shape(self) -> list[int]: ...
    @property
    def strides(self) -> list[int] | None: ...

def import_dlpack(producer: object, stream: int | None) -> DLPackImport: ...
