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
"""Test KernelLibrary.has_shape_function."""

import os
from pathlib import Path

import pytest
from max.graph import Graph


@pytest.fixture
def kernel_verification_ops_path() -> Path:
    return Path(os.environ["MODULAR_KERNEL_VERIFICATION_OPS_PATH"])


def test_has_shape_function(kernel_verification_ops_path: Path) -> None:
    with Graph("test_has_shape_function") as graph:
        kernels = graph._kernel_library
        kernels.add_path(kernel_verification_ops_path)

        # `my_add` registers a shape function; `op_with_int_parameter` does
        # not.
        assert kernels.has_shape_function("my_add") is True
        assert kernels.has_shape_function("op_with_int_parameter") is False
        with pytest.raises(KeyError, match="no_such_kernel_zzz"):
            kernels.has_shape_function("no_such_kernel_zzz")
