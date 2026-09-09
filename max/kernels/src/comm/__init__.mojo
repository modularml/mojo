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
"""Provides communication primitives for multi-GPU workloads.

This package implements collective operations: allreduce, allgather,
reducescatter, broadcast, and scatter, along with the low-level
synchronization signals and Lamport-protocol primitives on which they are built.
Both NVIDIA and AMD GPU targets are supported.

These APIs are consumed by the model-serving layer when running distributed
inference across multiple GPUs within a node.
"""

from .allgather_rmsnorm import allgather_rmsnorm
from .lamport import (
    LamportGeneration,
    has_neg_zero,
    remove_neg_zero,
    set_neg_zero,
)
from .reducescatter_rmsnorm import reducescatter_rmsnorm
from .sync import Signal, MAX_GPUS, group_start, group_end
