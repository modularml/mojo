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
"""Helper for test_compile_pool: die hard with a compile still in flight.

Prints a worker's pid, then ``os._exit``s — no atexit hooks, no executor
shutdown, simulating a crashed parent. The test asserts the worker does
not outlive us (the fork server exits on our death, and the worker's
``PR_SET_PDEATHSIG`` fires when the fork server goes).
"""

import os
import time

from max.driver import DeviceSpec
from max.dtype import DType
from max.experimental.compile_pool import ProcessCompilePool
from max.graph import DeviceRef, Graph, TensorType

with Graph(
    "addself",
    input_types=[TensorType(DType.float32, [4], device=DeviceRef.CPU())],
) as g:
    (x,) = g.inputs
    g.output(x.tensor + x.tensor)

pool = ProcessCompilePool(device_specs=[DeviceSpec.cpu()])
pool.compile(g)
# Workers are created lazily on submit; wait for one to exist.
deadline = time.monotonic() + 240
while not pool._executor._processes and time.monotonic() < deadline:
    time.sleep(0.05)
(worker_pid,) = pool._executor._processes.keys()
print(worker_pid, flush=True)
os._exit(0)
