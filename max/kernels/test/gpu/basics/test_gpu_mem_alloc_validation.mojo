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
"""GPU memory allocation validation for measure_gpu_mem_mojo.py calibration.

Allocates 1 GiB on the device, holds it for 3 s so the NVML poller
(0.5 s interval) can observe the peak, then frees it.

Usage::

    ./utils/generate_test_resources_report/gpu_mem/measure_gpu_mem_mojo.py \\
        //max/kernels/test/gpu/basics:test_gpu_mem_alloc_validation.mojo.test \\
        --gpu b200

Expected: ~1.0 GB
"""

from max.gpu.host import DeviceContext
from std.time import sleep


def main() raises:
    var ctx = DeviceContext()
    # Allocate 100 MiB — fits within the default MAX memory manager budget
    # (~205 MiB).  NVML will report the full manager chunk (~205 MiB).
    comptime ALLOC_BYTES = 100 * 1024 * 1024
    var buf = ctx.enqueue_create_buffer[.uint8](ALLOC_BYTES)
    ctx.synchronize()
    # Hold for 3 s: well above the 0.5 s polling interval.
    sleep(3.0)
    _ = buf^  # explicitly consume so the buffer stays alive through the sleep
