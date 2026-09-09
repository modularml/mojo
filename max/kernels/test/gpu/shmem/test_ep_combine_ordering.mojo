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
"""Single-GPU regression guard for the EP-combine data-ready ordering fix.

Guards the device-scope acquire fence in
`EPCombineKernel.reduce_and_copy_to_output` (`shmem/ep_comm.mojo`): after the
AMD `volatile` data-ready poll, an acquire fence must re-establish the
release/acquire happens-before with the wait SM and invalidate the reduce SM's
vector L1. Without it the reduce SM reads stale peer data -> NaN.

This standalone kernel mirrors that same-GPU wait-SM -> reduce-SM handoff (no
rocshmem, no multi-GPU). Teeth: a producer CTA writes a payload then
RELEASE-stores the flag; oversubscribed consumer CTAs pre-touch a NaN-poisoned
slot (seeding a stale L1 line), volatile-poll, acquire-fence, then read. With
the fence removed the consumers read NaN ~55% of the time; with it, exact.
MI355X (gfx950) only.
"""

from std.atomic import Atomic, Ordering, fence
from max.gpu import WARP_SIZE, block_dim, block_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.testing import assert_equal
from std.utils.numerics import isnan, nan

comptime DEVICE_SCOPE = "agent"
comptime _flag_atomic = Atomic[Int32, scope=DEVICE_SCOPE]
comptime DATA_READY = Int32(1024)

comptime N_ITERS = 100
comptime N_CONSUMERS = 512  # oversubscribed vs CU count -> widens the window
comptime MSG_ELEMS = 256


def ordering_guard_kernel(
    payload: MutPointer[Float32, MutAnyOrigin],
    flags: MutPointer[Int32, MutAnyOrigin],
    output: MutPointer[Float32, MutAnyOrigin],
    scratch: MutPointer[Float32, MutAnyOrigin],
):
    """Block 0 is the producer (~ wait SM); blocks 1.. are consumers (~ reduce
    SMs), one payload slot each."""
    var tid = Int(thread_idx.x)
    var nthreads = Int(block_dim.x)

    if block_idx.x == 0:
        for c in range(N_CONSUMERS):
            var base = c * MSG_ELEMS
            for e in range(tid, MSG_ELEMS, nthreads):
                payload[base + e] = Float32(c)
            barrier()
            if tid == 0:
                _flag_atomic.store[ordering=Ordering.RELEASE](
                    flags + c, DATA_READY
                )
    else:
        var c = Int(block_idx.x) - 1
        if c >= N_CONSUMERS:
            return
        var base = c * MSG_ELEMS

        # Sink to scratch so the stale-L1 pre-touch isn't DCE'd.
        var seed = Float32(0)
        for e in range(tid, MSG_ELEMS, nthreads):
            seed += payload[base + e]
        if tid == 0:
            scratch[c] = seed

        if tid == 0:
            while flags.load[volatile=True](c) != DATA_READY:
                pass
            fence[ordering=Ordering.ACQUIRE, scope=DEVICE_SCOPE]()
        barrier()

        for e in range(tid, MSG_ELEMS, nthreads):
            output[base + e] = payload[base + e]


def main() raises:
    var ctx = DeviceContext()
    comptime total_elems = N_CONSUMERS * MSG_ELEMS
    var payload = ctx.enqueue_create_buffer[.float32](total_elems)
    var flags = ctx.enqueue_create_buffer[.int32](N_CONSUMERS)
    var output = ctx.enqueue_create_buffer[.float32](total_elems)
    var scratch = ctx.enqueue_create_buffer[.float32](N_CONSUMERS)
    var host_output = alloc[Float32](total_elems)

    var stale: Int = 0
    for _it in range(N_ITERS):
        # NaN so a stale read is detectable.
        ctx.enqueue_memset(payload, nan[.float32]())
        ctx.enqueue_memset(flags, Int32(0))
        ctx.enqueue_memset(output, Float32(0))

        ctx.enqueue_function[ordering_guard_kernel](
            payload,
            flags,
            output,
            scratch,
            grid_dim=N_CONSUMERS + 1,
            block_dim=WARP_SIZE,
        )

        ctx.enqueue_copy(host_output, output)
        ctx.synchronize()

        for c in range(N_CONSUMERS):
            var base = c * MSG_ELEMS
            for e in range(MSG_ELEMS):
                var got = host_output[base + e]
                if isnan(got) or got != Float32(c):
                    stale += 1

    host_output.free()

    assert_equal(
        stale,
        0,
        (
            "stale reads: device-scope acquire fence missing (ep_comm.mojo"
            " combine data-ready spin)"
        ),
    )
    print(
        "ep_combine ordering guard: 0 stale reads across",
        N_ITERS * total_elems,
        "reads",
    )
