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

"""Extensions for `std.benchmark` to work with accelerator `DeviceContext`s."""


from std.os import abort
from std.benchmark import (
    Bench,
    ThroughputMeasure,
    Bencher,
    BenchId,
    BenchmarkInfo,
    Report,
)
from max.gpu.host import DeviceContext
from max.algorithm import sync_parallelize


# ===----------------------------------------------------------------------=== #
# Bench extensions
# ===----------------------------------------------------------------------=== #


@always_inline
def bench_multicontext[
    FuncType: def(mut Bencher, DeviceContext, Int) raises -> None,
](
    mut self_: Bench,
    func: FuncType,
    list_of_ctx: List[DeviceContext],
    bench_id: BenchId,
    measures: List[ThroughputMeasure] = {},
) raises:
    """Benchmarks or Tests an input function across multiple device contexts.

    The metric returned represents the *slowest* performing device.

    Parameters:
        FuncType: The body function type.

    Args:
        self_: The benchmark configuration.
        func: The closure carrying the captured state of the body function.
        list_of_ctx: A list of device contexts on which the bench_fn is run in parallel.
        bench_id: The benchmark Id object used for identification.
        measures: Optional arg used to represent a list of ThroughputMeasure's.

    Raises:
        If the operation fails.
    """

    var num_ctxs = len(list_of_ctx)
    assert num_ctxs > 1, "list_of_ctx must contain at least 2 DeviceContexts"
    # Necessary to fill this List w/ default BenchmarkInfo otherwise each
    # thread attempts to free uninitialized BenchmarkInfo when copying below.
    var default_info = BenchmarkInfo(
        name="",
        result=Report(),
        measures=List[ThroughputMeasure](),
    )
    var results_b = List[BenchmarkInfo](length=num_ctxs, fill=default_info)

    # This closure runs in parallel on the host, 1 host thread per context.
    def per_gpu(i: Int) raises {mut results_b, imm}:
        def context_closure(mut b: Bencher) raises {imm}:
            func(b, list_of_ctx[i], i)

        var b = Bench()
        b.bench_function(context_closure, bench_id, measures)
        results_b[i] = b.info_vec[0].copy()

    sync_parallelize(per_gpu, num_ctxs)

    # Collect and print the worst-case GPU time.
    var max_time = 0.0
    var max_loc = 0

    for i in range(num_ctxs):
        var val = results_b[i].result.mean()
        if val > max_time:
            max_time = val
            max_loc = i

    self_.info_vec.append(results_b[max_loc].copy())


# ===----------------------------------------------------------------------=== #
# Bencher extensions
# ===----------------------------------------------------------------------=== #


def bencher_iter_custom[
    FuncType: def(DeviceContext) raises -> None,
](mut self_: Bencher, func: FuncType, ctx: DeviceContext):
    """Times a target GPU closure with custom number of iterations via DeviceContext ctx.

    Parameters:
        FuncType: The target GPU kernel launch closure type.

    Args:
        self_: The bencher state.
        func: The closure carrying the captured state of the kernel launch.
        ctx: The GPU DeviceContext for launching kernel.

    Notes:

    Nested launch closures that capture benchmark-local state are closure
    values. This value-taking overload forwards the closure to
    `DeviceContext.execution_time()` so `FuncType` carries the captured
    state.
    """

    try:
        self_.elapsed = ctx.execution_time(func, self_.num_iters)
    except e:
        abort(String(e))


def bencher_iter_custom[
    FuncType: def(DeviceContext, Int) raises -> None,
](mut self_: Bencher, func: FuncType, ctx: DeviceContext):
    """Times a target GPU closure with custom number of iterations via DeviceContext ctx.

    Parameters:
        FuncType: The target GPU kernel launch closure type.

    Args:
        self_: The bencher state.
        func: The closure carrying the captured state of the kernel launch.
        ctx: The GPU DeviceContext for launching kernel.

    Notes:

    Nested launch closures that capture benchmark-local state are closure
    values. This value-taking overload forwards the closure to
    `DeviceContext.execution_time_iter()` so `FuncType` carries the
    captured state.
    """

    try:
        self_.elapsed = ctx.execution_time_iter(func, self_.num_iters)
    except e:
        abort(String(e))
