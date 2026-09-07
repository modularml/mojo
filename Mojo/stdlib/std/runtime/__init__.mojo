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
"""Runtime services: runtime initialization and worker-thread queries.

This package exposes `initialize_runtime()`, for initializing the runtime
explicitly when Mojo code built as a shared library is called from a non-Mojo
host program, and `parallelism_level()`, for querying how many worker threads
the runtime has available.

The async task primitives that back `parallelize()` and friends live in the
private `_asyncrt` module. Mojo's async and coroutine support is unfinished and
carries no stability guarantees, so it is deliberately not part of this
package's public API. Do not build async patterns on it yet.
"""

from std.builtin._startup import _ensure_runtime_init
from std.ffi import external_call


def initialize_runtime():
    """Initializes the global Mojo runtime if it is not already initialized.

    The Mojo runtime manages the thread pool used by parallel APIs such as
    `parallelize()`. Programs with a Mojo `main()` function initialize the
    runtime automatically at startup, so most programs never need to call this
    function.

    However, when Mojo code is compiled into a shared library (with
    `mojo build --emit shared-lib`) and called from a non-Mojo host program
    (such as C or C++), no Mojo `main()` function runs and the runtime is
    never initialized. In that case, call this function before using any API
    that depends on the runtime — for example, at the start of each function
    exported with `@export`. This function is idempotent and inexpensive when
    the runtime is already initialized.

    Initializing the runtime once covers all threads in the process. The
    runtime remains alive for the remainder of the process.

    Examples:

    ```mojo
    from max.algorithm import parallelize
    from std.runtime import initialize_runtime


    @export("fill_squares")
    def fill_squares(
        data: Pointer[Int64, MutUntrackedOrigin], len: Int
    ) abi("C"):
        initialize_runtime()

        def fill(i: Int):
            data.unsafe_store(i, Int64(i * i))

        parallelize(fill, len)
    ```
    """
    _ensure_runtime_init()


@always_inline
def parallelism_level() -> Int:
    """Gets the parallelism level of the Runtime.

    Returns:
        The number of worker threads available in the async runtime.
    """
    return Int(
        external_call[
            "KGEN_CompilerRT_AsyncRT_ParallelismLevel",
            Int32,
        ]()
    )
