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

"""Regression test for UCX plugin load inside a ``spawn`` subprocess.

The upstream NIXL plugin manager ``dlopen``s the CUDA-flavor ``libplugin_UCX.so``
with ``RTLD_NOW | RTLD_LOCAL``; the plugin's CUDA/NVML symbols must already be in
the process under ``RTLD_GLOBAL`` or the load fails and
``get_plugin_params("UCX")`` returns ``NIXL_ERR_NOT_FOUND``.

The test conftest pre-loads those libraries, but only in the MAIN pytest
process. A ``spawn``-ed child re-imports everything fresh and does NOT inherit
the parent's ``RTLD_GLOBAL`` handles, so the child must perform the preload
itself. ``KVTransferEngine`` does this in-library via
``preload_nixl_plugin_deps`` (``_nixl_plugin_deps.py``). This test stands up a
``KVTransferEngine`` inside a ``spawn`` child to prove that path works WITHOUT
relying on the conftest preload.

Without the in-library preload this child raises ``NIXL_ERR_NOT_FOUND`` while
constructing the engine; with it, the child exits 0.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
from max.driver import Accelerator
from max.driver.buffer import Buffer


def _construct_engine_in_child(result_queue: mp.Queue) -> None:  # type: ignore[type-arg]
    # NOTE: this runs in a spawn-ed interpreter. The transfer_engine conftest's
    # main-process RTLD_GLOBAL preload is NOT in effect here. Importing inside
    # the child mirrors the real serve worker, which is also spawn-ed.
    try:
        from max.dtype import DType
        from max.nn.kv_cache.cache_params import KVCacheMemory
        from max.pipelines.kv_cache import KVTransferEngine

        device = Accelerator(0)
        total_num_pages = 2
        blocks = Buffer.from_numpy(
            np.arange(total_num_pages * 4, dtype=np.int16)
        ).to(device)

        # KVTransferEngine.__init__ -> create_agent -> get_plugin_params("UCX")
        # which is the line that raises NIXL_ERR_NOT_FOUND if the plugin's CUDA
        # deps were not pre-loaded RTLD_GLOBAL in this process.
        bytes_per_page = (
            blocks.num_elements * blocks.dtype.size_in_bytes // total_num_pages
        )
        group = KVCacheMemory(
            replicated=False,
            buffers=[
                blocks.view(DType.uint8, [total_num_pages, bytes_per_page])
            ],
        )
        engine = KVTransferEngine(
            "spawn_preload_engine",
            [[group]],
        )
        engine.cleanup()
        result_queue.put(("ok", ""))
    except BaseException as exc:
        result_queue.put((type(exc).__name__, str(exc)))


def test_kv_transfer_engine_constructs_in_spawn_child() -> None:
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()  # type: ignore[type-arg]
    proc = ctx.Process(target=_construct_engine_in_child, args=(result_queue,))
    proc.start()
    proc.join(timeout=120)

    assert proc.exitcode is not None, "spawn child did not exit (hang)"
    status, detail = result_queue.get(timeout=5)
    assert status == "ok", (
        f"KVTransferEngine failed to construct in spawn child: "
        f"{status}: {detail}"
    )
    assert proc.exitcode == 0, f"spawn child exited non-zero: {proc.exitcode}"
