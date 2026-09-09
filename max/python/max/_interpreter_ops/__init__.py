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

"""Python bindings for the MO interpreter ops.

This module defines the operation handler registry and the graph-compiler-
backed op bindings for the MO graph interpreter.
"""

import time

from max import _eager_policy

from . import (
    band_part_gc,
    cast_gc,
    conv_gc,
    data_movement_gc,
    elementwise_binary_gc,
    gather_gc,
    gc_compile,
    group_norm_gc,
    layer_norm_gc,
    matmul_gc,
    nms_gc,
    nonzero_gc,
    pooling_gc,
    random_gc,
    range_gc,
    reduce_axis_gc,
    resize_gc,
    rms_norm_gc,
    roi_align_gc,
    scatter_gc,
    scatter_nd_gc,
    select_gc,
    shape_rearrange_gc,
    topk_gc,
    unary_elementwise_gc,
)

# Import handlers after the op modules to avoid circular import issues:
# handlers.py imports the op modules above (via the package).
# Re-export the warm-adoption query (from gc_compile) so a consumer can assert
# the ops were force-loaded from the manifest rather than cold-compiled.
from .gc_compile import adopted_from_manifest
from .handlers import _MO_OP_HANDLERS, lookup_handler, register_op_handler

# Every warm path iterates this; each ``*_gc.py`` self-registers at import
# (MXF-533), so there's no hand-maintained list to drift.
GC_FAMILIES: tuple[gc_compile.GCOpFamily, ...] = (
    gc_compile.registered_families()
)


def compile_all_families() -> None:
    """Compile every registered GC family's full sweep into the cache.

    Stamps once, after every family; a manifest force-load does not stamp.
    """
    swept = [family.compile_sweep() for family in GC_FAMILIES]
    if swept and all(swept):
        gc_compile.write_warm_stamp()


# Opt-in (MAX_EAGER_OP_PRECOMPILE=1) precompile of the full GC matrix; lazy
# per-dispatch otherwise (MXF-508).
def _precompile_gc_models() -> None:
    if not gc_compile.should_precompile():
        return
    provisioned = gc_compile.provisioned()
    if not _eager_policy.allow_lazy_compile() and not provisioned:
        # Fail at import rather than mid-request. (MXF-569)
        raise _eager_policy.EagerLazyCompileDisallowed(
            "MAX_EAGER_OP_PRECOMPILE=1 asks to compile the eager op matrix,"
            f" but {_eager_policy.ALLOW_LAZY_COMPILE_ENV_VAR}=0 forbids it and"
            " this machine has no warm cache to load.\n\n"
            "Run `max warm-interpreter-cache` on this machine."
        )
    will_compile = not provisioned
    if will_compile:
        _eager_policy.note_sweep_start()
    start = time.perf_counter()
    compile_all_families()
    if will_compile:
        _eager_policy.note_sweep_end(time.perf_counter() - start)


_precompile_gc_models()

__all__ = [
    "GC_FAMILIES",
    "_MO_OP_HANDLERS",
    "adopted_from_manifest",
    "compile_all_families",
    "lookup_handler",
    "register_op_handler",
]
