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
"""Input batching for the fused Qwen3.5 MTP graph -- export only, no serving."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NoReturn

from max.driver import Buffer
from max.nn.kv_cache import KVCacheInputsInterface
from max.pipelines.context import TextContext

from ..qwen3_5.batch_processor import Qwen3_5BatchProcessor


class UnifiedMTPQwen3_5BatchProcessor(Qwen3_5BatchProcessor):
    """Compiles and exports the fused graph; refuses to drive a decode step.

    The other unified MTP architectures batch for MAX's own serving loop. This
    one is compiled to a MEF and executed by Mach's spec-step executor, which
    owns the state pools and supplies the whole live/shadow tail itself. MAX
    has no code that fills that tail, so serving here would fail deep inside
    the base processor -- on a state cache ``load_model`` never allocates --
    rather than at the boundary where the support actually stops.
    """

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> NoReturn:
        raise NotImplementedError(
            "UnifiedMTPQwen3_5 cannot be served by MAX: the fused graph's"
            " live and shadow state pools are owned by the serving engine, and"
            " MAX allocates neither. Export the graph to a MEF with"
            " mach/tools/gen-mef (--self-draft --spec-graph-name"
            " qwen3_5_with_mtp_graph) and serve it through Mach. To serve"
            " this checkpoint from MAX, drop --speculative-method mtp and it"
            " runs unspeculated on the base Qwen3_5 architecture."
        )
