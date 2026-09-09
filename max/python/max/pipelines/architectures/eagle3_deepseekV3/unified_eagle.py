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
"""Eagle3 + DeepseekV3: the MLA draft wired to the spec-decode driver."""

from __future__ import annotations

from max.nn.transformer import ReturnHiddenStates
from max.pipelines.speculative.config import SpeculativeConfig
from max.pipelines.speculative.driver import SequentialDriver
from max.pipelines.speculative.spec_input_types import SpecDecodeInputTypeSpec

from ..deepseekV3.deepseekV3 import DeepseekV3
from ..deepseekV3.model_config import DeepseekV3Config
from ..deepseekV3.spec_adapters import DeepseekV3MLAProposer, DeepseekV3Target
from .eagle3_draft import Eagle3DeepseekV3


class Eagle3DeepseekV3Proposer(DeepseekV3MLAProposer):
    """The Eagle3 draft, proposing one token per step."""

    split_prefix = "eagle3"
    carry_dim_prefix = "draft_step"
    # Steps 1..K run in decode mode (one token per batch element), where
    # ALL-hs == LAST-hs. ALL returns per-device hidden states directly,
    # avoiding the LAST path's allgather, and needs no per-replica slice.
    step_hidden_mode = ReturnHiddenStates.ALL


class Eagle3DeepseekV3Unified(SequentialDriver):
    """Fused nn.Module: merge + target forward + rejection + shift.

    The target returns concatenated hidden states from 3 intermediate layers
    (first, middle, last); the draft fuses these via ``fc`` and generates the
    next speculative token. The loop itself lives in
    :class:`SequentialDriver`.
    """

    target: DeepseekV3
    draft: Eagle3DeepseekV3

    def __init__(
        self,
        config: DeepseekV3Config,
        draft_config: DeepseekV3Config | None = None,
        speculative_config: SpeculativeConfig | None = None,
        enable_structured_output: bool = False,
    ) -> None:
        assert draft_config is not None
        target = DeepseekV3(config)
        draft = Eagle3DeepseekV3(draft_config)
        super().__init__(
            DeepseekV3Target(target),
            Eagle3DeepseekV3Proposer(draft),
            target_model=target,
            draft_model=draft,
            devices=config.devices,
            data_parallel_degree=config.data_parallel_degree,
            input_spec=SpecDecodeInputTypeSpec(
                distributed=True,
                data_parallel_degree=config.data_parallel_degree,
            ),
            speculative_config=speculative_config,
            enable_structured_output=enable_structured_output,
        )
        self.config = config
