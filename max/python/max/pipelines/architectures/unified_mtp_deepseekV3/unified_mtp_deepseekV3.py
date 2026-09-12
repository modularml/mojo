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
"""DeepseekV3 with MTP: the NextN draft wired to the spec-decode driver."""

from __future__ import annotations

from max.nn.transformer import ReturnHiddenStates
from max.pipelines.speculative.config import SpeculativeConfig
from max.pipelines.speculative.driver import SequentialDriver
from max.pipelines.speculative.spec_input_types import SpecDecodeInputTypeSpec

from ..deepseekV3.deepseekV3 import DeepseekV3
from ..deepseekV3.model_config import DeepseekV3Config
from ..deepseekV3.spec_adapters import DeepseekV3MLAProposer, DeepseekV3Target
from ..deepseekV3_nextn.deepseekV3_nextn import DeepseekV3NextN
from ..deepseekV3_nextn.model_config import DeepseekV3NextNConfig


class MTPDeepseekV3Proposer(DeepseekV3MLAProposer):
    """The NextN head, proposing one token per step."""

    split_prefix = "mtp"
    carry_dim_prefix = "mtp_step"
    # Steps 1..K use LAST_PER_DEVICE. The underlying LAST path's internal
    # allgather acts as a collective fence between successive draft
    # invocations -- without it, the draft's EP/MoE dispatch leaves pending
    # async work that surfaces `mgp.sync` on the 2nd decoder call during CUDA
    # graph capture. LAST_PER_DEVICE returns the post-allgather full-batch
    # tensors, so the driver slices each replica's rows back out under DP.
    step_hidden_mode = ReturnHiddenStates.LAST_PER_DEVICE
    uses_thinking_phase = True
    draft_takes_ep_inputs = True


class UnifiedMTPDeepseekV3(SequentialDriver):
    """Fused nn.Module: merge + target forward + rejection + shift.

    The loop lives in :class:`SequentialDriver`; this class only names the
    target, the draft and the two adapters that call them.
    """

    target: DeepseekV3
    draft: DeepseekV3NextN

    def __init__(
        self,
        config: DeepseekV3Config,
        draft_config: DeepseekV3NextNConfig | None = None,
        speculative_config: SpeculativeConfig | None = None,
        enable_structured_output: bool = False,
    ) -> None:
        assert draft_config is not None
        target = DeepseekV3(config)
        draft = DeepseekV3NextN(draft_config)
        super().__init__(
            DeepseekV3Target(target),
            MTPDeepseekV3Proposer(draft),
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
