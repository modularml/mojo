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

from max.pipelines.lib import Speculator

from ..deepseekV3.arch import deepseekV3_arch
from .batch_processor import (
    Eagle3DeepseekV3BatchProcessor,
    Eagle3MHADeepseekV3BatchProcessor,
)
from .mha_pipeline import Eagle3MHADeepseekV3Model
from .model import Eagle3DeepseekV3Model

# Eagle3 with an MLA draft shipped as its own repo.
eagle3_deepseekV3_speculator = Speculator(
    name="Eagle3DeepseekV3ForCausalLM",
    base=deepseekV3_arch,
    draft_arch="Eagle3DeepseekV2ForCausalLM",
    method="eagle",
    pipeline_model=Eagle3DeepseekV3Model,
    batching=Eagle3DeepseekV3BatchProcessor,
    # TODO(MXSERV-7): Move ``austinpowers/...`` to the official Modular HF org
    # so CI doesn't depend on a personal account.
    example_repo_ids=["austinpowers/Kimi-K2.5-NVFP4-DeepseekV3"],
    opt_out_cascade=True,
)

# Eagle3 with a Llama-style MHA draft, which runs Kimi's fused graph.
eagle3_mha_deepseekV3_speculator = Speculator(
    name="Eagle3MHADeepseekV3ForCausalLM",
    base=deepseekV3_arch,
    draft_arch="LlamaForCausalLMEagle3",
    method="eagle",
    pipeline_model=Eagle3MHADeepseekV3Model,
    batching=Eagle3MHADeepseekV3BatchProcessor,
    example_repo_ids=["austinpowers/Kimi-K2.5-NVFP4-DeepseekV3"],
    opt_out_cascade=True,
)
