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

from max.graph.weights import WeightsFormat
from max.pipelines.lib import Speculator

from ..deepseekV3.arch import deepseekV3_arch
from .batch_processor import UnifiedMTPDeepseekV3BatchProcessor
from .model import UnifiedMTPDeepseekV3Model
from .weight_adapters import convert_with_mtp_state_dict

# MTP, whose NextN head is baked into the target checkpoint.
unified_mtp_deepseekV3_speculator = Speculator(
    name="UnifiedMTPDeepseekV3ForCausalLM",
    base=deepseekV3_arch,
    draft_arch=None,
    method="mtp",
    pipeline_model=UnifiedMTPDeepseekV3Model,
    batching=UnifiedMTPDeepseekV3BatchProcessor,
    weight_adapters={
        WeightsFormat.safetensors: convert_with_mtp_state_dict,
    },
    opt_out_cascade=True,
)
