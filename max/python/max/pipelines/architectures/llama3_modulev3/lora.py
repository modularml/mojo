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
"""ModuleV3 LoRA targets for the ModuleV3 Llama3."""

from __future__ import annotations

from max.pipelines.lora import LoRATargetModule

LLAMA3_LORA_TARGETS: tuple[LoRATargetModule, ...] = (
    LoRATargetModule("self_attn.qkv_proj", ("q_proj", "k_proj", "v_proj")),
    LoRATargetModule("self_attn.o_proj", ("o_proj",)),
)
