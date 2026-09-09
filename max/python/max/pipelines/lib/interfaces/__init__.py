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
"""Interfaces for MAX pipelines."""

from max.pipelines.lib.interfaces.batch_processor import (
    BatchProcessor,
    BatchProcessorRuntime,
    ModuleV3SingleReplicaBatchProcessor,
    RaggedBatchProcessor,
    SingleReplicaRaggedBatchProcessor,
    build_single_replica_ragged_token_arrays,
    process_ragged_kv_outputs,
    ragged_kv_symbolic_inputs,
    single_replica_context_batch,
)

from .arch_config import (
    ArchConfig,
    ArchConfigWithAttentionKVCache,
    ArchConfigWithBoundedMaxSeqLen,
    ArchConfigWithKVCache,
    ArchConfigWithPermissiveMaxSeqLen,
    ArchConfigWithStoredKVParams,
    ArchConfigWithVisionCache,
    ArchVLConfigWithTextSubconfig,
    arch_has_vision_tower,
)
from .generate import GenerateMixin
from .pipeline_model import (
    AlwaysSignalBuffersMixin,
    GraphPipelineModel,
    GraphPipelineModelWithKVCache,
    ModelInputs,
    ModelOutputs,
    ModuleV3MultiGraphPipelineModelWithKVCache,
    ModuleV3PipelineModel,
    ModuleV3PipelineModelWithKVCache,
    MultiGraphPipelineModelWithKVCache,
    PipelineModel,
    PipelineModelWithKVCache,
    UnifiedEagleOutputs,
    UnifiedSpecDecodeInputs,
)

__all__ = [
    "AlwaysSignalBuffersMixin",
    "ArchConfig",
    "ArchConfigWithAttentionKVCache",
    "ArchConfigWithBoundedMaxSeqLen",
    "ArchConfigWithKVCache",
    "ArchConfigWithPermissiveMaxSeqLen",
    "ArchConfigWithStoredKVParams",
    "ArchConfigWithVisionCache",
    "ArchVLConfigWithTextSubconfig",
    "BatchProcessor",
    "BatchProcessorRuntime",
    "GenerateMixin",
    "GraphPipelineModel",
    "GraphPipelineModelWithKVCache",
    "ModelInputs",
    "ModelOutputs",
    "ModuleV3MultiGraphPipelineModelWithKVCache",
    "ModuleV3PipelineModel",
    "ModuleV3PipelineModelWithKVCache",
    "ModuleV3SingleReplicaBatchProcessor",
    "MultiGraphPipelineModelWithKVCache",
    "PipelineModel",
    "PipelineModelWithKVCache",
    "RaggedBatchProcessor",
    "SingleReplicaRaggedBatchProcessor",
    "UnifiedEagleOutputs",
    "UnifiedSpecDecodeInputs",
    "arch_has_vision_tower",
    "build_single_replica_ragged_token_arrays",
    "process_ragged_kv_outputs",
    "ragged_kv_symbolic_inputs",
    "single_replica_context_batch",
]
