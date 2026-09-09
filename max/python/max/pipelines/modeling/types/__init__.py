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

"""Pipeline modeling types: request/input/pipeline interfaces."""

from collections.abc import Callable

from max.pipelines.context.logit_processors_type import (
    BatchLogitsProcessor,
    BatchProcessorInputs,
    LogitsProcessor,
    ProcessorInputs,
)
from max.pipelines.request import (
    DUMMY_REQUEST_ID,
    OpenResponsesRequest,
    Request,
    RequestID,
    RequestType,
)

from .pipeline import (
    Pipeline,
    PipelineInputs,
    PipelineInputsType,
    PipelineOutput,
    PipelineOutputsDict,
    PipelineOutputType,
)
from .pipeline_variants import (
    AudioGenerationInputs,
    BatchType,
    CompletedBatchStats,
    EmbeddingsContext,
    EmbeddingsGenerationContextType,
    EmbeddingsGenerationInputs,
    EmbeddingsGenerationOutput,
    ImageContentPart,
    MessageContent,
    PixelGenerationInputs,
    TextContentPart,
    TextGenerationInputs,
    TextGenerationRequest,
    TextGenerationRequestFunction,
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
    VideoContentPart,
)
from .reasoning import (
    ParsedReasoningDelta,
    ReasoningParser,
    ReasoningPipelineTokenizer,
    ReasoningSpan,
)
from .task import InputModality, PipelineTask
from .tokenizer import PipelineTokenizer, TokenizerEncoded, UnboundContextType
from .tool_parsing import (
    ParsedToolCall,
    ParsedToolCallDelta,
    ParsedToolResponse,
    ToolParser,
)
from .utils import (
    SharedMemoryArray,
    msgpack_numpy_decoder,
    msgpack_numpy_encoder,
    msgpack_numpy_oob_decoder,
    msgpack_numpy_oob_encoder,
)

PipelinesFactory = Callable[
    [], Pipeline[PipelineInputsType, PipelineOutputType]
]

__all__ = [
    "DUMMY_REQUEST_ID",
    "AudioGenerationInputs",
    "BatchLogitsProcessor",
    "BatchProcessorInputs",
    "BatchType",
    "CompletedBatchStats",
    "EmbeddingsContext",
    "EmbeddingsGenerationContextType",
    "EmbeddingsGenerationInputs",
    "EmbeddingsGenerationOutput",
    "ImageContentPart",
    "InputModality",
    "LogitsProcessor",
    "MessageContent",
    "OpenResponsesRequest",
    "ParsedReasoningDelta",
    "ParsedToolCall",
    "ParsedToolCallDelta",
    "ParsedToolResponse",
    "Pipeline",
    "PipelineInputs",
    "PipelineInputsType",
    "PipelineOutput",
    "PipelineOutputType",
    "PipelineOutputsDict",
    "PipelineTask",
    "PipelineTokenizer",
    "PipelinesFactory",
    "PixelGenerationInputs",
    "ProcessorInputs",
    "ReasoningParser",
    "ReasoningPipelineTokenizer",
    "ReasoningSpan",
    "Request",
    "RequestID",
    "RequestType",
    "SharedMemoryArray",
    "TextContentPart",
    "TextGenerationInputs",
    "TextGenerationRequest",
    "TextGenerationRequestFunction",
    "TextGenerationRequestMessage",
    "TextGenerationRequestTool",
    "TokenizerEncoded",
    "ToolParser",
    "UnboundContextType",
    "VideoContentPart",
    "msgpack_numpy_decoder",
    "msgpack_numpy_encoder",
    "msgpack_numpy_oob_decoder",
    "msgpack_numpy_oob_encoder",
]
