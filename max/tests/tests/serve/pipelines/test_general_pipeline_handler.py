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
"""Tests for GeneralPipelineHandler."""

import asyncio
from collections.abc import AsyncGenerator
from unittest.mock import Mock

import numpy as np
from max.pipelines.context import GenerationStatus
from max.pipelines.context.outputs import GenerationOutput
from max.pipelines.modeling.types import RequestID
from max.pipelines.request import (
    OpenResponsesRequest,
    OpenResponsesRequestBody,
)
from max.pipelines.request.open_responses import OutputImageContent
from max.serve.pipelines.general_handler import GeneralPipelineHandler


class MockGeneralPipelineHandler(GeneralPipelineHandler):
    """Mock implementation of GeneralPipelineHandler for testing."""

    def __init__(self, mock_chunks: list[GenerationOutput]) -> None:
        # Skip the parent constructor that requires real dependencies
        self.model_name = "test-model"
        self.logger = Mock()
        self.debug_logging = False
        self._mock_chunks = mock_chunks

    async def next(
        self, request: OpenResponsesRequest
    ) -> AsyncGenerator[GenerationOutput, None]:
        """Mock implementation that yields predefined chunks."""
        for chunk in self._mock_chunks:
            yield chunk


def create_test_request() -> OpenResponsesRequest:
    """Create a test OpenResponsesRequest."""
    body = OpenResponsesRequestBody(
        model="test-model",
        input="Test prompt for generation",
    )
    return OpenResponsesRequest(
        request_id=RequestID("test-request-1"),
        body=body,
    )


def test_next_single_chunk() -> None:
    """Test next with a single chunk."""
    # Create test image data
    pixel_data = np.array([[1, 2, 3]], dtype=np.uint8)

    chunks = [
        GenerationOutput(
            request_id=RequestID("test-request-1"),
            final_status=GenerationStatus.END_OF_SEQUENCE,
            output=[OutputImageContent.from_numpy(pixel_data, format="png")],
        ),
    ]

    pipeline = MockGeneralPipelineHandler(chunks)
    request = create_test_request()

    # Collect chunks
    async def collect_chunks() -> list[GenerationOutput]:
        result = []
        async for chunk in pipeline.next(request):
            result.append(chunk)
        return result

    streamed_chunks = asyncio.run(collect_chunks())

    # Verify
    assert len(streamed_chunks) == 1
    assert streamed_chunks[0].is_done is True
    assert len(streamed_chunks[0].output) == 1

    # Narrow type for mypy
    output_content = streamed_chunks[0].output[0]
    assert isinstance(output_content, OutputImageContent)
    assert output_content.format == "png"


def test_next_multiple_chunks() -> None:
    """Test next with multiple chunks."""
    chunks = [
        GenerationOutput(
            request_id=RequestID("test-request-1"),
            final_status=GenerationStatus.ACTIVE,
            output=[
                OutputImageContent.from_numpy(
                    np.array([[1]], dtype=np.uint8), format="png"
                )
            ],
        ),
        GenerationOutput(
            request_id=RequestID("test-request-1"),
            final_status=GenerationStatus.ACTIVE,
            output=[
                OutputImageContent.from_numpy(
                    np.array([[2]], dtype=np.uint8), format="png"
                )
            ],
        ),
        GenerationOutput(
            request_id=RequestID("test-request-1"),
            final_status=GenerationStatus.END_OF_SEQUENCE,
            output=[
                OutputImageContent.from_numpy(
                    np.array([[3]], dtype=np.uint8), format="png"
                )
            ],
        ),
    ]

    pipeline = MockGeneralPipelineHandler(chunks)
    request = create_test_request()

    # Collect chunks
    async def collect_chunks() -> list[GenerationOutput]:
        result = []
        async for chunk in pipeline.next(request):
            result.append(chunk)
        return result

    streamed_chunks = asyncio.run(collect_chunks())

    # Verify all chunks were streamed
    assert len(streamed_chunks) == 3
    assert streamed_chunks[0].final_status == GenerationStatus.ACTIVE
    assert streamed_chunks[1].final_status == GenerationStatus.ACTIVE
    assert streamed_chunks[2].final_status == GenerationStatus.END_OF_SEQUENCE
    assert streamed_chunks[2].is_done is True


def test_next_empty() -> None:
    """Test next with no chunks."""
    pipeline = MockGeneralPipelineHandler([])
    request = create_test_request()

    # Collect chunks
    async def collect_chunks() -> list[GenerationOutput]:
        result = []
        async for chunk in pipeline.next(request):
            result.append(chunk)
        return result

    streamed_chunks = asyncio.run(collect_chunks())

    # Verify no chunks were returned
    assert len(streamed_chunks) == 0


def test_next_streaming() -> None:
    """Test that next properly streams chunks."""
    chunks = [
        GenerationOutput(
            request_id=RequestID("test-request-1"),
            final_status=GenerationStatus.ACTIVE,
            output=[
                OutputImageContent.from_numpy(
                    np.array([[1]], dtype=np.uint8), format="png"
                )
            ],
        ),
        GenerationOutput(
            request_id=RequestID("test-request-1"),
            final_status=GenerationStatus.END_OF_SEQUENCE,
            output=[
                OutputImageContent.from_numpy(
                    np.array([[2]], dtype=np.uint8), format="png"
                )
            ],
        ),
    ]

    pipeline = MockGeneralPipelineHandler(chunks)
    request = create_test_request()

    # Collect all chunks from streaming
    async def collect_chunks() -> list[GenerationOutput]:
        result = []
        async for chunk in pipeline.next(request):
            result.append(chunk)
        return result

    streamed_chunks = asyncio.run(collect_chunks())

    # Verify all chunks were streamed
    assert len(streamed_chunks) == 2
    assert streamed_chunks[0].final_status == GenerationStatus.ACTIVE
    assert streamed_chunks[1].final_status == GenerationStatus.END_OF_SEQUENCE
    assert streamed_chunks[1].is_done is True
