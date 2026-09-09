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

"""Integration test for file URI support in chat completions."""

import os
import tempfile
from io import BytesIO
from pathlib import Path

import pytest
from async_asgi_testclient import TestClient
from fastapi import FastAPI
from max.driver import DeviceSpec
from max.pipelines import PipelineArgs
from max.pipelines.lib import KVCacheConfig, PipelineRuntimeConfig
from PIL import Image

pipeline_config = PipelineArgs(
    model_path="OpenGVLab/InternVL3-1B-Instruct",
    device_specs=[DeviceSpec.accelerator()],
    quantization_encoding="bfloat16",
    trust_remote_code=True,
    max_length=512,
    kv_cache=KVCacheConfig(),
    runtime=PipelineRuntimeConfig(max_batch_size=1),
)


def create_test_image_bytes() -> bytes:
    """Create a minimal test image and return it as JPEG bytes."""
    # Create a simple 10x10 red image
    img = Image.new("RGB", (10, 10), color="red")
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config",
    [pipeline_config],
    indirect=True,
)
@pytest.mark.parametrize(
    "settings_config",
    [
        {
            "MAX_SERVE_USE_HEARTBEAT": True,
            "MAX_SERVE_ALLOWED_IMAGE_ROOTS": ["/tmp"],
        }
    ],
    indirect=True,
)
async def test_chat_completion_with_file_uri(
    app: FastAPI, tmp_path: Path
) -> None:
    """Test chat completion with file:// URI for images."""
    # Create a valid test image file in /tmp (which is in allowed roots)
    image_path = tmp_path / "test_image.jpg"
    image_path.write_bytes(create_test_image_bytes())

    async with TestClient(app, timeout=720.0) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "OpenGVLab/InternVL3-1B-Instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What's in this image?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"file://{image_path}"},
                            },
                        ],
                    }
                ],
                "stream": False,
                "top_k": 1,
                "max_tokens": 3,
            },
        )

        # Should succeed with 200 OK.
        assert response.status_code == 200
        result = response.json()
        assert "choices" in result
        assert len(result["choices"]) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config",
    [pipeline_config],
    indirect=True,
)
@pytest.mark.parametrize(
    "settings_config",
    [
        {
            "MAX_SERVE_USE_HEARTBEAT": True,
            "MAX_SERVE_ALLOWED_IMAGE_ROOTS": [],  # No allowed roots
        }
    ],
    indirect=True,
)
async def test_file_uri_security_check(app: FastAPI, tmp_path: Path) -> None:
    """Test that file URIs are rejected when no roots are allowed."""
    # Create a valid test image file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(create_test_image_bytes())
        outside_path = f.name

    try:
        async with TestClient(app, timeout=60.0) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "OpenGVLab/InternVL3-1B-Instruct",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "What's in this image?",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"file://{outside_path}"
                                    },
                                },
                            ],
                        }
                    ],
                    "stream": False,
                    "top_k": 1,
                    "max_tokens": 3,
                },
            )

            # Should fail with error
            assert response.status_code == 400  # ValueError becomes 400
            error = response.json()
            # The API returns a generic "Value error." for security reasons
            assert error["error"]["message"] == "Value error."
    finally:
        os.unlink(outside_path)
