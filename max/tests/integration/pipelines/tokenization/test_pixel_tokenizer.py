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
"""Tests for PixelGenerationTokenizer.

These tests require network access to HuggingFace and are marked as manual
in the BUILD.bazel file. They are not run in CI by default.

To run these tests manually:
    ./bazelw test //max/tests/integration/pipelines/tokenization:test_pixel_tokenizer
"""

from __future__ import annotations

import numpy as np
import pytest
from max.pipelines.lib import MAXModelConfig, PixelGenerationTokenizer
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from max.pipelines.modeling.types import RequestID
from max.pipelines.request import OpenResponsesRequest
from max.pipelines.request.open_responses import OpenResponsesRequestBody
from max.pipelines.request.provider_options import (
    ImageProviderOptions,
    ProviderOptions,
)


class TestPixelGenerationTokenizer:
    """Test suite for PixelGenerationTokenizer.

    These tests use the real Flux 2.0 model from HuggingFace and require
    network access. They are marked as manual to prevent running in CI.
    """

    @pytest.fixture
    def flux_model_path(self) -> str:
        """Flux 2.0 model path from HuggingFace."""
        return "black-forest-labs/FLUX.2-dev"

    @pytest.fixture
    def flux_pipeline_config(self, flux_model_path: str) -> PipelineConfig:
        """Pipeline config for Flux model."""
        return PipelineConfig(
            models=ModelManifest.from_model_path(flux_model_path),
            runtime=PipelineRuntimeConfig(),
        )

    @pytest.fixture
    def zimage_model_path(self) -> str:
        """Z-Image model path from HuggingFace."""
        return "Tongyi-MAI/Z-Image-Turbo"

    @pytest.fixture
    def zimage_pipeline_config(self, zimage_model_path: str) -> PipelineConfig:
        """Pipeline config for Z-Image model."""
        return PipelineConfig(
            models=ModelManifest.from_model_path(zimage_model_path),
            runtime=PipelineRuntimeConfig(),
        )

    def test_initialization_basic(
        self, flux_model_path: str, flux_pipeline_config: PipelineConfig
    ) -> None:
        """Test basic initialization of PixelGenerationTokenizer.

        Uses FLUX.2 which has a single text encoder (Mistral-Small-3.2-24B),
        verifying that delegate_2 is None when no secondary tokenizer is specified.
        """
        tokenizer = PixelGenerationTokenizer(
            model_path=flux_model_path,
            pipeline_config=flux_pipeline_config,
            subfolder="tokenizer",
            max_length=77,
        )

        assert tokenizer.model_path == flux_model_path
        assert tokenizer.max_length == 77
        assert tokenizer.delegate is not None
        assert tokenizer.delegate_2 is None
        assert tokenizer._pipeline_class_name == "Flux2Pipeline"

    def test_initialization_without_diffusion_manifest(
        self, flux_model_path: str
    ) -> None:
        """Test that initialization fails without a diffusion ModelManifest."""
        # Use a non-diffusion model (text-generation model) which won't have
        # diffusion metadata (_class_name) in its ModelManifest.
        non_diffusion_model = "gpt2"
        config = PipelineConfig(
            models=ModelManifest(
                {"main": MAXModelConfig(model_path=non_diffusion_model)}
            ),
            runtime=PipelineRuntimeConfig(),
        )

        with pytest.raises(
            ValueError, match="metadata is missing required key"
        ):
            PixelGenerationTokenizer(
                model_path=flux_model_path,
                pipeline_config=config,
                subfolder="tokenizer",
                max_length=2048,
            )

    def test_static_config_caching(
        self, flux_model_path: str, flux_pipeline_config: PipelineConfig
    ) -> None:
        """Test that static configuration values are cached during initialization."""
        tokenizer = PixelGenerationTokenizer(
            model_path=flux_model_path,
            pipeline_config=flux_pipeline_config,
            subfolder="tokenizer",
            max_length=2048,
        )

        # Verify cached values
        assert tokenizer._vae_scale_factor == 8
        assert tokenizer._default_sample_size == 128
        assert (
            tokenizer._num_channels_latents == 32
        )  # Flux2 has in_channels=128, so 128//4=32
        assert tokenizer._scheduler.base_image_seq_len == 256
        assert tokenizer._scheduler.max_image_seq_len == 4096
        assert tokenizer._scheduler.base_shift == 0.5
        assert tokenizer._scheduler.max_shift == 1.15
        # PixelGenerationTokenizer forces use_empirical_mu=True for FLUX2 and
        # FLUX2_KLEIN unconditionally (see pixel_tokenizer.py).
        assert tokenizer._scheduler._use_empirical_mu is True

    @pytest.mark.asyncio
    async def test_encode_primary_tokenizer(
        self, flux_model_path: str, flux_pipeline_config: PipelineConfig
    ) -> None:
        """Test encoding with the primary tokenizer."""
        tokenizer = PixelGenerationTokenizer(
            model_path=flux_model_path,
            pipeline_config=flux_pipeline_config,
            subfolder="tokenizer",
            max_length=77,
        )

        prompt = "A beautiful sunset over the ocean"
        token_ids, attention_mask = await tokenizer.encode(prompt)

        assert isinstance(token_ids, np.ndarray)
        assert isinstance(attention_mask, np.ndarray)
        assert token_ids.dtype == np.int64
        assert attention_mask.dtype == np.bool_
        # FLUX2 wraps the prompt with system/user chat-template messages
        # before tokenizing; the output length depends on the formatted
        # string and is bounded above by max_length but not padded to it.
        assert 0 < len(token_ids) <= 77
        assert len(attention_mask) == len(token_ids)

    @pytest.mark.asyncio
    async def test_new_context_flux(
        self, flux_model_path: str, flux_pipeline_config: PipelineConfig
    ) -> None:
        """Test creating a PixelContext for Flux model."""
        tokenizer = PixelGenerationTokenizer(
            model_path=flux_model_path,
            pipeline_config=flux_pipeline_config,
            subfolder="tokenizer",
            max_length=77,
        )

        body = OpenResponsesRequestBody(
            model="flux-dev",
            input="A majestic mountain landscape",
            seed=42,
            provider_options=ProviderOptions(
                image=ImageProviderOptions(
                    height=1024,
                    width=1024,
                    steps=50,
                    guidance_scale=3.5,
                )
            ),
        )
        request = OpenResponsesRequest(
            request_id=RequestID("test-request-1"), body=body
        )

        context = await tokenizer.new_context(request)

        # Verify context properties
        assert context.request_id == request.request_id
        assert context.height == 1024
        assert context.width == 1024
        assert context.num_inference_steps == 50
        assert context.guidance_scale == 3.5
        assert context.tokens is not None
        assert context.latents is not None
        assert context.timesteps is not None

        # Verify timesteps are normalized correctly for Flux (standard)
        assert np.all(context.timesteps >= 0.0)
        assert np.all(context.timesteps <= 1.0)

    @pytest.mark.asyncio
    async def test_new_context_zimage(
        self, zimage_model_path: str, zimage_pipeline_config: PipelineConfig
    ) -> None:
        """Test creating a PixelContext for Z-Image model with inverted timesteps."""
        tokenizer = PixelGenerationTokenizer(
            model_path=zimage_model_path,
            pipeline_config=zimage_pipeline_config,
            subfolder="tokenizer",
            max_length=77,
        )

        body = OpenResponsesRequestBody(
            model="z-image-turbo",
            input="A futuristic cityscape",
            seed=123,
            provider_options=ProviderOptions(
                image=ImageProviderOptions(
                    height=1024,
                    width=1024,
                    steps=8,
                    guidance_scale=3.5,
                )
            ),
        )
        request = OpenResponsesRequest(
            request_id=RequestID("test-request-2"), body=body
        )

        context = await tokenizer.new_context(request)

        # Verify Z-Image uses inverted timestep normalization
        assert context.timesteps is not None
        assert np.all(context.timesteps >= 0.0)
        assert np.all(context.timesteps <= 1.0)

        # Z-Image should have different timestep values than Flux
        # due to inverted normalization: (1000 - t) / 1000 vs t / 1000

    @pytest.mark.asyncio
    async def test_new_context_default_dimensions(
        self, flux_model_path: str, flux_pipeline_config: PipelineConfig
    ) -> None:
        """Test that default dimensions are computed correctly."""
        tokenizer = PixelGenerationTokenizer(
            model_path=flux_model_path,
            pipeline_config=flux_pipeline_config,
            subfolder="tokenizer",
            max_length=2048,
        )

        body = OpenResponsesRequestBody(
            model="flux-dev",
            input="Test prompt",
            provider_options=ProviderOptions(
                image=ImageProviderOptions(
                    steps=10,
                    # No height/width specified
                )
            ),
        )
        request = OpenResponsesRequest(
            request_id=RequestID("test-request-4"), body=body
        )

        context = await tokenizer.new_context(request)

        # Default: 128 * vae_scale_factor (8) = 1024
        assert context.height == 1024
        assert context.width == 1024

    @pytest.mark.asyncio
    async def test_postprocess(
        self, flux_model_path: str, flux_pipeline_config: PipelineConfig
    ) -> None:
        """Test pixel data postprocessing."""
        tokenizer = PixelGenerationTokenizer(
            model_path=flux_model_path,
            pipeline_config=flux_pipeline_config,
            subfolder="tokenizer",
            max_length=2048,
        )

        # Create mock pixel data (NCHW format, normalized to [-1, 1])
        pixel_data = np.random.randn(1, 3, 64, 64).astype(np.float32)

        processed = await tokenizer.postprocess(pixel_data)

        # postprocess only denormalizes from [-1, 1] to [0, 1]; it does not
        # change the layout. NCHW→NHWC happens later in the pipeline variant.
        assert processed.shape == pixel_data.shape
        assert np.all(processed >= 0.0)
        assert np.all(processed <= 1.0)

    def test_calculate_shift(
        self, flux_model_path: str, flux_pipeline_config: PipelineConfig
    ) -> None:
        """Test the shift calculation for timestep scheduling."""
        tokenizer = PixelGenerationTokenizer(
            model_path=flux_model_path,
            pipeline_config=flux_pipeline_config,
            subfolder="tokenizer",
            max_length=2048,
        )

        # Test with different image sequence lengths
        mu_small = tokenizer._scheduler._calculate_mu(256, 50)
        mu_large = tokenizer._scheduler._calculate_mu(4096, 50)

        # Mu should increase with sequence length. We don't pin specific
        # values: PixelGenerationTokenizer enables use_empirical_mu=True for
        # FLUX2, which uses an empirical formula rather than the linear
        # interpolation between base_shift and max_shift.
        assert mu_small < mu_large

    def test_prepare_latents(
        self, flux_model_path: str, flux_pipeline_config: PipelineConfig
    ) -> None:
        """Test latent preparation."""
        tokenizer = PixelGenerationTokenizer(
            model_path=flux_model_path,
            pipeline_config=flux_pipeline_config,
            subfolder="tokenizer",
            max_length=2048,
        )

        latents, latent_image_ids = tokenizer._prepare_latents(
            batch_size=2,
            num_channels_latents=16,
            latent_height=128,
            latent_width=128,
            seed=42,
        )

        # Verify latents shape
        assert latents.shape == (2, 16, 128, 128)
        assert latents.dtype == np.float32

        # FLUX2 uses 4D (T, H, W, L) coordinates with a leading batch dim;
        # other pipelines use 3D (zero, H, W) coordinates without batching.
        # FLUX2 builds the IDs from np.arange / np.array([0]) and stacks
        # without an explicit cast, so the result is int64 (unlike the
        # non-FLUX2 path which explicitly casts to float32).
        assert latent_image_ids.shape == (2, 64 * 64, 4)
        assert latent_image_ids.dtype == np.int64

    def test_prepare_latents_deterministic(
        self, flux_model_path: str, flux_pipeline_config: PipelineConfig
    ) -> None:
        """Test that latent preparation is deterministic with the same seed."""
        tokenizer = PixelGenerationTokenizer(
            model_path=flux_model_path,
            pipeline_config=flux_pipeline_config,
            subfolder="tokenizer",
            max_length=2048,
        )

        latents1, _ = tokenizer._prepare_latents(
            batch_size=1,
            num_channels_latents=16,
            latent_height=64,
            latent_width=64,
            seed=12345,
        )

        latents2, _ = tokenizer._prepare_latents(
            batch_size=1,
            num_channels_latents=16,
            latent_height=64,
            latent_width=64,
            seed=12345,
        )

        # Same seed should produce identical latents
        np.testing.assert_array_equal(latents1, latents2)

    @pytest.mark.asyncio
    async def test_decode_not_implemented(
        self, flux_model_path: str, flux_pipeline_config: PipelineConfig
    ) -> None:
        """Test that decode raises NotImplementedError."""
        tokenizer = PixelGenerationTokenizer(
            model_path=flux_model_path,
            pipeline_config=flux_pipeline_config,
            subfolder="tokenizer",
            max_length=2048,
        )

        dummy_encoded = (np.array([1, 2, 3]), np.array([True, True, True]))

        with pytest.raises(NotImplementedError):
            await tokenizer.decode(dummy_encoded)

    def test_properties(
        self, flux_model_path: str, flux_pipeline_config: PipelineConfig
    ) -> None:
        """Test tokenizer properties."""
        tokenizer = PixelGenerationTokenizer(
            model_path=flux_model_path,
            pipeline_config=flux_pipeline_config,
            subfolder="tokenizer",
            max_length=2048,
        )

        # Test eos_token_ids property
        assert tokenizer.eos_token_ids and all(
            isinstance(t, int) for t in tokenizer.eos_token_ids
        )

        # Test expects_content_wrapping property
        assert tokenizer.expects_content_wrapping is False
