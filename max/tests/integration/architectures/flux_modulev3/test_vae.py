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
"""Tests for the standalone unified ``Vae`` Module in ``flux_modulev3``.

Exercises the single-Module-with-two-forward-methods-and-shared-weights
pattern (MXF-460):

* Construction + structure (encoder, decoder, shared BN parameters).
* :meth:`Vae.forward` is wired to ``NotImplementedError``.
* :meth:`Vae.input_types` exposes only the encoder input.
* :meth:`Vae.adapt_state_dict` translates the four legacy key
  prefixes (encoder, quant_conv, decoder, post_quant_conv) plus the
  shared BN buffers, and drops everything else.
* A small wrapper Module embedding the ``Vae`` sees the shared BN
  parameters exactly once in its parameter tree, and compiling its
  ``forward`` traces both ``encode`` and ``decode`` into one graph --
  the headline demonstration of the pattern.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from max.driver import (
    Accelerator,
    DeviceSpec,
    accelerator_api,
    accelerator_count,
)
from max.dtype import DType
from max.experimental.nn import Module, module_dataclass
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType
from max.graph.weights import WeightData
from max.pipelines.architectures.flux_modulev3 import Vae
from max.pipelines.lib.weight_loader import dict_loader

# The compile test below needs an NVIDIA GPU: Conv2d's FCRS filter
# layout has no CPU implementation, and the FCRS path /
# ``flash_attention_gpu`` used in ``VAEAttention`` are CUDA-only -- the
# AMD ROCm Conv2d path crashes on these tiny shapes.  Pure structural /
# state-dict tests are CPU-friendly.
_REQUIRES_NVIDIA_GPU = pytest.mark.skipif(
    accelerator_count() == 0 or accelerator_api() != "cuda",
    reason=(
        "Encoder/decoder forward requires NVIDIA GPU (Conv2d FCRS layout +"
        " flash_attention_gpu are CUDA-only)"
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_flux2_vae_config() -> dict[str, Any]:
    """Two-block FLUX.2 VAE config sized for fast CPU unit tests."""
    return {
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
        "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D"],
        "block_out_channels": [16, 32],
        "layers_per_block": 1,
        "act_fn": "silu",
        "latent_channels": 4,
        "norm_num_groups": 8,
        "sample_size": 32,
        "scaling_factor": 1.0,
        "shift_factor": 0.0,
        "use_quant_conv": True,
        "use_post_quant_conv": True,
        "patch_size": (2, 2),
        "batch_norm_eps": 1e-4,
        "batch_norm_momentum": 0.1,
    }


def _fake_weight(name: str) -> WeightData:
    """Sentinel ``WeightData`` for verifying key translations.

    Pure key-translation tests don't need real tensor data; we just
    need a unique value to compare identity-preservation through the
    adapter.
    """
    return WeightData.from_numpy(np.zeros([1], dtype=np.float32), name)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def device_spec() -> DeviceSpec:
    """Picks accelerator if available, else CPU.

    Tests that only inspect structure (parameters, descendants, etc.)
    work on either; tests that compile the encoder or decoder require
    GPU because ``Conv2d`` has no CPU implementation of its FCRS
    filter layout.
    """
    if accelerator_count() > 0:
        return DeviceSpec.accelerator(0)
    return DeviceSpec.cpu()


@pytest.fixture
def vae(device_spec: DeviceSpec) -> Vae:
    """Eagerly-constructed ``Vae`` with zero-initialised weights.

    Function-scoped so each test gets a fresh ``Vae``.  Mojo kernel
    caching across tests keeps the per-test construction cost low
    after the first invocation, and per-test isolation avoids
    CUDA-state pollution that surfaces when GPU tests share a long-
    lived Module with structural-only tests.
    """
    return Vae(
        huggingface_config=_tiny_flux2_vae_config(),
        quantization_encoding="bfloat16",
        device_specs=[device_spec],
    )


# ---------------------------------------------------------------------------
# 1. Construction smoke
# ---------------------------------------------------------------------------


def test_vae_construction_smoke(vae: Vae) -> None:
    assert vae.encoder is not None
    assert vae.decoder is not None
    assert list(vae.bn_mean.shape) == [vae.num_channels]
    assert list(vae.bn_var.shape) == [vae.num_channels]
    assert vae.bn_mean.dtype == DType.bfloat16
    assert vae.bn_var.dtype == DType.bfloat16


# ---------------------------------------------------------------------------
# 2. num_channels property
# ---------------------------------------------------------------------------


def test_vae_num_channels(vae: Vae) -> None:
    # latent_channels=4 in the tiny config; 2x2 patchify multiplies by 4.
    assert vae.num_channels == 16


# ---------------------------------------------------------------------------
# 3. forward() raises
# ---------------------------------------------------------------------------


def test_vae_forward_raises_with_no_args(vae: Vae) -> None:
    with pytest.raises(NotImplementedError, match="encode"):
        vae.forward()


def test_vae_call_raises_with_no_args(vae: Vae) -> None:
    # __call__ dispatches through forward(); same NotImplementedError.
    with pytest.raises(NotImplementedError, match="decode"):
        vae()


# ---------------------------------------------------------------------------
# 4. input_types() exposes only the encoder input
# ---------------------------------------------------------------------------


def test_vae_input_types_encoder_only(vae: Vae) -> None:
    types = vae.input_types()
    assert len(types) == 1
    t = types[0]
    assert t.dtype == DType.uint8
    # ``(image_height, image_width, 3)`` -- the trailing 3 is concrete.
    assert len(t.shape) == 3


# ---------------------------------------------------------------------------
# 5. Shared BN parameters live only on the Vae root
# ---------------------------------------------------------------------------


def test_vae_shared_bn_parameters(vae: Vae) -> None:
    local_param_names = {name for name, _ in vae.local_parameters}
    # The Vae root owns exactly ``bn_mean`` and ``bn_var`` as direct
    # parameters; everything else lives under ``encoder.*`` / ``decoder.*``.
    assert local_param_names == {"bn_mean", "bn_var"}

    all_param_names = {name for name, _ in vae.parameters}
    # Encoder + decoder sub-trees must NOT carry their own copies of
    # bn_mean / bn_var -- the whole point of merging the two modules
    # was to avoid that duplication.
    assert "encoder.bn_mean" not in all_param_names
    assert "encoder.bn_var" not in all_param_names
    assert "decoder.bn_mean" not in all_param_names
    assert "decoder.bn_var" not in all_param_names
    # And we should see real encoder + decoder params, so the test is
    # actually exercising the sub-trees.
    assert any(n.startswith("encoder.") for n in all_param_names)
    assert any(n.startswith("decoder.") for n in all_param_names)


# ---------------------------------------------------------------------------
# 6-9. adapt_loader key translation
# ---------------------------------------------------------------------------


def test_vae_adapt_loader_encoder_keys() -> None:
    src = {
        "encoder.conv_in.weight": _fake_weight("encoder.conv_in.weight"),
        "quant_conv.weight": _fake_weight("quant_conv.weight"),
    }
    loader = Vae.adapt_loader(dict_loader(src))
    # Canonical Module names resolve to the source weights by identity.
    assert loader("encoder.conv_in.weight") is src["encoder.conv_in.weight"]
    # ``encoder.quant_conv.*`` queries route to the un-prefixed source.
    assert loader("encoder.quant_conv.weight") is src["quant_conv.weight"]


def test_vae_adapt_loader_decoder_keys() -> None:
    src = {
        "decoder.conv_out.weight": _fake_weight("decoder.conv_out.weight"),
        "post_quant_conv.weight": _fake_weight("post_quant_conv.weight"),
    }
    loader = Vae.adapt_loader(dict_loader(src))
    assert loader("decoder.conv_out.weight") is src["decoder.conv_out.weight"]
    assert (
        loader("decoder.post_quant_conv.weight")
        is src["post_quant_conv.weight"]
    )


def test_vae_adapt_loader_bn_keys() -> None:
    src = {
        "bn.running_mean": _fake_weight("bn.running_mean"),
        "bn.running_var": _fake_weight("bn.running_var"),
    }
    loader = Vae.adapt_loader(dict_loader(src))
    assert loader("bn_mean") is src["bn.running_mean"]
    assert loader("bn_var") is src["bn.running_var"]


def test_vae_adapt_loader_unmapped_queries_raise() -> None:
    # The adapter only translates the parameters the Module declares; it
    # never invents weights, so a query with no backing source key
    # raises ``KeyError`` (the Module simply never issues such queries).
    src = {"encoder.conv_in.weight": _fake_weight("encoder.conv_in.weight")}
    loader = Vae.adapt_loader(dict_loader(src))
    # A known canonical key resolves.
    assert loader("encoder.conv_in.weight") is src["encoder.conv_in.weight"]
    # Unknown / absent canonical names have no source backing.
    with pytest.raises(KeyError):
        loader("text_encoder.foo")
    with pytest.raises(KeyError):
        loader("bn_mean")  # bn.running_mean absent from this source


# ---------------------------------------------------------------------------
# 11-12. Headline test: parent Module sees the shared BN tensors exactly once
# ---------------------------------------------------------------------------


@module_dataclass
class _VaeWrapper(Module[..., Tensor]):
    """Test-only wrapper Module that embeds the unified ``Vae``.

    Models what a real parent (``FLUXModule``) does: own the ``Vae``
    as a sub-Module and invoke both :meth:`Vae.encode` and
    :meth:`Vae.decode` from a single ``forward``.  Because the parent
    owns one ``Vae``, the shared ``bn_mean`` / ``bn_var`` parameters
    appear once in the parent's parameter tree -- no checkpoint
    aliasing across encoder/decoder siblings required.
    """

    vae: Vae

    def forward(
        self,
        image: Tensor,
        latents: Tensor,
        h_carrier: Tensor,
        w_carrier: Tensor,
    ) -> Tensor:
        _ = self.vae.encode(image)
        return self.vae.decode(latents, h_carrier, w_carrier)


def test_vae_wrapper_construction(vae: Vae) -> None:
    """The wrapper Module can be built around a ``Vae`` cleanly."""
    wrapper = _VaeWrapper(vae=vae)
    assert wrapper.vae is vae
    children = dict(wrapper.children)
    assert set(children.keys()) == {"vae"}
    assert children["vae"] is vae


def test_vae_wrapper_parameter_sharing(vae: Vae) -> None:
    """Parent's parameter tree sees one ``bn_mean`` / ``bn_var``.

    This is the headline demonstration of MXF-460's
    *single-Module-with-two-forward-methods-and-shared-weights*
    pattern: by owning a single ``Vae`` sub-Module (rather than two
    siblings for encoder and decoder), a parent automatically gets
    one ``vae.bn_mean`` and one ``vae.bn_var`` in its parameter tree
    -- not two copies that need to be aliased across modules.
    """
    wrapper = _VaeWrapper(vae=vae)
    wrapper_params = {name for name, _ in wrapper.parameters}

    # Shared BN buffers surface under the ``vae.`` prefix, exactly once.
    bn_names = {n for n in wrapper_params if n.endswith(("bn_mean", "bn_var"))}
    assert bn_names == {"vae.bn_mean", "vae.bn_var"}

    # The encoder + decoder sub-trees are both reachable from the
    # wrapper (so a parent's ``compile`` would trace into both halves).
    assert any(n.startswith("vae.encoder.") for n in wrapper_params)
    assert any(n.startswith("vae.decoder.") for n in wrapper_params)

    # And critically: no duplicate BN buffers under either child.
    assert "vae.encoder.bn_mean" not in wrapper_params
    assert "vae.decoder.bn_mean" not in wrapper_params
    assert "vae.encoder.bn_var" not in wrapper_params
    assert "vae.decoder.bn_var" not in wrapper_params


# ---------------------------------------------------------------------------
# 13. Headline test: encode + decode trace into one parent graph
# ---------------------------------------------------------------------------


@_REQUIRES_NVIDIA_GPU
def test_vae_wrapper_compile(vae: Vae, device_spec: DeviceSpec) -> None:
    """Compile ``_VaeWrapper.forward`` -- both halves trace into one graph.

    Exercises the headline claim of the
    *single-Module-with-two-forward-methods-and-shared-weights* pattern:
    a parent that calls both :meth:`Vae.encode` and :meth:`Vae.decode`
    from its own ``forward`` traces them into a single graph along with
    the shared ``bn_mean`` / ``bn_var`` parameters.  A successful
    compile is the regression signal -- if the shared BN tensors didn't
    flow into the parent's graph correctly, or if the symbolic-shape
    rebinds in patchify/unpatchify drifted, compilation would fail.
    """
    wrapper = _VaeWrapper(vae=vae)
    device = Accelerator(device_spec.id)
    device_ref = DeviceRef.from_device(device)

    # Tiny shapes: a 16x16 image encodes to
    # (1, packed_h * packed_w, num_channels) packed latents with
    # packed_h = packed_w = 4 after one downsample + 2x2 patchify.
    image_type = TensorType(DType.uint8, shape=[16, 16, 3], device=device_ref)
    latents_type = TensorType(
        DType.bfloat16,
        shape=[1, 4 * 4, vae.num_channels],
        device=device_ref,
    )
    h_carrier_type = TensorType(DType.bfloat16, shape=[4], device=device_ref)
    w_carrier_type = TensorType(DType.bfloat16, shape=[4], device=device_ref)

    compiled = wrapper.compile(
        image_type, latents_type, h_carrier_type, w_carrier_type
    )
    assert compiled is not None
