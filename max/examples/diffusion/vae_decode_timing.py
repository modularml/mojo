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

"""Time a Wan VAE decode through the native Mojo conv path.

Loads the Wan VAE, decodes a deterministic random latent, and reports
execution-only wall time on the accelerator plus basic tensor statistics
(value range, NaN/inf fractions, post-clamp saturation fractions). A
single warm-up run is issued before the timed run so first-launch JIT /
autotune cost is excluded.

Usage:

.. code-block:: bash

    ./bazelw run //max/examples/diffusion:vae_decode_timing

    # Custom latent shape and seed:
    ./bazelw run //max/examples/diffusion:vae_decode_timing -- \\
        --shape 1 16 5 30 52 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import cast

import numpy as np

logger = logging.getLogger("vae_decode_timing")


# T2V-A14B (and Wan2.1) has the standard z_dim=16 / base_dim=96 /
# decoder_base_dim=None config that MAX's AutoencoderKLWanModel supports.
_DEFAULT_MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
# (B, z_dim, T, H, W). T=5 exercises first-frame + rest-frame cached decode
# (4 cached frames after the first); H/W are chosen small so the run
# completes in a few seconds on B200 without blowing memory.
_DEFAULT_SHAPE = [1, 16, 5, 30, 52]


def _decode_and_time(
    model_repo: str,
    seed: int,
    latent_shape: tuple[int, ...],
    encoding: str,
) -> int:
    """Decode a deterministic latent through the Wan VAE and report timing."""
    from huggingface_hub import snapshot_download
    from max.driver import CPU, DeviceSpec, load_devices
    from max.engine import InferenceSession
    from max.graph.weights import load_weights
    from max.pipelines.architectures.autoencoders.autoencoder_kl_wan import (
        AutoencoderKLWanModel,
        _buffer_to_numpy_f32,
        _numpy_f32_to_buffer,
    )
    from max.pipelines.lib import SupportedEncoding

    logger.info("Downloading VAE from %s", model_repo)
    snapshot = snapshot_download(repo_id=model_repo, allow_patterns=["vae/*"])
    vae_dir = Path(snapshot) / "vae"
    config_path = vae_dir / "config.json"
    if not config_path.exists():
        raise RuntimeError(f"VAE config.json not found at {config_path}")
    with open(config_path) as f:
        config_dict = json.load(f)

    weight_paths = sorted(vae_dir.glob("*.safetensors"))
    if not weight_paths:
        raise RuntimeError(
            f"No .safetensors files under {vae_dir}. "
            "The VAE subfolder may not have been downloaded."
        )
    weights = load_weights(weight_paths)

    # Pre-flight: MAX's Wan VAE always uses base_dim for the decoder; fail
    # early on non-standard configs instead of 200 lines later in
    # load_state_dict.
    if config_dict.get("decoder_base_dim") is not None:
        raise RuntimeError(
            f"Model {model_repo!r} has decoder_base_dim="
            f"{config_dict['decoder_base_dim']} which MAX's "
            "AutoencoderKLWanModel does not currently wire into the "
            "decoder. Pick a VAE with a standard config "
            "(e.g. Wan-AI/Wan2.2-T2V-A14B-Diffusers)."
        )
    cfg_z_dim = config_dict.get("z_dim", 16)
    if latent_shape[1] != cfg_z_dim:
        raise RuntimeError(
            f"Latent z-dim {latent_shape[1]} (from --shape) does not "
            f"match VAE config z_dim={cfg_z_dim}. Re-run with "
            f"--shape {latent_shape[0]} {cfg_z_dim} "
            f"{' '.join(str(s) for s in latent_shape[2:])}."
        )

    devices = load_devices([DeviceSpec.accelerator()])
    device = devices[0]
    session = InferenceSession(devices=devices)

    logger.info("Building AutoencoderKLWanModel")
    encoding_literal = cast("SupportedEncoding", encoding)
    vae = AutoencoderKLWanModel(
        config=config_dict,
        encoding=encoding_literal,
        devices=devices,
        weights=weights,
        session=session,
    )

    rng = np.random.RandomState(seed)
    latent_np = rng.randn(*latent_shape).astype(np.float32)
    target_dtype = vae.config.dtype
    latent_buf = _numpy_f32_to_buffer(latent_np, target_dtype, device)

    logger.info("Decoding latent shape=%s seed=%d", tuple(latent_shape), seed)

    # Warm up once so the timed run excludes first-launch JIT / autotune.
    _ = vae.decode_5d(latent_buf)
    device.synchronize()

    # Execution-only wall time: block on the GPU before taking the end
    # timestamp. Model download, weight load, and graph compile ran earlier
    # and are excluded; the CPU-side copy also runs after.
    t0 = time.perf_counter()
    decoded_buf = vae.decode_5d(latent_buf)
    device.synchronize()
    decode_s = time.perf_counter() - t0

    decoded_np = _buffer_to_numpy_f32(decoded_buf, CPU())

    print("=" * 70)
    print("Wan VAE decode timing")
    print("=" * 70)
    print(f"Model:       {model_repo}")
    print(f"Latent:      shape={tuple(latent_shape)} seed={seed}")
    print(f"Decode wall: {decode_s * 1000:.2f} ms ({decode_s:.3f} s)")
    print()

    finite = decoded_np[np.isfinite(decoded_np)]
    nan_frac = float(np.isnan(decoded_np).mean())
    inf_frac = float(np.isinf(decoded_np).mean())
    print(f"Decoded shape: {decoded_np.shape}")
    print(
        f"  range=[{finite.min():.4f}, {finite.max():.4f}] "
        f"mean={finite.mean():.4f} std={finite.std():.4f}"
    )
    if nan_frac > 0 or inf_frac > 0:
        print(f"  WARNING: nan_frac={nan_frac:.6f} inf_frac={inf_frac:.6f}")

    # Post-VAE values clamp to [-1, 1] for image pipelines; report fractions
    # near the clamp boundaries so kernel regressions that drive outputs
    # toward black/white show up.
    clip_lo = float((decoded_np < -0.99).mean())
    clip_hi = float((decoded_np > 0.99).mean())
    print(
        f"  fraction <-0.99 (maps to black): {clip_lo:.6f}   "
        f"fraction > 0.99 (maps to white): {clip_hi:.6f}"
    )

    return 0 if (nan_frac == 0 and inf_frac == 0) else 2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Time a Wan VAE decode through the native Mojo conv path."
    )
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help="HuggingFace repo hosting the Wan VAE (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the deterministic random latent (default: %(default)s).",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=5,
        default=_DEFAULT_SHAPE,
        metavar=("B", "Z", "T", "H", "W"),
        help="Latent shape (default: %(default)s).",
    )
    parser.add_argument(
        "--encoding",
        default="bfloat16",
        help="Model encoding (default: %(default)s).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    rc = _decode_and_time(
        args.model,
        args.seed,
        tuple(args.shape),
        args.encoding,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
