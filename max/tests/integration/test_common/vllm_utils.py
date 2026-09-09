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
"""Utilities for running vLLM models for testing."""

from __future__ import annotations

import hashlib
import os
import pathlib
import sys
import tempfile
from collections.abc import Iterable
from typing import Any

# Force vLLM to use 'spawn' instead of 'fork' for multiprocessing.
# The calling process (generate_llm_logits) may initialize CUDA before
# vLLM starts (e.g. via device_specs_from_normalized_device_handle), and
# CUDA cannot be re-initialized in a forked subprocess.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# Disable DeepGEMM. vLLM bundles a vendored `vllm.third_party.deep_gemm`, so
# `has_deep_gemm()` reports it as available and vLLM tries to JIT-compile and
# warm up DeepGEMM FP8 kernels for FP8 models (e.g. Kimi-K2.6). That JIT cannot
# run under Bazel's read-only pycross runfiles, so warmup raises "DeepGEMM
# backend is not available or outdated". Turning it off makes FP8 linear/MoE
# layers fall back to the CUTLASS/Triton GEMM path (numerically equivalent
# within logit-verification tolerances) for both warmup and execution. The env
# var is inherited by vLLM's spawned workers.
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")

import numpy as np
from test_common.test_data import MockTextGenerationRequest


def _setup_ninja_path() -> None:
    """Add ninja binary to PATH for FlashInfer JIT compilation.

    FlashInfer relies on ninja to JIT-compile kernels. In Bazel's
    pycross_wheel_library environment, ninja.BIN_DIR can be empty, so we
    locate the binary relative to the installed ninja package and prepend
    it to PATH. This must run before FlashInfer is imported or initialized.
    """
    try:
        import ninja  # type: ignore[import-not-found, unused-ignore]
    except ImportError:
        # ninja not available: let flashinfer import fail separately.
        return

    ninja_bin_dir = ninja.BIN_DIR
    if not ninja_bin_dir:
        # In Bazel pycross_wheel_library, bin is at ../../bin relative to
        # the package location.
        ninja_bin_dir = os.path.normpath(
            os.path.join(os.path.dirname(ninja.__file__), "..", "..", "bin")
        )
    if ninja_bin_dir and os.path.isdir(ninja_bin_dir):
        if ninja_bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = (
                ninja_bin_dir + os.pathsep + os.environ.get("PATH", "")
            )


def _setup_cutlass_path() -> None:
    """Expose the ``cutlass`` package shipped by ``nvidia-cutlass-dsl-libs-base``.

    vLLM's CUTE flash-attention path (used by the Kimi-K2.6 vision tower, among
    others) lazily does ``import cutlass.cute``. The ``nvidia-cutlass-dsl`` wheel
    is an empty metapackage; the actual ``cutlass`` package is shipped by
    ``nvidia-cutlass-dsl-libs-base`` under ``nvidia_cutlass_dsl/python_packages``
    and exposed via a ``nvidia_cutlass_dsl.pth`` file. Python's ``site`` machinery
    processes ``.pth`` files at startup, but Bazel's ``pycross`` runfiles layout
    does not, so that directory never lands on ``sys.path`` and ``import cutlass``
    fails with ``ModuleNotFoundError``.

    Locate the relocated package directory and prepend it to both ``sys.path``
    (for this process) and ``PYTHONPATH`` (so it is inherited by vLLM's spawned
    worker processes, which is where the import actually happens). This must run
    before vLLM is constructed.
    """
    # The libs-base wheel's site-packages root is on sys.path, with cutlass
    # nested under `nvidia_cutlass_dsl/python_packages`. Scan for it rather than
    # hard-coding version- or hash-specific runfiles paths.
    for entry in list(sys.path):
        if not entry:
            continue
        candidate = os.path.join(entry, "nvidia_cutlass_dsl", "python_packages")
        if os.path.isdir(os.path.join(candidate, "cutlass")):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            pythonpath = os.environ.get("PYTHONPATH", "")
            if candidate not in pythonpath.split(os.pathsep):
                os.environ["PYTHONPATH"] = (
                    candidate + os.pathsep + pythonpath
                    if pythonpath
                    else candidate
                )
            return


# Self-contained sitecustomize, auto-imported at interpreter startup so it runs
# in vLLM's spawned worker processes before flashinfer's cubin loader is
# imported. It rebuilds/reuses the writable cubin mirror itself (rather than
# relying on the parent's filesystem state) and repoints flashinfer at it.
_FLASHINFER_CUBIN_SITECUSTOMIZE = '''\
"""Auto-generated by MAX vllm_utils; redirects flashinfer's cubin directory.

Bazel pycross runfiles are read-only, but flashinfer wants to create alias
symlinks inside the `flashinfer-cubin` package directory. Point it at a writable
per-host mirror (symlinks to the bundled cubins) so those writes succeed.
"""
try:
    import hashlib
    import pathlib
    import tempfile

    import flashinfer_cubin

    _real = pathlib.Path(flashinfer_cubin.__file__).resolve().parent / "cubins"
    _digest = hashlib.sha1(str(_real).encode()).hexdigest()[:12]
    _mirror = pathlib.Path(tempfile.gettempdir()) / (
        "max_flashinfer_cubin_" + _digest
    )
    _mirror.mkdir(parents=True, exist_ok=True)
    if _real.is_dir():
        for _entry in _real.iterdir():
            _link = _mirror / _entry.name
            if not _link.exists() and not _link.is_symlink():
                try:
                    _link.symlink_to(_entry)
                except FileExistsError:
                    pass  # created concurrently by another worker
    flashinfer_cubin.CUBIN_DIR = _mirror
    flashinfer_cubin.get_cubin_dir = lambda: str(_mirror)
except Exception:
    # Best-effort: never break interpreter startup.
    pass
'''


def _redirect_flashinfer_cubin_dir() -> bool:
    """Point flashinfer's cubin directory at a writable per-host mirror.

    When the ``flashinfer-cubin`` package is installed, flashinfer resolves its
    cubin directory to ``flashinfer_cubin/cubins`` inside that (read-only under
    Bazel) package and ignores the ``FLASHINFER_CUBIN_DIR`` env var. It then
    tries to ``mkdir``/``symlink`` convenience alias paths under that directory
    (e.g. ``flashinfer/trtllm/batched_gemm`` for the TRT-LLM NVFP4 MoE kernels
    used by Kimi-K2.6), which fails on read-only runfiles.

    Build a writable mirror directory whose top-level entries symlink to the
    read-only bundled cubins (so reads resolve to the real artifacts) while the
    mirror root stays writable for the alias subdirectories flashinfer creates,
    and repoint ``flashinfer_cubin.get_cubin_dir`` at it. The mirror path is
    derived deterministically from the bundled cubin location, so the parent
    process and every spawned vLLM worker independently build/reuse the same
    mirror (no reliance on a shared parent-created directory). Returns whether
    ``flashinfer-cubin`` was present.
    """
    try:
        import flashinfer_cubin  # type: ignore[import-not-found, unused-ignore]
    except ImportError:
        # flashinfer-cubin not present (e.g. non-vLLM path): nothing to do.
        return False

    real = pathlib.Path(flashinfer_cubin.__file__).resolve().parent / "cubins"
    digest = hashlib.sha1(str(real).encode()).hexdigest()[:12]
    mirror = (
        pathlib.Path(tempfile.gettempdir()) / f"max_flashinfer_cubin_{digest}"
    )
    mirror.mkdir(parents=True, exist_ok=True)
    if real.is_dir():
        # Symlink each top-level bundled entry; the mirror root itself stays a
        # real, writable directory so flashinfer can create its alias subdirs.
        for entry in real.iterdir():
            link = mirror / entry.name
            if not link.exists() and not link.is_symlink():
                try:
                    link.symlink_to(entry)
                except FileExistsError:
                    pass  # created concurrently by another worker

    flashinfer_cubin.CUBIN_DIR = mirror
    flashinfer_cubin.get_cubin_dir = lambda: str(mirror)
    return True


def _setup_flashinfer_cubin_cache() -> None:
    """Redirect flashinfer's cubin directory to a writable mirror.

    Applies the redirect in this process and, via a ``sitecustomize`` injected
    onto ``PYTHONPATH``, in vLLM's spawned worker processes (where the cubin
    loader actually runs). Must run before vLLM is constructed. See
    ``_redirect_flashinfer_cubin_dir`` for the underlying mechanism.
    """
    if not _redirect_flashinfer_cubin_dir():
        return

    # Apply the same redirect in spawned workers via a sitecustomize on
    # PYTHONPATH (auto-imported at their interpreter startup).
    hook_dir = os.environ.get("_MAX_FLASHINFER_HOOK_DIR")
    if not hook_dir or not os.path.isdir(hook_dir):
        hook_dir = tempfile.mkdtemp(prefix="flashinfer_hook_")
        with open(os.path.join(hook_dir, "sitecustomize.py"), "w") as f:
            f.write(_FLASHINFER_CUBIN_SITECUSTOMIZE)
        os.environ["_MAX_FLASHINFER_HOOK_DIR"] = hook_dir
    pythonpath = os.environ.get("PYTHONPATH", "")
    if hook_dir not in pythonpath.split(os.pathsep):
        os.environ["PYTHONPATH"] = (
            hook_dir + os.pathsep + pythonpath if pythonpath else hook_dir
        )


def _resolve_vocab_size(*, llm: Any, tokenizer: Any, model_path: str) -> int:
    """Resolve vocab size for dense logit reconstruction.

    vLLM returns logprobs as a sparse mapping of {token_id: logprob}. For
    verification we rebuild a dense vector of shape (vocab_size,) and place
    each logprob at its token ID index. This requires a vocab size that matches
    the model's output head, not just the tokenizer's current size.

    Why this matters:
    - Token IDs are produced by the tokenizer, so we must index by tokenizer
      IDs.
    - The dense vector must be sized to the model's true vocab size (the logits
      dimension) so it aligns with MAX/Torch outputs during verification.
    - Some models (e.g. DeepSeek R1 NVFP4) report a vocab size in the model
      config that differs from len(tokenizer) or tokenizer.vocab_size. Using
      the tokenizer size yields vectors with the wrong length (e.g. 128000 or
      128815 vs 129280), which breaks element-wise comparison and tolerance
      checks.

    Resolution order:
    1) vLLM model config (hf_config.vocab_size) if available.
    2) Local HF config.json (no network) via huggingface_hub cache.
    3) Tokenizer length or vocab_size as a fallback.
    """
    vocab_size: int | None = None
    try:
        model_config = llm.llm_engine.model_config
        hf_config = getattr(model_config, "hf_config", None)
        vocab_size = getattr(hf_config, "vocab_size", None)
    except Exception:
        vocab_size = None

    if vocab_size is None:
        try:
            import json

            from huggingface_hub import (  # type: ignore[import-not-found, unused-ignore]
                hf_hub_download,
            )

            config_path = hf_hub_download(
                repo_id=model_path,
                filename="config.json",
                local_files_only=True,
            )
            with open(config_path) as f:
                vocab_size = json.load(f).get("vocab_size")
        except Exception:
            vocab_size = None

    if vocab_size is None:
        try:
            vocab_size = len(tokenizer)
        except TypeError:
            vocab_size = tokenizer.vocab_size

    return int(vocab_size)


def _messages_as_dicts(messages: Iterable[Any]) -> list[dict[str, Any]]:
    """Convert request messages to OpenAI-style dicts for chat templating.

    ``request.messages`` are ``TextGenerationRequestMessage`` (Pydantic) objects.
    HF chat templates (e.g. Kimi-K2.6) access fields with ``message.get(...)``,
    which Pydantic objects do not support — passing them directly raises
    ``jinja2.exceptions.UndefinedError: ... has no attribute 'get'``.
    ``model_dump`` yields the standard ``{role, content, ...}`` dict the
    templates expect (and Jinja ``message.role`` access still works on dicts).
    """
    return [m.model_dump() if hasattr(m, "model_dump") else m for m in messages]


def run_text_generation(
    model_path: str,
    textgen_requests: Iterable[MockTextGenerationRequest],
    num_steps: int = 10,
    print_outputs: bool = False,
    encoding_name: str | None = None,
    trust_remote_code: bool = False,
    gpu_memory_utilization: float = 0.9,
    max_batch_size: int | None = None,
    tensor_parallel_size: int = 1,
    extra_kwargs: dict[str, Any] | None = None,
    mm_data_key: str = "image",
) -> list[dict[str, Any]]:
    """Run text generation using vLLM.

    NOTE: We import vLLM inside this function to avoid triggering any
    CUDA initialization or multiprocessing side-effects at module-import time.
    """

    # Set `ninja` path since vLLM V1 defaults to `FLASHINFER`, which may
    # require `ninja` to JIT compile kernels at runtime.
    _setup_ninja_path()

    # Make the `cutlass` package importable for vLLM's CUTE flash-attention
    # path. The wheel relocates it behind a `.pth` file that Bazel does not
    # process; see `_setup_cutlass_path` for details.
    _setup_cutlass_path()

    # Redirect flashinfer's cubin directory to a writable mirror so it can
    # create its kernel symlinks (the bundled `flashinfer-cubin` package dir is
    # read-only under Bazel); see `_setup_flashinfer_cubin_cache` for details.
    _setup_flashinfer_cubin_cache()

    try:
        from vllm import (  # type: ignore[import-not-found, unused-ignore]
            LLM,
            SamplingParams,
        )
    except ImportError as e:
        raise SystemExit(
            f"Failed to import vLLM ({e}). vLLM is an opt-in dependency "
            "gated by the `--//:use_vllm` Bazel flag and is only supported "
            "on NVIDIA GPUs. Rebuild the target with `--//:use_vllm` (and "
            "run on an NVIDIA GPU) to use the vLLM framework."
        ) from None

    # Map encoding_name to vLLM dtype/quantization
    dtype = "auto"
    quantization = None

    if encoding_name:
        if encoding_name in ["float32", "float16", "bfloat16"]:
            dtype = encoding_name
        elif encoding_name == "float8_e4m3fn":
            # vLLM often runs FP8 models automatically if hardware supports it,
            # but usually setting dtype to float16/bfloat16 is safer for the container
            dtype = "float16"
        elif encoding_name == "float4_e2m1fnx2":
            # NVFP4 models - vLLM loads these natively with auto dtype detection
            dtype = "auto"
        elif encoding_name in ["awq", "gptq", "squeezellm", "fp8"]:
            quantization = encoding_name
        else:
            raise ValueError(f"Unrecognized encoding: {encoding_name}")

    # Handle batch size limit if provided
    # vLLM uses max_num_seqs to control how many sequences are processed at once
    max_num_seqs = max_batch_size if max_batch_size is not None else 256

    # Initialize vLLM
    # We set gpu_memory_utilization explicitly to avoid OOM if the runner
    # has some overhead, though vLLM usually dominates.
    llm_kwargs: dict[str, Any] = {
        "model": model_path,
        "dtype": dtype,
        "quantization": quantization,
        "trust_remote_code": trust_remote_code,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_num_seqs": max_num_seqs,
        # Default max_logprobs is 20. We increase this to support full logits
        # retrieval. 262144 covers large vocabs (e.g. Qwen2.5 is ~152k).
        "max_logprobs": 262144,
        # Force eager mode for stability.
        "enforce_eager": True,
        # Avoid vLLM custom all-reduce path for stability.
        "disable_custom_all_reduce": True,
        # Tensor parallelism for multi-GPU models
        "tensor_parallel_size": tensor_parallel_size,
    }
    # Allow oracles to pass model-specific kwargs (e.g. mm_encoder_tp_mode,
    # limit_mm_per_prompt). Intentionally overrides defaults if keys collide.
    if extra_kwargs:
        llm_kwargs.update(extra_kwargs)

    llm: Any = LLM(**llm_kwargs)

    tokenizer = llm.get_tokenizer()
    vocab_size = _resolve_vocab_size(
        llm=llm, tokenizer=tokenizer, model_path=model_path
    )

    prompts: list[str | dict[str, Any]] = []
    sampling_params_list = []

    for request in textgen_requests:
        if request.is_multimodal and request.images:
            # Lazy import — only needed for multimodal requests.
            from test_common.storage import load_image

            pil_images = [load_image(img) for img in request.images]
            if mm_data_key == "vision_chunk":
                mm_items: Any = [
                    {"type": "image", "image": img} for img in pil_images
                ]
            else:
                mm_items = pil_images
            # Build the prompt from messages with the tokenizer chat template
            # so vLLM's prefill tokens match what MAX produces from the same
            # messages. The template preserves the image placeholder, so vLLM
            # still binds multi_modal_data to it.
            if request.messages:
                mm_prompt = tokenizer.apply_chat_template(
                    _messages_as_dicts(request.messages),
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                mm_prompt = request.prompt
            prompts.append(
                {
                    "prompt": mm_prompt,
                    "multi_modal_data": {mm_data_key: mm_items},
                }
            )
        elif request.messages:
            # Build the prompt from messages with the tokenizer chat template
            # so vLLM's tokens match what MAX's tokenizer produces from the
            # same messages.
            templated = tokenizer.apply_chat_template(
                _messages_as_dicts(request.messages),
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(templated)
        else:
            prompts.append(request.prompt)

        # We use logprobs=vocab_size to get the full distribution. This is the
        # closest approximation to logits we can get via the vLLM API.
        sp: Any = SamplingParams(
            max_tokens=num_steps,
            temperature=0,
            logprobs=vocab_size,
        )
        sampling_params_list.append(sp)

    outputs = llm.generate(prompts, sampling_params_list)

    results = []

    # Process outputs to match the format of torch_utils.py
    for request, output in zip(textgen_requests, outputs, strict=False):
        saved_logits = []

        # `output.outputs[0].logprobs` is a list of dicts (one per step). vLLM
        # may return `None` for the first step (prompt) depending on version,
        # but it usually returns generation steps.
        generated_data = output.outputs[0]

        if generated_data.logprobs:
            for step_logprobs in generated_data.logprobs:
                # Initialize with a proxy for -inf logprob
                logits_np = np.full((vocab_size,), -100.0, dtype=np.float32)

                # Fill in the values returned by vLLM
                # vLLM returns {token_id: LogprobObject}
                for token_id, logprob_obj in step_logprobs.items():
                    val = getattr(logprob_obj, "logprob", logprob_obj)
                    if token_id < vocab_size:
                        logits_np[token_id] = val

                next_token = logits_np.argmax()

                # Save vLLM logprobs explicitly; verification normalizes as needed.
                saved_logits.append(
                    {
                        "next_token": next_token,
                        "next_token_logprobs": float(logits_np[next_token]),
                        "logprobs": logits_np,
                    }
                )

        if print_outputs:
            print(
                "Prompt:",
                f"{request.prompt[:100]}...{request.prompt[-100:]}"
                if len(request.prompt) > 200
                else request.prompt,
            )
            print("Output:", request.prompt + output.outputs[0].text)

        results.append({"prompt": request.prompt, "values": saved_logits})

    return results
