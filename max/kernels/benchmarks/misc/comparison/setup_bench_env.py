#!/usr/bin/env python3
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

"""Unified setup script for prefill & grouped GEMM benchmarks.

Modes:
1. Install mode (default): Creates/uses a venv, installs PyTorch + external baselines.
2. Wheel build mode (--build-wheels): Builds reproducible wheels for CI/Bazel integration.

External baselines: DeepGEMM, FlashInfer, FlashAttention.
Repositories are auto-cloned to ~/.cache/blackwell_bench/ if not present.

Wheel Build Usage:
    # Build all wheels for SM100 (Blackwell B200)
    python setup_bench_env.py --build-wheels --wheel-dir ./sm100_wheels

    # Build only DeepGEMM wheel
    python setup_bench_env.py --build-wheels --wheel-dir ./sm100_wheels \\
        --no-flashinfer --no-flashattn

    # Use custom source path (skips auto-clone)
    python setup_bench_env.py --build-wheels --wheel-dir ./sm100_wheels \\
        --deepgemm-src /path/to/DeepGEMM

After building, upload wheels to S3:
    ./utils/upload-public-bazel-artifact.sh deep_gemm sm100 <wheel_path>
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Script is at Kernels/benchmarks/comparison/, so REPO_ROOT is 3 levels up
REPO_ROOT = HERE.parent.parent.parent
DEFAULT_FI_BASE = Path.home() / ".cache/flashinfer_bench"
DEFAULT_FI_CACHE = DEFAULT_FI_BASE / ".cache/flashinfer"
DEFAULT_MAX_JOBS = 64
DEFAULT_WHEEL_DIR = REPO_ROOT / "sm100_wheels"
DEFAULT_SRC_CACHE = Path.home() / ".cache/blackwell_bench"

# External baseline repositories
REPOS = {
    "deepgemm": {
        "url": "https://github.com/deepseek-ai/DeepGEMM.git",
        "branch": "main",
    },
    "flashinfer": {
        "url": "https://github.com/flashinfer-ai/flashinfer.git",
        "branch": "main",
    },
    "flashattn": {
        "url": "https://github.com/Dao-AILab/flash-attention.git",
        "branch": "main",
    },
}


@dataclass
class WheelBuildResult:
    """Result of a wheel build operation."""

    package: str
    success: bool
    wheel_path: Path | None = None
    sha256: str | None = None
    git_hash: str | None = None
    error: str | None = None


def sh(
    cmd: list[str], env: Mapping | None = None, cwd: str | None = None
) -> None:
    subprocess.run(cmd, check=True, env=env, cwd=cwd)


def ensure_venv(venv: Path, py: str) -> Path:
    if not venv.exists():
        sh([py, "-m", "venv", str(venv)])
    return venv / "bin/python"


def build_env(python: Path) -> dict:
    env = os.environ.copy()
    vbin = python.parent
    env["PATH"] = f"{vbin}:" + env.get("PATH", "")
    return env


# ---------------------------------------------------------------------------
# Repository management
# ---------------------------------------------------------------------------


def ensure_repo(name: str, src: Path) -> Path:
    """Ensure a source repository exists, cloning if necessary.

    Args:
        name: Repository key in REPOS dict (e.g., "deepgemm", "flashinfer")
        src: Target directory path for the repository

    Returns:
        Path to the repository (same as src)

    Raises:
        ValueError: If name is not in REPOS
        subprocess.CalledProcessError: If git clone fails
    """
    if src.exists():
        print(f"[info] Using existing {name} at {src}")
        return src

    if name not in REPOS:
        raise ValueError(f"Unknown repository: {name}")

    repo_info = REPOS[name]
    url = repo_info["url"]
    branch = repo_info.get("branch", "main")

    print(f"[info] Cloning {name} from {url}")
    print(f"       Branch: {branch}")
    print(f"       Target: {src}")

    src.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "git",
        "clone",
        "--recursive",
        "--branch",
        branch,
        "--depth",
        "1",
        url,
        str(src),
    ]
    subprocess.run(cmd, check=True)

    print(f"[ok] Cloned {name} to {src}")
    return src


# ---------------------------------------------------------------------------
# Wheel building functions
# ---------------------------------------------------------------------------


def get_git_info(src: Path) -> tuple[str, str]:
    """Get git commit hash and branch/tag for a source directory.

    Returns (short_hash, ref_name) tuple. Returns ("unknown", "unknown") on error.
    """
    try:
        hash_result = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            cwd=str(src),
            capture_output=True,
            text=True,
            check=True,
        )
        short_hash = hash_result.stdout.strip()

        # Try to get tag or branch name
        ref_result = subprocess.run(
            ["git", "describe", "--tags", "--exact-match"],
            cwd=str(src),
            capture_output=True,
            text=True,
        )
        if ref_result.returncode == 0:
            ref_name = ref_result.stdout.strip()
        else:
            # Fall back to branch name
            branch_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(src),
                capture_output=True,
                text=True,
            )
            ref_name = (
                branch_result.stdout.strip()
                if branch_result.returncode == 0
                else "unknown"
            )

        return short_hash, ref_name
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", "unknown"


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def sanitize_wheel_filename(wheel_path: Path) -> Path:
    """Rename wheel if filename contains '+' to avoid S3 URL encoding issues.

    S3 objects with '+' in filenames require URL encoding (%2B) for HTTP access.
    Bazel's http_file doesn't URL-encode, causing 403 errors. We rename wheels
    to replace '+' with '_' which preserves the version info while being URL-safe.

    PEP 440 local versions use '+' (e.g., 2.2.0+38f8ef7), which appears in wheel
    filenames like: deep_gemm-2.2.0+38f8ef7-cp312-cp312-linux_x86_64.whl

    Args:
        wheel_path: Path to the wheel file.

    Returns:
        New path if renamed, original path if no '+' present.
    """
    if "+" not in wheel_path.name:
        return wheel_path

    new_name = wheel_path.name.replace("+", "_")
    new_path = wheel_path.parent / new_name

    print("[info] Renaming wheel to avoid S3 URL encoding issues:")
    print(f"       {wheel_path.name}")
    print(f"    -> {new_name}")

    wheel_path.rename(new_path)
    return new_path


def clean_build_artifacts(src: Path) -> None:
    """Clean build artifacts from a source directory."""
    for pattern in ["build", "dist", "*.egg-info"]:
        if pattern.startswith("*"):
            for p in src.glob(pattern):
                shutil.rmtree(p, ignore_errors=True)
        else:
            path = src / pattern
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)


def build_wheel_generic(
    python: Path,
    src: Path,
    wheel_dir: Path,
    package_name: str,
    env_extras: dict[str, str] | None = None,
    clean: bool = True,
) -> WheelBuildResult:
    """Build a wheel from source directory.

    Args:
        python: Path to Python interpreter with torch installed.
        src: Source directory containing setup.py/pyproject.toml.
        wheel_dir: Output directory for the wheel.
        package_name: Human-readable package name for logging.
        env_extras: Additional environment variables for the build.
        clean: Whether to clean build artifacts before building.

    Returns:
        WheelBuildResult with build status and wheel info.
    """
    if not src.exists():
        return WheelBuildResult(
            package=package_name,
            success=False,
            error=f"Source directory not found: {src}",
        )

    git_hash, git_ref = get_git_info(src)
    print(f"\n{'=' * 60}")
    print(f"Building {package_name}")
    print(f"  Source: {src}")
    print(f"  Git: {git_hash} ({git_ref})")
    print(f"  Output: {wheel_dir}")
    print("=" * 60)

    if clean:
        print(f"[info] Cleaning build artifacts in {src}")
        clean_build_artifacts(src)

    wheel_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(python)
    if env_extras:
        for k, v in env_extras.items():
            env[k] = v

    # Log key environment variables
    print("[info] Build environment:")
    for var in [
        "TORCH_CUDA_ARCH_LIST",
        "FLASHINFER_CUDA_ARCH_LIST",
        "FLASH_ATTN_CUDA_ARCHS",
        "FLASH_ATTENTION_FORCE_BUILD",
        "FLASH_ATTENTION_SKIP_CUDA_BUILD",
        "MAX_JOBS",
        "NVCC_THREADS",
    ]:
        if var in env:
            print(f"  {var}={env[var]}")

    try:
        cmd = [
            str(python),
            "-m",
            "pip",
            "wheel",
            "--no-build-isolation",
            "--no-deps",
            "-v",
            f"--wheel-dir={wheel_dir}",
            str(src),
        ]
        print(f"[info] Running: {' '.join(cmd[:6])} ...")
        subprocess.run(cmd, env=env, check=True)

        # Find the built wheel (most recent .whl file)
        # Match by package name patterns
        wheel_patterns = [
            f"{package_name.lower().replace('-', '_')}*.whl",
            f"{package_name.lower()}*.whl",
            f"{package_name}*.whl",
        ]
        wheels: list[Path] = []
        for pattern in wheel_patterns:
            wheels.extend(wheel_dir.glob(pattern))

        if not wheels:
            # Fallback: find any new wheel
            wheels = list(wheel_dir.glob("*.whl"))

        if not wheels:
            return WheelBuildResult(
                package=package_name,
                success=False,
                git_hash=git_hash,
                error="No wheel file found after build",
            )

        # Get the most recently modified wheel
        wheel_path = max(wheels, key=lambda p: p.stat().st_mtime)

        # Sanitize filename to avoid S3 URL encoding issues with '+'
        wheel_path = sanitize_wheel_filename(wheel_path)

        sha256 = compute_sha256(wheel_path)

        print(f"\n[ok] Successfully built {package_name}")
        print(f"  Wheel: {wheel_path.name}")
        print(f"  SHA256: {sha256}")

        return WheelBuildResult(
            package=package_name,
            success=True,
            wheel_path=wheel_path,
            sha256=sha256,
            git_hash=git_hash,
        )

    except subprocess.CalledProcessError as e:
        return WheelBuildResult(
            package=package_name,
            success=False,
            git_hash=git_hash,
            error=f"Build failed with exit code {e.returncode}",
        )
    except Exception as e:
        return WheelBuildResult(
            package=package_name,
            success=False,
            git_hash=git_hash,
            error=str(e),
        )


def build_deepgemm_wheel(
    python: Path, src: Path, wheel_dir: Path, clean: bool = True
) -> WheelBuildResult:
    """Build DeepGEMM wheel with correct ABI settings."""
    env_extras = {
        "TORCH_CXX11_ABI": "1",
        "CXXFLAGS": "-U_GLIBCXX_USE_CXX11_ABI -D_GLIBCXX_USE_CXX11_ABI=1",
    }
    return build_wheel_generic(
        python, src, wheel_dir, "deep_gemm", env_extras, clean
    )


def build_flashinfer_wheel(
    python: Path, src: Path, wheel_dir: Path, clean: bool = True
) -> WheelBuildResult:
    """Build FlashInfer wheel with JIT cache settings."""
    env_extras = {
        "NVCC_THREADS": os.environ.get("NVCC_THREADS", "4"),
        "FLASHINFER_WORKSPACE_BASE": str(DEFAULT_FI_BASE),
        "FLASHINFER_CACHE_DIR": str(DEFAULT_FI_CACHE),
    }
    return build_wheel_generic(
        python, src, wheel_dir, "flashinfer", env_extras, clean
    )


def build_flashattn_wheel(
    python: Path, src: Path, wheel_dir: Path, clean: bool = True
) -> WheelBuildResult:
    """Build flash-attention wheel as pure Python (cute interface only).

    We build a pure Python wheel because:
    1. The cute interface uses nvidia-cutlass-dsl's JIT compiler, not precompiled CUDA
    2. Building CUDA extensions requires matching CUDA/PyTorch versions
    3. The prebuilt wheels from GitHub may be outdated vs the source

    Environment variables:
    - FLASH_ATTENTION_FORCE_BUILD: Force building from source instead of downloading
      prebuilt wheels from GitHub releases (which may be outdated).
    - FLASH_ATTENTION_SKIP_CUDA_BUILD: Skip compiling CUDA extensions (flash_attn_2_cuda).
      The cute interface we use doesn't need these - it uses cutlass-dsl JIT.
    """
    env_extras: dict[str, str] = {
        "NVCC_THREADS": os.environ.get("NVCC_THREADS", "4"),
        # Force building from source - don't download prebuilt wheels from GitHub
        # which may be outdated relative to the source checkout.
        "FLASH_ATTENTION_FORCE_BUILD": "TRUE",
        # Skip CUDA extension compilation - we only need the pure Python cute
        # interface which uses nvidia-cutlass-dsl's JIT compiler at runtime.
        "FLASH_ATTENTION_SKIP_CUDA_BUILD": "TRUE",
    }

    return build_wheel_generic(
        python, src, wheel_dir, "flash_attn", env_extras, clean
    )


def print_wheel_summary(results: list[WheelBuildResult]) -> None:
    """Print a summary of wheel build results for easy S3 upload."""
    print("\n" + "=" * 70)
    print("WHEEL BUILD SUMMARY")
    print("=" * 70)

    success_count = sum(1 for r in results if r.success)
    print(f"\nBuilt {success_count}/{len(results)} wheels successfully.\n")

    # Print successful builds
    successful = [r for r in results if r.success]
    if successful:
        print("Successful builds:")
        print("-" * 70)
        for r in successful:
            print(f"\n  Package: {r.package}")
            print(f"  Wheel:   {r.wheel_path}")
            print(f"  SHA256:  {r.sha256}")
            print(f"  Git:     {r.git_hash}")

        print("\n" + "-" * 70)
        print("Upload commands:")
        print("-" * 70)
        for r in successful:
            if r.wheel_path:
                pkg_name = r.package.replace("-", "_")
                print(
                    "  ./utils/upload-public-bazel-artifact.sh"
                    f" {pkg_name} sm100 {r.wheel_path}"
                )

        print("\n" + "-" * 70)
        print("MODULE.bazel http_file entries (copy after upload):")
        print("-" * 70)
        for r in successful:
            if r.wheel_path and r.sha256:
                pkg_name = r.package.replace("-", "_")
                wheel_name = r.wheel_path.name
                print(
                    f"""
http_file(
    name = "{pkg_name}_sm100_wheel",
    downloaded_file_path = "{wheel_name}",
    sha256 = "{r.sha256}",
    urls = ["https://modular-bazel-artifacts-public.s3.amazonaws.com/artifacts/{pkg_name}/sm100/{r.sha256}/{wheel_name}"],
)"""
                )

    # Print failed builds
    failed = [r for r in results if not r.success]
    if failed:
        print("\n" + "-" * 70)
        print("Failed builds:")
        print("-" * 70)
        for r in failed:
            print(f"\n  Package: {r.package}")
            print(f"  Error:   {r.error}")

    print("\n" + "=" * 70)


def install_base(
    python: Path,
    torch_index: str,
    torch_version: str = "2.8.0",
    install_torch: bool = True,
) -> None:
    pip = [str(python), "-m", "pip"]
    env = build_env(python)
    sh(
        pip
        + [
            "install",
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
            "packaging",
            "ninja",
            "psutil",
            "numpy",
        ],
        env=env,
    )
    if install_torch:
        # Pin torch version to match monorepo for ABI compatibility
        sh(
            pip
            + [
                "install",
                "--index-url",
                torch_index,
                f"torch=={torch_version}",
                "torchvision",
            ],
            env=env,
        )


def install_deepgemm(python: Path, src: Path) -> None:
    if not src.exists():
        print(f"[warn] DeepGEMM not found at {src}, skipping")
        return

    env = build_env(python)
    env.setdefault("TORCH_CXX11_ABI", "1")
    env.setdefault(
        "CXXFLAGS", "-U_GLIBCXX_USE_CXX11_ABI -D_GLIBCXX_USE_CXX11_ABI=1"
    )
    env.setdefault("MAX_JOBS", str(DEFAULT_MAX_JOBS))
    pip = [
        str(python),
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
        "-v",
        str(src),
    ]
    try:
        sh(pip, env=env)
    except subprocess.CalledProcessError:
        print("[warn] DeepGEMM install failed")


def install_flashinfer(python: Path, src: Path) -> None:
    if not src.exists():
        print(f"[warn] flashinfer not found at {src}, skipping")
        return
    env = build_env(python)
    env.setdefault("MAX_JOBS", str(DEFAULT_MAX_JOBS))
    env.setdefault("NVCC_THREADS", "4")
    env.setdefault("FLASHINFER_WORKSPACE_BASE", str(DEFAULT_FI_BASE))
    env.setdefault("FLASHINFER_CACHE_DIR", str(DEFAULT_FI_CACHE))
    try:
        sh(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-build-isolation",
                "-v",
                str(src),
            ],
            env=env,
        )
    except subprocess.CalledProcessError:
        print("[warn] flashinfer install failed")


def install_flashattn(python: Path, src: Path) -> None:
    if not src.exists():
        print(f"[warn] flash-attention not found at {src}, skipping")
        return

    env = build_env(python)
    env.setdefault("MAX_JOBS", str(DEFAULT_MAX_JOBS))
    env.setdefault("NVCC_THREADS", "4")

    # Pin CUDA toolchain to 13.0 to match torch cu130 wheels
    for cuda_dir in (
        "/usr/local/cuda-13.0",
        "/usr/local/cuda-13",
        env.get("CUDA_HOME"),
    ):
        if cuda_dir and os.path.isdir(cuda_dir):
            env["CUDA_HOME"] = cuda_dir
            env["PATH"] = f"{cuda_dir}/bin:" + env.get("PATH", "")
            ldpaths = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = (
                f"{cuda_dir}/lib64:{ldpaths}"
                if ldpaths
                else f"{cuda_dir}/lib64"
            )
            env["CUDACXX"] = f"{cuda_dir}/bin/nvcc"
            break
    try:
        sh(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-build-isolation",
                "-v",
                str(src),
            ],
            env=env,
        )
    except subprocess.CalledProcessError:
        print("[warn] flash-attention install failed")


def ensure_max_venv() -> Path:
    """Create MAX pipelines venv via bazel if missing."""

    target_venv = REPO_ROOT / ".max+python+max+entrypoints+pipelines.venv"
    if target_venv.exists():
        return target_venv / "bin/python"

    env = os.environ.copy()
    env.setdefault("MAX_JOBS", str(DEFAULT_MAX_JOBS))
    sh(
        ["./bazelw", "run", "//max/python/max/_entrypoints:pipelines.venv"],
        env=env,
        cwd=str(REPO_ROOT),
    )
    return target_venv / "bin/python"


def smoke(python: Path, env: dict | None = None) -> None:
    code = r"""
import importlib, torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())

def try_import(name: str):
    try:
        importlib.import_module(name)
        print('[ok] import', name)
        return True
    except Exception as e:
        print('[warn] import failed', name, e)
        return False

has_flashinfer = try_import('flashinfer')
has_flashattn = try_import('flash_attn.cute.interface')
try_import('deep_gemm')
try_import('max')

# FlashInfer mini-run
if has_flashinfer and torch.cuda.is_available():
    try:
        import flashinfer
        dtype = torch.bfloat16
        bs, seqlen, heads, hd = 1, 16, 8, 64
        q = torch.randn(bs*seqlen, heads, hd, device='cuda', dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        offs = torch.arange(0, bs+1, device='cuda', dtype=torch.int32) * seqlen
        work = torch.empty(8*1024*1024, device='cuda', dtype=dtype)
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(work, kv_layout='NHD', backend='cutlass')
        wrapper.plan(offs, offs, heads, heads, hd, head_dim_vo=hd, causal=False, q_data_type=dtype, kv_data_type=dtype)
        out = wrapper.run(q, k, v)
        torch.cuda.synchronize()
        print('[ok] flashinfer run', tuple(out.shape))
    except Exception as e:
        print('[warn] flashinfer run failed', e)

# FlashAttention mini-run
if has_flashattn and torch.cuda.is_available():
    try:
        from flash_attn.cute.interface import flash_attn_varlen_func
        bs, seqlen, heads, hd = 1, 16, 8, 64
        dtype = torch.bfloat16
        q = torch.randn(bs*seqlen, heads, hd, device='cuda', dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        offs = torch.arange(0, bs+1, device='cuda', dtype=torch.int32) * seqlen
        out, _ = flash_attn_varlen_func(q, k, v, cu_seqlens_q=offs, cu_seqlens_k=offs, causal=False, pack_gqa=False)
        torch.cuda.synchronize()
        print('[ok] flash-attn run', tuple(out.shape))
    except Exception as e:
        print('[warn] flash-attn run failed', e)
"""
    sh([str(python), "-c", code], env=env)


def run_wheel_build(args: argparse.Namespace) -> int:
    """Run wheel building mode.

    Returns exit code (0 for success, 1 if any builds failed).
    """
    print("=" * 70)
    print("WHEEL BUILD MODE")
    print("=" * 70)
    print(f"Target architecture: {args.cuda_arch_list}")
    print(f"Output directory: {args.wheel_dir}")
    print("=" * 70)

    # Get Python interpreter
    if args.no_max:
        vpy = ensure_venv(args.venv, args.python)
        install_base(vpy, args.torch_index, args.torch_version)
    else:
        vpy = ensure_max_venv()
        # Ensure torch is installed for building
        install_base(
            vpy, args.torch_index, args.torch_version, install_torch=True
        )

    print(f"\n[info] Using Python: {vpy}")

    # Verify torch is available
    try:
        subprocess.run(
            [
                str(vpy),
                "-c",
                "import torch; print(f'PyTorch {torch.__version__}')",
            ],
            check=True,
        )
    except subprocess.CalledProcessError:
        print("[error] PyTorch not available in the environment")
        return 1

    results: list[WheelBuildResult] = []

    # Build requested wheels (clone repos if needed)
    if not args.no_deepgemm:
        try:
            src = ensure_repo("deepgemm", args.deepgemm_src)
            result = build_deepgemm_wheel(
                vpy, src, args.wheel_dir, clean=args.clean_build
            )
        except subprocess.CalledProcessError as e:
            result = WheelBuildResult(
                package="deep_gemm", success=False, error=f"Clone failed: {e}"
            )
        results.append(result)

    if not args.no_flashinfer:
        try:
            src = ensure_repo("flashinfer", args.flashinfer_src)
            result = build_flashinfer_wheel(
                vpy, src, args.wheel_dir, clean=args.clean_build
            )
        except subprocess.CalledProcessError as e:
            result = WheelBuildResult(
                package="flashinfer", success=False, error=f"Clone failed: {e}"
            )
        results.append(result)

    if not args.no_flashattn:
        try:
            src = ensure_repo("flashattn", args.flashattn_src)
            result = build_flashattn_wheel(
                vpy, src, args.wheel_dir, clean=args.clean_build
            )
        except subprocess.CalledProcessError as e:
            result = WheelBuildResult(
                package="flash_attn", success=False, error=f"Clone failed: {e}"
            )
        results.append(result)

    # Print summary
    print_wheel_summary(results)

    # Return exit code based on success
    return 0 if all(r.success for r in results) else 1


def run_install(args: argparse.Namespace) -> None:
    """Run installation mode (original behavior)."""
    if args.no_max:
        vpy = ensure_venv(args.venv, args.python)
        install_base(vpy, args.torch_index, args.torch_version)
    else:
        vpy = ensure_max_venv()
        # Install the standard stack with torch so that all baselines
        # (flashinfer/flash-attn/DeepGEMM) work consistently.
        install_base(
            vpy, args.torch_index, args.torch_version, install_torch=True
        )

    # Install baselines (clone repos if needed)
    if not args.no_deepgemm:
        try:
            src = ensure_repo("deepgemm", args.deepgemm_src)
            install_deepgemm(vpy, src)
        except subprocess.CalledProcessError as e:
            print(f"[warn] DeepGEMM clone failed: {e}")
    if not args.no_flashinfer:
        try:
            src = ensure_repo("flashinfer", args.flashinfer_src)
            install_flashinfer(vpy, src)
        except subprocess.CalledProcessError as e:
            print(f"[warn] FlashInfer clone failed: {e}")
    if not args.no_flashattn:
        try:
            src = ensure_repo("flashattn", args.flashattn_src)
            install_flashattn(vpy, src)
        except subprocess.CalledProcessError as e:
            print(f"[warn] flash-attention clone failed: {e}")

    # Run smoke test with PATH including venv/bin so helper tools (ninja) are found.
    smoke_env = build_env(vpy)
    smoke(vpy, env=smoke_env)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Set up venv for MHA and grouped GEMM benchmarks, or build wheels"
            " for CI."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Install mode (default): set up venv with all baselines
  python setup_bench_env.py

  # Build wheels for SM100 (Blackwell B200)
  python setup_bench_env.py --build-wheels --wheel-dir ./sm100_wheels

  # Build only DeepGEMM wheel
  python setup_bench_env.py --build-wheels --no-flashinfer --no-flashattn

  # Build with clean artifacts
  python setup_bench_env.py --build-wheels --clean-build
""",
    )

    # Mode selection
    ap.add_argument(
        "--build-wheels",
        action="store_true",
        help="Build wheels instead of installing packages",
    )
    ap.add_argument(
        "--wheel-dir",
        type=Path,
        default=DEFAULT_WHEEL_DIR,
        help=(
            f"Output directory for built wheels (default: {DEFAULT_WHEEL_DIR})"
        ),
    )
    ap.add_argument(
        "--clean-build",
        action="store_true",
        default=True,
        help="Clean build artifacts before building (default: True)",
    )
    ap.add_argument(
        "--no-clean-build",
        action="store_false",
        dest="clean_build",
        help="Don't clean build artifacts before building",
    )

    # Common options
    ap.add_argument("--venv", default=REPO_ROOT / ".venv/bench-bw", type=Path)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument(
        "--torch-index",
        default="https://download.pytorch.org/whl/cu128",
        help="PyTorch wheel index URL (default: cu128 to match monorepo)",
    )
    ap.add_argument(
        "--torch-version",
        default="2.8.0",
        help="PyTorch version to install (default: 2.8.0 to match monorepo)",
    )
    ap.add_argument(
        "--cuda-arch-list",
        default="10.0a",
        help=(
            "Space- or semicolon-separated list for TORCH_CUDA_ARCH_LIST /"
            " FLASHINFER_CUDA_ARCH_LIST (e.g., '10.0a' for SM100 Blackwell; use"
            " '12.0f' for SM120)."
        ),
    )
    ap.add_argument(
        "--flashattn-archs",
        default="100",
        help=(
            "Semicolon- or space-separated list passed to FLASH_ATTN_CUDA_ARCHS"
            " (default SM100-only; per upstream, SM100 maps to Blackwell"
            " data-center parts such as B200/GB200)."
        ),
    )

    # Source paths (auto-cloned to cache dir if not present)
    ap.add_argument(
        "--deepgemm-src",
        default=DEFAULT_SRC_CACHE / "DeepGEMM",
        type=Path,
        help=f"DeepGEMM source (default: {DEFAULT_SRC_CACHE / 'DeepGEMM'})",
    )
    ap.add_argument(
        "--flashinfer-src",
        default=DEFAULT_SRC_CACHE / "flashinfer",
        type=Path,
        help=f"FlashInfer source (default: {DEFAULT_SRC_CACHE / 'flashinfer'})",
    )
    ap.add_argument(
        "--flashattn-src",
        default=DEFAULT_SRC_CACHE / "flash-attention",
        type=Path,
        help=(
            "flash-attention source (default:"
            f" {DEFAULT_SRC_CACHE / 'flash-attention'})"
        ),
    )

    # Package selection
    ap.add_argument("--no-deepgemm", action="store_true")
    ap.add_argument("--no-flashinfer", action="store_true")
    ap.add_argument("--no-flashattn", action="store_true")
    ap.add_argument(
        "--no-max",
        action="store_true",
        help="Skip using MAX venv and use a fresh venv instead",
    )

    args = ap.parse_args()

    # Set up common environment variables
    for var, val in {
        "MAX_JOBS": str(DEFAULT_MAX_JOBS),
        "NVCC_THREADS": "4",
        "CMAKE_BUILD_PARALLEL_LEVEL": str(DEFAULT_MAX_JOBS),
        "OMP_NUM_THREADS": str(DEFAULT_MAX_JOBS),
    }.items():
        os.environ.setdefault(var, val)

    # Limit codegen targets (torch extension and flashinfer) to reduce ptxas load.
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", args.cuda_arch_list)
    os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", args.cuda_arch_list)
    # Limit flash-attention build targets.
    os.environ.setdefault(
        "FLASH_ATTN_CUDA_ARCHS", args.flashattn_archs.replace(" ", ";")
    )

    # Reset flashinfer cache to avoid stale builds pointing at other venvs.
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", str(DEFAULT_FI_BASE))

    # Clean default and bench cache dirs to avoid stale references.
    shutil.rmtree(Path.home() / ".cache/flashinfer", ignore_errors=True)
    cache_dir = Path(os.environ.get("FLASHINFER_CACHE_DIR", DEFAULT_FI_CACHE))
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.environ.setdefault("FLASHINFER_CACHE_DIR", str(cache_dir))

    # Run appropriate mode
    if args.build_wheels:
        exit_code = run_wheel_build(args)
        sys.exit(exit_code)
    else:
        run_install(args)


if __name__ == "__main__":
    main()
