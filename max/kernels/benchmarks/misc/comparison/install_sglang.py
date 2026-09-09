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

import subprocess
import sys


def update_pip() -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"]
    )


def install_pip_package(package: str, index_url: str = "") -> None:
    cmd = [sys.executable, "-m", "pip", "install", package]
    if index_url:
        cmd += ["--index-url", index_url]
    subprocess.check_call(cmd)


def _torch_cuda_index() -> str:
    result = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.version.cuda or '')"],
        capture_output=True,
        text=True,
        check=True,
    )
    cuda_ver = result.stdout.strip().replace(".", "")  # "13.0" -> "130"
    if not cuda_ver:
        return ""
    return f"https://download.pytorch.org/whl/cu{cuda_ver}"


def _matching_torchvision_version() -> str:
    result = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.__version__)"],
        capture_output=True,
        text=True,
        check=True,
    )
    # Strip local suffix: "2.11.0+cu130" -> "2.11.0"
    version = result.stdout.strip().split("+")[0]
    _, minor, _ = version.split(".")
    # torch 2.N -> torchvision 0.(N+15): 2.8->0.23, 2.11->0.26, etc.
    return f"0.{int(minor) + 15}.0"


if __name__ == "__main__":
    update_pip()
    install_pip_package("sglang[all]")
    install_pip_package("sgl-kernel")

    # sglang upgrades torch and installs a CPU-only torchvision from PyPI.
    # Reinstall the matching CUDA torchvision so its C extension links against
    # the correct torch ABI and operators like torchvision::nms are registered.
    tv_version = _matching_torchvision_version()
    torch_index = _torch_cuda_index()
    install_pip_package(f"torchvision=={tv_version}", index_url=torch_index)
