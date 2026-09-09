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

import argparse
import json
import os
import sys
from pathlib import Path

from jupyter_client.kernelspec import KernelSpecManager


def create_argparser() -> argparse.ArgumentParser:
    """Helper for CL option definition and parsing logic."""
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(help="sub-command help", dest="command")
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--mojo-install-channel",
        type=str,
        help="The package channel in which Mojo was installed.",
        default="stable",
    )

    parser_install = subparsers.add_parser(
        "install",
        help="Install the mojo jupyter kernel",
        parents=[common_parser],
    )
    parser_install.add_argument(
        "--python",
        default=sys.executable,
        help="The python interpreter to use when launching the kernel.",
    )
    parser_install.add_argument(
        "--no-user",
        dest="user",
        action="store_false",
        help="Install kernel to system-wide location",
    )
    parser_install.add_argument(
        "--modular-home", type=str, help="Modular home path"
    )
    parser_install.set_defaults(user=True, modular_home="")

    subparsers.add_parser(
        "uninstall",
        help="Uninstall the mojo jupyter kernel",
        parents=[common_parser],
    )

    return parser


def install_kernel(
    python: str,
    user: bool,
    modular_home: str,
    kernel_name: str,
    install_channel: str,
) -> None:
    """Install the kernel spec."""
    kernel_dir = Path(__file__).parent / "kernel"
    kernel_install_dir = Path(
        KernelSpecManager().install_kernel_spec(
            str(kernel_dir), kernel_name, user
        )
    )

    if modular_home == "":
        modular_home = os.environ.get("MODULAR_HOME")
        if not modular_home:
            modular_home = os.environ.get("MODULAR_DERIVED_PATH")
        if not modular_home:
            raise RuntimeError("unable to resolve MODULAR_HOME path")

    # Generate the kernel.json file.
    logo_name = "nightly-logo" if "nightly" in install_channel else "logo"
    kernel_json = {
        "display_name": "Mojo"
        + (f" ({install_channel})" if install_channel != "stable" else ""),
        "argv": [
            python,
            str(kernel_install_dir / "mojokernel.py"),
            "-f",
            "{connection_file}",
            "--modular-home",
            str(modular_home),
        ],
        "language": "mojo",
        "codemirror_mode": "mojo",
        "language_info": {
            "name": "mojo",
            "mimetype": "text/x-mojo",
            "file_extension": ".mojo",
            "codemirror_mode": {"name": "mojo"},
        },
        "resources": {
            "logo-64x64": str(kernel_install_dir / f"{logo_name}-64x64.png"),
            "logo-svg": str(kernel_install_dir / f"{logo_name}.svg"),
        },
    }
    kernel_json_path = kernel_install_dir / "kernel.json"
    kernel_json_path.write_text(json.dumps(kernel_json, indent=2))


def uninstall_kernel(kernel_name: str) -> None:
    """Uninstall the kernel spec."""

    try:
        KernelSpecManager().remove_kernel_spec(kernel_name)
    except Exception as e:
        print(e)


def main() -> None:
    parser = create_argparser()
    args = parser.parse_args()

    # Handle the installation channel. If present, prefix the kernel name with
    # the channel name.
    if args.mojo_install_channel == "stable":
        kernel_name = "mojo-jupyter-kernel"
    else:
        kernel_name = f"mojo-{args.mojo_install_channel}-jupyter-kernel"

    if args.command == "install":
        install_kernel(
            args.python,
            args.user,
            args.modular_home,
            kernel_name,
            args.mojo_install_channel,
        )
    elif args.command == "uninstall":
        uninstall_kernel(kernel_name)
    else:
        raise Exception(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
