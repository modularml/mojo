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

import json
import os
import sys
import venv
from pathlib import Path
from typing import Any

_PEP_491_SHEBANG = b"#!python"


def _is_shared_lib(p: Path) -> bool:
    # Linux: libfoo.so or libfoo.so.N[.M]; macOS: libfoo.dylib
    return p.suffix in (".dylib", ".so") or ".so." in p.name


def _write_pth(site_packages: Path, imports: list[str]) -> None:
    pth_imports = []
    for imp in imports:
        path = Path(imp)
        if path.parts[0] == "_main":
            pth_imports.append(Path(*path.parts[1:]).absolute().as_posix())
        elif not path.parts[0].startswith("rules_pycross"):
            # Add imports for first parties and third parties that are included
            # through bazel like protobuf and grpc
            pth_imports.append((Path("..") / path).resolve().as_posix())
    (site_packages / "modular_venv.pth").write_text(
        "\n".join(pth_imports) + "\n"
    )


def _create_symlink(src: Path, dest: Path, overwrite: bool = False) -> None:
    if overwrite:
        dest.unlink(missing_ok=True)

    # NOTE: Ignore duplicate files that would end up in the same place. First one wins.
    if dest.exists():
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.symlink_to(src.resolve())


def _symlink_files(
    venv_path: Path, site_packages: Path, manifest: dict[str, Any]
) -> None:
    for src in manifest["files"]:
        src = Path(src)
        if src.is_dir():
            paths = [x for x in src.glob("**/*") if not x.is_dir()]
        else:
            paths = [src]

        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"error: '{path}': does not exist")

            assert path.as_posix().startswith("../")
            actual_file = path.relative_to("..")

            has_import = False
            for imp in manifest["imports"]:
                if actual_file.as_posix().startswith(imp):
                    has_import = True
                    dependency_base_dir = actual_file.relative_to(imp)
                    destination = site_packages / dependency_base_dir
                    _create_symlink(path, destination)

            if not has_import:
                # Assume all remaining files are from pycross deps
                dependency_base_dir = Path(
                    *actual_file.relative_to(
                        "rules_pycross++lock_file+modular_pip_lock_file_repo/deps"
                    ).parts[1:]
                )
                if dependency_base_dir.parts[0] == "bin":
                    destination = venv_path / dependency_base_dir
                    contents = path.read_bytes()
                    if path.read_bytes().startswith(_PEP_491_SHEBANG):
                        shebang = f"#!{venv_path / 'bin/python3'}".encode()
                        contents = shebang + contents[len(_PEP_491_SHEBANG) :]
                        destination.write_bytes(contents)
                        destination.chmod(0o755)
                    else:
                        _create_symlink(path, destination)
                elif dependency_base_dir.parts[0] == "data":
                    destination = venv_path / Path(
                        *dependency_base_dir.parts[1:]
                    )
                    _create_symlink(path, destination)
                else:
                    raise FileNotFoundError(
                        f"error: not sure where to link '{actual_file}' please report this"
                    )

    for src in manifest["data_files"]:
        src = Path(src)
        assert not src.is_dir(), (
            f"error: data_files cannot be directories: '{src}'"
        )
        if src.suffix == ".mojoc":
            _create_symlink(src, venv_path / "lib/mojo" / src.name)
        elif _is_shared_lib(src):
            _create_symlink(src, venv_path / "lib" / src.name)
        elif src.read_bytes().startswith(b"#!/usr/bin/env python3"):
            # Skip py_binary data dependency
            continue
        else:
            _create_symlink(src, venv_path / "bin" / src.name)


def _create_venv(manifest: dict[str, Any], venv_path: Path) -> None:
    builder = venv.EnvBuilder(clear=True, symlinks=True, with_pip=True)
    builder.create(venv_path.as_posix())

    site_packages = (
        venv_path
        / "lib"
        / f"python{sys.version_info[0]}.{sys.version_info[1]}"
        / "site-packages"
    )
    if not site_packages.exists():
        raise FileNotFoundError(
            f"error: could not find site-packages at '{site_packages}'"
        )

    _write_pth(site_packages, manifest["imports"])
    _symlink_files(venv_path, site_packages, manifest)
    short_venv_path = Path(os.environ["BUILD_WORKSPACE_DIRECTORY"]) / ".venv"
    _create_symlink(venv_path, short_venv_path, overwrite=True)
    print(
        f"Created virtual environment at:\n\n{venv_path}\n\nActivate it with:\n\nsource {venv_path / 'bin/activate'}\n\nOr:\n\nsource {short_venv_path / 'bin/activate'}\n"
    )


if __name__ == "__main__":
    manifest_path = os.environ["VENV_MANIFEST"]
    venv_path = (
        Path(os.environ["BUILD_WORKSPACE_DIRECTORY"]) / os.environ["VENV_NAME"]
    )

    with open(manifest_path) as f:
        manifest = json.load(f)
    _create_venv(manifest, venv_path)
