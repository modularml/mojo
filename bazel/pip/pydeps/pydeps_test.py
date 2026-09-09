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
from pathlib import Path

from import_dependency_compare import check_dependencies_against_imports
from python_module import PythonModule


def _rerender_label(label: str) -> str:
    """Rerenders a Bazel label to a more human-readable form."""
    if label.startswith("@@rules_pycross++"):
        # Format: @@rules_pycross++lock_file+modular_pip_lock_file_repo//deps:pytest@8.2.2
        # TODO: Dedup this with the logic in get_imports.py
        return label.split(":")[-1].split("@")[0]
    if label.startswith("@@//"):
        label = label[2:]
    package, target = label.split(":", maxsplit=1)
    last_package = package.split("/")[-1]
    if last_package == target:
        # //foo/bar:bar -> //foo/bar
        return package
    return label


def main() -> int:
    # Mapping from source files to target labels
    deps_info: dict[PythonModule, str] = {}
    no_src_deps: set[str] = set()

    with open(os.environ["PYDEPS_TEST_ARGS_FILE"]) as f:
        json_content = json.load(f)

    for label, srcs in json_content["DEP_SOURCES"].items():
        if not srcs:
            no_src_deps.add(label)
            continue
        for src in srcs:
            deps_info[PythonModule.from_path(Path(src))] = label

    target_srcs = {Path(src) for src in json_content["TARGET_SOURCES"]}

    working_dir = (
        Path(json_content["WORKING_DIR"]).resolve().relative_to(Path.cwd())
    )

    third_party_deps = set(json_content["THIRD_PARTY_DEPS"])
    ignore_extra_deps = set(json_content["IGNORE_EXTRA_DEPS"])
    ignore_unresolved_imports = set(json_content["IGNORE_UNRESOLVED_IMPORTS"])

    target_label = json_content.get("TARGET_LABEL")
    assert target_label
    target_label = target_label[: -len(".pydeps_test")]

    final_srcs: set[Path] = set()
    adjust_working_dir = False
    for tsrc in target_srcs:
        if tsrc.is_relative_to(working_dir):
            final_srcs.add(tsrc.relative_to(working_dir))
        else:
            new_source = Path(str(tsrc).removeprefix("oss/modular/"))
            # Some targets are actually in oss/modular/bazel/..., but are referenced
            # as bazel/... In this case, rewrite the sources and working directory.
            if new_source.is_relative_to(working_dir):
                final_srcs.add(new_source.relative_to(working_dir))
                adjust_working_dir = True
            # Ignore .mojo and .so files that aren't in the working dir, since they may be unrelated.
            elif new_source.suffix not in (".mojo", ".so"):
                raise ValueError(
                    f"Source file {tsrc} is not in working directory {working_dir}"
                )

    if adjust_working_dir:
        working_dir = "oss/modular" / working_dir

    unresolved_imports, unused_deps = check_dependencies_against_imports(
        working_dir.absolute(),
        final_srcs,
        target_srcs,
        third_party_deps,
        ignore_extra_deps,
        ignore_unresolved_imports,
        deps_info,
        no_src_deps,
    )

    result = 0

    if unresolved_imports:
        print(
            f"{_rerender_label(target_label)} has imports that could not be mapped to dependencies. "
            "These are likely coming from transitive dependencies that need to be explicitly declared. "
            "If these imports are optional at runtime, consider adding them to the `ignore_unresolved_imports` attribute of the target."
        )
        for imp in unresolved_imports:
            print(f"  {imp}")
        result = 1

    if unused_deps:
        print(
            f"{_rerender_label(target_label)} has unused dependencies. "
            "If these are actually necessary at runtime, consider also adding them to the `ignore_extra_deps` attribute of the target."
        )
        for dep in unused_deps:
            print(f"  {_rerender_label(dep)}")
        result = 1

    return result


if __name__ == "__main__":
    sys.exit(main())
