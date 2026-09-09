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

# Adapted from https://github.com/bazel-contrib/rules_pydeps/blob/1c3eae19c4cd4b854e91a6ea48e21666b08d7ecc/pydeps/private/py/python_module.py

from pathlib import Path

from typing_extensions import override


class PythonModule:
    def __init__(self, module: str) -> None:
        if not module:
            raise ValueError("Module name cannot be empty")
        self._module = module

    @classmethod
    def from_path(cls, path: Path) -> "PythonModule":
        """
        Convert a path to a particular file to the corresponding Python module name.
        """
        module = list(path.parts)

        if module[-1] == "__init__.py":
            module = module[:-1]
            # Special case __init__s, they're the only cases where the module can get shorter
            if not module:
                return cls(".")
        elif module[-1] == "__init__.pyi":
            module = module[:-1]
            if not module:
                return cls(".")
        elif path.suffix == ".py":
            module[-1] = module[-1].removesuffix(".py")
        elif path.suffix == ".pyi":
            module[-1] = module[-1].removesuffix(".pyi")
        elif path.suffix == ".mojo":
            module[-1] = module[-1].removesuffix(".mojo")
        elif path.suffix == ".so":
            # these are files of the form:
            #   lxml/etree.cpython-310-darwin.so
            # and these capture the module
            #   lxml.etree
            # so we extract the `etree` component of the shared lib filename
            root, _ = module[-1].split(".", maxsplit=1)
            module[-1] = root
        else:
            raise ValueError(f"Unsupported module path: {path}")

        return cls(".".join(module))

    def has_parent(self) -> bool:
        return "." in self._module

    def parent(self) -> "PythonModule":
        return PythonModule(".".join(self._module.split(".")[:-1]))

    def root(self) -> str:
        return self._module.split(".")[0]

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, PythonModule) and self._module == other._module

    @override
    def __hash__(self) -> int:
        return hash(self._module)

    @override
    def __repr__(self) -> str:
        return f"PythonModule({self._module})"

    @override
    def __str__(self) -> str:
        return self._module
