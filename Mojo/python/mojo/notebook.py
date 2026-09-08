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
import tempfile
from pathlib import Path

try:
    # Don't require including IPython as a dependency
    from IPython.core.magic import register_cell_magic  # type: ignore
except ImportError:

    def register_cell_magic(fn):  # noqa: ANN001, ANN201
        return fn


from .paths import MojoCompilationError
from .run import subprocess_run_mojo


@register_cell_magic
def mojo(line, cell) -> None:  # noqa: ANN001
    """A Mojo cell.

    Usage:
        - Run Mojo code in a cell

            ```mojo
            %%mojo
            def main():
                print("Hello from Mojo!")
            ```

        - Compile a python extension SO file

            ```mojo
            %%mojo build --emit shared-lib -o mojo_module.so

            from python import PythonObject
            from python.bindings import PythonModuleBuilder
            from os import abort

            @export
            def PyInit_mojo_module() abi("C") -> PythonObject:
                try:
                    var m = PythonModuleBuilder("thing")
                    m.def_function[hello]("hello", docstring="Hello!")
                    return m.finalize()
                except e:
                    abort(String("error creating Python Mojo module:", e))

            def hello() -> PythonObject:
                return "Hello from Mojo!"
            ```

            then in another cell

            ```python
            from mojo_module import hello

            hello()
            ```

        - Compile a package for kernel development.
            The following produces a `kernels.mojoc` which may be included
            as custom ops in a graph via the `custom_extensions` mechanism.

            ```mojo
            %%mojo precompile -o kernels.mojoc

            from gpu.host import DeviceContext
            from tensor import InputTensor, ManagedTensorSlice, OutputTensor

            @extensibility.register("histogram")
            struct Histogram:
                @staticmethod
                fn execute[
                    target: StaticString
                ](
                    output: OutputTensor[dtype = DType.int64, rank=1],
                    input: InputTensor[dtype = DType.uint8, rank=1],
                    ctx: DeviceContext,
                ) raises:
                    ...
            ```

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", default="run")

    args, extra_args = parser.parse_known_args(line.strip().split())

    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir)
        mojo_path = path / "cell.mojo"
        with open(mojo_path, "w") as f:
            f.write(cell)
        (path / "__init__.mojo").touch()

        input_path = path if args.command == "package" else mojo_path
        command = [
            args.command,
            str(input_path),
            *extra_args,
        ]

        result = subprocess_run_mojo(command, capture_output=True)

    if not result.returncode:
        print(result.stdout.decode())
    else:
        raise MojoCompilationError(
            input_path, command, result.stdout.decode(), result.stderr.decode()
        )
