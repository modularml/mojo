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
from __future__ import annotations

import contextlib
import copy
import csv
import ctypes
import functools
import glob
import json
import logging
import math
import multiprocessing
import os
import queue
import shutil
import signal
import string
import subprocess
import sys
import threading
from collections import deque
from collections.abc import Generator, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import product
from multiprocessing import Pool
from pathlib import Path
from subprocess import list2cmdline
from time import time
from typing import Any

# Under ubsan, force the `spawn` start method so the libubsan-preloaded
# parent never forks. With fork, the child inherits libubsan's runtime
# state plus any queue feeder threads from the parent and intermittently
# wedges in popen_fork._launch. Spawn launches a fresh interpreter via
# execve, sidestepping both interactions. Gated on KBENCH_LIBUBSAN_PATH,
# which the kbench bazel rules export only under --config=ubsan
# (MOTO-1576).
if os.environ.get("KBENCH_LIBUBSAN_PATH"):
    multiprocessing.set_start_method("spawn", force=True)


@contextlib.contextmanager
def _redirect_output(
    stdout_path: Path, stderr_path: Path
) -> Generator[None, None, None]:
    """Redirect OS-level stdout/stderr to named files on disk.

    Uses raw os.open() fds (not Python file objects) to avoid
    BufferedWriter __del__ double-close in forked workers.
    """
    _OPEN_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    stdout_fd = os.open(str(stdout_path), _OPEN_FLAGS, 0o644)
    stderr_fd = os.open(str(stderr_path), _OPEN_FLAGS, 0o644)
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        yield
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(stdout_fd)
        os.close(stderr_fd)


import numpy as np
import pandas as pd
import utils
import yaml
from rich.progress import (
    Progress,
)

ScalarValue = str | int | float | bool

_WRAPPER_SOURCE = """\
from {module_name} import main as _bench_main
from std.builtin._startup import _ensure_runtime_init


@export
def benchmark_entry() abi("C") -> Int32:
    # Shared libraries don't get the __wrap_and_execute_main
    # startup that executables do, so the Mojo async runtime is
    # never registered.  Benchmarks that use CPU parallelism
    # (e.g. elementwise) will abort on a null Runtime* without
    # this.  The call is idempotent — a no-op after the first.
    _ensure_runtime_init()
    try:
        _bench_main()
        return 0
    except:
        return 1
"""


@dataclass
class ProcessOutput:
    stdout: str | None = None
    stderr: str | None = None
    return_code: int = -1
    path: Path | None = None

    def log(self) -> None:
        if self.stdout:
            logging.debug("output " + self.stdout + utils.LINE)
        if self.stderr:
            logging.debug("error " + self.stderr + utils.LINE)


# TODO: remove and replace directly with subprocess.run
def _run_cmdline(
    cmd: Sequence[str],
    dryrun: bool = False,
    timeout: int | None = None,
    env: dict[str, str] | None = None,
) -> ProcessOutput:
    """Execute a shell command with error handling."""
    if env is None:
        env = {}
    try:
        if dryrun:
            print(list2cmdline(cmd))
            return ProcessOutput(None, None, -1, None)

        # Pass the current environment to subprocess, including MODULAR_MOJO_MAX_IMPORT_PATH
        _env = os.environ.copy()
        _env.update(dict(env))
        if timeout is None:
            output = subprocess.run(
                cmd, check=False, capture_output=True, env=_env
            )
        else:
            try:
                output = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    env=_env,
                    timeout=timeout,
                )
            except Exception as e:
                return ProcessOutput(None, str(e), os.EX_OSERR)

        return ProcessOutput(
            output.stdout.decode("utf-8"),
            output.stderr.decode("utf-8"),
            output.returncode,
        )

    except Exception as exc:
        return ProcessOutput(
            None,
            f"Unable to run command {list2cmdline(cmd)}: {exc}",
            os.EX_OSERR,
        )


@dataclass(frozen=True)
class Lang:
    name: str
    extensions: list[str]
    path: str
    needs_compilation: bool


# TODO: enabled cached property option
# @functools.cached_property
# @staticmethod
def mojo_binary() -> str:
    """Find mojo binary in PATH."""
    # Check for Bazel-provided mojo binary first
    if mojo_path := os.environ.get("MODULAR_MOJO_MAX_DRIVER_PATH"):
        if os.path.exists(mojo_path):
            return mojo_path
        else:
            raise FileNotFoundError(
                f"MODULAR_MOJO_MAX_DRIVER_PATH '{mojo_path}' does not exist."
            )
    # Fall back to searching in PATH
    if mojo := shutil.which("mojo"):
        return mojo
    raise FileNotFoundError("Could not find the `mojo` binary.")


def python_binary() -> str:
    """Find python binary in PATH."""
    return sys.executable


class SupportedLangs:
    MOJO = Lang("mojo", [".mojo"], mojo_binary(), needs_compilation=True)
    PYTHON = Lang("python", [".py"], python_binary(), needs_compilation=False)

    @staticmethod
    def which_executor(file: Path) -> Lang:
        if file.suffix in SupportedLangs.PYTHON.extensions:
            return SupportedLangs.PYTHON
        elif file.suffix in SupportedLangs.MOJO.extensions:
            return SupportedLangs.MOJO
        else:
            raise ValueError(f"Extension {file.suffix} is not supported!")


@dataclass
class Param:
    name: str
    value: ScalarValue

    def define(self, lang: Lang) -> list[str]:
        """Generate command line arguments for this parameter."""

        if lang == SupportedLangs.MOJO:
            if self.name.startswith("$"):
                var_name = self.name.removeprefix("$")
                return [f"--{var_name}={self.value}"]
            return ["-D", f"{self.name}={self.value}"]
        if lang == SupportedLangs.PYTHON:
            var_name = self.name.removeprefix("$")
            return [f"--{var_name}={self.value}"]
        return [""]


@dataclass
class ParamSpace:
    name: str
    value: ScalarValue | list[ScalarValue] | None
    value_set: list[ScalarValue] = field(default_factory=list)
    length: int = 0

    def __post_init__(self) -> None:
        """Initialize value set from flattened values."""
        values: list[ScalarValue]
        if not isinstance(self.value, list):
            values = [self.value] if self.value is not None else []
        else:
            values = self.value
        # Try evaluating values as arithmetic expressions:
        try:
            values = [eval(str(x)) for x in values]
        except Exception:
            pass
        # Note: as of python3.7+ the built-in dict is guaranteed to maintain insertion order.
        self.value_set = list(dict.fromkeys(utils.flatten(values)))
        self.value = None
        self.length = len(self.value_set)


# Singleton build failed state
@dataclass(frozen=True)
class _BuildFailed:
    pass


BuildFailed = _BuildFailed()


class KBENCH_MODE(Enum):
    RUN = auto()
    BUILD = auto()
    BUILD_AND_RUN = auto()


class KbenchCache:
    """Cache for compiled binaries."""

    def __init__(
        self,
        path: Path | str = "kbench_cache.pkl",
        base_dir: Path | None = None,
    ) -> None:
        self.base_dir = base_dir
        if base_dir:
            self.path = Path(base_dir) / "kbench_cache.pkl"
            os.makedirs(base_dir, exist_ok=True)
        else:
            self.path = Path(path)
        self.data: dict[str, str | _BuildFailed] = {}
        self.is_active = False

    def clear(self) -> None:
        """Remove cache file if it exists."""
        logging.debug(f"Removing kbench-cache: {self.path}")
        if self.path.exists():
            subprocess.run(["rm", str(self.path)])

    def load(self) -> None:
        """Load cache from file."""
        if self.path.exists():
            self.data = utils.load_pickle(self.path)
        self.is_active = True

    def dump(self) -> None:
        """Save cache to file."""
        if self.is_active and self.data:
            utils.store_pickle(self.path, self.data)

    def query(self, key: str) -> str | _BuildFailed | None:
        """Get cached path for given key if it exists."""
        if not self.is_active:
            return None
        obj_path = self.data.get(key)
        if isinstance(obj_path, str):
            if self.base_dir:
                resolved = str((Path(self.base_dir) / obj_path).resolve())
            else:
                resolved = obj_path
            return resolved if Path(resolved).exists() else None
        return obj_path

    def store(self, key: str, obj_path: Path) -> Path | None:
        """Store object in cache and return its new path."""
        if not self.is_active:
            return None
        # TODO: revise the following conflict.
        if key in self.data:
            logging.debug(f"overwriting {key} already in obj-cache")
        if self.base_dir:
            self.data[key] = os.path.relpath(str(obj_path), str(self.base_dir))
        else:
            self.data[key] = str(obj_path)
        return obj_path

    def store_failed(self, key: str) -> None:
        """Store build failure result for the specified key."""
        if not self.is_active:
            return
        # TODO: revise the following conflict.
        if key in self.data:
            logging.debug(f"overwriting {key} already in obj-cache")
        self.data[key] = BuildFailed


@dataclass(frozen=True)
class SpecInstance:
    name: str
    file: Path
    executor: Lang
    params: list[Param] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.params)

    @functools.cached_property
    def _get_defines(self) -> list[str]:
        defines = []
        for param in self.params:
            if not param.name.startswith("$"):
                defines.append(param.define(self.executor))

        return [item for sublist in defines for item in sublist]

    @functools.cached_property
    def _get_vars(self) -> list[str]:
        vars = []
        for param in self.params:
            if param.name.startswith("$"):
                vars.append(param.define(self.executor))

        return [item for sublist in vars for item in sublist]

    def build(
        self,
        *,
        output_dir: Path,
        build_opts: Sequence[str],
        dryrun: bool = False,
        idx: int = -1,
        enable_logging: bool = True,
    ) -> ProcessOutput:
        """Build the spec instance. Use set of compile-time
        parameters as path of the compiled binary and store
        the executable in 'output_dir'.
        """

        bin_name = self.hash(with_variables=False)
        bin_path = output_dir / Path(bin_name)

        if enable_logging:
            logging.info(f"building [{idx}][{bin_name}]")
            logging.debug(
                f"defines: {self._get_defines}"
                + "\n"
                + f"vars   : {self._get_vars}"
            )

        executor = self.executor

        if not executor.needs_compilation:
            return ProcessOutput(return_code=os.EX_OK, path=self.file)

        if executor == SupportedLangs.MOJO:
            cmd = [executor.path]
            cmd.extend(["build"])
            cmd.extend(build_opts)
            cmd.extend(
                [
                    *self._get_defines,
                    str(self.file),
                    "-o",
                    str(bin_path),
                ]
            )
            out = _run_cmdline(cmd, dryrun)
            if out.return_code == os.EX_OK:
                out.path = bin_path
            else:
                out.path = None
            return out

        return ProcessOutput()

    def build_shared_lib(
        self,
        *,
        output_dir: Path,
        build_opts: Sequence[str],
        dryrun: bool = False,
        idx: int = -1,
        enable_logging: bool = True,
    ) -> ProcessOutput:
        """Build the spec instance as a shared library (.so).

        Generates a wrapper .mojo file that imports main() from the benchmark
        and exports benchmark_entry() as a C-ABI symbol, then compiles to .so.
        """
        bin_name = self.hash(with_variables=False)
        so_path = output_dir / Path(bin_name + ".so")

        if enable_logging:
            logging.info(f"building .so [{idx}][{bin_name}]")
            logging.debug(
                f"defines: {self._get_defines}"
                + "\n"
                + f"vars   : {self._get_vars}"
            )

        executor = self.executor

        if not executor.needs_compilation:
            return ProcessOutput(return_code=os.EX_OK, path=self.file)

        if executor == SupportedLangs.MOJO:
            # Generate wrapper file that imports and calls the benchmark's main().
            module_name = Path(self.file).with_suffix("").name
            wrapper_source = _WRAPPER_SOURCE.format(module_name=module_name)
            wrapper_path = output_dir / f"{bin_name}_wrapper.mojo"
            wrapper_path.write_text(wrapper_source)

            cmd = [executor.path, "build", "--emit", "shared-lib"]
            cmd.extend(["-I", str(Path(self.file).parent)])
            cmd.extend(build_opts)
            cmd.extend(
                [
                    "-D",
                    "KBENCH_USE_ENV_ARGS=True",
                    *self._get_defines,
                    str(wrapper_path),
                    "-o",
                    str(so_path),
                ]
            )
            out = _run_cmdline(cmd, dryrun)
            if out.return_code == os.EX_OK:
                out.path = so_path
            else:
                out.path = None
            return out

        return ProcessOutput()

    def execute(
        self,
        binary_path: Path,
        output_file: Path,
        dryrun: bool = False,
        exec_prefix: list[str] = [],  # noqa: B006
        exec_suffix: list[str] = [],  # noqa: B006
        env: dict[str, str] = {},  # noqa: B006
        timeout_secs: int | None = None,
    ) -> ProcessOutput:
        if self.executor == SupportedLangs.PYTHON:
            exec_prefix = exec_prefix + [self.executor.path]
            vars = self._get_defines + self._get_vars
        else:
            vars = self._get_vars

        cmd = []
        if exec_prefix:
            logging.debug(f"exec-prefix: {exec_prefix}")
            cmd.extend(exec_prefix)
        cmd.extend([str(binary_path), *vars, "-o", str(output_file)])
        if exec_suffix:
            cmd.extend(exec_suffix)
            logging.debug(f"exec-suffix: {exec_suffix}")
        out = _run_cmdline(cmd, dryrun, timeout=timeout_secs, env=env)
        return out

    def to_obj(self) -> dict[str, Any]:
        return {param.name: param.value for param in self.params}

    @functools.cached_property
    def file_stem(self) -> str:
        return Path(self.file).with_suffix("").stem

    def __str__(self) -> str:
        tokens = [self.file_stem]
        for param in self.params:
            tokens.append(f"{param.name}={param.value}")
        return "/".join(tokens)

    def hash(self, with_variables: bool = True) -> str:
        MAX_FILENAME_LEN = 224

        tokens = [self.file_stem]
        for param in self.params:
            name = param.name
            # just use compile-time parameters and ignore runtime variables.
            if name.startswith("$") and not with_variables:
                continue
            name = name.replace("$", "")
            tokens.append(f"{name}-{param.value}")

        hash_str = "_".join(tokens)
        if len(hash_str) < MAX_FILENAME_LEN:
            return hash_str
        else:
            MAX_HASH_DIGITS = 8
            hash_hex = hash(hash_str) % (10**MAX_HASH_DIGITS)
            return f"{hash_str[: MAX_FILENAME_LEN - MAX_HASH_DIGITS]}{hash_hex}"


class GridSearchStrategy:
    instances: list[SpecInstance] = field(default_factory=list)

    def __init__(
        self, name: str, file: Path, params: list[list[ParamSpace]]
    ) -> None:
        self.instances: list[SpecInstance] = []

        # Expand the product of all the param:value-set's per each group of parameters
        for cfg in params:
            name_list = [p.name for p in cfg]
            param_list = [p.value_set for p in cfg]
            param_mesh = list(product(*param_list))
            num_params = len(cfg)
            for idx in range(len(param_mesh)):
                s = SpecInstance(
                    name=name,
                    file=file,
                    params=[
                        Param(name=name_list[i], value=param_mesh[idx][i])
                        for i in range(num_params)
                    ],
                    executor=SupportedLangs.which_executor(file),
                )
                self.instances.append(s)

    def __iter__(self):
        self.offset = 0
        return self

    def __next__(self):
        # Stop condition
        if self.offset == len(self.instances):
            raise StopIteration

        res = self.instances[self.offset]
        self.offset += 1
        return res

    def __getitem__(self, i: int) -> SpecInstance:
        return self.instances[i]

    def __len__(self) -> int:
        return len(self.instances)

    def extend(self, other: GridSearchStrategy) -> None:
        self.instances.extend(other.instances)


@dataclass(repr=True)
class Spec:
    name: str = ""
    file: Path = Path("")
    params: list[list[ParamSpace]] = field(default_factory=list)
    mesh_idx: int = 0
    mesh: list[SpecInstance] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)

    @staticmethod
    def load_yaml(file: Path) -> Spec:
        """
        Loads the spec from a YAML file

        Args:
            file: the yaml file Path

        Returns:
            Spec: the spec
        """
        if not file.exists():
            raise FileNotFoundError(
                f'Unable to find the spec file at "{file}".'
            )
        try:
            logging.debug(f"Loading yaml [{file}]" + utils.LINE)
            return Spec.loads(file.read_text())
        except Exception as e:
            raise ValueError(f"Could not load spec from {file}\nException: {e}")  # noqa: B904

    @staticmethod
    def load_yaml_list(yaml_path_list: Sequence[Path]) -> Spec:
        spec = Spec.load_yaml(Path(yaml_path_list[0]))
        for yaml_path in yaml_path_list[1:]:
            spec.join(Spec.load_yaml(Path(yaml_path)))
        return spec

    @staticmethod
    def parse_params(param_list: Sequence[str]) -> dict[str, list[Any]]:
        """
        Parse the parameters as (key,value) dictionary.
        The parameters can be defined as follows:
        - `PARAM_NAME:PARAM_VALUE` (single value)
        - `PARAM_NAME:[PARAM_VALUE0, PARAM_VALUE1]` (Pythonic list of values)

        Args:
            param_list: a list of param-value's as strings/

        Returns:
            Spec: Dictionary of with extra param names as keys and param values.
        """
        d: dict[str, list[Any]] = {}
        IFS = ":"
        for p in param_list:
            name = ""
            val = ""
            if IFS in p:
                name, val = p.split(IFS)

            if name not in d:
                d[name] = []

            # This supports list of params per one definition
            # The following works for parsing a single-value, or a Pythonic list of values.
            vals = val.split(",")
            vals[0] = vals[0].strip("[")
            vals[-1] = vals[-1].strip("]")
            for i, v in enumerate(vals):
                v = v.strip()
                try:
                    vals[i] = eval(v)
                except:
                    vals[i] = v
            d[name].extend(vals)
        return d

    @staticmethod
    def _param_names_match(a: str, b: str) -> bool:
        """Check whether two param names refer to the same parameter.

        Names are considered equal when they match after stripping any
        leading ``$`` prefix, so ``$batch_size`` matches ``batch_size``
        and vice-versa.
        """
        return a.lstrip("$") == b.lstrip("$")

    def extend_params(self, param_list: Sequence[str]) -> None:
        # Expand with CLI params
        extra_params = self.parse_params(param_list)

        # For all params in each config, if a matching param exists
        # (with or without the '$' prefix), *replace* its value_set
        # with the CLI-provided values so that ``--param batch_size:"[1]"``
        # restricts the sweep to only that value.  When no match is found
        # a brand-new ParamSpace is appended.
        for cfg in self.params:
            for k, v in extra_params.items():
                found = False
                for ps in cfg:
                    if self._param_names_match(ps.name, k):
                        ps.value_set = list(dict.fromkeys(utils.flatten(v)))
                        ps.length = len(ps.value_set)
                        found = True
                        break
                if not found:
                    cfg.append(ParamSpace(k, v))

        self.setup_mesh()

    def extend_shape_params(self, param_set: Sequence[Param]) -> None:
        # TODO: check for collisions in param-names

        extra_params: list[ParamSpace] = []
        for ps in param_set:
            extra_params.append(ParamSpace(ps.name, ps.value))

        # add extended set of parameter to each bundle of parameters:
        for p in self.params:
            p.extend(extra_params)

        if not self.params:
            self.params = [extra_params]
        self.setup_mesh()

    def dump_yaml(self, out_path: Path) -> None:
        assert self.mesh, "There are no instances to write to YAML!"
        obj = {
            "name": self.name,
            "file": self.file,
            "params": [s.to_obj() for s in self.mesh],
        }
        with open(out_path, "w") as f:
            yaml.dump(obj, f, sort_keys=False)
        logging.debug(f"dumped {len(self.mesh)} instances to [{out_path}]")

    @staticmethod
    def loads(yaml_str: str) -> Spec:
        """
        Deserializes a Spec object from the given yaml string.

        Args:
            yaml_str: the yaml string representation of the model manifest

        Returns:
            Spec: a Spec loaded from the given yaml string
        """
        obj = yaml.safe_load(yaml_str)

        if "name" not in obj:
            logging.warning("Field [name] is not set in YAML")
        if "file" not in obj:
            logging.warning("Field [file] is not set in YAML")

        params: list[list[ParamSpace]] = []
        if "params" in obj:
            for cfg in obj["params"]:
                e: list[ParamSpace] = []
                for k, v in cfg.items():
                    if k == "metadata":
                        continue
                    e.append(ParamSpace(name=k, value=v))
                params.append(e)

        return Spec(
            name=obj.get("name", ""),
            file=obj.get("file", ""),
            params=params,
            rules=obj.get("rules", []),
        )

    def __len__(self) -> int:
        return len(self.mesh)

    def __post_init__(self):
        # checking if the file source path is valid
        file_abs_path = Path(
            string.Template(str(self.file)).substitute(os.environ)
        ).absolute()
        assert file_abs_path.exists(), (
            f"error: '{file_abs_path}' does not exist."
        )
        self.file = file_abs_path

        # setup mesh
        if self.params:
            self.setup_mesh()
        else:
            # default values for empty mesh
            self.mesh = [
                SpecInstance("", Path("./"), executor=SupportedLangs.MOJO)
            ]

    def setup_mesh(self) -> int:
        """
        Setup a mesh (cartesian product) of all values for all params. For example,
        if we have 2 set of params M=[64,256] and N=[A,B,C], the mesh will include
        to the following values:

        M=[64,256] x N=[A,B,C]
        ======================
        idx  : values
        0    : [64,A]
        1    : [64,B]
        2    : [64,C]
        3    : [256,A]
        4    : [256,B]
        5    : [256,C]

        At the end, append the configs with fixed parameters, if any exists in YAML.

        Return the total size of mesh.
        """
        grid_mesh = list(GridSearchStrategy(self.name, self.file, self.params))
        self.mesh = self.apply_rules(grid_mesh, self.rules)
        return len(self.mesh)

    def join(self, other: Spec) -> None:
        assert self.name == other.name
        assert self.file == other.file
        assert len(other.mesh) > 0

        self.mesh_idx = 0
        self.params.extend(other.params)
        self.mesh.extend(other.mesh)

    @staticmethod
    def apply_rules(
        mesh: Sequence[SpecInstance], rules: Sequence[str]
    ) -> list[SpecInstance]:
        new_mesh: list[SpecInstance] = []

        if not rules:
            return list(mesh)

        def remove_dlr(s: str) -> str:
            return s.replace("$", "")

        for s in mesh:
            valid = True
            for r in rules:
                # TODO: revise handling of $ in string.
                locals = {remove_dlr(p.name): p.value for p in s.params}
                r = remove_dlr(r)

                try:
                    e = eval(r, {}, locals)
                # the following exception is required in case a parameter
                # is present in rule and missing from spec-instance combination.
                except NameError:
                    e = True
                valid = valid & e
                if not valid:
                    break
            if valid:
                new_mesh.append(s)
        return new_mesh

    def filter(self, filter_list: Sequence[str]) -> None:
        filters: dict[str, list[str]] = {}
        for f in filter_list:
            if "=" in f:
                name, val = f.split("=")
            elif ":" in f:
                name, val = f.split(":")

            if name not in filters:
                filters[name] = []
            filters[name].append(val)

        # Skip filter keys that don't appear in any shape's params. An
        # unknown key (e.g. a function argument embedded in a benchmark name
        # that isn't a YAML parameter) would otherwise require num_filters
        # matches per instance, silently eliminating every shape.
        all_param_names: set[str] = {
            p.name for s in self.mesh for p in s.params
        }
        known_filters = {
            k: v
            for k, v in filters.items()
            if any(self._param_names_match(pn, k) for pn in all_param_names)
        }
        unknown_keys = sorted(set(filters) - set(known_filters))
        if unknown_keys:
            print(
                f"kbench --filter: ignoring unknown parameter(s) {unknown_keys}"
                " (not present in any YAML shape)",
                file=sys.stderr,
            )

        filtered_insts: list[SpecInstance] = []
        num_filters = len(known_filters)

        # Count the number of valid filters in each instance.
        # If the count==num_filters then add the instance to the result.
        valid_cnt = np.zeros(len(self.mesh), dtype=np.int32)

        for k_filter, v_filter in known_filters.items():
            for i, s in enumerate(self.mesh):
                for p in s.params:
                    if (
                        self._param_names_match(p.name, k_filter)
                        and str(p.value) in v_filter
                    ):
                        valid_cnt[i] += 1

        for i, idx in enumerate(valid_cnt):
            if idx == num_filters:
                filtered_insts.append(self.mesh[i])

        self.mesh = filtered_insts[:]
        self.mesh_idx = 0

    def __iter__(self):
        self.iter_offset = 0
        return self

    def __next__(self) -> SpecInstance:
        assert self.mesh is not None, (
            "Should call self.init_mesh after loading or in postinit."
        )

        # Stop condition
        if self.iter_offset == len(self.mesh):
            raise StopIteration

        # Retrieve and update self.mesh_idx
        idx = self.iter_offset
        self.iter_offset += 1
        return self.mesh[idx]

    def __repr__(self) -> str:
        rs = [f"[{i}] {str(s)}" for i, s in enumerate(self.mesh)]
        rs += [utils.LINE]
        rs += [f"Num Instances: {len(self.mesh)}"]
        rs += [utils.LINE]
        return "\n".join(rs)


@dataclass
class BuildItem:
    """
    To store all necessary details for building a spec item (instance).

    Args:
        idx: unique index of item in the list of scheduler items
        spec_instance: the parameter set used as the basis of build
        output_dir: output directory specific for this build item
        dryrun: set to True to enable dryrun
        use_shared_lib: set to True to build as shared library
        output_path: path to output file
        bin_path: path to executable binary
        build_output: output message for build
        build_elapsed_time: elapsed time for build
        exec_output: output message for exec
        exec_benchmark_time: measured time for executing the entire benchmark
    """

    idx: int
    spec_instance: SpecInstance
    output_dir: Path
    build_opts: list[str]
    dryrun: bool = False
    use_shared_lib: bool = False
    build_output_dir: Path | None = None
    output_path: Path = Path()
    bin_path: Path | None = None
    stdout_capture_path: Path = field(init=False)
    stderr_capture_path: Path = field(init=False)

    build_output: ProcessOutput = field(default_factory=ProcessOutput)
    build_elapsed_time: float = 0
    exec_output: ProcessOutput = field(default_factory=ProcessOutput)
    exec_benchmark_time: float = 0

    def __post_init__(self) -> None:
        spec_hash = self.spec_instance.hash(with_variables=True)
        self.stdout_capture_path = self.output_dir / f"{spec_hash}_stdout.log"
        self.stderr_capture_path = self.output_dir / f"{spec_hash}_stderr.log"


@dataclass(frozen=True)
class MkdirArgs:
    """Arguments for kbench_mkdir, passed through multiprocessing pool."""

    output_dir: Path
    output_suffix: str
    run_only: bool
    has_cache_dir: bool


@dataclass
class ExecItemTask:
    """Single benchmark item to execute."""

    build_item: BuildItem
    use_shared_lib: bool
    profile: str = ""
    exec_prefix: list[str] = field(default_factory=list)
    exec_suffix: list[str] = field(default_factory=list)


class ItemPool:
    """Thread-safe pool with .so affinity and work-stealing."""

    def __init__(self, binary_groups: list[list[ExecItemTask]]) -> None:
        self._lock = threading.Lock()
        self._binary_groups: deque[list[ExecItemTask]] = deque(
            sorted(binary_groups, key=len, reverse=True)
        )
        self._gpu_queues: dict[int | str, deque[ExecItemTask]] = {}

    def register_gpu(self, gpu_id: int | str) -> None:
        with self._lock:
            self._gpu_queues[gpu_id] = deque()

    def next_for(self, gpu_id: int | str) -> ExecItemTask | None:
        with self._lock:
            # 1. Try own per-GPU queue (items from current binary group).
            if self._gpu_queues[gpu_id]:
                return self._gpu_queues[gpu_id].popleft()
            # 2. Grab entire next binary group from global pool.
            if self._binary_groups:
                group = self._binary_groups.popleft()
                self._gpu_queues[gpu_id].extend(group)
                return self._gpu_queues[gpu_id].popleft()
            # 3. Work-steal: take from longest per-GPU queue.
            longest = max(
                self._gpu_queues, key=lambda g: len(self._gpu_queues[g])
            )
            if self._gpu_queues[longest]:
                return self._gpu_queues[longest].popleft()
            # 4. All done.
            return None


_libubsan_preloaded = False


def _maybe_preload_libubsan() -> None:
    """Load libubsan into the process before any user .so dlopen.

    Mojo's `libKGENCompilerRTShared.so` (which kbench-built .so's depend on)
    leaves `__ubsan_handle_*_abort` references unresolved under UBSan
    (`-Wl,-z,undefs` allows it; the executable normally provides them via
    static UBSan link). When kbench loads a user .so via `ctypes.CDLL` from
    a non-UBSan Python interpreter, those symbols can't resolve → dlopen
    fails with "undefined symbol".

    Fix: bring libubsan into the process's global symbol namespace by
    `ctypes.CDLL(..., mode=RTLD_GLOBAL)` once, before opening any user .so.
    Subsequent dlopens then resolve `__ubsan_handle_*` against the loaded
    libubsan.

    The path comes from the `KBENCH_LIBUBSAN_PATH` env var that the kbench
    `modular_py_binary` rule sets (and includes the .so in runfiles) only
    under `--config=ubsan` on Linux. No-op everywhere else. Tracks MOTO-1576.
    """
    global _libubsan_preloaded
    if _libubsan_preloaded:
        return
    libubsan_rel = os.environ.get("KBENCH_LIBUBSAN_PATH")
    if not libubsan_rel:
        return
    # `KBENCH_LIBUBSAN_PATH` is a runfiles-relative path emitted by bazel's
    # `$(rootpath ...)` substitution. Resolve via the runfiles helper so the
    # lookup works under both `bazel run` and `bazel test`. If resolution
    # fails the helper returns None and we let ctypes raise OSError so the
    # caller can produce a libubsan-specific diagnostic.
    from python.runfiles import runfiles

    r = runfiles.Create()
    libubsan_path = (
        r.Rlocation(libubsan_rel) if r is not None else None
    ) or libubsan_rel
    ctypes.CDLL(libubsan_path, mode=ctypes.RTLD_GLOBAL)
    _libubsan_preloaded = True
    logging.info(f"UBSan: preloaded libubsan from {libubsan_path}")


class _SharedLibExecutor:
    """Manages a cached ctypes shared library for worker-process benchmarks."""

    def __init__(self) -> None:
        self._so_path: Path | None = None
        self._lib: ctypes.CDLL | None = None

    def execute(self, bi: BuildItem) -> BuildItem:
        """Load the library (if changed) and run the benchmark."""
        assert bi.bin_path is not None
        if self._so_path != bi.bin_path:
            self._so_path = bi.bin_path
            try:
                _maybe_preload_libubsan()
            except OSError as e:
                kbench_libubsan_path = os.environ.get("KBENCH_LIBUBSAN_PATH")
                logging.error(
                    f"Failed to preload libubsan from"
                    f" KBENCH_LIBUBSAN_PATH={kbench_libubsan_path!r}: {e}"
                )
                self._lib = None
            else:
                try:
                    self._lib = ctypes.CDLL(str(self._so_path))
                    self._lib.benchmark_entry.restype = ctypes.c_int32
                except OSError as e:
                    logging.error(f"Failed to load {self._so_path}: {e}")
                    self._lib = None

        # Benchmark execution — redirect output to capture files.
        with _redirect_output(bi.stdout_capture_path, bi.stderr_capture_path):
            result = _worker_exec_shared_lib(bi, self._lib)

        # Read captured output after redirect is restored.
        result.exec_output.stdout = bi.stdout_capture_path.read_text(
            errors="replace"
        )
        result.exec_output.stderr = bi.stderr_capture_path.read_text(
            errors="replace"
        )
        return result


def _start_worker(
    gpu_id: int | str,
    visible_device_prefix: str,
    task_queue: multiprocessing.Queue[ExecItemTask | None],
    result_queue: multiprocessing.Queue[BuildItem],
) -> multiprocessing.Process:
    """Start a worker process for a specific GPU."""
    proc = multiprocessing.Process(
        target=_gpu_worker_loop,
        args=(gpu_id, visible_device_prefix, task_queue, result_queue),
        daemon=True,
    )
    proc.start()
    return proc


def _gpu_worker_loop(
    gpu_id: int | str,
    visible_device_prefix: str,
    task_queue: multiprocessing.Queue[ExecItemTask | None],
    result_queue: multiprocessing.Queue[BuildItem],
) -> None:
    """Worker loop running in a child process, one per GPU."""
    os.setpgrp()  # New process group so we can kill the entire tree
    if visible_device_prefix:
        os.environ[visible_device_prefix] = str(gpu_id)
    logging.debug(
        f"Worker pid={os.getpid()} assigned {visible_device_prefix}={gpu_id}"
    )

    executor = _SharedLibExecutor()
    while (task := task_queue.get()) is not None:
        bi = task.build_item
        try:
            if task.use_shared_lib:
                bi = executor.execute(bi)
            else:
                bi = Scheduler.execute_item(
                    build_item=bi,
                    profile=task.profile,
                    exec_prefix=task.exec_prefix,
                    exec_suffix=task.exec_suffix,
                )
        except Exception as e:
            bi.exec_output = ProcessOutput(
                return_code=1,
                stderr=f"Worker error: {type(e).__name__}: {e}",
            )
        result_queue.put(bi)


def _worker_exec_shared_lib(
    bi: BuildItem,
    lib: ctypes.CDLL | None,
) -> BuildItem:
    """Execute a single BuildItem via ctypes in the worker process."""
    if lib is None:
        bi.exec_output = ProcessOutput(
            return_code=1,
            stderr=f"Failed to load {bi.bin_path}",
        )
        return bi

    # Set env vars for runtime args ($-prefixed params).
    env_keys: list[str] = []
    for param in bi.spec_instance.params:
        if param.name.startswith("$"):
            var_name = param.name.removeprefix("$")
            key = f"KBENCH_ARG_{var_name}"
            os.environ[key] = str(param.value)
            env_keys.append(key)
    os.environ["KBENCH_OUTFILE"] = str(bi.output_path)

    t_start = time()
    rc = 1
    try:
        rc = lib.benchmark_entry()
    except Exception:
        pass

    bi.exec_benchmark_time = time() - t_start

    # Clean up env vars.
    os.environ.pop("KBENCH_OUTFILE", None)
    for key in env_keys:
        os.environ.pop(key, None)

    # stdout/stderr are captured from redirect files by the caller.
    bi.exec_output = ProcessOutput(return_code=rc)
    return bi


def _poll_result(
    result_queue: multiprocessing.Queue[BuildItem],
    proc: multiprocessing.Process,
    timeout_secs: int | None,
) -> tuple[BuildItem | None, str]:
    """Poll for a result, detecting worker death between attempts.

    Returns (result, status) where result is None on timeout or worker death.
    """
    deadline = None if timeout_secs is None else time() + timeout_secs
    while True:
        wait = 1.0
        if deadline is not None:
            remaining = deadline - time()
            if remaining <= 0:
                return None, "timed out"
            wait = min(wait, remaining)
        try:
            bi = result_queue.get(timeout=wait)
            status = (
                "succeeded"
                if bi.exec_output.return_code == os.EX_OK
                else "crashed"
            )
            return bi, status
        except queue.Empty:
            # Detect worker crash (e.g. signal killed the process)
            # so we don't wait the full timeout for a dead process.
            if not proc.is_alive():
                return None, f"worker died (exit code {proc.exitcode})"


def _respawn_worker(
    gpu_id: int | str,
    visible_device_prefix: str,
    proc: multiprocessing.Process,
    task_queue: multiprocessing.Queue[ExecItemTask | None],
    result_queue: multiprocessing.Queue[BuildItem],
) -> tuple[
    multiprocessing.Process,
    multiprocessing.Queue[ExecItemTask | None],
    multiprocessing.Queue[BuildItem],
]:
    """Terminate *proc* and return a fresh (proc, task_q, result_q) tuple."""
    pid = proc.pid
    if pid is not None:
        try:
            os.killpg(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
    else:
        proc.terminate()
    proc.join(timeout=5)
    if proc.is_alive():
        pid = proc.pid
        if pid is not None:
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
        else:
            proc.kill()
        proc.join(timeout=2)
    task_queue.close()
    result_queue.close()
    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    proc = _start_worker(
        gpu_id, visible_device_prefix, task_queue, result_queue
    )
    return proc, task_queue, result_queue


def _gpu_manager(
    gpu_id: int | str,
    visible_device_prefix: str,
    item_pool: ItemPool,
    timeout_secs: int | None,
    results_list: list[BuildItem],
    results_lock: threading.Lock,
    progress: Any,
    progress_task: Any,
    shutdown_event: threading.Event,
) -> None:
    """Manager thread for a single GPU. Owns one worker process."""
    task_queue: multiprocessing.Queue[ExecItemTask | None] = (
        multiprocessing.Queue()
    )
    result_queue: multiprocessing.Queue[BuildItem] = multiprocessing.Queue()
    proc = _start_worker(
        gpu_id, visible_device_prefix, task_queue, result_queue
    )

    while not shutdown_event.is_set():
        item = item_pool.next_for(gpu_id)
        if item is None:
            break

        task_queue.put(item)
        completed_bi, status = _poll_result(result_queue, proc, timeout_secs)

        if completed_bi is None:
            # Worker died or timed out — read any captured output.
            bi = item.build_item
            stdout = ""
            stderr = status
            if bi.stdout_capture_path.exists():
                stdout = bi.stdout_capture_path.read_text(errors="replace")
            if bi.stderr_capture_path.exists():
                captured_stderr = bi.stderr_capture_path.read_text(
                    errors="replace"
                )
                if captured_stderr:
                    stderr += "\n" + captured_stderr
            bi.exec_output = ProcessOutput(
                return_code=1, stdout=stdout, stderr=stderr
            )
            completed_bi = bi

        with results_lock:
            results_list[completed_bi.idx] = completed_bi
            progress.update(progress_task, advance=1)
            done = int(progress.tasks[progress_task].completed)
            total = int(progress.tasks[progress_task].total)
        logging.info(
            f"{status} [{done}/{total}] ({utils._percentage(done, total)}%)"
        )
        completed_bi.exec_output.log()

        completed_bi.stdout_capture_path.unlink(missing_ok=True)
        completed_bi.stderr_capture_path.unlink(missing_ok=True)

        if completed_bi.exec_output.return_code != os.EX_OK:
            proc, task_queue, result_queue = _respawn_worker(
                gpu_id,
                visible_device_prefix,
                proc,
                task_queue,
                result_queue,
            )

    # Clean shutdown.
    task_queue.put(None)
    proc.join(timeout=10)
    if proc.is_alive():
        pid = proc.pid
        if pid is not None:
            try:
                os.killpg(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                proc.terminate()
        else:
            proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            pid = proc.pid
            if pid is not None:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
            else:
                proc.kill()
            proc.join(timeout=2)
    task_queue.close()
    result_queue.close()


def _get_similar_files(path: Path) -> list[Path]:
    """Returns a list of files that belong to the same benchmark but are
    created by different processes, e.g. due to using mpirun
    """
    dir_name = os.path.dirname(path)
    stem = path.stem
    suffix = path.suffix
    pattern = os.path.join(dir_name, f"{stem}*{suffix}")
    return [Path(p) for p in sorted(glob.glob(pattern))]


class Scheduler:
    """
    Kbench singleton scheduler class to coordinate building and running all items in spec.

    Args:
        num_cpu: number of cpu's (cores) used for building items
        num_gpu: number of gpu's used for executing items
        build_items: list of spec items to build (BuildItem's)
        obj_cache: kbench obj-cache
        output_dir: parent output directory for all results
        num_specs: total number of spec items added to scheduler (to build+run)
    """

    num_cpu: int
    num_gpu: int
    build_items: list[BuildItem]
    obj_cache: KbenchCache
    run_only: bool
    output_suffix: str
    output_dir: Path
    num_specs: int
    num_unique_build_items: int = 0
    t_build_total: float = 0.0
    t_benchmark_total: float = 0.0
    t_elapsed_total: float = 0.0

    CHUNK_SIZE: int = 1

    def __init__(
        self,
        num_cpu: int,
        num_gpu: int,
        obj_cache: KbenchCache,
        run_only: bool,
        spec_list: list[SpecInstance],
        output_dir: Path,
        build_opts: list[str],
        dryrun: bool,
        output_suffix: str = "output.csv",
        progress: Progress = Progress(),  # noqa: B008
        use_shared_lib: bool = False,
        output_dir_list: list[Path] | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.num_cpu = num_cpu
        self.num_gpu = num_gpu
        if not (0 < num_gpu <= num_cpu):
            raise ValueError(
                "num_gpu must be greater than 0 and less than or equal to num_cpu."
            )
        self.obj_cache = obj_cache
        self.num_specs = len(spec_list)

        if output_dir_list is not None:
            # Per-item output dirs: assign sequential out_0/, out_1/, ...
            # within each base dir.
            dir_counters: dict[Path, int] = {}
            resolved_dirs: list[Path] = []
            for base_dir in output_dir_list:
                idx = dir_counters.get(base_dir, 0)
                resolved_dirs.append(base_dir / f"out_{idx}")
                dir_counters[base_dir] = idx + 1
            item_output_dirs = resolved_dirs
        else:
            item_output_dirs = [
                Path(f"{output_dir}/out_{i}") for i in range(self.num_specs)
            ]

        self.output_suffix = output_suffix
        self.output_dir = output_dir
        self.run_only = run_only
        self.cache_dir = cache_dir

        self.build_items = [
            BuildItem(
                idx=i,
                spec_instance=spec_list[i],
                output_dir=item_output_dirs[i],
                build_opts=build_opts,
                dryrun=dryrun,
                use_shared_lib=use_shared_lib
                and spec_list[i].executor == SupportedLangs.MOJO,
                build_output_dir=cache_dir,
                output_path=item_output_dirs[i] / output_suffix,
            )
            for i in range(self.num_specs)
        ]

        self._shutdown_event: threading.Event | None = None
        self.setup_build_pool()
        self.mk_output_dirs()
        self.progress = progress

    @staticmethod
    def kbench_mkdir(args: MkdirArgs) -> Path:
        """Run the following command:
        `mkdir -p {output_dir}`
        """

        path_exists: bool = os.path.exists(args.output_dir) and os.path.isdir(
            args.output_dir
        )
        if not args.run_only:
            if path_exists:
                logging.debug(
                    f"Output dir already exists, will be overwritten:"
                    f" [{args.output_dir}]"
                )
                # Check for existing output files and remove them (if any):
                existing_csv = _get_similar_files(
                    args.output_dir / args.output_suffix
                )
                for f in existing_csv:
                    os.remove(f)

            os.makedirs(args.output_dir, exist_ok=True)
        else:
            if not path_exists:
                if args.has_cache_dir:
                    # With --cache-dir, per-spec output dirs won't exist from
                    # a previous build phase — they only hold result CSVs.
                    os.makedirs(args.output_dir, exist_ok=True)
                else:
                    raise ValueError(
                        f"--run-only specified but output directory does not exist: {args.output_dir}"
                    )
        return args.output_dir

    def get_chunksize(self, num_elements: int) -> int:
        elements_per_cpu = math.ceil(num_elements / self.num_cpu)
        return min(elements_per_cpu, self.CHUNK_SIZE)

    def mk_output_dirs(self) -> None:
        """
        Make output directories for kbench results (one per spec-instance)
        """
        output_dir_list = [
            MkdirArgs(
                output_dir=b.output_dir,
                output_suffix=self.output_suffix,
                run_only=self.run_only,
                has_cache_dir=self.cache_dir is not None,
            )
            for b in self.build_items
        ]

        for r in self.build_pool.imap(
            self.kbench_mkdir,
            output_dir_list,
            chunksize=self.CHUNK_SIZE,
        ):
            logging.debug(f"mkdir [{r}]")
        logging.debug(
            "Created directories for all instances in spec." + utils.LINE
        )

    def schedule_unique_build_items(
        self,
    ) -> tuple[dict[str, int], dict[str, Path]]:
        # Stores items that need to be build (i.e. not in cache)
        unique_build_items: dict[str, int] = {}
        # Stores paths to real binaries that have been cached beforehand
        unique_build_paths: dict[str, Path] = {}

        for b in self.build_items:
            i = b.idx
            s = b.spec_instance
            bin_name = s.hash(with_variables=False)
            logging.debug(f"schedule [{i}][{bin_name}]")
            debug_msg = [
                f"defines: {s._get_defines}",
                f"vars   : {s._get_vars}",
            ]

            # first, check cache for build from previous rounds
            bin_path = self.obj_cache.query(bin_name)
            debug_msg += [f"In cache: {bool(bin_path)}"]
            if isinstance(bin_path, str):
                unique_build_paths[bin_name] = Path(bin_path)
            elif bin_path is BuildFailed:
                # This binary failed to build before and would just fail again.
                # Skip it.
                continue
            else:
                # Neither found in the cache, nor exists in the unique_build_items
                if bin_name not in unique_build_items:
                    unique_build_items[bin_name] = i
                    debug_msg += [f"Added to schedule (ref_idx=[{i}])"]
                else:
                    # Already in the unique_build_items list
                    idx = unique_build_items[bin_name]
                    debug_msg += [f"Currently in schedule (ref_idx=[{idx}])"]
            logging.debug("\n".join(debug_msg) + utils.LINE)
        return unique_build_items, unique_build_paths

    @staticmethod
    def _pool_build_wrapper(bi: BuildItem) -> BuildItem:
        t_start_item = time()
        build_fn = (
            bi.spec_instance.build_shared_lib
            if bi.use_shared_lib
            else bi.spec_instance.build
        )
        effective_output_dir = bi.build_output_dir or bi.output_dir
        bi.build_output = build_fn(
            output_dir=effective_output_dir,
            build_opts=bi.build_opts,
            dryrun=bi.dryrun,
            idx=bi.idx,
            enable_logging=False,
        )
        elapsed_ms = int((time() - t_start_item) * 1e3)
        bi.build_elapsed_time = elapsed_ms
        return bi

    def build_all(self) -> None:
        """
        Build all unique items scheduled by the scheduler.
        """
        t_start = time()

        unique_build_items_dict, unique_build_paths = (
            self.schedule_unique_build_items()
        )
        self.num_unique_build_items = len(unique_build_items_dict)

        if self.run_only and len(unique_build_items_dict) > 0:
            logging.error("Run only but not all binaries are found")
            raise ValueError(
                f"--run-only specified but {len(unique_build_items_dict)} binaries not found in cache. "
                "Please build first or remove --run-only flag."
            )

        unique_build_items = [
            self.build_items[i] for i in list(unique_build_items_dict.values())
        ]

        logging.info(
            f"scheduled {len(unique_build_items)} unique build items out of {self.num_specs}"
            + utils.LINE
        )
        if unique_build_items:
            obj_cache = self.obj_cache

            build_progress = self.progress.add_task(
                "build",
                total=len(unique_build_items),
                auto_refresh=False,
            )

            for cnt, b in enumerate(
                self.build_pool.imap(
                    self._pool_build_wrapper,
                    unique_build_items,
                    chunksize=self.CHUNK_SIZE,
                    # alternatively: self.get_chunksize(len(unique_build_items))
                )
            ):
                build_output = b.build_output
                # update the data with build_output result
                self.build_items[b.idx].build_output = build_output
                self.build_items[
                    b.idx
                ].build_elapsed_time = b.build_elapsed_time

                bin_name = b.spec_instance.hash(with_variables=False)

                num_unique_build_items = len(unique_build_items)
                logging.info(
                    f"build [{b.idx}][{bin_name}] ({utils._percentage(cnt + 1, num_unique_build_items)}%)"
                )

                # print build_output stdout and stderr using log function.
                build_output.log()

                # Try storing the executable in cache if:
                # - cache is active
                # - no error is reported in stderr
                # - build_output path is found
                if build_output.return_code == os.EX_OK and build_output.path:
                    binary_path = build_output.path
                    obj_cache.store(bin_name, binary_path)
                    unique_build_paths[bin_name] = binary_path
                else:
                    obj_cache.store_failed(bin_name)

                self.progress.update(build_progress, advance=1)
            logging.info(
                f"finished building {self.num_unique_build_items} unique items"
                + utils.LINE
            )

        self.close_build_pool()
        self.t_build_total = time() - t_start

        # update all build items with their binary path
        for b in self.build_items:
            bin_name = b.spec_instance.hash(with_variables=False)
            self.build_items[b.idx].bin_path = unique_build_paths.get(bin_name)

        # Log build summary
        seen: set[str] = set()
        build_succeeded = 0
        build_failed = 0
        for b in self.build_items:
            h = b.spec_instance.hash(with_variables=False)
            if h in seen:
                continue
            seen.add(h)
            if b.bin_path is not None:
                build_succeeded += 1
            else:
                build_failed += 1
        total = build_succeeded + build_failed
        parts = [f"{build_succeeded} succeeded"]
        if build_failed:
            parts.append(f"{build_failed} failed")
        logging.info(
            f"Build summary: {', '.join(parts)} ({total} unique items)"
        )

    @staticmethod
    def execute_item(
        build_item: BuildItem,
        profile: str,
        exec_prefix: list[str],
        exec_suffix: list[str],
        timeout_secs: int | None = None,
    ) -> BuildItem:
        """Execute all the items in the scheduler"""

        env: dict[str, str] = {}
        bin_name = build_item.spec_instance.hash(with_variables=False)

        exec_prefix_item = copy.deepcopy(exec_prefix)
        exec_suffix_item = copy.deepcopy(exec_suffix)
        env_item = copy.deepcopy(env)

        profile_output = f"{build_item.output_dir}/{bin_name}_profile"
        if profile in ["ncu", "ncu-single"]:
            exec_prefix_item.extend(["ncu", "-o", profile_output])
            if profile == "ncu-single":
                exec_suffix_item.extend(
                    ["--bench-max-iters=0", "--bench-max-batch-size=1"]
                )
        if profile in ["rocm", "rocprof-compute"]:
            exec_prefix_item.extend(
                f"rocprof-compute profile --name NAME -p {profile_output} --".split()
            )
            logging.info(f"writing profiling results to {profile_output}")

        if build_item.bin_path:
            t_start = time()
            exec_output = build_item.spec_instance.execute(
                build_item.bin_path,
                build_item.output_path,
                dryrun=build_item.dryrun,
                exec_prefix=exec_prefix_item,
                exec_suffix=exec_suffix_item,
                env=env_item,
                timeout_secs=timeout_secs,
            )
            build_item.exec_output = exec_output
            build_item.exec_benchmark_time = time() - t_start
            exec_output.log()
        else:
            logging.error(f"Could not find binary [{bin_name}]")

        return build_item

    def setup_build_pool(self) -> None:
        self.build_pool = Pool(self.num_cpu)

    def close_build_pool(self) -> None:
        self.build_pool.close()
        self.build_pool.join()

    def _make_gpu_ids(self, visible_device_prefix: str) -> list[int | str]:
        """Return list of GPU IDs from env or range(num_gpu)."""
        existing = os.environ.get(visible_device_prefix, "")
        if existing.strip():
            visible_ids: list[int | str] = list(
                dict.fromkeys(v.strip() for v in existing.split(","))
            )
            return visible_ids[: self.num_gpu]
        return list(range(self.num_gpu))

    def execute_all(
        self,
        visible_device_prefix: str,
        timeout_secs: int | None,
        profile: str,
        exec_prefix: Sequence[str],
        exec_suffix: Sequence[str],
    ) -> None:
        """Execute all build items using process-per-GPU with manager threads.

        Each GPU gets a dedicated worker process managed by a parent thread.
        Items are distributed via an ItemPool with .so affinity and
        work-stealing.
        """
        t_start = time()
        num_build_items = len(self.build_items)
        exec_progress = self.progress.add_task("run", total=num_build_items)

        # Pre-filter: log and skip items that failed to build.
        executable_items: list[BuildItem] = []
        no_binary_count = 0
        for bi in self.build_items:
            if bi.bin_path:
                executable_items.append(bi)
            else:
                bin_name = bi.spec_instance.hash(with_variables=False)
                logging.error(
                    f"Skipping [{bin_name}]: build failed (no binary produced)"
                )
                self.progress.update(exec_progress, advance=1)
                no_binary_count += 1

        # Group executable items by binary hash (items sharing a .so).
        groups: dict[str, list[ExecItemTask]] = {}
        for bi in executable_items:
            key = bi.spec_instance.hash(with_variables=False)
            groups.setdefault(key, []).append(
                ExecItemTask(
                    build_item=bi,
                    use_shared_lib=bi.use_shared_lib,
                    profile=profile,
                    exec_prefix=list(exec_prefix),
                    exec_suffix=list(exec_suffix),
                )
            )

        binary_groups = [g for g in groups.values() if g]
        gpu_ids = self._make_gpu_ids(visible_device_prefix)

        # Only isolate GPUs when kbench itself is parallelizing across GPUs
        # and not using mpirun. Otherwise let benchmarks see all GPUs.
        use_mpirun = any("mpirun" in p for p in exec_prefix)
        if len(gpu_ids) <= 1 or use_mpirun:
            visible_device_prefix = ""

        num_items = num_build_items - no_binary_count
        largest_group = (
            max(len(g) for g in binary_groups) if binary_groups else 0
        )

        logging.info(
            f"Distributing {num_items} items ({len(binary_groups)} binary "
            f"groups) across {len(gpu_ids)} GPUs "
            f"(largest group: {largest_group} items)"
        )

        item_pool = ItemPool(binary_groups)
        for gpu_id in gpu_ids:
            item_pool.register_gpu(gpu_id)

        # Launch one manager thread per GPU.
        results_lock = threading.Lock()
        shutdown_event = threading.Event()
        self._shutdown_event = shutdown_event
        threads: list[threading.Thread] = []
        for gpu_id in gpu_ids:
            t = threading.Thread(
                target=_gpu_manager,
                args=(
                    gpu_id,
                    visible_device_prefix,
                    item_pool,
                    timeout_secs,
                    self.build_items,
                    results_lock,
                    self.progress,
                    exec_progress,
                    shutdown_event,
                ),
            )
            t.start()
            threads.append(t)

        for t in threads:
            t.join()
        self.t_benchmark_total = time() - t_start

        # Log execution summary
        exec_succeeded = 0
        exec_crashed = 0
        exec_timed_out = 0
        exec_no_binary = 0
        for b in self.build_items:
            if b.bin_path is None:
                exec_no_binary += 1
            elif b.exec_output.return_code == os.EX_OK:
                exec_succeeded += 1
            elif (
                b.exec_output.stderr
                and "timed out" in b.exec_output.stderr.lower()
            ):
                exec_timed_out += 1
            else:
                exec_crashed += 1
        total = exec_succeeded + exec_crashed + exec_timed_out + exec_no_binary
        parts = [f"{exec_succeeded} succeeded"]
        if exec_crashed:
            parts.append(f"{exec_crashed} crashed")
        if exec_timed_out:
            parts.append(f"{exec_timed_out} timed out")
        if exec_no_binary:
            parts.append(f"{exec_no_binary} failed to build")
        logging.info(f"Execution summary: {', '.join(parts)} ({total} total)")

    def shutdown_workers(self) -> None:
        """Signal manager threads to stop after current item."""
        if self._shutdown_event is not None:
            self._shutdown_event.set()

    @staticmethod
    def get_build_df(bi_list: Sequence[BuildItem]) -> pd.DataFrame:
        build_df = pd.DataFrame(
            {
                "name": ["build" for b in bi_list],
                "spec": [f"{str(b.spec_instance)}" for b in bi_list],
            }
        )

        build_elapsed_time_list = [b.build_elapsed_time for b in bi_list]
        build_df.insert(
            len(build_df.columns),
            "met (ms)",
            pd.Series(build_elapsed_time_list),
        )
        build_df.insert(len(build_df.columns), "iters", 1)
        build_df.insert(
            len(build_df.columns),
            "mesh_idx",
            pd.Series([bi.idx for bi in bi_list]),
        )
        build_df["met (ms)"] = build_df["met (ms)"].fillna(0)

        build_df["name"] = build_df["name"].astype("string")
        build_df["spec"] = build_df["spec"].astype("string")
        build_df["met (ms)"] = build_df["met (ms)"].astype("float64")

        return pd.DataFrame(
            build_df.loc[:, ["mesh_idx", "name", "met (ms)", "iters", "spec"]]
        )

    @staticmethod
    def load_csv_to_pd(
        mesh_idx: int, current_spec: SpecInstance, files: list[Path]
    ) -> list[pd.DataFrame]:
        valid_specs: list[pd.DataFrame] = []
        for f in files:
            df = pd.read_csv(f, index_col=None, header=0)
            if not df.empty:
                df.insert(0, "mesh_idx", mesh_idx)
                df.insert(len(df.columns), "spec", str(current_spec))
                kbench_filter = " ".join(
                    f"{p.name.lstrip('$')}={p.value}"
                    for p in current_spec.params
                )
                df.insert(len(df.columns), "kbench_filter", kbench_filter)
                # If there are more than one entries in CSV then bencher
                # has added an extra column at the end of name with input_id.
                # TODO: This will create multiple rows with same mesh_idx.
                # Ensure this doesn't cause issues with 'kprofile' utilities.
                # TODO: Set an alternative index if input_id is missing.
                if len(df) > 1:
                    if df["name"].str.contains("/input:id").all():
                        raise ValueError(
                            "Detected multiple lines in output. All entries should have /input_id:"
                        )
                    id_column = df["name"].str.split("/input_id:").str[-1]
                    df["spec"] = (
                        df["spec"].astype(str) + "/input_id=" + id_column
                    )
                valid_specs.append(df)
        return valid_specs

    # Retrieve, sort, and pick top choices
    @staticmethod
    def get_valid_specs(
        bi_list: Sequence[BuildItem],
        spec: Spec,
        mode: KBENCH_MODE = KBENCH_MODE.BUILD_AND_RUN,
    ) -> tuple[list[pd.DataFrame], list[int]]:
        valid_specs: list[pd.DataFrame] = []
        invalid_specs: list[int] = []

        for idx, b in enumerate(bi_list):
            valid = False
            files = _get_similar_files(b.output_path)

            if mode == KBENCH_MODE.BUILD:
                # In build-only mode, success = binary was produced
                if b.bin_path is not None:
                    df = pd.DataFrame.from_dict(
                        {
                            "mesh_idx": [b.idx],
                            "name": ["build"],
                            "met (ms)": [b.build_elapsed_time],
                            "iters": [1],
                            "spec": [str(spec.mesh[b.idx])],
                        }
                    )
                    valid_specs.append(df)
                    valid = True
            elif b.exec_output.return_code == os.EX_OK:
                if files:
                    current_valid_specs = Scheduler.load_csv_to_pd(
                        mesh_idx=b.idx,
                        current_spec=spec.mesh[b.idx],
                        files=files,
                    )
                    valid_specs.extend(current_valid_specs)
                    valid = len(current_valid_specs) > 0

                # TODO: is this case still needed? why should successful
                # build without output.csv be considered as valid result?

                if not valid:
                    df = pd.DataFrame().from_dict(
                        {
                            "mesh_idx": [b.idx],
                            "name": ["-"],
                            "met (ms)": [0],
                            "iters": [0],
                            "spec": [str(spec.mesh[b.idx])],
                        }
                    )
                    valid_specs.append(df)
                    valid = True

            if not valid:
                invalid_specs.append(idx)

        return valid_specs, invalid_specs

    @staticmethod
    def dump(
        bi_list: list[BuildItem],
        spec: Spec,
        output_path: Path = Path(),
        mode: KBENCH_MODE = KBENCH_MODE.BUILD_AND_RUN,
        t_build_total: float = 0.0,
        t_benchmark_total: float = 0.0,
        t_elapsed_total: float = 0.0,
        verbose: bool = False,
    ) -> None:
        output_lines = []
        output_dict: dict[str, Any] = {}

        build_df = Scheduler.get_build_df(bi_list)
        output_dict["build_df"] = build_df

        output_lines += [utils.LINE]
        output_lines += ["Build time stats:"]
        output_lines += [build_df.to_string(index=False)]

        output_lines += [utils.LINE]
        output_lines += [f"Running ['{spec.file}']"]

        ###############################
        valid_specs, invalid_specs = Scheduler.get_valid_specs(
            bi_list, spec, mode
        )
        num_invalid_specs = len(invalid_specs)
        num_valid_specs = len(valid_specs)

        # Build structured failure records for all invalid specs
        failure_records = []
        if num_invalid_specs:
            output_lines += [utils.LINE]
            output_lines += [
                f"Number of invalid specs: {num_invalid_specs} (out of {len(spec)})"
            ]

            for idx in invalid_specs:
                s = bi_list[idx].spec_instance
                build_output = bi_list[idx].build_output
                # Determine failure type
                if (
                    build_output.return_code != os.EX_OK
                    and not bi_list[idx].bin_path
                ):
                    failure_type = "build"
                elif (
                    stderr := bi_list[idx].exec_output.stderr
                ) and "timed out" in stderr.lower():
                    failure_type = "timeout"
                else:
                    failure_type = "execution"
                failure_records.append(
                    {
                        "mesh_idx": idx,
                        "params": s.to_obj(),
                        "failure_type": failure_type,
                        "exec_stderr": bi_list[idx].exec_output.stderr or "",
                        "exec_stdout": bi_list[idx].exec_output.stdout or "",
                    }
                )
                # check build failure
                if build_output.stdout or build_output.stderr:
                    output_lines += [utils.LINE]
                    output_lines += [f"mesh_idx: [{idx}][{s.to_obj()}]"]
                    if build_output.stdout:
                        output_lines.append(build_output.stdout)
                    if build_output.stderr:
                        output_lines.append(build_output.stderr)

        label = "built" if mode == KBENCH_MODE.BUILD else "executed"
        output_lines += [utils.LINE]
        output_lines += [
            f"Number of valid {label} specs: {num_valid_specs} (out of {len(spec)})"
        ]

        if num_valid_specs:
            merged_df = pd.concat(valid_specs, axis=0, ignore_index=True)
            # Convert 'name' and 'spec' columns to pandas string
            merged_df["name"] = merged_df["name"].astype("string")
            merged_df["spec"] = merged_df["spec"].astype("string")

            ###############################
            # Get the name of column 2 (met (ms))
            output_dict["merged_df"] = merged_df

            output_lines += [merged_df.to_string(index=False)]
            output_lines += [utils.LINE]
            ###############################
        t_overhead = t_elapsed_total - (t_build_total + t_benchmark_total)
        timing_details = pd.DataFrame(
            {
                "Step": ["build", "benchmark", "kbench overhead", "TOTAL"],
                "Total (s)": [
                    t_build_total,
                    t_benchmark_total,
                    t_overhead,
                    t_elapsed_total,
                ],
            }
        ).round(3)
        timing_str = "Total elapsed time per step:\n" + str(
            timing_details.to_markdown(index=False, tablefmt="rounded_grid")
        )
        output_lines += [timing_str]
        output_str = "\n".join(output_lines)
        if verbose:
            print(output_str)
        else:
            logging.info(timing_str)

        if output_path:
            output_dict["name"] = spec.name
            output_dict["file"] = spec.file
            output_suffix = output_path.suffix
            pkl_path = output_path.with_suffix(output_suffix + ".pkl")
            csv_path = output_path.with_suffix(output_suffix + ".csv")
            txt_path = output_path.with_suffix(output_suffix + ".txt")

            utils.store_pickle(f"{pkl_path}", output_dict)

            # KBENCH_MODE.RUN overrides everything else and just dumps the running results.
            # THIS IS CRITICAL for CI automated kernel benchmarks workflow.
            if (
                mode in [KBENCH_MODE.RUN, KBENCH_MODE.BUILD_AND_RUN]
            ) and valid_specs:
                merged_df.drop(columns=["mesh_idx"]).to_csv(
                    csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC
                )
            elif mode == KBENCH_MODE.BUILD:
                build_df.to_csv(
                    csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC
                )

            with open(txt_path, "w") as f:
                f.write(output_str + "\n")

            # Write structured failure data for downstream reporting
            failures_json_path = output_path.with_suffix(
                output_suffix + ".failures.json"
            )
            failures_data = {
                "spec_name": spec.name,
                "spec_file": str(spec.file),
                "num_valid": num_valid_specs,
                "num_total": len(spec),
                "failures": failure_records,
            }
            with open(failures_json_path, "w") as f:
                json.dump(failures_data, f, indent=2, default=str)

            logging.info(
                f"wrote results to [{output_path}]"
                " (.txt, .csv, .pkl, .failures.json)"
            )
