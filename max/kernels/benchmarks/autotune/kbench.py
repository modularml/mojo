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

import gc
import logging
import math
import os
import pickle
import sys
from collections.abc import Sequence
from pathlib import Path
from time import time

import click
import pandas as pd
import rich
from kbench_model import (
    KBENCH_MODE,
    KbenchCache,
    Scheduler,
    Spec,
    SpecInstance,
)
from rich import traceback
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from terminal_viz import render_results
from utils import (
    LINE,
    _get_core_count,
    _get_gpu_count,
    _get_visible_device_prefix,
    check_gpu_clock,
    check_valid_target_accelerator,
    get_target_accelerator_helpstr,
    pretty_exception_handler,
)

##### Utilities and configurations #####

CONSOLE = Console(width=80)

pd.set_option("display.float_format", str)


def configure_logging(
    quiet: bool = False, verbose: bool = False, pretty_output: bool = True
) -> Console:
    """Configure logging with rich formatting."""
    global CONSOLE

    if pretty_output:
        debug_handler = RichHandler(
            show_path=False, show_time=False, console=CONSOLE
        )
        logging.basicConfig(format="%(message)s", handlers=[debug_handler])
    else:
        logging.basicConfig(format="%(levelname)s: %(message)s")
        CONSOLE = Console(width=80, force_terminal=False, color_system=None)

    log_level = (
        logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    )
    logging.getLogger().setLevel(log_level)

    if verbose and pretty_output:
        traceback.install(suppress=[click, rich])
    elif pretty_output:
        sys.excepthook = pretty_exception_handler

    return CONSOLE


def log_and_raise_error(message: str, param_hint: str | None = None) -> None:
    """Log an error and raise a Click exception.

    Args:
        message: The error message to log and display
        param_hint: Optional parameter hint for BadParameter exception
    """
    logging.error(message)
    if param_hint:
        raise click.BadParameter(message, param_hint=param_hint)
    else:
        raise click.UsageError(message)


def run(
    yaml_path_list: list[Path] | None,
    obj_cache: KbenchCache,
    shape: SpecInstance,
    output_path: Path = Path(),
    mode: KBENCH_MODE = KBENCH_MODE.BUILD_AND_RUN,
    param_list: Sequence[str] | None = None,
    filter_list: Sequence[str] | None = None,
    build_opts: list[str] = [],  # noqa: B006
    profile: str = "",
    exec_prefix: list[str] = [],  # noqa: B006
    exec_suffix: list[str] = [],  # noqa: B006
    dryrun: bool = False,
    verbose: bool = False,
    output_dir: Path | None = None,
    num_cpu: int = 1,
    num_gpu: int = 1,
    target_accelerator: str | None = None,
    timeout_secs: int | None = None,
    plot: str = "bars",
    use_shared_lib: bool = False,
    cache_dir: Path | None = None,
) -> None:
    if yaml_path_list:
        # Load specs from a list of YAML files and join them in 'spec'.
        assert len(yaml_path_list), "There should be at least 1 YAML as input."
        spec = Spec.load_yaml_list(yaml_path_list)
    else:
        # Just load an empty Spec with identical name and file as shape
        spec = Spec(shape.name, shape.file)

    # Set output_dir='./kbench-output' if it is not specified.
    if not output_dir:
        output_dir = Path("./kbench-output")

    # Set output_path (for storing results) relative to output_dir
    output_path = output_dir / output_path
    os.makedirs(output_path.parent, exist_ok=True)
    # strip output_path suffix
    if output_path.suffix in [".csv", ".pkl", ".txt"]:
        output_path = output_path.with_suffix("")

    if shape:
        spec.extend_shape_params(shape.params)
        # Each shape should have its own temporary directory.
        output_dir = output_dir / Path(shape.hash(with_variables=True))
    logging.info(f"output-dir: [{output_dir}]")

    # Expand with CLI params
    if param_list:
        spec.extend_params(param_list)

    # Apply the filters, if any.
    if filter_list:
        spec.filter(filter_list)

    if verbose:
        for i, s in enumerate(spec):
            logging.debug(f"[{i}]{s}")
        logging.debug(LINE)

    # Run the code over the mesh of param/values
    t_start_total = time()
    progress = Progress(
        *Progress.get_default_columns(),
        MofNCompleteColumn(),
        TextColumn("|"),
        TimeElapsedColumn(),
        console=CONSOLE,
        expand=True,
        transient=True,
    )

    # Kbench Singleton Scheduler
    scheduler = Scheduler(
        num_cpu=num_cpu,
        num_gpu=num_gpu,
        obj_cache=obj_cache,
        run_only=(mode == KBENCH_MODE.RUN),
        spec_list=list(spec),
        output_dir=output_dir,
        build_opts=build_opts,
        dryrun=dryrun,
        output_suffix="output.csv",
        progress=progress,
        use_shared_lib=use_shared_lib,
        cache_dir=cache_dir,
    )

    visible_device_prefix = _get_visible_device_prefix(str(target_accelerator))

    # 0 means no timeout (infinite wait).
    effective_timeout = timeout_secs or None

    # Run the code over the mesh of param/values
    t_start_total = time()

    with progress:
        try:
            scheduler.build_all()
            obj_cache.dump()

            if mode == KBENCH_MODE.BUILD:
                # In build-only mode, skip execution, dump, and
                # visualization. Print the summary line that
                # bench_compare.py parses.
                num_valid = sum(
                    1 for bi in scheduler.build_items if bi.bin_path is not None
                )
                num_total = len(spec)
                print(
                    f"Number of valid built specs: {num_valid}"
                    f" (out of {num_total})"
                )
                return

            # Only RUN and BUILD_AND_RUN reach here.
            scheduler.execute_all(
                visible_device_prefix=visible_device_prefix,
                timeout_secs=effective_timeout,
                profile=profile,
                exec_prefix=exec_prefix,
                exec_suffix=exec_suffix,
            )

        except KeyboardInterrupt:
            scheduler.close_build_pool()
            scheduler.shutdown_workers()
            obj_cache.dump()
            sys.exit(0)

    t_elapsed_total = time() - t_start_total

    ###############################
    # dump all the details
    gc.collect()
    Scheduler.dump(
        scheduler.build_items,
        spec,
        output_path,
        mode,
        scheduler.t_build_total,
        scheduler.t_benchmark_total,
        t_elapsed_total,
        verbose=verbose,
    )
    # Render terminal visualization if requested
    if plot != "none":
        pkl_path = output_path.with_suffix(output_path.suffix + ".pkl")
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                pkl_data = pickle.load(f)
            if "merged_df" in pkl_data:
                render_results(
                    pkl_data["merged_df"], mode=plot, console=CONSOLE
                )


def _validate_partition(partition: str) -> list[int]:
    assert ":" in partition
    partition_idx, num_partitions = [int(x) for x in partition.split(":")]
    assert num_partitions > 0
    assert 0 <= partition_idx < num_partitions, (
        "Condition: 0 <= partition_idx < num_partitions"
    )
    return [partition_idx, num_partitions]


def set_build_opts(
    debug_level: str | None = None,
    optimization_level: str | None = None,
    use_experimental_kernels: bool | None = None,
    target_accelerator: str | None = None,
    disable_warnings: bool | None = None,
    march: str | None = None,
) -> list[str]:
    build_opts = []
    if debug_level:
        build_opts.extend(["--debug-level", debug_level])
    if optimization_level:
        build_opts.extend([f"-O{optimization_level}"])
    if use_experimental_kernels:
        build_opts.extend(["-D", "USE_EXPERIMENTAL_KERNELS=1"])
    if target_accelerator:
        build_opts.extend(["--target-accelerator", target_accelerator])
    if disable_warnings:
        build_opts.extend(["--disable-warnings"])
    if march:
        build_opts.extend(["--march", march])
    # TODO: add num_threads to CLI
    # num_threads_per_build = 1
    # build_opts.extend(["--num-threads", num_threads_per_build])
    return build_opts


@click.command(
    help="Benchmarking toolkit for Mojo kernels",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "-help", "--help"]},
)
@click.option(
    "--filter",
    "filter",
    help=(
        "Define a single filter (should match a valid parameter, can have"
        " multiple ones). The filters should of the format '--filter"
        " PARAM=VALUE', that is, the subset of parameters that satisfy this"
        " condition will be included."
    ),
    multiple=True,
)
@click.option(
    "--output",
    "-o",
    "output_path",
    default="output.csv",
    help="Path to output file.",
)
@click.option(
    "--output-dir",
    "output_dir",
    default="kbench-output",
    help="Path to output directory for all results (default='./kbench-output')",
)
@click.option(
    "--build",
    "build",
    is_flag=True,
    default=False,
    help="Just build the binary and report the build time.",
)
@click.option(
    "--run-only",
    "run_only",
    is_flag=True,
    default=False,
    help="Only run, do not build. Cache must exist, -c is implied",
)
@click.option(
    "--param",
    default=(),
    help="Set extra params in the format of 'PARAM:VALUE'. Example: '--param use_vendor_blas:True'",
    multiple=True,
)
@click.option(
    "--debug-level", default=None, help="The debug level used during the build."
)
@click.option(
    "--use-experimental-kernels",
    is_flag=True,
    default=False,
    help="If enabled, then experimental kernels are used.",
)
@click.option(
    "-O",
    "--optimization-level",
    default=None,
    help="The optimization level used during the build.",
)
@click.option(
    "--target-accelerator",
    default=None,
    help="Specify the mojo target accelerator. Allowed values for this option:"
    + get_target_accelerator_helpstr(),
)
@click.option(
    "--march",
    default=None,
    help="Set the host CPU architecture for cross-compilation (e.g. x86-64-v3). "
    "Passed directly to mojo as --march. Useful when compiling on a different "
    "machine than the one that will run the benchmarks.",
)
@click.option(
    "--disable-warnings",
    is_flag=True,
    default=False,
    help="Disable mojo build warnings.",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    default=False,
    help="This option will be deprecated soon. See --skip-clock-check instead.",
)
@click.option(
    "--skip-clock-check",
    is_flag=True,
    default=False,
    help="Run even if accelerator clocks are not set to maximum.",
)
@click.option(
    "--cached",
    "-c",
    is_flag=True,
    default=False,
    help="Enable Kbench cache (WARNING: doesn't check for source changes).",
)
@click.option(
    "--cache-dir",
    "cache_dir",
    default=None,
    help="Fixed directory for compiled binaries and cache pickle. "
    "Enables portable caching for split build/run workflows.",
    type=click.Path(path_type=Path),
)
@click.option(
    "--clear-cache",
    "-cc",
    is_flag=True,
    default=False,
    help="Clear Kbench cache.",
)
@click.option(
    "--num-cpu",
    default=-1,
    help="Set the total number of cpu cores for building. Set to -1 for max number of cores (default=-1).",
)
@click.option(
    "--num-gpu",
    default=1,
    help="Set the total number of GPU devices for running, it can only be used with '--target-accelerator' (default=1).",
)
@click.option(
    "--mpirun-np",
    default=1,
    help="Set the total number of GPU devices for running with mpirun, it cannot be combined with '--num-gpus' (default=1)."
    "Make sure to call 'Bench.check_mpirun()' in mojo benchmark.",
)
@click.option(
    "--dryrun",
    "-dryrun",
    is_flag=True,
    default=False,
    help="Do not execute the config, just show the parameters.",
)
@click.option(
    "--verbose", "-v", is_flag=True, default=False, help="Verbose printing."
)
@click.option(
    "--shapes",
    default=(),
    help="Set of shapes passed as extra params.",
    multiple=True,
)
@click.option(
    "--build-opts",
    default="",
    help="Any build options (treated as str and directly passed to mojo compiler.)",
    multiple=False,
)
@click.option(
    "--profile",
    default="",
    help="Set the profiler [ncu, ncu-single, rocm, rocprof-compute].",
    multiple=False,
)
@click.option(
    "--exec-prefix",
    default="",
    help="Any prefix options (treated as str and directly passed before binary.)",
    multiple=False,
)
@click.option(
    "--exec-suffix",
    default="",
    help="Any suffix options (treated as str and directly passed after binary.)",
    multiple=False,
)
@click.option(
    "--timeout-secs",
    default=120,
    show_default=True,
    help="Timeout seconds for executing each benchmark. 0 disables timeout.",
    multiple=False,
    type=click.INT,
)
@click.option(
    "--partition",
    default="0:1",
    help="Formatted as fraction 'm:n', divide the shapes "
    "into n partitions and limit the space to m'th partition "
    "(default='0:1' running everything). Note that it has no "
    "effect on parameter set and is only applied to shapes.",
    multiple=False,
    type=click.STRING,
)
@click.option(
    "--plot",
    type=click.Choice(["bars", "table", "summary", "none"]),
    default="bars",
    help="Terminal visualization: bars (default), table, summary, or none to disable.",
)
@click.argument("files", nargs=-1, type=click.UNPROCESSED)
def cli(
    files: tuple[str, ...],
    filter: tuple[str, ...],
    output_path: str,
    output_dir: str,
    build: bool,
    run_only: bool,
    param: tuple[str, ...],
    debug_level: str | None,
    use_experimental_kernels: bool,
    optimization_level: str | None,
    target_accelerator: str | None,
    march: str | None,
    disable_warnings: bool,
    force: bool,
    skip_clock_check: bool,
    cached: bool,
    cache_dir: Path | None,
    clear_cache: bool,
    num_cpu: int,
    num_gpu: int,
    mpirun_np: int,
    dryrun: bool,
    verbose: bool,
    shapes: tuple[str, ...],
    build_opts: str,
    profile: str,
    exec_prefix: str,
    exec_suffix: str,
    timeout_secs: int | None,
    partition: str,
    plot: str,
) -> bool:
    configure_logging(verbose=verbose)

    if not verbose:
        sys.tracebacklimit = 1

    if force:
        logging.warning(
            "'--force' option is deprecated and will be removed soon. Please use '--skip-clock-check' instead."
        )
        skip_clock_check = True

    mode = KBENCH_MODE.BUILD_AND_RUN
    if run_only:
        mode = KBENCH_MODE.RUN

    if run_only and clear_cache:
        log_and_raise_error(
            "Cannot clear cache when in run-only mode. Need cache to run.",
            param_hint="'--clear-cache'",
        )
    if run_only and build_opts:
        log_and_raise_error("Cannot provide build options when run-only mode")

    partition_idx, num_partitions = _validate_partition(partition)

    if build:
        mode = KBENCH_MODE.BUILD

    if cache_dir:
        cache_dir = Path(cache_dir).resolve()
        obj_cache = KbenchCache(base_dir=cache_dir)
        cached = True  # --cache-dir implies --cached
    else:
        obj_cache = KbenchCache()

    # check kbench_cache and load it if exists:
    if clear_cache and run_only:
        log_and_raise_error("Trying to clear cache when run_only")
    elif clear_cache:
        obj_cache.clear()

    if cached or (mode == KBENCH_MODE.RUN):
        obj_cache.load()

    if len(obj_cache.data) == 0 and mode == KBENCH_MODE.RUN:
        log_and_raise_error(
            "Run Only requires an active cache object but the object is empty",
            param_hint="'--run-only'",
        )

    if not len(files) and not len(shapes):
        logging.info(
            "Nothing more to do without parameter or shape YAML provided!"
        )
        return True

    # Resolve YAML file paths from the input user globs
    yaml_files = []
    for f in files:
        yaml_files.append(Path(f).resolve())

    if len(yaml_files) == 0:
        log_and_raise_error(
            "No valid YAML files found from the input globs.",
            param_hint="'FILES'",
        )

    if not skip_clock_check:
        check_gpu_clock()

    # If `shapes` is not specified, pick an empty Spec and '-o output_path'.
    shape_list = list(
        Spec.load_yaml_list([Path(s) for s in shapes]) if shapes else Spec()
    )
    shape_path_list = (
        [Path(sh.hash(with_variables=True)) for sh in shape_list]
        if shapes
        else [Path(output_path)]
    )

    assert len(shape_path_list) == len(shape_list), (
        "Number of shapes doesn't equal number of paths."
    )

    if target_accelerator and not check_valid_target_accelerator(
        target_accelerator
    ):
        log_and_raise_error(
            f"Invalid target accelerator '{target_accelerator}'. "
            f"Should be one of the following {get_target_accelerator_helpstr()}",
            param_hint="'--target-accelerator'",
        )

    build_opts_list: list[str] = build_opts.split(" ") if build_opts else []
    build_opts_list.extend(
        set_build_opts(
            debug_level,
            optimization_level,
            use_experimental_kernels,
            target_accelerator,
            disable_warnings,
            march,
        )
    )

    if num_gpu > 1 and not target_accelerator:
        raise ValueError(
            "Cannot use --num-gpu>1 without specifying --target-accelerator"
        )
    if mpirun_np > 1 and num_gpu > 1:
        raise ValueError(
            "Cannot use --num-gpu>1 and --mpirun-np>1 at the same time!"
        )

    # Validate requested GPU count against available hardware
    gpu_request = num_gpu if num_gpu > 1 else mpirun_np
    if gpu_request > 1 and target_accelerator:
        available = _get_gpu_count(target_accelerator)
        if available is not None and gpu_request > available:
            flag = "--num-gpu" if num_gpu > 1 else "--mpirun-np"
            raise ValueError(
                f"{flag}={gpu_request} exceeds the {available} GPUs"
                " detected on this machine."
            )

    exec_suffix_list: list[str] = exec_suffix.split(" ") if exec_suffix else []
    exec_prefix_list: list[str] = exec_prefix.split(" ") if exec_prefix else []
    if mpirun_np > 1:
        exec_prefix_list.extend(["mpirun", "-np", str(mpirun_np)])

    # Auto-select: use shared lib (.so) mode when possible for faster execution.
    # Falls back to subprocess mode when profiling or using custom exec wrappers.
    # Under sanitizers, kbench_model._maybe_preload_libubsan handles the
    # dlopen-from-non-instrumented-Python case, so shared-lib mode works there
    # too (see MOTO-1576).
    use_shared_lib = not profile and not exec_prefix and not exec_suffix
    if use_shared_lib:
        logging.info("Using shared library (.so) mode for faster execution")

    # Resolve num_cpu sentinel value once before the shape loop.
    if num_cpu == -1:
        num_cpu = max(_get_core_count() // 2, 1)
    logging.info(f"num cpu's: {num_cpu}, num gpu's: {num_gpu}")

    shapes_per_partition = math.ceil(len(shape_list) / num_partitions)
    shape_idx_lb = partition_idx * shapes_per_partition
    shape_idx_ub = min(shape_idx_lb + shapes_per_partition, len(shape_list))

    output_dir_path: Path | None = Path(output_dir) if output_dir else None
    for i in range(shape_idx_lb, shape_idx_ub):
        run(
            yaml_path_list=yaml_files,
            obj_cache=obj_cache,
            shape=shape_list[i],
            output_path=shape_path_list[i],
            mode=mode,
            param_list=param,
            filter_list=filter,
            build_opts=build_opts_list,
            profile=profile,
            exec_prefix=exec_prefix_list,
            exec_suffix=exec_suffix_list,
            dryrun=dryrun,
            verbose=verbose,
            output_dir=output_dir_path,
            num_cpu=num_cpu,
            num_gpu=num_gpu,
            target_accelerator=target_accelerator,
            timeout_secs=timeout_secs,
            plot=plot,
            use_shared_lib=use_shared_lib,
            cache_dir=cache_dir,
        )
        if obj_cache.is_active:
            obj_cache.dump()
    logging.info(f"Number of shapes: {len(shape_list)}")
    return True


def main() -> None:
    try:
        cli()
    except Exception:
        CONSOLE.print_exception(suppress=[click, rich])


if __name__ == "__main__":
    if directory := os.environ.get("BUILD_WORKING_DIRECTORY"):
        os.chdir(directory)

    main()
