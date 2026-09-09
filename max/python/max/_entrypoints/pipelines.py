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

import functools
import importlib
import logging
import os
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from datetime import timedelta
from typing import Any, TypeVar

import click
from click import shell_completion
from max import _eager_policy
from typing_extensions import ParamSpec

# Please keep all max imports inside their respective functions.
# This is best practice to keep the CLI invocation fast.
# Exception: max._eager_policy is a stdlib-only leaf, so it costs nothing here.


logger = logging.getLogger("max._entrypoints")

_P = ParamSpec("_P")
_R = TypeVar("_R")

# Subcommands that should not configure/emit telemetry.
_TELEMETRY_OPT_OUT_COMMANDS = {
    "benchmark",
    "list",
    "warm-interpreter-cache",
}


def check_model_flag_conflict(args: list[str]) -> None:
    """Raises if both --model and --model-path are specified.

    Raises:
        ValueError: If both --model and --model-path are specified.
    """
    saw_model = False
    saw_model_path = False
    for arg in args:
        if arg == "--model" or arg.startswith("--model="):
            saw_model = True
        elif arg == "--model-path" or arg.startswith("--model-path="):
            saw_model_path = True
        if saw_model and saw_model_path:
            raise ValueError("model_path and model cannot both be specified")


class WithLazyPipelineOptions(click.Command):
    """Command wrapper that defers loading pipeline configuration options

    Lazily applies pipeline_config_options to the callback only when
    command help or execution is actually requested, improving startup time.
    This is somewhat of a hack,
    and should be removed when the pipeline_config_options decorator is fast.
    """

    def __init__(self, *args, **kwargs) -> None:
        self._options_loaded = False
        super().__init__(*args, **kwargs)

    @staticmethod
    def _add_options(callback: Callable[_P, _R]) -> Callable[_P, _R]:
        from max._entrypoints.cli.config import pipeline_config_options

        return pipeline_config_options(callback)

    def _ensure_options_loaded(self) -> None:
        if self._options_loaded:
            return

        # Lazily load and apply pipeline_config_options decorator

        # In Click, each command has a callback function that's executed when the command runs.
        # The callback contains the actual implementation of the command.
        # Here, we're applying the pipeline_config_options decorator to add CLI parameters
        # to our callback function dynamically, rather than statically at import time.
        assert self.callback is not None
        self.callback = self._add_options(self.callback)
        self._options_loaded = True

        # When Click decorators (like @click.option) are applied to a function,
        # they attach Parameter objects to the function via a __click_params__ attribute.
        # We need to extract these parameters and add them to the command's params list
        # so Click knows about them for argument parsing, help text generation, etc.
        # Create a copy to avoid modifying the original list
        self.params = self.params.copy()
        for param in getattr(self.callback, "__click_params__", []):
            self.params.append(param)

    def get_help(self, ctx: click.Context) -> str:
        self._ensure_options_loaded()
        return super().get_help(ctx)

    def invoke(self, ctx: click.Context) -> Any:
        self._ensure_options_loaded()
        return super().invoke(ctx)

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        self._ensure_options_loaded()
        check_model_flag_conflict(args)
        return super().parse_args(ctx, args)

    def get_params(self, ctx: click.Context) -> list[click.Parameter]:
        self._ensure_options_loaded()
        return super().get_params(ctx)

    def shell_complete(
        self, ctx: click.Context, incomplete: str
    ) -> list[shell_completion.CompletionItem]:
        self._ensure_options_loaded()
        return super().shell_complete(ctx, incomplete)


class WithLazySamplingAndPipelineOptions(WithLazyPipelineOptions):
    @staticmethod
    def _add_options(callback: Callable[_P, _R]) -> Callable[_P, _R]:
        from max._entrypoints.cli.config import (
            pipeline_config_options,
            sampling_params_options,
        )

        return sampling_params_options(pipeline_config_options(callback))


class ModelGroup(click.Group):
    def get_command(
        self, ctx: click.Context, cmd_name: str
    ) -> click.Command | None:
        rv = click.Group.get_command(self, ctx, cmd_name)
        if rv is not None:
            return rv
        supported = ", ".join(self.list_commands(ctx))
        ctx.fail(
            f"Command not supported: {cmd_name}\nSupported commands:"
            f" {supported}"
        )


def _handle_import_error(
    e: ImportError, subcommand: str, suggestion: tuple[str, str]
) -> None:
    """Gives actionable feedback on import errors.

    Args:
        e: The raised error
        subcommand: The subcommand that was run (i.e. "serve" for `max serve`)
        suggestion: Which package to install, first item is when using conda, second is for wheels.
    """

    if sys.version_info < (3, 11):
        # Backport `add_note()`
        import exceptiongroup  # noqa: F401

    suggest = suggestion[0] if os.getenv("CONDA_PREFIX") else suggestion[1]
    e.add_note(  # type: ignore
        f"To use the `max {subcommand}` command, install the `{suggest}` package."
    )
    raise e


@click.command(cls=ModelGroup)
@click.option(
    "--version",
    is_flag=True,
    callback=lambda ctx, param, value: print_version(ctx, param, value),
    expose_value=False,
    is_eager=True,  # Eager ensures this runs before other options/commands
    help="Show the MAX version and exit.",
)
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False
    ),
    default="INFO",
    help="Set logging level explicitly (ignored if --verbose or --quiet is used).",
)
@click.pass_context
def main(ctx: click.Context, log_level: str = "INFO") -> None:
    from max._entrypoints.cli.entrypoint import configure_cli_logging

    # Configure logging first, before any other initialization
    configure_cli_logging(
        level=log_level, log_prefix=os.getenv("MAX_SERVE_LOG_PREFIX")
    )

    # Some subcommands opt out of telemetry.
    if ctx.invoked_subcommand not in _TELEMETRY_OPT_OUT_COMMANDS:
        configure_telemetry(ctx.invoked_subcommand or "")


def configure_telemetry(subcommand: str) -> None:
    try:
        from max.serve.config import Settings
        from max.serve.telemetry.common import (
            configure_metrics,
            configure_tracing,
        )
    except ImportError as e:
        # Note: most commands import main(), and thus run this, so this catches most subcommands.
        _handle_import_error(
            e,
            subcommand,
            # All subcommands that invoke telemetry need serve deps anyways, so always suggest these.
            ("max-pipelines", "max[serve]"),
        )

    settings = Settings()
    configure_metrics(settings)
    configure_tracing(settings)


def common_server_options(func: Callable[_P, _R]) -> Callable[_P, _R]:
    @click.option(
        "--port",
        type=int,
        help="Port for the HTTP API. Defaults to ``8000``.",
    )
    @click.option(
        "--headless",
        is_flag=True,
        show_default=True,
        default=False,
        help="Run only the dispatcher service and model worker without the API server.",
    )
    @click.option(
        "--log-prefix",
        type=str,
        help="Optional prefix to add to all log messages for this server instance.",
    )
    @click.option(
        "--eplb-stats",
        type=click.Path(exists=True, dir_okay=False, readable=True),
        default=None,
        help="Path to a snapshot JSON from /max_internal/eplb_stats. "
        "Triggers an EPLB rebalance at startup.",
    )
    @click.option(
        "--max-queue-size",
        type=int,
        default=None,
        help=(
            "Cap (N) on the request queue to the model worker. Once this many "
            "requests are in transit to the worker, new requests are rejected "
            "with HTTP 429 instead of being enqueued, providing "
            "self-calibrating backpressure to keep latency within SLAs. Pair "
            "with --max-pending-requests. Defaults to unbounded."
        ),
    )
    @click.option(
        "--max-pending-requests",
        type=int,
        default=None,
        help=(
            "Cap (M) on the scheduler's pending (prefill) queue depth. The "
            "worker stops pulling from the request queue once it holds this "
            "many not-yet-running requests, so the request queue backs up and "
            "exerts backpressure (see --max-queue-size) instead of growing an "
            "unbounded pending pool. Should be at least --max-batch-size. "
            "Defaults to unbounded."
        ),
    )
    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        return func(*args, **kwargs)

    return wrapper


def _apply_interpreter_cache_policy(allow_cold: bool) -> None:
    """Publishes the eager cold-cache policy to this process and its workers.

    Travels through the environment because the model workers that dispatch
    eager ops are subprocesses (see ``start_workers``).

    Args:
        allow_cold: Whether ``--allow-cold-interpreter-cache`` was passed. The
            flag wins over an exported value; serve's default does not.
    """
    if allow_cold:
        os.environ[_eager_policy.ALLOW_LAZY_COMPILE_ENV_VAR] = "1"
    else:
        os.environ.setdefault(_eager_policy.ALLOW_LAZY_COMPILE_ENV_VAR, "0")


@main.command(name="serve", cls=WithLazyPipelineOptions)
@common_server_options
@click.option(
    "--task-arg",
    multiple=True,
    type=str,  # Take them all in as strings
    help="Task-specific arguments to pass to the underlying model (can be used multiple times).",
)
@click.option(
    "--pretty-print-config",
    is_flag=True,
    default=False,
    help="Pretty Print Entire Config",
)
@click.option(
    "--allow-cold-interpreter-cache",
    is_flag=True,
    show_default=True,
    default=False,
    help="Permit compiling eager interpreter ops on demand, instead of "
    "refusing on a machine that `max warm-interpreter-cache` has not warmed. "
    "Equivalent to setting MAX_EAGER_ALLOW_LAZY_COMPILE=1.",
)
def cli_serve(
    port: int,
    headless: bool,
    log_prefix: str | None,
    eplb_stats: str | None,
    max_queue_size: int | None,
    max_pending_requests: int | None,
    task_arg: tuple[str, ...],
    pretty_print_config: bool,
    allow_cold_interpreter_cache: bool,
    **config_kwargs: Any,
) -> None:
    """Start a model serving endpoint for inference.

    Loads a model from a Hugging Face model ID or local path and
    exposes OpenAI-compatible HTTP endpoints for inference requests.
    """
    _apply_interpreter_cache_policy(allow_cold_interpreter_cache)

    from max._entrypoints.cli.serve import serve_api_server_and_model_worker
    from max._entrypoints.workers import start_workers
    from max.pipelines import PipelineArgs
    from max.pipelines.context import SamplingParams, SamplingParamsInput
    from max.pipelines.lib import MAXModelConfig
    from max.serve.config import Settings
    from max.serve.telemetry.common import configure_logging

    # Initialize Settings for API Server
    setting_kwargs: dict[str, Any] = {}
    if port is not None:
        setting_kwargs["MAX_SERVE_PORT"] = port

    if log_prefix is not None:
        setting_kwargs["MAX_SERVE_LOG_PREFIX"] = log_prefix

    if headless is not None:
        setting_kwargs["MAX_SERVE_HEADLESS"] = headless

    if eplb_stats is not None:
        os.environ["MAX_SERVE_EPLB_STATS"] = eplb_stats
        setting_kwargs["MAX_SERVE_EPLB_STATS"] = eplb_stats

    if max_queue_size is not None:
        setting_kwargs["MAX_SERVE_MAX_QUEUE_SIZE"] = max_queue_size

    if max_pending_requests is not None:
        setting_kwargs["MAX_SERVE_MAX_PENDING_REQUESTS"] = max_pending_requests

    settings = Settings(**setting_kwargs)

    # Initialize config, and serve.
    # Load tokenizer & pipeline.
    pipeline_args = PipelineArgs.from_flat_kwargs(**config_kwargs)
    if not pipeline_args.model_path:
        raise click.UsageError(
            "No model specified. Pass --model with a Hugging Face repo ID "
            "or local path, e.g.:\n"
            "  max serve --model modularai/Llama-3.1-8B-Instruct-GGUF"
        )

    # Log Pipeline and Sampling Configuration
    if pretty_print_config:
        # Log Default Sampling Configuration (only for single-model pipelines)
        if pipeline_args.model_path:
            model_config = MAXModelConfig.from_pipeline_args(pipeline_args)
            sampling_params = SamplingParams.from_input_and_generation_config(
                SamplingParamsInput(),
                sampling_params_defaults=model_config.sampling_params_defaults,
            )
            sampling_params.log_sampling_info()

        # Log API Server Related Info
        settings.log_server_info()

    # Configure Logging Globally
    configure_logging(settings)

    if headless:
        start_workers(
            settings=settings,
            pipeline_args=pipeline_args,
        )
    else:
        serve_api_server_and_model_worker(
            settings=settings, pipeline_args=pipeline_args
        )


@main.command(name="generate", cls=WithLazySamplingAndPipelineOptions)
@click.option(
    "--prompt",
    type=str,
    default="I believe the meaning of life is",
    help="The text prompt to use for further generation.",
)
@click.option(
    "--image_url",
    type=str,
    multiple=True,
    default=[],
    help=(
        "Images to include along with prompt, specified as URLs."
        " The images are ignored if the model does not support"
        " image inputs."
    ),
)
@click.option(
    "--num-warmups",
    type=int,
    default=0,
    show_default=True,
    help="Number of warmup iterations to run before the final timed run.",
)
@click.option(
    "--profile",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "Capture a rudimentary profile of the timed run. If Nsight Systems "
        "(`nsys`) and an NVIDIA GPU are available, captures the GPU kernel "
        "trace into an .nsys-rep file and prints the top kernels. Always "
        "captures a Python/CPU profile via cProfile."
    ),
)
@click.option(
    "--profile-output",
    type=str,
    default=None,
    help=(
        "Path for the .nsys-rep file when `--profile` is on. Default: "
        "$BUILD_WORKSPACE_DIRECTORY/max-profile.nsys-rep, or "
        "./max-profile.nsys-rep."
    ),
)
@click.option(
    "--profile-top-n",
    type=int,
    default=15,
    show_default=True,
    help="Number of rows to show in the GPU kernel and Python profile tables.",
)
def cli_pipeline(
    prompt: str,
    image_url: list[str],
    num_warmups: int,
    profile: bool,
    profile_output: str | None,
    profile_top_n: int,
    top_k: int,
    top_p: float,
    min_p: float,
    temperature: float,
    frequency_penalty: float,
    presence_penalty: float,
    repetition_penalty: float,
    max_new_tokens: int,
    min_new_tokens: int,
    ignore_eos: bool,
    stop: list[str],
    stop_token_ids: list[int],
    detokenize: bool,
    seed: int,
    **config_kwargs: Any,
) -> None:
    """Generate text using the specified model.

    This command runs text generation using the loaded model, optionally
    accepting image inputs for multimodal models.
    """
    from max._entrypoints.cli.generate import generate_text_for_pipeline
    from max.pipelines.context import SamplingParams, SamplingParamsInput
    from max.profiler import maybe_reexec_under_nsys

    # When --profile is set and we have a usable nsys + NVIDIA GPU, re-exec
    # under nsys *before* loading any model state. The child process picks
    # up the same argv, detects it is running under nsys, and runs normally.
    # This keeps the parent process's wasted work to ~zero.
    if profile:
        maybe_reexec_under_nsys(profile_output, top_n=profile_top_n)

    params = SamplingParamsInput(
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
        # Limit generate default max_new_tokens to 100.
        max_new_tokens=max_new_tokens or 100,
        min_new_tokens=min_new_tokens,
        ignore_eos=ignore_eos,
        stop=stop,
        stop_token_ids=stop_token_ids,
        detokenize=detokenize,
        seed=seed,
    )

    # Load tokenizer & pipeline.
    from max.pipelines import PipelineArgs
    from max.pipelines.lib import MAXModelConfig

    pipeline_args = PipelineArgs.from_flat_kwargs(**config_kwargs)
    model_config = MAXModelConfig.from_pipeline_args(pipeline_args)
    generate_text_for_pipeline(
        pipeline_args,
        sampling_params=SamplingParams.from_input_and_generation_config(
            params,
            sampling_params_defaults=model_config.sampling_params_defaults,
        ),
        prompt=prompt,
        image_urls=image_url,
        num_warmups=num_warmups,
        profile=profile,
        profile_top_n=profile_top_n,
    )


@main.command(name="encode", cls=WithLazyPipelineOptions)
@click.option(
    "--prompt",
    type=str,
    default="I believe the meaning of life is",
    help="The text prompt to use for further generation.",
)
@click.option(
    "--num-warmups",
    type=int,
    default=0,
    show_default=True,
    help="Number of warmup iterations to run before the final timed run.",
)
def encode(prompt: str, num_warmups: int, **config_kwargs: Any) -> None:
    """Encode text input into model embeddings.

    This command processes the input text through the model's encoder, producing
    embeddings that can be used for various downstream tasks.
    """
    from max._entrypoints.cli.encode import pipeline_encode
    from max.pipelines import PipelineArgs

    # Load tokenizer & pipeline.
    pipeline_args = PipelineArgs.from_flat_kwargs(**config_kwargs)
    pipeline_encode(pipeline_args, prompt=prompt, num_warmups=num_warmups)


@main.command(name="warm-cache", cls=WithLazyPipelineOptions)
@click.option(
    "--target",
    type=str,
    default=None,
    help=(
        "Target API and architecture to compile for (e.g., cuda, cuda:sm_90, "
        "hip:gfx942, metal). When specified, uses virtual devices for "
        "compilation without requiring physical hardware."
    ),
)
def cli_warm_cache(target: str | None, **config_kwargs) -> None:
    """Load and compile the model to prepare caches."""
    from max.pipelines import PIPELINE_REGISTRY, PipelineArgs, PipelineConfig

    # Log what we're doing if target mode is enabled
    if target:
        from max.serve.config import parse_api_and_target_arch

        api, target_arch = parse_api_and_target_arch(target)
        logging.info(
            f"Compiling for target: {api} ({target_arch}) using virtual devices"
        )

    pipeline_args = PipelineArgs.from_flat_kwargs(**config_kwargs)
    PIPELINE_REGISTRY.retrieve(PipelineConfig.from_args(pipeline_args))


def _render_warm_progress(
    events: Iterator[tuple[str, int]], family_names: Sequence[str]
) -> int:
    """Renders warm progress, one spinner row per op family still compiling.

    Each completion retires its row into a permanent check-mark line above
    the live display, so the live region shrinks to what is still in flight.
    On a non-terminal stdout rich suppresses the live region, so the output
    degrades to the same per-completion lines plus a final summary row.
    Returns the total op count.
    """
    # Lazy import to keep it off the startup path of every other CLI command.
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    total_ops = 0
    with Progress(
        SpinnerColumn(finished_text="[green]✓[/green]"),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
    ) as progress:
        began = progress.get_time()
        overall = progress.add_task(
            f"0/{len(family_names)} op families", total=len(family_names)
        )
        rows = {name: progress.add_task(name, total=1) for name in family_names}
        for done, (name, op_count) in enumerate(events, start=1):
            total_ops += op_count
            plural = "" if op_count == 1 else "s"
            finished_at = timedelta(seconds=int(progress.get_time() - began))
            progress.remove_task(rows[name])
            progress.console.print(
                f"[green]✓[/green] {name} ({op_count} op{plural})"
                f" [yellow]{finished_at}[/yellow]",
                highlight=False,
            )
            progress.update(
                overall,
                advance=1,
                description=f"{done}/{len(family_names)} op families",
            )
    return total_ops


@main.command(name="warm-interpreter-cache")
@click.option(
    "--check",
    "check_only",
    is_flag=True,
    default=False,
    help="Report whether this machine is already warmed, compiling nothing "
    "(exit 0 if so, 1 if not).",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-warm even if this machine is already warmed, e.g. after a "
    "toolchain change.",
)
@click.option(
    "--jobs",
    type=click.IntRange(min=1),
    default=None,
    help="Concurrent compile worker processes. Defaults to one per op "
    "family, capped at the CPU count; 1 compiles in-process, serially.",
)
def cli_warm_interpreter_cache(
    check_only: bool, force: bool, jobs: int | None
) -> None:
    """Compile the eager interpreter's graph-compiler models to prepare caches.

    Batch-compiles every registered op family and stamps this machine as
    provisioned, so a later eager process adopts the warm in one batched load
    instead of compiling per target. Families compile concurrently in worker
    processes (see ``--jobs``); the artifacts land in the shared on-disk
    model cache, which is what a later process reads. Does nothing on an
    already-warmed machine unless ``--force``.
    """
    # Reject conflicting values, not the variables merely being set.
    if _eager_policy.should_precompile():
        var = _eager_policy.OP_PRECOMPILE_ENV_VAR
        detail = (
            "would compile the whole matrix at import, defeating --check's"
            " promise of compiling nothing"
            if check_only
            else "moves the compile sweep into the import instead of this"
            " command's tracked loop, losing the progress display and summary"
        )
        raise click.ClickException(
            f"{var}=1 {detail}. Unset it:"
            f" `env -u {var} max warm-interpreter-cache`."
        )

    if not check_only and not _eager_policy.allow_lazy_compile():
        var = _eager_policy.ALLOW_LAZY_COMPILE_ENV_VAR
        raise click.ClickException(
            f"{var}=0 forbids compiling, which is what this command does."
            f" Unset it: `env -u {var} max warm-interpreter-cache`."
        )

    # Dynamic import: a static ``from`` import fails mypy on Linux (Mojo-backed
    # package with no stub) and pulls the package's heavy import into every CLI
    # command. importlib keeps it lazy and invisible to static analysis.
    # Importing the package (even unused) is what registers its GC families.
    importlib.import_module("max._interpreter_ops")
    gc_compile = importlib.import_module("max._interpreter_ops.gc_compile")

    signature = gc_compile._context_signature()

    if check_only:
        if gc_compile.provisioned():
            click.echo(f"provisioned for {signature}")
            return
        click.echo(f"not provisioned: no warm cache for {signature}")
        raise SystemExit(1)

    cache_dir = gc_compile._cache_dir()
    if cache_dir is None:
        raise click.ClickException(
            "Cannot locate the model cache directory, so a warm here could"
            " not be recorded and nothing would adopt it. Set"
            " MODULAR_DERIVED_PATH to a writable location and retry."
        )

    if gc_compile.provisioned() and not force:
        click.echo(
            f"Already warmed for {signature}"
            " (nothing to do; pass --force to re-warm)"
        )
        return

    devices = gc_compile.DISCOVERED_DEVICES
    click.echo(
        f"Compiling interpreter ops for {len(devices)} device(s):"
        f" {', '.join(gc_compile.device_class_of(d) for d in devices)}"
    )

    warm = importlib.import_module("max._interpreter_ops.warm")

    families = gc_compile.registered_families()
    if jobs is None:
        jobs = min(len(families), os.cpu_count() or 1)
    start = time.perf_counter()
    try:
        total_ops = _render_warm_progress(
            warm.warm_families(jobs), [f.name for f in families]
        )
    except RuntimeError as e:
        raise click.ClickException(str(e)) from e
    elapsed = time.perf_counter() - start

    if gc_compile.write_warm_stamp():
        click.echo(
            f"Compiled {total_ops} ops in {elapsed:.1f}s\n"
            f"Stamp: {cache_dir / gc_compile._WARM_STAMP_NAME}"
        )
    else:
        click.echo(
            f"Compiled {total_ops} ops in {elapsed:.1f}s, but the stamp"
            " could not be written, so later processes will not adopt this"
            " warm."
        )


@main.command(name="list")
@click.option(
    "--json",
    is_flag=True,
    show_default=True,
    default=False,
    help="Print the list of pipelines options in JSON format.",
)
def cli_list(json: bool) -> None:
    """List available pipeline configurations and models.

    This command displays information about all registered pipelines and their
    configurations. Output can be formatted as human-readable text or JSON.
    """
    try:
        from max._entrypoints.cli.list import (
            list_pipelines_to_console,
            list_pipelines_to_json,
        )
    except ImportError as e:
        # This one does not enable telemetry, and this is the first import, so we have to handle this here.
        _handle_import_error(e, "list", ("max-pipelines", "max[serve]"))

    if json:
        list_pipelines_to_json()
    else:
        list_pipelines_to_console()


# Argument parsing is handled by benchmark_serving.parse_args.
# All CLI args other than the explicit ``--profile*`` options below are
# forwarded as-is so the two entry points stay in sync.
@main.command(
    name="benchmark",
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
        "help_option_names": [],
    },
)
@click.option(
    "--profile",
    "profile",
    is_flag=True,
    default=False,
    help=(
        "Translate to `--trace`, then render a ranked top-N GPU kernel "
        "table from the resulting `.nsys-rep` after the run completes. "
        "Unlike `max generate --profile`, this targets the *server* "
        "process (which must already be running under `nsys launch`) — "
        "the benchmark client just sends start/stop signals to nsys."
    ),
)
@click.option(
    "--profile-output",
    "profile_output",
    type=str,
    default=None,
    help=(
        "Path for the .nsys-rep file when `--profile` is on. Default: "
        "$BUILD_WORKSPACE_DIRECTORY/max-profile.nsys-rep, or "
        "./max-profile.nsys-rep."
    ),
)
@click.option(
    "--profile-top-n",
    "profile_top_n",
    type=int,
    default=15,
    show_default=True,
    help="Number of rows to show in the GPU kernel summary table.",
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def cli_benchmark(
    profile: bool,
    profile_output: str | None,
    profile_top_n: int,
    args: Sequence[str],
) -> None:
    """Run benchmark tests on a serving model.

    This command runs comprehensive benchmark tests on a model server to measure
    performance metrics including throughput, latency, and resource utilization.
    Make sure that the MAX server is running and hosting a model before running
    this command.
    """
    # ``--profile`` here is fundamentally different from ``max generate
    # --profile``. Generate runs in-process and re-execs under
    # ``nsys profile``; benchmark is a client that sends requests to a
    # separately running server, so the server must be launched under
    # ``nsys launch`` and we just signal start/stop on the existing nsys
    # session via the existing ``--trace`` flow.
    #
    # Import lazily to avoid importing benchmark modules at module load
    # time.
    try:
        from max.benchmark.sweep_benchmark_serving import main as sweep_main
        from max.profiler import oneshot
    except ImportError as e:
        # This one does not enable telemetry, and this is the first import, so we have to handle this here.
        _handle_import_error(
            e, "benchmark", ("max-benchmark", "max[benchmark]")
        )

    args = list(args)
    profile_path: str | None = None
    if profile:
        profile_path = profile_output or oneshot.default_profile_output()
        args = oneshot.inject_trace_flags(args, profile_path)

    logger.debug("Running benchmark subcommand with args: %s", args)

    benchmark_succeeded = False
    try:
        click.echo("Starting benchmark...")
        sweep_main(args, app_name="max-benchmark")
        click.echo("Benchmark completed successfully!")
        benchmark_succeeded = True
    except SystemExit as e:
        # cyclopts calls sys.exit() for help and errors, we need to handle this
        if e.code == 0:
            # Help was requested and printed, just return
            return
        else:
            # There was an error, exit with the same code
            sys.exit(e.code)
    except Exception as e:
        click.echo(f"Benchmark failed: {e}", err=True)
        sys.exit(1)

    # Render outside the try so a renderer error doesn't get reported as
    # "Benchmark failed".
    if benchmark_succeeded and profile_path is not None:
        if os.path.exists(profile_path):
            oneshot.render_nsys_kernel_summary(
                profile_path, top_n=profile_top_n, out=sys.stdout
            )
            sys.stdout.write(f"\nFull profile saved to: {profile_path}\n")
            sys.stdout.write(f"Open with: nsys-ui {profile_path}\n")
        else:
            sys.stdout.write(
                f"\n--profile: no .nsys-rep found at {profile_path}. The "
                f"server must be launched under `nsys launch` for "
                f"--profile to produce a GPU trace.\n"
            )


def print_version(
    ctx: click.Context, param: click.Parameter, value: bool
) -> None:
    if not value or ctx.resilient_parsing:
        return
    from max import _core

    click.echo(f"MAX {_core.__version__}")
    ctx.exit()


if __name__ == "__main__":
    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)

    main()
