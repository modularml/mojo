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

import math
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import torch


@dataclass
class Stat:
    """Aggregated timing statistics for a labeled section."""

    calls: int = 0
    total_s: float = 0.0
    samples: list[float] = field(default_factory=list)

    def add(self, dt_s: float) -> None:
        """Accumulate a duration sample in seconds."""
        self.calls += 1
        self.total_s += dt_s
        self.samples.append(dt_s)

    @property
    def avg_s(self) -> float:
        """Average duration per call in seconds."""
        return self.total_s / self.calls if self.calls else 0.0

    @property
    def std_s(self) -> float:
        """Sample standard deviation of per-call durations in seconds."""
        if len(self.samples) < 2:
            return 0.0
        mean = self.avg_s
        var = sum((s - mean) ** 2 for s in self.samples) / (
            len(self.samples) - 1
        )
        return math.sqrt(var)

    @property
    def min_s(self) -> float:
        """Minimum per-call duration in seconds."""
        return min(self.samples) if self.samples else 0.0

    @property
    def max_s(self) -> float:
        """Maximum per-call duration in seconds."""
        return max(self.samples) if self.samples else 0.0


class _Patcher:
    """Patch attributes and restore them on context exit."""

    def __init__(self) -> None:
        self._patched: list[tuple[object, str, Any]] = []

    def patch(self, obj: object, name: str, new_value: Any) -> None:
        """Replace `obj.name` with `new_value`, recording the original value."""
        old_value: Any = getattr(obj, name)
        self._patched.append((obj, name, old_value))
        setattr(obj, name, new_value)

    def restore(self) -> None:
        """Restore patched attributes in reverse order."""
        for obj, name, old_value in reversed(self._patched):
            setattr(obj, name, old_value)
        self._patched.clear()


class _TimedFn:
    """Callable wrapper that measures wall-clock time."""

    def __init__(
        self, fn: Callable[..., Any], on_time: Callable[[float], None]
    ) -> None:
        self._fn: Callable[..., Any] = fn
        self._on_time: Callable[[float], None] = on_time

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        torch.cuda.synchronize()
        t0: float = perf_counter()
        try:
            return self._fn(*args, **kwargs)
        finally:
            torch.cuda.synchronize()
            self._on_time(perf_counter() - t0)


class _TimedCallableProxy:
    """Callable proxy that times `obj(*args, **kwargs)`.

    Additionally, selected method names can be timed when accessed, e.g.
    `vae.encode(...)` and `vae.decode(...)`.
    """

    def __init__(
        self,
        obj: Any,
        on_time_call: Callable[[float], None],
        timed_methods: Mapping[str, Callable[[float], None]] | None = None,
    ) -> None:
        self._obj: Any = obj
        self._on_time_call: Callable[[float], None] = on_time_call
        self._timed_methods: Mapping[str, Callable[[float], None]] = (
            timed_methods or {}
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        torch.cuda.synchronize()
        t0: float = perf_counter()
        try:
            return self._obj(*args, **kwargs)
        finally:
            torch.cuda.synchronize()
            self._on_time_call(perf_counter() - t0)

    def __getattr__(self, name: str) -> Any:
        attr: Any = getattr(self._obj, name)
        on_time: Callable[[float], None] | None = self._timed_methods.get(name)
        if on_time is not None and callable(attr):
            return _TimedFn(attr, on_time)
        return attr


class ExecuteProfiler(AbstractContextManager["ExecuteProfiler"]):
    """Profile diffusion pipelines with method- and component-level aggregation.

    This profiler provides:
      - method-level labels in `stats`
      - module-level labels in `component_stats`

    It guarantees component-level timings by replacing `pipeline.<component>`
    attributes with proxies during the context.
    """

    def __init__(
        self,
        pipeline_or_wrapper: Any,
        *,
        enabled: bool = True,
        patch_tensor_ops: bool = False,
        is_diffusers: bool = False,
    ) -> None:
        self._obj: Any = pipeline_or_wrapper
        self._enabled: bool = enabled
        self._patch_tensor_ops: bool = patch_tensor_ops
        self._is_diffusers: bool = is_diffusers

        self.stats: dict[str, Stat] = {}
        self.component_stats: dict[str, Stat] = {}

        self._patcher: _Patcher = _Patcher()
        self._model: Any | None = None
        self._pipeline_module_label: str = "pipeline/unknown"

    def __enter__(self) -> ExecuteProfiler:
        if not self._enabled:
            return self

        self._model = self._resolve_inner_model(self._obj)
        target: Any = self._model if self._model is not None else self._obj
        self._pipeline_module_label = f"pipeline/{type(target).__name__}"

        self._wrap_methods(target)
        self._wrap_components(target)
        if self._patch_tensor_ops:
            self._wrap_tensor_ops()

        return self

    def __exit__(
        self, exc_type: object, exc: BaseException | None, tb: object
    ) -> None:
        self._patcher.restore()

    def report(self, unit: str = "ms") -> None:
        """Print a full profiling report with component and method tables."""
        print("==================== PROFILING REPORT ==")
        print(f"Component Timings:\n{self.report_components(unit=unit)}\n")
        print(f"Method Timings:\n{self.report_methods(unit=unit)}")
        print("==========================================")

    def report_methods(self, *, unit: str = "ms") -> str:
        """Render a method-level timing table."""
        mul: float = 1000.0 if unit == "ms" else 1.0
        items: list[tuple[str, Stat]] = sorted(
            self.stats.items(), key=lambda kv: kv[1].total_s, reverse=True
        )
        lines: list[str] = [
            f"{'methods':<30} {'calls':>7} {'total':>12} {'avg':>12} "
            f"{'std':>12} {'min':>12} {'max':>12} ({unit})"
        ]
        for name, st in items:
            lines.append(
                f"{name:<30} {st.calls:>7d} {st.total_s * mul:>12.3f} "
                f"{st.avg_s * mul:>12.3f} {st.std_s * mul:>12.3f} "
                f"{st.min_s * mul:>12.3f} {st.max_s * mul:>12.3f}"
            )
        return "\n".join(lines)

    def report_components(self, *, unit: str = "ms") -> str:
        """Render a component-level timing table."""
        mul: float = 1000.0 if unit == "ms" else 1.0
        items: list[tuple[str, Stat]] = sorted(
            self.component_stats.items(),
            key=lambda kv: kv[1].total_s,
            reverse=True,
        )
        lines: list[str] = [
            f"{'components':<30} {'calls':>7} {'total':>12} {'avg':>12} "
            f"{'std':>12} {'min':>12} {'max':>12} ({unit})"
        ]
        for name, st in items:
            lines.append(
                f"{name:<30} {st.calls:>7d} {st.total_s * mul:>12.3f} "
                f"{st.avg_s * mul:>12.3f} {st.std_s * mul:>12.3f} "
                f"{st.min_s * mul:>12.3f} {st.max_s * mul:>12.3f}"
            )
        return "\n".join(lines)

    def _resolve_inner_model(self, obj: Any) -> Any | None:
        """Find an inner diffusion model instance inside wrapper pipelines."""
        candidates: tuple[str, ...] = (
            "pipeline_model",
            "model",
            "_model",
            "_pipeline_model",
            "diffusion_pipeline",
            "_diffusion_pipeline",
        )
        for name in candidates:
            if hasattr(obj, name):
                inner: Any = getattr(obj, name)
                if (
                    inner is not None
                    and hasattr(inner, "execute")
                    and callable(inner.execute)
                ):
                    return inner
        return None

    def _accum(self, label: str, dt_s: float) -> None:
        self.stats.setdefault(label, Stat()).add(dt_s)

    def _accum_component(self, label: str, dt_s: float) -> None:
        self.component_stats.setdefault(label, Stat()).add(dt_s)

    def _resolve_attr_path(self, obj: Any, path: str) -> tuple[Any, str]:
        """Resolve a dotted attribute path, returning (parent_obj, attr_name)."""
        parts = path.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        return obj, parts[-1]

    def _wrap_methods(self, target: Any) -> None:
        """Wrap pipeline method calls for timing."""
        if not self._is_diffusers:
            specs = [
                ("execute", "E2E execute"),
                ("prepare_prompt_embeddings", "prepare_embeddings"),
                ("decode_latents", "decode_latents"),
                ("prepare_scheduler", "prepare_scheduler"),
                ("scheduler_step", "scheduler_step"),
                ("preprocess_latents", "preprocess_latents"),
                ("prepare_image_latents", "prepare_image_latents"),
            ]
        else:
            specs = [
                ("__call__", "E2E execute"),
                ("encode_prompt", "encode_prompt"),
                ("prepare_latents", "prepare_latents"),
                ("retrieve_timesteps", "retrieve_timesteps"),
                ("_unpack_latents_with_ids", "_unpack_latents_with_ids"),
                ("_unpatchify_latents", "_unpatchify_latents"),
                ("scheduler.step", "scheduler.step"),
                ("image_processor.postprocess", "image_processor.postprocess"),
            ]
        for method_name, label in specs:
            self._wrap_method_if_exists(target, method_name, label)

    def _wrap_method_if_exists(
        self, obj: Any, method_name: str, label: str
    ) -> None:
        """Wrap `obj.method_name` and aggregate into method stats."""
        try:
            parent, attr_name = self._resolve_attr_path(obj, method_name)
        except AttributeError:
            return

        # Special handling for __call__: patch on the type
        if attr_name == "__call__":
            target_obj: Any = type(parent)
        else:
            target_obj = parent

        if not hasattr(target_obj, attr_name):
            return
        original: Any = getattr(target_obj, attr_name)
        if not callable(original):
            return
        if getattr(original, "__is_execute_profiler_wrapper__", False):
            return

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            torch.cuda.synchronize()
            t0: float = perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                torch.cuda.synchronize()
                dt: float = perf_counter() - t0
                self._accum(label, dt)

        wrapper.__is_execute_profiler_wrapper__ = True  # type: ignore[attr-defined]
        wrapper.__wrapped__ = original  # type: ignore[attr-defined]

        try:
            self._patcher.patch(target_obj, attr_name, wrapper)
        except (AttributeError, TypeError):
            pass

    def _wrap_components(self, target: Any) -> None:
        """Replace `target.<component>` with a proxy or wrap its methods.

        For the VAE, also time `.encode()` and `.decode()` calls explicitly.
        """
        comps: Any = getattr(target, "components", {}) or {}
        if not isinstance(comps, dict):
            return

        for name in comps:
            if name in ("scheduler", "tokenizer"):
                continue
            if not hasattr(target, name):
                continue
            comp: Any = getattr(target, name)
            if comp is None:
                continue

            base_label: str = f"component/{name}"

            if self._is_diffusers:
                self._wrap_component_method_if_exists(
                    comp, "forward", base_label
                )
                if name == "vae":
                    self._wrap_component_method_if_exists(
                        comp, "encode", "component/vae.encode"
                    )
                    self._wrap_component_method_if_exists(
                        comp, "decode", "component/vae.decode"
                    )
                continue

            # Non-diffusers (MAX): keep proxy approach
            def make_on_time(lab: str) -> Callable[[float], None]:
                def on_time(dt: float) -> None:
                    self._accum(lab, dt)
                    self._accum_component(lab, dt)

                return on_time

            timed_methods: dict[str, Callable[[float], None]] = {}

            if name == "vae":
                timed_methods["encode"] = make_on_time("component/vae.encode")
                timed_methods["decode"] = make_on_time("component/vae.decode")

            proxy: _TimedCallableProxy = _TimedCallableProxy(
                comp,
                on_time_call=make_on_time(base_label),
                timed_methods=timed_methods,
            )

            try:
                self._patcher.patch(target, name, proxy)
            except (AttributeError, TypeError):
                continue

    def _wrap_component_method_if_exists(
        self, obj: Any, method_name: str, label: str
    ) -> None:
        """Wrap `obj.method_name` and aggregate into component stats."""
        if not hasattr(obj, method_name):
            return
        original: Any = getattr(obj, method_name)
        if not callable(original):
            return
        if getattr(original, "__is_execute_profiler_wrapper__", False):
            return

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            torch.cuda.synchronize()
            t0: float = perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                torch.cuda.synchronize()
                dt: float = perf_counter() - t0
                self._accum_component(label, dt)

        wrapper.__is_execute_profiler_wrapper__ = True  # type: ignore[attr-defined]
        wrapper.__wrapped__ = original  # type: ignore[attr-defined]

        try:
            self._patcher.patch(obj, method_name, wrapper)
        except (AttributeError, TypeError):
            pass

    def _wrap_tensor_ops(self) -> None:
        """Patch Tensor conversion/movement ops (process-wide while active)."""
        try:
            from max.experimental.tensor import Tensor
        except Exception:
            return

        if hasattr(Tensor, "from_dlpack") and callable(Tensor.from_dlpack):
            orig_from: Any = Tensor.from_dlpack
            if not getattr(orig_from, "__is_execute_profiler_wrapper__", False):

                def from_dlpack_wrapped(*args: Any, **kwargs: Any) -> Any:
                    torch.cuda.synchronize()
                    t0: float = perf_counter()
                    try:
                        return orig_from(*args, **kwargs)
                    finally:
                        torch.cuda.synchronize()
                        dt: float = perf_counter() - t0
                        self._accum("tensor/from_dlpack", dt)

                from_dlpack_wrapped.__is_execute_profiler_wrapper__ = True  # type: ignore[attr-defined]
                from_dlpack_wrapped.__wrapped__ = orig_from  # type: ignore[attr-defined]
                try:
                    self._patcher.patch(
                        Tensor, "from_dlpack", from_dlpack_wrapped
                    )
                except (AttributeError, TypeError):
                    pass

        for meth, label in (("to", "tensor/to"), ("cast", "tensor/cast")):
            if hasattr(Tensor, meth) and callable(getattr(Tensor, meth)):
                orig_m: Any = getattr(Tensor, meth)
                if getattr(orig_m, "__is_execute_profiler_wrapper__", False):
                    continue

                def _make_wrapped(
                    o: Callable[..., Any], meth_label: str
                ) -> Callable[..., Any]:
                    def wrapped(self_: Any, *args: Any, **kwargs: Any) -> Any:
                        torch.cuda.synchronize()
                        t0: float = perf_counter()
                        try:
                            return o(self_, *args, **kwargs)
                        finally:
                            torch.cuda.synchronize()
                            dt: float = perf_counter() - t0
                            self._accum(meth_label, dt)

                    wrapped.__is_execute_profiler_wrapper__ = True  # type: ignore[attr-defined]
                    wrapped.__wrapped__ = o  # type: ignore[attr-defined]
                    return wrapped

                wrapped_fn: Callable[..., Any] = _make_wrapped(orig_m, label)
                try:
                    self._patcher.patch(Tensor, meth, wrapped_fn)
                except (AttributeError, TypeError):
                    pass


def profile_execute(
    pipeline_or_wrapper: Any,
    *,
    enabled: bool = True,
    patch_tensor_ops: bool = False,
    is_diffusers: bool = False,
) -> ExecuteProfiler:
    """Create a profiler for a diffusion pipeline or a wrapper pipeline."""
    return ExecuteProfiler(
        pipeline_or_wrapper,
        enabled=enabled,
        patch_tensor_ops=patch_tensor_ops,
        is_diffusers=is_diffusers,
    )
