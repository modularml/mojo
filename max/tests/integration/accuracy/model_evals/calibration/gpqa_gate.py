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
"""GPQA logit-shift gate: histogram, sizer, named specs, and live verdict."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

CLAMP = 1e-4
N_PROMPTS = 198
NMAX = 4000
# Explorer sizer: pin H at these named means, R = H - delta.
PARK_STOP = 0.985
PARK_ACC = 0.925
DEFAULT_DELTA_STOP = 0.005
DEFAULT_DELTA_ACC = 0.01
DEFAULT_CAP = 0.01
HIST_MODES = ("bins", "per_prompt")
SUBSETS = ("noisy_15", "plus_bucket", "ever_trunc")
_HIST = Path(__file__).resolve().parent / "data" / "gpqa_gate_hist.json"


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def expit(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def clamp_rate(p: float) -> float:
    return min(max(p, CLAMP), 1.0 - CLAMP)


def apply_shift(rates: list[float], shift: float) -> list[float]:
    return [expit(logit(clamp_rate(p)) + shift) for p in rates]


def shift_for_mean(rates: list[float], target: float) -> float:
    lo, hi = -25.0, 25.0
    n = len(rates)
    assert n > 0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        mean = sum(expit(logit(clamp_rate(p)) + mid) for p in rates) / n
        lo, hi = (mid, hi) if mean < target else (lo, mid)
    return (lo + hi) / 2.0


def _binom_sparse(
    n: int, q: float, tail: float = 12.0
) -> tuple[int, list[float]]:
    if n == 0 or q <= 0:
        return 0, [1.0]
    if q >= 1:
        return n, [1.0]
    mean, sd = n * q, math.sqrt(n * q * (1.0 - q))
    lo = max(0, int(mean - tail * sd - 10))
    hi = min(n, int(mean + tail * sd + 10) + 1)
    lg = math.lgamma
    lp = (
        lg(n + 1)
        - lg(lo + 1)
        - lg(n - lo + 1)
        + lo * math.log(q)
        + (n - lo) * math.log1p(-q)
    )
    vals = [math.exp(lp)]
    ratio = q / (1.0 - q)
    for j in range(lo, hi):
        vals.append(vals[-1] * (n - j) / (j + 1) * ratio)
    return lo, vals


def _conv(
    left: tuple[int, list[float]], right: tuple[int, list[float]]
) -> tuple[int, list[float]]:
    off_a, va = left
    off_b, vb = right
    out = [0.0] * (len(va) + len(vb) - 1)
    for i, x in enumerate(va):
        if x < 1e-300:
            continue
        for j, y in enumerate(vb):
            out[i + j] += x * y
    return off_a + off_b, out


def total_pmf(qs: list[float], n_repeats: int) -> tuple[int, list[float]]:
    dist: tuple[int, list[float]] = (0, [1.0])
    for q in qs:
        if q > 0:
            dist = _conv(dist, _binom_sparse(n_repeats, q))
    off, vals = dist
    lo, hi = 0, len(vals) - 1
    while lo < hi and vals[lo] < 1e-18:
        lo += 1
    while hi > lo and vals[hi] < 1e-18:
        hi -= 1
    return off + lo, vals[lo : hi + 1]


def _cdf(dist: tuple[int, list[float]], k: int) -> float:
    off, vals = dist
    return (
        0.0 if k < off else min(1.0, sum(vals[: min(len(vals), k - off + 1)]))
    )


@dataclass(frozen=True)
class GateSolve:
    n_repeats: int
    cutoff: int
    false_positive: float
    false_negative: float
    mean_h: float
    mean_r: float


def evaluate_repeats(
    qs_h: list[float], qs_r: list[float], n: int, alpha: float
) -> tuple[int, float, float] | None:
    dist_h, dist_r = total_pmf(qs_h, n), total_pmf(qs_r, n)
    kmin, lo, hi = None, 0, n * len(qs_h)
    while lo <= hi:
        mid = (lo + hi) // 2
        if _cdf(dist_h, mid) >= 1.0 - alpha:
            kmin, hi = mid, mid - 1
        else:
            lo = mid + 1
    if kmin is None:
        return None
    return kmin, 1.0 - _cdf(dist_h, kmin), _cdf(dist_r, kmin)


def solve_repeats(
    qs_h: list[float], qs_r: list[float], alpha: float, beta: float
) -> GateSolve | None:
    def ok(n: int) -> tuple[int, float, float] | None:
        found = evaluate_repeats(qs_h, qs_r, n, alpha)
        return found if found is not None and found[2] <= beta else None

    n = 1
    while n <= NMAX and ok(n) is None:
        n *= 2
    if n > NMAX:
        return None
    lo, hi = n // 2 + 1, n
    while lo < hi:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if ok(mid) is not None else (mid + 1, hi)
    found = ok(lo)
    assert found is not None
    k, fp, fn = found
    return GateSolve(lo, k, fp, fn, sum(qs_h), sum(qs_r))


def snr(qs_h: list[float], qs_r: list[float]) -> float:
    var_h = sum(q * (1.0 - q) for q in qs_h)
    return 0.0 if var_h <= 0 else (sum(qs_r) - sum(qs_h)) / math.sqrt(var_h)


@dataclass(frozen=True)
class GateHist:
    q_trunc: list[float]
    q_acc: list[float]
    ever_trunc_ids: list[int]
    noisy_ids: list[int]
    rare_ids: list[int]
    hot_rare_ids: list[int]
    bin_hist: list[int]


def load_gate_hist(path: Path | None = None) -> GateHist:
    payload = json.loads((path or _HIST).read_text())
    q_trunc = [float(x) for x in payload["q_trunc"]]
    assert len(q_trunc) == N_PROMPTS
    return GateHist(
        q_trunc=q_trunc,
        q_acc=[float(x) for x in payload["q_acc"]],
        ever_trunc_ids=[int(i) for i in payload["ever_trunc_ids"]],
        noisy_ids=[int(i) for i in payload["noisy_ids"]],
        rare_ids=[int(i) for i in payload["rare_ids"]],
        hot_rare_ids=[int(i) for i in payload["hot_rare_ids"]],
        bin_hist=[int(c) for c in payload["bin_hist"]],
    )


def subset_ids(hist: GateHist, subset: str) -> list[int]:
    if subset == "noisy_15":
        return list(hist.noisy_ids)
    if subset == "plus_bucket":
        return sorted(set(hist.noisy_ids) | set(hist.hot_rare_ids))
    if subset == "ever_trunc":
        return list(hist.ever_trunc_ids)
    raise KeyError(f"unknown subset {subset!r}")


@dataclass(frozen=True)
class GateSpec:
    name: str
    hist_mode: str
    subset: str
    prompt_ids: list[int]
    alpha: float
    beta: float
    delta_stop: float
    delta_acc: float
    want_stop: bool
    want_acc: bool


def make_spec(
    hist_mode: str,
    subset: str,
    *,
    alpha: float = DEFAULT_CAP,
    beta: float = DEFAULT_CAP,
    delta_stop: float = DEFAULT_DELTA_STOP,
    delta_acc: float = DEFAULT_DELTA_ACC,
    want_stop: bool = True,
    want_acc: bool = True,
    prompt_ids: list[int] | None = None,
    hist: GateHist | None = None,
) -> GateSpec:
    if hist_mode not in HIST_MODES or subset not in SUBSETS:
        raise KeyError(f"hist={hist_mode!r} subset={subset!r}")
    if not want_stop and not want_acc:
        raise ValueError("need stop and/or accuracy")
    data = hist or load_gate_hist()
    ids = subset_ids(data, subset) if prompt_ids is None else list(prompt_ids)
    if not ids:
        raise ValueError("need at least one prompt id")
    if any(i < 0 or i >= N_PROMPTS for i in ids):
        raise ValueError(f"prompt ids must be in 0..{N_PROMPTS - 1}")
    return GateSpec(
        name=f"{hist_mode}:{subset}",
        hist_mode=hist_mode,
        subset=subset,
        prompt_ids=ids,
        alpha=alpha,
        beta=beta,
        delta_stop=delta_stop,
        delta_acc=delta_acc,
        want_stop=want_stop,
        want_acc=want_acc,
    )


def catalog(
    hist: GateHist | None = None,
    *,
    alpha: float = DEFAULT_CAP,
    beta: float = DEFAULT_CAP,
    delta_stop: float = DEFAULT_DELTA_STOP,
    delta_acc: float = DEFAULT_DELTA_ACC,
    want_stop: bool = True,
    want_acc: bool = True,
) -> list[GateSpec]:
    data = hist or load_gate_hist()
    return [
        make_spec(
            hist_mode,
            subset,
            alpha=alpha,
            beta=beta,
            delta_stop=delta_stop,
            delta_acc=delta_acc,
            want_stop=want_stop,
            want_acc=want_acc,
            hist=data,
        )
        for hist_mode in HIST_MODES
        for subset in SUBSETS
    ]


def _bin_raw(hist: GateHist, n_bucket: int, bucket_rate: float) -> list[float]:
    counts = list(hist.bin_hist)
    if n_bucket:
        if counts[0] < n_bucket:
            raise ValueError("rare bucket larger than bin 0")
        counts[0] -= n_bucket
    raw = [k / 10.0 for k, count in enumerate(counts) for _ in range(count)]
    raw.extend([bucket_rate] * n_bucket)
    return raw


def _bins_raw(spec: GateSpec, hist: GateHist) -> list[float]:
    if spec.subset == "noisy_15":
        return _bin_raw(hist, 0, 0.0)
    if spec.subset == "plus_bucket":
        ids = hist.hot_rare_ids
        return _bin_raw(
            hist, len(ids), sum(hist.q_trunc[i] for i in ids) / len(ids)
        )
    ids = hist.rare_ids
    return _bin_raw(
        hist, len(ids), sum(hist.q_trunc[i] for i in ids) / len(ids)
    )


def _stop_park(spec: GateSpec) -> tuple[float, float, float]:
    return PARK_STOP, PARK_STOP, PARK_STOP - spec.delta_stop


def _acc_park(spec: GateSpec) -> tuple[float, float, float]:
    return PARK_ACC, PARK_ACC, PARK_ACC - spec.delta_acc


def _stop_coins(
    spec: GateSpec, hist: GateHist
) -> tuple[list[float], list[float]]:
    _, stop_h, stop_r = _stop_park(spec)
    raw = (
        hist.q_trunc
        if spec.hist_mode == "per_prompt"
        else _bins_raw(spec, hist)
    )
    qh = apply_shift(raw, shift_for_mean(raw, 1.0 - stop_h))
    qr = apply_shift(raw, shift_for_mean(raw, 1.0 - stop_r))
    if spec.hist_mode == "per_prompt":
        return [qh[i] for i in spec.prompt_ids], [
            qr[i] for i in spec.prompt_ids
        ]
    return [p for p, rate in zip(qh, raw, strict=True) if rate > 0], [
        p for p, rate in zip(qr, raw, strict=True) if rate > 0
    ]


def _acc_coins(
    spec: GateSpec, hist: GateHist
) -> tuple[list[float], list[float]]:
    _, acc_h, acc_r = _acc_park(spec)
    q = hist.q_acc
    qh = apply_shift(q, shift_for_mean(q, acc_h))
    qr = apply_shift(q, shift_for_mean(q, acc_r))
    return [1.0 - qh[i] for i in spec.prompt_ids], [
        1.0 - qr[i] for i in spec.prompt_ids
    ]


@dataclass(frozen=True)
class ScoredGate:
    spec: GateSpec
    n_repeats: int
    stop_cutoff: int | None
    acc_cutoff: int | None
    stop_fp: float
    stop_fn: float
    acc_fp: float
    acc_fn: float
    cost: int
    stop_snr: float
    acc_snr: float
    base_stop: float
    stop_h: float
    stop_r: float
    base_acc: float
    acc_h: float
    acc_r: float


def _need(solve: GateSolve | None, msg: str) -> GateSolve:
    if solve is None:
        raise ValueError(msg)
    return solve


def _solve_at(
    qs_h: list[float],
    qs_r: list[float],
    n: int,
    alpha: float,
    msg: str,
) -> GateSolve:
    ev = evaluate_repeats(qs_h, qs_r, n, alpha)
    return _need(
        None
        if ev is None
        else GateSolve(n, ev[0], ev[1], ev[2], sum(qs_h), sum(qs_r)),
        msg,
    )


def score_spec(
    spec: GateSpec,
    hist: GateHist | None = None,
    *,
    n_repeats: int | None = None,
) -> ScoredGate:
    if n_repeats is not None and n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")
    data = hist or load_gate_hist()
    stop_solve: GateSolve | None = None
    acc_solve: GateSolve | None = None
    stop_h_coins: list[float] = []
    stop_r_coins: list[float] = []
    acc_h_coins: list[float] = []
    acc_r_coins: list[float] = []
    needed: list[int] = []
    if spec.want_stop:
        stop_h_coins, stop_r_coins = _stop_coins(spec, data)
        if n_repeats is None:
            stop_solve = _need(
                solve_repeats(
                    stop_h_coins, stop_r_coins, spec.alpha, spec.beta
                ),
                f"{spec.name}: stop infeasible",
            )
            needed.append(stop_solve.n_repeats)
        else:
            stop_solve = _solve_at(
                stop_h_coins,
                stop_r_coins,
                n_repeats,
                spec.alpha,
                f"{spec.name}: stop infeasible",
            )
    if spec.want_acc:
        acc_h_coins, acc_r_coins = _acc_coins(spec, data)
        if n_repeats is None:
            acc_solve = _need(
                solve_repeats(acc_h_coins, acc_r_coins, spec.alpha, spec.beta),
                f"{spec.name}: acc infeasible",
            )
            needed.append(acc_solve.n_repeats)
        else:
            acc_solve = _solve_at(
                acc_h_coins,
                acc_r_coins,
                n_repeats,
                spec.alpha,
                f"{spec.name}: acc infeasible",
            )
    m = n_repeats if n_repeats is not None else max(needed)
    if (
        n_repeats is None
        and stop_solve is not None
        and m != stop_solve.n_repeats
    ):
        stop_solve = _solve_at(
            stop_h_coins,
            stop_r_coins,
            m,
            spec.alpha,
            f"{spec.name}: stop infeasible",
        )
    if n_repeats is None and acc_solve is not None and m != acc_solve.n_repeats:
        acc_solve = _solve_at(
            acc_h_coins,
            acc_r_coins,
            m,
            spec.alpha,
            f"{spec.name}: acc infeasible",
        )
    base_stop, park_stop_h, park_stop_r = _stop_park(spec)
    base_acc, park_acc_h, park_acc_r = _acc_park(spec)
    return ScoredGate(
        spec,
        m,
        stop_solve.cutoff if stop_solve is not None else None,
        acc_solve.cutoff if acc_solve is not None else None,
        stop_solve.false_positive if stop_solve is not None else 0.0,
        stop_solve.false_negative if stop_solve is not None else 0.0,
        acc_solve.false_positive if acc_solve is not None else 0.0,
        acc_solve.false_negative if acc_solve is not None else 0.0,
        m * len(spec.prompt_ids),
        snr(stop_h_coins, stop_r_coins) if stop_solve is not None else 0.0,
        snr(acc_h_coins, acc_r_coins) if acc_solve is not None else 0.0,
        base_stop,
        park_stop_h,
        park_stop_r,
        base_acc,
        park_acc_h,
        park_acc_r,
    )


def score_catalog(
    hist: GateHist | None = None,
    *,
    alpha: float = DEFAULT_CAP,
    beta: float = DEFAULT_CAP,
    delta_stop: float = DEFAULT_DELTA_STOP,
    delta_acc: float = DEFAULT_DELTA_ACC,
    want_stop: bool = True,
    want_acc: bool = True,
) -> list[ScoredGate]:
    data = hist or load_gate_hist()
    return [
        score_spec(spec, data)
        for spec in catalog(
            data,
            alpha=alpha,
            beta=beta,
            delta_stop=delta_stop,
            delta_acc=delta_acc,
            want_stop=want_stop,
            want_acc=want_acc,
        )
    ]


VerdictStatus = Literal["pass", "fail", "error"]


@dataclass(frozen=True)
class LiveVerdict:
    status: VerdictStatus
    rationale: str
    n_rows: int
    n_trunc: int
    n_wrong: int
    stop_cutoff: int | None
    acc_cutoff: int | None


def score_results(results_path: Path, scored: ScoredGate) -> LiveVerdict:
    rows: list[dict[str, object]] = []
    n_err = 0
    for line in results_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "error" in row:
            n_err += 1
        else:
            rows.append(row)
    expected = scored.cost
    if n_err or len(rows) < expected:
        return LiveVerdict(
            "error",
            f"{n_err} errors, {len(rows)}/{expected} rows",
            len(rows) + n_err,
            0,
            0,
            scored.stop_cutoff,
            scored.acc_cutoff,
        )
    n_trunc = sum(1 for row in rows if row.get("finish_reason") == "length")
    n_wrong = sum(1 for row in rows if not row.get("correct"))
    stop_ok = scored.stop_cutoff is None or n_trunc <= scored.stop_cutoff
    acc_ok = scored.acc_cutoff is None or n_wrong <= scored.acc_cutoff
    status: VerdictStatus = "pass" if stop_ok and acc_ok else "fail"
    parts: list[str] = []
    if scored.stop_cutoff is not None:
        parts.append(f"stop: {n_trunc} vs k={scored.stop_cutoff}")
    if scored.acc_cutoff is not None:
        parts.append(f"accuracy: {n_wrong} vs k={scored.acc_cutoff}")
    return LiveVerdict(
        status,
        "; ".join(parts),
        len(rows),
        n_trunc,
        n_wrong,
        scored.stop_cutoff,
        scored.acc_cutoff,
    )
