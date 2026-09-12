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

"""Probability distribution abstractions used for parameterized values.

Supports parsing distribution specifications from strings like "N(mean, std)"
for normal distributions, "U(lower, upper)" for continuous uniform distributions,
"DU(lower, upper)" for discrete uniform distributions, "NB(n, p)" for
negative binomial distributions, "G(shape, scale)" for gamma distributions,
"LN(mean, std)" for log-normal distributions, "Burr12(c, d, scale)" for
Burr Type XII distributions, "Cat(v1:w1, v2:w2, ...)" for an explicit weighted
set of values, as well as plain float values for constant returns.

The class hierarchy separates continuous (float-valued) and discrete
(int-valued) distributions. `CategoricalDistribution` sits alongside
`ConstantDistribution` outside that split (like `ConstantDistribution`, its
values are whatever the caller supplied; callers needing an int round the
sampled value themselves, the same way they already do for
`ConstantDistribution`):

    BaseDistribution
    ├── ConstantDistribution
    ├── CategoricalDistribution        Cat(v1:w1, v2:w2, ...)
    ├── ContinuousDistribution
    │   ├── NormalDistribution        N(mean, std)
    │   ├── UniformDistribution       U(lower, upper)
    │   ├── GammaDistribution         G(shape, scale)
    │   ├── LogNormalDistribution     LN(mean, std)
    │   └── Burr12Distribution        Burr12(c, d, scale)
    └── DiscreteDistribution
        ├── DiscreteUniformDistribution   DU(lower, upper)
        └── NegativeBinomialDistribution  NB(n, p)

`CategoricalDistribution` is the general-purpose escape hatch for empirical
distributions that don't fit a standard parametric shape (for example, a
handful of image dimensions measured from real production traffic). It works
with any `DistributionParameter` field in this package, not just images.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

DistributionParameter = int | float | str
"""Type alias for parameters that accept a float, an int, or a distribution string."""


def _match_distribution_args(
    schema: str, prefix_pattern: str, num_args: int
) -> list[str] | None:
    """Match a "PREFIX(a, b, ...)" distribution string and return its raw,
    unparsed argument strings, or None if it doesn't match.

    Shared by every parametric distribution's ``parse_from_str_schema``; each
    still owns its own numeric conversion (float vs. int) and error messages.

    Args:
        schema: The full distribution string, e.g. "N(100, 5)".
        prefix_pattern: A regex matching the case-insensitive prefix before
            the opening paren, e.g. ``r"[Nn]"`` or ``r"[Ll][Nn]"``.
        num_args: Expected number of comma-separated arguments.

    Returns:
        The ``num_args`` raw argument strings, in order, or None if
        ``schema`` doesn't match ``prefix_pattern(...)`` with exactly that
        many comma-separated arguments.
    """
    arg_pattern = r"\s*,\s*".join([r"([^,)]+)"] * num_args)
    match = re.match(rf"{prefix_pattern}\(\s*{arg_pattern}\s*\)", schema)
    return list(match.groups()) if match else None


class BaseDistribution(ABC):
    """Abstract base class for probability distributions used in benchmarks."""

    @abstractmethod
    def sample_value(self) -> float | int:
        """Sample a single value from this distribution.

        Returns:
            A sampled value (float for continuous, int for discrete).
        """
        ...

    @classmethod
    def from_distribution_parameter_or_raise(
        cls, param: DistributionParameter
    ) -> BaseDistribution:
        """Parse a distribution parameter that the caller must supply.

        Args:
            param: An int or float, or a distribution string.

        Returns:
            A concrete distribution.

        Raises:
            ValueError: If ``param`` does not resolve to a distribution.
        """
        dist = cls.from_distribution_parameter(param)
        if dist is None:
            raise ValueError(
                f"a resolved distribution is required; got {param!r}."
            )
        return dist

    @classmethod
    def from_distribution_parameter(
        cls, param: DistributionParameter | None
    ) -> BaseDistribution | None:
        """Parse a distribution parameter into a concrete distribution instance.

        Args:
            param: An int or float, a string like "N(mean,std)",
                "U(lower,upper)", "DU(lower,upper)", "NB(n,p)",
                "G(shape,scale)", "LN(mean,std)", "Burr12(c,d,scale)",
                "Cat(v1:w1,v2:w2,...)", or None.

        Returns:
            A BaseDistribution instance, or None if param is None.

        Raises:
            ValueError: If the string format is unrecognized or unparseable.
        """
        if param is None:
            return None

        elif isinstance(param, (float, int)):
            return ConstantDistribution(value=float(param))

        elif isinstance(param, str):
            stripped = param.strip()

            # Try parsing as a plain float string
            try:
                return ConstantDistribution(value=float(stripped))
            except ValueError:
                pass

            upper = stripped.upper()
            if upper.startswith("BURR12("):
                return Burr12Distribution.parse_from_str_schema(stripped)
            elif upper.startswith("CAT("):
                return CategoricalDistribution.parse_from_str_schema(stripped)
            elif upper.startswith("LN("):
                return LogNormalDistribution.parse_from_str_schema(stripped)
            elif upper.startswith("NB("):
                return NegativeBinomialDistribution.parse_from_str_schema(
                    stripped
                )
            elif upper.startswith("N("):
                return NormalDistribution.parse_from_str_schema(stripped)
            elif upper.startswith("DU("):
                return DiscreteUniformDistribution.parse_from_str_schema(
                    stripped
                )
            elif upper.startswith("U("):
                return UniformDistribution.parse_from_str_schema(stripped)
            elif upper.startswith("G("):
                return GammaDistribution.parse_from_str_schema(stripped)
            else:
                raise ValueError(
                    f"Unrecognized distribution format: '{param}'. "
                    "Expected a float, 'N(mean,std)', "
                    "'U(lower,upper)', 'DU(lower,upper)', 'NB(n,p)', "
                    "'G(shape,scale)', 'LN(mean,std)', "
                    "'Burr12(c,d,scale)', or 'Cat(v1:w1,v2:w2,...)'."
                )

        else:
            raise TypeError(
                "Expected float, str, or None for distribution parameter, got "
                "{type(param)}"
            )


class ContinuousDistribution(BaseDistribution):
    """Abstract base for continuous distributions that sample float values."""

    @abstractmethod
    def sample_value(self) -> float: ...


class DiscreteDistribution(BaseDistribution):
    """Abstract base for discrete distributions that sample integer values."""

    @abstractmethod
    def sample_value(self) -> int: ...


@dataclass
class ConstantDistribution(BaseDistribution):
    """A degenerate distribution that always returns the same value."""

    value: float

    def sample_value(self) -> float:
        return self.value


@dataclass
class CategoricalDistribution(BaseDistribution):
    """A distribution over an explicit, weighted set of values.

    The general-purpose escape hatch for empirical distributions that don't
    fit one of the standard parametric shapes above — for example, a
    handful of image dimensions measured from real production traffic.
    Works with any `DistributionParameter` field, not just images.

    String schema: ``Cat(v1:w1, v2:w2, ...)``. The ``:weight`` suffix is
    optional per entry and defaults to ``1.0``, so ``Cat(v1, v2, v3)``
    samples uniformly among the three values. Weights need not sum to 1;
    they're normalized internally.
    """

    values: list[float]
    weights: list[float]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError(
                "Categorical distribution requires at least one value"
            )
        if len(self.values) != len(self.weights):
            raise ValueError(
                f"values ({len(self.values)} entries) and weights"
                f" ({len(self.weights)} entries) must have the same length"
            )
        # nan slips past every comparison below (nan < 0 and nan <= 0 are both
        # False), so it would reach np.random.choice and fail there instead of
        # at the typo that produced it.
        if any(not math.isfinite(v) for v in self.values):
            raise ValueError("Values must be finite")
        if any(not math.isfinite(w) for w in self.weights):
            raise ValueError("Weights must be finite")
        if any(w < 0 for w in self.weights):
            raise ValueError("Weights must be non-negative")
        if sum(self.weights) <= 0:
            raise ValueError("Weights must sum to a positive value")

    def sample_value(self) -> float:
        # Individually finite weights can still sum to inf, which would send
        # every probability to 0. Scaling by the max preserves the ratios and
        # bounds the sum by len(weights).
        scale = max(self.weights)
        normalized = [w / scale for w in self.weights]
        total = sum(normalized)
        probabilities = [w / total for w in normalized]
        return float(np.random.choice(self.values, p=probabilities))

    @classmethod
    def parse_from_str_schema(cls, schema: str) -> CategoricalDistribution:
        """Parse a string like "Cat(v1:w1, v2:w2, ...)" into a CategoricalDistribution.

        Args:
            schema: A string in the format "Cat(v1:w1, v2:w2, ...)"
                (case-insensitive prefix). The ":weight" suffix is optional
                per entry and defaults to 1.0 (uniform) when omitted.

        Returns:
            A CategoricalDistribution instance.

        Raises:
            ValueError: If the string cannot be parsed.
        """
        match = re.match(r"(?i)cat\((.*)\)\s*$", schema)
        if not match:
            raise ValueError(
                f"Cannot parse categorical distribution from '{schema}'. "
                "Expected format: 'Cat(v1:w1, v2:w2, ...)'."
            )
        entries = [entry.strip() for entry in match.group(1).split(",")]
        values: list[float] = []
        weights: list[float] = []
        for entry in entries:
            parts = entry.split(":")
            if not entry or len(parts) not in (1, 2):
                raise ValueError(
                    f"Cannot parse categorical entry '{entry}' in '{schema}'."
                    " Expected 'value' or 'value:weight'."
                )
            try:
                value = float(parts[0])
                weight = float(parts[1]) if len(parts) == 2 else 1.0
            except ValueError as e:
                raise ValueError(
                    f"Cannot parse numeric values from '{entry}' in"
                    f" '{schema}': {e}"
                ) from e
            values.append(value)
            weights.append(weight)
        return cls(values=values, weights=weights)


@dataclass
class NormalDistribution(ContinuousDistribution):
    """A normal (Gaussian) distribution parameterized by mean and std."""

    mean: float
    std: float

    def __post_init__(self) -> None:
        if self.std < 0:
            raise ValueError(
                f"Standard deviation must be non-negative, got {self.std}"
            )

    def sample_value(self) -> float:
        return float(np.random.normal(loc=self.mean, scale=self.std))

    @classmethod
    def parse_from_str_schema(cls, schema: str) -> NormalDistribution:
        """Parse a string like "N(mean, std)" into a NormalDistribution.

        Args:
            schema: A string in the format "N(mean, std)" (case-insensitive prefix).

        Returns:
            A NormalDistribution instance.

        Raises:
            ValueError: If the string cannot be parsed.
        """
        args = _match_distribution_args(schema, r"[Nn]", 2)
        if args is None:
            raise ValueError(
                f"Cannot parse normal distribution from '{schema}'. "
                "Expected format: 'N(mean, std)'."
            )
        try:
            mean = float(args[0])
            std = float(args[1])
        except ValueError as e:
            raise ValueError(
                f"Cannot parse numeric values from '{schema}': {e}"
            ) from e
        return cls(mean=mean, std=std)


@dataclass
class UniformDistribution(ContinuousDistribution):
    """A continuous uniform distribution over [lower, upper)."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError(
                f"Lower bound ({self.lower}) must be <= upper bound"
                f" ({self.upper})"
            )

    def sample_value(self) -> float:
        return float(np.random.uniform(low=self.lower, high=self.upper))

    @classmethod
    def parse_from_str_schema(cls, schema: str) -> UniformDistribution:
        """Parse a string like "U(lower, upper)" into a UniformDistribution.

        Args:
            schema: A string in the format "U(lower, upper)" (case-insensitive prefix).

        Returns:
            A UniformDistribution instance.

        Raises:
            ValueError: If the string cannot be parsed.
        """
        args = _match_distribution_args(schema, r"[Uu]", 2)
        if args is None:
            raise ValueError(
                f"Cannot parse uniform distribution from '{schema}'. "
                "Expected format: 'U(lower, upper)'."
            )
        try:
            lower = float(args[0])
            upper = float(args[1])
        except ValueError as e:
            raise ValueError(
                f"Cannot parse numeric values from '{schema}': {e}"
            ) from e
        return cls(lower=lower, upper=upper)


@dataclass
class GammaDistribution(ContinuousDistribution):
    """A gamma distribution parameterized by shape (k) and scale (theta)."""

    shape: float
    scale: float

    def __post_init__(self) -> None:
        if self.shape <= 0:
            raise ValueError(f"Shape must be positive, got {self.shape}")
        if self.scale <= 0:
            raise ValueError(f"Scale must be positive, got {self.scale}")

    def sample_value(self) -> float:
        return float(np.random.gamma(shape=self.shape, scale=self.scale))

    @classmethod
    def parse_from_str_schema(cls, schema: str) -> GammaDistribution:
        """Parse a string like "G(shape, scale)" into a GammaDistribution.

        Args:
            schema: A string in the format "G(shape, scale)" (case-insensitive prefix).

        Returns:
            A GammaDistribution instance.

        Raises:
            ValueError: If the string cannot be parsed.
        """
        args = _match_distribution_args(schema, r"[Gg]", 2)
        if args is None:
            raise ValueError(
                f"Cannot parse gamma distribution from '{schema}'. "
                "Expected format: 'G(shape, scale)'."
            )
        try:
            shape = float(args[0])
            scale = float(args[1])
        except ValueError as e:
            raise ValueError(
                f"Cannot parse numeric values from '{schema}': {e}"
            ) from e
        return cls(shape=shape, scale=scale)


@dataclass
class LogNormalDistribution(ContinuousDistribution):
    """A log-normal distribution parameterized by the mean and std of the
    underlying normal distribution.

    Samples are drawn from exp(N(mean, std)), so the resulting values are
    always positive.
    """

    mean: float
    std: float

    def __post_init__(self) -> None:
        if self.std < 0:
            raise ValueError(
                f"Standard deviation must be non-negative, got {self.std}"
            )

    def sample_value(self) -> float:
        return float(np.random.lognormal(mean=self.mean, sigma=self.std))

    @classmethod
    def parse_from_str_schema(cls, schema: str) -> LogNormalDistribution:
        """Parse a string like "LN(mean, std)" into a LogNormalDistribution.

        Args:
            schema: A string in the format "LN(mean, std)" (case-insensitive prefix).

        Returns:
            A LogNormalDistribution instance.

        Raises:
            ValueError: If the string cannot be parsed.
        """
        args = _match_distribution_args(schema, r"[Ll][Nn]", 2)
        if args is None:
            raise ValueError(
                f"Cannot parse log-normal distribution from '{schema}'. "
                "Expected format: 'LN(mean, std)'."
            )
        try:
            mean = float(args[0])
            std = float(args[1])
        except ValueError as e:
            raise ValueError(
                f"Cannot parse numeric values from '{schema}': {e}"
            ) from e
        return cls(mean=mean, std=std)


@dataclass
class Burr12Distribution(ContinuousDistribution):
    """A Burr Type XII distribution with shape parameters ``c``, ``d`` and
    a ``scale`` parameter.

    The CDF is ``F(x) = 1 - (1 + (x/scale)**c)**(-d)`` for ``x >= 0``.
    Samples are drawn via the inverse-CDF transform on a uniform draw, so
    no scipy dependency is required:

        x = scale * ((1 - U)**(-1/d) - 1)**(1/c),  U ~ Uniform[0, 1)

    The implementation rewrites ``(1 - U)**(-1/d) - 1`` as
    ``expm1(-log1p(-U) / d)`` to avoid catastrophic cancellation when ``U``
    is near 0 (where ``(1 - U)**(-1/d)`` is close to 1 and the naive
    subtraction loses precision).
    """

    c: float
    d: float
    scale: float

    def __post_init__(self) -> None:
        if self.c <= 0:
            raise ValueError(f"c (shape) must be positive, got {self.c}")
        if self.d <= 0:
            raise ValueError(f"d (shape) must be positive, got {self.d}")
        if self.scale <= 0:
            raise ValueError(f"scale must be positive, got {self.scale}")

    def sample_value(self) -> float:
        u = np.random.uniform(low=0.0, high=1.0)
        # expm1/log1p form of (1 - u)**(-1/d) - 1; avoids cancellation when u ~ 0.
        inner = np.expm1(-np.log1p(-u) / self.d)
        return float(self.scale * inner ** (1.0 / self.c))

    @classmethod
    def parse_from_str_schema(cls, schema: str) -> Burr12Distribution:
        """Parse a string like "Burr12(c, d, scale)" into a Burr12Distribution.

        Args:
            schema: A string in the format ``Burr12(c, d, scale)``
                (case-insensitive prefix). All three parameters must be
                positive floats.

        Returns:
            A Burr12Distribution instance.

        Raises:
            ValueError: If the string cannot be parsed.
        """
        args = _match_distribution_args(schema, r"[Bb][Uu][Rr][Rr]12", 3)
        if args is None:
            raise ValueError(
                f"Cannot parse Burr12 distribution from '{schema}'. "
                "Expected format: 'Burr12(c, d, scale)'."
            )
        try:
            c = float(args[0])
            d = float(args[1])
            scale = float(args[2])
        except ValueError as e:
            raise ValueError(
                f"Cannot parse numeric values from '{schema}': {e}"
            ) from e
        return cls(c=c, d=d, scale=scale)


@dataclass
class DiscreteUniformDistribution(DiscreteDistribution):
    """A discrete uniform distribution over integers in [lower, upper].

    Uses ``np.random.randint`` internally, which samples from ``[low, high)``.
    The ``upper`` bound stored here is *inclusive*, so ``randint`` is called with
    ``high=upper + 1``.
    """

    lower: int
    upper: int

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError(
                f"Lower bound ({self.lower}) must be <= upper bound"
                f" ({self.upper})"
            )

    def sample_value(self) -> int:
        return int(np.random.randint(low=self.lower, high=self.upper + 1))

    @classmethod
    def parse_from_str_schema(cls, schema: str) -> DiscreteUniformDistribution:
        """Parse a string like "DU(lower, upper)" into a DiscreteUniformDistribution.

        Args:
            schema: A string in the format "DU(lower, upper)" (case-insensitive
                prefix). Both lower and upper must be integers.

        Returns:
            A DiscreteUniformDistribution instance.

        Raises:
            ValueError: If the string cannot be parsed or values are not integers.
        """
        args = _match_distribution_args(schema, r"[Dd][Uu]", 2)
        if args is None:
            raise ValueError(
                f"Cannot parse discrete uniform distribution from '{schema}'. "
                "Expected format: 'DU(lower, upper)'."
            )
        try:
            lower = int(args[0])
            upper = int(args[1])
        except ValueError as e:
            raise ValueError(
                f"Cannot parse integer values from '{schema}': {e}"
            ) from e
        return cls(lower=lower, upper=upper)


@dataclass
class NegativeBinomialDistribution(DiscreteDistribution):
    """A 1-shifted negative binomial distribution parameterized by n and p.

    ``np.random.negative_binomial(n, p)`` counts the number of failures before
    ``n`` successes, yielding values in {0, 1, 2, ...}. This class adds 1 to
    every sample so the support becomes {1, 2, 3, ...}, which is appropriate
    for quantities that must be at least 1 (e.g. number of turns).

    String schema: ``NB(n, p)``
    """

    n: float
    p: float

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError(
                f"n (number of successes) must be positive, got {self.n}"
            )
        if not (0.0 <= self.p <= 1.0):
            raise ValueError(
                f"p (success probability) must be in [0, 1], got {self.p}"
            )

    def sample_value(self) -> int:
        return int(np.random.negative_binomial(n=self.n, p=self.p)) + 1

    @classmethod
    def parse_from_str_schema(cls, schema: str) -> NegativeBinomialDistribution:
        """Parse a string like "NB(n, p)" into a NegativeBinomialDistribution.

        Args:
            schema: A string in the format "NB(n, p)" (case-insensitive
                prefix). ``n`` must be a positive float and ``p`` a float
                in [0, 1].

        Returns:
            A NegativeBinomialDistribution instance.

        Raises:
            ValueError: If the string cannot be parsed.
        """
        args = _match_distribution_args(schema, r"[Nn][Bb]", 2)
        if args is None:
            raise ValueError(
                f"Cannot parse negative binomial distribution from "
                f"'{schema}'. Expected format: 'NB(n, p)'."
            )
        try:
            n = float(args[0])
            p = float(args[1])
        except ValueError as e:
            raise ValueError(f"Cannot parse values from '{schema}': {e}") from e
        return cls(n=n, p=p)
