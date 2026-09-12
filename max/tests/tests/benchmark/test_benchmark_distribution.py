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

"""Unit tests for benchmark_shared.distribution module."""

import numpy as np
import pytest
from max.benchmark.benchmark_shared.datasets.distribution import (
    BaseDistribution,
    Burr12Distribution,
    CategoricalDistribution,
    ConstantDistribution,
    ContinuousDistribution,
    DiscreteDistribution,
    DiscreteUniformDistribution,
    GammaDistribution,
    LogNormalDistribution,
    NegativeBinomialDistribution,
    NormalDistribution,
    UniformDistribution,
)

# ---------------------------------------------------------------------------
# ConstantDistribution
# ---------------------------------------------------------------------------


def test_float_returns_constant() -> None:
    dist = BaseDistribution.from_distribution_parameter(42.0)
    assert isinstance(dist, ConstantDistribution)
    assert dist.value == 42.0


def test_str_float_returns_constant() -> None:
    dist = BaseDistribution.from_distribution_parameter("42.0")
    assert isinstance(dist, ConstantDistribution)
    assert dist.value == 42.0


def test_returns_same_value() -> None:
    dist = ConstantDistribution(value=42.0)
    assert dist.sample_value() == 42.0
    assert dist.sample_value() == 42.0


# ---------------------------------------------------------------------------
# CategoricalDistribution
# ---------------------------------------------------------------------------


def test_categorical_empty_values_raises() -> None:
    with pytest.raises(ValueError, match="at least one value"):
        CategoricalDistribution(values=[], weights=[])


def test_categorical_mismatched_lengths_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        CategoricalDistribution(values=[1.0, 2.0], weights=[1.0])


def test_categorical_negative_weight_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        CategoricalDistribution(values=[1.0, 2.0], weights=[1.0, -1.0])


def test_categorical_nonfinite_weight_raises() -> None:
    """nan passes both `w < 0` and `sum(w) <= 0`, so it must be caught here or
    it surfaces later inside np.random.choice."""
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            CategoricalDistribution(values=[1.0, 2.0], weights=[1.0, bad])


def test_categorical_nonfinite_value_raises() -> None:
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            CategoricalDistribution(values=[1.0, bad], weights=[1.0, 1.0])


def test_categorical_all_zero_weights_raises() -> None:
    with pytest.raises(ValueError, match="positive value"):
        CategoricalDistribution(values=[1.0, 2.0], weights=[0.0, 0.0])


def test_categorical_weights_summing_to_inf_still_sample() -> None:
    """Each weight is finite, but their sum overflows; dividing by that sum
    would send every probability to 0 and np.random.choice would reject it."""
    np.random.seed(0)
    dist = CategoricalDistribution(values=[1.0, 2.0], weights=[1e308, 1e308])
    samples = [dist.sample_value() for _ in range(200)]
    assert set(samples) == {1.0, 2.0}
    assert 0.3 < samples.count(1.0) / len(samples) < 0.7


def test_categorical_string_with_weights() -> None:
    dist = BaseDistribution.from_distribution_parameter(
        "Cat(1024:0.7, 512:0.2, 2048:0.1)"
    )
    assert isinstance(dist, CategoricalDistribution)
    assert dist.values == [1024.0, 512.0, 2048.0]
    assert dist.weights == [0.7, 0.2, 0.1]


def test_categorical_string_lowercase() -> None:
    dist = BaseDistribution.from_distribution_parameter("cat(1, 2, 3)")
    assert isinstance(dist, CategoricalDistribution)


def test_categorical_string_uniform_weights_default_to_one() -> None:
    dist = BaseDistribution.from_distribution_parameter("Cat(512, 1024)")
    assert isinstance(dist, CategoricalDistribution)
    assert dist.weights == [1.0, 1.0]


def test_categorical_string_single_value() -> None:
    dist = BaseDistribution.from_distribution_parameter("Cat(512)")
    assert isinstance(dist, CategoricalDistribution)
    assert dist.sample_value() == 512.0


def test_categorical_string_empty_entry_raises() -> None:
    with pytest.raises(ValueError, match="Cat"):
        BaseDistribution.from_distribution_parameter("Cat()")


def test_categorical_string_non_numeric_raises() -> None:
    with pytest.raises(ValueError, match="Cannot parse numeric"):
        BaseDistribution.from_distribution_parameter("Cat(1:1, 2:notaweight)")


def test_categorical_samples_only_given_values() -> None:
    dist = CategoricalDistribution(values=[1.0, 2.0, 3.0], weights=[1, 1, 1])
    samples = {dist.sample_value() for _ in range(200)}
    assert samples <= {1.0, 2.0, 3.0}


def test_categorical_sample_frequencies_match_weights() -> None:
    np.random.seed(0)
    dist = CategoricalDistribution(values=[10.0, 20.0], weights=[0.9, 0.1])
    samples = [dist.sample_value() for _ in range(5000)]
    frequency_of_10 = samples.count(10.0) / len(samples)
    assert abs(frequency_of_10 - 0.9) < 0.03


def test_categorical_weights_need_not_sum_to_one() -> None:
    np.random.seed(0)
    dist = CategoricalDistribution(values=[1.0, 2.0], weights=[9.0, 1.0])
    samples = [dist.sample_value() for _ in range(5000)]
    frequency_of_1 = samples.count(1.0) / len(samples)
    assert abs(frequency_of_1 - 0.9) < 0.03


def test_categorical_is_not_continuous_or_discrete() -> None:
    assert not issubclass(CategoricalDistribution, ContinuousDistribution)
    assert not issubclass(CategoricalDistribution, DiscreteDistribution)


# ---------------------------------------------------------------------------
# NormalDistribution
# ---------------------------------------------------------------------------


def test_negative_std_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        NormalDistribution(mean=100.0, std=-1.0)


def test_normal_string() -> None:
    dist = BaseDistribution.from_distribution_parameter("N(100, 1)")
    assert isinstance(dist, NormalDistribution)
    assert dist.mean == 100.0
    assert dist.std == 1.0


def test_samples_near_mean_std() -> None:
    np.random.seed(0)
    dist = NormalDistribution(mean=100.0, std=1.0)
    samples = [dist.sample_value() for _ in range(500)]
    assert abs(np.mean(samples) - 100.0) < 5.0
    assert abs(np.std(samples) - 1.0) < 0.05


# ---------------------------------------------------------------------------
# UniformDistribution
# ---------------------------------------------------------------------------


def test_lower_greater_than_upper_raises() -> None:
    with pytest.raises(ValueError, match="Lower bound"):
        UniformDistribution(lower=10.0, upper=1.0)


def test_equal_bounds_returns_that_value() -> None:
    dist = UniformDistribution(lower=3.0, upper=3.0)
    assert dist.sample_value() == 3.0


def test_uniform_string() -> None:
    dist = BaseDistribution.from_distribution_parameter("U(1, 10)")
    assert isinstance(dist, UniformDistribution)
    assert dist.lower == 1.0
    assert dist.upper == 10.0


def test_get_value_within_bounds() -> None:
    dist = UniformDistribution(lower=1.0, upper=10.0)
    for _ in range(100):
        val = dist.sample_value()
        assert 1.0 <= val <= 10.0


def test_uniform_is_continuous() -> None:
    dist = UniformDistribution(lower=0.0, upper=1.0)
    assert isinstance(dist, ContinuousDistribution)
    assert isinstance(dist, BaseDistribution)


# ---------------------------------------------------------------------------
# DiscreteUniformDistribution
# ---------------------------------------------------------------------------


def test_discrete_uniform_lower_greater_than_upper_raises() -> None:
    with pytest.raises(ValueError, match="Lower bound"):
        DiscreteUniformDistribution(lower=10, upper=1)


def test_discrete_uniform_equal_bounds() -> None:
    dist = DiscreteUniformDistribution(lower=5, upper=5)
    assert dist.sample_value() == 5


def test_discrete_uniform_string() -> None:
    dist = BaseDistribution.from_distribution_parameter("DU(1, 10)")
    assert isinstance(dist, DiscreteUniformDistribution)
    assert dist.lower == 1
    assert dist.upper == 10


def test_discrete_uniform_returns_ints_within_bounds() -> None:
    dist = DiscreteUniformDistribution(lower=1, upper=10)
    for _ in range(200):
        val = dist.sample_value()
        assert isinstance(val, int)
        assert 1 <= val <= 10


def test_discrete_uniform_is_discrete() -> None:
    dist = DiscreteUniformDistribution(lower=0, upper=5)
    assert isinstance(dist, DiscreteDistribution)
    assert isinstance(dist, BaseDistribution)


def test_discrete_uniform_non_int_raises() -> None:
    with pytest.raises(ValueError, match="Cannot parse integer"):
        DiscreteUniformDistribution.parse_from_str_schema("DU(1.5, 3.7)")


# ---------------------------------------------------------------------------
# NegativeBinomialDistribution
# ---------------------------------------------------------------------------


def test_nb_non_positive_n_raises() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        NegativeBinomialDistribution(n=0, p=0.5)
    with pytest.raises(ValueError, match="must be positive"):
        NegativeBinomialDistribution(n=-1, p=0.5)


def test_nb_string() -> None:
    dist = BaseDistribution.from_distribution_parameter("NB(3, 0.4)")
    assert isinstance(dist, NegativeBinomialDistribution)
    assert dist.n == 3
    assert dist.p == 0.4


def test_nb_samples_always_at_least_one() -> None:
    dist = NegativeBinomialDistribution(n=1, p=0.99)
    for _ in range(500):
        val = dist.sample_value()
        assert isinstance(val, int)
        assert val >= 1


def test_nb_samples_near_expected_mean() -> None:
    np.random.seed(0)
    n, p = 3, 0.4
    dist = NegativeBinomialDistribution(n=n, p=p)
    # Shifted NB mean = 1 + n*(1-p)/p
    expected_mean = 1 + n * (1 - p) / p
    samples = [dist.sample_value() for _ in range(2000)]
    assert abs(np.mean(samples) - expected_mean) < 0.5


def test_nb_is_discrete() -> None:
    dist = NegativeBinomialDistribution(n=1, p=0.5)
    assert isinstance(dist, DiscreteDistribution)
    assert isinstance(dist, BaseDistribution)


# ---------------------------------------------------------------------------
# GammaDistribution
# ---------------------------------------------------------------------------


def test_non_positive_shape_raises() -> None:
    with pytest.raises(ValueError, match="Shape must be positive"):
        GammaDistribution(shape=0.0, scale=1.0)
    with pytest.raises(ValueError, match="Shape must be positive"):
        GammaDistribution(shape=-1.0, scale=1.0)


def test_non_positive_scale_raises() -> None:
    with pytest.raises(ValueError, match="Scale must be positive"):
        GammaDistribution(shape=1.0, scale=0.0)
    with pytest.raises(ValueError, match="Scale must be positive"):
        GammaDistribution(shape=1.0, scale=-1.0)


def test_gamma_string() -> None:
    dist = GammaDistribution.parse_from_str_schema("G(2, 100)")
    assert dist.shape == 2.0
    assert dist.scale == 100.0


def test_get_value_near_mean() -> None:
    dist = GammaDistribution(shape=2.0, scale=100.0)
    # Gamma(k=2, theta=100) has mean = k*theta = 200
    values = [dist.sample_value() for _ in range(500)]
    mean = float(np.mean(values))
    assert 190 < mean < 210, f"Expected mean ~200, got {mean}"


# ---------------------------------------------------------------------------
# LogNormalDistribution
# ---------------------------------------------------------------------------


def test_lognormal_negative_std_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        LogNormalDistribution(mean=0.0, std=-1.0)


def test_lognormal_string() -> None:
    dist = BaseDistribution.from_distribution_parameter("LN(1, 0.5)")
    assert isinstance(dist, LogNormalDistribution)
    assert dist.mean == 1.0
    assert dist.std == 0.5


def test_lognormal_samples_positive() -> None:
    dist = LogNormalDistribution(mean=0.0, std=1.0)
    for _ in range(100):
        assert dist.sample_value() > 0.0


def test_lognormal_samples_near_expected_mean() -> None:
    np.random.seed(0)
    dist = LogNormalDistribution(mean=1.0, std=0.5)
    # LogNormal(mu=1, sigma=0.5) has mean = exp(mu + sigma^2/2) ≈ 3.08
    expected_mean = np.exp(1.0 + 0.5**2 / 2)
    samples = [dist.sample_value() for _ in range(1000)]
    assert abs(np.mean(samples) - expected_mean) < 0.5


# ---------------------------------------------------------------------------
# BaseDistribution.from_distribution_parameter
# ---------------------------------------------------------------------------


def test_unrecognized_string_raises() -> None:
    with pytest.raises(ValueError, match="Unrecognized distribution"):
        BaseDistribution.from_distribution_parameter("1,2,3,4,5")
    with pytest.raises(ValueError, match="Unrecognized distribution"):
        BaseDistribution.from_distribution_parameter("X(1, 2)")


# ---------------------------------------------------------------------------
# Hierarchy checks
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Burr12Distribution
# ---------------------------------------------------------------------------


def test_burr12_non_positive_c_raises() -> None:
    with pytest.raises(ValueError, match="c"):
        Burr12Distribution(c=0.0, d=1.0, scale=1.0)


def test_burr12_non_positive_d_raises() -> None:
    with pytest.raises(ValueError, match="d"):
        Burr12Distribution(c=1.0, d=-0.5, scale=1.0)


def test_burr12_non_positive_scale_raises() -> None:
    with pytest.raises(ValueError, match="scale"):
        Burr12Distribution(c=1.0, d=1.0, scale=0.0)


def test_burr12_string() -> None:
    dist = BaseDistribution.from_distribution_parameter(
        "Burr12(2.389, 0.569, 214.8)"
    )
    assert isinstance(dist, Burr12Distribution)
    assert dist.c == pytest.approx(2.389)
    assert dist.d == pytest.approx(0.569)
    assert dist.scale == pytest.approx(214.8)


def test_burr12_string_lowercase() -> None:
    dist = BaseDistribution.from_distribution_parameter("burr12(2, 1, 5)")
    assert isinstance(dist, Burr12Distribution)


def test_burr12_string_missing_param_raises() -> None:
    with pytest.raises(ValueError, match="Burr12"):
        BaseDistribution.from_distribution_parameter("Burr12(1, 2)")


def test_burr12_sample_is_stable_for_tiny_u(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # For u below ~2^-53, (1 - u) rounds to 1.0 and the naive form
    # (1 - u)**(-1/d) - 1.0 collapses to 0.0. The expm1/log1p form preserves
    # precision: the true value is ~u/d for small u.
    tiny_u = 1e-18
    d = 0.569
    monkeypatch.setattr(np.random, "uniform", lambda low=0.0, high=1.0: tiny_u)

    # Sanity check: the naive form is broken at this u.
    assert (1.0 - tiny_u) ** (-1.0 / d) - 1.0 == 0.0

    dist = Burr12Distribution(c=2.389, d=d, scale=214.8)
    sample = dist.sample_value()
    assert sample > 0.0
    # Expected: scale * (u/d)**(1/c). Within 1% of the analytic small-u limit.
    expected = dist.scale * (tiny_u / d) ** (1.0 / dist.c)
    assert abs(sample - expected) / expected < 0.01


def test_burr12_samples_positive_and_match_median() -> None:
    np.random.seed(0)
    c, d, scale = 2.389, 0.569, 214.8
    dist = Burr12Distribution(c=c, d=d, scale=scale)
    samples = np.array([dist.sample_value() for _ in range(20_000)])
    assert (samples > 0).all()
    expected_median = scale * (2.0 ** (1.0 / d) - 1.0) ** (1.0 / c)
    assert abs(np.median(samples) - expected_median) / expected_median < 0.05


def test_continuous_distributions_are_continuous() -> None:
    for dist_cls in (
        NormalDistribution,
        UniformDistribution,
        GammaDistribution,
        LogNormalDistribution,
        Burr12Distribution,
    ):
        assert issubclass(dist_cls, ContinuousDistribution)


def test_discrete_distributions_are_discrete() -> None:
    for dist_cls in (
        DiscreteUniformDistribution,
        NegativeBinomialDistribution,
    ):
        assert issubclass(dist_cls, DiscreteDistribution)
