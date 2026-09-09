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
"""Pseudorandom number generation with uniform and normal distributions.

The `random` package provides pseudorandom number generation for simulations,
games, and statistical applications. It offers both a convenient global PRNG
state accessed through package-level functions and explicit generator types for
reproducible sequences. The package supports uniform and normal (Gaussian)
distributions for various numeric types.

Use this package for Monte Carlo simulations, stochastic algorithms, random
sampling, or testing with randomized inputs. This package's functionality
is not cryptographically
secure and should not be used for security-sensitive applications.
"""

from .random import (
    rand,
    randint,
    randn,
    randn_float64,
    random_float64,
    random_si64,
    random_ui64,
    seed,
    shuffle,
)
from .philox import Random, NormalRandom
