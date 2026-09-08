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

"""
Random number generation using the Philox algorithm.

This module implements a high-performance random number generator using the Philox algorithm,
which is designed for parallel computing and works efficiently on both CPU and GPU. The Philox
algorithm is a counter-based random number generator that provides high-quality random numbers
with excellent statistical properties.

The main class is Random which generates both uniform random numbers and raw 32-bit integers.
It supports:
- Seeding for reproducible sequences
- Multiple independent subsequences
- Configurable number of rounds for quality vs performance tradeoff
- Vectorized operations for efficiency
- Cross-platform support (CPU and GPU)

Example:

```mojo
from std.random.philox import Random
    rng = Random(seed=42)
    uniform_values = rng.step_uniform()  # Returns 4 random floats in [0,1)
    raw_values = rng.step()  # Returns 4 raw 32-bit integers
```
"""

from std.sys import is_little_endian

from std.math import cos, fma, log, pi, sin, sqrt

from std.memory import bitcast

from std._gpu.intrinsics import mulwide


def _mulhilow(a: UInt32, b: UInt32) -> SIMD[.uint32, 2]:
    var res = mulwide(a, b)
    return bitcast[.uint32, 2](res)


struct Random[rounds: Int = 10](Copyable):
    """A high-performance random number generator using the Philox algorithm.

    The Philox algorithm is a counter-based random number generator designed for parallel
    computing. It provides high-quality random numbers with excellent statistical properties
    and works efficiently on both CPU and GPU.

    Parameters:
        rounds: Number of mixing rounds to perform. Higher values provide better statistical
               quality at the cost of performance. Default is 10.
    """

    var _key: SIMD[.uint32, 2]
    var _counter: SIMD[.uint32, 4]

    def __init__(
        out self,
        *,
        seed: UInt64 = 0x3D30F19CD101,
        subsequence: UInt64 = 0,
        offset: UInt64 = 0,
    ):
        """Initialize the random number generator.

        Args:
            seed: Initial seed value for reproducible sequences. Default is 0.
            subsequence: Subsequence number for generating independent streams. Default is 0.
            offset: Starting offset in the sequence. Default is 0.
        """
        self._key = bitcast[.uint32, 2](seed)
        self._counter = bitcast[.uint32, 4](
            SIMD[.uint64, 2](offset, subsequence)
        )

    @always_inline
    def step(mut self) -> SIMD[.uint32, 4]:
        """Generate 4 random 32-bit unsigned integers.

        Returns:
            SIMD vector containing 4 random 32-bit unsigned integers.
        """
        comptime K_PHILOX_10 = SIMD[.uint32, 2](0x9E3779B9, 0xBB67AE85)

        var counter = self._counter
        var key = self._key

        comptime for i in range(Self.rounds - 1):
            counter = self._single_round(counter, key)
            key += K_PHILOX_10
        var res = self._single_round(counter, key)
        self._incrn(1)
        return res

    @always_inline
    def step_uniform(mut self) -> SIMD[.float32, 4]:
        """Generate 4 random floating point numbers uniformly distributed in [0,1).

        Returns:
            SIMD vector containing 4 random float32 values in range [0,1).
        """
        # maximum value such that `MAX_INT * scale < 1.0` (with float rounding)
        comptime SCALE = 4.6566127342e-10
        return (self.step() & 0x7FFFFFFF).cast[.float32]() * SCALE

    @always_inline
    def step_uniform_unbiased(mut self) -> SIMD[.float32, 4]:
        """Generate 4 uniform float32 values in (0, 1), unbiased.

        Uses all 32 raw bits of each Philox output via the conversion
        `u = (raw + 0.5) / 2^32`, which is fused into a single FMA. The
        resulting range is `(2^-33, 1 - 2^-33)` — never exactly 0 or 1.
        Compared to `step_uniform`, this uses one extra bit of randomness
        and is bounded away from 0, which removes the need for a `1.0 - u`
        guard before `log(u)` in Box-Muller transforms.

        Returns:
            SIMD vector containing 4 random float32 values in (0, 1).
        """
        comptime SCALE = Float32(2.3283064e-10)  # 1 / 2^32
        comptime SCALE_HALF = SCALE * Float32(0.5)
        return fma(
            self.step().cast[.float32](),
            SIMD[.float32, 4](SCALE),
            SIMD[.float32, 4](SCALE_HALF),
        )

    @always_inline
    def _incrn(mut self, n: Int64):
        """Increment the internal counter by n.

        Args:
            n: Amount to increment the counter by.
        """
        var hilo = bitcast[.uint32, 2](n)
        var hi, lo = (hilo[1], hilo[0]) if is_little_endian() else (
            hilo[0],
            hilo[1],
        )

        self._counter[0] += lo
        if self._counter[0] < lo:
            hi += 1
            self._counter[1] += hi
            if hi != 0 and hi <= self._counter[1]:
                return
        else:
            self._counter[1] += hi
            if hi <= self._counter[1]:
                return
        self._counter[2] += 1
        if self._counter[2]:
            return
        self._counter[3] += 1

    @always_inline
    @staticmethod
    def _single_round(
        counter: SIMD[.uint32, 4], key: SIMD[.uint32, 2]
    ) -> SIMD[.uint32, 4]:
        """Perform a single round of the Philox mixing function.

        Args:
            counter: Current counter state as 4 32-bit values.
            key: Current key state as 2 32-bit values.

        Returns:
            Mixed output as 4 32-bit values.
        """
        comptime K_PHILOX_SA = 0xD2511F53
        comptime K_PHILOX_SB = 0xCD9E8D57

        var res0 = _mulhilow(K_PHILOX_SA, counter[0])
        var res1 = _mulhilow(K_PHILOX_SB, counter[2])
        return SIMD[.uint32, 4](
            res1[1] ^ counter[1] ^ key[0],
            res1[0],
            res0[1] ^ counter[3] ^ key[1],
            res0[0],
        )


struct NormalRandom[rounds: Int = 10](Copyable):
    """A high-performance random number generator using the Box-Muller transform.

    The Box-Muller transform is a method for generating pairs of independent standard normal
    random variables. Works efficiently on both CPU and GPU.

    Parameters:
        rounds: Number of mixing rounds to perform for the underlying random uniform generator that serves as
               input to the Box-Muller transform. Higher values provide better statistical quality at the cost of
               performance. Default is 10.
    """

    var _rng: Random[Self.rounds]

    def __init__(
        out self,
        *,
        seed: UInt64 = 0x3D30F19CD101,
        subsequence: UInt64 = 0,
        offset: UInt64 = 0,
    ):
        """Initializes the normal distribution random number generator.

        Args:
            seed: Seed value for the RNG.
            subsequence: Subsequence number for the RNG.
            offset: Offset value for the RNG.
        """
        self._rng = Random[Self.rounds](
            seed=seed, subsequence=subsequence, offset=offset
        )

    def step_normal(
        mut self, mean: Float32 = 0.0, stddev: Float32 = 1.0
    ) -> SIMD[.float32, 8]:
        """Generate 8 normally distributed random numbers using Box-Muller transform.

        Args:
            mean: Mean of the normal distribution.
            stddev: Standard deviation of the normal distribution.

        Returns:
            SIMD vector containing 8 random float32 values from a normal distribution with mean `mean` and standard deviation `stddev`.
        """
        # Convert from range of [0,1) to (0,1]. This avoids having 0 and passing to log.
        var u1 = 1.0 - self._rng.step_uniform()
        var u2 = 1.0 - self._rng.step_uniform()

        var r = sqrt(-2.0 * log(u1))
        var theta = 2.0 * pi * u2
        var z0 = r * cos(theta)
        var z1 = r * sin(theta)

        # Scale and shift both sets.
        z0 = z0 * stddev + mean
        z1 = z1 * stddev + mean

        # Combine z0 and z1 into a single SIMD[DType.float32, 8]
        return z0.join(z1)

    @always_inline
    def step_normal_4(
        mut self, mean: Float32 = 0.0, stddev: Float32 = 1.0
    ) -> SIMD[.float32, 4]:
        """Generate 4 normal floats from a single Philox step.

        Pairs adjacent uint32 outputs `(raw[0], raw[1])` and
        `(raw[2], raw[3])` into Box-Muller pairs (one Philox step total,
        vs two for `step_normal`). Uses the unbiased
        `(raw + 0.5) / 2^32` uniform conversion, which is bounded away
        from 0 — no `1.0 - u` guard before `log(u)` is needed.

        Result lane order: `(sin(v_0)*s_0, cos(v_0)*s_0, sin(v_1)*s_1,
        cos(v_1)*s_1)` — sin/cos interleaved per pair, matching the
        natural output of a single Box-Muller call.

        Args:
            mean: Mean of the normal distribution.
            stddev: Standard deviation of the normal distribution.

        Returns:
            SIMD vector of 4 normal float32 values.
        """
        comptime SCALE = Float32(1.0) / Float32(UInt64(1) << 32)
        comptime SCALE_2PI = SCALE * Float32(2.0) * Float32(pi)
        comptime SCALE_HALF = SCALE * Float32(0.5)
        comptime SCALE_2PI_HALF = SCALE_2PI * Float32(0.5)

        var raw = self._rng.step().cast[.float32]()

        # Pair 0 → (raw[0], raw[1]); Pair 1 → (raw[2], raw[3]).
        # u from even lanes, v (angle) from odd lanes.
        var u_raw = SIMD[.float32, 2](raw[0], raw[2])
        var v_raw = SIMD[.float32, 2](raw[1], raw[3])

        var u = fma(
            u_raw,
            SIMD[.float32, 2](SCALE),
            SIMD[.float32, 2](SCALE_HALF),
        )
        var v = fma(
            v_raw,
            SIMD[.float32, 2](SCALE_2PI),
            SIMD[.float32, 2](SCALE_2PI_HALF),
        )
        var s = sqrt(log(u) * Float32(-2.0))
        var sin_v = sin(v)
        var cos_v = cos(v)

        var result = SIMD[.float32, 4](
            s[0] * sin_v[0],
            s[0] * cos_v[0],
            s[1] * sin_v[1],
            s[1] * cos_v[1],
        )
        return result * stddev + mean
