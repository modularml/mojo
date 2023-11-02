# ===----------------------------------------------------------------------=== #
# Copyright (c) 2023, Modular Inc. All rights reserved.
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

# This sample implements the nbody benchmarking in
# https://benchmarksgame-team.pages.debian.net/benchmarksgame/performance/nbody.html

from utils.index import StaticTuple
from math import sqrt
from benchmark import run

alias PI = 3.141592653589793
alias SOLAR_MASS = 4 * PI * PI
alias DAYS_PER_YEAR = 365.24


@register_passable("trivial")
struct Planet:
    var pos: SIMD[DType.float64, 4]
    var velocity: SIMD[DType.float64, 4]
    var mass: Float64

    fn __init__(
        pos: SIMD[DType.float64, 4],
        velocity: SIMD[DType.float64, 4],
        mass: Float64,
    ) -> Self:
        return Self {
            pos: pos,
            velocity: velocity,
            mass: mass,
        }


alias NUM_BODIES = 5


fn offset_momentum(inout bodies: StaticTuple[NUM_BODIES, Planet]):
    var p = SIMD[DType.float64, 4]()

    @unroll
    for i in range(NUM_BODIES):
        p += bodies[i].velocity * bodies[i].mass

    var body = bodies[0]
    body.velocity = -p / SOLAR_MASS

    bodies[0] = body


fn advance(inout bodies: StaticTuple[NUM_BODIES, Planet], dt: Float64):
    @unroll
    for i in range(NUM_BODIES):
        for j in range(NUM_BODIES - i - 1):
            var body_i = bodies[i]
            var body_j = bodies[j + i + 1]
            let diff = body_i.pos - body_j.pos
            let diff_sqr = (diff * diff).reduce_add()
            let mag = dt / (diff_sqr * sqrt(diff_sqr))

            body_i.velocity -= diff * body_j.mass * mag
            body_j.velocity += diff * body_i.mass * mag

            bodies[i] = body_i
            bodies[j + i + 1] = body_j

    @unroll
    for i in range(NUM_BODIES):
        var body = bodies[i]
        body.pos += dt * body.velocity
        bodies[i] = body


fn energy(bodies: StaticTuple[NUM_BODIES, Planet]) -> Float64:
    var e: Float64 = 0

    @unroll
    for i in range(NUM_BODIES):
        let body_i = bodies[i]
        e += (
            0.5
            * body_i.mass
            * ((body_i.velocity * body_i.velocity).reduce_add())
        )

        for j in range(NUM_BODIES - i - 1):
            let body_j = bodies[j + i + 1]
            let diff = body_i.pos - body_j.pos
            let distance = sqrt((diff * diff).reduce_add())
            e -= (body_i.mass * body_j.mass) / distance

    return e


fn _run():
    let Sun = Planet(
        0,
        0,
        SOLAR_MASS,
    )

    let Jupiter = Planet(
        SIMD[DType.float64, 4](
            4.84143144246472090e00,
            -1.16032004402742839e00,
            -1.03622044471123109e-01,
            0,
        ),
        SIMD[DType.float64, 4](
            1.66007664274403694e-03 * DAYS_PER_YEAR,
            7.69901118419740425e-03 * DAYS_PER_YEAR,
            -6.90460016972063023e-05 * DAYS_PER_YEAR,
            0,
        ),
        9.54791938424326609e-04 * SOLAR_MASS,
    )

    let Saturn = Planet(
        SIMD[DType.float64, 4](
            8.34336671824457987e00,
            4.12479856412430479e00,
            -4.03523417114321381e-01,
            0,
        ),
        SIMD[DType.float64, 4](
            -2.76742510726862411e-03 * DAYS_PER_YEAR,
            4.99852801234917238e-03 * DAYS_PER_YEAR,
            2.30417297573763929e-05 * DAYS_PER_YEAR,
            0,
        ),
        2.85885980666130812e-04 * SOLAR_MASS,
    )

    let Uranus = Planet(
        SIMD[DType.float64, 4](
            1.28943695621391310e01,
            -1.51111514016986312e01,
            -2.23307578892655734e-01,
            0,
        ),
        SIMD[DType.float64, 4](
            2.96460137564761618e-03 * DAYS_PER_YEAR,
            2.37847173959480950e-03 * DAYS_PER_YEAR,
            -2.96589568540237556e-05 * DAYS_PER_YEAR,
            0,
        ),
        4.36624404335156298e-05 * SOLAR_MASS,
    )

    let Neptune = Planet(
        SIMD[DType.float64, 4](
            1.53796971148509165e01,
            -2.59193146099879641e01,
            1.79258772950371181e-01,
            0,
        ),
        SIMD[DType.float64, 4](
            2.68067772490389322e-03 * DAYS_PER_YEAR,
            1.62824170038242295e-03 * DAYS_PER_YEAR,
            -9.51592254519715870e-05 * DAYS_PER_YEAR,
            0,
        ),
        5.15138902046611451e-05 * SOLAR_MASS,
    )
    var system = StaticTuple[NUM_BODIES, Planet](
        Sun, Jupiter, Saturn, Uranus, Neptune
    )
    offset_momentum(system)

    print("Energy of System:", energy(system))

    for i in range(50_000_000):
        advance(system, 0.01)

    print("Energy of System:", energy(system))


fn benchmark():
    print(run[_run]().mean())


fn main():
    print("Starting nbody...")
    _run()
