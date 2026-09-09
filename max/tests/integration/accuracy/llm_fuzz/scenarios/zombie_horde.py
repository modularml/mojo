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
Zombie Horde — all scenarios unleashed simultaneously in random order.

Grabs every registered scenario (except long-running and itself), shuffles
them into randomised waves, and fires them concurrently.  The server either
survives the horde or it doesn't.

    python3 fuzz.py --scenarios zombie_horde

Uses ``config.endurance_duration_sec`` as the wall-clock timeout (default
120 s).  Waves keep launching until time runs out or the server dies.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import TYPE_CHECKING

from scenarios import (
    BaseScenario,
    ScenarioResult,
    Verdict,
    get_all_scenarios,
    register_scenario,
)

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig

# Scenarios excluded from the horde:
#   - endurance_soak: designed for multi-minute runs, would dominate wall-clock
#   - connection_exhaustion: raw sockets interfere with concurrent HTTP work
#   - zombie_horde: prevent infinite recursion
_EXCLUDE = frozenset(
    {"endurance_soak", "connection_exhaustion", "zombie_horde"}
)

# How many scenarios run in parallel per wave.
_WAVE_SIZE = 4

# Default timeout if config.endurance_duration_sec is not set.
_DEFAULT_TIMEOUT_SEC = 120.0

# ---------------------------------------------------------------------------
# ASCII banners
# ---------------------------------------------------------------------------

_BANNER_DEAD = r"""

                                  _.---._
                              .-'         '-.
                             /               \
                            |   R . I . P .   |
                            |                 |
                            |    S E R V E R  |
                            |                 |
                            |{date_line}|
                            |                 |
                            |  "It served us  |
                            |     well..."    |
                           _|_________________|_
                     ,'`. |_____________________| ,'`.
                    `'`'`/   _             _     \`'`'`
                 ,'`.   /   | |    ___    | |     \  ,'`.
                `'`'`__/    | |   /   \   | |     _\__'`'`
          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
           ~~      ~~       ~~      ~~       ~~      ~~

                ,                                      (()))
            _,-""-._               ,                  /|x x|
          ,"        ".          ,-""-._              /\( - )
         /    ,-,  ,"\        ,"   _   ".    ___.-._/\/
        "    /   \ | o|      /  ,-/ \,  "|  /=`_'-'-'/  !!
        \    `-o-"  `-',    "  /  o  \ o|   |-{-_-_-}     !
         `,   _.--'`'--`    \   `-o-" `-',  (-{-_-_-}    !
           `--`---'          `,  _.'`'--`    \{_-_-_}   !
             ,' '      _      `-`---'         }-_-_-}
           ./ ,  `,    _)       ,' '    _     {-_|-_}
           / /     \  _ \     ./ , `,   _)    {-_|_-}
          (_)))_ _," \___/    / /    \  _ \   {_-|-_}
             _))))_,         (_)))_ _," \___/ {_-|-_}  ZOT
    --------(_,-._)))---------- _))))_,-------%%@ @%%----------
                              (_,-._)))

"""

_BANNER_SURVIVED = r"""

                              .    |     .
                               \   |    /
                           `.   \  '   /   .'
                             `. .-*""*-. .'
                         "*-._ /.*"  "*.\ _.-*"
                              :    ;  ____
                         ----':    ..    ;
                         _.-*" \ `.__.' / "*-._
                             .' `-.__.-' `.
                           .'   /   .  \   `.
                               /    |   \
                              '     |    `


                           __ _.--..--._ _
                        .-' _/   _/\_   \_'-.
                       |__ /   _/\__/\_   \__|
                          |___/\_\__/  \___|
                                 \__/
                                 \__/
                                  \__/
                                   \__/
                                ____\__/___
                          . - '             ' -.
                         /    SERVER IS ALIVE    \
                   ~~~~~~~  ~~~~~ ~~~~~  ~~~ ~~~  ~~~~~
                     ~~~   ~~~~~   ~~~~   ~~ ~  ~ ~ ~~

    ================================================================
    ||                                                            ||
    ||    _____                                                   ||
    ||   / ____|                                                  ||
    ||  | (___   ___ _ ____   _____ _ __                          ||
    ||   \___ \ / _ \ '__\ \ / / _ \ '__|                         ||
    ||   ____) |  __/ |   \ V /  __/ |                            ||
    ||  |_____/ \___|_|    \_/ \___|_|                            ||
    ||                                                            ||
    ||   _____                  _                _   _            ||
    ||  / ____|                (_)              | | | |           ||
    ||  | (___  _   _ _ ____   _____   _____  __| | | |           ||
    ||   \___ \| | | | '__\ \ / / \ \ / / _ \/ _` | | |           ||
    ||   ____) | |_| | |   \ V /   \ V /  __/ (_| | |_|           ||
    ||  |_____/ \__,_|_|    \_/     \_/ \___|\__,_| (_)           ||
    ||                                                            ||
    ||      The server stands. The horde retreats at dawn.        ||
    ||                                                            ||
    ================================================================
"""


@register_scenario
class ZombieHorde(BaseScenario):
    name = "zombie_horde"
    description = (
        "All scenarios unleashed simultaneously in random order "
        "— total server survival test"
    )
    tags = ["chaos", "crash", "load"]

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        timeout_sec = getattr(
            config, "endurance_duration_sec", _DEFAULT_TIMEOUT_SEC
        )
        deadline = time.perf_counter() + timeout_sec

        # Collect every scenario except exclusions and those needing a validator
        # that isn't available.
        registry = get_all_scenarios()
        has_validator = getattr(config, "validator", None) is not None
        candidates = [
            cls
            for name, cls in registry.items()
            if name not in _EXCLUDE
            and (not cls.requires_validator or has_validator)
            and cls.model_filter is None  # skip model-specific scenarios
        ]

        if not candidates:
            results.append(
                self.make_result(
                    self.name,
                    "no_scenarios_found",
                    Verdict.ERROR,
                    detail="No candidate scenarios found for the horde",
                )
            )
            return results

        # Run waves in cycles until the deadline.
        horde_t0 = time.perf_counter()
        server_alive = True
        wave_num = 0
        cycle_num = 0

        while time.perf_counter() < deadline and server_alive:
            cycle_num += 1
            random.shuffle(candidates)

            waves = [
                candidates[i : i + _WAVE_SIZE]
                for i in range(0, len(candidates), _WAVE_SIZE)
            ]

            for wave in waves:
                wave_num += 1

                # Check timeout before launching the next wave.
                remaining_sec = deadline - time.perf_counter()
                if remaining_sec <= 0:
                    elapsed_so_far = (time.perf_counter() - horde_t0) * 1000
                    results.append(
                        self.make_result(
                            self.name,
                            "horde_timeout",
                            Verdict.PASS,
                            elapsed_ms=elapsed_so_far,
                            detail=(
                                f"Timeout reached ({timeout_sec:.0f}s) after "
                                f"{wave_num - 1} waves ({cycle_num} cycles) — "
                                f"server still standing"
                            ),
                        )
                    )
                    break

                wave_names = [cls.name for cls in wave]
                wave_t0 = time.perf_counter()

                # Launch every scenario in this wave concurrently.
                tasks = [
                    asyncio.create_task(
                        self._run_one(cls(), client, config),
                        name=cls.name,
                    )
                    for cls in wave
                ]

                # Enforce the remaining wall-clock budget for this wave.
                try:
                    wave_outcomes = await asyncio.wait_for(
                        asyncio.gather(*tasks),
                        timeout=remaining_sec,
                    )
                except asyncio.TimeoutError:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

                    wave_ms = (time.perf_counter() - wave_t0) * 1000
                    elapsed_so_far = (time.perf_counter() - horde_t0) * 1000

                    for cls in wave:
                        results.append(
                            self.make_result(
                                self.name,
                                f"{cls.name}__horde_timeout",
                                Verdict.ERROR,
                                elapsed_ms=wave_ms,
                                detail=(
                                    f"Scenario {cls.name!r} did not complete "
                                    f"before the horde timeout "
                                    f"({timeout_sec:.0f}s) elapsed"
                                ),
                            )
                        )

                    results.append(
                        self.make_result(
                            self.name,
                            "horde_timeout",
                            Verdict.PASS,
                            elapsed_ms=elapsed_so_far,
                            detail=(
                                f"Timeout reached ({timeout_sec:.0f}s) during "
                                f"wave {wave_num} (cycle {cycle_num}) — "
                                f"server still standing"
                            ),
                        )
                    )
                    break

                wave_ms = (time.perf_counter() - wave_t0) * 1000

                # Collect results from each scenario.
                wave_results = []
                for cls, (scenario_results, error) in zip(
                    wave, wave_outcomes, strict=False
                ):
                    if error:
                        wave_results.append(
                            self.make_result(
                                self.name,
                                f"{cls.name}__scenario_crashed",
                                Verdict.ERROR,
                                error=error,
                                detail=f"Scenario {cls.name!r} threw during the horde",
                            )
                        )
                    else:
                        for r in scenario_results:
                            wave_results.append(
                                self.make_result(
                                    self.name,
                                    f"{r.scenario_name}__{r.test_name}",
                                    r.verdict,
                                    status_code=r.status_code,
                                    elapsed_ms=r.elapsed_ms,
                                    detail=r.detail,
                                    response_body=r.response_body,
                                    error=r.error,
                                )
                            )
                results.extend(wave_results)

                # Wave summary.
                wave_passes = sum(
                    1 for r in wave_results if r.verdict == Verdict.PASS
                )
                wave_fails = sum(
                    1 for r in wave_results if r.verdict == Verdict.FAIL
                )

                # Health check between waves.
                health = await client.health_check()
                survived = health.status == 200 and not health.error

                results.append(
                    self.make_result(
                        self.name,
                        f"wave_{wave_num}_cycle_{cycle_num}_health_check",
                        Verdict.PASS if survived else Verdict.FAIL,
                        status_code=health.status,
                        elapsed_ms=wave_ms,
                        detail=(
                            f"Wave {wave_num} (cycle {cycle_num}) "
                            f"[{', '.join(wave_names)}] "
                            f"— {wave_passes} passed, {wave_fails} failed, "
                            f"{wave_ms:.0f}ms"
                            + (
                                ""
                                if survived
                                else " — SERVER DOWN, aborting horde"
                            )
                        ),
                    )
                )

                if not survived:
                    server_alive = False
                    results.append(
                        self.make_result(
                            self.name,
                            "horde_aborted",
                            Verdict.FAIL,
                            detail=(
                                f"Server died after wave {wave_num} "
                                f"(cycle {cycle_num}) — horde wins"
                            ),
                        )
                    )
                    break
            else:
                # Inner for-loop completed without break — continue cycling.
                continue
            break  # Inner loop broke (timeout/death) — exit outer while.

        # Final summary result.
        total_ms = (time.perf_counter() - horde_t0) * 1000
        total_pass = sum(1 for r in results if r.verdict == Verdict.PASS)
        total_fail = sum(1 for r in results if r.verdict == Verdict.FAIL)
        total_err = sum(1 for r in results if r.verdict == Verdict.ERROR)

        if total_fail == 0 and total_err == 0:
            summary_verdict = Verdict.PASS
            summary_detail = (
                f"Server survived the entire horde: "
                f"{total_pass} passed across {wave_num} waves "
                f"({cycle_num} cycles) in {total_ms:.0f}ms"
            )
        else:
            summary_verdict = (
                Verdict.FAIL if total_fail > 0 else Verdict.INTERESTING
            )
            summary_detail = (
                f"Horde results: {total_pass} passed, {total_fail} failed, "
                f"{total_err} errors across {wave_num} waves "
                f"({cycle_num} cycles) in {total_ms:.0f}ms"
            )

        # Print the appropriate ASCII banner.
        import datetime

        current_date = datetime.date.today().isoformat()
        if server_alive:
            print(_BANNER_SURVIVED)
        else:
            print(_BANNER_DEAD.replace("{date_line}", current_date.center(17)))

        results.append(
            self.make_result(
                self.name,
                "horde_final_verdict",
                summary_verdict,
                elapsed_ms=total_ms,
                detail=summary_detail,
            )
        )

        return results

    @staticmethod
    async def _run_one(
        scenario: BaseScenario,
        client: FuzzClient,
        config: RunConfig,
    ) -> tuple[list[ScenarioResult], str | None]:
        """Run a single scenario, catching any exception."""
        try:
            results = await scenario.run(client, config)
            return results, None
        except Exception as exc:
            return [], f"{type(exc).__name__}: {exc}"
