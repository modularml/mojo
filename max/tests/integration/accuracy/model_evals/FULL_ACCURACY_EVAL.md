# Full accuracy eval suite

One `workflow_dispatch` of
[`fullAccuracyEval.yaml`](../../../../../.github/workflows/fullAccuracyEval.yaml)
runs the complete accuracy eval suite against a build and produces one
consolidated `results.json` plus a score table in the run's step summary.
MiniMax M3 is the first profile.

## Trigger a run

```bash
gh workflow run fullAccuracyEval.yaml \
  --ref <branch-or-sha-to-qualify> \
  -f model=MiniMaxAI/MiniMax-M3-MXFP8 \
  -f backend=max-serve \
  -f runner=modrunner-b200-8x
```

The build under test is the git ref you dispatch on: each eval job builds
and serves from that ref on its own runner.

## Comparing two runs (branch or backend A/B)

Dispatch twice with only the variable under test changed (the `--ref` for a
branch-vs-branch comparison such as a release candidate against the current
approved release, or `backend`/`runner` for MAX vs vLLM), download both
`consolidated-results` artifacts, and re-run the collector with the baseline
leg as `--compare-to`:

```bash
gh run download <max-run-id> -n consolidated-results -D /tmp/ab/max
gh run download <vllm-run-id> -D /tmp/ab/vllm-artifacts
python3 max/tests/integration/accuracy/model_evals/collect_scores.py \
  --artifacts-dir /tmp/ab/vllm-artifacts \
  --out /tmp/ab/vllm-vs-max.json \
  --compare-to /tmp/ab/max/results.json
```

This adds a per-benchmark delta table and a `comparison` section to the
output; benchmarks present in only one leg show `n/a` rather than being
dropped.

## What runs

One job per dataset, all in parallel, each on its own runner:

| Toggle                        | Jobs                                                |
|-------------------------------|-----------------------------------------------------|
| run_text                      | aime25, gpqa, hle, aa-lcr, aa-omniscience           |
| run_scicode                   | scicode                                             |
| run_multimodal                | mmmu-pro, omnidocbench, mmmu-video                  |
| run_agent_class (default off) | agent-class (SWE-bench + GAIA health check)         |
| run_provider_verifier         | provider-verifier (format-correctness health check) |

The wall clock is the slowest single dataset rather than a serial group; the
cost is one server startup per dataset, and when the runner pool is smaller
than the fan-out, jobs queue and effectively serialize. Agent-class is
opt-in (33h+), and even when enabled the consolidate job does not wait for
it; its artifacts land in the same run when it finishes.

## Read the results

- The **consolidate** job's step summary shows the per-benchmark score table
  and a group status table.
- The `consolidated-results` artifact holds `results.json`: run metadata,
  one row per benchmark (score, errors, totals, token stats, and the raw
  `score.json` payload), and a `missing` list for expected benchmarks that
  produced no score.
- Per-dataset artifacts keep the full `results.jsonl` + `score.json`
  detail, prefixed by the job that produced them (`aime25-`, `scicode-`,
  `mmmu-video-`, ...).

## Re-run a failed piece

Re-run the failed job (GitHub allows re-running a single job of a run),
or dispatch the suite again with only that dataset's toggle enabled, then
re-run the consolidate job, or run the collector locally over downloaded
artifacts:

```bash
gh run download <run-id> -D /tmp/suite-artifacts
python3 max/tests/integration/accuracy/model_evals/collect_scores.py \
  --artifacts-dir /tmp/suite-artifacts --out /tmp/results.json
```

For failed individual rows within a text eval, the eval modules support
re-running exact dataset rows; see `--row-ids` on the de-embedded evals.

## Notes

- The multimodal and agent-class benchmarks are not yet in the
  collector's `--expect` list (their score directory names need one
  validation run to pin down); their scores still appear in `results.json`
  whenever present. MMMU-Video previously wrote only `summary.json` and so
  never appeared in the collector's output at all (it only globs
  `score.json`); it now writes both.
- The provider verifier's unexpected-failure list (format-correctness tests
  that failed and are not on its `KNOWN_FAILURES` allowlist) and pass@10
  batch-verification metrics are folded into its own `score.json` and
  rendered as extra sections below the main score table.
- Verdicts (PASS/FAIL/CONSULT vs baselines) attach to the consolidated
  output once the report-only eval gate lands.
