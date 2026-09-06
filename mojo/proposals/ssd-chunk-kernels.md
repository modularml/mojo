# Proposal: Mamba2 SSD chunk-scan prefill kernels

- **Status:** Draft (proposal)
- **Author:** Evan Owen
- **Area:** MAX AI kernels (`max/kernels/src/state_space/`)
- **Tracking:** Phase 2 of the Linear-Attention / SSM roadmap

## Summary

Add the State Space Duality (SSD) **chunk-scan prefill** algorithm for Mamba2
as a set of Mojo kernels under `max/kernels/src/state_space/`, plus a graph op
`ssd_chunk_scan_combined`. This is the only genuinely new kernel work for
Mamba2 — the decode path and all preprocessing already exist (see *Reuse*).
Because state-space kernels are not on the
[`max/kernels/CONTRIBUTING.md`](../../max/kernels/CONTRIBUTING.md) "changes we
accept" list, this proposal seeks a kernels-lead go-ahead before
implementation PRs.

## Motivation

Mamba2 ([Transformers are SSMs](https://arxiv.org/abs/2405.21060)) replaces
Mamba1's diagonal selective scan with a scalar-times-identity `A` per head,
which makes the recurrence equivalent to a structured form of linear attention.
That equivalence lets prefill be computed **chunk-wise with dense matmuls** that
map onto tensor cores, instead of a sequential scan. Mamba2 reference
implementations report 2–8× faster training/prefill from this. MAX currently
has no chunk-SSD kernel: the existing `ssd_combined` (`selective_scan.mojo`) is,
despite its name, a Mamba1 *sequential* scan fused with norm+residual.

## Reuse (no new work)

Confirmed present in `max/kernels/src/state_space/`:

| Capability                     | Symbol / file                                                                  | Mamba2 role                                                                                                    |
|--------------------------------|--------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| Causal conv1d (prefill+update) | `causal_conv1d.mojo`                                                           | identical, reused                                                                                              |
| Varlen conv1d                  | `varlen_causal_conv1d.mojo`                                                    | identical, reused                                                                                              |
| Fused RMSNorm + residual       | `rms_norm_fused_residual.mojo`                                                 | identical, reused                                                                                              |
| Single-token update (scalar A) | `selective_scan_update` (`selective_scan.mojo`)                                | **decode path**, reused as-is                                                                                  |
| Fused conv+scan, multi-head    | `mamba_split_conv1d_scan_combined_{cpu,gpu}` (`selective_scan.mojo:2260,2766`) | usable Mamba2 **decode** path (already has `nheads`/`headdim`/`ngroups`, scalar A/head; sequential internally) |

So **only the prefill chunk algorithm is new.**

## Proposed design

New file `max/kernels/src/state_space/ssd_chunk.mojo` (keep the 3.2k-line
`selective_scan.mojo` untouched). The reference is `ssd_minimal_discrete`
(Listing 1 of the paper). Note the discretization contract: the fused op takes
raw `x, dt, A, B, C` and internally applies `x ← x·dt`, `A ← A·dt` — i.e. the
op equals `ssd_minimal_discrete(x·dt, A·dt, B, C, chunk_size)`. Decompose into
four kernels (each a separate PR), all operating on chunked
tensors `(batch, n_chunks, chunk_len, n_heads, …)`:

1. **Intra-chunk (diagonal) output.** Within-chunk token×token contribution.
   `L = exp(segsum(A))`;
   `Y_diag = einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)`.
   Dense GEMM over `chunk_len`; the tensor-core hot path.
2. **Per-chunk state.** Reduce each chunk to its end-state.
   `decay = exp(A_cumsum[...,-1:] − A_cumsum)`;
   `states = einsum("bclhn,bhcl,bclhp->bchpn", B, decay, X)`.
3. **Inter-chunk recurrence.** Sequential scan over the per-chunk states
   (length = `n_chunks` ≪ `seqlen`).
   `decay_chunk = exp(segsum(pad(A_cumsum[...,−1])))`;
   `new_states = einsum("bhzc,bchpn->bzhpn", decay_chunk, states)`; carries an
   optional `initial_states` for varlen / cache continuation, emits
   `final_state` for the SSM cache.
4. **Output recombination.** Add the inter-chunk contribution.
   `Y_off = einsum("bclhn,bchpn,bhcl->bclhp", C, states, exp(A_cumsum))`;
   `Y = Y_diag + Y_off`.

A combined op `ssd_chunk_scan_combined` (registered like
`selective_scan_ops.mojo`'s `@compiler.register`) fuses dt-discretization,
conv (optional), the four stages, and the final-state write so the Python
pipeline calls a single op.

### Dimensions & dtypes

- `n_heads · head_dim = d_model`; `head_dim` 64–128; scalar `A` per head;
  `n_groups` for B/C sharing; `d_state` (`N`) 64–128.
- `chunk_size` tunable 64–256, default **128** (trades tensor-core utilization
  against per-chunk state overhead).
- Compute in fp32 accumulation; inputs bf16/fp16/fp32. The varlen scan path
  already supports `MAX_DSTATE=256`, covering Mamba2's range.

## Correctness / parity

Gate every kernel against the PyTorch/CUDA reference: compare
`ssd_chunk_scan_combined` to `mamba_chunk_scan_combined` and the pure-torch
`ssd_minimal_discrete` on identical seeded inputs. Tolerances: fp32 rtol
1e-5/atol 1e-6; fp16 1e-2/1e-3; bf16 2e-2/1e-2. Capture goldens for in-tree
`assert_almost_equal` tests (`max/kernels/test/{,gpu/}state_space/`), so
committed tests need no CUDA.

## Testing

- Mojo `std.testing` (`assert_almost_equal`), GPU tests gated on accelerator
  count; CPU reference path where practical.
- Per-stage unit tests + a combined-op test + the parity golden test.

## Alternatives considered

- **Extend `ssd_combined`** — rejected: it is Mamba1 sequential scan fused with
  norm; the chunk algorithm is structurally different.
- **Sequential scan for prefill** (reuse `mamba_split_conv1d_scan_combined`) —
  works and is the decode path, but forgoes the tensor-core speedup that is the
  entire point of Mamba2 prefill.

## Rollout (PR split)

One PR each: (1) intra-chunk kernel, (2) chunk-state kernel, (3) inter-chunk
scan, (4) output recombination, (5) `ssd_chunk_scan_combined` op wrapper,
(6) ops wrappers for the existing fused kernels (closes `TODO(MSTDL-2472)`),
(7) Mamba2 dim parameterization. Each carries its own tests and parity report.
