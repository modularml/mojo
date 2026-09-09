# TileTensor MHA Conformance Investigation — Full Findings

**Date**: April 9-10, 2026
**System**: 8x AMD Instinct MI355X OAM (gfx950), TP=8
**Model**: Llama-3.1-405B (`RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8-dynamic`)
**PRs under investigation**:

- PR #82119 (`6936b243e61`): Structured prefill kernel
- PR #82541 (`e25b574fad6`): TileTensor conversion of AMD CDNA attention
  internals

## 1. Problem Statement

The 405B smoke test accuracy on 8xMI355 dropped from ~1.0 to 0.0 after
landing two MHA PRs. The test response became gibberish
(e.g., "Hello. the. has in..") instead of coherent text.

## 2. Commit Bisection

| Checkpoint                   | Commit                                   | Accuracy | Status |
|------------------------------|------------------------------------------|----------|--------|
| Pre-#82119 baseline          | `0aaa35b19c4`                            | 1.0      | PASS   |
| Post-#82119 (structured ON)  | `6936b243e61`                            | 1.0      | PASS   |
| Post-#82119 (structured OFF) | `6936b243e61` + `MHA_NO_STRUCTURED=True` | 1.0      | PASS   |
| Post-#82541 (TileTensor)     | `e25b574fad6`                            | 0.0      | FAIL   |
| HEAD of main                 | various                                  | 0.0      | FAIL   |

**Conclusion**: PR #82119 (structured prefill) is clean. The regression is
entirely in PR #82541 (TileTensor conversion).

## 3. Test Coverage Gap — group=16

**405B with TP=8 per-GPU shapes**:
`num_heads=16, kv_num_heads=1, depth=128, group=16`

Existing tests in `test_mha_causal_mask_amd.mojo` only tested up to
**group=8**. The group=16 configuration (16 Q heads per 1 KV head) was
never tested on AMD. With group <= 8, AMD buffer resource OOB clamping
silently masked errors on inactive MMA rows. With group=16, all 16 MFMA
rows are active, exposing bugs.

## 4. Contiguous Kernel Tests (flash_attention with contiguous Q/K/V)

### At original commit e25b574fad6

- **Prefill** (seq_len > 1) with group=16: **PASSED** all shapes
- **Decode** (seq_len=1) with group=16, depth=128: **FAILED**
  - 2 out of 2048 values exceeded 2% relative tolerance
  - BF16 specific (FP32 passed with 0 errors)
  - group=4: PASS, group=8: PASS, group=16: FAIL (exact boundary)

### After UInt-to-Int refactoring on main (commits a89010e1e19, 54d3612089a, 2ad7fdb062a)

- **All contiguous tests PASS**, including group=16 decode at depth=128
- The UInt-to-Int type changes resolved the precision issue that caused
  the contiguous test failures

### Key insight: contiguous test dispatches to wrong kernel

The contiguous `flash_attention` with `seq_len=1` dispatches to the
**prefill kernel** (`mha[...]` with BM=128), NOT to the **decode kernel**
(`mha_decoding[...]` with BM=16). The decode kernel is only triggered
through the KV cache `flash_attention` overload when
`is_token_generation=True`. So contiguous tests never actually exercised
the `mha_decoding` code path.

## 5. Paged KV Cache Tests

### Mixed CE (prefill with paged KV)

- `test_mha_mixed_ce_tg.mojo` with group=16, bf16: **PASSED**
- This test exercises prefill with cached context, NOT decode

### Paged decode (TG with paged KV)

- `test_batch_kv_cache_flash_attention_causal_mask_ragged_paged.mojo`
  with group=16 shapes added (previously gated behind
  `has_nvidia_gpu_accelerator()`)
- Single-partition (cache <= 256): **PASSED**
- Split-k (cache > 256): **PASSED**
- bs=1 with various cache sizes: **PASSED**
- Stress test (20 seeds, bs=4): **0 failures**
- Full suite (many shapes, many seeds): **rare data-dependent failure**
  (~0.01 absolute diff on ~0.5 values, approximately 1 ULP in bf16)

### Critical finding: continuous-vs-paged precision gap is PRE-EXISTING

The same rare continuous-vs-paged mismatch (batch=2, diff=0.01171875)
occurs on **main without TileTensor changes**. This is NOT a TileTensor
regression — it is a pre-existing precision difference between the
continuous and paged KV cache paths at group=16 that was never tested
on AMD before.

## 6. PRegisterBuffer.copy_to_shared Analysis

### At original commit e25b574fad6

The TileTensor `distribute[tt_col_major[warp_m, warp_n, 1]()]` approach
produced incorrect SMEM lane assignments. Reverting to the old
`copy_local_to_shared[thread_layout=warp_layout]` approach fixed the
contiguous kernel test.

### After UInt-to-Int refactoring

The `copy_to_shared` bug was resolved by the UInt-to-Int refactoring —
the contiguous test passes without any changes to `copy_to_shared`.
The signed vs unsigned type semantics likely changed how the `distribute`
mapping computed SMEM offsets.

### Smoke test

Applying the `copy_to_shared` fix (reverting to `copy_local_to_shared`)
did NOT fix the 405B smoke test — accuracy remained 0.0. This confirms
`copy_to_shared` was not the primary cause of the smoke test failure.

## 7. Smoke Test Status

The 405B smoke test (accuracy=0.0) fails consistently with the TileTensor
changes cherry-picked onto current main, despite ALL kernel-level tests
passing:

- Contiguous group=16 decode: PASS
- Paged group=16 decode: PASS (with rare pre-existing precision gaps)
- Paged group=16 prefill: PASS
- All existing test shapes: PASS

### What the smoke test does differently

1. Compiles through **bazel** (`./bazelw run smoke-test`) which builds
   `nn.mojopkg` as a compiled package, not from source
2. Uses the **graph compiler** for JIT compilation of the model graph
3. Runs **126 transformer layers** where small errors compound
4. Uses **TP=8** across 8 GPUs
5. Uses **FP8 quantized** model weights
6. Runs through the **full serving pipeline** (not a direct kernel call)

### Open question

The smoke test failure mechanism is not yet identified. All kernel-level
tests pass, the continuous-vs-paged precision gap is pre-existing, and
the `copy_to_shared` fix doesn't help. The root cause may be in:

- How the **bazel-compiled package** differs from `mojo` source compilation
- An interaction with the **graph compiler JIT**
- A **different code path** activated by the full pipeline that kernel
  tests don't cover
- An error that **compounds across 126 layers** from a very small
  per-layer precision difference

## 8. Op Shapes Reference (405B TP=8 per GPU)

### Prefill kernel (`mha[...]`)

- BM=128, BN=64, BK=32, WM=32, WN=64
- Grid: (num_heads=16, ceil(seq/BM), batch)

### Decode kernel (`mha_decoding[...]`)

- BM=16, BN=128, BK=32, WM=16, WN=32
- num_threads=256 (1 × 4 × 64)
- Grid: (num_partitions, num_heads//group=1, batch)
- Split-k when cache_length > 256

### MMA shape (gfx950, token_gen, depth=128)

- 16×16×32 MFMA
- fragment_layout: row_major(1, 4)
- warp_layout: col_major(16, 4)

## 9. Files Changed in PR #82541

| File                  | Changes                                                         |
|-----------------------|-----------------------------------------------------------------|
| `attention.mojo`      | `_dma_loop`, `RegTileWriter`, TileTensor MMA orchestration      |
| `buffers.mojo`        | KV buffer DMA, `PRegisterBuffer.copy_to_shared` rewrite         |
| `softmax.mojo`        | Score/output args changed from LayoutTensor to TileTensor       |
| `utils.mojo`          | `GlobalMemoryManager` with `valid_rows` + TileTensor tiles      |
| `mma.mojo`            | Free `mma()` replaced by `TiledMmaOp`                           |
| `mha_gfx942.mojo`     | Decode loop, KV tile construction, `num_b_rows` removal         |
| `mha_gfx950.mojo`     | Prefill MMA subtiles as TileTensor views                        |
| `mha_structured.mojo` | Structured prefill with TileTensor KV buffers                   |
| `mla.mojo`            | k_rope as TileTensor with MixedLayout (includes -O2 workaround) |
| `kv_buffer.mojo`      | New structured gfx950 KV path                                   |

## 10. Branches and Patches

- **Revert PR**: https://github.com/modularml/modular/pull/82960 (merged)
- **Re-land branch**: `reland-tiletensor-mha-v2` (cherry-pick of e25b574fad6 +
  test additions)
- **copy_to_shared patch**: `~/copy_to_shared_fix_and_tests.patch` (saved but
  NOT needed — UInt-to-Int fix resolved it)

## 11. Critical Finding: Contiguous Path is Bitwise Identical

Using Anand's hash-based reproducibility test (PR #83067), we confirmed
that the TileTensor code on current main produces **bitwise identical**
results to the pre-TileTensor code for ALL tested shapes:

| Test                     | TileTensor Hash      | Main Hash            | Match |
|--------------------------|----------------------|----------------------|-------|
| group=4 prefill 128x128  | 15578270350029038373 | 15578270350029038373 | YES   |
| group=16 prefill 128x128 | 18282737304533803813 | 18282737304533803813 | YES   |
| group=16 decode 1x12031  | 4927928534235661093  | 4927928534235661093  | YES   |
| group=8 decode 1x12031   | 8439226780733698853  | 8439226780733698853  | YES   |

The UInt-to-Int refactoring that landed on main (commits a89010e1e19,
54d3612089a, 2ad7fdb062a) appears to have resolved ALL the numerical
differences that were present in the original e25b574fad6 commit.

## 12. Pipeline Dispatch Traced

The serving pipeline dispatches through:

1. Python: `flash_attention_ragged()` →
   `ops.inplace_custom("mo.mha.ragged.paged")`
2. MOGG: `_execute_mha_ragged_paged_scalar_args()` →
   `generic_flash_attention_kv_cache_ragged()`
3. Mojo: `_flash_attention_dispatch()` → `gpu_flash_attention[ragged=True]()`
   (alias for `flash_attention[ragged=True]`)
4. This is the KV cache overload (line 304 of mha.mojo) which computes
   `is_token_generation` from cache state

This is the SAME overload our paged KV cache tests use. And our tests pass.

## 13. The Remaining Mystery

After clean build, all kernel tests pass (bitwise identical), paged
decode tests pass, yet the 405B smoke test produces gibberish output.
The only files changed are MHA attention code + tensor_core.mojo
(removed unused overloads) + amd_tile_io.mojo (new TileTensor helpers).
None of these affect non-MHA kernels.

## 14. THE GAP: paged test compares wrong-vs-wrong

The paged KV cache test
(`test_batch_kv_cache_flash_attention_causal_mask_ragged_paged`) compares
**continuous-batching vs paged** KV cache results. Both paths dispatch through
the SAME `mha_decoding` kernel with the SAME TileTensor code. If `mha_decoding`
has a group=16 bug, BOTH produce the same wrong answer, and the comparison
passes!

The contiguous hash test uses `flash_attention` with dense Q/K/V which
dispatches to `mha` (prefill kernel), NOT `mha_decoding`. So the hash
comparison proves prefill is correct but says nothing about decode.

**No test compares `mha_decoding` output against a known-correct
reference for group=16.** This is the missing test.

## 15. Definitive Repro: mha_decoding via KV cache path

Using a hash-based test that compares `flash_attention` output through
the KV cache overload (which triggers `mha_decoding`) between the
TileTensor branch and main:

| Config                            | Main Hash           | TileTensor Hash      | Match  |
|-----------------------------------|---------------------|----------------------|--------|
| group=4, kv_heads=8               | 3842828076352676645 | 3842828076352676645  | YES    |
| group=8, kv_heads=1               | 7229237656969765669 | 7229237656969765669  | YES    |
| group=16, kv_heads=1, small cache | 3388747208356569893 | 15015457017215230757 | **NO** |
| group=16, kv_heads=1, large cache | 7153609791466414885 | 12939245640425186085 | **NO** |

Both single-partition (no split-k) and multi-partition fail. The bug is
in the core `mha_decoding` kernel, not in split-k reduction.

Test file: `max/kernels/test/gpu/kv_cache/test_mha_decoding_vs_naive.mojo`
(~39 seconds to run).

## 16. Root Cause: TileTensor vs LayoutTensor SMEM write/read mismatch

### The smoking gun

In `KVBufferImpl` (`buffers.mojo`), the `load_from_shared` method for
`token_gen=True` has an explicit comment and LayoutTensor fallback:

```mojo
else:
    # Token-gen: use LayoutTensor path (TileTensor distribute
    # produces different offsets for single-row token-gen tiles).
```

This means the SMEM **read** path uses LayoutTensor distribution
(`mma_op.load_b` with `_load_matrix_frag` / `ds_read_tr16_b64`).

But the SMEM **write** path uses TileTensor distribution:

- `load_from_dram`: `RegTileLoader.load()` — TileTensor distribute
- `copy_to_shared`: `tt_copy_local_to_shared` — TileTensor distribute

### Why it breaks

The `RegTileLoader.load` stores data in registers using **col-major**
indexing (`dst_idx = i + j * M`), while the old `copy_dram_to_local`
stored using **row-major** indexing. Then `tt_copy_local_to_shared`
reads registers in col-major order (matching RegTileLoader), and
`copy_local_to_shared` reads in row-major order (matching old DMA).

Within each convention (old: all row-major, new: all col-major) the
register ↔ SMEM mapping is internally consistent. BUT the
`load_from_shared` workaround switches from TileTensor back to
LayoutTensor, breaking the consistency:

```text
WRITE PATH (TileTensor convention):
  DRAM → registers (RegTileLoader, col-major registers)
  registers → SMEM (tt_copy_local_to_shared, col-major read)
  → SMEM content is in TileTensor-ordered positions

READ PATH (LayoutTensor convention, workaround):
  SMEM → MMA registers (mma_op.load_b, expects LayoutTensor SMEM order)
  → reads from wrong SMEM positions
```

For group <= 8, `valid_rows < 16` means OOB-clamped MMA rows mask
the errors. For group=16, all rows are valid and the wrong data
produces wrong attention scores.

### Experiments performed

1. **Removed load_from_shared workaround** (use TileTensor load_b for
   all paths): group=4 ALSO broke. Confirms TiledMmaOp.load_b genuinely
   produces different results from mma_op.load_b for these tile shapes.

2. **Made copy_to_shared use LayoutTensor** (matching load_from_shared):
   group=16 hash still different because load_from_dram (RegTileLoader)
   still writes registers in col-major, but LayoutTensor copy_to_shared
   reads row-major.

3. **Made both load_from_dram AND copy_to_shared use LayoutTensor**:
   compilation error — `copy_dram_to_local` expects LayoutTensor src
   with specific element_layout, but TileTensor.to_layout_tensor()
   produces scalar elements.

### TileTensor distribute analysis

Deep analysis of `tile_tensor.mojo` `distribute` / `distribute_with_offset`
vs LayoutTensor `distribute` shows they produce **identical offset formulas**
for flat 2D row-major layouts. Both compute:

```text
thread_coord_i = (thread_id // thread_stride[i]) % thread_shape[i]
offset = sum(thread_coord_i * data_stride[i])
```

The actual difference is NOT in `distribute` itself but in:

1. **`TiledMmaOp.load_b`** uses `distribute[col_major[...]]` + vectorize
   on the MMA sub-tile — a generic approach

2. **`mma_op.load_b`** (old LayoutTensor) uses `_load_matrix_frag` which
   calls **`ds_read_tr16_b64`** — a hardware-specific LDS transposed-read
   intrinsic that reads elements in a specific hardware-defined pattern

These two approaches produce different register-level MMA operand layouts.
The hardware intrinsic performs a physical transpose during the LDS read
that the generic `distribute` doesn't replicate.

### `RegTileLoader.load` vs `copy_dram_to_local`

- `RegTileLoader.load`: uses `worker_idx = lane_id()` (warp scope) or
  `thread_idx.x` (block scope), stores dst in **col-major** order
- `copy_dram_to_local`: stores dst in **row-major** order (LayoutTensor
  native order)
- Both use the same thread distribution formula

The col-major vs row-major register storage is internally consistent
within each convention (RegTileLoader↔tt_copy_local_to_shared, and
copy_dram_to_local↔copy_local_to_shared). The break happens when
the workaround mixes conventions (TileTensor write + LayoutTensor read).

## 17. Fix Options

### Option A: Make all three steps use LayoutTensor for token_gen

Fix `load_from_dram` and `copy_to_shared` to use LayoutTensor for
token_gen, matching the existing `load_from_shared` workaround.
**Blocked**: `copy_dram_to_local` has type mismatches with TileTensor-
derived LayoutTensors (element_layout/SIMD width).

### Option B: Fix TiledMmaOp.load_b to match mma_op.load_b

Make `TiledMmaOp.load_b` use the same `_load_matrix_frag` /
`ds_read_tr16_b64` hardware intrinsic as the old `mma_op.load_b`.
Then remove the `load_from_shared` workaround and use TileTensor
consistently for all three steps.

### Option C: Fix TiledMmaOp.load_b to produce correct layout via distribute

Understand exactly what register layout `ds_read_tr16_b64` produces
and replicate it using TileTensor `distribute` with the correct
thread layout and vectorization pattern. This is the cleanest
TileTensor-native fix.

## 18. NEXT: compare mha_decoding against mha_gpu_naive

Need a test that:

1. Constructs a KV cache with data
2. Calls `flash_attention` with KV cache (triggering `mha_decoding`)
3. Also computes the same attention using `mha_gpu_naive`
4. Compares the two results for group=16

Possible explanations:

1. A code compilation difference that only manifests in the full
   nn.mojopkg with all ops compiled together (interaction between
   modules at compilation time)
2. A runtime difference in how the graph compiler invokes the kernel
   that our direct tests don't exercise
3. An issue in the `structured_kernels/amd_tile_io.mojo` file that
   affects compilation of adjacent code

## 14. Recommendations

1. **The kernel-level MHA code is correct** — all tests pass at group=16
   with both contiguous and paged KV cache
2. **The continuous-vs-paged precision gap at group=16 is pre-existing**
   and should be investigated separately (not blocking TileTensor re-land)
3. **The smoke test failure requires further investigation** into the
   bazel compilation / graph compiler / full pipeline interaction
4. **Add group=16 test cases permanently** to prevent future regressions
   (both `test_mha_causal_mask_amd.mojo` and
   `test_batch_kv_cache_flash_attention_causal_mask_ragged_paged.mojo`)
