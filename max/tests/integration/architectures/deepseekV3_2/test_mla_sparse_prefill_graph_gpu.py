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
"""GPU consumers for the sparse MLA prefill CPU/GPU split.

The graphs are compiled to MEFs by a CPU-only build action
(``:mla_sparse_prefill_mefs`` via ``mef_precompile.bzl``); these tests do NOT
compile. ``test_prefill_from_precompiled_mef`` initializes each
:data:`PREFILL_SPECS` MEF and drives a fresh prefill through the combined op's
sparse-prefill branch (``mla_sm100_prefill_sparse``), asserting shape + finite.
``test_prefill_only_matches_auto`` initializes the precompiled ``auto`` and
``prefill`` MEFs with shared weights and asserts they produce matching output.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from _mla_sparse_graphs import (
    HIDDEN_SIZE,
    MAX_BATCH_SIZE,
    PREFILL_LEN,
    PREFILL_SPECS,
    SPECS_BY_NAME,
    TOTAL_NUM_PAGES,
    PrefillSpec,
    make_multi_kv,
    weights_for,
)
from max.driver import Accelerator, Buffer
from max.engine import InferenceSession
from max.nn.kv_cache import MultiKVCacheInputs
from max.pipelines.kv_cache import PagedKVCacheManager
from test_common.context_utils import create_text_context
from test_common.mef_precompile import init_from_mef, mefs_from_env
from torch.utils.dlpack import from_dlpack


@pytest.mark.parametrize("spec", PREFILL_SPECS, ids=lambda s: s.name)
def test_prefill_from_precompiled_mef(spec: PrefillSpec) -> None:
    """Init a CPU-precompiled MEF on GPU and run a fresh sparse prefill."""
    mef_path = mefs_from_env("MLA_MEF_RLOCATIONS")[f"{spec.name}.mef"]
    assert mef_path.is_file(), f"precompiled MEF missing: {mef_path}"

    device = Accelerator(0)
    session = InferenceSession(devices=[Accelerator()])

    model = init_from_mef(session, mef_path, weights_for(spec))

    multi_kv = make_multi_kv(spec.num_heads)
    kv_manager = PagedKVCacheManager(
        params=multi_kv,
        total_num_pages=TOTAL_NUM_PAGES,
        session=session,
        max_batch_size=MAX_BATCH_SIZE,
    )
    context = create_text_context(np.empty(PREFILL_LEN))
    kv_manager.claim(context)
    kv_manager.alloc(context)
    kv_ri_pref = kv_manager.runtime_inputs([[context]])
    assert isinstance(kv_ri_pref, MultiKVCacheInputs)

    t_pref = (
        torch.randn((PREFILL_LEN, HIDDEN_SIZE), dtype=torch.float32) * 0.02
    ).to(torch.bfloat16)
    hidden_prefill = Buffer.from_dlpack(t_pref).to(device)
    row_prefill = Buffer.from_numpy(
        np.array([0, PREFILL_LEN], dtype=np.uint32)
    ).to(device)
    out_pref = model.execute(
        hidden_prefill, row_prefill, *kv_ri_pref.flatten()
    )[0]

    out_t = from_dlpack(out_pref).cpu()
    out_np = (
        out_t.float().numpy()
        if out_t.dtype == torch.bfloat16
        else out_t.numpy()
    )
    assert out_np.shape == (PREFILL_LEN, HIDDEN_SIZE)
    assert not np.isnan(out_np).any()
    assert np.all(np.isfinite(out_np))


@pytest.mark.parametrize("num_heads", [128, 8])
def test_prefill_only_matches_auto(num_heads: int) -> None:
    """Disaggregated ``prefill_only`` routes through the same combined op as ``auto``.

    Initializes the CPU-precompiled ``auto`` and ``prefill`` MEFs (same head
    count) with shared weights, runs the same fresh prefill through each, and
    asserts the outputs match within a tight tolerance -- the only expected
    delta is nondeterministic reduction order across two kernel launches. For a
    pure prefill batch both modes assemble the identical graph, so a larger
    divergence would mean the prefill branch isn't reaching
    ``mla_sm100_prefill_sparse``. Runs at 128 heads (DeepSeek V3.2) and the GLM
    5.2 TP8-sharded count of 8 heads per device.
    """
    mefs = mefs_from_env("MLA_MEF_RLOCATIONS")
    # Shared weights across both modes, so a divergence isolates the graph-mode
    # routing rather than a weight mismatch (weights are mode-independent).
    weights = weights_for(SPECS_BY_NAME[f"prefill_fp8_{num_heads}h"])
    multi_kv = make_multi_kv(num_heads)

    device = Accelerator(0)
    session = InferenceSession(devices=[Accelerator()])

    t_pref = (
        torch.randn((PREFILL_LEN, HIDDEN_SIZE), dtype=torch.float32) * 0.02
    ).to(torch.bfloat16)
    hidden_prefill = Buffer.from_dlpack(t_pref).to(device)
    row_prefill = Buffer.from_numpy(
        np.array([0, PREFILL_LEN], dtype=np.uint32)
    ).to(device)

    def run(mef_name: str) -> np.ndarray:
        model = init_from_mef(session, mefs[mef_name], weights)
        # A fresh manager keeps each run's prefill cache writes independent.
        kv_manager = PagedKVCacheManager(
            params=multi_kv,
            total_num_pages=TOTAL_NUM_PAGES,
            session=session,
            max_batch_size=MAX_BATCH_SIZE,
        )
        context = create_text_context(np.empty(PREFILL_LEN))
        kv_manager.claim(context)
        kv_manager.alloc(context)
        kv_ri = kv_manager.runtime_inputs([[context]])
        assert isinstance(kv_ri, MultiKVCacheInputs)
        out_buf = model.execute(hidden_prefill, row_prefill, *kv_ri.flatten())[
            0
        ]
        out_t = from_dlpack(out_buf).cpu()
        return (
            out_t.float().numpy()
            if out_t.dtype == torch.bfloat16
            else out_t.numpy()
        )

    out_auto = run(f"prefill_fp8_{num_heads}h.mef")
    out_prefill = run(f"prefill_only_fp8_{num_heads}h.mef")

    assert out_prefill.shape == (PREFILL_LEN, HIDDEN_SIZE)
    assert not np.isnan(out_prefill).any()
    assert np.all(np.isfinite(out_prefill))
    np.testing.assert_allclose(out_prefill, out_auto, atol=1e-3, rtol=1e-2)
