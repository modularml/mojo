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

from __future__ import annotations

from _attention_helpers import (
    MAX_DTYPE,
    build_max_attention_graph,
    make_attention_weights_global,
    make_attention_weights_local,
    make_text_config,
)
from max.driver import Accelerator
from max.dtype import DType
from max.graph import DeviceRef, Graph
from test_common.mef_precompile import precompile_entrypoint


def build_graph(spec: str) -> Graph:
    text_config = make_text_config()
    attn_weights = (
        make_attention_weights_local(text_config)
        if "local" in spec
        else make_attention_weights_global(text_config)
    )
    layer_idx = 0 if "local" in spec else 5
    cache_dtype = DType.float8_e4m3fn if "native_fp8" in spec else None
    return build_max_attention_graph(
        text_config=text_config,
        attention_weights=attn_weights,
        dtype=MAX_DTYPE,
        device_ref=DeviceRef.from_device(Accelerator()),
        layer_idx=layer_idx,
        cache_dtype=cache_dtype,
    )[0]


if __name__ == "__main__":
    precompile_entrypoint(build_graph)
