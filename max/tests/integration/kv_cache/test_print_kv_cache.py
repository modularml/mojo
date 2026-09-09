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

from dataclasses import dataclass

import pytest
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn.kv_cache import KVCacheParams, MHAKVCacheParams, PagedCacheValues


@dataclass(frozen=True)
class PrintKVCacheModel:
    """Model containing a single print KV cache op."""

    kv_params: KVCacheParams
    """Hyperparameters describing this instance of the KV cache."""

    layer_idx: int
    """Layer index of the KV cache collection."""

    def __call__(
        self,
        valid_lengths: TensorValue,
        *kv_inputs: TensorValue,
    ) -> None:
        """Stages a graph consisting of a print KV cache op.

        This contains both the print KV cache op and a "fetch" op to get a
        KVCacheCollection.
        """
        kv_collection = PagedCacheValues(
            kv_blocks=kv_inputs[0].buffer,
            cache_lengths=kv_inputs[1].tensor,
            lookup_table=kv_inputs[2].tensor,
            max_prompt_length=kv_inputs[3].tensor,
            max_cache_length=kv_inputs[4].tensor,
        )
        page_size = self.kv_params.page_size
        if page_size is None:
            raise ValueError(
                "KVCacheParams.page_size cannot be none, when printing."
            )
        ops.inplace_custom(
            "mo.print_kv_cache.paged",
            device=valid_lengths.device,
            values=[
                valid_lengths,
                *kv_collection.flatten(),
                ops.constant(
                    self.layer_idx, DType.uint32, device=DeviceRef.CPU()
                ),
                ops.constant(True, DType.bool, device=DeviceRef.CPU()),
            ],
        )


@pytest.mark.parametrize(
    "dtype",
    [
        # Test representative dtypes for compilation: one integer, one float.
        # The print_kv_cache kernel handles dtypes uniformly, so testing all
        # 10+ supported dtypes adds time without proportional coverage value.
        DType.uint32,
        DType.float32,
    ],
)
def test_print_kv_cache(dtype: DType) -> None:
    """Tests compiling a print KV cache op."""
    kv_params = MHAKVCacheParams(
        dtype=dtype,
        # Use minimal model parameters for faster compilation.
        n_kv_heads=1,
        head_dim=16,
        num_layers=1,
        page_size=16,
        devices=[DeviceRef.CPU()],
    )

    batch_size = 2
    graph = Graph(
        "print_kv_cache",
        forward=PrintKVCacheModel(kv_params, layer_idx=0),
        input_types=[
            TensorType(
                dtype=DType.uint32, shape=[batch_size], device=DeviceRef.CPU()
            ),
            *kv_params.flattened_kv_inputs(),
        ],
    )

    # Compile and init the print KV cache model.
    InferenceSession(devices=[CPU()]).load(graph)
