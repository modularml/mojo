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
"""Shared graph construction for the vision transformer CPU/GPU split.

The CPU producer (``precompile_transformer``) and the GPU consumer
(``test_transformer``) both import this module so they build the *identical*
graph -- the producer compiles it to a MEF with no GPU, the consumer
initializes that MEF and executes it. Keeping the construction in one place is
what guarantees the compiled artifact and the runtime inputs can't drift apart.

Every input dimension that varies across the test's cases (patch count, video
count, sequence count) is symbolic, so a single compiled graph serves all of
them and the module needs only one spec.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.pipelines.architectures.kimik2_5.layers.vision.transformer import (
    Transformer,
)
from max.pipelines.architectures.kimik2_5.model_config import VisionConfig

MAX_DTYPE = DType.bfloat16

NUM_HEADS = 16
HIDDEN_DIM = 1152
MLP_DIM = 4304

ROPE_MAX_HEIGHT = 512
ROPE_MAX_WIDTH = 512
ROPE_THETA = 10000.0

PATCH_SIZE = 14
IN_CHANNELS = 3
INIT_POS_EMB_HEIGHT = 64
INIT_POS_EMB_WIDTH = 64
INIT_POS_EMB_TIME = 4
MERGE_KERNEL_SIZE = (2, 2)
DECODER_HIDDEN_SIZE = 7168
VT_NUM_LAYERS = 2

TRANSFORMER_SPEC = "kimik2_5_transformer"
"""Stable identifier used for the MEF filename and the producer's ``--spec``."""


def build_transformer_graph() -> Graph:
    """Builds the vision transformer graph (device-independent).

    This is the single construction path the CPU producer compiles and the GPU
    consumer initializes.

    Returns:
        The constructed :class:`Graph`, ready to compile.
    """
    vision_config = VisionConfig(
        dtype=MAX_DTYPE,
        devices=[DeviceRef.GPU()],
        init_pos_emb_height=INIT_POS_EMB_HEIGHT,
        init_pos_emb_time=INIT_POS_EMB_TIME,
        init_pos_emb_width=INIT_POS_EMB_WIDTH,
        merge_kernel_size=list(MERGE_KERNEL_SIZE),
        mm_hidden_size=HIDDEN_DIM,
        patch_size=PATCH_SIZE,
        projector_ln_eps=1e-5,
        text_hidden_size=DECODER_HIDDEN_SIZE,
        vt_hidden_size=HIDDEN_DIM,
        vt_intermediate_size=MLP_DIM,
        vt_num_attention_heads=NUM_HEADS,
        vt_num_hidden_layers=VT_NUM_LAYERS,
        in_channels=IN_CHANNELS,
        rope_max_height=ROPE_MAX_HEIGHT,
        rope_max_width=ROPE_MAX_WIDTH,
        rope_theta=ROPE_THETA,
    )
    vision_tower = Transformer(vision_config)
    # Materialize the layer's weights so each one carries its fully-qualified
    # name; that name is the key the consumer's weights registry binds to.
    vision_tower.state_dict()

    with Graph(
        TRANSFORMER_SPEC,
        input_types=[
            TensorType(
                MAX_DTYPE,
                ["n_patches", IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
                device=DeviceRef.GPU(),
            ),
            TensorType(
                DType.int64,
                ["n_videos", 3],
                device=DeviceRef.GPU(),
            ),
            TensorType(
                DType.uint32,
                ["num_seqs"],
                device=DeviceRef.GPU(),
            ),
            TensorType(DType.uint32, [1], device=DeviceRef.CPU()),
            TensorType(
                DType.int64,
                ["n_patches"],
                device=DeviceRef.GPU(),
            ),
        ],
    ) as graph:
        (
            pixel_values_in,
            grid_thws_in,
            input_row_offsets_in,
            max_seq_len_in,
            position_ids_in,
        ) = graph.inputs
        assert isinstance(pixel_values_in, TensorValue)
        assert isinstance(grid_thws_in, TensorValue)
        assert isinstance(input_row_offsets_in, TensorValue)
        assert isinstance(max_seq_len_in, TensorValue)
        assert isinstance(position_ids_in, TensorValue)
        outs = vision_tower(
            [pixel_values_in],
            [grid_thws_in],
            [input_row_offsets_in],
            [max_seq_len_in],
            [position_ids_in],
            [],
        )
        graph.output(outs[0])
    return graph
