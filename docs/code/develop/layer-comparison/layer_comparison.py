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
# DOC: max/develop/layer-comparison.mdx

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from max.driver import CPU, Buffer, load_max_buffer
from max.dtype import DType
from max.engine import InferenceSession, PrintStyle
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn import Embedding, Linear, RMSNorm
from max.nn.hooks import PrintHook
from max.nn.layer import LayerList, Module
from torch import nn

VOCAB = 32
HIDDEN = 8
NUM_LAYERS = 2
RMS_EPS = 1e-6
INPUT_IDS = [[1, 7, 13, 3]]  # one sequence of four tokens
SEQ_LEN = len(INPUT_IDS[0])

# Maps each MAX print name (the ``.max`` filename stem that ``PrintHook`` emits)
# to the matching PyTorch submodule path. ``PrintHook.name_layers`` names the
# root module ``model`` and, because the decoder blocks live in a ``LayerList``
# attribute named ``layers``, doubles it to ``model.layers.layers.N``.
LAYER_MAP = {
    "model.embed_tokens-output": "embed_tokens",
    "model.layers.layers.0-output": "layers.0",
    "model.layers.layers.1-output": "layers.1",
    "model.norm-output": "norm",
}


# --- PyTorch reference model (the source of truth) ------------------- #
class ReferenceBlock(nn.Module):
    """A residual ``silu(linear(x))`` block."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(HIDDEN, HIDDEN, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + F.silu(self.proj(x))


class ReferenceModel(nn.Module):
    """A two-layer decoder: ``embed_tokens`` -> ``layers`` -> ``norm``."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.layers = nn.ModuleList(
            [ReferenceBlock() for _ in range(NUM_LAYERS)]
        )
        self.norm = nn.RMSNorm(HIDDEN, eps=RMS_EPS)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden)


def build_reference_model() -> ReferenceModel:
    """Builds a seeded, random-weight reference model in float32."""
    torch.manual_seed(0)
    model = ReferenceModel().eval().to(torch.float32)
    with torch.no_grad():
        # Scale weights up so activations are O(1) and the SiLU/GELU gap moves
        # cosine similarity, not just the absolute error.
        nn.init.normal_(model.embed_tokens.weight, std=1.0)
        for block in model.layers:
            nn.init.normal_(block.proj.weight, std=0.5)
    return model


# --- MAX model, from public max.nn building blocks ------------------------- #
class Block(Module):
    """A residual ``activation(linear(x))`` block."""

    def __init__(self, activation: str) -> None:
        super().__init__()
        self.activation = activation
        self.proj = Linear(HIDDEN, HIDDEN, DType.float32, DeviceRef.CPU())

    def __call__(self, x: TensorValue) -> TensorValue:
        activate = ops.silu if self.activation == "silu" else ops.gelu
        return x + activate(self.proj(x))


class Model(Module):
    """A two-layer decoder: ``embed_tokens`` -> ``layers`` -> ``norm``."""

    def __init__(self, activations: list[str]) -> None:
        super().__init__()
        self.embed_tokens = Embedding(
            VOCAB, HIDDEN, DType.float32, DeviceRef.CPU()
        )
        self.layers = LayerList(
            [Block(activation) for activation in activations]
        )
        self.norm = RMSNorm(HIDDEN, DType.float32, eps=RMS_EPS)

    def __call__(self, input_ids: TensorValue) -> TensorValue:
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden)


# --- Capture MAX layer outputs --------------------------------------------- #
def capture_max_tensors(
    reference_model: ReferenceModel,
    activations: list[str],
    output_dir: str,
) -> None:
    """Runs the MAX model and writes each layer's output to a ``.max`` file.

    Loads the reference model's weights into the MAX model so the two match,
    attaches a ``PrintHook`` to name and print every layer, and configures the
    session to serialize each printed tensor as a MAX checkpoint file.
    """
    # Reuse the reference weights: the torch and MAX parameter names line up.
    weights = {
        name: tensor.detach().numpy()
        for name, tensor in reference_model.state_dict().items()
    }
    model = Model(activations)
    model.load_state_dict(weights, weight_alignment=1)

    # Name the layers before building the graph so the print ops are included.
    hook = PrintHook()
    hook.name_layers(model)

    graph = Graph(
        "layer_comparison",
        forward=model,
        input_types=[TensorType(DType.int64, [1, SEQ_LEN], DeviceRef.CPU())],
    )

    session = InferenceSession(devices=[CPU()])
    session.set_debug_print_options(
        style=PrintStyle.BINARY_MAX_CHECKPOINT,
        output_directory=output_dir,
    )
    compiled = session.compile(graph)
    runnable_model = session.init(compiled, weights_registry=model.state_dict())
    runnable_model.execute(
        Buffer.from_numpy(np.asarray(INPUT_IDS, dtype=np.int64))
    )
    hook.remove()


# --- Capture PyTorch layer outputs ----------------------------------------- #
def capture_reference_tensors(
    reference_model: ReferenceModel,
) -> dict[str, torch.Tensor]:
    """Runs the reference model and returns each layer's output by MAX name."""
    captured: dict[str, torch.Tensor] = {}

    def make_hook(name: str) -> Callable[..., None]:
        def hook(_module: object, _inputs: object, output: object) -> None:
            if isinstance(output, tuple):
                output = output[0]
            if isinstance(output, torch.Tensor):
                captured[name] = output.detach().cpu()

        return hook

    handles = []
    try:
        for max_name, torch_name in LAYER_MAP.items():
            module = reference_model.get_submodule(torch_name)
            handles.append(module.register_forward_hook(make_hook(max_name)))
        with torch.no_grad():
            reference_model(input_ids=torch.tensor(INPUT_IDS))
    finally:
        for handle in handles:
            handle.remove()
    return captured


# --- Compare the layers ---------------------------------------------------- #
@dataclass
class LayerComparison:
    name: str
    cosine_similarity: float
    max_abs_diff: float
    max_rel_diff: float


def compare_layer(
    name: str, reference: torch.Tensor, max_tensor_dir: str
) -> LayerComparison:
    """Compares one layer's MAX output against the reference output."""
    buffer = load_max_buffer(Path(max_tensor_dir) / f"{name}.max")
    max_output = torch.from_dlpack(buffer).cpu().to(torch.float32)
    reference = reference.to(torch.float32)

    abs_diff = (max_output - reference).abs()
    rel_diff = abs_diff / reference.abs().clamp_min(1e-10)
    cosine = F.cosine_similarity(
        max_output.flatten(), reference.flatten(), dim=0
    ).item()
    return LayerComparison(
        name, cosine, abs_diff.max().item(), rel_diff.max().item()
    )


def print_report(results: list[LayerComparison]) -> None:
    header = f"{'layer':<32}{'cosine':>12}{'max_abs':>14}{'max_rel':>14}"
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.name:<32}{result.cosine_similarity:>12.6f}"
            f"{result.max_abs_diff:>14.3e}{result.max_rel_diff:>14.3e}"
        )


def main() -> None:
    reference_model = build_reference_model()

    with tempfile.TemporaryDirectory() as max_tensor_dir:
        # Introduce a bug: the MAX model uses GELU in layer 1 instead of SiLU.
        capture_max_tensors(reference_model, ["silu", "gelu"], max_tensor_dir)
        reference_tensors = capture_reference_tensors(reference_model)
        results = [
            compare_layer(name, reference_tensors[name], max_tensor_dir)
            for name in LAYER_MAP
        ]

    print_report(results)

    # The first diverging layer localizes the bug: everything up to and
    # including layer 0 matches, while layer 1 and the downstream final norm
    # diverge. These assertions keep the example honest if an API changes.
    by_name = {result.name: result for result in results}
    assert by_name["model.embed_tokens-output"].cosine_similarity > 0.9999
    assert by_name["model.layers.layers.0-output"].cosine_similarity > 0.9999
    assert by_name["model.layers.layers.1-output"].cosine_similarity < 0.9999
    assert by_name["model.norm-output"].cosine_similarity < 0.9999


if __name__ == "__main__":
    main()
