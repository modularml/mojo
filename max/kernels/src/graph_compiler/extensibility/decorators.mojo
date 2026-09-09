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
"""Decorators for registering MAX Graph kernels.

`@register` is the public decorator for DPS-style custom kernels.
`@register_internal` is used by built-in MAX Graph operations.
"""


def register_internal(name: StaticString):
    """
    Registers the given mojo function as an implementation of an `mo` op or a
    `mo.custom` op. Used by built-in
    [MAX Graph operations](/api/python/graph.ops).

    For registering [custom operations](/develop/custom-ops/), use the
    [@extensibility.register](/api/mojo/extensibility/decorators/register/) decorator,
    instead.

    For instance:

    ```mojo
    @register_internal("mo.add")
    def my_op[...](...):
      ...
    ```

    Registers `my_op` as an implementation of `mo.add`.

    Args:
      name: The name of the op to register.
    """
    return


def register(
    name: StaticString,
    type: StaticString = "",
    api: StaticString = "",
    arch: StaticString = "",
    model: StaticString = "",
):
    """Registers a struct as a kernel implementation for a MAX Graph op.

    At compile time, the Graph Compiler selects the most specific registered
    kernel that matches the runtime device. The four device fields narrow the
    target from coarse to fine:

    - `type`: broad device category: `"cpu"`, `"gpu"`, or a custom label.
    - `api`: compute backend: `"cuda"`, `"hip"`, `"metal"`, etc.
    - `arch`: microarchitecture: `"sm_100a"`, `"sm_90a"`, `"gfx942"`, etc.
    - `model`: exact hardware model: `"NVIDIA B200"`, `"AMD Instinct MI355X"`, etc.

    An empty string for any field acts as a wildcard: it matches any device
    value for that field.
    Fields are ordered from least to most specific: `type` < `api` < `arch` <
    `model`. Among all matching candidates, the most specific registration wins:
    a registration that sets `arch` is more specific than one that sets only
    `type` and `api`, because `arch` sits higher in the hierarchy.

    If a user-defined kernel and a built-in kernel are equally specific for a
    device, the user-defined kernel takes precedence.
    Registering the same op with identical device fields more than once,
    whether within the same library or across multiple user libraries, is an
    error reported by the Graph Compiler.

    Example: device field combinations and their reach:

    ```mojo
    # Matches all devices (backward-compatible default).
    @extensibility.register("mo.matmul")

    # Matches all CPU devices.
    @extensibility.register("mo.matmul", type="cpu")

    # Matches all GPU devices regardless of vendor or architecture.
    @extensibility.register("mo.matmul", type="gpu")

    # Matches all CUDA GPUs (any architecture).
    @extensibility.register("mo.matmul", type="gpu", api="cuda")

    # Matches only the NVIDIA SM100A architecture.
    @extensibility.register("mo.matmul", type="gpu", api="cuda", arch="sm_100a")

    # Matches only the NVIDIA B200.
    @extensibility.register("mo.matmul", type="gpu", api="cuda", arch="sm_100a", model="NVIDIA B200")
    ```

    Example: selection with multiple registrations in scope:

    ```mojo
    @extensibility.register("mo.matmul")                                          # wildcard
    @extensibility.register("mo.matmul", type="cpu")                              # least specific
    @extensibility.register("mo.matmul", type="gpu")                              # least specific
    @extensibility.register("mo.matmul", type="gpu", api="cuda", arch="sm_100a")  # more specific
    ```

    - SM100A CUDA GPU → all four match; `arch` registration wins (most specific).
    - SM90A CUDA GPU  → wildcard and `type="gpu"` match; `type="gpu"` wins.
    - CPU             → wildcard and `type="cpu"` match; `type="cpu"` wins.
    - Some NPU        → only the wildcard matches; wildcard wins.

    Args:
        name: The MAX Graph op name to register (e.g. `"mo.matmul"`).
        type: Broad device category: `"cpu"`, `"gpu"`, or a custom accelerator
            label such as `"npu-xxx"`. Corresponds to the label used in
            `DeviceRef` (e.g. `DeviceRef.GPU()`). Empty matches all.
        api: Programming API or compute backend for the device, for example
            `"cuda"` (NVIDIA CUDA), `"hip"` (AMD ROCm/HIP), or `"metal"`
            (Apple Metal). Empty matches all.
        arch: Microarchitecture identifier, for example `"sm_90a"` (Hopper),
            `"sm_100a"` (Blackwell), or `"gfx942"` (AMD CDNA3). Empty matches
            all.
        model: Specific device model, for example `"NVIDIA B200"` or
            `"AMD Instinct MI355X"`. Empty matches all.
    """
    pass


def register_shape_function(name: StaticString):
    """Registers a standalone shape function for a MAX Graph op.

    The decorated function computes the output shapes of the named op from its
    input shapes at compile time, without executing the op itself.

    Args:
        name: The name of the op whose output shapes the function computes.
    """
    pass


def view_kernel():
    """Marks a DPS kernel as a view operation that aliases its input memory.

    Decorated kernels return a tensor that shares storage with one of their
    inputs rather than allocating new memory, so the Graph Compiler can preserve
    aliasing relationships across the operation.
    """
    return
