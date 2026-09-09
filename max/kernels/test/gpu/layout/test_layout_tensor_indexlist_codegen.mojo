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
"""Test that LayoutTensor IndexList codegen matches variadic codegen.

This test verifies that using IndexList to index into a LayoutTensor produces
the same quality code as using variadic arguments. The key requirement is that
IndexList should not generate worse code than the variadic overload.

Before the fix (MSTDL-2166), IndexList generated inefficient code that loaded
strides from memory and computed offsets at runtime, while variadic args
could use compile-time constants.

After the fix, both should produce equivalent code.
"""

from std.compile import compile_info
from max.gpu.host import get_gpu_target
from layout import Layout, LayoutTensor
from std.testing import assert_true
from std.utils import Index


comptime layout_2d = Layout.row_major(10, 20)


def test_store_codegen_equivalence() raises:
    """Test that store(IndexList) produces equivalent code to store(m, n)."""

    def variadic_kernel(
        output: LayoutTensor[.int32, layout_2d, MutAnyOrigin],
    ):
        output.store[1](2, 3, 1234)

    def indexlist_kernel(
        output: LayoutTensor[.int32, layout_2d, MutAnyOrigin],
    ):
        output.store[1](Index(2, 3), 1234)

    var variadic_asm = String(
        compile_info[
            variadic_kernel, emission_kind="asm", target=get_gpu_target()
        ]()
    )
    var indexlist_asm = String(
        compile_info[
            indexlist_kernel, emission_kind="asm", target=get_gpu_target()
        ]()
    )

    # Check that IndexList doesn't load strides from memory when variadic doesn't
    # ld.param.v2.b32 is the instruction used to load stride pairs from params
    var variadic_loads_strides = "ld.param.v2.b32" in variadic_asm
    var indexlist_loads_strides = "ld.param.v2.b32" in indexlist_asm

    # IndexList should not load strides if variadic doesn't
    if not variadic_loads_strides:
        assert_true(
            not indexlist_loads_strides,
            "IndexList loads strides from memory but variadic doesn't. "
            + "IndexList should produce equivalent or better code.\n\n"
            + "Variadic ASM:\n"
            + variadic_asm
            + "\n\nIndexList ASM:\n"
            + indexlist_asm,
        )


def test_load_codegen_equivalence() raises:
    """Test that load(IndexList) produces equivalent code to load(m, n)."""

    def variadic_kernel(
        input: LayoutTensor[.int32, layout_2d, MutAnyOrigin],
        output: LayoutTensor[.int32, layout_2d, MutAnyOrigin],
    ):
        var val = input.load[1](2, 3)
        output.store[1](0, 0, val)

    def indexlist_kernel(
        input: LayoutTensor[.int32, layout_2d, MutAnyOrigin],
        output: LayoutTensor[.int32, layout_2d, MutAnyOrigin],
    ):
        var val = input.load[1](Index(2, 3))
        output.store[1](0, 0, val)

    var variadic_asm = String(
        compile_info[
            variadic_kernel, emission_kind="asm", target=get_gpu_target()
        ]()
    )
    var indexlist_asm = String(
        compile_info[
            indexlist_kernel, emission_kind="asm", target=get_gpu_target()
        ]()
    )

    # Check stride loading behavior
    var variadic_loads_strides = "ld.param.v2.b32" in variadic_asm
    var indexlist_loads_strides = "ld.param.v2.b32" in indexlist_asm

    if not variadic_loads_strides:
        assert_true(
            not indexlist_loads_strides,
            "IndexList loads strides from memory but variadic doesn't. "
            + "IndexList should produce equivalent or better code.\n\n"
            + "Variadic ASM:\n"
            + variadic_asm
            + "\n\nIndexList ASM:\n"
            + indexlist_asm,
        )


def test_aligned_load_codegen_equivalence() raises:
    """Test that aligned_load(IndexList) produces equivalent code to aligned_load(m, n).
    """

    def variadic_kernel(
        input: LayoutTensor[.int32, layout_2d, MutAnyOrigin],
        output: LayoutTensor[.int32, layout_2d, MutAnyOrigin],
    ):
        var val = input.aligned_load[1](2, 3)
        output.store[1](0, 0, val)

    def indexlist_kernel(
        input: LayoutTensor[.int32, layout_2d, MutAnyOrigin],
        output: LayoutTensor[.int32, layout_2d, MutAnyOrigin],
    ):
        var val = input.aligned_load[1](Index(2, 3))
        output.store[1](0, 0, val)

    var variadic_asm = String(
        compile_info[
            variadic_kernel, emission_kind="asm", target=get_gpu_target()
        ]()
    )
    var indexlist_asm = String(
        compile_info[
            indexlist_kernel, emission_kind="asm", target=get_gpu_target()
        ]()
    )

    # Check stride loading behavior
    var variadic_loads_strides = "ld.param.v2.b32" in variadic_asm
    var indexlist_loads_strides = "ld.param.v2.b32" in indexlist_asm

    if not variadic_loads_strides:
        assert_true(
            not indexlist_loads_strides,
            "IndexList loads strides from memory but variadic doesn't. "
            + "IndexList should produce equivalent or better code.\n\n"
            + "Variadic ASM:\n"
            + variadic_asm
            + "\n\nIndexList ASM:\n"
            + indexlist_asm,
        )


def main() raises:
    test_store_codegen_equivalence()
    test_load_codegen_equivalence()
    test_aligned_load_codegen_equivalence()
