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

from max.gpu.host.compile import _compile_code
from max.gpu.host.info import H100
from max.gpu.memory import *
from max.gpu.intrinsics import Scope
from std.testing import *


def test_multimem_ld_reduce() raises:
    print("== test_multimem_ld_reduce")
    assert_true(
        "multimem.ld_reduce.weak.gpu.global.max.v2.bf16 "
        in _compile_code[
            multimem_ld_reduce[
                DType.bfloat16,
                count=2,
                reduction=ReduceOp.MAX,
                scope=Scope.GPU,
                accum_type=DType.bfloat16,
                consistency=Consistency.WEAK,
            ],
            target=H100.target(),
        ]().asm
    )

    assert_true(
        "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16 "
        in _compile_code[
            multimem_ld_reduce[
                DType.bfloat16,
                count=4,
                reduction=ReduceOp.ADD,
                scope=Scope.SYSTEM,
                accum_type=DType.bfloat16,
                consistency=Consistency.RELAXED,
            ],
            target=H100.target(),
        ]().asm
    )

    assert_true(
        "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16 "
        in _compile_code[
            multimem_ld_reduce[
                DType.bfloat16,
                count=4,
                reduction=ReduceOp.ADD,
                scope=Scope.SYSTEM,
                accum_type=DType.bfloat16,
                consistency=Consistency.RELAXED,
            ],
            target=H100.target(),
        ]().asm
    )

    assert_true(
        "multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16 "
        in _compile_code[
            multimem_ld_reduce[
                DType.bfloat16,
                count=4,
                reduction=ReduceOp.ADD,
                scope=Scope.SYSTEM,
                consistency=Consistency.RELAXED,
            ],
            target=H100.target(),
        ]().asm
    )

    assert_true(
        "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 "
        in _compile_code[
            multimem_ld_reduce[
                DType.bfloat16,
                count=4,
                reduction=ReduceOp.ADD,
                scope=Scope.SYSTEM,
                consistency=Consistency.RELAXED,
                accum_type=DType.bfloat16,
                output_width=2,
            ],
            target=H100.target(),
        ]().asm
    )

    assert_true(
        "multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2 "
        in _compile_code[
            multimem_ld_reduce[
                DType.bfloat16,
                count=4,
                reduction=ReduceOp.ADD,
                scope=Scope.SYSTEM,
                consistency=Consistency.RELAXED,
                accum_type=DType.float32,
                output_width=2,
            ],
            target=H100.target(),
        ]().asm
    )

    assert_true(
        "multimem.ld_reduce.relaxed.sys.global.max.v4.bf16x2 "
        in _compile_code[
            multimem_ld_reduce[
                DType.bfloat16,
                count=4,
                reduction=ReduceOp.MAX,
                scope=Scope.SYSTEM,
                consistency=Consistency.RELAXED,
                accum_type=DType.bfloat16,
                output_width=2,
            ],
            target=H100.target(),
        ]().asm
    )

    # Test count=1 (scalar operations)
    assert_true(
        "multimem.ld_reduce.relaxed.gpu.global.add.bf16x2 "
        in _compile_code[
            multimem_ld_reduce[
                DType.bfloat16,
                count=1,
                reduction=ReduceOp.ADD,
                scope=Scope.GPU,
                consistency=Consistency.RELAXED,
                accum_type=DType.bfloat16,
                output_width=2,
            ],
            target=H100.target(),
        ]().asm
    )

    assert_true(
        "multimem.ld_reduce.weak.sys.global.max.f32 "
        in _compile_code[
            multimem_ld_reduce[
                DType.float32,
                count=1,
                reduction=ReduceOp.MAX,
                scope=Scope.SYSTEM,
                consistency=Consistency.WEAK,
                accum_type=DType.float32,
                output_width=1,
            ],
            target=H100.target(),
        ]().asm
    )

    assert_true(
        "multimem.ld_reduce.relaxed.gpu.global.add.f64 "
        in _compile_code[
            multimem_ld_reduce[
                DType.float64,
                count=1,
                reduction=ReduceOp.ADD,
                scope=Scope.GPU,
                consistency=Consistency.RELAXED,
                accum_type=DType.float64,
                output_width=1,
            ],
            target=H100.target(),
        ]().asm
    )

    # Test count=8 (v8 operations)
    assert_true(
        "multimem.ld_reduce.relaxed.gpu.global.add.v8.bf16 "
        in _compile_code[
            multimem_ld_reduce[
                DType.bfloat16,
                count=8,
                reduction=ReduceOp.ADD,
                scope=Scope.GPU,
                consistency=Consistency.RELAXED,
                accum_type=DType.bfloat16,
                output_width=1,
            ],
            target=H100.target(),
        ]().asm
    )

    assert_true(
        "multimem.ld_reduce.weak.sys.global.max.v8.f16 "
        in _compile_code[
            multimem_ld_reduce[
                DType.float16,
                count=8,
                reduction=ReduceOp.MAX,
                scope=Scope.SYSTEM,
                consistency=Consistency.WEAK,
                accum_type=DType.float16,
                output_width=1,
            ],
            target=H100.target(),
        ]().asm
    )

    assert_true(
        "multimem.ld_reduce.relaxed.cluster.global.add.v8.f16 "
        in _compile_code[
            multimem_ld_reduce[
                DType.float16,
                count=8,
                reduction=ReduceOp.ADD,
                scope=Scope.CLUSTER,
                consistency=Consistency.RELAXED,
                accum_type=DType.float16,
                output_width=1,
            ],
            target=H100.target(),
        ]().asm
    )


def test_multimem_st() raises:
    print("== test_multimem_st")

    assert_true(
        "multimem.st.weak.cta.global.v2.bf16x2 "
        in _compile_code[
            multimem_st[
                DType.bfloat16,
                count=2,
                scope=Scope.BLOCK,
                consistency=Consistency.WEAK,
                width=2,
            ],
            target=H100.target(),
        ]().asm
    )

    assert_true(
        "multimem.st.relaxed.gpu.global.v4.f32 "
        in _compile_code[
            multimem_st[
                DType.float32,
                count=4,
                scope=Scope.GPU,
                consistency=Consistency.RELAXED,
            ],
            target=H100.target(),
        ]().asm
    )

    assert_true(
        "multimem.st.release.sys.global.v4.f16x2 "
        in _compile_code[
            multimem_st[
                DType.float16,
                count=4,
                scope=Scope.SYSTEM,
                consistency=Consistency.RELEASE,
                width=2,
            ],
            target=H100.target(),
        ]().asm
    )

    assert_true(
        "multimem.st.relaxed.cluster.global.v4.bf16 "
        in _compile_code[
            multimem_st[
                DType.bfloat16,
                count=4,
                scope=Scope.CLUSTER,
                consistency=Consistency.RELAXED,
            ],
            target=H100.target(),
        ]().asm
    )

    # Test count=1 (scalar operations)
    assert_true(
        "multimem.st.relaxed.gpu.global.bf16x2 "
        in _compile_code[
            multimem_st[
                DType.bfloat16,
                count=1,
                scope=Scope.GPU,
                consistency=Consistency.RELAXED,
                width=2,
            ],
            target=H100.target(),
        ]().asm
    )

    assert_true(
        "multimem.st.weak.sys.global.f32 "
        in _compile_code[
            multimem_st[
                DType.float32,
                count=1,
                scope=Scope.SYSTEM,
                consistency=Consistency.WEAK,
                width=1,
            ],
            target=H100.target(),
        ]().asm
    )

    assert_true(
        "multimem.st.release.gpu.global.f64 "
        in _compile_code[
            multimem_st[
                DType.float64,
                count=1,
                scope=Scope.GPU,
                consistency=Consistency.RELEASE,
                width=1,
            ],
            target=H100.target(),
        ]().asm
    )

    # Test count=8 (v8 operations)
    assert_true(
        "multimem.st.relaxed.gpu.global.v8.f16 "
        in _compile_code[
            multimem_st[
                DType.float16,
                count=8,
                scope=Scope.GPU,
                consistency=Consistency.RELAXED,
                width=1,
            ],
            target=H100.target(),
        ]().asm
    )

    assert_true(
        "multimem.st.weak.cluster.global.v8.bf16 "
        in _compile_code[
            multimem_st[
                DType.bfloat16,
                count=8,
                scope=Scope.CLUSTER,
                consistency=Consistency.WEAK,
                width=1,
            ],
            target=H100.target(),
        ]().asm
    )


def main() raises:
    test_multimem_ld_reduce()
    test_multimem_st()
