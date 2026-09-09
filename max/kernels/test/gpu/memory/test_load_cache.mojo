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
from max.gpu.memory import CacheEviction, CacheOperation, load
from std.testing import assert_equal, assert_true


def load_value[
    *,
    dtype: DType = .uint32,
    width: Int = 1,
    read_only: Bool = False,
    prefetch_size: Optional[Int] = None,
    cache_policy: CacheOperation = CacheOperation.ALWAYS,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
](ptr: ImmPointer[Scalar[dtype], ImmutAnyOrigin]) -> SIMD[dtype, width]:
    return load[
        width=width,
        read_only=read_only,
        prefetch_size=prefetch_size,
        cache_policy=cache_policy,
        eviction_policy=eviction_policy,
    ](ptr)


def test_load() raises:
    assert_true(
        "ld.global "
        in _compile_code[
            load_value[width=1, prefetch_size=128], emission_kind="asm"
        ]()
    )

    assert_true(
        "ld.global.L2::128B.v2.u32 "
        in _compile_code[
            load_value[width=2, prefetch_size=128], emission_kind="asm"
        ]()
    )

    assert_true(
        "ld.global.L2::128B.v4.u32 "
        in _compile_code[
            load_value[width=4, prefetch_size=128], emission_kind="asm"
        ]()
    )

    assert_true(
        "ld.global.L2::256B.v4.u32 "
        in _compile_code[
            load_value[width=4, prefetch_size=256], emission_kind="asm"
        ]()
    )

    assert_equal(
        String(
            _compile_code[
                load_value[width=64, prefetch_size=128], emission_kind="asm"
            ]()
        ).count("ld.global.L2::128B.v8.u32 "),
        8,
    )

    assert_true(
        "ld.global.lu.v2.u32 "
        in _compile_code[
            load_value[
                dtype=DType.uint32,
                width=2,
                prefetch_size=None,
                cache_policy=CacheOperation.LAST_USE,
            ],
            emission_kind="asm",
        ]()
    )

    assert_true(
        "ld.global.nc.v2.u32 "
        in _compile_code[
            load_value[dtype=DType.uint32, width=2, read_only=True],
            emission_kind="asm",
        ]()
    )


def main() raises:
    test_load()
