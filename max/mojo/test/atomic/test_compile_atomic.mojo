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

from std.atomic import Atomic, Ordering, fence

from std.compile import compile_info
from std.testing import TestSuite, assert_false, assert_true

from max.gpu.host import get_gpu_target
from max.gpu.host.info import A100

comptime _MI300X_TARGET = get_gpu_target["mi300x"]()


def _assert_ordered(text: String, first: String, second: String) raises:
    """Asserts that both substrings appear in `text` and that `first`
    occurs before `second`."""
    var first_pos = text.find(first)
    var second_pos = text.find(second)
    assert_true(first_pos >= 0, String(t"expected to find {first}"))
    assert_true(second_pos >= 0, String(t"expected to find {second}"))
    assert_true(
        first_pos < second_pos,
        String(t"{first} must precede {second}"),
    )


def test_compile_store_nvptx_default_scope() raises:
    # Default (empty) scope maps to NVPTX's system syncscope.
    def my_store(ptr: Pointer[Int32, MutAnyOrigin], v: Int32):
        Atomic[Int32].store[ordering=Ordering.RELEASE](ptr, v)

    var ptx = String(compile_info[my_store, target=A100.target()]())

    assert_true("st.release.sys.global" in ptx)
    # Guards against regressing back to the pre-MOLIB-2603 lowering
    # (`atom.exch.release.<scope>`), which costs an atomic-unit roundtrip.
    assert_false("atom.exch" in ptx)


def test_compile_store_nvptx_device_scope() raises:
    def my_store(ptr: Pointer[Int32, MutAnyOrigin], v: Int32):
        Atomic[Int32, scope="device"].store[ordering=Ordering.RELEASE](ptr, v)

    var ptx = String(compile_info[my_store, target=A100.target()]())

    assert_true("st.release.gpu.global" in ptx)
    assert_false("atom.exch" in ptx)


def test_compile_load_nvptx_default_scope() raises:
    def my_load(
        dst: Pointer[Int32, MutAnyOrigin],
        ptr: Pointer[Int32, MutAnyOrigin],
    ):
        dst[] = Atomic[Int32].load[ordering=Ordering.ACQUIRE](ptr)

    var ptx = String(compile_info[my_load, target=A100.target()]())

    assert_true("ld.acquire.sys.global" in ptx)


def test_compile_load_nvptx_device_scope() raises:
    def my_load(
        dst: Pointer[Int32, MutAnyOrigin],
        ptr: Pointer[Int32, MutAnyOrigin],
    ):
        dst[] = Atomic[Int32, scope="device"].load[ordering=Ordering.ACQUIRE](
            ptr
        )

    var ptx = String(compile_info[my_load, target=A100.target()]())

    assert_true("ld.acquire.gpu.global" in ptx)


def test_compile_store_amdgpu_default_scope() raises:
    # AMDGPU release at system scope: write-back fence before the store and
    # SC0/SC1 bypass-cache flags on the `global_store` itself. Memory model:
    # https://llvm.org/docs/AMDGPUUsage.html#memory-model-gfx942.
    def my_store(ptr: Pointer[Int32, MutAnyOrigin], v: Int32):
        Atomic[Int32].store[ordering=Ordering.RELEASE](ptr, v)

    var asm = String(compile_info[my_store, target=_MI300X_TARGET]())

    assert_true("buffer_wbl2 sc0 sc1" in asm)
    assert_true("global_store_dword" in asm)
    _assert_ordered(asm, "buffer_wbl2", "global_store_dword")


def test_compile_load_amdgpu_default_scope() raises:
    # AMDGPU acquire at system scope: `global_load` with SC0/SC1, then a
    # `buffer_inv` to invalidate stale cache lines.
    def my_load(
        dst: Pointer[Int32, MutAnyOrigin],
        ptr: Pointer[Int32, MutAnyOrigin],
    ):
        dst[] = Atomic[Int32].load[ordering=Ordering.ACQUIRE](ptr)

    var asm = String(compile_info[my_load, target=_MI300X_TARGET]())

    assert_true("global_load_dword" in asm)
    assert_true("buffer_inv sc0 sc1" in asm)
    _assert_ordered(asm, "global_load_dword", "buffer_inv")


def test_compile_store_amdgpu_agent_scope() raises:
    # `scope="agent"` narrows the release fence to the device, so the
    # system-wide L2 write-back (`buffer_wbl2 sc0 sc1`) emitted at system
    # scope is no longer required. Absence of that sequence is the
    # concrete witness that scope narrowing reaches AMDGPU codegen.
    def my_store(ptr: Pointer[Int32, MutAnyOrigin], v: Int32):
        Atomic[Int32, scope="agent"].store[ordering=Ordering.RELEASE](ptr, v)

    var asm = String(compile_info[my_store, target=_MI300X_TARGET]())

    assert_true("global_store_dword" in asm)
    assert_false("buffer_wbl2 sc0 sc1" in asm)


def test_compile_store_load_amdgpu_llvm_ir() raises:
    def my_kernel(
        dst: Pointer[Int32, MutAnyOrigin],
        ptr: Pointer[Int32, MutAnyOrigin],
        v: Int32,
    ):
        Atomic[Int32].store[ordering=Ordering.RELEASE](ptr, v)
        dst[] = Atomic[Int32].load[ordering=Ordering.ACQUIRE](ptr)

    var ir = String(
        compile_info[
            my_kernel, target=_MI300X_TARGET, emission_kind="llvm-opt"
        ]()
    )

    assert_true("store atomic" in ir)
    assert_true("release" in ir)
    assert_true("load atomic" in ir)
    assert_true("acquire" in ir)
    # `atomicrmw xchg` would mean we regressed to the pre-MOLIB-2603 path.
    assert_false("atomicrmw xchg" in ir)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
