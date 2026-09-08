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
# RUN: %mojo -debug-level full %s | FileCheck %s

# Nested-scope capture into a unified closure must compile with full debug
# info. The capture load is emitted in the enclosing function and must not
# inherit the inner closure's debug subprogram (MOCO-4664).


def outer(K: Int):
    def task_func(task_id: Int) {var K, imm}:
        for ko in range(0, K, 16):

            @always_inline
            def inner[tile_n: Int](n_idx: Int) {imm}:
                var x = ko + n_idx * tile_n
                _ = x

            inner[2](task_id)

    task_func(0)


def plain_fn_loop_capture(K: Int):
    for g in range(0, K, 16):

        def pack(n: Int) {imm} -> Int:
            return g + n

        _ = pack(0)


def main():
    outer(64)
    plain_fn_loop_capture(32)
    # CHECK: ok
    print("ok")
