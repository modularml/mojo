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
# RUN: %parse-mojo-isolated %s -debug-level full -mlir-print-debuginfo | kgen-opt -lower-semantic-cf -check-lifetimes -mlir-print-debuginfo | FileCheck %s

# Capturing a loop variable into a unified closure materializes a load in the
# enclosing function. That load (and the CheckLifetimes debuginfo.kill that
# follows it) must carry the enclosing function's debug scope, not the inner
# closure's subprogram (MOCO-4664). Succeeding through check-lifetimes with
# full debug info is the verifier regression; the CHECKs pin the load scope.

def take[T: def (x: Int) -> Int, //](state: T, x: Int):
    _ = state(x)


def plain_fn_loop_capture(items: List[Int]):
    for g in items:
        def pack(n: Int) {imm} -> Int:
            return g + n

        _ = take(pack, 0)


def nested_closure_captures_loop_var(items: List[Int]):
    def task(task_id: Int) {imm}:
        for ko in items:
            def inner(n: Int) {imm} -> Int:
                return ko + n

            _ = take(inner, task_id)

    task(0)


# CHECK-DAG: pack::__storage
# CHECK-DAG: inner::__storage

# CHECK-DAG: lit.ref.load %g{{.*}}loc(#[[G_LOAD:loc[0-9]+]])
# CHECK-DAG: lit.ref.load %ko{{.*}}loc(#[[KO_LOAD:loc[0-9]+]])

# CHECK-DAG: #[[PLAIN_NAME:.*]] = #debuginfo.source_name<(fn)"plain_fn_loop_capture"
# CHECK-DAG: #[[TASK_NAME:.*]] = #debuginfo.source_name<(fn)"task"
# CHECK-DAG: #[[G_LOAD]] = loc(fused<#[[PLAIN_SP:.*]]>
# CHECK-DAG: #[[KO_LOAD]] = loc(fused<#[[TASK_SP:.*]]>
# CHECK-DAG: #[[PLAIN_SP]] = #debuginfo.subprogram{{.*}}sourceName = #[[PLAIN_NAME]]
# CHECK-DAG: #[[TASK_SP]] = #debuginfo.subprogram{{.*}}sourceName = #[[TASK_NAME]]
