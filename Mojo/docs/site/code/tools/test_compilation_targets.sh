#!/usr/bin/env bash
##===----------------------------------------------------------------------===##
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
##===----------------------------------------------------------------------===##
# Compilation targets full test suite
# See README-Compilation-Targets.md for full context
#
# Tests the examples and claims on the compilation-targets.mdx page.
# Run from any directory. Test artifacts are created in /tmp and
# cleaned up on exit.

set -o pipefail

WORKDIR=$(mktemp -d /tmp/mojo-xc-test.XXXXXX)
trap 'rm -rf $WORKDIR' EXIT

pass=0
fail=0
total=0

# Create test source files
cat > "$WORKDIR/test.mojo" <<'EOF'
"""Minimal executable for compilation-target testing."""


def main():
    var x: Int = 42
    var y: Int = x + 1
EOF

cat > "$WORKDIR/testlib.mojo" <<'EOF'
"""Minimal library for compilation-target testing."""


def add(x: Int, y: Int) -> Int:
    return x + y
EOF

cat > "$WORKDIR/test-gpu.mojo" <<'GPUEOF'
"""Minimal GPU kernel for compilation-target testing."""

from std.memory import Pointer
from max.gpu.host import DeviceContext

comptime `✅`: Int32 = 1
comptime `❌`: Int32 = 0


def kernel(
    value: Pointer[Scalar[DType.int32], MutAnyOrigin],
):
    value[0] = `✅`


def main() raises:
    with DeviceContext() as ctx:
        var out = ctx.enqueue_create_buffer[DType.int32](1)
        out.enqueue_fill(`❌`)
        ctx.enqueue_function[kernel](
            out, grid_dim=1, block_dim=1
        )
        with out.map_to_host() as out_host:
            print(
                "GPU responded:",
                "✅" if out_host[0] == `✅` else "❌",
            )
GPUEOF

# --- Test runners ---

run_test() {
    local num=$1
    local desc=$2
    shift 2
    ((total++))
    if "$@" > /dev/null 2>&1; then
        echo "PASS  $num  $desc"
        ((pass++))
    else
        echo "FAIL  $num  $desc"
        ((fail++))
    fi
}

run_test_expect_fail() {
    local num=$1
    local desc=$2
    shift 2
    ((total++))
    if "$@" > /dev/null 2>&1; then
        echo "FAIL  $num  $desc (expected failure, got success)"
        ((fail++))
    else
        echo "PASS  $num  $desc"
        ((pass++))
    fi
}

run_test_check_arch() {
    local num=$1
    local desc=$2
    local outfile=$3
    local expected=$4
    shift 4
    ((total++))
    if ! "$@" > /dev/null 2>&1; then
        echo "FAIL  $num  $desc (compilation failed)"
        ((fail++))
        return
    fi
    if file "$outfile" | grep -q "$expected"; then
        echo "PASS  $num  $desc"
        ((pass++))
    else
        echo "FAIL  $num  $desc (wrong arch: $(file -b "$outfile"))"
        ((fail++))
    fi
}

# --- Header ---

echo "Compilation target test suite"
echo "Mojo:    $(mojo --version)"
echo "Host:    $(uname -m) $(uname -s) $(uname -r)"
echo "Workdir: $WORKDIR"
echo ""

# --- Group 1: Query flags ---

echo "## Group 1: Query flags"

run_test 1 "print-effective-target" \
    mojo build --print-effective-target

run_test 2 "print-supported-targets" \
    mojo build --print-supported-targets

run_test 3 "print-supported-cpus (aarch64)" \
    mojo build --print-supported-cpus --target-triple=aarch64-apple-macosx

run_test 4 "print-supported-cpus (x86_64)" \
    mojo build --print-supported-cpus --target-triple=x86_64-unknown-linux-gnu

run_test 5 "print-supported-accelerators" \
    mojo build --print-supported-accelerators

echo ""

# --- Group 2: Mojo target flags ---

echo "## Group 2: Mojo target flags (--target-* path)"

run_test_check_arch 6 "target-cpu aarch64 object" \
    "$WORKDIR/test6.o" "aarch64" \
    mojo build \
        --target-triple aarch64-unknown-linux-gnu \
        --target-cpu cortex-a72 \
        --emit object -o "$WORKDIR/test6.o" "$WORKDIR/test.mojo"

run_test_check_arch 7 "target-cpu x86_64 object" \
    "$WORKDIR/test7.o" "x86-64" \
    mojo build \
        --target-triple x86_64-unknown-linux-gnu \
        --target-cpu x86-64-v3 \
        --target-features "+avx512f" \
        --emit object -o "$WORKDIR/test7.o" "$WORKDIR/test.mojo"

echo ""

# --- Group 3: GCC/Clang-compatible flags ---

echo "## Group 3: GCC/Clang flags (--march/--mcpu/--mtune path)"
echo "   (warnings expected from MOCO-3686 when cross-arch)"

run_test_check_arch 8 "mcpu haswell object" \
    "$WORKDIR/test8.o" "x86-64" \
    mojo build \
        --target-triple x86_64-unknown-linux-gnu \
        --mcpu=haswell \
        --emit object -o "$WORKDIR/test8.o" "$WORKDIR/test.mojo"

run_test 9 "march skylake-avx512 asm" \
    mojo build \
        --target-triple x86_64-unknown-linux-gnu \
        --march=skylake-avx512 \
        --emit asm -o "$WORKDIR/test9.s" "$WORKDIR/test.mojo"

run_test_check_arch 10 "march + mcpu + mtune object" \
    "$WORKDIR/test10.o" "x86-64" \
    mojo build \
        --target-triple x86_64-unknown-linux-gnu \
        --march=x86-64 --mcpu=haswell --mtune=skylake \
        --emit object -o "$WORKDIR/test10.o" "$WORKDIR/test.mojo"

echo ""

# --- Group 4: Mixing errors ---

echo "## Group 4: Flag family mixing (all should fail to PASS)"

run_test_expect_fail 11 "target-cpu + mcpu" \
    mojo build --target-cpu=haswell --mcpu=skylake "$WORKDIR/test.mojo"

run_test_expect_fail 12 "target-cpu + march" \
    mojo build --target-cpu=haswell --march=x86-64 "$WORKDIR/test.mojo"

run_test_expect_fail 13 "mcpu + target-features" \
    mojo build --mcpu=haswell --target-features="+avx512f" "$WORKDIR/test.mojo"

echo ""

# --- Group 5: Emit options ---

echo "## Group 5: Emit options with cross-compilation"

run_test 14 "emit llvm cross-target" \
    mojo build \
        --target-triple x86_64-unknown-linux-gnu \
        --target-cpu x86-64 \
        --emit llvm -o "$WORKDIR/test14.ll" "$WORKDIR/test.mojo"

run_test 15 "emit llvm-bitcode cross-target" \
    mojo build \
        --target-triple x86_64-unknown-linux-gnu \
        --target-cpu x86-64 \
        --emit llvm-bitcode -o "$WORKDIR/test15.bc" "$WORKDIR/test.mojo"

run_test_expect_fail 16 "emit shared-lib cross-target (linker)" \
    mojo build \
        --target-triple x86_64-unknown-linux-gnu \
        --target-cpu x86-64 \
        --emit shared-lib -o "$WORKDIR/test16.so" "$WORKDIR/testlib.mojo"

run_test_expect_fail 17 "emit exe cross-target (linker)" \
    mojo build \
        --target-triple aarch64-unknown-linux-gnu \
        --target-cpu cortex-a72 \
        --emit exe -o "$WORKDIR/test17" "$WORKDIR/test.mojo"

echo ""

# --- Group 6: GPU/accelerator targets ---

echo "## Group 6: GPU/accelerator targets"

# This test checks for the sidecar file, not just compilation success.
# It requires a host with GPU support (Metal on Apple Silicon).
((total++))
if mojo build \
    --target-accelerator=metal:4 \
    --emit asm -o "$WORKDIR/test-gpu.s" "$WORKDIR/test-gpu.mojo" \
    > /dev/null 2>&1 && \
    [[ -f "$WORKDIR/test-gpu_kernel.ll" ]]; then
    echo "PASS  18  GPU asm + Metal sidecar"
    ((pass++))
else
    echo "FAIL  18  GPU asm + Metal sidecar"
    ((fail++))
fi

echo ""

# --- Summary ---

echo "==========================="
echo "$pass passed, $fail failed, $total total"

if [[ $fail -gt 0 ]]; then
    exit 1
fi
