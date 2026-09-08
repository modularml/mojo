# Post Parser Debugging

This document describes tools and tips to debug generated MLIR or LLVM IR.

## Tools

There are several internal tools in the Mojo compiler that can help you debug

### `kgen`

The `kgen` tool compiles Mojo program until some point that is controlled by the
option it takes:

- `-elaborate` stops compilation right after Elaboration (for the host)
  and prints result MLIR
- `-emit=llvm` stops compilation right after lowering of MLIR to LLVM IR
  (for the host) and prints result LLVM IR
- `-emit=llvm-bitcode` stops compilation right after lowering of MLIR to LLVM IR
  (for the host) and prints LLVM bitcode
- `-emit=llvm-opt` stops compilation after our custom LLVM optimization
  pipeline and prints result LLVM IR
- `-emit=llvm-opt-bitcode` stops compilation after our custom LLVM optimization
  pipeline and prints LLVM bitcode
- `-emit=asm` runs our custom LLVM optimization pipeline, backends and emit
  result assembly
- `-emit=asm-verbose` runs our custom LLVM optimization pipeline, backends and
  emit result assembly, with more information inlined
- `-emit=object` prints binary output
- others (see `--help`) that are less used for debugging

The tool also supports upstream's `mlir-opt` flags, such as
`--mlir-print-ir-after[-all|=<pass>]`, `--mlir-pass-statistics` etc.

Other 2 notable options are `--save-temps` and `--temps-dir=<dir>/<prefix>` that
needs to be used all together to save IR after certain points to files.

***NOTE***: It's essential to note that `--mlir-print-ir-[before|after]` does
only work for the "main" `PassManager`, i.e. that is built to construct the main
pipeline. Since Mojo compiler is focused on fast compilation time, many passes
are doing crazy parallelization by constructing own nested `PassManager`s that
don't inherit "main" `PassManager`'s options, like `applyPassManagerCLOptions`.
Therefore `--mlir-print-ir-[before|after]` won't print IR for these nested
`PassManager`s.

***NOTE***: To emphasize the NOTE above: `--mlir-print-ir-[before|after]` won't
print MLIR for the offload target.

### `kgen-opt`

The `kgen-opt` tool is similar to `mlir-opt` but for the Mojo compiler. It can
be used to run individual passes on a given MLIR.

***NOTE***: Passes that build own nested `PassManager`s and pipeline may require
to add extra argument to its CL option.

### `llvm-module-split`

The `llvm-module-split` tool allows testing of LLVM IR splitter that splits
LLVM IR to different modules.

### `kgen-llvm-opt`

The `kgen-llvm-opt` tool allows testing of our custom LLVM optimization
pipeline. Currently it doesn't allow to test individual passes, unlike
`opt`

### `kgen-reduce`

***NOT TESTED***

The tool should help to reduce MLIR to be able to easily debug it, but no one
seems to use it.

## Debugging

As always, first it's essential to find a guilty pass that causes a problem.
Unlike LLVM, Mojo compiler does not have `-opt-bisect-limit`-like option (yet).
Therefore it might require to manually play with the pipeline to find the
guilty pass.

***NOTE***: Be aware of nested `PassManager`s and their passes.

After the guilty pass is identified, IR can be dumped by using `kgen` tool.

***NOTE***: For offload compilation if MLIR before `KGEN -> LLVM` pass is
needed, for now it requires to manually add `op.dump()` into the pass and
possibly disable multithreading with `-workqueue=single-thread`

After IR on hand, `kgen-opt`, `opt` or other tool mentioned above can be
used to find the root cause of the problem

## Tips

- If program fails at runtime for the user, first try to build it with a
  `release` compiler (`br //:install --config=release`).

  *Explanation*: `production` (`br //:install --config=production`)
  build disables all verifications and asserts, therefore invalid MLIR can be
  generated

- Don't use `debug` (`br //:install --config=debug-modular` or
  `br //:install --config=debug-everything`) build for `kgen` tool if not
  debugging Mojo Parser

  *Explanation*: `debug` build is ***REALLY*** slow.

- There are a few pass-specific CL options that can be used with `mojo` tool by
  passing them through `KGEN_OPTIONS` env var:

```bash
KGEN_OPTIONS="-kgen-parametric-inline-threshold=35" mojo build test.mojo
```

  It requires to define `KGEN_ENABLE_PASS_OPTIONS` in `mojo`'s sources and
  rebuild it.
  Supported options are defined in `KGENPassCLOptions`.

- Rarely there are issues with LLVM module splitter that can result in
  compile or runtime problems. If you suspect it, try to disable parallel
  compilation.

- Build only tools you need for debugging. It's still essential to run
  `br //:install --config=....` first on a fresh workspace to properly setup
  symlinks, but after that it's usually enough to rebuild only tools you need.
  For example, `br //:kgen-tool //:kgen-opt` command can be used to only build
  `kgen` and `kgen-opt` tools

## Debugger

On MacOS `lldb` is the simplest debugger that can be used to debug Mojo.
On Linux, `gdb` or `lldb` can be used.

There are several ways to run debugger on Mojo's tools:

- Use `bd [--gdb] [--config=...] //<bazel-test>` command to run debugger on
  `<bazel-test>`
- Use `bd [--gdb] [--config=...] //:<tool> -- <options>` command to run
  debugger on `<tool>` with `<options>`. For example:

```bash
  bd --config=debug-modular --gdb KGEN/tools/mojo -- build \
              --target-accelerator="amdgpu:gfx942" test.mojo
```

- Use `lldb` or `gdb` directly to run debugger.
  **NOTE**: debuginfo contains relative path, therefore running this command
  outside of Mojo's workspace won't show any symbols in debugger.
  **NOTE**: If there's a desire to keep all tests outside of Mojo's workspace,
  following aliases can be used:

```bash
alias blldb='blldb_() {
  if [[ -z ${MODULAR_PATH} ]]; then
    echo "${MODULAR_PATH} must be set"
  fi
  args=();
  for arg in $@; do
    if [[ -f ${arg} || -d ${arg} ]]; then
      args+=($(readlink -f ${arg}))
    else
      args+=(${arg})
    fi
  done
  echo "(cd ${MODULAR_PATH}; lldb ${args})"
  (cd ${MODULAR_PATH}; lldb ${args})
  unset -f blldb_;
}
blldb_'

alias bgdb='bgdb_() {
  if [[ -z ${MODULAR_PATH} ]]; then
    echo "${MODULAR_PATH} must be set"
  fi
  args=();
  for arg in $@; do
    if [[ -f ${arg} || -d ${arg} ]]; then
      args+=($(readlink -f ${arg}))
    else
      args+=(${arg})
    fi
  done
  echo "(cd ${MODULAR_PATH}; gdb ${args})"
  (cd ${MODULAR_PATH}; gdb ${args})
  unset -f bgdb_;
}
bgdb_'
```

  which allows to stay within test directory that could be outside of the
  workspace. The aliases do have obvious rare corner cases.

## Apple GPU debugging

Since Apple GPU does not have LLVM upstream support, its support in Mojo comes
from reverse engineering.
This section contains common problems and how to debug them efficiently.

The best commands here are

- `kgen -elaborate --save-temps --temps-dir` to get a file containing
  generated LLVM IR before transformation to AIR and generated AIR file.
- `xcrun -sdk macosx air-*` to see how Metal compiler behaves on a given AIR
  file
- `kgen-llvm-opt -O3 -S` to get LLVM IR after transformation to AIR.
- `kgen-llvm-opt --disable-optimization-passes -O3` to get AIR file from LLVM IR
  without running any passes to transform LLVM IR to AIR-compatible LLVM IR.

- ```bash
   KGEN_OPTIONS="-kgen-object-compiler-use-custom-air=<your-air-file>" mojo \
                 build/run test.mojo
   ```

   to test your AIR file at runtime

- `llvm-reduce` to reduce LLVM IR

### Add NVidia/AMDGPU feature `X` to Apple GPU

That first requires understanding how Metal supports this. LLM (the best friend)
or Metal Shading Language spec can be used to write a metal kernel. Then metal
kernel can be compiled and its LLVM IR can be dumped to see how feature `X` is
implemented:

```bash
/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/metal \
        -mllvm -print-after-all test.metal &> dump
```

After that its implementation can be added to the compiler and/or stdlib.

### Invalid bitcode

Metal compiler still uses LLVM 17.0 as it needs typed pointers and 5.0 bitcode
reader. The Mojo compiler has its own bitcode writer that *should be* compatible
with 5.0, however there could be some differences.

If you see `metallib compilation failed with code` problem, it could indicate
that generated bitcode is invalid for Metal.

To find what is wrong:

```bash
kgen -elaborate test.mojo --save-temps --temps-dir=/tmp/apple_gpu
xcrun -sdk macosx air-objdump --disassemble /tmp/apple_gpu.*.air
```

that is supposed to print opcode that Metal compiler didn't like. If it's still
not clear what's a problem, try to reduce a test case using `llvm-reduce` with a
script

```bash
! kgen-llvm-opt --disable-optimization-passes -O3 $1 -o /tmp/kernel_$$.air || exit 1
xcrun -sdk macosx air-objdump --disassemble /tmp/kernel_$$.air \
          -o /tmp/metalllib$$.metallib &> /tmp/reduce_$$.log
grep -q "<the error you care about>" /tmp/reduce_$$.log || exit 1
```

and then

```bash
kgen-llvm-opt -O3 -S /tmp/apple_gpu.pre-split*.ll -o /tmp/apple_gpu.kernel.ll
llvm-reduce --test=reduce.sh /tmp/apple_gpu.kernel.ll
```

### Incorrect output

That problem is the most time consuming to debug. So far there were 2 problems
that resulted to incorrect output:

- incorrect address space on a `load` or `store`.
- Some `MTL::Buffer` was not passed to the encoder.

The best option here is to manually play with AIR file to find out which
instruction(s) are incorrect:

```bash
kgen -elaborate test.mojo --save-temps --temps-dir=/tmp/apple_gpu
kgen-llvm-opt  -O3 /tmp/apple_gpu.pre-split.*.ll -S -o /tmp/kernel.ll
```

Now `/tmp/kernel.ll` contains LLVM IR transformed to AIR-compatible LLVM IR.
After that the following commands can be used to test it at runtime

```bash
kgen-llvm-opt --disable-optimization-passes -O3 /tmp/kernel.ll \
        -o /tmp/apple_gpu.air
clear-cache
KGEN_OPTIONS="-kgen-object-compiler-use-custom-air=/tmp/apple_gpu.air" \
              mojo run test.mojo
```

Now it's about to manually modify AIR and find out which instruction(s),
attributes, metadata etc are incorrect.

### `Failed to create compute pipeline state`

That is the most unclear problem so far. The problem occurs when
AsyncRT tries to get number of arguments for which `MTL:Buffer` has to be
passed. That means that Metal compiler finds generated AIR file as good, but
something still goes wrong at reflection time.

The best option so far is to use combination of steps from
[Invalid bitcode](#invalid-bitcode) and [Incorrect output](#incorrect-output)
sections with more complicated script for `llvm-reduce`.

***NOTE*** Since it's required to run a program, some tests may have infinite
recursion or loop during `llvm-reduce`, so make sure to put a timeout when
running a binary.
