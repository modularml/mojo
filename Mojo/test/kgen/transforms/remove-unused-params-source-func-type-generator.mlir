// RUN: kgen-opt --split-input-file --remove-unused-params --eliminate-dead-symbols -mlir-print-local-scope %s | FileCheck %s

// `RemoveUnusedParams` rewrites the live `funcTypeGenerator` (and the printed
// signature) to drop the unused argument, but must leave the
// `sourceFuncTypeGenerator` snapshot untouched. This is the load-bearing
// invariant behind the snapshot: it preserves the original, pre-pruning source
// signature. `RemoveUnusedParams` clones the op and rewrites only the live
// signature via targeted setters, so the snapshot survives verbatim.

// `%arg0` is unused, so it is pruned from the signature of the rewritten
// `@callee_REMOVED_ARG` — but the snapshot still carries both arguments.
// CHECK-LABEL: kgen.generator @callee_REMOVED_ARG(%arg0: !kgen.pointer<index>) -> index
// CHECK-SAME:    sourceFuncTypeGenerator = #kgen.type<(index, !kgen.pointer<index>) -> index> : !kgen.type
kgen.generator @callee(%arg0: index, %arg1: !kgen.pointer<index>) -> index attributes {sourceFuncTypeGenerator = #kgen.type<(index, !kgen.pointer<index>) -> index> : !kgen.type} {
  %l = pop.load %arg1 : !kgen.pointer<index>
  kgen.return %l : index
}

kgen.generator export @entry(%arg0: index, %arg1: !kgen.pointer<index>) {
  %0 = kgen.call @callee(%arg0, %arg1) : (index, !kgen.pointer<index>) -> index
  kgen.return
}
