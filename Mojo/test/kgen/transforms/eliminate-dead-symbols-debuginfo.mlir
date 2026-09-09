// RUN: kgen-opt %s -eliminate-dead-symbols -mlir-print-debuginfo -allow-unregistered-dialect | FileCheck %s

// Test that EliminateDeadSymbols preserves symbols that are only referenced
// inside debug info locations, not in regular op bodies/types/attrs.
// Regression test for MOCO-3440.

// @only_in_debug is a struct declaration referenced only in the DISubroutineType
// of the subprogram attached to @exported's location. With the fix it must
// survive EDS; without it, EDS removes it because it never appears in a
// reachable op's attribute dictionary or result/argument types.
// CHECK: @only_in_debug
lit.struct.decl @only_in_debug {
  lit.struct.field value : index
}

// @truly_dead is unreferenced everywhere (no debug info, no code). Always gone.
// CHECK-NOT: @truly_dead
lit.struct.decl @truly_dead {
  lit.struct.field value : index
}

// The subprogram type embeds !kgen.pointer<@only_in_debug> as an argument type.
// This creates a FlatSymbolRefAttr(@only_in_debug) inside the location attribute,
// but nowhere in the op's regular attribute dictionary or op types.
#subprogram = #debuginfo.subprogram<sourceName = <"exported">> :
    !debuginfo.subroutine<(!kgen.pointer<@only_in_debug>) -> (): DW_CC_normal>

#loc = loc("test.mlir":1:1)
#loc1 = loc(fused<#subprogram>[#loc])

// CHECK: @exported
kgen.func export @exported(%arg0: index) {
  kgen.return loc(#loc1)
} loc(#loc1)
