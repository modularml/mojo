// RUN: kgen-opt -split-input-file %s

// Verify that ConstantLike ops with a non-null mismatched scope do not trigger
// a verification error. ConstantLike ops are freely movable across scopes and
// so their scope should not matter / should not be verified.

#file = #debuginfo.file<"foo.mlir" in "/">
#subprogram = #debuginfo.subprogram<sourceName = <"foo">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#subprogram1 = #debuginfo.subprogram<sourceName = <"foo1">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#lexical_block = #debuginfo.lexical_block<scope = #subprogram1, file = #file, line = 104, column = 17>
#lexical_block1 = #debuginfo.lexical_block<scope = #lexical_block, file = #file, line = 120, column = 22>

#loc = loc("foo.mlir":7:8)
#loc1 = loc("foo.mlir":10:13)
#funcLoc = loc(fused<#subprogram>[#loc])

kgen.func @foo() {
  // kgen.param.constant has ConstantLike trait: a non-null mismatched scope
  // (from a different subprogram) must not trigger a verification error.
  %index1 = kgen.param.constant = <2> loc(fused<#lexical_block1>[#loc1])
  kgen.return loc(#funcLoc)
} loc(#funcLoc)
