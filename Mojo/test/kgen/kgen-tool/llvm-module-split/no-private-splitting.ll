; RUN: llvm-module-split %s --per-func -debug 2> %t | FileCheck %s
; RUN: cat %t | FileCheck --check-prefix=DUMP %s

; REQUIRES: ASSERTIONS

; COM: At this moment we do not support splitting of private symbols as they
; COM: could be wrongly linked

; DUMP: global .memset_pattern cannot be split

; CHECK-NOT: @.memset_pattern = external.*
; CHECK: @.memset_pattern_0 = weak dso_local constant [1 x float] zeroinitializer
; CHECK-NOT: @.memset_pattern = external.*

@image_info = external global ptr
@.memset_pattern = private constant [1 x float] zeroinitializer

define void @foo() {
entry:
  store ptr null, ptr @image_info, align 8
  ret void
}

define void @bench_main() {
entry:
  call void null(ptr @image_info, ptr @.memset_pattern, i64 0)
  ret void
}

define i32 @main() {
  tail call void @bench_main()
  ret i32 0
}
