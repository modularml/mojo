// RUN: kgen-opt %s -lower-semantic-cf -verify-parameters -verify-diagnostics -allow-unregistered-dialect | FileCheck %s

lit.fn @my_abort() -> !kgen.never {
  kgen.unreachable
}

// CHECK-LABEL: lit.struct.decl @SomeStruct
lit.struct.decl @SomeStruct {
  // CHECK-LABEL: lit.fn @dead_returns
  lit.fn @dead_returns(%c: !kgen.scalar<bool>, %a: i32, %b: i32) -> i32 {
    // CHECK: hlcf.if %c
    hlcf.if %c {
      // CHECK-NEXT: kgen.return %b : i32
      lit.return %b: i32
      lit.return %a: i32 // expected-warning {{unreachable code after return statement}}
      hlcf.yield
    // CHECK-NEXT: else
    } else {
      hlcf.yield
    }
    // CHECK: kgen.return %a : i32
    lit.return %a : i32
    lit.return %b : i32 // expected-warning {{unreachable code after return statement}}
    lit.end_fn
  // CHECK-NEXT: }
  }

  // Derived from MOCO-2978.
  // CHECK-LABEL: lit.fn @calls_unreachable_in_deinit
  lit.fn @calls_unreachable_in_deinit[mut *"self"](%self: !lit.ref<!lit.struct<@SomeStruct>, mut *"self"> deinit_mem) -> !kgen.none {
    // CHECK: lit.call tail @my_abort()
    // CHECK: kgen.unreachable {isAfterUnreachableCall = true}
    %0 = lit.call tail @my_abort() : !lit.generator<() -> !kgen.never>
    lit.end_fn
  }
}

// CHECK-LABEL: lit.file_module @FileModule
lit.file_module @FileModule {
  // CHECK-LABEL: lit.struct.decl @SomeStruct
  lit.struct.decl @SomeStruct {
    // CHECK-LABEL: lit.fn @try_and_raise
    lit.fn @try_and_raise(%a: i32) throws {
      // CHECK-NEXT: lit.try
      lit.try {
        // CHECK-NEXT: lit.try.raise
        lit.raise
        lit.try.yield
      // CHECK-NEXT: except
      } except {
        // CHECK-NEXT: kgen.return
        lit.return
        lit.try.yield
      // CHECK-NEXT: else
      } else {
        // CHECK-NEXT: kgen.unreachable

        // expected-warning @+1 {{'else' logic in 'try' is unreachable}}
        lit.return
        lit.try.yield
      // CHECK-NEXT: }
      } finally {
        lit.try.yield
      }

      // CHECK-NEXT: kgen.unreachable
      // expected-warning @+1 {{unreachable code after try statement that doesn't fall through}}
      lit.return
      lit.end_fn
    }
  }

  // CHECK-LABEL: lit.fn @break_and_continue
  lit.fn @break_and_continue(%c: !kgen.scalar<bool>) {
    // CHECK-NEXT: hlcf.loop
    // CHECK-NEXT: hlcf.if %c {
    // CHECK-NEXT:   hlcf.yield
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   hlcf.break
    // CHECK-NEXT: }
    lit.loop {
      hlcf.if %c {
        hlcf.yield
      } else {
        lit.loop.break.else
      }

      // CHECK-NEXT: hlcf.if %c {
      hlcf.if %c {
        // CHECK-NEXT: hlcf.break
        lit.break
        lit.continue // expected-warning {{unreachable code after break statement}}
        hlcf.yield
      // CHECK-NEXT: else
      } else {
        // CHECK-NEXT: hlcf.continue
        lit.continue
        lit.break  // expected-warning {{unreachable code after continue statement}}
        hlcf.yield
      // CHECK-NEXT: }
      }
      // CHECK-NEXT: kgen.unreachable
      // CHECK-NEXT: }
      lit.return  // expected-warning {{unreachable code after if statement with then/else that do not fall through}}
      lit.loop.continue
    } else {
      lit.loop.yield
    }

    // CHECK-NEXT: kgen.return
    lit.return
    lit.end_fn
  }
}

// CHECK-LABEL: lit.fn @no_return
lit.fn @no_return() -> !kgen.none {
  // CHECK: kgen.return
  %0 = kgen.param.constant: none = <#kgen.none>
  lit.return %0 :  !kgen.none
  lit.end_fn
}

lit.fn @if_true_return() -> index {
  %0 = index.constant 0
  %true = kgen.param.constant: scalar<bool> = <true>
  hlcf.if %true {
    lit.return %0 : index
    hlcf.yield
  } else {
    // expected-warning @+1 {{unreachable code after 'if True'}}
    lit.return %0 : index
    hlcf.yield
  }
  lit.end_fn
}

lit.fn @while_true() -> index {
  %true = kgen.param.constant: scalar<bool> = <true>
  lit.loop {
    hlcf.if %true {
      hlcf.yield
    } else {
      lit.loop.break.else
    }

    hlcf.if %true {
      lit.continue
      hlcf.yield
    } else {
      hlcf.yield
    }
    lit.break // expected-warning {{unreachable code after if statement with then/else that do not fall through}}
    lit.loop.continue
  } else {
    lit.loop.yield
  }
  lit.end_fn
}

// CHECK-LABEL: lit.fn @if_false_raise
lit.fn @if_false_raise() throws -> !kgen.scalar<bool> {
  %false = kgen.param.constant: scalar<bool> = <false>
  hlcf.if %false {
    hlcf.yield
  // CHECK: else
  } else {
    // CHECK-NEXT: [[TRUE:%.*]] = kgen.param.constant: scalar<bool> = <true>
    // CHECK-NEXT: lit.error_return [[TRUE]]
    lit.raise
    hlcf.yield
  }
  lit.end_fn
}

// CHECK-LABEL: lit.fn @for_else_raise
lit.fn @for_else_raise() throws -> !kgen.scalar<bool> {
  lit.loop {
    %cond = "foo"() : () -> i1
    %cond_sb = pop.cast_from_builtin %cond : i1 to !kgen.scalar<bool>
    hlcf.if %cond_sb {
      hlcf.yield
    } else {
      lit.loop.break.else
    }
    lit.loop.continue
  // CHECK: else
  } else {
    // CHECK-NEXT: [[TRUE:%.*]] = kgen.param.constant: scalar<bool> = <true>
    // CHECK-NEXT: lit.error_return [[TRUE]]
    lit.raise
    lit.loop.yield
  }
  lit.end_fn
}

// CHECK-LABEL: lit.fn @raise_raise
lit.fn @raise_raise() throws {
  // CHECK: lit.try
  lit.try {
    // CHECK: lit.try.raise
    lit.raise
    lit.try.yield
  // CHECK-NEXT: except
  } except {
    // CHECK-NEXT: kgen.return
    lit.return
    lit.try.yield
  // CHECK-NEXT: else
  } else {
    lit.try.yield
  // CHECK-NOT: finally
  } finally {
    lit.try.yield
  }

  lit.end_fn
}

// CHECK-LABEL: lit.fn @throwing_func
lit.fn @throwing_func[mut elt, mut lt](
    %0[*""]: !lit.ref<@Error, mut elt> byref_error,
    %1[*""]: !lit.ref<none, mut *[0,1]> byref_result
) throws -> !kgen.scalar<bool> {
  // CHECK-NEXT: [[TRUE:%.*]] = kgen.param.constant: scalar<bool> = <true>
  // CHECK-NEXT: lit.error_return [[TRUE]]
  lit.raise
  lit.end_fn
}

lit.struct.decl @Error {}

// CHECK-LABEL: lit.fn @throwing_calls
lit.fn @throwing_calls(
    %f: !lit.generator<[2](!lit.ref<@Error, mut *[0,0]> byref_error, !lit.ref<none, mut *[0,1]> byref_result) throws -> !kgen.scalar<bool>>
) throws -> !kgen.scalar<bool> {
  %err = lit.var.decl "err" synth : !lit.ref<@Error, mut elt>
  %result = lit.var.decl "result" synth : !lit.ref<none, mut lt>

  // CHECK:      [[IS_ERR:%.*]] = lit.call @throwing_func
  // CHECK-NEXT: hlcf.if [[IS_ERR]]
  // CHECK-NEXT:   mark_consumed %result
  // CHECK-NEXT:   [[TRUE:%.*]] = kgen.param.constant: scalar<bool> = <true>
  // CHECK-NEXT:   lit.error_return [[TRUE]]
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   mark_consumed %err
  // CHECK-NEXT:   yield
  // CHECK-NEXT: }
  lit.call @throwing_func[mut elt, mut lt](%err, %result) : !lit.generator<[2](!lit.ref<@Error, mut *[0,0]> byref_error, !lit.ref<none, mut *[0,1]> byref_result) throws -> !kgen.scalar<bool>>

  %error = lit.var.decl "error" synth : !lit.ref<@Error, mut tlt>
  // CHECK: lit.try "try0" {
  lit.try %error : !lit.ref<@Error, mut tlt> {
    // CHECK-NEXT: [[IS_ERR:%.*]] = lit.call_indirect %f
    // CHECK-NEXT: hlcf.if [[IS_ERR]]
    // CHECK-NEXT:   mark_consumed %result
    // CHECK-NEXT:   lit.try.raise "try0"
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   mark_consumed %error
    // CHECK-NEXT:   yield
    // CHECK-NEXT: }
    lit.call_indirect %f[mut tlt, mut lt](%error, %result) :  !lit.generator<[2](!lit.ref<@Error, mut *[0,0]> byref_error, !lit.ref<none, mut *[0,1]> byref_result) throws -> !kgen.scalar<bool>>
    lit.try.yield
  } except {
    lit.try.yield
  } else {
    lit.try.yield
  } finally {
    lit.try.yield
  }
  kgen.unreachable
}

// CHECK-LABEL: lit.fn @unreachable_try
lit.fn @unreachable_try() {
  lit.try {
    lit.try.yield
  } except {
    // expected-warning @+1 {{'except' logic is unreachable, try doesn't raise an exception}}
    index.constant 0
    lit.try.yield
  } else {
    lit.return
    lit.try.yield
  } finally {
    lit.try.yield
  }
  // CHECK: kgen.unreachable
  // expected-warning @+1 {{unreachable code after try statement that doesn't fall through}}
  index.constant 0
  lit.end_fn
}

// CHECK-LABEL: lit.fn @suppressed_try
lit.fn @suppressed_try() {
  lit.try {
    lit.try.yield
  } except {
    // CHECK: except
    // CHECK-NEXT: kgen.unreachable
    index.constant 0
    lit.try.yield
  } else {
    index.constant 0
    lit.return
    lit.try.yield
  } finally {
    lit.try.yield
  } {"suppressWarnings" = true}
  // CHECK: kgen.unreachable
  // expected-warning @+1 {{unreachable code after try statement that doesn't fall through}}
  index.constant 0
  lit.end_fn
}

// CHECK-LABEL: lit.fn @coroutine() async -> index
lit.fn @coroutine() async -> index {
  %idx0 = index.constant 0
  // CHECK: return %idx0
  lit.return %idx0 : index
  lit.end_fn
}

// CHECK-LABEL: lit.fn @call_coroutine
// CHECK-SAME: coro: () async -> !kgen.none
// CHECK-SAME: ) async -> !kgen.none
lit.fn @call_coroutine<coro: () async -> !kgen.none>() async -> !kgen.none {
  // CHECK-NEXT: lit.async.call[() async -> !kgen.none: coro]()
  lit.async.call[() async -> !kgen.none: coro]()
  %0 = kgen.param.constant: none = <#kgen.none>
  lit.return %0 :  !kgen.none
  lit.end_fn
}

// CHECK-LABEL: lit.fn @return_after_return
lit.fn @return_after_return() -> !kgen.none {
  %0 = kgen.param.constant: none = <#kgen.none>
  // CHECK: kgen.return %none : !kgen.none
  lit.return %0 : !kgen.none
  %1 = kgen.param.constant: scalar<bool> = <true>  // expected-warning {{unreachable code after return statement}}
  hlcf.if %1 {
    %2 = kgen.param.constant: none = <#kgen.none>
    lit.return %2 : !kgen.none
    hlcf.yield
  } else {
    hlcf.yield
  }
  lit.end_fn
}

// CHECK-LABEL: lit.fn @if_else_return
lit.fn @if_else_return(%cond: !kgen.scalar<bool>) -> index {
  %0 = index.constant 0
  hlcf.if %cond {
    lit.return %0 : index
    hlcf.yield
  } else {
    lit.return %0 : index
    hlcf.yield
  }
  // CHECK: kgen.unreachable
  lit.end_fn
}

// CHECK-LABEL: lit.fn @if_else_raise
lit.fn @if_else_raise[mut elt, mut lt](%cond: !kgen.scalar<bool>,
    %error[*""]: !lit.ref<@Error, mut elt> byref_error,
    %result[*""]: !lit.ref<none, mut lt> byref_result
) throws -> !kgen.scalar<bool> {
  %0 = kgen.param.constant: scalar<bool> = <false>
  hlcf.if %cond {
    lit.return %0 : !kgen.scalar<bool>
    hlcf.yield
  } else {
    lit.call @throwing_func[mut elt, mut lt](%error, %result) : !lit.generator<[2](!lit.ref<@Error, mut *[0,0]> byref_error, !lit.ref<none, mut *[0,1]> byref_result) throws -> !kgen.scalar<bool>>
    lit.raise
    hlcf.yield
  }
  // CHECK: kgen.unreachable
  lit.end_fn
}

// CHECK-LABEL: lit.fn @coroutine2
lit.fn @coroutine2() async -> index {
  %0 = index.constant 0
  %true = kgen.param.constant: scalar<bool> = <true>

  lit.loop  {
    hlcf.if %true {
      hlcf.yield
    } else {
      lit.loop.break.else
    }

    lit.return %0 : index
    lit.break  // expected-warning {{unreachable code after return statement}}
    lit.loop.continue
  } else {
    lit.loop.yield
  }

  // CHECK: kgen.unreachable
  lit.end_fn
}

// CHECK-LABEL: lit.fn @pointlessTry
lit.fn @pointlessTry() -> !kgen.none {
  lit.try { // expected-warning {{try body doesn't raise an exception}}
    lit.try.yield
  } except {
    lit.try.yield
  } else {
    lit.try.yield
  } finally {
    lit.try.yield
  }
  %0 = kgen.param.constant: none = <#kgen.none>
  lit.return %0 :  !kgen.none
  lit.end_fn
}

// CHECK-LABEL: lit.fn @reraise_in_try
lit.fn @reraise_in_try() {
  // CHECK-NEXT: lit.try
  lit.try {
    // CHECK-NEXT: lit.try
    lit.try {
      // CHECK-NEXT: lit.try.raise
      lit.raise
      lit.try.yield
    // CHECK-NEXT: except
    } except {
      // CHECK-NEXT: lit.try.raise
      lit.raise
      lit.try.yield
    // CHECK-NEXT: else
    } else {
      // CHECK-NEXT: unreachable
      lit.try.yield
    // CHECK-NOT: finally
    } finally {
      lit.try.yield
    }
    // CHECK: unreachable
    lit.try.yield
  // CHECK-NEXT: except
  } except {
    // CHECK-NEXT: yield
    lit.try.yield
  // CHECK-NEXT: else
  } else {
    // CHECK-NEXT: unreachable
    lit.try.yield
  } finally {
    lit.try.yield
  }
  kgen.return
}

// CHECK-LABEL: lit.fn @nested_try_inner_catch
lit.fn @nested_try_inner_catch() {
  // CHECK-NEXT: lit.try
  lit.try {
    %err = lit.var.decl "err" synth : !lit.ref<@Error, mut elt>
    %result = lit.var.decl "result" synth : !lit.ref<none, mut lt>
    // CHECK: lit.call @throwing_func
    lit.call @throwing_func[mut elt, mut lt](%err, %result) : !lit.generator<[2](!lit.ref<@Error, mut *[0,0]> byref_error, !lit.ref<none, mut *[0,1]> byref_result) throws -> !kgen.scalar<bool>>
    // CHECK-NEXT: hlcf.if
    // CHECK-NEXT:   lit.ownership.mark_consumed %result
    // CHECK-NEXT:   lit.try.raise
    // CHECK-NEXT: else
    // CHECK-NEXT:   lit.ownership.mark_consumed %err
    // CHECK-NEXT:   yield

    // CHECK: lit.try
    lit.try {
      // CHECK-NEXT: lit.try.raise
      lit.raise
      lit.try.yield
    // CHECK-NEXT: except
    } except {
      // CHECK-NEXT: kgen.return
      lit.return
      lit.try.yield
    // CHECK-NEXT: else
    } else {
      // CHECK-NEXT: kgen.unreachable
      lit.try.yield
    } finally {
      lit.try.yield
    }
    // CHECK: kgen.unreachable
    lit.try.yield
  // CHECK-NEXT: except
  } except {
    // CHECK-NEXT: lit.try.yield
    lit.try.yield
  // CHECK-NEXT: else
  } else {
    // CHECK-NEXT: kgen.unreachable
    lit.try.yield
  } finally {
    lit.try.yield
  }
  // CHECK: kgen.return
  kgen.return
}

// CHECK-LABEL: lit.fn @finally_breaks
lit.fn @finally_breaks() -> index {
  // CHECK-LABEL: lit.try
  lit.try {
    // CHECK-NEXT: lit.try.yield
    lit.try.yield
  // CHECK-NEXT: except
  } except (%e: index) {
    // CHECK-NEXT: unreachable
    lit.try.yield
  // CHECK-NEXT: else
  } else {
    // CHECK: kgen.return %idx0
    lit.try.yield
  // CHECK-NOT: finally
  } finally {
    %idx0 = index.constant 0
    lit.return %idx0 : index
    lit.try.yield
  }
  // CHECK: kgen.unreachable
  lit.end_fn
}

// CHECK-LABEL: lit.fn @try_finally
lit.fn @try_finally(%arg0: !kgen.scalar<bool>, %arg1: i32, %arg2: i64) -> (i32, i64) {
  %true = kgen.param.constant: scalar<bool> = <true>

  // CHECK: hlcf.loop "_loop_0" {
  // CHECK-NEXT: hlcf.if %simd {
  // CHECK-NEXT:         hlcf.yield
  // CHECK-NEXT:       } else {
  // CHECK-NEXT:         kgen.unreachable
  // CHECK-NEXT:       }
  lit.loop {
    hlcf.if %true {
      hlcf.yield
    } else {
      lit.loop.break.else
    }

    // CHECK-NEXT: lit.try
    lit.try {
      // CHECK-NEXT: hlcf.if %arg0
      hlcf.if %arg0 {
        // CHECK: clean.up
        // CHECK-NEXT: break
        hlcf.break
      // CHECK-NEXT: else
      } else {
        // CHECK-NEXT: yield
        hlcf.yield
      }
      // CHECK: clean.up
      // CHECK-NEXT: return %arg1, %arg2
      kgen.return %arg1, %arg2 : i32, i64
    // CHECK-NEXT: except
    } except (%err: index) {
      // CHECK-NEXT: unreachable
      lit.try.yield
    // CHECK-NEXT: else
    } else {
      // CHECK-NEXT: unreachable
      lit.try.yield
    // CHECK-NOT: finally
    } finally {
      "clean.up"() : () -> ()
      lit.try.yield
    }
    // CHECK: unreachable
    lit.break // expected-warning {{unreachable code after try statement that doesn't fall through}}
    lit.loop.continue
  } else {
    lit.loop.yield
  }
  // CHECK: return %arg1, %arg2
  kgen.return %arg1, %arg2 : i32, i64
}

// CHECK-LABEL: lit.fn @try_finally_return
lit.fn @try_finally_return(%arg0: index, %arg1: index, %arg2: !kgen.scalar<bool>) -> index {
  %true = kgen.param.constant: scalar<bool> = <true>

  // CHECK: hlcf.loop "_loop_0" {
  // CHECK-NEXT: hlcf.if %simd {
  // CHECK-NEXT:         hlcf.yield
  // CHECK-NEXT:       } else {
  // CHECK-NEXT:         kgen.unreachable
  // CHECK-NEXT:       }

  lit.loop {
    hlcf.if %true {
      hlcf.yield
    } else {
      lit.loop.break.else
    }

    // CHECK-NEXT: lit.try
    lit.try {
      // CHECK-NEXT: hlcf.if %arg2
      hlcf.if %arg2 {
        // CHECK-NEXT: return %arg1
        hlcf.break
      // CHECK-NEXT: else
      } else {
        // CHECK-NEXT: return %arg1
        hlcf.continue
      }
      // CHECK: unreachable
      kgen.return %arg0 : index
    } except (%err: index) {
      lit.try.yield
    } else {
      lit.try.yield
    } finally {
      kgen.return %arg1 : index
    }
    lit.break // expected-warning {{unreachable code after try statement that doesn't fall through}}
    lit.loop.continue
  } else {
    lit.loop.yield
  }

  kgen.return %arg1 : index
}

// CHECK-LABEL: lit.fn @nested_try_finally
lit.fn @nested_try_finally() {
  // CHECK-NEXT: lit.try
  lit.try {
    // CHECK-NEXT: lit.try
    lit.try {
      // CHECK-NEXT: clean.up0
      // CHECK-NEXT: clean.up1
      // CHECK-NEXT: return
      kgen.return
    } except (%err: index) {
      lit.try.yield
    // CHECK: else
    } else {
      lit.try.yield
    } finally {
      "clean.up0"() : () -> ()
      lit.try.yield
    }
    lit.try.yield
  } except (%err: index) {
    lit.try.yield
  // CHECK: else
  } else {
    // CHECK-NEXT: unreachable
    lit.try.yield
  } finally {
    "clean.up1"() : () -> ()
    lit.try.yield
  }
  // CHECK: unreachable
  kgen.return
}

// CHECK-LABEL: lit.fn @try_in_loop
lit.fn @try_in_loop(%arg0: !kgen.scalar<bool>) {
  lit.loop {
    hlcf.if %arg0 {
      hlcf.yield
    } else {
      lit.loop.break.else
    }

    lit.try {
      lit.try.yield
    // CHECK: except
    } except (%e: index) {
      // CHECK-NEXT: kgen.unreachable
      lit.try.yield
    } else {
      lit.try.yield
    } finally {
      lit.try.yield
    } {"suppressWarnings" = true}
    // CHECK: hlcf.continue
    lit.loop.continue
  } else {
    lit.loop.yield
  }
  // CHECK: after.loop
  "after.loop"() : () -> ()
  kgen.return
}

// CHECK-LABEL: lit.fn @recurse
// CHECK-SAME (%x: !kgen.scalar<index>) -> !kgen.scalar<index> {
// CHECK-NEXT: %0 = kgen.call @recurse(%x) : !lit.generator<("x": !kgen.scalar<index>) -> !kgen.scalar<index>>
// CHECK-NEXT: kgen.return %0 : !kgen.scalar<index>
// CHECK-NEXT:}
lit.fn @recurse(%x: !kgen.scalar<index>) -> !kgen.scalar<index> {
  %0 = kgen.call @recurse(%x) : !lit.generator<("x": !kgen.scalar<index>) -> !kgen.scalar<index>>
  lit.return %0 : !kgen.scalar<index>
  lit.end_fn
}

// CHECK-LABEL: lit.fn @coroutine_await
lit.fn @coroutine_await(%arg0: !kgen.scalar<bool>) {
  // CHECK-NEXT: co.suspend
  co.suspend (%hdl0) {
    hlcf.if %arg0 {
      // CHECK: kgen.return
      lit.return
      hlcf.yield
    } else {
      hlcf.yield
    }
    // CHECK: co.suspend.end
    co.suspend.end
  }
  lit.return
  lit.end_fn
}

// CHECK-LABEL: lit.fn @loop_with_else
lit.fn @loop_with_else(%arg0: !kgen.scalar<bool>) {
  // CHECK: hlcf.loop "_loop_0"
  lit.loop {
    hlcf.if %arg0 {
      hlcf.yield
    } else {
      lit.loop.break.else
    }

    lit.loop {
      hlcf.if %arg0 {
        hlcf.yield
      } else {
        lit.loop.break.else
      }

      // CHECK: hlcf.if %arg0 {
      // CHECK-NEXT:   hlcf.yield
      // CHECK-NEXT: } else {
      // CHECK-NEXT:   hlcf.break
      // CHECK-NEXT: }
      // CHECK-NEXT: hlcf.loop "_loop_1" {
      // CHECK-NEXT:   hlcf.if %arg0 {
      // CHECK-NEXT:     hlcf.yield
      // CHECK-NEXT:   } else {
      // CHECK-NEXT:     hlcf.continue "_loop_0"
      // CHECK-NEXT:   }
      // CHECK-NEXT:   hlcf.continue
      // CHECK-NEXT: }
      // CHECK-NEXT: kgen.unreachable
      lit.loop.continue
    } else {
      lit.continue
      lit.break     // expected-warning {{unreachable code after continue statement}}
      lit.loop.yield
    }
    lit.loop.continue
  } else {
    lit.loop.yield
  }

  lit.return
  lit.end_fn
}

// CHECK-LABEL: lit.trait.decl @Trait
lit.trait.decl @Trait {
  lit.fn @trait_fn() {
    kgen.unreachable
  }
}

// CHECK-LABEL: lit.fn @loop_with_cond_raise
// Crash handling exception
// https://github.com/modularml/modular/issues/27937
// Checking the loop body clobbered the "can raise" flag for the try block.
lit.fn @loop_with_cond_raise(%cond: !kgen.scalar<bool>) {
  lit.try {
    hlcf.if %cond {
      lit.raise
      hlcf.yield
    } else {
      hlcf.yield
    }

    lit.loop {
      hlcf.if %cond {
        hlcf.yield
      } else {
        hlcf.break
      }

      hlcf.if %cond {
        hlcf.yield
      } else {
        hlcf.break
      }
      lit.loop.continue
    } else {
      lit.loop.yield
    }
    lit.try.yield
  // CHECK: } except {
  } except {
    // CHECK-NEXT: kgen.return
    lit.return
    kgen.unreachable
  } else {
    lit.try.yield
  } finally {
    lit.try.yield
  }
  lit.return
  lit.end_fn
}

// [QoI] Generate error for obviously self recursive functions
// https://github.com/modular/mojo/issues/222
lit.fn @self_recursive() -> !kgen.none {
  // expected-warning @+1 {{self recursive call will cause an infinite loop}}
  %0 = lit.call @self_recursive() : !lit.generator<() -> !kgen.none>
  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_fn
}
lit.fn @self_recursive_arg(%a: index, %cond: i1) -> !kgen.none {
  // expected-warning @+1 {{self recursive call will cause an infinite loop}}
  %0 = lit.call @self_recursive_arg(%a, %cond) : !lit.generator<("a": index, "cond": i1) -> !kgen.none>
  %cond_sb = pop.cast_from_builtin %cond : i1 to !kgen.scalar<bool>
  hlcf.if %cond_sb {
    %4 = kgen.param.constant: index = <1>
    %5 = index.sub %a, %4
    // No warning.
    %6 = lit.call @self_recursive_arg(%5, %cond) : !lit.generator<("a": index, "cond": i1) -> !kgen.none>
    hlcf.yield
  } else {
    hlcf.yield
  }
  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_fn
}

lit.fn @self_recursive_param<a: index, cond: scalar<bool>>() -> !kgen.none attributes {sourceName = "self_recursive_param", specialFnKind = 0 : i8} {
  // expected-warning @+1 {{self recursive call will cause an infinite loop}}
  %0 = lit.call @self_recursive_param<a, :scalar<bool> cond>() : !lit.generator<() -> !kgen.none>
  kgen.param.if <cond> {
    // No warning.
    %1 = lit.call @self_recursive_param<a, :scalar<bool> cond>() : !lit.generator<() -> !kgen.none>
    kgen.param.yield
  } else {
    kgen.param.yield
  }
  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_fn
}

// #28551: Should report infinite recursion on this testcase
lit.fn @self_recursive_arg_diff(%a: index) -> !kgen.none {
  %one = kgen.param.constant: index = <1>
  %b = index.sub %a, %one
  // expected-warning @+1 {{self recursive call will cause an infinite loop}}
  lit.call @self_recursive_arg_diff(%b) : !lit.generator<("a": index) -> !kgen.none>

  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_fn
}

// CHECK-LABEL: lit.fn @elif
// CHECK-NEXT: %idx0 = index.constant 0
// CHECK-NEXT: %idx1 = index.constant 1
// CHECK-NEXT: %idx2 = index.constant 2
// CHECK-NEXT: [[V1:%*.]] = index.cmp eq(%arg0, %idx0)
// CHECK-NEXT: [[V1SB:%.*]] = pop.cast_from_builtin [[V1]] : i1 to !kgen.scalar<bool>
// CHECK-NEXT: [[V0:%.*]] = hlcf.elif [[V1SB]] -> index {
// CHECK-NEXT: hlcf.yield %arg0 : index
// CHECK-NEXT: } else {
// CHECK-NEXT: [[V2:%*.]] = index.cmp eq(%arg0, %idx1)
// CHECK-NEXT: [[V2SB:%.*]] = pop.cast_from_builtin [[V2]] : i1 to !kgen.scalar<bool>
// CHECK-NEXT: hlcf.elif.yield [[V2SB]]
// CHECK-NEXT: } then {
// CHECK-NEXT: kgen.return %arg1 : index
// CHECK-NEXT: } else {
// CHECK-NEXT: kgen.return %arg1 : index
// CHECK-NEXT: }
lit.fn @elif(%arg0: index, %arg1: index, %arg2: index) -> index {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx2 = index.constant 2
  %c0 = index.cmp eq(%arg0, %idx0)
  %c0_sb = pop.cast_from_builtin %c0 : i1 to !kgen.scalar<bool>
  %0 = hlcf.elif %c0_sb -> index {
    hlcf.yield %arg0 : index
  } else {
    %c = index.cmp eq(%arg0, %idx1)
    %c_sb = pop.cast_from_builtin %c : i1 to !kgen.scalar<bool>
    hlcf.elif.yield %c_sb
  } then {
    lit.return %arg1 : index
    hlcf.yield %arg1 : index
  } else {
    lit.return %arg1 : index
    hlcf.yield %arg2 : index
  }
  kgen.return %0 : index
}


// COM: https://github.com/modularml/modular/issues/33570
// COM: When cloning the finally block, we must uniquely mangle parameters to
// COM: avoid duplicate parameter name errors.
// CHECK-LABEL: lit.fn @mangle_params_finally_1
lit.fn @mangle_params_finally_1<x>(%c: !kgen.scalar<bool> imm) -> !kgen.none {
  lit.try {
    // CHECK: hlcf.if %c
    hlcf.if %c {
      %none_0 = kgen.param.constant: none = <#kgen.none>
      // CHECK: lit.alias.decl *"y`"
      // CHECK-NEXT: kgen.return
      lit.return %none_0 : !kgen.none
      hlcf.yield
    // CHECK-NEXT: } else {
    } else {
      // CHECK-NEXT: hlcf.yield
      hlcf.yield
    }
    lit.try.yield
  // CHECK: } except
  } except {
    // CHECK-NEXT: kgen.unreachable
    lit.try.yield
  // CHECK-NEXT: } else {
  } else {
    // CHECK-NEXT: lit.alias.decl *"y`f0"
    // CHECK-NEXT: lit.try.yield
    lit.try.yield
  } finally {
    lit.alias.decl *"y`" = <x>
    lit.try.yield
  }
  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_fn
}


// CHECK-LABEL: lit.fn @mangle_params_finally_2
lit.fn @mangle_params_finally_2<x>(%c: !kgen.scalar<bool> imm) -> !kgen.none {
  lit.try {
    // CHECK: hlcf.if %c
    hlcf.if %c {
      %none_1 = kgen.param.constant: none = <#kgen.none>
      // CHECK: lit.alias.decl *"y`"
      // CHECK-NEXT: kgen.return
      lit.return %none_1 : !kgen.none
      hlcf.yield
    } else {
      hlcf.yield
    }

    // CHECK: hlcf.if %c
    hlcf.if %c {
      %none_1 = kgen.param.constant: none = <#kgen.none>
      // CHECK: lit.alias.decl *"y`f0"
      // CHECK-NEXT: kgen.return
      lit.return %none_1 : !kgen.none
      hlcf.yield
    } else {
      hlcf.yield
    }

    %none_0 = kgen.param.constant: none = <#kgen.none>
    // CHECK: lit.alias.decl *"y`f1"
    // CHECK: kgen.return
    lit.return %none_0 : !kgen.none
    lit.try.yield
  // CHECK: } except
  } except {
    // CHECK-NEXT: kgen.unreachable
    lit.try.yield
  // CHECK-NEXT: } else {
  } else {
    // CHECK-NEXT: kgen.unreachable
    lit.try.yield
  } finally {
    lit.alias.decl *"y`" = <x>
    lit.try.yield
  }
  // CHECK: kgen.unreachable

  // expected-warning @+1 {{unreachable code after try statement that doesn't fall through}}
  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_fn
}


// CHECK-LABEL: lit.fn @mangle_params_finally_3
lit.fn @mangle_params_finally_3<x>(%c: !kgen.scalar<bool> imm) -> !kgen.none {
  lit.try {
    // CHECK: lit.fn nested()
    lit.fn nested() -> !kgen.none {
      // CHECK-NEXT: %[[NONE:.*]] = kgen.param.constant: none
      %none_0 = kgen.param.constant: none = <#kgen.none>
      // CHECK-NEXT: kgen.return %[[NONE:.*]]
      lit.return %none_0 : !kgen.none
      lit.end_fn
    }
    // CHECK: hlcf.if
    hlcf.if %c {
      %none_0 = kgen.param.constant: none = <#kgen.none>
      // CHECK: lit.alias.decl *"y`"
      // CHECK: kgen.return
      lit.return %none_0 : !kgen.none
      hlcf.yield
    // CHECK: } else {
    } else {
      // CHECK-NEXT: hlcf.yield
      hlcf.yield
    }
    lit.try.yield
  } except {
    lit.try.yield
  // CHECK: } else {
  } else {
    // CHECK: lit.alias.decl *"y`f0"
    // CHECK: lit.try.yield
    lit.try.yield
  } finally {
    lit.alias.decl *"y`" = <x>
    lit.try.yield
  }
  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_fn
}

// CHECK-LABEL: lit.fn @containsEarlyReturn
lit.fn @containsEarlyReturn(%arg: !kgen.scalar<bool>) -> !kgen.none {
  // CHECK: hlcf.elif %arg {
  // CHECK:     %none = kgen.param.constant: none = <#kgen.none>
  // CHECK:     kgen.return %none : !kgen.none
  // CHECK:   } else {
  // CHECK:     %none = kgen.param.constant: none = <#kgen.none>
  // CHECK:     kgen.return %none : !kgen.none
  // CHECK:   }
  // CHECK:   kgen.unreachable
  hlcf.elif %arg {
    %none_0 = kgen.param.constant: none = <#kgen.none>
    lit.return %none_0 : !kgen.none
    hlcf.yield
  } else {
    %none_0 = kgen.param.constant: none = <#kgen.none>
    lit.return %none_0 : !kgen.none
    hlcf.yield
  }
  lit.end_fn
}

// CHECK-LABEL: lit.fn @fallthrough
lit.fn @fallthrough<cond0: scalar<bool>, cond1: scalar<bool>>(%lhs: index, %rhs: index, %cond2 : !kgen.scalar<bool>) -> index {
// CHECK: kgen.param.if <cond0> {
// CHECK-NEXT:   kgen.return %lhs : index
// CHECK-NEXT: } else {
// CHECK-NEXT: kgen.param.if <cond1> {
// CHECK-NEXT:   kgen.return %rhs : index
// CHECK-NEXT:  } else {
// CHECK-NEXT:  hlcf.elif %cond2 {
// CHECK-NEXT:    hlcf.yield
// CHECK-NEXT:  } else {
// CHECK-NEXT:    hlcf.yield
// CHECK-NEXT:  }
// CHECK-NEXT:  %index0 = kgen.param.constant = <0>
// CHECK-NEXT:  kgen.return %index0 : index
// CHECK-NEXT:  }
// CHECK-NEXT:  kgen.unreachable
// CHECK-NEXT: }
// CHECK-NEXT: kgen.unreachable
 kgen.param.if <cond0> {
   lit.return %lhs : index
   kgen.param.yield
 } else {
   kgen.param.if <cond1> {
     lit.return %rhs : index
     kgen.param.yield
   } else {
     hlcf.elif %cond2 {
       hlcf.yield
     } else {
       hlcf.yield
     }
     %0 = kgen.param.constant: index = <0>
     lit.return %0 : index
     kgen.param.yield
   }
   kgen.param.yield
 }
 lit.end_fn
}


// CHECK-LABEL: lit.fn @consecutiveElifs
lit.fn @consecutiveElifs(%arg0: index, %arg1: index) -> index {
  %idx0 = index.constant 0
  %idx1 = index.constant 1

  // CHECK:  index.cmp eq(%arg0, %idx0)
  // CHECK-NEXT: pop.cast_from_builtin
  // CHECK-NEXT: hlcf.elif {{%.*}} -> index {
  %c0 = index.cmp eq(%arg0, %idx0)
  %c0_sb = pop.cast_from_builtin %c0 : i1 to !kgen.scalar<bool>
  %0 = hlcf.elif %c0_sb -> index {
    hlcf.yield %arg0 : index
  } else {
    hlcf.yield %arg1 : index
  }
  // CHECK:  index.cmp eq(%arg0, %idx1)
  // CHECK-NEXT:   pop.cast_from_builtin
  // CHECK-NEXT: hlcf.elif {{%.*}} {
  // CHECK-NEXT:   kgen.return %arg0 : index
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   kgen.return %arg1 : index
  // CHECK-NEXT: }
  %c1 = index.cmp eq(%arg0, %idx1)
  %c1_sb = pop.cast_from_builtin %c1 : i1 to !kgen.scalar<bool>
  hlcf.elif %c1_sb {
    lit.return %arg0 : index
    hlcf.yield
  } else {
    lit.return %arg1 : index
    hlcf.yield
  }
  // CHECK-NEXT: kgen.unreachable
  // CHECK-NEXT: }
  lit.end_fn
}

// CHECK-LABEL: lit.fn @param_if_call_throws
lit.fn @param_if_call_throws<paramb: scalar<bool>>() throws -> !kgen.scalar<bool> {
  // CHECK: kgen.param.if <paramb> {
  kgen.param.if <paramb> {
    %err = lit.var.decl "err" synth : !lit.ref<@Error, mut elt>
    %result = lit.var.decl "result" synth : !lit.ref<none, mut lt>
    // CHECK: lit.call @throwing_func
    // CHECK: hlcf.if
    // CHECK:   lit.error_return
    // CHECK: else
    // CHECK:   hlcf.yield
    lit.call @throwing_func[mut elt, mut lt](%err, %result) : !lit.generator<[2](!lit.ref<@Error, mut *[0,0]> byref_error, !lit.ref<none, mut *[0,1]> byref_result) throws -> !kgen.scalar<bool>>
    kgen.param.if <paramb> {
      // CHECK: %[[THEN_RETURN:.*]] = kgen.param.constant: scalar<bool> = <false>
      // CHECK: kgen.return %[[THEN_RETURN]]
      %then_return = kgen.param.constant: scalar<bool> = <false>
      lit.return %then_return : !kgen.scalar<bool>
      kgen.param.yield
    } else {
      // CHECK: %[[ELSE_RETURN:.*]] = kgen.param.constant: scalar<bool> = <true>
      // CHECK: kgen.return %[[ELSE_RETURN]]
      %else_return = kgen.param.constant: scalar<bool> = <true>
      lit.return %else_return : !kgen.scalar<bool>
      kgen.param.yield
    }
    // CHECK: kgen.unreachable
    kgen.param.yield
  } else {
    %else_return = kgen.param.constant: scalar<bool> = <true>
    lit.return %else_return : !kgen.scalar<bool>
    // CHECK: %[[ELSE_RETURN:.*]] = kgen.param.constant: scalar<bool> = <true>
    // CHECK: kgen.return %[[ELSE_RETURN]]
    kgen.param.yield
  }
  // CHECK: kgen.unreachable
  lit.end_fn
}

// Derived from MOCO-1475
lit.fn @crashing_try_warning(%cond: !kgen.scalar<bool>) -> !kgen.none {
  lit.loop {
    hlcf.if %cond {
      hlcf.yield
    } else {
      lit.loop.break.else
    }
    lit.loop.continue
  } else {
    lit.try { // expected-warning {{try body doesn't raise an exception}}
      lit.try.yield
    } except {
      lit.try.yield
    } else {
      lit.try.yield
    } finally {
      lit.try.yield
    }
    lit.loop.yield
  }
  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_fn
}

// Derived from MOCO-2119.
// We had some weirdness where the outer kgen.param.if was incorrectly using
// the hlcf.elif's doesFallThrough as its own doesFallThrough, and then that was
// causing it to not replace the lit.end_fn with a kgen.unreachable.
lit.fn @weird_fallthroughs<parambool: scalar<bool>>(%runbool: i1) -> i1 {
  kgen.param.if <parambool> {
    lit.return %runbool : i1
    kgen.param.yield
  } else {
    %runbool_sb = pop.cast_from_builtin %runbool : i1 to !kgen.scalar<bool>
    hlcf.elif %runbool_sb {
      hlcf.yield
    } else {
      hlcf.yield
    }
    kgen.param.if <parambool> {
      lit.return %runbool : i1
      kgen.param.yield
    } else {
      lit.return %runbool : i1
      kgen.param.yield
    }
    kgen.param.yield
  }
  lit.end_fn
}

// kgen.param.assert with a statically false condition means the assertion
// fails at elaboration time; all code after it is dead.

// CHECK-LABEL: lit.fn @dead_code_after_param_assert_false
lit.fn @dead_code_after_param_assert_false<cond: i1>() -> !kgen.none {
  // A true assert is not a terminator.
  // CHECK: kgen.param.assert <true>
  kgen.param.assert <true>, "this always passes"

  // A false assert causes everything after it to be dead.
  // CHECK-NEXT: kgen.param.assert <false>
  // CHECK-NEXT: kgen.unreachable
  kgen.param.assert <false>, "this always fails"

  // expected-warning @+1 {{unreachable code after compile-time assertion failure}}
  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_fn
}
