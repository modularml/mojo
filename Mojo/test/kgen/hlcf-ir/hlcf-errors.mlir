// RUN: kgen-opt %s -allow-unregistered-dialect -verify-diagnostics -split-input-file

kgen.func @loop_args() {
  // expected-error @below {{'hlcf.loop' op specifies 0 branch inputs but target expected 1 along control-flow edge from here}}
  "hlcf.loop"() ({
  ^bb0(%arg0: i32):
    kgen.return // expected-note {{to beginning of region #0 here}}
  }) : () -> ()
  kgen.return
}

// -----

kgen.func @return_mismatch_result_count(%arg0: i32) {
  hlcf.loop {
    // expected-error @below {{'kgen.return' op expected 0 operands, but given 1}}
    kgen.return %arg0 : i32
  }
}

// -----

kgen.func @return_mismatch_result_count(%arg0: i32) -> i64 {
  hlcf.loop {
    // expected-error @below {{'kgen.return' op operand #0 has type 'i32' but expected 'i64'}}
    kgen.return %arg0 : i32
  }
}

// -----

kgen.func @yield_mismatch(%arg0: !kgen.scalar<bool>, %arg1 : i32) {
  // expected-note @below {{to end of parent operation here}}
  %0 = hlcf.if %arg0 -> i64 {
    // expected-error @below {{'hlcf.yield' op branch input #0 has type 'i32' but target expected 'i64' along control-flow edge from here}}
    hlcf.yield %arg1 : i32
  } else {
    kgen.return
  }
}

// -----

// expected-note @below {{see control-flow root here}}
kgen.func @break_no_loop(%arg0: !kgen.scalar<bool>) {
  hlcf.if %arg0 {
    kgen.return
  } else {
    // expected-error @below {{'hlcf.break' op is not nested within a suitable parent operation}}
    hlcf.break
  }
}

// -----

kgen.func @break_wrong_types(%arg0: i32) {
  // expected-note @below {{to end of parent operation here}}
  %0 = hlcf.loop () -> i64 {
    // expected-error @below {{'hlcf.break' op branch input #0 has type 'i32' but target expected 'i64' along control-flow edge from here}}
    hlcf.break %arg0 : i32
  }
}

// -----

kgen.func @continue_wrong_types(%arg0: i32, %arg1 : i64) {
  hlcf.loop (%0 = %arg0 : i32) -> () {
    // expected-error @below {{'hlcf.continue' op branch input #0 has type 'i64' but target expected 'i32' along control-flow edge from here}}
    // expected-note @below {{to beginning of region #0 here}}
    hlcf.continue %arg1 : i64
  }
}

// -----

kgen.func @labelled_break_mismatch(%arg0: i32) {
  // expected-note @below {{to end of parent operation here}}
  hlcf.loop "foo" () -> index {
    hlcf.loop () -> i32 {
      // expected-error @below {{'hlcf.break' op branch input #0 has type 'i32' but target expected 'index' along control-flow edge from here}}
      hlcf.break "foo" %arg0 : i32
    }
    hlcf.continue
  }
  kgen.return
}
