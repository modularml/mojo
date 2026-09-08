// RUN: kgen-opt %s -verify-diagnostics -split-input-file

// expected-note @below {{see function here}}
kgen.func @coroutine_handle() {
  // expected-error @below {{'co.handle' op surrounding function must have 1 result}}
  %hdl = co.handle : index
  kgen.return
}

// -----

// expected-note @below {{surrounding function returns 'index'}}
kgen.func @coroutine_handle() -> index {
  // expected-error @below {{'co.handle' op surrounding function result type does not match coroutine handle type}}
  %hdl = co.handle : index
  %idx0 = index.constant 0
  kgen.return %idx0 : index
}

// -----

kgen.func @invalid_callee() {
  // expected-error @below {{callable must be 'async'}}
  %0 = co.invoke[() -> (): @not_async]()
  kgen.return
}
