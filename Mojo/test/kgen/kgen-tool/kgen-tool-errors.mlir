// RUN: kgen %s -execute -func="some_func:f32()" -func="some_func:f32(f32)" -func="badkernel:f32()" -ignore-failure -verify-diagnostics

// expected-error@-3 {{could not find func '@badkernel'}}

// expected-error@below {{command-line specified signature does not match the IR signature}}
kgen.generator export @some_func(%arg0: f32) -> (f32, f32) {
  kgen.return %arg0, %arg0 : f32, f32
}
