// RUN: kgen-opt %s -eliminate-dead-symbols -allow-unregistered-dialect | FileCheck %s

// CHECK-NOT: @unused
kgen.func @unused() {
  kgen.return
}

// CHECK: @used
kgen.func @used() {
  kgen.return
}

// CHECK: @addr
kgen.func @addr() {
  kgen.return
}

// CHECK: @someOp
kgen.func @someOp() {
  kgen.return
}

// CHECK: @exported
kgen.func export @exported() {
  kgen.call @used() : () -> ()
  kgen.call @addr() : () -> ()
  "some.op"() {foo=@someOp} : () -> ()
  kgen.return
}

// CHECK: @A
kgen.func export @A() {
  kgen.call @B() : () -> ()
  kgen.return
}

// CHECK: @B
kgen.func @B() {
  kgen.call @A() : () -> ()
  kgen.return
}
