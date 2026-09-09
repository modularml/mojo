// RUN: kgen-opt %s -split-input-file -reorder-param-ops -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: @simple
kgen.generator @simple() {
  // COM: reorder param decl to before use.
  // CHECK: kgen.param.declare q
  // CHECK-NEXT: kgen.param.declare w
  // CHECK-NEXT: kgen.call @g2
  %0 = kgen.call @g2<q, w>() : () -> index
  kgen.param.declare q = <3>
  kgen.param.declare w = <5>

  kgen.return
}

// CHECK-LABEL: @nestedRegions()
kgen.generator @nestedRegions() {
  // COM: reorder param decl to before use.
  // CHECK: kgen.param.declare cond_var
  %0 = kgen.param.if <lt(cond_var, 10)> -> index {
    // CHECK: kgen.param.declare next_lt
    // CHECK-NEXT: "should.not.appear"
    %1 = "should.not.appear"() : () -> index
    kgen.param.declare next_lt = <add(cond_var, 10)>
    kgen.param.yield %1 : index
  } else {
    // CHECK: kgen.param.declare next_gt
    // CHECK-NEXT: "should.appear"
    %3 = "should.appear"() : () -> index
    kgen.param.declare next_gt = <add(cond_var, 20)>
    kgen.param.yield %3 : index
  }

  kgen.param.declare cond_var = <32>
  kgen.return
}

// CHECK-LABEL: @reorder_asserts
kgen.generator @reorder_asserts() {
  // COM: reorder param decl to before use.
  // CHECK: kgen.param.declare q
  // CHECK-NEXT: kgen.param.assert <ne(:scalar<index> from_builtin(q), 0)>
  // CHECK-NEXT: kgen.param.declare w
  // CHECK-NEXT: kgen.param.assert <ne(:scalar<index> from_builtin(w), 0)>
  // CHECK-NEXT: kgen.param.assert <gt(:scalar<index> from_builtin(w), from_builtin(q))>
  // CHECK-NEXT: kgen.param.assert <lt(:scalar<index> from_builtin(q), from_builtin(w))>
  // CHECK-NEXT: kgen.call @g2
  kgen.param.assert <ne(q, 0)>, "q is not 0"
  %0 = kgen.call @g2<q, w>() : () -> index
  kgen.param.declare q = <3>
  kgen.param.assert <ne(w, 0)>, "w is not 0"
  kgen.param.assert <gt(w, q)>, "w is greater than q"
  kgen.param.declare w = <5>
  kgen.param.assert <lt(q, w)>, "q is less than w"

  // CHECK: kgen.param.if
  %1 = kgen.param.if <lt(q, w)> -> index {
    // CHECK-NEXT: kgen.param.assert <ge(:scalar<index> from_builtin(q), 3)>
    // CHECK-NEXT: kgen.param.declare next_lt
    // CHECK-NEXT: kgen.param.assert <lt(:scalar<index> from_builtin(next_lt), from_builtin(q))>
    // CHECK-NEXT: "produce.value"
    %2 = "produce.value"() : () -> index
    kgen.param.declare next_lt = <add(q, 10)>
    kgen.param.assert <lt(next_lt, q)>, "next_lt is less than q"
    kgen.param.assert <ge(q, 3)>, "q is no less than 3"
    kgen.param.yield %2 : index
  } else {
    // CHECK: kgen.param.declare next_gt
    // CHECK-NEXT: kgen.param.assert <lt(:scalar<index> from_builtin(next_gt), from_builtin(w))>
    // CHECK-NEXT: "produce.value"
    %3 = "produce.value"() : () -> index
    kgen.param.assert <lt(next_gt, w)>, "next_gt is less than w"
    kgen.param.declare next_gt = <add(w, 20)>
    kgen.param.yield %3 : index
  }
  kgen.return
}

// CHECK-LABEL: @reorder_asserts_def_in_parent
kgen.generator @reorder_asserts_def_in_parent<q, w>() {
  // COM: lift param asserts to the top of the current param scope.
  // CHECK-NEXT: kgen.param.assert <ne(:scalar<index> from_builtin(w), 0)>
  // CHECK-NEXT: kgen.param.assert <gt(:scalar<index> from_builtin(w), from_builtin(q))>
  // CHECK-NEXT: kgen.call @g2
  %0 = kgen.call @g2<q, w>() : () -> index
  kgen.param.assert <ne(w, 0)>, "w is not 0"
  kgen.param.assert <gt(w, q)>, "w is greater than q"

  // CHECK: kgen.param.for
  kgen.param.for iter in ?
      has_next :() -> i1 ?
      get_next_iter :() -> () ? {
        // CHECK: kgen.param.assert <ge(:scalar<index> from_builtin(q), 3)>, "q is no less than 3"
        kgen.param.assert <ge(q, 3)>, "q is no less than 3"
        // CHECK: kgen.param.if
        %1 = kgen.param.if <lt(q, iter)> -> index {
          // CHECK-NEXT: kgen.param.assert <ge(:scalar<index> from_builtin(iter), 3)>
          // CHECK-NEXT: kgen.param.assert <ge(:scalar<index> from_builtin(q), 3)>
          // CHECK-NEXT: %2 = "produce.value"() : () -> index
          %2 = "produce.value"() : () -> index
          kgen.param.assert <ge(iter, 3)>, "iter is no less than 3"
          kgen.param.assert <ge(q, 3)>, "q is no less than 3"
          kgen.param.yield %2 : index
        } else {
          %3 = "produce.value"() : () -> index
          kgen.param.yield %3 : index
        }
        kgen.param.for.continue
      } else {
        // CHECK: kgen.param.assert <lt(:scalar<index> from_builtin(w), 3)>, "w is less than 3"
        // CHECK: kgen.param.if
        %1 = kgen.param.if <lt(q, iter)> -> index {
          // CHECK-NEXT: kgen.param.assert <ge(:scalar<index> from_builtin(q), 3)>
          %2 = "produce.value"() : () -> index
          kgen.param.assert <ge(q, 3)>, "q is no less than 3"
          kgen.param.yield %2 : index
        } else {
          // CHECK: else
          // CHECK-NEXT: kgen.param.assert <ge(:scalar<index> from_builtin(iter), 3)>
          kgen.param.assert <ge(iter, 3)>, "iter is no less than 3"
          %3 = "produce.value"() : () -> index
          kgen.param.yield %3 : index
        }
        kgen.param.assert <lt(w, 3)>, "w is less than 3"
        kgen.param.for.break
      }

  kgen.return
}
