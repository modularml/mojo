// RUN: kgen-opt -verify-parameters -lift-and-fold-apply %s | FileCheck %s

kgen.generator @pass(%arg0: index) -> index {
  kgen.return %arg0 : index
}

// CHECK-LABEL: kgen.generator @take_and_pass
// CHECK-SAME: !pop.array<*[[L0:.*]], index>
kgen.generator @take_and_pass<N>() -> !pop.array<apply(:(index) -> index @pass, N), index> {
  // CHECK-NEXT: apply *[[L0]] = [(index) -> index: @pass](N)
  kgen.unreachable
}

// CHECK-LABEL: kgen.generator @lift_apply
kgen.generator @lift_apply() {
  kgen.param.declare p0 = <1>
  kgen.param.declare p1 = <add(p0, 1)>
  // CHECK: apply *[[L0:.*]] = [(index) -> index: @pass](p0)
  // CHECK: apply *[[L1:.*]] = [(index) -> index: @pass](*[[L0]])
  // CHECK: apply *[[L2:.*]] = [(index) -> index: @pass](p1)
  // CHECK: apply *[[L3:.*]] = [(index) -> index: @pass](*[[L2]])
  // CHECK: constant = <*[[L1]]>
  kgen.param.constant = <apply(:(index) -> index @pass, apply(:(index) -> index @pass, p0))>
  // CHECK: constant = <*[[L1]]>
  kgen.param.constant = <apply(:(index) -> index @pass, apply(:(index) -> index @pass, p0))>
  // CHECK: constant = <*[[L0]]>
  kgen.param.constant = <apply(:(index) -> index @pass, p0)>
  // CHECK: call @take_and_pass<*[[L2]]>() : () -> !pop.array<*[[L3]], index>
  %0 = kgen.call @take_and_pass<apply(:(index) -> index @pass, p1)>() : ()
    -> !pop.array<apply(:(index) -> index @pass, apply(:(index) -> index @pass, p1)), index>

  // CHECK: region F = <p2>
  kgen.param.declare.region F = <p2>() {
    // CHECK: apply *[[L4:.*]] = [(index) -> index: @pass](p2)
    // CHECK: constant = <*[[L4]]>
    kgen.param.constant = <apply(:(index) -> index @pass, p2)>
    kgen.return
  }

  kgen.return
}

kgen.generator @consume<N>(%arg0: !pop.array<N, index>) {
  kgen.return
}

// CHECK-LABEL: kgen.generator @apply_value_crosses
kgen.generator @apply_value_crosses(%arg0: !pop.array<apply(:(index) -> index @pass, 1), index>) {
  // CHECK-NEXT: apply *[[L0:.*]] = [(index) -> index: @pass](1)
  // CHECK: kgen.param.if
  kgen.param.if <false> {
    // CHECK-NEXT: apply *[[L1:.*]] = [(index) -> index: @pass](2)
    // CHECK: constant = <*[[L1]]>
    kgen.param.constant = <apply(:(index) -> index @pass, 2)>
    // CHECK-NEXT: call @consume<*[[L0]]>(%arg0) : (!pop.array<*[[L0]], index>)
    kgen.call @consume<apply(:(index) -> index @pass, 1)>(%arg0)
      : (!pop.array<apply(:(index) -> index @pass, 1), index>) -> ()
    kgen.return
  } else {
    kgen.param.yield
  }
  kgen.return
}


// CHECK-LABEL: kgen.generator @lift_from_bind_params
kgen.generator @lift_from_bind_params<
    dispatch: <() -> !pop.array<apply(:(index) -> index @pass, 1), index>>() -> ()>() {
  // CHECK-NEXT: apply *[[L0:.*]] = [(index) -> index: @pass](1)
  // CHECK-NEXT: bind_params({{.*}} dispatch, :() -> !pop.array<apply(:(index) -> index @pass, 1), index> rebind(:() -> !pop.array<*[[L0]], index> @take_and_pass<1>))
  kgen.param.declare fn: () -> () = <bind_params(
    :<() -> !pop.array<apply(:(index) -> index @pass, 1), index>>() -> () dispatch,
    :() -> !pop.array<apply(:(index) -> index @pass, 1), index> @take_and_pass<1>)>
  kgen.return
}

// CHECK-LABEL: kgen.generator @preserve_gen_attr
kgen.generator @preserve_gen_attr() {
  // COM: Make sure that the `apply` inside the generator attr is not lifted, otherwise the index reference will be invalid.
  // CHECK-NEXT: <#kgen.gen<apply(:(index) -> index @pass, *(0,0))>
  kgen.param.constant: !kgen.generator<<index>index> = <
     #kgen.gen<apply(:(index) -> index @pass, *(0,0))>
  >
  kgen.return
}

kgen.generator @bad() -> index {
  kgen.unreachable
}

// CHECK-LABEL: kgen.generator @nohoist_cond
kgen.generator @nohoist_cond() {
  kgen.param.declare cond: !kgen.scalar<bool> = <false>
  // CHECK-NOT: kgen.param.apply
  kgen.param.declare value = <cond(cond, apply(:() -> index @bad), 1)>
  kgen.return
}

// CHECK-LABEL: kgen.generator @hlcf_if_apply
kgen.generator @hlcf_if_apply(%cond: !kgen.scalar<bool>) {
  // COM: make sure that the apply is being lifted to the beginning
  // COM: of the generator since hlcf.if regions don't create
  // COM: new parameter decl scopes.

  kgen.param.declare p0 = <1>
  // CHECK: apply *[[L0:.*]] = [(index) -> index: @pass](p0)
  // CHECK: apply *[[L1:.*]] = [(index) -> index: @pass](*[[L0]])
  // CHECK: hlcf.if
  hlcf.if %cond  {
    // CHECK: constant = <*[[L1]]>
    kgen.param.constant = <apply(:(index) -> index @pass, p0)>
    kgen.param.constant = <apply(:(index) -> index @pass, apply(:(index) -> index @pass, p0))>
    hlcf.yield
  } else {
    // CHECK: constant = <*[[L1]]>
    kgen.param.constant = <apply(:(index) -> index @pass, apply(:(index) -> index @pass, p0))>
    hlcf.yield
  }

  kgen.return
}

// CHECK-LABEL: kgen.generator @hlcf_if_in_param_if_apply
kgen.generator @hlcf_if_in_param_if_apply(%cond0: !kgen.scalar<bool>, %cond1: !kgen.scalar<bool>) {
  // COM: make sure that the apply is being lifted to the beginning
  // COM: of the paramDecl region since hlcf.if regions don't create
  // COME: new parameter decl scopes.

  kgen.param.declare p0 = <1>
  // CHECK: kgen.param.if
  kgen.param.if <false> {
    // CHECK: apply *[[L0:.*]] = [(index) -> index: @pass](p0)
    // CHECK: apply *[[L1:.*]] = [(index) -> index: @pass](*[[L0]])
    // CHECK: hlcf.if
    hlcf.if %cond0  {
      // CHECK: constant = <*[[L0]]>
      // CHECK: constant = <*[[L1]]>
      kgen.param.constant = <apply(:(index) -> index @pass, p0)>
      kgen.param.constant = <apply(:(index) -> index @pass, apply(:(index) -> index @pass, p0))>
      hlcf.yield
    } else {
      // CHECK: constant = <*[[L1]]>
      kgen.param.constant = <apply(:(index) -> index @pass, apply(:(index) -> index @pass, p0))>
      hlcf.yield
    }

    hlcf.if %cond1  {
      // CHECK: constant = <*[[L0]]>
      // CHECK: constant = <*[[L1]]>
      kgen.param.constant = <apply(:(index) -> index @pass, p0)>
      kgen.param.constant = <apply(:(index) -> index @pass, apply(:(index) -> index @pass, p0))>
      hlcf.yield
    } else {
      // CHECK: constant = <*[[L1]]>
      kgen.param.constant = <apply(:(index) -> index @pass, apply(:(index) -> index @pass, p0))>
      hlcf.yield
    }

    kgen.param.yield
  } else {
    kgen.param.yield
  }
  kgen.return
}
