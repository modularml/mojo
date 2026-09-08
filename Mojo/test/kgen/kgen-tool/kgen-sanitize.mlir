// RUN: kgen %s -sanitize=address -emit=llvm | FileCheck %s --check-prefix=CHECK-ASAN
// RUN: kgen %s -sanitize=thread -emit=llvm | FileCheck %s --check-prefix=CHECK-TSAN

// CHECK-ASAN: define dso_local float @exp_f32(float noundef %0) #[[FNATTRS:.*]]
// CHECK-ASAN: attributes #[[FNATTRS:.*]] = {{.*}} sanitize_address

// CHECK-TSAN: define dso_local float @exp_f32(float noundef %0) #[[FNATTRS:.*]]
// CHECK-TSAN: attributes #[[FNATTRS:.*]] = {{.*}} sanitize_thread

kgen.generator export @exp_f32(%arg: f32) -> f32 {
  kgen.return %arg : f32
}
