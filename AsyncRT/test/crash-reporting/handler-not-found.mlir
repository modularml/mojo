// TODO(#19240): These tests should support Windows.  The same steps should be
// doable on Windows, but these commands would need to be rewritten in Batch or
// PowerShell.
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: ln -s %crash-report-path-info %t/crash-report-path-info
// RUN: (env -i %t/crash-report-path-info -get crashpad-handler 2>&1; true) | FileCheck %s
// CHECK: could not determine crashpad handler path: {{.*}}
