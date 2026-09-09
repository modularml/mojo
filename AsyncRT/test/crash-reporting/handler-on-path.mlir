// TODO(#19240): These tests should support Windows.  The same steps should be
// doable on Windows, but these commands would need to be rewritten in Batch or
// PowerShell.
// RUN: rm -rf %t
// RUN: mkdir -p %t/fake-path
// RUN: ln -s %crash-report-path-info %t/crash-report-path-info
// RUN: ln -s %modular-crashpad-handler %t/fake-path/modular-crashpad-handler
// RUN: env -i PATH=%t/fake-path %t/crash-report-path-info -get crashpad-handler | FileCheck %s
// CHECK: {{.*}}fake-path{{[\\/]}}modular-crashpad-handler
