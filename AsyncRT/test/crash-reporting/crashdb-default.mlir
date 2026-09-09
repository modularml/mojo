// TODO(#19240): These tests should support Windows.  The same steps should be
// doable on Windows, but these commands would need to be rewritten in Batch or
// PowerShell.
// RUN: rm -rf %t
// RUN: mkdir -p %t/home
// RUN: env -i MODULAR_HOME=%t/home %crash-report-path-info -get crashdb | FileCheck %s
// CHECK: {{.*}}home{{[\\/]}}crashdb
