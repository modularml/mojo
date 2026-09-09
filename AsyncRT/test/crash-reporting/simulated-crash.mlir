// TODO(#19240): These tests should support Windows.  The same steps should be
// doable on Windows, but these commands would need to be rewritten in Batch or
// PowerShell.
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: printf '[crash_reporting]\nurl = http://invalid.\n' > %t/modular.cfg
// RUN: env MODULAR_HOME=%t crash-test-dummy -simulate
// RUN: (cd %t && find .) | FileCheck %s
// CHECK: ./crashdb/{{pending|completed}}/{{.*}}.dmp
