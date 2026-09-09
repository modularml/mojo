// TODO(#19240): These tests should support Windows.  The same steps should be
// doable on Windows, but these commands would need to be rewritten in Batch or
// PowerShell.
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: ln -s %crash-report-path-info %t/crash-report-path-info
// RUN: ln -s %modular-crashpad-handler %t/nonstandard-handler
// RUN: printf '[crash_reporting]\nhandler_path = %t/nonstandard-handler\n' > %t/modular.cfg
// RUN: env -u MODULAR_CRASH_REPORTING_HANDLER_PATH -u TEST_SRCDIR -u RUNFILES_DIR MODULAR_HOME=%t %t/crash-report-path-info -get crashpad-handler | FileCheck %s
// CHECK: {{.*}}nonstandard-handler
