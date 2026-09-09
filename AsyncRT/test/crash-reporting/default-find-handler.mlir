// RUN: %crash-report-path-info -get crashpad-handler | FileCheck %s
// CHECK: {{.*}}modular-crashpad-handler{{(\.exe)?}}
