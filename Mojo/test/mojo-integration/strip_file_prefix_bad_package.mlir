// RUN: not mojo precompile %S/inputs/bad_package -strip-file-prefix=. 2>&1 | FileCheck %s
// CHECK: {{^}}inputs/bad_package/bad_file.mojo:14:5: error
