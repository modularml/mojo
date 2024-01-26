# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# This file is only run on linux targets with amx_tile
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: linux
# REQUIRES: amx_tile
# RUN: %mojo -debug-level full %s | FileCheck %s


from sys.info import has_intel_amx, os_is_linux

from IntelAMX import init_intel_amx


# CHECK-LABEL: test_has_intel_amx
fn test_has_intel_amx():
    print("== test_intel_amx_amx")
    # CHECK: True
    print(os_is_linux())
    # CHECK: True
    print(has_intel_amx())
    # CHECK: True
    print(init_intel_amx())


fn main():
    test_has_intel_amx()
