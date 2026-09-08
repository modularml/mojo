# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

# Test that number parsing lookup tables (powers_of_5_table and POWERS_OF_10)
# are optimized to use global constants instead of stack allocation.
#
# This verifies that the global_constant optimization is working correctly.
# Before the optimization, these lookup tables would be materialized on the
# stack (10,416 bytes for powers_of_5_table, 184 bytes for POWERS_OF_10) every
# time they were accessed. With the optimization, they live as global constants
# in the read-only data section.

# RUN: mkdir -p %t
# RUN: %mojo-build %s -o %t/test_number_parsing_lut.ll --emit llvm
# RUN: FileCheck %s --check-prefix=GLOBAL-CONSTANTS --input-file=%t/test_number_parsing_lut.ll
# RUN: FileCheck %s --check-prefix=NO-STACK-ALLOC --input-file=%t/test_number_parsing_lut.ll

# Verify that powers_of_5_table (1302 x i64) exists as a global constant
# GLOBAL-CONSTANTS-DAG: @{{.*}}global_constant{{.*}} = {{.*}}constant {{.*}}[1302 x i64]

# Verify that POWERS_OF_10 (23 x double) exists as a global constant
# GLOBAL-CONSTANTS-DAG: @{{.*}}global_constant{{.*}} = {{.*}}constant {{.*}}[23 x double]

# Verify that powers_of_5_table is NOT allocated on the stack
# NO-STACK-ALLOC-NOT: alloca {{.*}}[1302 x i64]

# Verify that POWERS_OF_10 is NOT allocated on the stack
# NO-STACK-ALLOC-NOT: alloca {{.*}}[23 x double]


def test_number_parsing() raises -> String:
    """Test that number parsing functions work correctly with optimized lookup tables.
    """
    var result = String()

    # Test float parsing that uses POWERS_OF_10
    var f1 = atof("1.23e10")
    var f2 = atof("4.56e-5")
    var f3 = atof("789.012")

    result += String(f1)
    result += ","
    result += String(f2)
    result += ","
    result += String(f3)

    return result


def main() raises:
    var results = test_number_parsing()

    # We don't actually need to print the results for the test,
    # but we need to use them so they don't get optimized away
    if results.byte_length() > 0:
        pass
