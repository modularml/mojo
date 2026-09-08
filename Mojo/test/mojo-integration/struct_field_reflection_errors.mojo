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

# Test compile-time errors for struct field reflection APIs.

# RUN: not kgen %s -elaborate -D TEST_NONEXISTENT_INDEX=1 2>&1 | FileCheck %s --check-prefix=CHECK-INDEX
# RUN: not kgen %s -elaborate -D TEST_NONEXISTENT_TYPE=1 2>&1 | FileCheck %s --check-prefix=CHECK-TYPE
# RUN: not kgen %s -elaborate -D TEST_OFFSET_NONEXISTENT_FIELD=1 2>&1 | FileCheck %s --check-prefix=CHECK-OFFSET-NAME
# RUN: not kgen %s -elaborate -D TEST_OFFSET_OUT_OF_BOUNDS=1 2>&1 | FileCheck %s --check-prefix=CHECK-OFFSET-INDEX
# RUN: not kgen %s -elaborate -D TEST_OFFSET_NEGATIVE_INDEX=1 2>&1 | FileCheck %s --check-prefix=CHECK-OFFSET-NEGATIVE

from std.sys import get_defined_bool


struct TestStruct:
    var x: Int
    var y: Float64


# Test that field_index produces an error for non-existent field.
# Use `comptime assert` to force compile-time evaluation.
def test_nonexistent_field_index():
    comptime if get_defined_bool["TEST_NONEXISTENT_INDEX", False]():
        # CHECK-INDEX: has no field named 'nonexistent'
        comptime assert (
            reflect[TestStruct].field_index["nonexistent"]() == 0
        ), "should not reach here"


# Test that `field` produces an error for non-existent field.
def test_nonexistent_field_type():
    comptime if get_defined_bool["TEST_NONEXISTENT_TYPE", False]():
        # CHECK-TYPE: has no field named 'missing_field'
        comptime field_handle = reflect[TestStruct].field["missing_field"]
        # Force evaluation by using the type
        comptime assert field_handle.name() == "Int", "should not reach here"


# Test that field_offset[name=] produces an error for non-existent field.
def test_offset_nonexistent_field():
    comptime if get_defined_bool["TEST_OFFSET_NONEXISTENT_FIELD", False]():
        # CHECK-OFFSET-NAME: has no field named 'does_not_exist'
        comptime assert (
            reflect[TestStruct].field_offset[name="does_not_exist"]() == 0
        ), "should not reach here"


# Test that field_offset[index=] produces an error for out-of-bounds index.
def test_offset_out_of_bounds():
    comptime if get_defined_bool["TEST_OFFSET_OUT_OF_BOUNDS", False]():
        # CHECK-OFFSET-INDEX: field index 99 is out of bounds for struct with 2 fields
        comptime assert (
            reflect[TestStruct].field_offset[index=99]() == 0
        ), "should not reach here"


# Test that field_offset[index=] produces an error for negative index.
def test_offset_negative_index():
    comptime if get_defined_bool["TEST_OFFSET_NEGATIVE_INDEX", False]():
        # CHECK-OFFSET-NEGATIVE: field index -1 is out of bounds for struct with 2 fields
        comptime assert (
            reflect[TestStruct].field_offset[index=-1]() == 0
        ), "should not reach here"


def main():
    test_nonexistent_field_index()
    test_nonexistent_field_type()
    test_offset_nonexistent_field()
    test_offset_out_of_bounds()
    test_offset_negative_index()
