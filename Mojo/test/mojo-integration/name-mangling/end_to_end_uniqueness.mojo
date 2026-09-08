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
#
# Regression test: a host program with a parameterized function and a custom
# linkage name. The linkage name is not a product of the function's parameters
# so will clash on repeated instantiations. Test that we are able to handle
# this with uniqueness suffixes.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo-build --emit asm %s -o %t.s
# RUN: FileCheck %s --input-file=%t.s


@no_inline
@__name("0this_name_is_long_and_will_clash_so_mangle_it")
def make_me_unique[n: Int]():
    print("hello", n)


# The two instantiations of the function 'make_me_unique' are separate concrete
# functions whose linkage names would clash if we didn't mangle.
# Check that the output assembly contains both functions with unique suffixes.
# Check also that the names are not sanitized or shortened, as this is not an
# accelerator target.

# CHECK-DAG: {{^_?}}0this_name_is_long_and_will_clash_so_mangle_it_{{[0-9a-f]+}}:
# CHECK-DAG: {{^_?}}0this_name_is_long_and_will_clash_so_mangle_it_{{[0-9a-f]+}}:


def main() raises:
    make_me_unique[0]()
    make_me_unique[1]()
