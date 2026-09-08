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

# Test that mojo-build accepts -Xlinker args. Specific behaviour will depend on
# the linker we use. This test assumes it's lld.

# RUN: mojo build -Xlinker --help %s 2>&1 | FileCheck %s --check-prefix HELP

# This test compiles a "library" containing a function "foo" to a library, and
# links into this program using the linker. It tests both static and shared libraries.

# RUN: mkdir -p %t.dir
# RUN: mojo build --emit object %S/inputs/libfoo.mojo -o %t.dir/libfoostatic.a
# RUN: mojo build --emit shared-lib %S/inputs/libfoo.mojo -o %t.dir/libfooshared.so

# RUN: mojo build -Xlinker -L%t.dir -Xlinker -lfoostatic %s -o %t.dir/run_static
# RUN: %t.dir/run_static | FileCheck %s

# RUN: mojo build -Xlinker -L%t.dir -Xlinker -lfooshared %s -o %t.dir/run_shared
# RUN: env LD_LIBRARY_PATH=%t.dir %t.dir/run_shared | FileCheck %s

# HELP: OVERVIEW: {{(lld|LLVM Linker)}}

from std.ffi import external_call


# CHECK: hello from foo: 0
def main() raises:
    external_call["foo", NoneType](0)
