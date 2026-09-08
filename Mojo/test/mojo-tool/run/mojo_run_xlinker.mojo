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

# End-to-end test that `mojo run` honors `-Xlinker` flags by loading the
# referenced shared library into the in-process JIT, so that `external_call`
# symbols resolve at runtime. Mirrors build/mojo_build_xlinker.mojo.

# RUN: mkdir -p %t.dir
# RUN: mojo build --emit shared-lib %S/../build/inputs/libfoo.mojo \
# RUN:   -o %t.dir/libfooshared.so

# `-Xlinker -L<dir> -Xlinker -l<name>` form — the same syntax the user passes
# to `mojo build`.
# RUN: mojo run -Xlinker -L%t.dir -Xlinker -lfooshared %s | FileCheck %s

# `-l<name>` resolution must succeed even when the `-L<dir>` follows it on the
# command line, matching `ld`'s order-independent search-path semantics.
# RUN: mojo run -Xlinker -lfooshared -Xlinker -L%t.dir %s | FileCheck %s

# Split-argument forms: `-L <dir>` and `-l <name>` as separate `-Xlinker`
# values. `ld` accepts both joined and split spellings.
# RUN: mojo run -Xlinker -L -Xlinker %t.dir -Xlinker -l -Xlinker fooshared %s \
# RUN:   | FileCheck %s

# `--library-path=<dir>` is a GNU `ld` synonym for `-L<dir>`.
# RUN: mojo run -Xlinker --library-path=%t.dir -Xlinker -lfooshared %s \
# RUN:   | FileCheck %s

# `-rpath <dir>` is conflated with `-L` under JIT — see the doc comment in
# mojo-run.cpp's `resolveXlinkerLibraries`.
# RUN: mojo run -Xlinker -rpath -Xlinker %t.dir -Xlinker -lfooshared %s \
# RUN:   | FileCheck %s

# An absolute path to an existing shared library, loaded directly.
# RUN: mojo run -Xlinker %t.dir/libfooshared.so %s | FileCheck %s

# CHECK: hello from foo: 0


from std.ffi import external_call


def main() raises:
    external_call["foo", NoneType](0)
