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

# Invoking the driver with an unknown subcommand results in an error.
# RUN: not mojo unknown 2>&1 | FileCheck %s --check-prefix CHECK-UNKNOWN
# CHECK-UNKNOWN: mojo{{.*}}: error: no such command 'unknown'

# Typos are diagnosed and similarly-typed commands are suggested.
# RUN: not mojo domangle 2>&1 | FileCheck %s --check-prefix CHECK-TYPO
# CHECK-TYPO: mojo{{.*}}: error: no such command 'domangle'. Did you mean 'demangle'?

# Invoking the driver with `--help` prints the driver's help text, which
# includes its subcommands.
# RUN: mojo --help | FileCheck %s --check-prefix CHECK-HELP
# CHECK-HELP: mojo

# Invoking the driver with `--version` prints the version, for example
# '0.4.0-release (eb70c661)':
# RUN: mojo --version | FileCheck %s --check-prefix CHECK-VERSION
# CHECK-VERSION: Mojo {{[0-9]+}}.{{[0-9]+}}.{{[0-9]+}}{{.*}}({{[a-f0-9]+}})

# Invoking the driver with `--print-cache-location` prints the resolved
# `.mojo_cache` directory. An explicit MODULAR_CACHE_DIR overrides
# the path to `<override>/.mojo_cache`.
# RUN: env MODULAR_CACHE_DIR=%t.cache mojo --print-cache-location \
# RUN:   | FileCheck %s --check-prefix CHECK-CACHE-OVERRIDE -DTMP=%t.cache
# CHECK-CACHE-OVERRIDE: [[TMP]]/.mojo_cache

# Without any override, the resolved path still ends in `.mojo_cache`.
# RUN: mojo --print-cache-location | FileCheck %s --check-prefix CHECK-CACHE
# CHECK-CACHE: {{.*}}/.mojo_cache

# `--clear-cache` against a non-existent cache reports "nothing to do".
# RUN: rm -rf %t.clear-empty && env MODULAR_CACHE_DIR=%t.clear-empty \
# RUN:   mojo --clear-cache < /dev/null \
# RUN:   | FileCheck %s --check-prefix CHECK-CLEAR-EMPTY
# CHECK-CLEAR-EMPTY: does not exist; nothing to do

# `--clear-cache` aborts when the user does not answer 'y'.
# RUN: rm -rf %t.clear-abort && mkdir -p %t.clear-abort/.mojo_cache/sub \
# RUN:   && touch %t.clear-abort/.mojo_cache/sub/file
# RUN: echo "" | env MODULAR_CACHE_DIR=%t.clear-abort \
# RUN:   mojo --clear-cache \
# RUN:   | FileCheck %s --check-prefix CHECK-CLEAR-ABORT
# RUN: test -d %t.clear-abort/.mojo_cache
# CHECK-CLEAR-ABORT: Proceed? [y/N]
# CHECK-CLEAR-ABORT: Aborted.

# `--clear-cache` removes the directory when the user confirms with 'y'.
# RUN: rm -rf %t.clear-yes && mkdir -p %t.clear-yes/.mojo_cache/sub \
# RUN:   && touch %t.clear-yes/.mojo_cache/sub/file
# RUN: echo y | env MODULAR_CACHE_DIR=%t.clear-yes \
# RUN:   mojo --clear-cache \
# RUN:   | FileCheck %s --check-prefix CHECK-CLEAR-YES -DTMP=%t.clear-yes
# RUN: not test -e %t.clear-yes/.mojo_cache
# CHECK-CLEAR-YES: Removed [[TMP]]/.mojo_cache

# `--clear-cache -f` removes the directory without prompting. Run with stdin
# closed so any accidental prompt would surface as a failure.
# RUN: rm -rf %t.clear-force && mkdir -p %t.clear-force/.mojo_cache/sub \
# RUN:   && touch %t.clear-force/.mojo_cache/sub/file
# RUN: env MODULAR_CACHE_DIR=%t.clear-force mojo --clear-cache -f < /dev/null \
# RUN:   | FileCheck %s --check-prefix CHECK-CLEAR-FORCE -DTMP=%t.clear-force
# RUN: not test -e %t.clear-force/.mojo_cache
# CHECK-CLEAR-FORCE-NOT: Proceed?
# CHECK-CLEAR-FORCE: Removed [[TMP]]/.mojo_cache

# The long spelling `--force` works the same as `-f`.
# RUN: rm -rf %t.clear-force-long && mkdir -p %t.clear-force-long/.mojo_cache \
# RUN:   && touch %t.clear-force-long/.mojo_cache/file
# RUN: env MODULAR_CACHE_DIR=%t.clear-force-long \
# RUN:   mojo --clear-cache --force < /dev/null \
# RUN:   | FileCheck %s --check-prefix CHECK-CLEAR-FORCE-LONG \
# RUN:           -DTMP=%t.clear-force-long
# RUN: not test -e %t.clear-force-long/.mojo_cache
# CHECK-CLEAR-FORCE-LONG: Removed [[TMP]]/.mojo_cache

# An unrecognised flag after `--clear-cache` is rejected with a clear error,
# so typos like `--forced` don't silently fall through to the prompt.
# RUN: not env MODULAR_CACHE_DIR=%t.clear-bogus mojo --clear-cache --forced \
# RUN:   2>&1 | FileCheck %s --check-prefix CHECK-CLEAR-BOGUS
# CHECK-CLEAR-BOGUS: error: unexpected argument '--forced' for --clear-cache
