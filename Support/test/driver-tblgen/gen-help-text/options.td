//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

// RUN: %driver-tblgen -gen-help-text %s | FileCheck %s
// RUN: %driver-tblgen -gen-help-hidden-text %s | FileCheck %s --check-prefix=CHECK-HIDDEN

// RUN: %driver-tblgen -gen-help-text %s 2>&1 >/dev/null | \
// RUN:   FileCheck %s --allow-empty --check-prefix CHECK-DIAG
// CHECK-DIAG-NOT: warning

include "Support/Driver/DriverCommand.td"

// CHECK: driver-tblgen-test [options] <inputs...> [inputs-arguments...]
// CHECK-NEXT: driver-tblgen-test <input>
// CHECK-NEXT: driver-tblgen-test
def : CommandDescription<
  "driver-tblgen-test", /*subcommand=*/"",
  "Test that options are output in help text.",
  "This command contains an option group with options.",
  [
    UsageInfo<"options", "inputs", /*variadicInput=*/true, /*inNameHasArgs=*/true>,
    UsageInfo<"", "input", /*variadicInput=*/false>,
    UsageInfo<"", "", /*variadicInput=*/false>
  ]>;

// CHECK: COMMANDS
// CHECK: subcommand — Test that subcommands are output in help text
def : CommandDescription<
  "driver-tblgen-test", "subcommand",
  "Test that subcommands are output in help text.",
  "This is the description of a subcommand.",
  [UsageInfo<"options", "input", /*variadicInput=*/false>]>, Index<0>;

// CHECK: OPTIONS
// CHECK: First Options
// CHECK: The first group of options.
def FirstOptionGroup : OptionGroup<"First Options">, Index<0>,
  HelpText<"The first group of options.">;

// CHECK: --foo <FOO>, -F<ALSO-FOO>, -F1 (FOO=1)
// CHECK-NEXT: Takes a foo and does something with it.
defm foo : CommandOption<["--", "-"], "foo", FirstOptionGroup, Index<0>,
  MetaVarName<"FOO">, HelpText<"Takes a foo and does something with it.">>;
def : Joined<["-"], "F">, Group<FirstOptionGroup>, Index<1>,
  MetaVarName<"ALSO-FOO">, Alias<foo>;
def : Joined<["-"], "F1">, Group<FirstOptionGroup>, Index<2>,
  Alias<foo>, AliasArgs<["1"]>;

// Values do not make their way into --help output.
// CHECK: --bar <BAR>
// CHECK-NEXT: Takes a bar and does something with it.
defm bar : CommandOptionWithValues<["--", "-"], "bar", FirstOptionGroup,
  Index<1>, MetaVarName<"BAR">,
  HelpText<"Takes a bar and does something with it.">, Values<"a,b,c">>;

// Hidden options don't appear in help text.
// CHECK-NOT: Don't show me this option in help text
// CHECK-HIDDEN: Don't show me this option in help text
def baz : Flag<["--"], "baz">, Group<FirstOptionGroup>, Index<2>,
  Flags<[HelpHidden]>, HelpText<"Don't show me this option in help text.">;

// Hidden option groups don't appear in help text.
// CHECK-NOT: Second Options
// CHECK-NOT: blong
// CHECK-HIDDEN: Second Options
// CHECK-HIDDEN: blong
def SecondOptionGroup : OptionGroup<"Second Options">, Index<1>,
  Flags<[HelpHidden]>;
def : Flag<["--"], "blong">, Group<SecondOptionGroup>, Index<0>,
  HelpText<"Blongs.">;
