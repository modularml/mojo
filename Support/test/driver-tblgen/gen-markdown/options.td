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

// RUN: %driver-tblgen -gen-markdown %s | FileCheck %s

// RUN: %driver-tblgen -gen-markdown %s 2>&1 >/dev/null | \
// RUN:   FileCheck %s --allow-empty --check-prefix CHECK-DIAG
// CHECK-DIAG-NOT: warning

include "Support/Driver/DriverCommand.td"

// CHECK: driver-tblgen-test [options] <input> [input-arguments...]
// CHECK: driver-tblgen-test <command>
def : CommandDescription<
  "driver-tblgen-test", /*subcommand=*/"",
  "Test that options are output in Markdown.",
  "This command contains an option group with options.",
  [
    UsageInfo<"options", "input", /*variadicInput=*/false, /*inNameHasArgs=*/true>,
    UsageInfo<"", "command", /*variadicInput=*/false>
  ]>;

// CHECK: ## Commands
// CHECK: [`subcommand`](subcommand.md) — Test that subcommands are output in Markdown
def : CommandDescription<
  "driver-tblgen-test", "subcommand",
  "Test that subcommands are output in Markdown.",
  "This is the description of a subcommand.",
  [UsageInfo<"options", "input", /*variadicInput=*/false>]>, Index<0>;

// CHECK: ## Options
// CHECK: ### First Options
def FirstOptionGroup : OptionGroup<"First Options">, Index<0>;

// CHECK: #### `--foo <FOO>`, `-F<ALSO-FOO>`, `-F1 (FOO=1)`
// CHECK: Takes a foo and does something with it.
defm foo : CommandOption<["--", "-"], "foo", FirstOptionGroup,
  Index<0>, MetaVarName<"FOO">,
  HelpText<"Takes a foo and does something with it.">>;
def : Joined<["-"], "F">, Group<FirstOptionGroup>, Index<1>,
  MetaVarName<"ALSO-FOO">, Alias<foo>;
def : Joined<["-"], "F1">, Group<FirstOptionGroup>, Index<2>,
  Alias<foo>, AliasArgs<["1"]>;

// Values do not make their way into Markdown output.
// CHECK: #### `--bar <BAR>`
// CHECK: Takes a bar and does something with it.
defm bar : CommandOptionWithValues<["--", "-"], "bar", FirstOptionGroup,
  Index<2>, MetaVarName<"BAR">,
  HelpText<"Takes a bar and does something with it.">, Values<"a,b,c">>;

// Hidden option groups don't appear in Markdown.
// CHECK-NOT: Second Options
// CHECK-NOT: blong
def SecondOptionGroup : OptionGroup<"Second Options">, Index<1>,
  Flags<[HelpHidden]>;
def : Flag<["--"], "blong">, Group<SecondOptionGroup>, Index<0>,
  HelpText<"Blongs.">;
