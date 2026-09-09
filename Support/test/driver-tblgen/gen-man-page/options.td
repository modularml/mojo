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

// RUN: %driver-tblgen -gen-man-page %s | FileCheck %s

// RUN: %driver-tblgen -gen-man-page %s 2>&1 >/dev/null | \
// RUN:   FileCheck %s --allow-empty --check-prefix CHECK-DIAG
// CHECK-DIAG-NOT: warning

include "Support/Driver/DriverCommand.td"

// CHECK: \fBdriver\-tblgen\-test\fR [\fIoptions\fR] \fIinputs...\fR [\fIinputs-arguments...\fR]
// CHECK-NEXT: .br
// CHECK-NEXT: \fBdriver\-tblgen\-test\fR \fIinput\fR
// CHECK-NEXT: .br
// CHECK-NEXT: \fBdriver\-tblgen\-test\fR
// CHECK-NEXT: .br
def : CommandDescription<
  "driver-tblgen-test", /*subcommand=*/"",
  "Test that options are output in a man page.",
  "This command contains several option groups, each of which has several "
  "options, along with aliases.",
  [
    UsageInfo<"options", "inputs", /*variadicInput=*/true, /*inNameHasArgs=*/true>,
    UsageInfo<"", "input", /*variadicInput=*/false>,
    UsageInfo<"", "", /*variadicInput=*/false>
  ]>;

// CHECK: .SH "COMMANDS"
// CHECK-NEXT: \fBsubcommand\fR \[em] Test that subcommands are output in a man page
// CHECK-NEXT: .br
def : CommandDescription<
  "driver-tblgen-test", "subcommand",
  "Test that subcommands are output in a man page.",
  "This is the description of a subcommand.",
  [UsageInfo<"options", "input", /*variadicInput=*/false>]>, Index<0>;

// CHECK: .SH "OPTIONS"
// CHECK: .SS "First Options"
def FirstOptionGroup : OptionGroup<"First Options">, Index<0>;

// Note that the following options have different indices than their source
// order. We test here that they appear in ascending index order.
// CHECK: \fB\-bar\fR, \fB\-\-baz\fR, \fB\-\-bazzer\fR, \fB\-\-bazzest\fR
// CHECK-NEXT: .RS 4
// CHECK-NEXT: Turns on bar support.
// CHECK-NEXT: .RE
// CHECK-NEXT: .sp
// CHECK: \fB\-\-foo\fR \fIFOO\fR, \fB\-F\fR\fIALSO\-FOO\fR, \fB\-F1\fR (\fIFOO\fR=\fI1\fR)
// CHECK-NEXT: .RS 4
// CHECK-NEXT: Takes a foo and does something with it.
// CHECK-NEXT: .RE
// CHECK-NEXT: .sp
defm foo : CommandOption<["--", "-"], "foo", FirstOptionGroup, Index<1>,
  MetaVarName<"FOO">, HelpText<"Takes a foo and does something with it.">>;
def : Joined<["-"], "F">, Group<FirstOptionGroup>, Index<1>,
  MetaVarName<"ALSO-FOO">, Alias<foo>;
def : Joined<["-"], "F1">, Group<FirstOptionGroup>, Index<2>,
  Alias<foo>, AliasArgs<["1"]>;

def bar : Flag<["-"], "bar">, Group<FirstOptionGroup>, Index<0>,
  HelpText<"Turns on bar support.">;
def : Flag<["--"], "bazzest">, Group<FirstOptionGroup>, Index<2>, Alias<bar>;
def : Flag<["--"], "baz">, Group<FirstOptionGroup>, Index<0>, Alias<bar>;
def : Flag<["--"], "bazzer">, Group<FirstOptionGroup>, Index<1>, Alias<bar>;

// Values do not make their way into man output.
// CHECK: \fB\-\-valued\fR \fIVALUE\fR
// CHECK-NEXT: .RS 4
// CHECK-NEXT: Values example.
// CHECK-NEXT: .RE
// CHECK-NEXT: .sp
defm valued : CommandOptionWithValues<["--", "-"], "valued", FirstOptionGroup,
  Index<3>, MetaVarName<"VALUE">, HelpText<"Values example.">, Values<"a,b,c">>;

// CHECK: .SS "Second Options"
// CHECK-NEXT: Another \-\- potentially less useful \-\- set of options.
// CHECK-NEXT: .sp
def SecondOptionGroup : OptionGroup<"Second Options">, Index<1>,
  HelpText<"Another -- potentially less useful -- set of options.">;

// CHECK: \fB\-I\fR
// CHECK-NEXT: .RS 4
// CHECK-NEXT: Specify the "I", as in "igloo."
// CHECK-NEXT: .RE
// CHECK-NEXT: .sp
def : Joined<["-"], "I">, Group<SecondOptionGroup>, Index<0>,
  HelpText<"Specify the \"I\", as in \"igloo.\"">, MetaVarName<"IGLOO">;

// Hidden option groups don't appear in manual pages.
// CHECK-NOT: Third Options
// CHECK-NOT: blong
def ThirdOptionGroup : OptionGroup<"Third Options">, Index<2>,
  Flags<[HelpHidden]>;
def : Flag<["--"], "blong">, Group<ThirdOptionGroup>, Index<0>,
  HelpText<"Blongs.">;
