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

// RUN: not %driver-tblgen -gen-man-page %s 2>&1 >/dev/null | FileCheck %s

include "Support/Driver/DriverCommand.td"

def : CommandDescription<
  "driver-tblgen-test", "option-validation", "Test option validation.",
  "Test that errors are emitted when options do not include help text, or "
  "when they do not match the style we expect.",
  [UsageInfo<"options", "path", /*variadicInput=*/false>]>;

def FunOptionGroup : OptionGroup<"Fun Options">;

// CHECK: error: help text for option 'chute' should not be empty
// CHECK: error: option 'chute' takes a value, but does not define a metavar name for that value
def chute : Separate<["-"], "chute">, Group<FunOptionGroup>, Index<0>;

// CHECK: error: help text for option 'ladder' should begin with a capital letter
// CHECK: error: help text for option 'ladder' should end with a period or quotation mark
// CHECK: error: option 'ladder' metavar name should not be empty
defm ladder : CommandOption<["--"], "ladder", FunOptionGroup, Index<1>,
  MetaVarName<"">, HelpText<"the ladder to use">>;
def stool : Flag<["--"], "stool">, Group<FunOptionGroup>, Index<2>,
  HelpText<"Whether to use a stool.">;
// CHECK: error: alias 's' has no index
def s : Joined<["-"], "s">, Group<FunOptionGroup>, Alias<stool>;

// CHECK: error: option 'slide' metavar name should be uppercase
// CHECK: error: option 'slide' has no index
def slide : JoinedOrSeparate<["--", "-"], "slide">, Group<FunOptionGroup>,
  HelpText<"The slide to use.">, MetaVarName<"path">;

// CHECK: error: group 'Fun Options' has no index

def HappyOptionGroup : OptionGroup<"Happy Options">, Flags<[HelpHidden]>;
// CHECK: error: option group 'Happy Options' is already hidden, marking its option 'smile' as hidden is redundant
def smile : Flag<["-"], "smile">, Group<HappyOptionGroup>, Index<0>,
  Flags<[HelpHidden]>, HelpText<"Sometimes it's all you can do.">;

def VoiceOptionGroup : OptionGroup<"Voice Options">;
def whisper : Flag<["-"], "whisper">, Group<VoiceOptionGroup>, Index<0>,
  Flags<[HelpHidden]>, HelpText<"Use an inside voice.">;

// CHECK: error: publicly documented group 'Voice Options' has no publicly documented options

def StanceOptions : OptionGroup<"Stance Options">, Flags<[HelpHidden]>;

// CHECK: error: group 'Stance Options' has no options
