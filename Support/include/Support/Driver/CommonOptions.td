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
//
// This file defines a "Common options" option group, as well as the "help"
// options "-h" and "--help".
//
// mojo-tblgen sorts option groups based on index; these common options are
// given a high index value so that they appear last on a list of options.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DRIVER_COMMONOPTIONS_TD
#define SUPPORT_DRIVER_COMMONOPTIONS_TD

include "Support/Driver/DriverCommand.td"
include "llvm/Option/OptParser.td"

def CommonOptionGroup : OptionGroup<"Common options">, Index<99>;

def help : Flag<["--", "-"], "help">, Group<CommonOptionGroup>, Index<99>,
  HelpText<"Displays help information.">;
def h : Flag<["-"], "h">, Group<CommonOptionGroup>, Alias<help>, Index<0>;

def help_hidden : Flag<["--", "-"], "help-hidden">, Group<CommonOptionGroup>, Index<100>,
  HelpText<"Displays help for hidden options.">;

#endif // SUPPORT_DRIVER_COMMONOPTIONS_TD
