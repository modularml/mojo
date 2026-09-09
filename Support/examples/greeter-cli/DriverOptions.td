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
// This file defines options for the top-level `greeter-cli` command
// specifically, as opposed to one of its subcommands, `hi` or `bye`.
// That is, we will treat `greeter-cli --busy` as valid, but not
// `greeter-cli bye --busy`.
//
// Most of the TableGen classes used in this file, such as `OptionGroup`,
// `Option`, and `Flag`, come from `llvm/Option/OptParser.td`. Refer to that
// file for (very sparse) documentation.
//
//===----------------------------------------------------------------------===//

include "DriverDescription.td"
include "Support/Driver/CommonOptions.td"

// All command line options must belong to a group.
//
// All options and groups have an explicit ordering when printed in
// documentation. This order is represented with an `Index` record.
// `driver-tblgen` will error if you forget to include an index, or if two
// indices overlap.
def GreetingOptionGroup : OptionGroup<"Greeting options">, Index<0>;

// This defines the `--busy` or `-busy` option. Some notes:
// - The first prefix specified in the list of `["--", "-"]` is the one that's
//   preferentially printed in documentation.
// - Options are indexed within their group: this is at index 0 within the
//   "Greeting options" group.
def busy : Flag<["--", "-"], "busy">, Group<GreetingOptionGroup>, Index<0>,
  HelpText<"Eschews a traditional greeting in favor of a more curt response.">;

// This defines `-b` as an alias for `--busy`.
// - Aliases are indexed across other aliases: this is at index 0 within all
//   other aliases for `--busy`.
// - Aliases do not require help text, since they're documented with the aliased
//   option's help text.
def b : Flag<["-"], "b">, Group<GreetingOptionGroup>, Alias<busy>, Index<0>;
