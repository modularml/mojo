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
// This file defines help text for the `greeter-cli hi` subcommand. The
// subcommand's options, on the other hand, are defined in `HiOptions.td`.
//
// The concepts used here are the same as those documented in
// `../DriverDescription.td`; read that file for more details.
//
//===----------------------------------------------------------------------===//

include "Support/Driver/DriverCommand.td"

def : CommandDescription<
  /*executable=*/"greeter-cli",
  /*subcommand=*/"hi",
  /*summary=*/"Prints \"hi\".",
  /*description=*/"Prints \"hi\" to stdout and exits. If a name is provided, "
                  "includes that name in the greeting.\n\nThis greeting is "
                  "typically used when meeting someone.",
  [
    UsageInfo</*optionsName=*/"options", /*inputName=*/"name",
              /*variadicInput=*/false>
  ]>, Index<0>;
