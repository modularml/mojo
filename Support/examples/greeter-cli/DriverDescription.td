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
// This file defines help text for the `greeter-cli` executable itself, as
// opposed to its command line options. This is included in generated help text
// and documentation.
//
//===----------------------------------------------------------------------===//

include "Support/Driver/DriverCommand.td"
include "Bye/ByeDescription.td"
include "Hi/HiDescription.td"

def : DriverDescription<
  /*executable=*/"greeter-cli",
  /*summary=*/"A friendly greeting tool.",
  /*description=*/"This handy tool greets you in the manner you choose.", [
    // Documentation for CLIs typically includes "usage" information, like
    // `greeter-cli [options]`. You can define any number of these using
    // `UsageInfo`. See its documentation for more details.
    UsageInfo</*optionsName=*/"", /*inputName=*/"command",
              /*variadicInput=*/false>,
    UsageInfo<"options", "", /*variadicInput=*/false>
  ]>;
