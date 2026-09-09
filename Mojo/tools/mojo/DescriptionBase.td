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

#ifndef MOJO_DRIVER_OPTIONS_BASE_TD
#define MOJO_DRIVER_OPTIONS_BASE_TD

include "Support/Driver/DriverCommand.td"

// A description of the Mojo top-level driver executable.
class MojoDriverDescription<string summary, string description,
                            list<UsageInfo> usages> :
  DriverDescription</*executable=*/"mojo", summary, description, usages>;

// A description of a Mojo driver subcommand.
class MojoCommandDescription<string subcommand, string summary,
                             string description, list<UsageInfo> usages> :
  CommandDescription</*executable=*/"mojo", subcommand, summary, description,
                     usages>;

#endif // MOJO_DRIVER_OPTIONS_BASE_TD
