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
// This file is nearly equivalent to `Hi/HiDescription.td`, you can read the
// comments there for an in-depth explanation of what's going on here.
//
//===----------------------------------------------------------------------===//

include "Support/Driver/DriverCommand.td"

def : CommandDescription<
  /*executable=*/"greeter-cli",
  /*subcommand=*/"bye",
  /*summary=*/"Prints \"bye\".",
  /*description=*/"Prints \"bye\" to stdout and exits.\n\nThis greeting is "
                  "typically used when parting ways with someone.",
  [
    UsageInfo</*optionsName=*/"options", /*inputName=*/"",
              /*variadicInput=*/false>
  ]>, Index<1>;
