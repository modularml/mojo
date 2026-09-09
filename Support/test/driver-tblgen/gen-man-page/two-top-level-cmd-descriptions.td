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

// RUN: not %driver-tblgen -gen-man-page %s 2>&1 | FileCheck %s
// CHECK: error: cannot define a second top-level 'CommandDescription' record

include "Support/Driver/DriverCommand.td"

def TopLevelDescriptionOne : CommandDescription<
  "exec", /*subcommand=*/"", "Test that two top-level commands result in error",
  "This is a top-level driver record, since it doesn't have a subcommand name.",
  [UsageInfo<"x", "y", /*variadicInput=*/true>]>;

def TopLevelDescriptionTwo : CommandDescription<
  "exec", /*subcommand=*/"", "Test that two top-level commands result in error",
  "Two top-level driver records cannot exist in the same set of records.",
  [UsageInfo<"x", "y", /*variadicInput=*/true>]>;
