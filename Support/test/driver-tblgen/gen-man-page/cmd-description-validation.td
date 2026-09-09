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

// CHECK: error: command executable name should not be empty
// CHECK: error: command summary should begin with a capital letter
// CHECK: error: command summary should end with a period
// CHECK: error: command description should begin with a capital letter
// CHECK: error: command description should end with a period
// CHECK: error: usage options name should be lowercase
// CHECK: error: usage input metavar name should be lowercase
def TopLevelDescription : CommandDescription<
  /*executable=*/"", /*subcommand=*/"", "test this record produces errors",
  "this is a top-level driver record, since it does not have a subcommand name",
  [
    UsageInfo<"Options", "path", /*variadicInput=*/false>,
    UsageInfo<"options", "patH", /*variadicInput=*/false>,
  ]>;

// CHECK: error: command executable name should not be empty
// CHECK: error: command summary should begin with a capital letter
// CHECK: error: command summary should end with a period
// CHECK: error: command description should begin with a capital letter
// CHECK: error: command description should end with a period
// CHECK: error: command should include at least one usage
// CHECK: error: subcommand 'sub-one' has no index
def SubcommandOneDescription : CommandDescription<
  "", "sub-one", "test this record also produces errors",
  "this is a subcommand record", /*usages=*/[]>;

// CHECK: error: subcommand '{{.*}}' has index 1, which has already been used
// CHECK: note: subcommand '{{.*}}' has already been defined with index 1 here
def SubcommandTwoDescription : CommandDescription<
  "exec", "sub-two", "Test this record is warned about because of its index.",
  "This record shares an index with the one below.",
  [UsageInfo<"x", "y", /*variadicInput=*/true>]>, Index<1>;
def SubcommandThreeDescription : CommandDescription<
  "exec", "sub-three", "Test this record is warned about because of its index.",
  "This record shares an index with the one above.",
  [UsageInfo<"x", "y", /*variadicInput=*/true>]>, Index<1>;
