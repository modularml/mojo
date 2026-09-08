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

include "DescriptionBase.td"

def RunDescription : MojoCommandDescription<
  "run", "Builds and executes a Mojo file.",
  "Compiles the Mojo file at the given path and immediately executes it. "
  "Another way to execute this command is to simply pass a file to `mojo`. "
  "For example:\n"
  "\n"
  "    mojo hello.mojo\n"
  "\n"
  "Options for this command itself, such as the ones listed below, must appear "
  "before the input file `path` argument. Any command line arguments that "
  "appear after the Mojo source file `path` are interpreted as arguments for "
  "that Mojo program.",
  [UsageInfo<"options", "path", /*variadicInput=*/false,
             /*inNameHasArgs=*/true>]>, Index<0>;
