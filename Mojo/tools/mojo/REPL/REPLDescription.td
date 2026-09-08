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

def REPLDescription : MojoCommandDescription<
  "repl", "Launches the Mojo REPL.",
  "Launches a Mojo read-evaluate-print loop (REPL) environment, which provides "
  "interactive development in the terminal. You can also start the REPL by "
  "running `mojo` without CLI arguments.\n"
  "\n"
  "Options and arguments are forwarded to the underlying LLDB tool, "
  "which runs the REPL.\n"
  "\n"
  "Beware: the REPL has known issues and isn't under active development. "
  "We don't offer any REPL stability guarantees. To check REPL status and "
  "future developments, watch the Mojo REPL roadmap discussion on "
  "https://forum.modular.com/t/mojo-repl-roadmap/1158.",
  [UsageInfo<"lldb-options", "", /*variadicInput=*/false>]>, Index<3>;
