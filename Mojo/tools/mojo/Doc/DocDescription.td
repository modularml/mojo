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

def DocDescription : MojoCommandDescription<
  "doc", "Compiles docstrings from a Mojo file.",
  "This is an early version of a documentation tool that generates an API "
  "reference from Mojo code comments. Currently, it generates a structured "
  "output of all docstrings into a JSON file, and it does not generate HTML. "
  "This output format is subject to change.\n"
  "\n"
  "The input may be a single file or a directory. If you specify a "
  "directory, it will generate a single JSON output with documentation "
  "for all modules found in that path, recursively.",
  [UsageInfo<"options", "path", /*variadicInput=*/false>]>, Index<7>;
