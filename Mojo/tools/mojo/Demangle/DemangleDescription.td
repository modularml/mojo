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

def DemangleDescription : MojoCommandDescription<
  "demangle", "Demangles the given name.",
  "If the given name is a mangled Mojo symbol name, prints the demangled "
  "name. If no name is provided, one is read from standard input.",
  [UsageInfo<"options", "name", /*variadicInput=*/false>]>, Index<8>;
