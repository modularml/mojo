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

def BuildDescription : MojoCommandDescription<
  "build", "Builds an executable from a Mojo file.",
  "Compiles the Mojo file at the given path into an executable.\n\n"
  "By default, the executable is saved to the current directory and named the "
  "same as the input file, but without a file extension.\n\n"
  "Beware that any Python libraries used in your Mojo project are not included "
  "in the executable binary, so they must be provided by the environment "
  "where you run the executable.",
  [UsageInfo<"options", "path", /*variadicInput=*/false>]>, Index<1>;
