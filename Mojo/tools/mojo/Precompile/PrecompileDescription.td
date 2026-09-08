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

def PrecompileDescription : MojoCommandDescription<
  "precompile", "Precompiles a Mojo package.",
  "Precompiles a directory of Mojo source files into a binary package suitable "
  "to share and import into other Mojo programs and modules. A precompiled Mojo"
  " package is faster to build with compared to building it from source. It is "
  " not intended as a distributable format as it is tied to the version of the "
  " compiler that produced it. Loading a precompiled package using a compiler "
  "of a different version will error. Despite this, it is technically portable "
  " across different systems because it is not an architecture-specific format "
  "(it includes only non-elaborated code). The code becomes an architecture-"
  "specific executable only after it's imported into a Mojo program that is "
  "then compiled with `mojo build`.\n"
  "\n"
  "To create a Mojo package, first add an `__init__.mojo` file to your "
  "package directory. Then pass that directory name to this "
  "command, and specify the output path and filename with `-o`.\n"
  "\n"
  "For more information, see [Mojo modules and "
  "packages](https://mojolang.org/docs/manual/packages/)." ,
  [UsageInfo<"options", "path", /*variadicInput=*/false>]>, Index<5>;
