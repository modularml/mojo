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

include "DriverDescription.td"
include "Support/Driver/CommonOptions.td"

//===----------------------------------------------------------------------===//
// Diagnostic Options
//===----------------------------------------------------------------------===//

def DiagnosticOptionGroup : OptionGroup<"Diagnostic options">, Index<0>;

def version : Flag<["--", "-"], "version">, Group<DiagnosticOptionGroup>,
  Index<0>, HelpText<"Prints the Mojo version and exits.">;

def v : Flag<["-"], "v">, Group<DiagnosticOptionGroup>, Alias<version>,
  Index<0>;

//===----------------------------------------------------------------------===//
// Cache Management Options
//===----------------------------------------------------------------------===//

def CacheOptionGroup : OptionGroup<"Cache management options">, Index<1>;

def print_cache_location : Flag<["--", "-"], "print-cache-location">,
  Group<CacheOptionGroup>, Index<0>,
  HelpText<"Prints the Mojo compile cache (.mojo_cache) location and exits.">;

def clear_cache : Flag<["--", "-"], "clear-cache">,
  Group<CacheOptionGroup>, Index<1>,
  HelpText<"Removes the Mojo compile cache (.mojo_cache) after confirmation. "
           "Pass '-f' / '--force' to skip the confirmation prompt.">;
