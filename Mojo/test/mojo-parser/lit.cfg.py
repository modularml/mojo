# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import os

from lit.llvm import llvm_config

# name: The name of this test suite.
config.name = "mojo-parser"

config.parser_stubs_source = os.path.abspath(
    os.path.join("Mojo", "test", "test-packages")
)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mojo", ".test"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join("Mojo", "test", "mojo-parser")

config.excludes = [
    "debuginfo/inputs",
    "lsp/inputs",
    "test_package",
    "test_package.foo",
    "test_package_user",
    "debuginfo_module.mojo",
    "imported_module.mojo",
    "imported_cached_module.mojo",
]

translate_with_prebuilt_packages = (
    "kgen-translate -import-mojo -mojo-enable-prebuilt-packages"
    " -mojo-search-paths={0}".format(config.parser_stubs_source)
)

config.substitutions.append(
    ("%parse-mojo-isolated", translate_with_prebuilt_packages)
)


config.environment["MODULAR_HOME"] = os.path.join(
    "KGEN", "test", "mojo-parser"
)
