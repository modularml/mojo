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

// This tool exposes the getter API's of `M::Config` class from
// `Support/Configuration.h` to enable testing the behavior of this
// foundational component.

#include "Support/CommandLine.h"
#include "Support/Configuration.h"

#include <cstdlib>
#include <iostream>

using namespace M;

namespace {

struct ConfigurationEnvTestCLIOptions {
  cl::opt<bool> ModularConfigFolderPath{
      "ModularConfigFolderPath",
      cl::desc("M::Config::getModularConfigFolderPath()"),
  };

  cl::opt<bool> ModularDataFolderPath{
      "ModularDataFolderPath",
      cl::desc("M::Config::getModularConfigFolderPath()"),
  };

  cl::opt<bool> ModularCacheFolderPath{
      "ModularCacheFolderPath",
      cl::desc("M::Config::getModularCacheFolderPath()"),
  };

  cl::opt<bool> ConfigFilePath{
      "ConfigFilePath",
      cl::desc("M::Config::getConfigFilePath()"),
  };
};

} // namespace

int main(int argc, char **argv) {
  ConfigurationEnvTestCLIOptions cli;
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    R"(Configuration Environment Test Tool

Multiple options may be specified at once.

Output is in JSON format and consists of a dictionary where the keys are the
name of the command-line option, which is in turn the name of getter member
function with the `get` prefix removed, for example to get the value returned
by `M::Config::getModularConfigFolderPath()`:

    $ env_test_cpp --ModularConfigFolderPath

    {
      "ModularConfigFolderPath": "/home/ubuntu/.config/modular"
    }
)");

  ErrorOr<Config> cfg = Config::open();
  if (cfg.isError()) {
    std::cerr << "FAILURE: M::Config::open(): " << cfg.getError() << "\n";
    return EXIT_FAILURE;
  }

  std::cout << "{";
  bool printComma = false;

  auto print = [&](const std::string &key, const std::string &value) {
    if (printComma) {
      std::cout << ",";
    } else {
      printComma = true;
    }
    std::cout << "\n  \"" << key << "\": " << "\"" << value << "\"";
  };

  if (cli.ConfigFilePath) {
    auto configFilePath = cfg->getConfigFilePath();
    if (configFilePath.isError()) {
      std::cerr << "FAILURE: cfg->getConfigFilePath():"
                << configFilePath.getError() << "\n";
      return EXIT_FAILURE;
    }
    print("ConfigFilePath", *configFilePath);
  }

  if (cli.ModularConfigFolderPath) {
    auto configFolderPath = cfg->getModularConfigFolderPath();
    if (configFolderPath.isError()) {
      std::cerr << "FAILURE: cfg->getModularConfigFolderPath(): "
                << configFolderPath.getError() << "\n";
      return EXIT_FAILURE;
    }
    print("ModularConfigFolderPath", *configFolderPath);
  }

  if (cli.ModularDataFolderPath) {
    auto dataFolderPath = cfg->getModularDataFolderPath();
    if (dataFolderPath.isError()) {
      std::cerr << "FAILURE: cfg->getModularConfigFolderPath(): "
                << dataFolderPath.getError() << "\n";
      return EXIT_FAILURE;
    }
    print("ModularDataFolderPath", *dataFolderPath);
  }

  if (cli.ModularCacheFolderPath) {
    auto cacheFolderPath = cfg->getModularCacheFolderPath();
    if (cacheFolderPath.isError()) {
      std::cerr << "FAILURE: cfg->getModularCacheFolderPath(): "
                << cacheFolderPath.getError() << "\n";
      return EXIT_FAILURE;
    }
    print("ModularCacheFolderPath", *cacheFolderPath);
  }

  std::cout << "\n}\n";

  return EXIT_SUCCESS;
}
