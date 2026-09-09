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

#include "Mojo/Support/Configuration.h"
#include "Support/Configuration.h"
#include "Support/Context.h"
#include "Support/RCRef.h"
#include "gtest/gtest.h"

using namespace M;
using namespace M::KGEN;

// fromContext stores a raw Config* whose lifetime is tied to the Context.
// Keep ctx alive for the duration of each test so the pointer stays valid.

TEST(MojoConfigTest, GetPluginPathsEmpty) {
  auto ctx = RCRef<Context>::take(new Context());
  ctx->set(std::make_unique<Config>());
  MojoConfig cfg = MojoConfig::fromContext(ctx);
  EXPECT_TRUE(cfg.getPluginPaths().empty());
}

TEST(MojoConfigTest, GetPluginPathsSingle) {
  auto ctx = RCRef<Context>::take(new Context());
  auto config = std::make_unique<Config>();
  config->setValue("mojo-max.mojo_plugin_paths", "/usr/lib/mojo/plugin.so");
  ctx->set(std::move(config));
  MojoConfig cfg = MojoConfig::fromContext(ctx);

  SmallVector<std::string> paths = cfg.getPluginPaths();
  ASSERT_EQ(paths.size(), 1u);
  EXPECT_EQ(paths[0], "/usr/lib/mojo/plugin.so");
}

TEST(MojoConfigTest, GetPluginPathsMultiple) {
  auto ctx = RCRef<Context>::take(new Context());
  auto config = std::make_unique<Config>();
  config->setValue("mojo-max.mojo_plugin_paths",
                   "/usr/lib/plugin_a.so;/usr/lib/plugin_b.so");
  ctx->set(std::move(config));
  MojoConfig cfg = MojoConfig::fromContext(ctx);

  SmallVector<std::string> paths = cfg.getPluginPaths();
  ASSERT_EQ(paths.size(), 2u);
  EXPECT_EQ(paths[0], "/usr/lib/plugin_a.so");
  EXPECT_EQ(paths[1], "/usr/lib/plugin_b.so");
}

TEST(MojoConfigTest, GetPluginPathsSkipsEmptyEntries) {
  auto ctx = RCRef<Context>::take(new Context());
  auto config = std::make_unique<Config>();
  config->setValue("mojo-max.mojo_plugin_paths",
                   "/usr/lib/plugin_a.so;;/usr/lib/plugin_b.so;");
  ctx->set(std::move(config));
  MojoConfig cfg = MojoConfig::fromContext(ctx);

  SmallVector<std::string> paths = cfg.getPluginPaths();
  ASSERT_EQ(paths.size(), 2u);
  EXPECT_EQ(paths[0], "/usr/lib/plugin_a.so");
  EXPECT_EQ(paths[1], "/usr/lib/plugin_b.so");
}
