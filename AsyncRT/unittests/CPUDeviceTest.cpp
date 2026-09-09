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

#include "AsyncRT/Runtime/HostSystem.h"

#include "gtest/gtest.h"

using namespace M;
using namespace M::AsyncRT;

namespace {

CPUDeviceRef createTestCPUDevice() {
  return AsyncRT::getOrCreateCPUDevice(AsyncRT::CPUDeviceSource::Test,
                                       AsyncRT::CPUDeviceOptions()
                                           .withLeakCheckedAllocator()
                                           .withMainWillNotDonate());
}

/// `getOrCreateCPUDevice` installs a single global CPUDevice; a second call on
/// the same thread with matching options returns another reference to that
/// CPUDevice.
TEST(CPUDeviceTest, GetOrCreateCPUDeviceReturnsSameGlobalInstance) {
  auto first = createTestCPUDevice();
  auto second = createTestCPUDevice();
  EXPECT_EQ(first.getPointer(), second.getPointer());
}

/// Test to ensure that the thread-local CPUDevice pointer is cleared when a
/// CPUDevice is destroyed.
TEST(CPUDeviceTest, CreateDestroyCreateClearsTls) {
  for (int i = 0; i < 5; ++i) {
    auto cpuDevice = createTestCPUDevice();
    cpuDevice.reset(); // Destructor clears thread-local CPUDevice pointer.
  }
}

TEST(CPUDeviceTest, DefaultAffinityBehavior) {
  // Ensure env var is not set (may already be in environment)
  unsetenv("MODULAR_ENABLE_AFFINITY");
  AsyncRT::CPUDeviceOptions options;
  // Disabled by default.
  EXPECT_FALSE(options.withAffinity);
}

TEST(CPUDeviceTest, EnvVarEnablesAffinity) {
  setenv("MODULAR_ENABLE_AFFINITY", "1", 1);
  AsyncRT::CPUDeviceOptions options;
  EXPECT_TRUE(options.withAffinity);
  unsetenv("MODULAR_ENABLE_AFFINITY");
}

TEST(CPUDeviceTest, EnvVarEnablesAffinityWithTrue) {
  setenv("MODULAR_ENABLE_AFFINITY", "true", 1);
  AsyncRT::CPUDeviceOptions options;
  EXPECT_TRUE(options.withAffinity);
  unsetenv("MODULAR_ENABLE_AFFINITY");
}

TEST(CPUDeviceTest, BuilderMethodOverridesEnvVar) {
  // Even when env var doesn't enable, builder can enable
  unsetenv("MODULAR_ENABLE_AFFINITY");
  AsyncRT::CPUDeviceOptions options;
  EXPECT_FALSE(options.withAffinity);
  options.withCPUAffinity(true);
  EXPECT_TRUE(options.withAffinity);
}
} // namespace
