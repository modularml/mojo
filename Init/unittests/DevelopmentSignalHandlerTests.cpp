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

#include "Init/Init.h"
#include "gtest/gtest.h"
#include <csignal>

#ifndef MODULAR_PRODUCTION
// The signal handle is only active for development (i.e. non-production)
// builds.

using namespace M::Init;

TEST(DevelopmentSignalHandlerDeathTest, HandlesSIGSEGV) {
  EXPECT_EXIT(
      {
        auto contextOrError = createContext("test_segv_program");
        std::raise(SIGSEGV);
      },
      ::testing::ExitedWithCode(128 + SIGSEGV),
      ".*test_segv_program crashed!.*Signal "
      "Information:.*SIGSEGV.*Segmentation fault.*Signal Code:.*Sending "
      "PID:.*Process ID:.*Thread ID:.*");
}

TEST(DevelopmentSignalHandlerDeathTest, HandlesSIGABRT) {
  EXPECT_EXIT(
      {
        auto contextOrError = createContext("test_abort_program");
        std::raise(SIGABRT);
      },
      ::testing::ExitedWithCode(128 + SIGABRT),
      ".*test_abort_program crashed!.*Signal "
      "Information:.*SIGABRT.*Abort signal.*Process ID:.*Thread ID:.*");
}

TEST(DevelopmentSignalHandlerDeathTest, HandlesSIGFPE) {
  EXPECT_EXIT(
      {
        auto contextOrError = createContext("test_fpe_program");
        std::raise(SIGFPE);
      },
      ::testing::ExitedWithCode(128 + SIGFPE),
      ".*test_fpe_program crashed!.*Signal Information:.*SIGFPE.*Floating "
      "point exception.*Signal Code:.*Sending PID:.*Process ID:.*Thread ID:.*");
}

TEST(DevelopmentSignalHandlerDeathTest, HandlesSIGILL) {
  EXPECT_EXIT(
      {
        auto contextOrError = createContext("test_ill_program");
        std::raise(SIGILL);
      },
      ::testing::ExitedWithCode(128 + SIGILL),
      ".*test_ill_program crashed!.*Signal "
      "Information:.*SIGILL.*Illegal instruction.*Process ID:.*Thread ID:.*");
}

TEST(DevelopmentSignalHandlerDeathTest, HandlesSIGBUS) {
  EXPECT_EXIT(
      {
        auto contextOrError = createContext("test_bus_program");
        std::raise(SIGBUS);
      },
      ::testing::ExitedWithCode(128 + SIGBUS),
      ".*test_bus_program crashed!.*Signal "
      "Information:.*SIGBUS.*Bus error.*Process ID:.*Thread ID:.*");
}

TEST(DevelopmentSignalHandlerDeathTest, HandlesSIGTRAP) {
  EXPECT_EXIT(
      {
        auto contextOrError = createContext("test_trap_program");
        std::raise(SIGTRAP);
      },
      ::testing::ExitedWithCode(128 + SIGTRAP),
      ".*test_trap_program crashed!.*Signal "
      "Information:.*SIGTRAP.*Trace/breakpoint trap.*Process ID:.*Thread "
      "ID:.*");
}

TEST(DevelopmentSignalHandlerDeathTest, HandlesSIGSYS) {
  EXPECT_EXIT(
      {
        auto contextOrError = createContext("test_sys_program");
        std::raise(SIGSYS);
      },
      ::testing::ExitedWithCode(128 + SIGSYS),
      ".*test_sys_program crashed!.*Signal "
      "Information:.*SIGSYS.*Bad system call.*Process ID:.*Thread ID:.*");
}

#endif //! MODULAR_PRODUCTION
