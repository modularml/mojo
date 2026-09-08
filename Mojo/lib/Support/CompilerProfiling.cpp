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

#include "Mojo/Support/CompilerProfiling.h"

using namespace M;
using namespace KGEN;

//===--------------------------------------------------------------------===//
// TraceProfiler
//===--------------------------------------------------------------------===//

void TraceProfiler::initialize(int timeTraceGranularity) {
  std::error_code ec;
  std::filesystem::path derived = std::filesystem::absolute(
      llvm::sys::Process::GetEnv("MODULAR_DERIVED_PATH").value_or("."), ec);

  profiler.emplace(timeTraceGranularity, "kgen",
                   (derived / "kgen.trace.json").string());
}

TraceProfiler::~TraceProfiler() {
  if (!profiler)
    return;
  if (auto err = profiler->write("-"))
    llvm::errs() << "unable to write trace file: " << err.getError();
}
