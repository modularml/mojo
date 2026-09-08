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

#ifndef KGEN_LIB_MOJOLLDB_LANGUAGE_MOJOLANGUAGERUNTIME_H
#define KGEN_LIB_MOJOLLDB_LANGUAGE_MOJOLANGUAGERUNTIME_H

#include <vector>

#include "llvm/ADT/StringMap.h"

#include "lldb/Core/PluginInterface.h"
#include "lldb/Target/LanguageRuntime.h"
#include "lldb/lldb-private.h"

namespace M::KGEN::Mojo {

class MojoLanguageRuntime : public lldb_private::LanguageRuntime {
public:
  ~MojoLanguageRuntime() override = default;

  static char ID;

  bool isA(const void *ClassID) const override {
    return ClassID == &ID || LanguageRuntime::isA(ClassID);
  }

  static bool classof(const LanguageRuntime *runtime) {
    return runtime->isA(&ID);
  }

  lldb::LanguageType GetLanguageType() const override {
    return lldb::eLanguageTypeMojo;
  }

  static MojoLanguageRuntime *Get(lldb_private::Process &process) {
    return llvm::cast_or_null<MojoLanguageRuntime>(
        process.GetLanguageRuntime(lldb::eLanguageTypeMojo));
  }

  llvm::Error GetObjectDescription(lldb_private::Stream &str,
                                   lldb_private::ValueObject &object) override {
    return llvm::createStringError("Mojo does not support object descriptions");
  }

  llvm::Error GetObjectDescription(
      lldb_private::Stream &str, lldb_private::Value &value,
      lldb_private::ExecutionContextScope *exe_scope) override {
    return llvm::createStringError("Mojo does not support object descriptions");
  }

  /// Obtain a ThreadPlan to get us into C++ constructs such as std::function.
  ///
  /// \param[in] thread
  ///     Current thread of execution.
  ///
  /// \param[in] stop_others
  ///     True if other threads should pause during execution.
  ///
  /// \return
  ///      A ThreadPlan Shared pointer
  lldb::ThreadPlanSP GetStepThroughTrampolinePlan(lldb_private::Thread &thread,
                                                  bool stop_others) override {

    lldb::ThreadPlanSP retPlanSp;
    return retPlanSp;
  }

  bool IsAllowedRuntimeValue(lldb_private::ConstString name) override {
    return false;
  }

  bool
  GetDynamicTypeAndAddress(lldb_private::ValueObject &in_value,
                           lldb::DynamicValueType use_dynamic,
                           lldb_private::TypeAndOrName &class_type_or_name,
                           lldb_private::Address &address,
                           lldb_private::Value::ValueType &value_type,
                           llvm::ArrayRef<uint8_t> &local_buffer) override {
    return false;
  }

  bool CouldHaveDynamicValue(lldb_private::ValueObject &in_value) override {
    return false;
  }

  lldb_private::TypeAndOrName
  FixUpDynamicType(const lldb_private::TypeAndOrName &typeAndOrName,
                   lldb_private::ValueObject &static_value) override {
    return typeAndOrName;
  }

  static void Initialize();

  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() { return "mojo-runtime"; }

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  static lldb_private::LanguageRuntime *
  CreateInstance(lldb_private::Process *process, lldb::LanguageType language);

  lldb::BreakpointResolverSP
  CreateExceptionResolver(const lldb::BreakpointSP &bkpt, bool catch_bp,
                          bool throw_bp) override;

private:
  MojoLanguageRuntime(lldb_private::Process *process)
      : lldb_private::LanguageRuntime(process) {}
};

} // namespace M::KGEN::Mojo

#endif // KGEN_LIB_MOJOLLDB_LANGUAGE_MOJOLANGUAGERUNTIME_H
