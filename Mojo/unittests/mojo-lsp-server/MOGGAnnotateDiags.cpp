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

// Disabled: MAXPLAT-223
#if 0

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;

TEST(MOGGAnnotateTests, MissingReadWriteParams) {
  Document doc("test:///foo.mojo", R"(
import compiler_internal as compiler
from tensor import ManagedTensorSlice, OutputTensor, InputTensor

@extensibility.register("Missing")
struct Missing:
    @staticmethod
    def execute(a: ManagedTensorSlice):
        ...
)");

  createTestClient()
      .open(doc)
      .onDiagnostics(
          doc,
          [](const std::vector<lsp::Diagnostic> &diags) {
            ASSERT_EQ((int)diags.size(), 2);
            EXPECT_EQ(diags[0].message, "Error for argument 'a': 'mut' "
                                        "inferred parameter must be set");
            EXPECT_EQ(diags[1].message, "Error for argument 'a': 'input' "
                                        "inferred parameter must be set");
          })
      .execute();
}

TEST(MOGGAnnotateTests, OutputAfterInput) {
  Document doc("test:///foo.mojo", R"(
import compiler_internal as compiler
from tensor import ManagedTensorSlice, OutputTensor, InputTensor

@extensibility.register("OutputAfterInput")
struct OutputAfterInput:
    @staticmethod
    def execute(a: InputTensor, b: OutputTensor):
        ...
)");

  createTestClient()
      .open(doc)
      .onDiagnostics(doc,
                     [](const std::vector<lsp::Diagnostic> &diags) {
                       ASSERT_EQ((int)diags.size(), 1);
                       EXPECT_EQ(diags[0].message,
                                 "Output tensor argument 'b' must come before "
                                 "other non-output tensor arguments");
                     })
      .execute();
}

TEST(MOGGAnnotateTests, InputTensorsForShape) {
  Document doc("test:///foo.mojo", R"(
import compiler_internal as compiler
from tensor import ManagedTensorSlice, OutputTensor, InputTensor

@extensibility.register("InputTensorsForShape")
struct InputTensorsForShape:
    @staticmethod
    def execute(a: InputTensor):
      pass

    @staticmethod
    def shape(a: ManagedTensorSlice):
      pass
)");

  createTestClient()
      .open(doc)
      .onDiagnostics(doc,
                     [](const std::vector<lsp::Diagnostic> &diags) {
                       ASSERT_EQ((int)diags.size(), 1);
                       EXPECT_EQ(diags[0].message,
                                 "Error for argument 'a': Tensor arguments to "
                                 "shape functions must be 'InputTensor'");
                     })
      .execute();
}

TEST(MOGGAnnotateTests, NonInputTensorList) {
  Document doc("test:///foo.mojo", R"(
from compiler_internal import StaticTensorSpec
import compiler_internal as compiler
from tensor import OutputTensor
from tensor.managed_tensor_slice import _MutableInputTensor as MutableInputTensor

@extensibility.register("non_input_tensor_list")
struct NonInputTensorList:
    @staticmethod
    def execute[
        type: DType,
        rank: Int,
        target: StringLiteral,
    ](
        output: List[
            OutputTensor[
                static_spec = StaticTensorSpec[type, rank, ...].get_unknown(),
            ]
        ],
    ):
      pass
)");

  createTestClient()
      .open(doc)
      .onDiagnostics(doc,
                     [](const std::vector<lsp::Diagnostic> &diags) {
                       ASSERT_EQ((int)diags.size(), 1);
                       EXPECT_EQ(
                           diags[0].message,
                           "Only input tensors are allowed as the element type "
                           "for list arguments at the moment.");
                     })
      .execute();
}

TEST(MOGGAnnotateTests, PytorchFallbackInvalidArgument) {
  Document doc("test:///foo.mojo", R"(
import compiler_internal as compiler
from python import Python, PythonObject

@extensibility.register("pytorch_fallback")
struct InvalidPytorchFallBackArgument:
    @staticmethod
    def execute():
      ...

    @staticmethod
    def pytorch_fallback(a: PythonObject, b: Int):
      return
)");

  createTestClient()
      .open(doc)
      .onDiagnostics(
          doc,
          [](const std::vector<lsp::Diagnostic> &diags) {
            ASSERT_EQ((int)diags.size(), 2);
            EXPECT_EQ(
                diags[0].message,
                "Error for argument 'b' all arguments to 'pytorch_fallback' "
                "functions must have type 'PythonObject'");
            EXPECT_EQ(diags[1].message,
                      "Error for result type: the only permitted return type "
                      "for 'pytorch_fallback' functions is 'PythonObject'");
          })
      .execute();
}

TEST(MOGGAnnotateTests, PytorchFallbackInvalidResult) {
  Document doc("test:///foo.mojo", R"(
import compiler_internal as compiler
from python import Python, PythonObject

@extensibility.register("pytorch_fallback")
struct InvalidPytorchFallBackResult:
    @staticmethod
    def execute():
      ...

    @staticmethod
    def pytorch_fallback(a: PythonObject, b: PythonObject) -> Int:
      return Int(0)
)");

  createTestClient()
      .open(doc)
      .onDiagnostics(
          doc,
          [](const std::vector<lsp::Diagnostic> &diags) {
            ASSERT_EQ((int)diags.size(), 1);
            EXPECT_EQ(diags[0].message,
                      "Error for result type: the only permitted return type "
                      "for 'pytorch_fallback' functions is 'PythonObject'");
          })
      .execute();
}

TEST(MOGGAnnotateTests, PytorchFallbackInvalidOutResult) {
  Document doc("test:///foo.mojo", R"(
import compiler_internal as compiler
from python import Python, PythonObject

@extensibility.register("pytorch_fallback")
struct InvalidPytorchFallBackResult:
    @staticmethod
    def execute():
      ...

    @staticmethod
    def pytorch_fallback(out output: Int, a: PythonObject, b: PythonObject):
      return Int(0)
)");

  createTestClient()
      .open(doc)
      .onDiagnostics(
          doc,
          [](const std::vector<lsp::Diagnostic> &diags) {
            ASSERT_EQ((int)diags.size(), 1);
            EXPECT_EQ(diags[0].message,
                      "Error for result type: the only permitted return type "
                      "for 'pytorch_fallback' functions is 'PythonObject'");
          })
      .execute();
}

#endif // #if 0
