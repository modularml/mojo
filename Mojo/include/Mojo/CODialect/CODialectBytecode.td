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

#ifndef CODIALECTBYTECODE_TD
#define CODIALECTBYTECODE_TD

include "Support/Compiler/Bytecode.td"

//===----------------------------------------------------------------------===//
// CODialect Types
//===----------------------------------------------------------------------===//

def CoroutineType : DialectType<(type)>;

def CODialectTypes : DialectTypes<"co"> {
  let elems = [
    CoroutineType
  ];
}

//===----------------------------------------------------------------------===//
// CODialect Attributes
//===----------------------------------------------------------------------===//

def CODialectAttributes : DialectAttributes<"co"> {
  let elems = [];
}

#endif // CODIALECTBYTECODE_TD
