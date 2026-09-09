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

#ifndef HLCFATTRS_TD
#define HLCFATTRS_TD

include "Mojo/HLCFDialect/HLCFDialect.td"
include "mlir/IR/EnumAttr.td"

class HLCF_Enum<string name, string summary, list<I32EnumAttrCase> cases> :
  I32EnumAttr<name, summary, cases> {
  let cppNamespace = "::M::HLCF";
  let genSpecializedAttr = 0;
}

class HLCF_EnumCase<string sym, int val, string str = sym> :
  I32EnumAttrCase<sym, val, str>;

class HLCF_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<HLCFDialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// UnrollLevel
//===----------------------------------------------------------------------===//

def HLCF_UnrollLevelAttr : HLCF_Attr<"UnrollLevel", "unroll_level"> {
  let summary = "loop unroll level";
  let description = [{
    The `#hlcf.unroll_level` attribute describes how much a loop should be
    unrolled. When set to `none`, the loop is not unrolled. When set to `full`,
    the loop will be fully unrolled. Alternatively, an unroll factor can be set.
  }];

  let parameters = (ins "UnrollLevel":$value);
  let assemblyFormat = "custom<UnrollLevel>($value)";

  let returnType = "HLCF::UnrollLevel";
  let constBuilderCall = "HLCF::UnrollLevelAttr::get($_builder.getContext(), $0)";
  let convertFromStorage = "$_self.getValue()";

  let extraClassDeclaration = [{
    /// Get UnrollLevel none attribute.
    static HLCF::UnrollLevelAttr getNone(MLIRContext* context);

    /// Get UnrollLevel full attribute.
    static HLCF::UnrollLevelAttr getFull(MLIRContext* context);
  }];
}

//===----------------------------------------------------------------------===//
// ForLoopBoundCmpPredicate
//===----------------------------------------------------------------------===//

def HLCF_ForLoopBoundCmpPredicate:
    HLCF_Enum<"ForLoopBoundCmpPredicate",
              "loop unroll bound compare predicate type", [
                           HLCF_EnumCase<"SGT", 0, "sgt">,
                           HLCF_EnumCase<"SLT", 1, "slt">,
                           HLCF_EnumCase<"SGE", 2, "sge">,
                           HLCF_EnumCase<"SLE", 3, "sle">]>;

def HLCF_ForLoopBoundCmpPredicateAttr
  : EnumAttr<HLCFDialect,
             HLCF_ForLoopBoundCmpPredicate,
             "for_loop_bound_cmp_predicate">;

//===----------------------------------------------------------------------===//
// ForLoopIndVarCompute
//===----------------------------------------------------------------------===//

def HLCF_ForLoopIndVarCompute:
    HLCF_Enum<"ForLoopIndVarCompute",
              "loop unroll induction variable compute type", [
                           HLCF_EnumCase<"ADD", 0, "add">,
                           HLCF_EnumCase<"SUB", 1, "sub">]>;

def HLCF_ForLoopIndVarComputeAttr
  : EnumAttr<HLCFDialect,
             HLCF_ForLoopIndVarCompute,
             "for_loop_ind_var_compute">;

#endif // HLCFATTRS_TD
