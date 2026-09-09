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

#ifndef SUPPORT_DEBUGINFODIALECT_DEBUGINFOINTERFACES_TD
#define SUPPORT_DEBUGINFODIALECT_DEBUGINFOINTERFACES_TD

include "mlir/IR/OpBase.td"

def DebugInfo_ScopedLocation : OpInterface<"ScopedLocation"> {
  let description = [{
    This interface represents an operation that can have a debuginfo scope in
    its location, and that scope can be retrieved quickly without recursing into
    nested locations.
  }];
  let cppNamespace = "::M::DebugInfo";

  let methods = [
    InterfaceMethod<[{
        Return the location scope, or a null attribute if not available
      }], "DebugInfo::DIScopeAttr", "getLocScope", (ins)>,
  ];
}

def DebugInfo_SubprogramScoped : OpInterface<
  "SubprogramScoped", [DebugInfo_ScopedLocation]
>{
  let description = [{
    This interface represents an operation that can have a debuginfo subprogram
    scope in its location, and a body whose operations (except other nested
    SubprogramScoped ops) must have the same scope.
  }];
  let cppNamespace = "::M::DebugInfo";

  let methods = [
    InterfaceMethod<[{
      Return the function-like region of the operation.
    }], "Region &", "getBodyRegion", (ins)>,
    InterfaceMethod<[{
      Return the location scope, or a null attribute if not available
    }], "DebugInfo::DISubprogramAttr", "getSubprogramScope", (ins), [{}],
    [{
      return dyn_cast_or_null<DebugInfo::DISubprogramAttr>(this->getLocScope());
    }]>,
  ];

  let extraTraitClassDeclaration = [{
    DebugInfo::DIScopeAttr getLocScope() {
      Location loc = this->getOperation()->getLoc();
      if (auto fusedLoc = dyn_cast<
          mlir::FusedLocWith<DebugInfo::DISubprogramAttr>>(loc))
        return fusedLoc.getMetadata();
      return {};
    }
  }];

  let verify = [{
    return Impl::verifySubprogramScoped(cast<SubprogramScoped>($_op));
  }];
}

def DebugInfo_InlinedSubprogramScoped : OpInterface<
  "InlinedSubprogramScoped", [DebugInfo_SubprogramScoped]
>{
  let description = [{
    This interface represents a SubprogramScoped operation that also acts as the
    callsite to the subprogram represented by the op's body region. These ops
    have to carry additional location information of where the call is, which
    could contain callstack information due to inlining, etc.

    This extra location info needs to be stored in the op's location, because if
    it is stored as an attribute, then different locations change an op's
    identity for optimizations like CSE.

    We encode the callLoc in the location as a fused location with a
    DICallLocAttr.  However, we maintain that a fused location with a
    DIScopeAttr is the top level location where applicable.

    So locations on InlinedSubprogramScoped ops can be (where fileLoc represents
    a location with no DIScope):

    * No callLoc -> Normal location, IE fileLoc or fused<scop>[fileLoc]
    * With callLoc, no DIScope -> fused<call_loc<callLoc>>[fileLoc]
    * With callLoc, with DIScope ->
      fused<scope>[fused<call_loc<callLoc>>[fileLoc]]

    Example:

    ```mlir
    #loc1 = loc("foo.mlir":12:34)
    #loc2 = loc("foo.mlir":45:67)
    #subprogram = #debuginfo.subprogram<name = <"foo">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
    #call_loc = #debuginfo.call_loc<#loc1>
    #loc3 = loc(fused<#call_loc>[#loc2])
    #loc4 = loc(fused<#subprogram>[#loc3])
    ```

  }];

  let dependentTraits = [DebugInfo_ScopedLocation];
  let cppNamespace = "::M::DebugInfo";

  let extraSharedClassDeclaration = [{
    /// Return the location of the op after removing any encoded callLoc info.
    Location getLocNoInlined() { return Impl::getLocNoInlined($_op); }
    /// Return any callLoc info encoded in the location.
    LocationAttr getCallLocAttr() { return Impl::getCallLocAttr($_op); }
    /// Adjust location info to encode the callLoc info.
    void setCallLocAttr(LocationAttr callLoc) {
      Impl::setCallLocAttr($_op, callLoc);
    }
  }];
}

#endif // SUPPORT_DEBUGINFODIALECT_DEBUGINFOINTERFACES_TD
