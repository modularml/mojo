# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics %s -I=%S/inputs

# expected-error @+2 {{@__doc_inline is not supported on wildcard imports}}
@__doc_inline
from target_module import *

# A rename would publish the source name against the destination name.
# expected-error @+2 {{@__doc_inline is not supported on renamed imports ('import ... as ...')}}
@__doc_inline
from target_module import InlinedStruct as RenamedStruct

# expected-error @+1 {{'from' statement supports only the @stable and @__doc_inline decorators; remove the decorator}}
@not_a_real_decorator
from target_module import InlinedStruct
