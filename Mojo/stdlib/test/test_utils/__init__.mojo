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
"""Provides test utility types and functions for the Mojo standard library tests."""

from .compare_helpers import compare
from .hash import assert_dif_hashes, assert_fill_factor, dif_bits
from .math_helpers import ulp_distance
from .test_utils import libm_call, check_write_to
from .types import (
    AbortOnCopy,
    AbortOnDel,
    CopyableExplicitDestroyKey,
    CopyCountedStruct,
    CopyCounter,
    DelCounter,
    DelRecorder,
    ExplicitCopyOnly,
    ExplicitDestroy,
    ExplicitDestroyKey,
    ImplicitCopyOnly,
    ConfigureTrivial,
    MoveCopyCounter,
    MoveCounter,
    MoveOnly,
    NonMovable,
    Observable,
    ObservableDel,
    ObservableMoveOnly,
    ExplicitDelOnly,
    Pinned,
    PinnedExplicitDelOnly,
    TriviallyCopyableMoveCounter,
)
from .words import (
    gen_word_pairs,
    words_ar,
    words_el,
    words_en,
    words_he,
    words_lv,
    words_pl,
    words_ru,
)
