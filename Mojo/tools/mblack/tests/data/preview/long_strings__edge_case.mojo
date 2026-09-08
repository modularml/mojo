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

# ===----------------------------------------------------------------------=== #
#
# File originates from:
#   Repo:   git@github.com:psf/black.git
#   Commit: d4a85643a465f5fae2113d07d22d021d4af4795a
#   Path:   tests/data/preview/long_strings__edge_case.py
#
# ===----------------------------------------------------------------------=== #

some_variable = (
    "This string is long but not so long that it needs to be split just yet"
)
some_variable = (
    "This string is long but not so long that it needs to be split just yet"
)
some_variable = (
    "This string is long, just long enough that it needs to be split, u get?"
)
some_variable = (
    "This string is long, just long enough that it needs to be split, u get?"
)
some_variable = (
    "This string is long, just long enough that it needs to be split, u get? So"
    " we stay"
)
some_variable = (
    "This string is long, just long enough that it needs to be split, u get? So"
    " we stay"
)
some_variable = (
    "This string is long, just long enough that it needs to be split, u get? So"
    " we split"
)
some_variable = (
    "This string is long, just long enough that it needs to be split, u get? So"
    " we split"
)
some_variable = (
    "This string is long but not so long that it needs hahahah toooooo be so"
    " greatttt {} that I just can't think of any more good words to say about"
    " it at alll".format("ha")
)
some_variable = (
    "This string is long but not so long that it needs hahahah toooooo be so"
    " greatttt {} that I just can't think of any more good words to say about"
    " it at allll".format("ha")
)
some_variable = (
    "This string is long but not so long that it needs hahahah toooooo be so"
    " greatttt {} that I just can't think of any more good words to say about"
    " it at alllllllllll".format("ha")
)
some_variable = (
    "This string is long but not so long that it needs hahahah toooooo be so"
    " greatttt {} that I just can't think of any more good words to say about"
    " it at allllllllllll".format("ha")
)
some_variable = (
    "This is a long string that will end with a method that is not calleddd"
    .format
)
addition_inside_tuple = (
    some_string_inside_a_variable
    + "Some string that is just long enough to cause a split to take"
    " place.............",
    xyz,
    "Some really long string that needs to get split eventually but I'm running"
    " out of things to say"
    + some_string_inside_a_variable,
)
addition_inside_tuple = (
    some_string_inside_a_variable
    + "Some string that is just long enough to cause a split to take"
    " place.............."
)
return (
    "Hi there. This is areally really reallllly long string that needs to be"
    " split!!!"
)
ternary_expression = (
    "Short String"
    if some_condition
    else (
        "This is a really long string that will eventually need to be split"
        " right here."
    )
)
return f"{x}/b/c/d/d/d/dadfjsadjsaidoaisjdsfjaofjdfijaidfjaodfjaoifjodjafojdoajaaaaaaaaaaa"
return f"{x}/b/c/d/d/d/dadfjsadjsaidoaisjdsfjaofjdfijaidfjaodfjaoifjodjafojdoajaaaaaaaaaaaa"
assert (
    str(result)
    == "This long string should be split at some point right close to or around"
    " hereeeeeee"
)
assert (
    str(result)
    < "This long string should be split at some point right close to or around"
    " hereeeeee"
)
assert (
    "A format string: %s"
    % "This long string should be split at some point right close to or around"
    " hereeeeeee"
    != result
)
msg += (
    "This long string should be wrapped in parens at some point right around"
    " hereeeee"
)
msg += (
    "This long string should be split at some point right close to or around"
    " hereeeeeeee"
)
msg += (
    "This long string should not be split at any point ever since it is just"
    " righttt"
)


# output


some_variable = (
    "This string is long but not so long that it needs to be split just yet"
)
some_variable = (
    "This string is long but not so long that it needs to be split just yet"
)
some_variable = (
    "This string is long, just long enough that it needs to be split, u get?"
)
some_variable = (
    "This string is long, just long enough that it needs to be split, u get?"
)
some_variable = (
    "This string is long, just long enough that it needs to be split, u get? So"
    " we stay"
)
some_variable = (
    "This string is long, just long enough that it needs to be split, u get? So"
    " we stay"
)
some_variable = (
    "This string is long, just long enough that it needs to be split, u get? So"
    " we split"
)
some_variable = (
    "This string is long, just long enough that it needs to be split, u get? So"
    " we split"
)
some_variable = (
    "This string is long but not so long that it needs hahahah toooooo be so"
    " greatttt {} that I just can't think of any more good words to say about"
    " it at alll".format("ha")
)
some_variable = (
    "This string is long but not so long that it needs hahahah toooooo be so"
    " greatttt {} that I just can't think of any more good words to say about"
    " it at allll".format("ha")
)
some_variable = (
    "This string is long but not so long that it needs hahahah toooooo be so"
    " greatttt {} that I just can't think of any more good words to say about"
    " it at alllllllllll".format("ha")
)
some_variable = (
    "This string is long but not so long that it needs hahahah toooooo be so"
    " greatttt {} that I just can't think of any more good words to say about"
    " it at allllllllllll".format("ha")
)
some_variable = (
    "This is a long string that will end with a method that is not calleddd"
    .format
)
addition_inside_tuple = (
    some_string_inside_a_variable
    + "Some string that is just long enough to cause a split to take"
    " place.............",
    xyz,
    "Some really long string that needs to get split eventually but I'm running"
    " out of things to say"
    + some_string_inside_a_variable,
)
addition_inside_tuple = (
    some_string_inside_a_variable
    + "Some string that is just long enough to cause a split to take"
    " place.............."
)
return (
    "Hi there. This is areally really reallllly long string that needs to be"
    " split!!!"
)
ternary_expression = "Short String" if some_condition else (
    "This is a really long string that will eventually need to be split"
    " right here."
)
return f"{x}/b/c/d/d/d/dadfjsadjsaidoaisjdsfjaofjdfijaidfjaodfjaoifjodjafojdoajaaaaaaaaaaa"
return f"{x}/b/c/d/d/d/dadfjsadjsaidoaisjdsfjaofjdfijaidfjaodfjaoifjodjafojdoajaaaaaaaaaaaa"
assert (
    str(result)
    == "This long string should be split at some point right close to or around"
    " hereeeeeee"
)
assert (
    str(result)
    < "This long string should be split at some point right close to or around"
    " hereeeeee"
)
assert (
    "A format string: %s"
    % "This long string should be split at some point right close to or around"
    " hereeeeeee"
    != result
)
msg += (
    "This long string should be wrapped in parens at some point right around"
    " hereeeee"
)
msg += (
    "This long string should be split at some point right close to or around"
    " hereeeeeeee"
)
msg += (
    "This long string should not be split at any point ever since it is just"
    " righttt"
)
