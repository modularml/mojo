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

from max.gpu.host import DeviceContext
from tensor_core_kernels import (
    test_load_and_mma_and_multiply_operands,
    test_load_operands_ldmatrix,
    test_write_res_operand,
)

from std.utils.index import Index


# CHECK-LABEL: == test_load_and_mma_f32_f16_16x16x16
# CHECK: thread 0 a_vals=[0 1 2 3], b_vals=[0 16 32 48], d_vals=[19841 50561 81281 112001]
# CHECK: thread 1 a_vals=[16 17 18 19], b_vals=[1 17 33 49], d_vals=[19961 50937 81913 112889]
# CHECK: thread 2 a_vals=[32 33 34 35], b_vals=[2 18 34 50], d_vals=[20081 51313 82545 113777]
# CHECK: thread 3 a_vals=[48 49 50 51], b_vals=[3 19 35 51], d_vals=[20201 51689 83177 114665]
# CHECK: thread 4 a_vals=[64 65 66 67], b_vals=[4 20 36 52], d_vals=[20321 52065 83809 115553]
# CHECK: thread 5 a_vals=[80 81 82 83], b_vals=[5 21 37 53], d_vals=[20441 52441 84441 116441]
# CHECK: thread 6 a_vals=[96 97 98 99], b_vals=[6 22 38 54], d_vals=[20561 52817 85073 117329]
# CHECK: thread 7 a_vals=[112 113 114 115], b_vals=[7 23 39 55], d_vals=[20681 53193 85705 118217]
# CHECK: thread 8 a_vals=[128 129 130 131], b_vals=[8 24 40 56], d_vals=[20801 53569 86337 119105]
# CHECK: thread 9 a_vals=[144 145 146 147], b_vals=[9 25 41 57], d_vals=[20921 53945 86969 119993]
# CHECK: thread 10 a_vals=[160 161 162 163], b_vals=[10 26 42 58], d_vals=[21041 54321 87601 120881]
# CHECK: thread 11 a_vals=[176 177 178 179], b_vals=[11 27 43 59], d_vals=[21161 54697 88233 121769]
# CHECK: thread 12 a_vals=[192 193 194 195], b_vals=[12 28 44 60], d_vals=[21281 55073 88865 122657]
# CHECK: thread 13 a_vals=[208 209 210 211], b_vals=[13 29 45 61], d_vals=[21401 55449 89497 123545]
# CHECK: thread 14 a_vals=[224 225 226 227], b_vals=[14 30 46 62], d_vals=[21521 55825 90129 124433]
# CHECK: thread 15 a_vals=[240 241 242 243], b_vals=[15 31 47 63], d_vals=[21641 56201 90761 125321]
# CHECK: thread 16 a_vals=[4 5 6 7], b_vals=[64 80 96 112], d_vals=[142721 173441 204161 234881]
# CHECK: thread 17 a_vals=[20 21 22 23], b_vals=[65 81 97 113], d_vals=[143865 174841 205817 236793]
# CHECK: thread 18 a_vals=[36 37 38 39], b_vals=[66 82 98 114], d_vals=[145009 176241 207473 238705]
# CHECK: thread 19 a_vals=[52 53 54 55], b_vals=[67 83 99 115], d_vals=[146153 177641 209129 240617]
# CHECK: thread 20 a_vals=[68 69 70 71], b_vals=[68 84 100 116], d_vals=[147297 179041 210785 242529]
# CHECK: thread 21 a_vals=[84 85 86 87], b_vals=[69 85 101 117], d_vals=[148441 180441 212441 244441]
# CHECK: thread 22 a_vals=[100 101 102 103], b_vals=[70 86 102 118], d_vals=[149585 181841 214097 246353]
# CHECK: thread 23 a_vals=[116 117 118 119], b_vals=[71 87 103 119], d_vals=[150729 183241 215753 248265]
# CHECK: thread 24 a_vals=[132 133 134 135], b_vals=[72 88 104 120], d_vals=[151873 184641 217409 250177]
# CHECK: thread 25 a_vals=[148 149 150 151], b_vals=[73 89 105 121], d_vals=[153017 186041 219065 252089]
# CHECK: thread 26 a_vals=[164 165 166 167], b_vals=[74 90 106 122], d_vals=[154161 187441 220721 254001]
# CHECK: thread 27 a_vals=[180 181 182 183], b_vals=[75 91 107 123], d_vals=[155305 188841 222377 255913]
# CHECK: thread 28 a_vals=[196 197 198 199], b_vals=[76 92 108 124], d_vals=[156449 190241 224033 257825]
# CHECK: thread 29 a_vals=[212 213 214 215], b_vals=[77 93 109 125], d_vals=[157593 191641 225689 259737]
# CHECK: thread 30 a_vals=[228 229 230 231], b_vals=[78 94 110 126], d_vals=[158737 193041 227345 261649]
# CHECK: thread 31 a_vals=[244 245 246 247], b_vals=[79 95 111 127], d_vals=[159881 194441 229001 263561]
# CHECK: thread 32 a_vals=[8 9 10 11], b_vals=[128 144 160 176], d_vals=[265601 296321 327041 357761]
# CHECK: thread 33 a_vals=[24 25 26 27], b_vals=[129 145 161 177], d_vals=[267769 298745 329721 360697]
# CHECK: thread 34 a_vals=[40 41 42 43], b_vals=[130 146 162 178], d_vals=[269937 301169 332401 363633]
# CHECK: thread 35 a_vals=[56 57 58 59], b_vals=[131 147 163 179], d_vals=[272105 303593 335081 366569]
# CHECK: thread 36 a_vals=[72 73 74 75], b_vals=[132 148 164 180], d_vals=[274273 306017 337761 369505]
# CHECK: thread 37 a_vals=[88 89 90 91], b_vals=[133 149 165 181], d_vals=[276441 308441 340441 372441]
# CHECK: thread 38 a_vals=[104 105 106 107], b_vals=[134 150 166 182], d_vals=[278609 310865 343121 375377]
# CHECK: thread 39 a_vals=[120 121 122 123], b_vals=[135 151 167 183], d_vals=[280777 313289 345801 378313]
# CHECK: thread 40 a_vals=[136 137 138 139], b_vals=[136 152 168 184], d_vals=[282945 315713 348481 381249]
# CHECK: thread 41 a_vals=[152 153 154 155], b_vals=[137 153 169 185], d_vals=[285113 318137 351161 384185]
# CHECK: thread 42 a_vals=[168 169 170 171], b_vals=[138 154 170 186], d_vals=[287281 320561 353841 387121]
# CHECK: thread 43 a_vals=[184 185 186 187], b_vals=[139 155 171 187], d_vals=[289449 322985 356521 390057]
# CHECK: thread 44 a_vals=[200 201 202 203], b_vals=[140 156 172 188], d_vals=[291617 325409 359201 392993]
# CHECK: thread 45 a_vals=[216 217 218 219], b_vals=[141 157 173 189], d_vals=[293785 327833 361881 395929]
# CHECK: thread 46 a_vals=[232 233 234 235], b_vals=[142 158 174 190], d_vals=[295953 330257 364561 398865]
# CHECK: thread 47 a_vals=[248 249 250 251], b_vals=[143 159 175 191], d_vals=[298121 332681 367241 401801]
# CHECK: thread 48 a_vals=[12 13 14 15], b_vals=[192 208 224 240], d_vals=[388481 419201 449921 480641]
# CHECK: thread 49 a_vals=[28 29 30 31], b_vals=[193 209 225 241], d_vals=[391673 422649 453625 484601]
# CHECK: thread 50 a_vals=[44 45 46 47], b_vals=[194 210 226 242], d_vals=[394865 426097 457329 488561]
# CHECK: thread 51 a_vals=[60 61 62 63], b_vals=[195 211 227 243], d_vals=[398057 429545 461033 492521]
# CHECK: thread 52 a_vals=[76 77 78 79], b_vals=[196 212 228 244], d_vals=[401249 432993 464737 496481]
# CHECK: thread 53 a_vals=[92 93 94 95], b_vals=[197 213 229 245], d_vals=[404441 436441 468441 500441]
# CHECK: thread 54 a_vals=[108 109 110 111], b_vals=[198 214 230 246], d_vals=[407633 439889 472145 504401]
# CHECK: thread 55 a_vals=[124 125 126 127], b_vals=[199 215 231 247], d_vals=[410825 443337 475849 508361]
# CHECK: thread 56 a_vals=[140 141 142 143], b_vals=[200 216 232 248], d_vals=[414017 446785 479553 512321]
# CHECK: thread 57 a_vals=[156 157 158 159], b_vals=[201 217 233 249], d_vals=[417209 450233 483257 516281]
# CHECK: thread 58 a_vals=[172 173 174 175], b_vals=[202 218 234 250], d_vals=[420401 453681 486961 520241]
# CHECK: thread 59 a_vals=[188 189 190 191], b_vals=[203 219 235 251], d_vals=[423593 457129 490665 524201]
# CHECK: thread 60 a_vals=[204 205 206 207], b_vals=[204 220 236 252], d_vals=[426785 460577 494369 528161]
# CHECK: thread 61 a_vals=[220 221 222 223], b_vals=[205 221 237 253], d_vals=[429977 464025 498073 532121]
# CHECK: thread 62 a_vals=[236 237 238 239], b_vals=[206 222 238 254], d_vals=[433169 467473 501777 536081]
# CHECK: thread 63 a_vals=[252 253 254 255], b_vals=[207 223 239 255], d_vals=[436361 470921 505481 540041]
def test_load_and_mma_f32_f16_16x16x16(ctx: DeviceContext) raises:
    print("== test_load_and_mma_f32_f16_16x16x16")
    test_load_and_mma_and_multiply_operands[
        DType.float32, DType.float16, Index(16, 16, 16)
    ](ctx)


# CHECK-LABEL: == test_load_and_mma_f32_bf16_16x16x16
# CHECK: thread 0 a_vals=[0 1 2 3], b_vals=[0 16 32 48], d_vals=[19841 50561 81281 112001]
# CHECK: thread 1 a_vals=[16 17 18 19], b_vals=[1 17 33 49], d_vals=[19961 50937 81913 112889]
# CHECK: thread 2 a_vals=[32 33 34 35], b_vals=[2 18 34 50], d_vals=[20081 51313 82545 113777]
# CHECK: thread 3 a_vals=[48 49 50 51], b_vals=[3 19 35 51], d_vals=[20201 51689 83177 114665]
# CHECK: thread 4 a_vals=[64 65 66 67], b_vals=[4 20 36 52], d_vals=[20321 52065 83809 115553]
# CHECK: thread 5 a_vals=[80 81 82 83], b_vals=[5 21 37 53], d_vals=[20441 52441 84441 116441]
# CHECK: thread 6 a_vals=[96 97 98 99], b_vals=[6 22 38 54], d_vals=[20561 52817 85073 117329]
# CHECK: thread 7 a_vals=[112 113 114 115], b_vals=[7 23 39 55], d_vals=[20681 53193 85705 118217]
# CHECK: thread 8 a_vals=[128 129 130 131], b_vals=[8 24 40 56], d_vals=[20801 53569 86337 119105]
# CHECK: thread 9 a_vals=[144 145 146 147], b_vals=[9 25 41 57], d_vals=[20921 53945 86969 119993]
# CHECK: thread 10 a_vals=[160 161 162 163], b_vals=[10 26 42 58], d_vals=[21041 54321 87601 120881]
# CHECK: thread 11 a_vals=[176 177 178 179], b_vals=[11 27 43 59], d_vals=[21161 54697 88233 121769]
# CHECK: thread 12 a_vals=[192 193 194 195], b_vals=[12 28 44 60], d_vals=[21281 55073 88865 122657]
# CHECK: thread 13 a_vals=[208 209 210 211], b_vals=[13 29 45 61], d_vals=[21401 55449 89497 123545]
# CHECK: thread 14 a_vals=[224 225 226 227], b_vals=[14 30 46 62], d_vals=[21521 55825 90129 124433]
# CHECK: thread 15 a_vals=[240 241 242 243], b_vals=[15 31 47 63], d_vals=[21641 56201 90761 125321]
# CHECK: thread 16 a_vals=[4 5 6 7], b_vals=[64 80 96 112], d_vals=[142721 173441 204161 234881]
# CHECK: thread 17 a_vals=[20 21 22 23], b_vals=[65 81 97 113], d_vals=[143865 174841 205817 236793]
# CHECK: thread 18 a_vals=[36 37 38 39], b_vals=[66 82 98 114], d_vals=[145009 176241 207473 238705]
# CHECK: thread 19 a_vals=[52 53 54 55], b_vals=[67 83 99 115], d_vals=[146153 177641 209129 240617]
# CHECK: thread 20 a_vals=[68 69 70 71], b_vals=[68 84 100 116], d_vals=[147297 179041 210785 242529]
# CHECK: thread 21 a_vals=[84 85 86 87], b_vals=[69 85 101 117], d_vals=[148441 180441 212441 244441]
# CHECK: thread 22 a_vals=[100 101 102 103], b_vals=[70 86 102 118], d_vals=[149585 181841 214097 246353]
# CHECK: thread 23 a_vals=[116 117 118 119], b_vals=[71 87 103 119], d_vals=[150729 183241 215753 248265]
# CHECK: thread 24 a_vals=[132 133 134 135], b_vals=[72 88 104 120], d_vals=[151873 184641 217409 250177]
# CHECK: thread 25 a_vals=[148 149 150 151], b_vals=[73 89 105 121], d_vals=[153017 186041 219065 252089]
# CHECK: thread 26 a_vals=[164 165 166 167], b_vals=[74 90 106 122], d_vals=[154161 187441 220721 254001]
# CHECK: thread 27 a_vals=[180 181 182 183], b_vals=[75 91 107 123], d_vals=[155305 188841 222377 255913]
# CHECK: thread 28 a_vals=[196 197 198 199], b_vals=[76 92 108 124], d_vals=[156449 190241 224033 257825]
# CHECK: thread 29 a_vals=[212 213 214 215], b_vals=[77 93 109 125], d_vals=[157593 191641 225689 259737]
# CHECK: thread 30 a_vals=[228 229 230 231], b_vals=[78 94 110 126], d_vals=[158737 193041 227345 261649]
# CHECK: thread 31 a_vals=[244 245 246 247], b_vals=[79 95 111 127], d_vals=[159881 194441 229001 263561]
# CHECK: thread 32 a_vals=[8 9 10 11], b_vals=[128 144 160 176], d_vals=[265601 296321 327041 357761]
# CHECK: thread 33 a_vals=[24 25 26 27], b_vals=[129 145 161 177], d_vals=[267769 298745 329721 360697]
# CHECK: thread 34 a_vals=[40 41 42 43], b_vals=[130 146 162 178], d_vals=[269937 301169 332401 363633]
# CHECK: thread 35 a_vals=[56 57 58 59], b_vals=[131 147 163 179], d_vals=[272105 303593 335081 366569]
# CHECK: thread 36 a_vals=[72 73 74 75], b_vals=[132 148 164 180], d_vals=[274273 306017 337761 369505]
# CHECK: thread 37 a_vals=[88 89 90 91], b_vals=[133 149 165 181], d_vals=[276441 308441 340441 372441]
# CHECK: thread 38 a_vals=[104 105 106 107], b_vals=[134 150 166 182], d_vals=[278609 310865 343121 375377]
# CHECK: thread 39 a_vals=[120 121 122 123], b_vals=[135 151 167 183], d_vals=[280777 313289 345801 378313]
# CHECK: thread 40 a_vals=[136 137 138 139], b_vals=[136 152 168 184], d_vals=[282945 315713 348481 381249]
# CHECK: thread 41 a_vals=[152 153 154 155], b_vals=[137 153 169 185], d_vals=[285113 318137 351161 384185]
# CHECK: thread 42 a_vals=[168 169 170 171], b_vals=[138 154 170 186], d_vals=[287281 320561 353841 387121]
# CHECK: thread 43 a_vals=[184 185 186 187], b_vals=[139 155 171 187], d_vals=[289449 322985 356521 390057]
# CHECK: thread 44 a_vals=[200 201 202 203], b_vals=[140 156 172 188], d_vals=[291617 325409 359201 392993]
# CHECK: thread 45 a_vals=[216 217 218 219], b_vals=[141 157 173 189], d_vals=[293785 327833 361881 395929]
# CHECK: thread 46 a_vals=[232 233 234 235], b_vals=[142 158 174 190], d_vals=[295953 330257 364561 398865]
# CHECK: thread 47 a_vals=[248 249 250 251], b_vals=[143 159 175 191], d_vals=[298121 332681 367241 401801]
# CHECK: thread 48 a_vals=[12 13 14 15], b_vals=[192 208 224 240], d_vals=[388481 419201 449921 480641]
# CHECK: thread 49 a_vals=[28 29 30 31], b_vals=[193 209 225 241], d_vals=[391673 422649 453625 484601]
# CHECK: thread 50 a_vals=[44 45 46 47], b_vals=[194 210 226 242], d_vals=[394865 426097 457329 488561]
# CHECK: thread 51 a_vals=[60 61 62 63], b_vals=[195 211 227 243], d_vals=[398057 429545 461033 492521]
# CHECK: thread 52 a_vals=[76 77 78 79], b_vals=[196 212 228 244], d_vals=[401249 432993 464737 496481]
# CHECK: thread 53 a_vals=[92 93 94 95], b_vals=[197 213 229 245], d_vals=[404441 436441 468441 500441]
# CHECK: thread 54 a_vals=[108 109 110 111], b_vals=[198 214 230 246], d_vals=[407633 439889 472145 504401]
# CHECK: thread 55 a_vals=[124 125 126 127], b_vals=[199 215 231 247], d_vals=[410825 443337 475849 508361]
# CHECK: thread 56 a_vals=[140 141 142 143], b_vals=[200 216 232 248], d_vals=[414017 446785 479553 512321]
# CHECK: thread 57 a_vals=[156 157 158 159], b_vals=[201 217 233 249], d_vals=[417209 450233 483257 516281]
# CHECK: thread 58 a_vals=[172 173 174 175], b_vals=[202 218 234 250], d_vals=[420401 453681 486961 520241]
# CHECK: thread 59 a_vals=[188 189 190 191], b_vals=[203 219 235 251], d_vals=[423593 457129 490665 524201]
# CHECK: thread 60 a_vals=[204 205 206 207], b_vals=[204 220 236 252], d_vals=[426785 460577 494369 528161]
# CHECK: thread 61 a_vals=[220 221 222 223], b_vals=[205 221 237 253], d_vals=[429977 464025 498073 532121]
# CHECK: thread 62 a_vals=[236 237 238 239], b_vals=[206 222 238 254], d_vals=[433169 467473 501777 536081]
# CHECK: thread 63 a_vals=[252 253 254 255], b_vals=[207 223 239 255], d_vals=[436361 470921 505481 540041]
def test_load_and_mma_f32_bf16_16x16x16(ctx: DeviceContext) raises:
    print("== test_load_and_mma_f32_bf16_16x16x16")
    test_load_and_mma_and_multiply_operands[
        DType.float32, DType.bfloat16, Index(16, 16, 16)
    ](ctx)


# CHECK-LABEL: == test_load_and_mma_f32_f32_16x16x4
# CHECK: thread 0 a_vals=[0], b_vals=[0], d_vals=[225 609 993 1377]
# CHECK: thread 1 a_vals=[4], b_vals=[1], d_vals=[231 631 1031 1431]
# CHECK: thread 2 a_vals=[8], b_vals=[2], d_vals=[237 653 1069 1485]
# CHECK: thread 3 a_vals=[12], b_vals=[3], d_vals=[243 675 1107 1539]
# CHECK: thread 4 a_vals=[16], b_vals=[4], d_vals=[249 697 1145 1593]
# CHECK: thread 5 a_vals=[20], b_vals=[5], d_vals=[255 719 1183 1647]
# CHECK: thread 6 a_vals=[24], b_vals=[6], d_vals=[261 741 1221 1701]
# CHECK: thread 7 a_vals=[28], b_vals=[7], d_vals=[267 763 1259 1755]
# CHECK: thread 8 a_vals=[32], b_vals=[8], d_vals=[273 785 1297 1809]
# CHECK: thread 9 a_vals=[36], b_vals=[9], d_vals=[279 807 1335 1863]
# CHECK: thread 10 a_vals=[40], b_vals=[10], d_vals=[285 829 1373 1917]
# CHECK: thread 11 a_vals=[44], b_vals=[11], d_vals=[291 851 1411 1971]
# CHECK: thread 12 a_vals=[48], b_vals=[12], d_vals=[297 873 1449 2025]
# CHECK: thread 13 a_vals=[52], b_vals=[13], d_vals=[303 895 1487 2079]
# CHECK: thread 14 a_vals=[56], b_vals=[14], d_vals=[309 917 1525 2133]
# CHECK: thread 15 a_vals=[60], b_vals=[15], d_vals=[315 939 1563 2187]
# CHECK: thread 16 a_vals=[1], b_vals=[16], d_vals=[1761 2145 2529 2913]
# CHECK: thread 17 a_vals=[5], b_vals=[17], d_vals=[1831 2231 2631 3031]
# CHECK: thread 18 a_vals=[9], b_vals=[18], d_vals=[1901 2317 2733 3149]
# CHECK: thread 19 a_vals=[13], b_vals=[19], d_vals=[1971 2403 2835 3267]
# CHECK: thread 20 a_vals=[17], b_vals=[20], d_vals=[2041 2489 2937 3385]
# CHECK: thread 21 a_vals=[21], b_vals=[21], d_vals=[2111 2575 3039 3503]
# CHECK: thread 22 a_vals=[25], b_vals=[22], d_vals=[2181 2661 3141 3621]
# CHECK: thread 23 a_vals=[29], b_vals=[23], d_vals=[2251 2747 3243 3739]
# CHECK: thread 24 a_vals=[33], b_vals=[24], d_vals=[2321 2833 3345 3857]
# CHECK: thread 25 a_vals=[37], b_vals=[25], d_vals=[2391 2919 3447 3975]
# CHECK: thread 26 a_vals=[41], b_vals=[26], d_vals=[2461 3005 3549 4093]
# CHECK: thread 27 a_vals=[45], b_vals=[27], d_vals=[2531 3091 3651 4211]
# CHECK: thread 28 a_vals=[49], b_vals=[28], d_vals=[2601 3177 3753 4329]
# CHECK: thread 29 a_vals=[53], b_vals=[29], d_vals=[2671 3263 3855 4447]
# CHECK: thread 30 a_vals=[57], b_vals=[30], d_vals=[2741 3349 3957 4565]
# CHECK: thread 31 a_vals=[61], b_vals=[31], d_vals=[2811 3435 4059 4683]
# CHECK: thread 32 a_vals=[2], b_vals=[32], d_vals=[3297 3681 4065 4449]
# CHECK: thread 33 a_vals=[6], b_vals=[33], d_vals=[3431 3831 4231 4631]
# CHECK: thread 34 a_vals=[10], b_vals=[34], d_vals=[3565 3981 4397 4813]
# CHECK: thread 35 a_vals=[14], b_vals=[35], d_vals=[3699 4131 4563 4995]
# CHECK: thread 36 a_vals=[18], b_vals=[36], d_vals=[3833 4281 4729 5177]
# CHECK: thread 37 a_vals=[22], b_vals=[37], d_vals=[3967 4431 4895 5359]
# CHECK: thread 38 a_vals=[26], b_vals=[38], d_vals=[4101 4581 5061 5541]
# CHECK: thread 39 a_vals=[30], b_vals=[39], d_vals=[4235 4731 5227 5723]
# CHECK: thread 40 a_vals=[34], b_vals=[40], d_vals=[4369 4881 5393 5905]
# CHECK: thread 41 a_vals=[38], b_vals=[41], d_vals=[4503 5031 5559 6087]
# CHECK: thread 42 a_vals=[42], b_vals=[42], d_vals=[4637 5181 5725 6269]
# CHECK: thread 43 a_vals=[46], b_vals=[43], d_vals=[4771 5331 5891 6451]
# CHECK: thread 44 a_vals=[50], b_vals=[44], d_vals=[4905 5481 6057 6633]
# CHECK: thread 45 a_vals=[54], b_vals=[45], d_vals=[5039 5631 6223 6815]
# CHECK: thread 46 a_vals=[58], b_vals=[46], d_vals=[5173 5781 6389 6997]
# CHECK: thread 47 a_vals=[62], b_vals=[47], d_vals=[5307 5931 6555 7179]
# CHECK: thread 48 a_vals=[3], b_vals=[48], d_vals=[4833 5217 5601 5985]
# CHECK: thread 49 a_vals=[7], b_vals=[49], d_vals=[5031 5431 5831 6231]
# CHECK: thread 50 a_vals=[11], b_vals=[50], d_vals=[5229 5645 6061 6477]
# CHECK: thread 51 a_vals=[15], b_vals=[51], d_vals=[5427 5859 6291 6723]
# CHECK: thread 52 a_vals=[19], b_vals=[52], d_vals=[5625 6073 6521 6969]
# CHECK: thread 53 a_vals=[23], b_vals=[53], d_vals=[5823 6287 6751 7215]
# CHECK: thread 54 a_vals=[27], b_vals=[54], d_vals=[6021 6501 6981 7461]
# CHECK: thread 55 a_vals=[31], b_vals=[55], d_vals=[6219 6715 7211 7707]
# CHECK: thread 56 a_vals=[35], b_vals=[56], d_vals=[6417 6929 7441 7953]
# CHECK: thread 57 a_vals=[39], b_vals=[57], d_vals=[6615 7143 7671 8199]
# CHECK: thread 58 a_vals=[43], b_vals=[58], d_vals=[6813 7357 7901 8445]
# CHECK: thread 59 a_vals=[47], b_vals=[59], d_vals=[7011 7571 8131 8691]
# CHECK: thread 60 a_vals=[51], b_vals=[60], d_vals=[7209 7785 8361 8937]
# CHECK: thread 61 a_vals=[55], b_vals=[61], d_vals=[7407 7999 8591 9183]
# CHECK: thread 62 a_vals=[59], b_vals=[62], d_vals=[7605 8213 8821 9429]
# CHECK: thread 63 a_vals=[63], b_vals=[63], d_vals=[7803 8427 9051 9675]
def test_load_and_mma_f32_f32_16x16x4(ctx: DeviceContext) raises:
    print("== test_load_and_mma_f32_f32_16x16x4")
    test_load_and_mma_and_multiply_operands[
        DType.float32, DType.float32, Index(16, 16, 4)
    ](ctx)


# CHECK-LABEL: test_write_f32_f32_16x16x4
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
# CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
# CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
# CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
# CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
# CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
# CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
# CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
# CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
# CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
# CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
# CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
# CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
def test_write_f32_f32_16x16x4(ctx: DeviceContext) raises:
    print("== test_write_f32_f32_16x16x4")
    test_write_res_operand[.float32, DType.float32, Index(16, 16, 4)](ctx)


# CHECK-LABEL: test_write_f32_f16_16x16x16
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
# CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
# CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
# CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
# CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
# CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
# CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
# CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
# CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
# CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
# CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
# CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
# CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
def test_write_f32_f16_16x16x16(ctx: DeviceContext) raises:
    print("== test_write_f32_f16_16x16x16")
    test_write_res_operand[.float32, DType.float16, Index(16, 16, 16)](ctx)


# CHECK-LABEL: test_write_f32_bf16_16x16x16
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
# CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
# CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
# CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
# CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
# CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
# CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
# CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
# CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
# CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
# CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
# CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
# CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
def test_write_f32_bf16_16x16x16(ctx: DeviceContext) raises:
    print("== test_write_f32_bf16_16x16x16")
    test_write_res_operand[.float32, DType.bfloat16, Index(16, 16, 16)](ctx)


# CHECK-LABEL: test_load_f32_f16_16x16x16_ldmatrix
# CHECK: thread 0 a_vals=[0 1 2 3], b_vals=[0 16 32 48]
# CHECK: thread 1 a_vals=[16 17 18 19], b_vals=[1 17 33 49]
# CHECK: thread 2 a_vals=[32 33 34 35], b_vals=[2 18 34 50]
# CHECK: thread 3 a_vals=[48 49 50 51], b_vals=[3 19 35 51]
# CHECK: thread 4 a_vals=[64 65 66 67], b_vals=[4 20 36 52]
# CHECK: thread 5 a_vals=[80 81 82 83], b_vals=[5 21 37 53]
# CHECK: thread 6 a_vals=[96 97 98 99], b_vals=[6 22 38 54]
# CHECK: thread 7 a_vals=[112 113 114 115], b_vals=[7 23 39 55]
# CHECK: thread 8 a_vals=[128 129 130 131], b_vals=[8 24 40 56]
# CHECK: thread 9 a_vals=[144 145 146 147], b_vals=[9 25 41 57]
# CHECK: thread 10 a_vals=[160 161 162 163], b_vals=[10 26 42 58]
# CHECK: thread 11 a_vals=[176 177 178 179], b_vals=[11 27 43 59]
# CHECK: thread 12 a_vals=[192 193 194 195], b_vals=[12 28 44 60]
# CHECK: thread 13 a_vals=[208 209 210 211], b_vals=[13 29 45 61]
# CHECK: thread 14 a_vals=[224 225 226 227], b_vals=[14 30 46 62]
# CHECK: thread 15 a_vals=[240 241 242 243], b_vals=[15 31 47 63]
# CHECK: thread 16 a_vals=[4 5 6 7], b_vals=[64 80 96 112]
# CHECK: thread 17 a_vals=[20 21 22 23], b_vals=[65 81 97 113]
# CHECK: thread 18 a_vals=[36 37 38 39], b_vals=[66 82 98 114]
# CHECK: thread 19 a_vals=[52 53 54 55], b_vals=[67 83 99 115]
# CHECK: thread 20 a_vals=[68 69 70 71], b_vals=[68 84 100 116]
# CHECK: thread 21 a_vals=[84 85 86 87], b_vals=[69 85 101 117]
# CHECK: thread 22 a_vals=[100 101 102 103], b_vals=[70 86 102 118]
# CHECK: thread 23 a_vals=[116 117 118 119], b_vals=[71 87 103 119]
# CHECK: thread 24 a_vals=[132 133 134 135], b_vals=[72 88 104 120]
# CHECK: thread 25 a_vals=[148 149 150 151], b_vals=[73 89 105 121]
# CHECK: thread 26 a_vals=[164 165 166 167], b_vals=[74 90 106 122]
# CHECK: thread 27 a_vals=[180 181 182 183], b_vals=[75 91 107 123]
# CHECK: thread 28 a_vals=[196 197 198 199], b_vals=[76 92 108 124]
# CHECK: thread 29 a_vals=[212 213 214 215], b_vals=[77 93 109 125]
# CHECK: thread 30 a_vals=[228 229 230 231], b_vals=[78 94 110 126]
# CHECK: thread 31 a_vals=[244 245 246 247], b_vals=[79 95 111 127]
# CHECK: thread 32 a_vals=[8 9 10 11], b_vals=[128 144 160 176]
# CHECK: thread 33 a_vals=[24 25 26 27], b_vals=[129 145 161 177]
# CHECK: thread 34 a_vals=[40 41 42 43], b_vals=[130 146 162 178]
# CHECK: thread 35 a_vals=[56 57 58 59], b_vals=[131 147 163 179]
# CHECK: thread 36 a_vals=[72 73 74 75], b_vals=[132 148 164 180]
# CHECK: thread 37 a_vals=[88 89 90 91], b_vals=[133 149 165 181]
# CHECK: thread 38 a_vals=[104 105 106 107], b_vals=[134 150 166 182]
# CHECK: thread 39 a_vals=[120 121 122 123], b_vals=[135 151 167 183]
# CHECK: thread 40 a_vals=[136 137 138 139], b_vals=[136 152 168 184]
# CHECK: thread 41 a_vals=[152 153 154 155], b_vals=[137 153 169 185]
# CHECK: thread 42 a_vals=[168 169 170 171], b_vals=[138 154 170 186]
# CHECK: thread 43 a_vals=[184 185 186 187], b_vals=[139 155 171 187]
# CHECK: thread 44 a_vals=[200 201 202 203], b_vals=[140 156 172 188]
# CHECK: thread 45 a_vals=[216 217 218 219], b_vals=[141 157 173 189]
# CHECK: thread 46 a_vals=[232 233 234 235], b_vals=[142 158 174 190]
# CHECK: thread 47 a_vals=[248 249 250 251], b_vals=[143 159 175 191]
# CHECK: thread 48 a_vals=[12 13 14 15], b_vals=[192 208 224 240]
# CHECK: thread 49 a_vals=[28 29 30 31], b_vals=[193 209 225 241]
# CHECK: thread 50 a_vals=[44 45 46 47], b_vals=[194 210 226 242]
# CHECK: thread 51 a_vals=[60 61 62 63], b_vals=[195 211 227 243]
# CHECK: thread 52 a_vals=[76 77 78 79], b_vals=[196 212 228 244]
# CHECK: thread 53 a_vals=[92 93 94 95], b_vals=[197 213 229 245]
# CHECK: thread 54 a_vals=[108 109 110 111], b_vals=[198 214 230 246]
# CHECK: thread 55 a_vals=[124 125 126 127], b_vals=[199 215 231 247]
# CHECK: thread 56 a_vals=[140 141 142 143], b_vals=[200 216 232 248]
# CHECK: thread 57 a_vals=[156 157 158 159], b_vals=[201 217 233 249]
# CHECK: thread 58 a_vals=[172 173 174 175], b_vals=[202 218 234 250]
# CHECK: thread 59 a_vals=[188 189 190 191], b_vals=[203 219 235 251]
# CHECK: thread 60 a_vals=[204 205 206 207], b_vals=[204 220 236 252]
# CHECK: thread 61 a_vals=[220 221 222 223], b_vals=[205 221 237 253]
# CHECK: thread 62 a_vals=[236 237 238 239], b_vals=[206 222 238 254]
# CHECK: thread 63 a_vals=[252 253 254 255], b_vals=[207 223 239 255]
def test_load_f32_f16_16x16x16_ldmatrix(ctx: DeviceContext) raises:
    print("== test_load_f32_f16_16x16x16_ldmatrix")
    test_load_operands_ldmatrix[
        DType.float32, DType.float16, Index(16, 16, 16)
    ](ctx)


# CHECK-LABEL: test_load_f32_bf16_16x16x16_ldmatrix
# CHECK: thread 0 a_vals=[0 1 2 3], b_vals=[0 16 32 48]
# CHECK: thread 1 a_vals=[16 17 18 19], b_vals=[1 17 33 49]
# CHECK: thread 2 a_vals=[32 33 34 35], b_vals=[2 18 34 50]
# CHECK: thread 3 a_vals=[48 49 50 51], b_vals=[3 19 35 51]
# CHECK: thread 4 a_vals=[64 65 66 67], b_vals=[4 20 36 52]
# CHECK: thread 5 a_vals=[80 81 82 83], b_vals=[5 21 37 53]
# CHECK: thread 6 a_vals=[96 97 98 99], b_vals=[6 22 38 54]
# CHECK: thread 7 a_vals=[112 113 114 115], b_vals=[7 23 39 55]
# CHECK: thread 8 a_vals=[128 129 130 131], b_vals=[8 24 40 56]
# CHECK: thread 9 a_vals=[144 145 146 147], b_vals=[9 25 41 57]
# CHECK: thread 10 a_vals=[160 161 162 163], b_vals=[10 26 42 58]
# CHECK: thread 11 a_vals=[176 177 178 179], b_vals=[11 27 43 59]
# CHECK: thread 12 a_vals=[192 193 194 195], b_vals=[12 28 44 60]
# CHECK: thread 13 a_vals=[208 209 210 211], b_vals=[13 29 45 61]
# CHECK: thread 14 a_vals=[224 225 226 227], b_vals=[14 30 46 62]
# CHECK: thread 15 a_vals=[240 241 242 243], b_vals=[15 31 47 63]
# CHECK: thread 16 a_vals=[4 5 6 7], b_vals=[64 80 96 112]
# CHECK: thread 17 a_vals=[20 21 22 23], b_vals=[65 81 97 113]
# CHECK: thread 18 a_vals=[36 37 38 39], b_vals=[66 82 98 114]
# CHECK: thread 19 a_vals=[52 53 54 55], b_vals=[67 83 99 115]
# CHECK: thread 20 a_vals=[68 69 70 71], b_vals=[68 84 100 116]
# CHECK: thread 21 a_vals=[84 85 86 87], b_vals=[69 85 101 117]
# CHECK: thread 22 a_vals=[100 101 102 103], b_vals=[70 86 102 118]
# CHECK: thread 23 a_vals=[116 117 118 119], b_vals=[71 87 103 119]
# CHECK: thread 24 a_vals=[132 133 134 135], b_vals=[72 88 104 120]
# CHECK: thread 25 a_vals=[148 149 150 151], b_vals=[73 89 105 121]
# CHECK: thread 26 a_vals=[164 165 166 167], b_vals=[74 90 106 122]
# CHECK: thread 27 a_vals=[180 181 182 183], b_vals=[75 91 107 123]
# CHECK: thread 28 a_vals=[196 197 198 199], b_vals=[76 92 108 124]
# CHECK: thread 29 a_vals=[212 213 214 215], b_vals=[77 93 109 125]
# CHECK: thread 30 a_vals=[228 229 230 231], b_vals=[78 94 110 126]
# CHECK: thread 31 a_vals=[244 245 246 247], b_vals=[79 95 111 127]
# CHECK: thread 32 a_vals=[8 9 10 11], b_vals=[128 144 160 176]
# CHECK: thread 33 a_vals=[24 25 26 27], b_vals=[129 145 161 177]
# CHECK: thread 34 a_vals=[40 41 42 43], b_vals=[130 146 162 178]
# CHECK: thread 35 a_vals=[56 57 58 59], b_vals=[131 147 163 179]
# CHECK: thread 36 a_vals=[72 73 74 75], b_vals=[132 148 164 180]
# CHECK: thread 37 a_vals=[88 89 90 91], b_vals=[133 149 165 181]
# CHECK: thread 38 a_vals=[104 105 106 107], b_vals=[134 150 166 182]
# CHECK: thread 39 a_vals=[120 121 122 123], b_vals=[135 151 167 183]
# CHECK: thread 40 a_vals=[136 137 138 139], b_vals=[136 152 168 184]
# CHECK: thread 41 a_vals=[152 153 154 155], b_vals=[137 153 169 185]
# CHECK: thread 42 a_vals=[168 169 170 171], b_vals=[138 154 170 186]
# CHECK: thread 43 a_vals=[184 185 186 187], b_vals=[139 155 171 187]
# CHECK: thread 44 a_vals=[200 201 202 203], b_vals=[140 156 172 188]
# CHECK: thread 45 a_vals=[216 217 218 219], b_vals=[141 157 173 189]
# CHECK: thread 46 a_vals=[232 233 234 235], b_vals=[142 158 174 190]
# CHECK: thread 47 a_vals=[248 249 250 251], b_vals=[143 159 175 191]
# CHECK: thread 48 a_vals=[12 13 14 15], b_vals=[192 208 224 240]
# CHECK: thread 49 a_vals=[28 29 30 31], b_vals=[193 209 225 241]
# CHECK: thread 50 a_vals=[44 45 46 47], b_vals=[194 210 226 242]
# CHECK: thread 51 a_vals=[60 61 62 63], b_vals=[195 211 227 243]
# CHECK: thread 52 a_vals=[76 77 78 79], b_vals=[196 212 228 244]
# CHECK: thread 53 a_vals=[92 93 94 95], b_vals=[197 213 229 245]
# CHECK: thread 54 a_vals=[108 109 110 111], b_vals=[198 214 230 246]
# CHECK: thread 55 a_vals=[124 125 126 127], b_vals=[199 215 231 247]
# CHECK: thread 56 a_vals=[140 141 142 143], b_vals=[200 216 232 248]
# CHECK: thread 57 a_vals=[156 157 158 159], b_vals=[201 217 233 249]
# CHECK: thread 58 a_vals=[172 173 174 175], b_vals=[202 218 234 250]
# CHECK: thread 59 a_vals=[188 189 190 191], b_vals=[203 219 235 251]
# CHECK: thread 60 a_vals=[204 205 206 207], b_vals=[204 220 236 252]
# CHECK: thread 61 a_vals=[220 221 222 223], b_vals=[205 221 237 253]
# CHECK: thread 62 a_vals=[236 237 238 239], b_vals=[206 222 238 254]
# CHECK: thread 63 a_vals=[252 253 254 255], b_vals=[207 223 239 255]
def test_load_f32_bf16_16x16x16_ldmatrix(ctx: DeviceContext) raises:
    print("== test_load_f32_bf16_16x16x16_ldmatrix")
    test_load_operands_ldmatrix[
        DType.float32, DType.bfloat16, Index(16, 16, 16)
    ](ctx)


# CHECK-LABEL: test_load_f32_f32_16x16x4_ldmatrix
# CHECK: thread 0 a_vals=[0], b_vals=[0]
# CHECK: thread 1 a_vals=[4], b_vals=[1]
# CHECK: thread 2 a_vals=[8], b_vals=[2]
# CHECK: thread 3 a_vals=[12], b_vals=[3]
# CHECK: thread 4 a_vals=[16], b_vals=[4]
# CHECK: thread 5 a_vals=[20], b_vals=[5]
# CHECK: thread 6 a_vals=[24], b_vals=[6]
# CHECK: thread 7 a_vals=[28], b_vals=[7]
# CHECK: thread 8 a_vals=[32], b_vals=[8]
# CHECK: thread 9 a_vals=[36], b_vals=[9]
# CHECK: thread 10 a_vals=[40], b_vals=[10]
# CHECK: thread 11 a_vals=[44], b_vals=[11]
# CHECK: thread 12 a_vals=[48], b_vals=[12]
# CHECK: thread 13 a_vals=[52], b_vals=[13]
# CHECK: thread 14 a_vals=[56], b_vals=[14]
# CHECK: thread 15 a_vals=[60], b_vals=[15]
# CHECK: thread 16 a_vals=[1], b_vals=[16]
# CHECK: thread 17 a_vals=[5], b_vals=[17]
# CHECK: thread 18 a_vals=[9], b_vals=[18]
# CHECK: thread 19 a_vals=[13], b_vals=[19]
# CHECK: thread 20 a_vals=[17], b_vals=[20]
# CHECK: thread 21 a_vals=[21], b_vals=[21]
# CHECK: thread 22 a_vals=[25], b_vals=[22]
# CHECK: thread 23 a_vals=[29], b_vals=[23]
# CHECK: thread 24 a_vals=[33], b_vals=[24]
# CHECK: thread 25 a_vals=[37], b_vals=[25]
# CHECK: thread 26 a_vals=[41], b_vals=[26]
# CHECK: thread 27 a_vals=[45], b_vals=[27]
# CHECK: thread 28 a_vals=[49], b_vals=[28]
# CHECK: thread 29 a_vals=[53], b_vals=[29]
# CHECK: thread 30 a_vals=[57], b_vals=[30]
# CHECK: thread 31 a_vals=[61], b_vals=[31]
# CHECK: thread 32 a_vals=[2], b_vals=[32]
# CHECK: thread 33 a_vals=[6], b_vals=[33]
# CHECK: thread 34 a_vals=[10], b_vals=[34]
# CHECK: thread 35 a_vals=[14], b_vals=[35]
# CHECK: thread 36 a_vals=[18], b_vals=[36]
# CHECK: thread 37 a_vals=[22], b_vals=[37]
# CHECK: thread 38 a_vals=[26], b_vals=[38]
# CHECK: thread 39 a_vals=[30], b_vals=[39]
# CHECK: thread 40 a_vals=[34], b_vals=[40]
# CHECK: thread 41 a_vals=[38], b_vals=[41]
# CHECK: thread 42 a_vals=[42], b_vals=[42]
# CHECK: thread 43 a_vals=[46], b_vals=[43]
# CHECK: thread 44 a_vals=[50], b_vals=[44]
# CHECK: thread 45 a_vals=[54], b_vals=[45]
# CHECK: thread 46 a_vals=[58], b_vals=[46]
# CHECK: thread 47 a_vals=[62], b_vals=[47]
# CHECK: thread 48 a_vals=[3], b_vals=[48]
# CHECK: thread 49 a_vals=[7], b_vals=[49]
# CHECK: thread 50 a_vals=[11], b_vals=[50]
# CHECK: thread 51 a_vals=[15], b_vals=[51]
# CHECK: thread 52 a_vals=[19], b_vals=[52]
# CHECK: thread 53 a_vals=[23], b_vals=[53]
# CHECK: thread 54 a_vals=[27], b_vals=[54]
# CHECK: thread 55 a_vals=[31], b_vals=[55]
# CHECK: thread 56 a_vals=[35], b_vals=[56]
# CHECK: thread 57 a_vals=[39], b_vals=[57]
# CHECK: thread 58 a_vals=[43], b_vals=[58]
# CHECK: thread 59 a_vals=[47], b_vals=[59]
# CHECK: thread 60 a_vals=[51], b_vals=[60]
# CHECK: thread 61 a_vals=[55], b_vals=[61]
# CHECK: thread 62 a_vals=[59], b_vals=[62]
# CHECK: thread 63 a_vals=[63], b_vals=[63]
def test_load_f32_f32_16x16x4_ldmatrix(ctx: DeviceContext) raises:
    print("== test_load_f32_f32_16x16x4_ldmatrix")
    test_load_operands_ldmatrix[.float32, DType.float32, Index(16, 16, 4)](ctx)


def main() raises:
    with DeviceContext() as ctx:
        test_load_and_mma_f32_f16_16x16x16(ctx)
        test_load_and_mma_f32_bf16_16x16x16(ctx)
        test_load_and_mma_f32_f32_16x16x4(ctx)
        test_write_f32_f32_16x16x4(ctx)
        test_write_f32_f16_16x16x16(ctx)
        test_write_f32_bf16_16x16x16(ctx)

        # ldmatrix
        test_load_f32_f16_16x16x16_ldmatrix(ctx)
        test_load_f32_bf16_16x16x16_ldmatrix(ctx)
        test_load_f32_f32_16x16x4_ldmatrix(ctx)
