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


# CHECK-LABEL: test_load_and_mma_f32_f32_16x8x8
# CHECK-DAG: thread 0 a_vals=[0 64 4 68], b_vals=[0 32], d_vals=[1121 1149 15457 15997]
# CHECK-DAG: thread 1 a_vals=[1 65 5 69], b_vals=[8 40], d_vals=[1177 1205 16537 17077]
# CHECK-DAG: thread 2 a_vals=[2 66 6 70], b_vals=[16 48], d_vals=[1233 1261 17617 18157]
# CHECK-DAG: thread 3 a_vals=[3 67 7 71], b_vals=[24 56], d_vals=[1289 1317 18697 19237]
# CHECK-DAG: thread 4 a_vals=[8 72 12 76], b_vals=[1 33], d_vals=[2913 3005 17249 17853]
# CHECK-DAG: thread 5 a_vals=[9 73 13 77], b_vals=[9 41], d_vals=[3097 3189 18457 19061]
# CHECK-DAG: thread 6 a_vals=[10 74 14 78], b_vals=[17 49], d_vals=[3281 3373 19665 20269]
# CHECK-DAG: thread 7 a_vals=[11 75 15 79], b_vals=[25 57], d_vals=[3465 3557 20873 21477]
# CHECK-DAG: thread 8 a_vals=[16 80 20 84], b_vals=[2 34], d_vals=[4705 4861 19041 19709]
# CHECK-DAG: thread 9 a_vals=[17 81 21 85], b_vals=[10 42], d_vals=[5017 5173 20377 21045]
# CHECK-DAG: thread 10 a_vals=[18 82 22 86], b_vals=[18 50], d_vals=[5329 5485 21713 22381]
# CHECK-DAG: thread 11 a_vals=[19 83 23 87], b_vals=[26 58], d_vals=[5641 5797 23049 23717]
# CHECK-DAG: thread 12 a_vals=[24 88 28 92], b_vals=[3 35], d_vals=[6497 6717 20833 21565]
# CHECK-DAG: thread 13 a_vals=[25 89 29 93], b_vals=[11 43], d_vals=[6937 7157 22297 23029]
# CHECK-DAG: thread 14 a_vals=[26 90 30 94], b_vals=[19 51], d_vals=[7377 7597 23761 24493]
# CHECK-DAG: thread 15 a_vals=[27 91 31 95], b_vals=[27 59], d_vals=[7817 8037 25225 25957]
# CHECK-DAG: thread 16 a_vals=[32 96 36 100], b_vals=[4 36], d_vals=[8289 8573 22625 23421]
# CHECK-DAG: thread 17 a_vals=[33 97 37 101], b_vals=[12 44], d_vals=[8857 9141 24217 25013]
# CHECK-DAG: thread 18 a_vals=[34 98 38 102], b_vals=[20 52], d_vals=[9425 9709 25809 26605]
# CHECK-DAG: thread 19 a_vals=[35 99 39 103], b_vals=[28 60], d_vals=[9993 10277 27401 28197]
# CHECK-DAG: thread 20 a_vals=[40 104 44 108], b_vals=[5 37], d_vals=[10081 10429 24417 25277]
# CHECK-DAG: thread 21 a_vals=[41 105 45 109], b_vals=[13 45], d_vals=[10777 11125 26137 26997]
# CHECK-DAG: thread 22 a_vals=[42 106 46 110], b_vals=[21 53], d_vals=[11473 11821 27857 28717]
# CHECK-DAG: thread 23 a_vals=[43 107 47 111], b_vals=[29 61], d_vals=[12169 12517 29577 30437]
# CHECK-DAG: thread 24 a_vals=[48 112 52 116], b_vals=[6 38], d_vals=[11873 12285 26209 27133]
# CHECK-DAG: thread 25 a_vals=[49 113 53 117], b_vals=[14 46], d_vals=[12697 13109 28057 28981]
# CHECK-DAG: thread 26 a_vals=[50 114 54 118], b_vals=[22 54], d_vals=[13521 13933 29905 30829]
# CHECK-DAG: thread 27 a_vals=[51 115 55 119], b_vals=[30 62], d_vals=[14345 14757 31753 32677]
# CHECK-DAG: thread 28 a_vals=[56 120 60 124], b_vals=[7 39], d_vals=[13665 14141 28001 28989]
# CHECK-DAG: thread 29 a_vals=[57 121 61 125], b_vals=[15 47], d_vals=[14617 15093 29977 30965]
# CHECK-DAG: thread 30 a_vals=[58 122 62 126], b_vals=[23 55], d_vals=[15569 16045 31953 32941]
# CHECK-DAG: thread 31 a_vals=[59 123 63 127], b_vals=[31 63], d_vals=[16521 16997 33929 34917]
def test_load_and_mma_f32_f32_16x8x8(ctx: DeviceContext) raises:
    print("== test_load_and_mma_f32_f32_16x8x8")
    test_load_and_mma_and_multiply_operands[
        DType.float32, DType.float32, Index(16, 8, 8)
    ](ctx)


# CHECK-LABEL: test_load_and_mma_f32_f32_16x8x8_b_transpose
# CHECK-DAG: thread 0 a_vals=[0 64 4 68], b_vals=[0 4], d_vals=[141 365 1933 6253]
# CHECK-DAG: thread 1 a_vals=[1 65 5 69], b_vals=[1 5], d_vals=[589 813 10573 14893]
# CHECK-DAG: thread 2 a_vals=[2 66 6 70], b_vals=[2 6], d_vals=[1037 1261 19213 23533]
# CHECK-DAG: thread 3 a_vals=[3 67 7 71], b_vals=[3 7], d_vals=[1485 1709 27853 32173]
# CHECK-DAG: thread 4 a_vals=[8 72 12 76], b_vals=[8 12], d_vals=[365 1101 2157 6989]
# CHECK-DAG: thread 5 a_vals=[9 73 13 77], b_vals=[9 13], d_vals=[1837 2573 11821 16653]
# CHECK-DAG: thread 6 a_vals=[10 74 14 78], b_vals=[10 14], d_vals=[3309 4045 21485 26317]
# CHECK-DAG: thread 7 a_vals=[11 75 15 79], b_vals=[11 15], d_vals=[4781 5517 31149 35981]
# CHECK-DAG: thread 8 a_vals=[16 80 20 84], b_vals=[16 20], d_vals=[589 1837 2381 7725]
# CHECK-DAG: thread 9 a_vals=[17 81 21 85], b_vals=[17 21], d_vals=[3085 4333 13069 18413]
# CHECK-DAG: thread 10 a_vals=[18 82 22 86], b_vals=[18 22], d_vals=[5581 6829 23757 29101]
# CHECK-DAG: thread 11 a_vals=[19 83 23 87], b_vals=[19 23], d_vals=[8077 9325 34445 39789]
# CHECK-DAG: thread 12 a_vals=[24 88 28 92], b_vals=[24 28], d_vals=[813 2573 2605 8461]
# CHECK-DAG: thread 13 a_vals=[25 89 29 93], b_vals=[25 29], d_vals=[4333 6093 14317 20173]
# CHECK-DAG: thread 14 a_vals=[26 90 30 94], b_vals=[26 30], d_vals=[7853 9613 26029 31885]
# CHECK-DAG: thread 15 a_vals=[27 91 31 95], b_vals=[27 31], d_vals=[11373 13133 37741 43597]
# CHECK-DAG: thread 16 a_vals=[32 96 36 100], b_vals=[32 36], d_vals=[1037 3309 2829 9197]
# CHECK-DAG: thread 17 a_vals=[33 97 37 101], b_vals=[33 37], d_vals=[5581 7853 15565 21933]
# CHECK-DAG: thread 18 a_vals=[34 98 38 102], b_vals=[34 38], d_vals=[10125 12397 28301 34669]
# CHECK-DAG: thread 19 a_vals=[35 99 39 103], b_vals=[35 39], d_vals=[14669 16941 41037 47405]
# CHECK-DAG: thread 20 a_vals=[40 104 44 108], b_vals=[40 44], d_vals=[1261 4045 3053 9933]
# CHECK-DAG: thread 21 a_vals=[41 105 45 109], b_vals=[41 45], d_vals=[6829 9613 16813 23693]
# CHECK-DAG: thread 22 a_vals=[42 106 46 110], b_vals=[42 46], d_vals=[12397 15181 30573 37453]
# CHECK-DAG: thread 23 a_vals=[43 107 47 111], b_vals=[43 47], d_vals=[17965 20749 44333 51213]
# CHECK-DAG: thread 24 a_vals=[48 112 52 116], b_vals=[48 52], d_vals=[1485 4781 3277 10669]
# CHECK-DAG: thread 25 a_vals=[49 113 53 117], b_vals=[49 53], d_vals=[8077 11373 18061 25453]
# CHECK-DAG: thread 26 a_vals=[50 114 54 118], b_vals=[50 54], d_vals=[14669 17965 32845 40237]
# CHECK-DAG: thread 27 a_vals=[51 115 55 119], b_vals=[51 55], d_vals=[21261 24557 47629 55021]
# CHECK-DAG: thread 28 a_vals=[56 120 60 124], b_vals=[56 60], d_vals=[1709 5517 3501 11405]
# CHECK-DAG: thread 29 a_vals=[57 121 61 125], b_vals=[57 61], d_vals=[9325 13133 19309 27213]
# CHECK-DAG: thread 30 a_vals=[58 122 62 126], b_vals=[58 62], d_vals=[16941 20749 35117 43021]
# CHECK-DAG: thread 31 a_vals=[59 123 63 127], b_vals=[59 63], d_vals=[24557 28365 50925 58829]
def test_load_and_mma_f32_f32_16x8x8_b_transpose(ctx: DeviceContext) raises:
    print("== test_load_and_mma_f32_f32_16x8x8_b_transpose")
    test_load_and_mma_and_multiply_operands[
        DType.float32, DType.float32, Index(16, 8, 8), transpose_b=True
    ](ctx)


# CHECK-LABEL: test_load_and_mma_f32_f32_16x8x4
# CHECK-DAG: thread 0 a_vals=[0 32], b_vals=[0], d_vals=[113 119 1649 1783]
# CHECK-DAG: thread 1 a_vals=[1 33], b_vals=[8], d_vals=[125 131 1917 2051]
# CHECK-DAG: thread 2 a_vals=[2 34], b_vals=[16], d_vals=[137 143 2185 2319]
# CHECK-DAG: thread 3 a_vals=[3 35], b_vals=[24], d_vals=[149 155 2453 2587]
# CHECK-DAG: thread 4 a_vals=[4 36], b_vals=[1], d_vals=[305 327 1841 1991]
# CHECK-DAG: thread 5 a_vals=[5 37], b_vals=[9], d_vals=[349 371 2141 2291]
# CHECK-DAG: thread 6 a_vals=[6 38], b_vals=[17], d_vals=[393 415 2441 2591]
# CHECK-DAG: thread 7 a_vals=[7 39], b_vals=[25], d_vals=[437 459 2741 2891]
# CHECK-DAG: thread 8 a_vals=[8 40], b_vals=[2], d_vals=[497 535 2033 2199]
# CHECK-DAG: thread 9 a_vals=[9 41], b_vals=[10], d_vals=[573 611 2365 2531]
# CHECK-DAG: thread 10 a_vals=[10 42], b_vals=[18], d_vals=[649 687 2697 2863]
# CHECK-DAG: thread 11 a_vals=[11 43], b_vals=[26], d_vals=[725 763 3029 3195]
# CHECK-DAG: thread 12 a_vals=[12 44], b_vals=[3], d_vals=[689 743 2225 2407]
# CHECK-DAG: thread 13 a_vals=[13 45], b_vals=[11], d_vals=[797 851 2589 2771]
# CHECK-DAG: thread 14 a_vals=[14 46], b_vals=[19], d_vals=[905 959 2953 3135]
# CHECK-DAG: thread 15 a_vals=[15 47], b_vals=[27], d_vals=[1013 1067 3317 3499]
# CHECK-DAG: thread 16 a_vals=[16 48], b_vals=[4], d_vals=[881 951 2417 2615]
# CHECK-DAG: thread 17 a_vals=[17 49], b_vals=[12], d_vals=[1021 1091 2813 3011]
# CHECK-DAG: thread 18 a_vals=[18 50], b_vals=[20], d_vals=[1161 1231 3209 3407]
# CHECK-DAG: thread 19 a_vals=[19 51], b_vals=[28], d_vals=[1301 1371 3605 3803]
# CHECK-DAG: thread 20 a_vals=[20 52], b_vals=[5], d_vals=[1073 1159 2609 2823]
# CHECK-DAG: thread 21 a_vals=[21 53], b_vals=[13], d_vals=[1245 1331 3037 3251]
# CHECK-DAG: thread 22 a_vals=[22 54], b_vals=[21], d_vals=[1417 1503 3465 3679]
# CHECK-DAG: thread 23 a_vals=[23 55], b_vals=[29], d_vals=[1589 1675 3893 4107]
# CHECK-DAG: thread 24 a_vals=[24 56], b_vals=[6], d_vals=[1265 1367 2801 3031]
# CHECK-DAG: thread 25 a_vals=[25 57], b_vals=[14], d_vals=[1469 1571 3261 3491]
# CHECK-DAG: thread 26 a_vals=[26 58], b_vals=[22], d_vals=[1673 1775 3721 3951]
# CHECK-DAG: thread 27 a_vals=[27 59], b_vals=[30], d_vals=[1877 1979 4181 4411]
# CHECK-DAG: thread 28 a_vals=[28 60], b_vals=[7], d_vals=[1457 1575 2993 3239]
# CHECK-DAG: thread 29 a_vals=[29 61], b_vals=[15], d_vals=[1693 1811 3485 3731]
# CHECK-DAG: thread 30 a_vals=[30 62], b_vals=[23], d_vals=[1929 2047 3977 4223]
# CHECK-DAG: thread 31 a_vals=[31 63], b_vals=[31], d_vals=[2165 2283 4469 4715]
def test_load_and_mma_f32_f32_16x8x4(ctx: DeviceContext) raises:
    print("== test_load_and_mma_f32_f32_16x8x4")
    test_load_and_mma_and_multiply_operands[
        DType.float32, DType.float32, Index(16, 8, 4)
    ](ctx)


# CHECK-LABEL: test_load_and_mma_f32_bf16_16x8x16
# CHECK-DAG: thread 0 a_vals=[0 1 128 129 8 9 136 137], b_vals=[0 8 64 72], d_vals=[9921 10041 132801 134969]
# CHECK-DAG: thread 1 a_vals=[2 3 130 131 10 11 138 139], b_vals=[16 24 80 88], d_vals=[10161 10281 137137 139305]
# CHECK-DAG: thread 2 a_vals=[4 5 132 133 12 13 140 141], b_vals=[32 40 96 104], d_vals=[10401 10521 141473 143641]
# CHECK-DAG: thread 3 a_vals=[6 7 134 135 14 15 142 143], b_vals=[48 56 112 120], d_vals=[10641 10761 145809 147977]
# CHECK-DAG: thread 4 a_vals=[16 17 144 145 24 25 152 153], b_vals=[1 9 65 73], d_vals=[25281 25657 148161 150585]
# CHECK-DAG: thread 5 a_vals=[18 19 146 147 26 27 154 155], b_vals=[17 25 81 89], d_vals=[26033 26409 153009 155433]
# CHECK-DAG: thread 6 a_vals=[20 21 148 149 28 29 156 157], b_vals=[33 41 97 105], d_vals=[26785 27161 157857 160281]
# CHECK-DAG: thread 7 a_vals=[22 23 150 151 30 31 158 159], b_vals=[49 57 113 121], d_vals=[27537 27913 162705 165129]
# CHECK-DAG: thread 8 a_vals=[32 33 160 161 40 41 168 169], b_vals=[2 10 66 74], d_vals=[40641 41273 163521 166201]
# CHECK-DAG: thread 9 a_vals=[34 35 162 163 42 43 170 171], b_vals=[18 26 82 90], d_vals=[41905 42537 168881 171561]
# CHECK-DAG: thread 10 a_vals=[36 37 164 165 44 45 172 173], b_vals=[34 42 98 106], d_vals=[43169 43801 174241 176921]
# CHECK-DAG: thread 11 a_vals=[38 39 166 167 46 47 174 175], b_vals=[50 58 114 122], d_vals=[44433 45065 179601 182281]
# CHECK-DAG: thread 12 a_vals=[48 49 176 177 56 57 184 185], b_vals=[3 11 67 75], d_vals=[56001 56889 178881 181817]
# CHECK-DAG: thread 13 a_vals=[50 51 178 179 58 59 186 187], b_vals=[19 27 83 91], d_vals=[57777 58665 184753 187689]
# CHECK-DAG: thread 14 a_vals=[52 53 180 181 60 61 188 189], b_vals=[35 43 99 107], d_vals=[59553 60441 190625 193561]
# CHECK-DAG: thread 15 a_vals=[54 55 182 183 62 63 190 191], b_vals=[51 59 115 123], d_vals=[61329 62217 196497 199433]
# CHECK-DAG: thread 16 a_vals=[64 65 192 193 72 73 200 201], b_vals=[4 12 68 76], d_vals=[71361 72505 194241 197433]
# CHECK-DAG: thread 17 a_vals=[66 67 194 195 74 75 202 203], b_vals=[20 28 84 92], d_vals=[73649 74793 200625 203817]
# CHECK-DAG: thread 18 a_vals=[68 69 196 197 76 77 204 205], b_vals=[36 44 100 108], d_vals=[75937 77081 207009 210201]
# CHECK-DAG: thread 19 a_vals=[70 71 198 199 78 79 206 207], b_vals=[52 60 116 124], d_vals=[78225 79369 213393 216585]
# CHECK-DAG: thread 20 a_vals=[80 81 208 209 88 89 216 217], b_vals=[5 13 69 77], d_vals=[86721 88121 209601 213049]
# CHECK-DAG: thread 21 a_vals=[82 83 210 211 90 91 218 219], b_vals=[21 29 85 93], d_vals=[89521 90921 216497 219945]
# CHECK-DAG: thread 22 a_vals=[84 85 212 213 92 93 220 221], b_vals=[37 45 101 109], d_vals=[92321 93721 223393 226841]
# CHECK-DAG: thread 23 a_vals=[86 87 214 215 94 95 222 223], b_vals=[53 61 117 125], d_vals=[95121 96521 230289 233737]
# CHECK-DAG: thread 24 a_vals=[96 97 224 225 104 105 232 233], b_vals=[6 14 70 78], d_vals=[102081 103737 224961 228665]
# CHECK-DAG: thread 25 a_vals=[98 99 226 227 106 107 234 235], b_vals=[22 30 86 94], d_vals=[105393 107049 232369 236073]
# CHECK-DAG: thread 26 a_vals=[100 101 228 229 108 109 236 237], b_vals=[38 46 102 110], d_vals=[108705 110361 239777 243481]
# CHECK-DAG: thread 27 a_vals=[102 103 230 231 110 111 238 239], b_vals=[54 62 118 126], d_vals=[112017 113673 247185 250889]
# CHECK-DAG: thread 28 a_vals=[112 113 240 241 120 121 248 249], b_vals=[7 15 71 79], d_vals=[117441 119353 240321 244281]
# CHECK-DAG: thread 29 a_vals=[114 115 242 243 122 123 250 251], b_vals=[23 31 87 95], d_vals=[121265 123177 248241 252201]
# CHECK-DAG: thread 30 a_vals=[116 117 244 245 124 125 252 253], b_vals=[39 47 103 111], d_vals=[125089 127001 256161 260121]
# CHECK-DAG: thread 31 a_vals=[118 119 246 247 126 127 254 255], b_vals=[55 63 119 127], d_vals=[128913 130825 264081 268041]


def test_load_and_mma_f32_bf16_16x8x16(ctx: DeviceContext) raises:
    print("== test_load_and_mma_f32_bf16_16x8x16")
    test_load_and_mma_and_multiply_operands[
        DType.float32, DType.bfloat16, Index(16, 8, 16)
    ](ctx)


# CHECK-LABEL: == test_load_and_mma_f32_bf16_16x8x16_b_transpose
# CHECK: thread 0 a_vals=[0 1 128 129 8 9 136 137], b_vals=[0 1 8 9], d_vals=[1241 3161 16601 51289]
# CHECK: thread 1 a_vals=[2 3 130 131 10 11 138 139], b_vals=[2 3 10 11], d_vals=[5081 7001 85977 120665]
# CHECK: thread 2 a_vals=[4 5 132 133 12 13 140 141], b_vals=[4 5 12 13], d_vals=[8921 10841 155353 190041]
# CHECK: thread 3 a_vals=[6 7 134 135 14 15 142 143], b_vals=[6 7 14 15], d_vals=[12761 14681 224729 259417]
# CHECK: thread 4 a_vals=[16 17 144 145 24 25 152 153], b_vals=[16 17 24 25], d_vals=[3161 9177 18521 57305]
# CHECK: thread 5 a_vals=[18 19 146 147 26 27 154 155], b_vals=[18 19 26 27], d_vals=[15193 21209 96089 134873]
# CHECK: thread 6 a_vals=[20 21 148 149 28 29 156 157], b_vals=[20 21 28 29], d_vals=[27225 33241 173657 212441]
# CHECK: thread 7 a_vals=[22 23 150 151 30 31 158 159], b_vals=[22 23 30 31], d_vals=[39257 45273 251225 290009]
# CHECK: thread 8 a_vals=[32 33 160 161 40 41 168 169], b_vals=[32 33 40 41], d_vals=[5081 15193 20441 63321]
# CHECK: thread 9 a_vals=[34 35 162 163 42 43 170 171], b_vals=[34 35 42 43], d_vals=[25305 35417 106201 149081]
# CHECK: thread 10 a_vals=[36 37 164 165 44 45 172 173], b_vals=[36 37 44 45], d_vals=[45529 55641 191961 234841]
# CHECK: thread 11 a_vals=[38 39 166 167 46 47 174 175], b_vals=[38 39 46 47], d_vals=[65753 75865 277721 320601]
# CHECK: thread 12 a_vals=[48 49 176 177 56 57 184 185], b_vals=[48 49 56 57], d_vals=[7001 21209 22361 69337]
# CHECK: thread 13 a_vals=[50 51 178 179 58 59 186 187], b_vals=[50 51 58 59], d_vals=[35417 49625 116313 163289]
# CHECK: thread 14 a_vals=[52 53 180 181 60 61 188 189], b_vals=[52 53 60 61], d_vals=[63833 78041 210265 257241]
# CHECK: thread 15 a_vals=[54 55 182 183 62 63 190 191], b_vals=[54 55 62 63], d_vals=[92249 106457 304217 351193]
# CHECK: thread 16 a_vals=[64 65 192 193 72 73 200 201], b_vals=[64 65 72 73], d_vals=[8921 27225 24281 75353]
# CHECK: thread 17 a_vals=[66 67 194 195 74 75 202 203], b_vals=[66 67 74 75], d_vals=[45529 63833 126425 177497]
# CHECK: thread 18 a_vals=[68 69 196 197 76 77 204 205], b_vals=[68 69 76 77], d_vals=[82137 100441 228569 279641]
# CHECK: thread 19 a_vals=[70 71 198 199 78 79 206 207], b_vals=[70 71 78 79], d_vals=[118745 137049 330713 381785]
# CHECK: thread 20 a_vals=[80 81 208 209 88 89 216 217], b_vals=[80 81 88 89], d_vals=[10841 33241 26201 81369]
# CHECK: thread 21 a_vals=[82 83 210 211 90 91 218 219], b_vals=[82 83 90 91], d_vals=[55641 78041 136537 191705]
# CHECK: thread 22 a_vals=[84 85 212 213 92 93 220 221], b_vals=[84 85 92 93], d_vals=[100441 122841 246873 302041]
# CHECK: thread 23 a_vals=[86 87 214 215 94 95 222 223], b_vals=[86 87 94 95], d_vals=[145241 167641 357209 412377]
# CHECK: thread 24 a_vals=[96 97 224 225 104 105 232 233], b_vals=[96 97 104 105], d_vals=[12761 39257 28121 87385]
# CHECK: thread 25 a_vals=[98 99 226 227 106 107 234 235], b_vals=[98 99 106 107], d_vals=[65753 92249 146649 205913]
# CHECK: thread 26 a_vals=[100 101 228 229 108 109 236 237], b_vals=[100 101 108 109], d_vals=[118745 145241 265177 324441]
# CHECK: thread 27 a_vals=[102 103 230 231 110 111 238 239], b_vals=[102 103 110 111], d_vals=[171737 198233 383705 442969]
# CHECK: thread 28 a_vals=[112 113 240 241 120 121 248 249], b_vals=[112 113 120 121], d_vals=[14681 45273 30041 93401]
# CHECK: thread 29 a_vals=[114 115 242 243 122 123 250 251], b_vals=[114 115 122 123], d_vals=[75865 106457 156761 220121]
# CHECK: thread 30 a_vals=[116 117 244 245 124 125 252 253], b_vals=[116 117 124 125], d_vals=[137049 167641 283481 346841]
# CHECK: thread 31 a_vals=[118 119 246 247 126 127 254 255], b_vals=[118 119 126 127], d_vals=[198233 228825 410201 473561]
def test_load_and_mma_f32_bf16_16x8x16_b_transpose(ctx: DeviceContext) raises:
    print("== test_load_and_mma_f32_bf16_16x8x16_b_transpose")
    test_load_and_mma_and_multiply_operands[
        DType.float32, DType.bfloat16, Index(16, 8, 16), transpose_b=True
    ](ctx)


# CHECK-LABEL: test_write_f32_f32_16x8x8
# CHECK: 0.0 0.0 1.0 1.0 2.0 2.0 3.0 3.0
# CHECK: 4.0 4.0 5.0 5.0 6.0 6.0 7.0 7.0
# CHECK: 8.0 8.0 9.0 9.0 10.0 10.0 11.0 11.0
# CHECK: 12.0 12.0 13.0 13.0 14.0 14.0 15.0 15.0
# CHECK: 16.0 16.0 17.0 17.0 18.0 18.0 19.0 19.0
# CHECK: 20.0 20.0 21.0 21.0 22.0 22.0 23.0 23.0
# CHECK: 24.0 24.0 25.0 25.0 26.0 26.0 27.0 27.0
# CHECK: 28.0 28.0 29.0 29.0 30.0 30.0 31.0 31.0
# CHECK: 0.0 0.0 1.0 1.0 2.0 2.0 3.0 3.0
# CHECK: 4.0 4.0 5.0 5.0 6.0 6.0 7.0 7.0
# CHECK: 8.0 8.0 9.0 9.0 10.0 10.0 11.0 11.0
# CHECK: 12.0 12.0 13.0 13.0 14.0 14.0 15.0 15.0
# CHECK: 16.0 16.0 17.0 17.0 18.0 18.0 19.0 19.0
# CHECK: 20.0 20.0 21.0 21.0 22.0 22.0 23.0 23.0
# CHECK: 24.0 24.0 25.0 25.0 26.0 26.0 27.0 27.0
# CHECK: 28.0 28.0 29.0 29.0 30.0 30.0 31.0 31.0
def test_write_f32_f32_16x8x8(ctx: DeviceContext) raises:
    print("== test_write_f32_f32_16x8x8")
    test_write_res_operand[.float32, DType.float32, Index(16, 8, 8)](ctx)


# CHECK-LABEL: test_write_f32_f32_16x8x4
# CHECK: 0.0 0.0 1.0 1.0 2.0 2.0 3.0 3.0
# CHECK: 4.0 4.0 5.0 5.0 6.0 6.0 7.0 7.0
# CHECK: 8.0 8.0 9.0 9.0 10.0 10.0 11.0 11.0
# CHECK: 12.0 12.0 13.0 13.0 14.0 14.0 15.0 15.0
# CHECK: 16.0 16.0 17.0 17.0 18.0 18.0 19.0 19.0
# CHECK: 20.0 20.0 21.0 21.0 22.0 22.0 23.0 23.0
# CHECK: 24.0 24.0 25.0 25.0 26.0 26.0 27.0 27.0
# CHECK: 28.0 28.0 29.0 29.0 30.0 30.0 31.0 31.0
# CHECK: 0.0 0.0 1.0 1.0 2.0 2.0 3.0 3.0
# CHECK: 4.0 4.0 5.0 5.0 6.0 6.0 7.0 7.0
# CHECK: 8.0 8.0 9.0 9.0 10.0 10.0 11.0 11.0
# CHECK: 12.0 12.0 13.0 13.0 14.0 14.0 15.0 15.0
# CHECK: 16.0 16.0 17.0 17.0 18.0 18.0 19.0 19.0
# CHECK: 20.0 20.0 21.0 21.0 22.0 22.0 23.0 23.0
# CHECK: 24.0 24.0 25.0 25.0 26.0 26.0 27.0 27.0
# CHECK: 28.0 28.0 29.0 29.0 30.0 30.0 31.0 31.0
def test_write_f32_f32_16x8x4(ctx: DeviceContext) raises:
    print("== test_write_f32_f32_16x8x4")
    test_write_res_operand[.float32, DType.float32, Index(16, 8, 4)](ctx)


# CHECK-LABEL: test_load_f32_bf16_16x8x16_ldmatrix
# CHECK-DAG thread 0 a_vals=[0 1 128 129 8 9 136 137], b_vals=[0 8 64 72]
# CHECK-DAG thread 1 a_vals=[2 3 130 131 10 11 138 139], b_vals=[16 24 80 88]
# CHECK-DAG thread 2 a_vals=[4 5 132 133 12 13 140 141], b_vals=[32 40 96 104]
# CHECK-DAG thread 3 a_vals=[6 7 134 135 14 15 142 143], b_vals=[48 56 112 120]
# CHECK-DAG thread 4 a_vals=[16 17 144 145 24 25 152 153], b_vals=[1 9 65 73]
# CHECK-DAG thread 5 a_vals=[18 19 146 147 26 27 154 155], b_vals=[17 25 81 89]
# CHECK-DAG thread 6 a_vals=[20 21 148 149 28 29 156 157], b_vals=[33 41 97 105]
# CHECK-DAG thread 7 a_vals=[22 23 150 151 30 31 158 159], b_vals=[49 57 113 121]
# CHECK-DAG thread 8 a_vals=[32 33 160 161 40 41 168 169], b_vals=[2 10 66 74]
# CHECK-DAG thread 9 a_vals=[34 35 162 163 42 43 170 171], b_vals=[18 26 82 90]
# CHECK-DAG thread 10 a_vals=[36 37 164 165 44 45 172 173], b_vals=[34 42 98 106]
# CHECK-DAG thread 11 a_vals=[38 39 166 167 46 47 174 175], b_vals=[50 58 114 122]
# CHECK-DAG thread 12 a_vals=[48 49 176 177 56 57 184 185], b_vals=[3 11 67 75]
# CHECK-DAG thread 13 a_vals=[50 51 178 179 58 59 186 187], b_vals=[19 27 83 91]
# CHECK-DAG thread 14 a_vals=[52 53 180 181 60 61 188 189], b_vals=[35 43 99 107]
# CHECK-DAG thread 15 a_vals=[54 55 182 183 62 63 190 191], b_vals=[51 59 115 123]
# CHECK-DAG thread 16 a_vals=[72 73 200 201 64 65 192 193], b_vals=[4 12 68 76]
# CHECK-DAG thread 17 a_vals=[74 75 202 203 66 67 194 195], b_vals=[20 28 84 92]
# CHECK-DAG thread 18 a_vals=[76 77 204 205 68 69 196 197], b_vals=[36 44 100 108]
# CHECK-DAG thread 19 a_vals=[78 79 206 207 70 71 198 199], b_vals=[52 60 116 124]
# CHECK-DAG thread 20 a_vals=[88 89 216 217 80 81 208 209], b_vals=[5 13 69 77]
# CHECK-DAG thread 21 a_vals=[90 91 218 219 82 83 210 211], b_vals=[21 29 85 93]
# CHECK-DAG thread 22 a_vals=[92 93 220 221 84 85 212 213], b_vals=[37 45 101 109]
# CHECK-DAG thread 23 a_vals=[94 95 222 223 86 87 214 215], b_vals=[53 61 117 125]
# CHECK-DAG thread 24 a_vals=[104 105 232 233 96 97 224 225], b_vals=[6 14 70 78]
# CHECK-DAG thread 25 a_vals=[106 107 234 235 98 99 226 227], b_vals=[22 30 86 94]
# CHECK-DAG thread 26 a_vals=[108 109 236 237 100 101 228 229], b_vals=[38 46 102 110]
# CHECK-DAG thread 27 a_vals=[110 111 238 239 102 103 230 231], b_vals=[54 62 118 126]
# CHECK-DAG thread 28 a_vals=[120 121 248 249 112 113 240 241], b_vals=[7 15 71 79]
# CHECK-DAG thread 29 a_vals=[122 123 250 251 114 115 242 243], b_vals=[23 31 87 95]
# CHECK-DAG thread 30 a_vals=[124 125 252 253 116 117 244 245], b_vals=[39 47 103 111]
# CHECK-DAG thread 31 a_vals=[126 127 254 255 118 119 246 247], b_vals=[55 63 119 127]


def test_load_f32_bf16_16x8x16_ldmatrix(ctx: DeviceContext) raises:
    print("== test_load_f32_bf16_16x8x16_ldmatrix")
    test_load_operands_ldmatrix[
        DType.float32, DType.bfloat16, Index(16, 8, 16)
    ](ctx)


# CHECK-LABEL: test_load_f32_f32_16x8x8_ldmatrix
# CHECK-DAG thread 0 a_vals=[0 64 4 68], b_vals=[0 32]
# CHECK-DAG thread 1 a_vals=[1 65 5 69], b_vals=[8 40]
# CHECK-DAG thread 2 a_vals=[2 66 6 70], b_vals=[16 48]
# CHECK-DAG thread 3 a_vals=[3 67 7 71], b_vals=[24 56]
# CHECK-DAG thread 4 a_vals=[8 72 12 76], b_vals=[1 33]
# CHECK-DAG thread 5 a_vals=[9 73 13 77], b_vals=[9 41]
# CHECK-DAG thread 6 a_vals=[10 74 14 78], b_vals=[17 49]
# CHECK-DAG thread 7 a_vals=[11 75 15 79], b_vals=[25 57]
# CHECK-DAG thread 8 a_vals=[16 80 20 84], b_vals=[2 34]
# CHECK-DAG thread 9 a_vals=[17 81 21 85], b_vals=[10 42]
# CHECK-DAG thread 10 a_vals=[18 82 22 86], b_vals=[18 50]
# CHECK-DAG thread 11 a_vals=[19 83 23 87], b_vals=[26 58]
# CHECK-DAG thread 12 a_vals=[24 88 28 92], b_vals=[3 35]
# CHECK-DAG thread 13 a_vals=[25 89 29 93], b_vals=[11 43]
# CHECK-DAG thread 14 a_vals=[26 90 30 94], b_vals=[19 51]
# CHECK-DAG thread 15 a_vals=[27 91 31 95], b_vals=[27 59]
# CHECK-DAG thread 16 a_vals=[36 100 32 96], b_vals=[4 36]
# CHECK-DAG thread 17 a_vals=[37 101 33 97], b_vals=[12 44]
# CHECK-DAG thread 18 a_vals=[38 102 34 98], b_vals=[20 52]
# CHECK-DAG thread 19 a_vals=[39 103 35 99], b_vals=[28 60]
# CHECK-DAG thread 20 a_vals=[44 108 40 104], b_vals=[5 37]
# CHECK-DAG thread 21 a_vals=[45 109 41 105], b_vals=[13 45]
# CHECK-DAG thread 22 a_vals=[46 110 42 106], b_vals=[21 53]
# CHECK-DAG thread 23 a_vals=[47 111 43 107], b_vals=[29 61]
# CHECK-DAG thread 24 a_vals=[52 116 48 112], b_vals=[6 38]
# CHECK-DAG thread 25 a_vals=[53 117 49 113], b_vals=[14 46]
# CHECK-DAG thread 26 a_vals=[54 118 50 114], b_vals=[22 54]
# CHECK-DAG thread 27 a_vals=[55 119 51 115], b_vals=[30 62]
# CHECK-DAG thread 28 a_vals=[60 124 56 120], b_vals=[7 39]
# CHECK-DAG thread 29 a_vals=[61 125 57 121], b_vals=[15 47]
# CHECK-DAG thread 30 a_vals=[62 126 58 122], b_vals=[23 55]
# CHECK-DAG thread 31 a_vals=[63 127 59 123], b_vals=[31 63]
def test_load_f32_f32_16x8x8_ldmatrix(ctx: DeviceContext) raises:
    print("== test_load_f32_f32_16x8x8_ldmatrix")
    test_load_operands_ldmatrix[.float32, DType.float32, Index(16, 8, 8)](ctx)


def main() raises:
    with DeviceContext() as ctx:
        test_load_and_mma_f32_f32_16x8x8(ctx)
        test_load_and_mma_f32_f32_16x8x8_b_transpose(ctx)
        test_load_and_mma_f32_f32_16x8x4(ctx)
        test_load_and_mma_f32_bf16_16x8x16(ctx)
        test_load_and_mma_f32_bf16_16x8x16_b_transpose(ctx)
        test_write_f32_f32_16x8x8(ctx)
        test_write_f32_f32_16x8x4(ctx)

        # ldmatrix
        test_load_f32_bf16_16x8x16_ldmatrix(ctx)
        test_load_f32_f32_16x8x8_ldmatrix(ctx)
