# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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

# RUN: %mojo %s -t

from benchmark import Bench, Bencher, BenchId, keep, BenchConfig, Unit, run
from utils.stringref import _memmem, _memchr, _align_down

from bit import countr_zero
from builtin.dtype import _uint_type_of_width

# ===----------------------------------------------------------------------===#
# Benchmark Data
# ===----------------------------------------------------------------------===#
var haystack = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer sed dictum est, et finibus ipsum. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Nam tincidunt vel lacus vitae pulvinar. Donec ac ligula elementum, mollis purus a, lacinia quam. Maecenas vulputate mauris quis sem euismod sollicitudin. Proin accumsan nulla vel nisl congue varius. Morbi a erat dui. Aliquam maximus interdum orci, vitae pretium lorem bibendum non. Vestibulum eu lacus ullamcorper, egestas dui vel, pharetra ipsum. Pellentesque sagittis, urna a tincidunt sodales, leo sem placerat eros, vitae molestie felis diam at dolor.

Donec viverra sem sit amet facilisis laoreet. Morbi semper convallis nisi, vitae congue velit tincidunt vel. Fusce ultrices, libero vel venenatis placerat, justo tellus porttitor massa, at volutpat tortor nunc id dui. Morbi eu ex quis odio porttitor ultricies vel eget massa. Aenean quis luctus nulla. Fusce sit amet leo at quam hendrerit mattis. Morbi sed quam nisl. Quisque purus enim, iaculis sed laoreet vel, pellentesque ut orci. Vivamus risus orci, varius eu pharetra quis, tincidunt non enim. Suspendisse bibendum lacus ex, quis blandit lectus malesuada a. Maecenas iaculis porta lacus, sit amet tristique ante scelerisque non. Proin auctor elit in lacus dictum egestas. Pellentesque tincidunt justo sed vehicula blandit. Pellentesque vehicula facilisis tellus in viverra.

Curabitur vel fermentum risus. Etiam ornare dolor in eros faucibus, sit amet sagittis orci blandit. Curabitur pulvinar pretium fermentum. Duis sit amet placerat ipsum. Sed dui lorem, gravida quis lacinia vel, maximus ut augue. Aenean a mauris ornare, fermentum orci vel, euismod dolor. Nullam et libero eget mi pellentesque congue. Donec venenatis sapien sit amet sem fringilla sagittis. Pellentesque in placerat mi. Curabitur congue fermentum rhoncus. Duis blandit mauris nec diam bibendum faucibus. Praesent consequat, purus sed viverra interdum, ante enim scelerisque augue, nec pharetra tellus sem nec eros. Etiam quis est tellus.

Donec lorem eros, hendrerit vel aliquet a, venenatis et lacus. Nulla quis nibh egestas dolor convallis cursus sit amet eu quam. Pellentesque sit amet pharetra nibh. Vestibulum aliquam tempor ex, nec tincidunt felis blandit tincidunt. Sed eu purus ac neque sagittis iaculis. In et sagittis erat, id gravida diam. Etiam pharetra enim tortor, id mollis odio faucibus quis. Etiam bibendum est pharetra neque convallis sollicitudin. Praesent consectetur lobortis nibh, vel interdum augue mattis eu. In tortor augue, venenatis vel semper vel, faucibus a est. Mauris efficitur aliquet sodales. Donec bibendum tempor elit, non pellentesque ligula pharetra vel. Praesent est sapien, sagittis ac lectus at, aliquam dictum neque. Ut efficitur commodo sapien et luctus. Nulla tincidunt justo nec vestibulum finibus. Maecenas eget aliquam ante, eu porta nisl.

Aliquam ac massa mi. Mauris gravida, nisl ac volutpat egestas, leo massa tincidunt libero, eu aliquet ante mi sed justo. Quisque mattis convallis mauris, a laoreet magna aliquet at. Maecenas vel libero vehicula, interdum mauris consequat, imperdiet enim. Cras et nisi nec erat varius condimentum. Morbi mollis metus eu condimentum aliquam. Proin commodo elit a diam interdum, sit amet auctor purus malesuada. Pellentesque laoreet ante et ex sodales, eu euismod enim venenatis. Vivamus lobortis eu velit vel tincidunt. Quisque elementum dapibus odio, ultrices vulputate massa finibus ac. Morbi at condimentum orci, non efficitur tellus. Aenean id quam dignissim, dapibus justo finibus, finibus lorem. Morbi eu tortor eget elit faucibus fermentum eget a eros. Interdum et malesuada fames ac ante ipsum primis in faucibus. Morbi maximus venenatis massa, ut auctor ex ullamcorper nec.

Suspendisse malesuada nisi augue, eget tincidunt nibh sagittis nec. Aliquam semper, sem id rhoncus varius, leo eros sodales dolor, sollicitudin vestibulum nisl sem in orci. Suspendisse quis neque rhoncus, placerat nisl id, vulputate nunc. Interdum et malesuada fames ac ante ipsum primis in faucibus. Praesent facilisis venenatis ante sit amet dignissim. Sed tellus nisi, sagittis nec ullamcorper vitae, placerat condimentum urna. Curabitur eget iaculis purus. Donec in congue odio, eu vestibulum massa. Sed maximus vulputate augue, nec pulvinar sem elementum sed. Aliquam dignissim risus tortor, at semper libero rhoncus sed. Aliquam eget lorem pellentesque, vehicula velit ut, vehicula lectus.

Sed arcu nunc, aliquam vel finibus vitae, mattis laoreet ante. Aenean sit amet nunc vehicula, faucibus tortor id, scelerisque dui. Morbi convallis elit sed leo fringilla, quis bibendum purus sollicitudin. Nulla vitae tincidunt tellus. Vivamus velit metus, maximus et lorem eget, elementum bibendum sem. Suspendisse eleifend pulvinar hendrerit. Maecenas tempus non urna aliquam lacinia. Vivamus sodales varius nibh eu maximus. Ut quis gravida ligula. Sed at molestie dui. Maecenas porttitor, quam et auctor lacinia, nisl enim dapibus sem, eget posuere orci odio sit amet velit. Nunc nisl turpis, ultrices ut pulvinar vel, iaculis nec erat. Vivamus bibendum, lorem id bibendum aliquam, erat augue egestas erat, a tristique massa tellus in magna. Curabitur porttitor pharetra dolor non maximus. Etiam eu felis porta, congue enim non, facilisis nisi.

Ut quam libero, consequat eget viverra nec, tristique vitae purus. Duis in dapibus ligula, at volutpat nibh. Maecenas varius diam et nunc cursus lacinia. Sed quis aliquet ligula. In viverra eros non leo accumsan accumsan. In pharetra, purus quis dictum ullamcorper, felis augue porta augue, id gravida ligula sem quis risus. Ut tempor efficitur ante non accumsan. Sed mattis quam at ante mattis, in ultricies mi venenatis. Donec eget lorem ac enim tincidunt pulvinar vel quis lacus. Quisque consequat suscipit mauris. Ut commodo id orci at consequat.

Vestibulum sem erat, fermentum et nibh non, imperdiet viverra mi. Donec lectus ipsum, laoreet at erat in, molestie rutrum orci. Mauris pretium neque ac tempus ullamcorper. Aliquam in massa interdum, aliquam turpis non, scelerisque erat. Curabitur commodo ligula ultricies justo ullamcorper, et sodales augue euismod. Sed a vehicula sem. Donec malesuada lorem ante, at dapibus magna tincidunt sit amet. Duis vel est quis libero semper mattis. Nam molestie tellus nec pharetra tristique.

Morbi egestas elit sapien, eget auctor libero sollicitudin ut. Aenean id dui tempor, pretium sapien id, fermentum orci. Aenean semper dolor orci, vel ultrices nisl lobortis vel. Duis at sem quis arcu vulputate auctor. Nulla quis pharetra elit, quis feugiat magna. Integer scelerisque tortor eros, vel convallis est congue vel. Vivamus id feugiat massa, non volutpat nulla. Maecenas sit amet facilisis lorem.

Nullam eu vulputate libero. Aenean rhoncus quis erat commodo commodo. Aenean scelerisque tortor quis nunc maximus, pulvinar cursus quam consequat. Quisque et maximus nisl. Nunc convallis eu purus vitae tempus. Aenean pulvinar feugiat libero, vel convallis diam blandit in. Vestibulum faucibus urna nibh, vel ornare nisl commodo at. Nullam interdum in sem non facilisis. Donec volutpat mi lectus, a malesuada enim eleifend eu. Proin tincidunt consequat diam, sed pharetra turpis placerat sed. Duis id est in ligula fermentum mattis.

Donec leo leo, finibus id imperdiet et, porta eget orci. Fusce malesuada turpis ac est egestas, vel euismod lectus accumsan. Phasellus a mauris hendrerit, ullamcorper diam nec, cursus nisi. Duis maximus, justo sit amet interdum faucibus, nisi lectus eleifend risus, commodo varius libero lorem vitae nibh. Suspendisse potenti. Nunc ornare nisi quis diam imperdiet rhoncus. Suspendisse suscipit nibh eget augue vestibulum, eget lacinia erat suscipit. Quisque sed enim feugiat diam accumsan dictum nec euismod elit. Duis leo arcu, placerat eget facilisis et, molestie at metus. Cras id tellus maximus, pharetra massa et, commodo ligula. Nam ac dui interdum, blandit tortor in, ullamcorper orci. Praesent rhoncus orci nisl, sed volutpat risus fringilla sit amet. Morbi a diam tortor.

Integer sed arcu pellentesque, euismod arcu a, ullamcorper diam. Praesent feugiat diam auctor risus egestas scelerisque. Quisque lacinia bibendum sem, id pretium justo convallis quis. Quisque vel massa vulputate, lobortis ligula ut, consequat nunc. Sed eget nisl eget lacus luctus egestas. In ornare pellentesque quam, ut sodales erat placerat eget. Nullam dignissim facilisis augue, vitae imperdiet tellus maximus eget. Duis et sodales leo. Vivamus vel urna at odio elementum bibendum sed eu ex. Donec imperdiet gravida mollis.

Aenean rutrum faucibus purus quis porttitor. Fusce scelerisque ullamcorper laoreet. Donec porttitor pulvinar dolor eget facilisis. Nunc libero urna, molestie sit amet dui in, posuere tempus massa. Nulla porta sed ligula vel congue. Praesent sed luctus tellus. Donec sagittis vel leo in commodo. Donec hendrerit risus vel condimentum ullamcorper.

Curabitur vel venenatis dui. Proin ante ante, condimentum sit amet quam non, aliquam porttitor metus. Donec cursus dignissim mollis. Morbi dictum porta libero, eu finibus nulla commodo et. Cras orci enim, eleifend id lacus ac, pharetra faucibus turpis. Duis ultrices luctus mi. Quisque metus erat, finibus ac fringilla et, ullamcorper et quam. Sed at leo sed mauris auctor maximus in faucibus nunc. Aenean nisi felis, commodo non feugiat at, varius sit amet justo. Vestibulum lobortis nisi arcu, quis facilisis ligula lacinia eget. Fusce ac commodo nisi, id euismod dui.

Cras nec dictum est. Nulla urna erat, bibendum sed elementum convallis, pellentesque id neque. In in dignissim dui. Cras pellentesque sed mauris sit amet venenatis. Maecenas eu ullamcorper velit, non rutrum nibh. Mauris bibendum varius erat, vitae tristique augue euismod at. Nunc ultricies nunc tortor, viverra maximus diam dapibus quis. Nullam malesuada tellus nec orci posuere, id varius dolor posuere.

Ut sit amet mauris sit amet nulla sollicitudin maximus. Nunc semper at diam in ullamcorper. Etiam ultrices tincidunt congue. Nullam nec quam sed sem maximus finibus. Praesent felis urna, ultrices in gravida vel, eleifend vel ligula. Sed nisi mi, iaculis eu dignissim ut, laoreet et nulla. Aenean scelerisque tempus massa. Morbi eget lorem leo. Integer mi velit, ornare vel vestibulum non, sodales vitae diam. Vestibulum viverra porta nulla non bibendum. Vivamus sed tellus et libero sodales ultricies.

Donec gravida posuere nunc sed pretium. Donec tempor scelerisque nulla, eu aliquet nulla sodales in. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Aenean vulputate turpis odio, non porttitor magna tempor non. Proin libero turpis, congue eu dolor vitae, congue sollicitudin nulla. Nulla eleifend sagittis ex, ut euismod tellus egestas ut. Nulla feugiat diam sed lacus accumsan accumsan. Sed arcu nibh, dapibus sit amet pulvinar ut, pulvinar at dolor. Vivamus elementum nisl sit amet fringilla ornare. Morbi dapibus diam at felis eleifend, vitae interdum nunc euismod. Duis vehicula nibh et mauris aliquam tempus. Sed volutpat ultrices scelerisque. Integer feugiat, mi sit amet ornare interdum, eros neque dapibus libero, ac euismod massa erat ut quam. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aliquam ac pellentesque felis. Vestibulum imperdiet sem at dapibus egestas.

Curabitur tempus, eros eget accumsan pellentesque, elit risus rhoncus sapien, non vestibulum nisi purus in tellus. In commodo elementum odio, nec viverra felis placerat a. Mauris cursus neque eget purus mattis consequat. Aliquam erat volutpat. Proin vel urna quam. Integer accumsan lobortis tempor. Aliquam sed leo metus. Duis nisi nisl, viverra vitae lobortis at, euismod at sem. Nunc quis nisi quam. Curabitur consectetur porttitor sem, sed condimentum nunc hendrerit ac. Fusce ultrices at dui at luctus. Pellentesque non suscipit lectus. Nunc sodales nec urna et mollis.

Aenean sed interdum justo. Ut mauris ex, interdum id accumsan vel, consequat sed risus. Pellentesque vel pulvinar lectus. Mauris ac lectus velit. Curabitur porta, felis sed euismod tempus, neque odio interdum massa, sed commodo sapien urna vel ex. Fusce lobortis sem nec lorem egestas, ut congue velit efficitur. In sodales commodo interdum. In tincidunt urna nunc, eget rhoncus elit maximus non.

Duis mauris orci, varius ut nisl eget, pharetra rhoncus felis. Proin erat augue, congue vel augue eget, aliquet aliquam urna. Curabitur feugiat accumsan nunc, ac imperdiet ipsum viverra vel. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aliquam ut imperdiet risus. Nam consectetur dictum placerat. Nullam sed lacus leo. Phasellus sit amet leo a mi mollis mattis. Duis ut quam elit. Aliquam semper hendrerit dolor eu sollicitudin. Fusce efficitur lorem nec hendrerit mattis. Phasellus est orci, auctor non felis quis, interdum dictum augue. Duis vitae accumsan purus.

Nullam a ligula blandit, rhoncus arcu et, iaculis velit. Praesent sed blandit mi. Sed commodo sem quis felis porta, a consequat enim mollis. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Curabitur tempus dui dolor, quis tempor risus commodo eu. Mauris dignissim, massa id elementum porta, risus ligula egestas neque, tempus rhoncus turpis risus eget lorem. Sed sit amet semper nulla. Phasellus ullamcorper, velit ac faucibus maximus, ligula sapien ultrices felis, sit amet malesuada justo metus a dui. Sed et dapibus sapien, id laoreet enim. Suspendisse et justo eget nisl mollis pellentesque vel id metus. Duis lobortis risus non turpis semper, ac ultricies nisl rutrum.

Donec porttitor a dolor ut commodo. Donec a odio id ante ultricies vehicula. Vivamus luctus quis est et maximus. Aliquam turpis ligula, tristique ut elit ac, vestibulum auctor diam. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam commodo blandit nisi non iaculis. Praesent tempor dictum enim, in feugiat nunc fermentum ac. Quisque eros velit, imperdiet et nisl vel, mollis viverra nisl.

Etiam porttitor consectetur erat et vulputate. Nam libero lacus, gravida quis dictum eget, venenatis quis mi. Aenean in arcu justo. Aenean sollicitudin, leo id mollis fermentum, metus tortor mollis leo, eu tincidunt justo odio et justo. Ut eget convallis elit. Nullam quis molestie lectus. Nam ullamcorper semper fringilla. Mauris sed finibus tellus, ac aliquam tellus. Maecenas feugiat ut justo non ullamcorper. Suspendisse vehicula ipsum enim, ut blandit dui malesuada ut. Nullam euismod nec urna vitae accumsan. Sed placerat mi laoreet dignissim posuere. Sed in diam diam.

Praesent vel ligula lacinia, faucibus magna eget, posuere est. Proin arcu massa, scelerisque vitae maximus a, ultricies sodales turpis. Nunc suscipit tellus in enim hendrerit, in efficitur metus porttitor. Phasellus sagittis, turpis eu feugiat ornare, augue augue pellentesque urna, id tempus sem magna at urna. Quisque tincidunt faucibus consequat. Ut venenatis nisl sed rutrum mattis. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Morbi non orci ut nibh facilisis vehicula. Nulla non metus eget turpis laoreet feugiat. Nam lacus lectus, tristique quis commodo in, hendrerit ut sem. Praesent pretium imperdiet rutrum.

Pellentesque vel ultrices metus. Maecenas leo purus, pretium quis eros vel, efficitur dignissim lorem. In hac habitasse platea dictumst. Pellentesque odio ligula, malesuada ac tincidunt maximus, commodo sed elit. Sed porttitor suscipit lectus sed pellentesque. Phasellus a commodo lacus. Donec ac commodo odio. Aenean faucibus ipsum sed nulla tempus luctus vel non dolor. Phasellus ornare elit lectus, eu sodales turpis molestie non.

Sed tempus vitae purus id molestie. Aliquam erat volutpat. In convallis sapien ut nunc elementum accumsan. Maecenas quam tortor, sodales vel venenatis nec, posuere et ex. Nam urna libero, pharetra ut pretium tempus, auctor vitae arcu. In luctus non augue vitae elementum. Sed efficitur congue convallis. Proin elementum nec orci nec malesuada. Nulla porttitor quis urna nec dapibus. Cras malesuada nisl eget libero porttitor, eu dignissim neque tempus. Vestibulum porttitor sodales dolor, quis rhoncus felis dapibus non. Maecenas egestas, nisi ac imperdiet bibendum, lectus tortor sagittis elit, vitae semper libero mi sit amet nunc. Nulla ac tortor at enim pellentesque scelerisque. Nunc dapibus interdum est ac molestie. Nullam condimentum iaculis urna, eget ultrices magna aliquet sed.

Duis posuere tortor eu libero suscipit, sed commodo mauris fermentum. Curabitur laoreet fermentum lacus. Maecenas venenatis eleifend quam, eu gravida tellus tincidunt quis. Donec ultricies, urna efficitur fringilla feugiat, dolor nisi dictum nulla, eu maximus mi nunc vel sem. Vestibulum est nibh, suscipit ut risus vitae, dapibus mattis augue. Ut mollis urna id eros elementum, sed pulvinar mauris ultricies. Quisque a enim purus. Etiam non dapibus neque. Vivamus urna lacus, aliquet eu sem sed, dictum bibendum enim. Sed cursus enim et sem dictum placerat. Mauris et nunc commodo, vulputate odio eget, pharetra ex. Sed vitae hendrerit leo, at bibendum turpis. Pellentesque tincidunt ullamcorper est, eget elementum est maximus et.

Curabitur quis vehicula dolor, at semper sem. Vestibulum vel velit vitae turpis venenatis luctus. Maecenas et urna condimentum, venenatis lacus ut, fringilla libero. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Duis sollicitudin turpis vulputate viverra porta. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Curabitur aliquet convallis nisl in molestie. Integer tempus diam a felis auctor, a ullamcorper eros dictum. Donec porta justo felis, non feugiat purus faucibus ut. Duis at ipsum mollis, vulputate augue id, efficitur nunc. Nullam mattis, nunc et maximus dictum, nibh lacus porttitor erat, et molestie massa lectus quis sapien. Maecenas egestas lorem ut condimentum tristique.

Maecenas eu accumsan neque. Cras vitae consequat ante, vel sagittis nibh. In rutrum odio nibh, ut bibendum lorem finibus eu. Proin vehicula erat in luctus iaculis. Praesent pulvinar tincidunt viverra. In et dignissim elit. Curabitur placerat tortor in vulputate consequat. Vestibulum a feugiat neque, condimentum elementum urna. Maecenas tincidunt turpis a ante aliquam, quis aliquet libero varius. Proin efficitur lorem mauris, in vehicula ex iaculis id. Curabitur rhoncus pulvinar varius. Morbi accumsan rutrum tellus. Vestibulum accumsan urna quis lacus interdum consectetur. Nulla facilisi.

Duis rutrum enim eu lectus imperdiet elementum id in justo. Praesent id elit et ex pharetra commodo. Phasellus a ipsum feugiat, interdum tellus sit amet, ultrices dolor. Maecenas rhoncus lectus at quam elementum pharetra. Aenean in vestibulum lorem. Mauris vel lorem leo. Ut arcu massa, pellentesque dignissim orci dictum, sodales tristique lacus. Ut eget finibus tellus. Integer rhoncus dignissim felis quis fringilla. Nullam urna odio, porttitor non nisl vel, eleifend vulputate nulla.

Ut sit amet ex sit amet tellus interdum ullamcorper. Fusce lacinia tellus et maximus lobortis. Donec eget elementum eros. Morbi commodo erat sed ligula aliquam laoreet. Cras bibendum pharetra arcu vel sollicitudin. Maecenas sodales orci eu nisl hendrerit, eget interdum lectus fringilla. Fusce tempus est nec turpis hendrerit accumsan. Suspendisse lacus risus, efficitur in turpis non, varius aliquam massa. Quisque vitae velit vel ipsum pulvinar dapibus vitae eu purus. Phasellus convallis nec magna eu sagittis.

Praesent non congue ex, nec interdum ante. Quisque elementum arcu id nulla placerat congue. In feugiat metus at tortor accumsan pharetra. Fusce tortor nunc, dictum vitae scelerisque vel, dictum sit amet arcu. Sed suscipit purus a ultricies vehicula. Vivamus ipsum sem, vehicula eget laoreet nec, suscipit ac mi. Mauris non nulla sit amet ligula faucibus ultrices vel eget urna.

Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Aliquam fringilla sed mi in euismod. Quisque odio justo, interdum vel urna eget, sodales vehicula quam. Vivamus feugiat varius elit. Pellentesque in mattis arcu. Suspendisse pharetra libero ex, ac ultrices felis volutpat id. Duis eu placerat mi. Proin dolor dolor, convallis eu lacinia a, lobortis eget velit. Maecenas ut varius nulla.

Cras ullamcorper augue id aliquet commodo. Vestibulum leo erat, porttitor tristique urna vel, consequat sodales elit. Donec finibus eros arcu, non auctor tortor laoreet ut. Duis a pulvinar nisi. Aliquam a urna fringilla enim vehicula blandit vel vitae metus. Maecenas vel aliquam ante. Ut condimentum elementum lacinia. Integer mattis felis non ipsum aliquam, ut porttitor libero vehicula. Maecenas consectetur, massa in venenatis dapibus, dui mi feugiat mauris, sit amet tincidunt lorem erat nec ante. Praesent facilisis neque commodo erat dignissim commodo. Maecenas blandit venenatis tempus. Cras imperdiet elementum urna molestie aliquam. Aenean efficitur lectus risus, quis tristique odio tristique ut. Donec quis sem quis quam maximus molestie. Duis scelerisque posuere malesuada. Maecenas gravida erat nec massa auctor, sit amet aliquet massa gravida.

Nam erat ligula, feugiat id imperdiet fringilla, fermentum non ligula. Praesent non elementum ex. Donec nec consectetur mauris, pulvinar tincidunt lacus. Nam varius, enim ut posuere sagittis, nisl massa condimentum arcu, vitae mattis justo ligula vel ipsum. Cras bibendum placerat orci ut sollicitudin. Aliquam rutrum et elit eu rutrum. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Duis tempus eros in egestas placerat. Fusce ullamcorper varius augue, sit amet interdum nunc finibus id.

Suspendisse potenti. Sed a arcu a justo eleifend ullamcorper. Duis id lobortis nisi. Cras tempor sem nibh. Nunc sapien erat, rhoncus nec cursus in, blandit facilisis erat. Suspendisse vitae enim tellus. Aliquam bibendum sed urna sed facilisis. Proin efficitur quam sem, et tempor lacus molestie faucibus. Nullam et lectus ex. Curabitur sollicitudin convallis pretium. Nulla enim nisi, ullamcorper sit amet leo sit amet, pellentesque vestibulum eros. Nulla finibus ipsum justo, at porta est feugiat quis. Mauris vestibulum tellus sed tincidunt gravida. Nam ornare diam non libero aliquet venenatis. Maecenas non faucibus nulla. Proin sed molestie nibh.

In hac habitasse platea dictumst. Phasellus ut massa varius enim accumsan vehicula. Donec sit amet dignissim sapien, quis sagittis augue. Suspendisse rhoncus arcu massa, ut dictum dolor venenatis eu. Donec egestas est sem, nec suscipit tortor fringilla et. Sed ullamcorper tellus arcu, ac ultrices mauris vestibulum eget. Duis ac tellus id odio semper rhoncus. Quisque aliquet maximus mattis. Proin imperdiet et augue sed egestas. Sed rutrum nisl et odio semper tincidunt. Fusce accumsan ante et est bibendum tincidunt. Ut pellentesque felis in lacus sagittis, at maximus nunc efficitur.

Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Nunc tempor rutrum tempus. Vivamus maximus non erat vel tristique. Proin est sapien, sollicitudin non interdum non, viverra non magna. Proin quis aliquet eros. Sed fermentum dolor eget facilisis euismod. Aliquam pulvinar, nisl ac imperdiet lacinia, velit diam imperdiet erat, quis vulputate tortor est eget sem. In a vulputate arcu, sed aliquam mi. Suspendisse at pharetra quam, non fringilla massa. Aliquam at tortor euismod, facilisis libero id, mattis eros. Donec pretium nulla et ante vulputate volutpat. Fusce turpis est, bibendum vitae tincidunt sed, vestibulum nec ipsum. Maecenas at dolor dignissim, varius orci a, egestas leo. Mauris blandit non neque ut convallis. Sed id elit nunc. Vestibulum maximus at arcu ut ullamcorper.

Nullam neque lorem, efficitur vitae molestie dignissim, sagittis sit amet felis. Aliquam a tortor quis velit gravida pulvinar. Sed interdum tristique pulvinar. Nulla in commodo odio, sed fermentum elit. Phasellus massa lacus, tristique vel finibus congue, tincidunt nec leo. Nunc rhoncus ex metus, eu ultrices augue feugiat vitae. Phasellus libero nisi, blandit at aliquet vitae, eleifend sit amet augue. Sed ac venenatis urna, ac pretium velit. Sed quis ultrices sapien, quis molestie elit. Sed euismod quam eget magna tristique, a lobortis massa blandit. Quisque lectus arcu, ornare at lorem non, venenatis mattis velit. Aenean congue magna vel ex finibus, vel porta magna condimentum. Maecenas cursus faucibus aliquet. Ut finibus velit erat, quis ultrices nunc sagittis id. Vestibulum auctor tristique venenatis. Aenean pharetra erat at ex dignissim gravida.

Nunc molestie massa nec vulputate imperdiet. Ut quis magna sapien. Pellentesque sit amet ex neque. Cras eget mattis lorem. Nam a fermentum ipsum, vitae facilisis velit. Aenean iaculis pharetra elit aliquam aliquam. Etiam ut aliquet nunc, eget suscipit risus.

Sed porttitor maximus neque vel consectetur. Aenean vel magna et purus posuere ultricies. Maecenas tempor varius neque. Aenean dictum urna at lorem tempor lacinia. Quisque eget congue elit, mattis consequat est. Maecenas sed odio gravida, facilisis odio at, vestibulum dui. Cras id ultrices nisl. Curabitur et accumsan ligula, et vestibulum urna. Curabitur a velit ac dolor suscipit congue. Etiam est nunc, porta efficitur nibh vitae, tristique pretium ipsum. Cras lobortis, nibh vel hendrerit gravida, sapien orci malesuada lectus, vitae iaculis sapien ex quis tellus. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Fusce libero urna, pulvinar non ultrices at, euismod id tortor.

Vivamus et dignissim nisi. Vestibulum eleifend sagittis finibus. Praesent vestibulum quis justo vitae cursus. Curabitur sagittis auctor aliquet. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Nam bibendum nisi lectus, ac euismod dui aliquet vitae. Praesent dignissim nunc malesuada dui suscipit, at sollicitudin massa tempor. Curabitur eget justo in sapien dictum hendrerit. Sed sagittis cursus tellus. Nunc eget tristique quam, eu luctus quam. Nunc nec semper magna. Maecenas vel consequat ante. Quisque fermentum lacus ac consectetur fermentum. Phasellus blandit in orci sit amet interdum. Vestibulum accumsan libero purus, non tincidunt erat rhoncus ut. Phasellus condimentum massa non lacus porta, ut efficitur lectus congue.

In dui neque, placerat eu gravida et, aliquet sed ex. Phasellus a diam iaculis, sagittis lacus in, consequat felis. Nullam finibus euismod tellus, et semper felis hendrerit ac. Morbi posuere viverra accumsan. Nam ac ullamcorper leo. Vivamus placerat ultrices commodo. Donec vitae eleifend lorem, in pretium sem. Maecenas turpis sem, gravida ac tortor ac, sollicitudin gravida lacus. Integer condimentum est ante, ut consequat odio scelerisque eget.

Nulla id varius ex. Integer efficitur condimentum blandit. Nulla facilisi. Ut non urna eu nisi mattis pulvinar. Etiam tincidunt orci sit amet lectus imperdiet, et aliquam est pharetra. Nulla convallis orci vitae feugiat dignissim. Ut faucibus, diam sed mattis aliquet, lacus mi fringilla arcu, nec accumsan metus urna nec neque.

Maecenas congue, dolor eu accumsan faucibus, ligula arcu suscipit ligula, quis pretium turpis ligula id leo. Curabitur accumsan dolor a condimentum elementum. Nunc eleifend ex enim, in tristique velit tincidunt vitae. Sed purus libero, rhoncus id elit at, scelerisque faucibus elit. Vestibulum sed dolor quam. Aliquam scelerisque enim risus, eu placerat lectus aliquam in. Nunc sapien diam, mattis quis mollis non, hendrerit id magna.

Morbi suscipit bibendum urna, semper bibendum quam blandit fermentum. Maecenas molestie magna fringilla nisi varius, et scelerisque libero faucibus. Nunc luctus odio orci, convallis vulputate leo euismod non. Aliquam vulputate viverra libero vel dignissim. Curabitur id rhoncus nisl. Nullam tincidunt, velit sit amet dignissim finibus, est nulla gravida libero, non rhoncus leo enim ac purus. Sed nec tellus velit. Aenean nec risus quam. Suspendisse feugiat felis quis elit laoreet placerat. Nullam tortor tortor, condimentum a varius rutrum, euismod porttitor ligula. Aliquam erat volutpat. Nullam hendrerit finibus euismod. Aliquam erat volutpat. Nulla a elementum libero, non varius ex.

Curabitur aliquet ligula ac molestie rhoncus. Praesent tempor, eros et pharetra facilisis, tellus elit lacinia justo, et porttitor augue justo eu enim. Fusce accumsan, urna nec faucibus interdum, dui augue feugiat nunc, eget pharetra mauris erat id est. Sed accumsan est sit amet nisl aliquam condimentum. Duis maximus est nec sem aliquam, eget hendrerit augue semper. Praesent a tristique sapien. Suspendisse egestas nunc eu justo dapibus consequat. Proin lectus magna, iaculis luctus scelerisque et, mattis non dui. Fusce tristique maximus mauris commodo tristique. Sed aliquet, erat eget sollicitudin iaculis, ante quam suscipit purus, at rhoncus mauris dolor sed augue.

Duis tempus eget augue id luctus. Quisque dapibus libero sed maximus cursus. Pellentesque posuere vitae quam in ultrices. Duis tortor purus, condimentum nec lorem sit amet, eleifend varius quam. Nulla risus diam, rhoncus nec enim ac, bibendum cursus nisl. Suspendisse elit nisl, aliquet vitae purus at, tincidunt consectetur nunc. Integer congue justo erat, ac ultricies urna ultricies ut. Donec eu est eu nulla luctus bibendum. Duis a sapien et ligula auctor molestie. Maecenas non quam sed elit volutpat imperdiet. In fringilla semper velit, id aliquam nisi maximus ut. Proin ornare convallis sagittis. Sed massa urna, commodo quis congue non, finibus et est. Donec tristique diam sit amet risus vehicula, viverra dapibus nunc auctor. Praesent non lorem orci.

Integer ullamcorper iaculis scelerisque. Nullam non lacus egestas, blandit quam nec, lobortis mi. Sed quis mauris gravida, rhoncus dolor id, venenatis tellus. Pellentesque iaculis consectetur velit. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Vestibulum tempor molestie metus, eu auctor elit ornare ac. Sed sed lectus vel libero convallis scelerisque eget lacinia eros. Pellentesque non eros at libero dapibus varius vel vitae arcu. Curabitur commodo volutpat tortor.

Curabitur ultricies eget risus tristique iaculis. Nunc consequat diam nisl, in fermentum nibh bibendum sit amet. Quisque sit amet risus vitae libero accumsan eleifend. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Aliquam volutpat finibus tristique. Nullam quis feugiat dui. Phasellus sit amet quam efficitur, consectetur mauris vitae, egestas neque. Integer pellentesque ligula posuere congue consectetur. Vestibulum vel enim ex. Duis eget sapien in tortor placerat cursus. Vivamus rhoncus sapien dui, non tempor est pharetra quis.

Proin ut leo vel tortor consequat mattis. Sed dictum nisi a nibh mattis gravida. Fusce in metus euismod, gravida quam sed, mattis ipsum. Morbi fermentum aliquam rutrum. Quisque commodo ac metus sed consequat. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aliquam iaculis ut leo id commodo. In enim dolor, dictum eget facilisis et, iaculis ac tellus. Nunc quis leo in lectus efficitur tincidunt. Vivamus pulvinar pretium erat, in malesuada tellus pulvinar non. Nulla maximus diam neque, a mattis diam pharetra vitae. Vestibulum sit amet felis non justo malesuada porttitor.

Vivamus nec nibh nibh. Nulla vel nulla magna. Duis tempor, dolor in lacinia gravida, dolor dolor efficitur quam, ac feugiat lacus tellus non risus. Suspendisse in turpis enim. Donec sollicitudin elit et dui gravida, at lacinia massa porttitor. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec pharetra sagittis leo, et viverra nibh finibus in. Vivamus cursus odio sed enim blandit maximus. Suspendisse laoreet tempus mattis. Nam vel purus in sem pellentesque convallis.

Nam scelerisque eget metus nec pretium. Donec interdum lectus quam, non lobortis nisi laoreet eu. Integer euismod sed leo a luctus. Aliquam sodales sit amet dolor sed dapibus. Nulla venenatis dignissim lorem. Cras sed velit et velit ullamcorper lobortis. Fusce mattis sagittis vulputate. Ut sollicitudin in erat et ullamcorper. Sed condimentum vehicula nulla quis tristique. Nullam vestibulum orci sit amet iaculis scelerisque. Phasellus tortor lacus, sollicitudin ac est viverra, iaculis laoreet elit. Pellentesque eu diam venenatis, vulputate augue ut, auctor mi.

Fusce vel dolor nec nibh fringilla placerat. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Suspendisse in rhoncus lorem. Etiam fringilla lacus rhoncus felis vulputate venenatis quis quis est. Vivamus accumsan fringilla ipsum vitae tristique. Vivamus at sapien dolor. Duis commodo neque id orci tincidunt cursus.

Fusce sit amet suscipit justo. Nam placerat eu orci nec lacinia. Etiam sollicitudin pharetra volutpat. Nullam pulvinar, nibh ac fermentum maximus, enim nulla accumsan est, vitae tempus erat neque ut risus. Vestibulum pharetra suscipit massa nec vestibulum. Donec lobortis tortor sed pellentesque finibus. Quisque vitae lacus turpis. Aenean dignissim ante a lorem sollicitudin mattis. Maecenas vel mi fringilla, bibendum orci eu, egestas velit. Phasellus volutpat fringilla ex, consequat scelerisque dolor euismod eu. Cras eget mi enim. Aliquam pretium, magna pellentesque varius dignissim, arcu mauris lacinia dui, et viverra ex urna sit amet quam.

Curabitur auctor volutpat diam vitae vehicula. Vivamus est arcu, efficitur nec interdum et, sagittis quis sem. Nam sodales vitae velit id pharetra. Mauris malesuada est quis nisi mattis, in facilisis lacus tempor. Integer cursus, risus sed molestie sollicitudin, nisi purus mattis justo, eget egestas tellus nisi mollis elit. Aenean sollicitudin justo luctus."""

var needle = "school"  # a word intentionally not in the test data


# ===----------------------------------------------------------------------===#
# Baseline `_memmem` implementation
# ===----------------------------------------------------------------------===#
@always_inline
fn _memmem_baseline[
    type: DType
](
    haystack: DTypePointer[type],
    haystack_len: Int,
    needle: DTypePointer[type],
    needle_len: Int,
) -> DTypePointer[type]:
    if not needle_len:
        return haystack
    if needle_len > haystack_len:
        return DTypePointer[type]()
    if needle_len == 1:
        return _memchr[type](haystack, needle[0], haystack_len)

    alias bool_mask_width = simdwidthof[DType.bool]()
    var first_needle = SIMD[type, bool_mask_width](needle[0])
    var vectorized_end = _align_down(
        haystack_len - needle_len + 1, bool_mask_width
    )
    for i in range(0, vectorized_end, bool_mask_width):
        var bool_mask = SIMD[size=bool_mask_width].load(
            haystack, i
        ) == first_needle
        var mask = bitcast[_uint_type_of_width[bool_mask_width]()](bool_mask)
        while mask:
            var offset = i + countr_zero(mask)
            if memcmp(haystack + offset + 1, needle + 1, needle_len - 1) == 0:
                return haystack + offset
            mask = mask & (mask - 1)

    for i in range(vectorized_end, haystack_len - needle_len + 1):
        if haystack[i] != needle[0]:
            continue

        if memcmp(haystack + i + 1, needle + 1, needle_len - 1) == 0:
            return haystack + i
    return DTypePointer[type]()


# ===----------------------------------------------------------------------===#
# Benchmarks
# ===----------------------------------------------------------------------===#
@parameter
fn bench_find_baseline(inout b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn():
        _ = _memmem_baseline(
            haystack.unsafe_ptr(),
            len(haystack),
            needle.unsafe_ptr(),
            len(needle),
        )

    b.iter[call_fn]()


@parameter
fn bench_find_optimized(inout b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn():
        _ = _memmem(
            haystack.unsafe_ptr(),
            len(haystack),
            needle.unsafe_ptr(),
            len(needle),
        )

    b.iter[call_fn]()


# ===----------------------------------------------------------------------===#
# Benchmark Main
# ===----------------------------------------------------------------------===#
def main():
    var m = Bench(BenchConfig(num_repetitions=1, warmup_iters=10000))
    m.bench_function[bench_find_baseline](BenchId("find_baseline"))
    m.bench_function[bench_find_optimized](BenchId("find_optimized"))
    m.dump_report()
