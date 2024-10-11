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
# RUN: %mojo %s

from bit import pop_count
from hashlib._ahash import AHasher
from hashlib.hash import hash as old_hash
from hashlib._hasher import _hash_with_hasher as hash
from testing import assert_equal, assert_not_equal, assert_true
from memory import memset_zero, UnsafePointer
from time import now
from utils import Span

# Source: https://www.101languages.net/arabic/most-common-arabic-words/
alias words_ar = """
لا, من, هذا, أن, في, أنا, على, ما, هل,
 يا, و, لقد, ذلك, ماذا, أنت, هنا, لم, إلى, نعم, كان, هو, ان, هذه, هناك, عن, فى, كل, ليس, فقط, كنت, الآن, يجب, انا,
 لك, مع, شيء, لكن, لن, الذي, حسنا, كيف, سوف, هيا, نحن, إنه, ـ, أجل, لماذا, إذا, عندما, انه, كذلك, لي, الى, بعد, انت,
 هي, أين, أنه, كانت, حتى, أي, إنها, أعرف, قد, قبل, تلك, الأمر, بعض, أو, مثل, أريد, رجل, لو, أعتقد, ربما, أيها, بخير,
 يكون, عليك, جيد, أنك, شخص, إن, التي, ولكن, أليس, علي, أحد, به, الوقت, يمكن, انها, اليوم, شئ, تعرف, تريد, صحيح, أكثر,
 تكون, لست, كما, أستطيع, منذ, جدا, سيدي, يمكنك, لذا, واحد, لديك, يبدو, أوه, كلا, الرجل, لدي, تفعل, غير, عليه, اذا,
 آخر, حدث, مرة, شكرا, لدينا, الناس, يوجد, له, مكان, سيد, سيكون, أعلم, رائع, مرحبا, آسف, بهذا, وقت, اللعنة, كم,
 ليست, أفضل, بها, معك, أنها, الذى, الكثير, قلت, بك, يحدث, الان, يكن, يوم, وأنا, واحدة, بي, أخرى, ولا, علينا,
 أبي, بأن, ثم, تعال, هكذا, يمكنني, هم, ألا, بالطبع, أنني, المكان, بذلك, معي, لهذا, ها, شىء, انك, إلهي, تستطيع,
 العمل, العالم, الحقيقة, الليلة, بالتأكيد, حقا, تعلم, أمي, الطريق, حال, لى, لها, الأن, هؤلاء, فعل, توقف,
 عمل, حول, لنا, خلال, اعتقد, السيد, انظر, منك, أى, أفعل, فعلت, لأن, إذن, قال, الجميع, تم, الجحيم, هى, فيه,
 جيدة, عنه, بشكل, بما, تقول, لديه, ثانية, لذلك, أكون, دعنا, ايها, المال, يمكننا, الذهاب, متى, تعتقد,
 اريد, عليها, أذهب, ستكون, فضلك, بدون, أرجوك, التى, شيئا, نذهب, لكي, نفسك, بنا, اين, وأنت, لكم, اي,
 بين, إنهم, أرى, المنزل, بحق, كنا, عند, أم, منه, نفس, اذهب, حيث, مجرد, أقول, تبدو, الحياة, أيضا,
 تحت, الأشياء, معه, يريد, أننا, أنظر, لما, اعرف, إلي, ثلاثة, انتظر, الرجال, الذين, حصلت, أني,
 سعيد, لابد, عزيزتي, الشيء,
  فكرة, انهم, الله, الباب, سيدى, دائما, رأيت, مشكلة, استطيع, تكن, تذهب, ليلة, شيئ, أظن, طوال,
  جميل, وهو, الشرطة, او, دولار, السيارة, وهذا, كبير, مني, بسرعة, النار, الأمور, سمعت, أشعر, يعرف, 
  أعني, لدى, بهذه, أحب, سنوات, بأس, الأفضل, بالنسبة,
   أنتم, عظيم, يقول, جميلة, جون, جاك, بسبب, الوحيد, أمر, بل, بالفعل, الشخص, الي, دعني, خارج, اجل, الخير, ــ,
   حالك, للغاية, فحسب, كانوا, أردت, فتاة, بشأن, يعني, كبيرة, ترى, آسفة, دقيقة, أنهم, يستطيع, احد, بأنك, تعمل,
   تريدين, فيها, اليس, رائعة, رجال, نوع, حياتي, الأرض, البيت, قتل, اوه, والآن, مات, بكل, تعرفين, أحتاج, نستطيع,
   جديد, صباح, ألم, عيد, منها, يعمل, الموت, إليك, جميع, لأنه, لحظة, لكني, الامر, عشر, لكنه, بحاجة, بأنه, أتمنى,
   إليه, عنك, الفتاة, لهم, بالضبط, سأكون, اعلم, اللعين, رقم, طريق, منهم, المدينة, الحب, لنذهب, خذ, أكن, فوق,
   عزيزي, دون, الـ, صغيرة, الرئيس, تتحدث, ترجمة, صديقي, فقد, الصغير, ولم, ساعة, يفعل, غرفة, وماذا, المرة, قام,
   إلا, عام, هذة, متأكد, دقائق, سيارة, فعله, سعيدة, مما, ومن, معنا, سبب, سأذهب, الطريقة, الأطفال, سنة, بينما,
   يرام, السبب, أننى, أول, اى, أريدك, قمت, الأولى, المدرسة, ذهبت, لطيف, نفسي, الا, الجنس, أية, أقصد, غريب, نفعل,
   الصباح, حالة, المزيد, أبدا, مهما, اسمع, لأنك, أحاول, وقد, ايضا, أحبك, اكثر, فرصة, رأيك, افعل, الحصول, صغير,
   الماء, جيدا, التحدث, يمكننى, الساعة, طريقة, أيتها, كثيرا, سيدة, خمسة, وجدت, قليلا, وانا, اخرى, الليل, تعني,
   تماما, نهاية, عرفت, اني, أفكر, معها, الأول, لكنك, تعالي, البعض, أفهم, أخبرك, حياة, أتعرف, نفسه, الواقع,
   أيام, انني, تأتي, لديهم, فهمت, لـ, لديها, الحرب, الأقل, أخبرني, إنك, بـ, الصغيرة, تحتاج, بدأت, حياتك,
   عني, إذهب, عندي, تقلق, نحتاج, إنتظر, أصبح, مجنون, يكفي, اننا, خطأ, الطفل, نصف, أكبر, الخاص, عليهم,
   نريد, لأنني, حان, تعلمين, نعرف, هنالك, رفاق, لكنني, معى, دكتور, جديدة, هلا, افضل, طفل, عنها, أتعلم, تقوم,
   أعمل, بد, الهاتف, بالخارج, السيدة, الطعام, ثلاث, أقوم, صديق, أتحدث, فرانك, الجديد, مالذي, للتو, سيدتي,
   طويلة, وما, السجن, أشياء, فأنا, أخبرتك, العديد, أعطني, أراك, أخي, سام, قالت, فريق, فيما, جو, يتم,
   نكون, وليس, يذهب, ممكن, لمدة, حق, اسف, يجري, تفعله, مثلك, وبعد, تشعر, تحب, اخر, رؤية, طويل, والدك,
   ذهب, آه, أقل, حصل, لكى, اللعنه, سأفعل, يعلم, كله, القيام, فتى, الممكن, أخرج, النوم, داخل, جورج,
   رجاء, أصبحت, الخاصة, اذن, ذات, جميعا, منا, الموضوع, الفتى, اللقاء, أخر, كي, كلمة, عبر, أود, بيت,
   تفهم, تفعلين, علاقة, بى, نيويورك, الآخر, بلا, مايكل, نظرة, ونحن, الخارج, تحاول, المشكلة, بواسطة, كن,
   المفترض, قل, يارجل, تظن, يقوم, مليون, أخذ, توم, يمكنه, مباشرة, سيئة, الحال, العودة, حاول, عندك,
   تكوني, ميت, الكبير, الفتيات, النساء, رئيس, أسرع, النهاية, قادم, أحضر, جزء,
    الهي, ذاهب, العام, لكنها, أتريد, بخصوص, الوغد, حقيقي, إنني, البقاء, حبيبتي, بهم, المساعدة, تصبح, عشرة, أحدهم,
    الخروج, قصة, مستحيل, أربعة, وهي, أبى, كلها, ضد, حاولت, القادمة, يأتي, تفضل, أسمع, تمت, توجد, لكل, العشاء,
    الغرفة, وانت, وسوف, خمس, تذكر, أصدق, ألف, بنفسك, شباب, الماضي, دعونا, الأسبوع, نتحدث, نسيت, بأنني, منزل,
    وضع, ولد, أنتي, جاهز, رسالة, دي, ابن, اكون, حقيقة, مايك, حين, عائلة, أدري, وكان, القائد, للمنزل, مساعدتك,
    غدا, ظننت, ولن, المرأة, لهذه, تحرك, يهم, تبقى, الطبيب, اسم, انظري, تبا, أتذكر, فترة, ساعات, تفكر, تحصل,
    بأي, النقود, لعبة, زوجتي, الكلام, ستفعل, أسف, فهو, الملك, مدينة, بكم, الوحيدة, أمام, عدد, اخرج, بول, سأعود,
    جئت, لأني, تحدث, السلامة, الماضية, أمك, اعتقدت, مره, مساء, بطريقة, الرب, ابدا, أهذا, وفي, وكل, أتيت, منكم,
    انتهى, بوب, بعيدا, ضع, وجود, تعود, زلت, اللعينة, نقوم, كلنا, أحصل, يريدون, تأخذ, المحتمل, الشمس, بدأ, 
    ارجوك, المسيح, جاء, كهذا, سنذهب, تعالى, إثنان, فعلا, حتي, سيحدث, الجيد, وشك, القادم,
     معرفة, صورة, أعود, اسمي, طلب, آنسة, الثانية, فقدت, حفلة, تنظر, مثير, اننى, وصلت, أنتظر, السماء, يقولون, الهراء,
     معهم, ابي, وعندما, مجموعة, العاهرة, ماري, حسن, الزواج, نحو, دعيني, الجديدة, مهم, أمس, اتصل, ابتعد, هراء, ستة,
     الأخرى, يحصل, ولكني, الطائرة, أصدقاء, الحظ, مشاكل, الترجمة, تبدين, لسنا, مستعد, ولكنه, اقول, أولئك, النوع, أثناء,
     اسمه, اسمك, مكتب, والدي, ينبغي, منى, كرة, بيتر, عدم, أطفال, الإطلاق, سوى, مضحك, الوضع, جي, الأخيرة, صعب, أحمق,
     يحاول, الشئ, حينما, الأشخاص, البحر, إليها, عرض, بأني, يحتاج, سيء, عالم, كثير, الداخل, الكتاب, ذو, الأيام, خلف,
     بعضنا, يعود, ام, اللعبة, إني, رأسك, شركة, زال, بشيء, الاشياء, قطعة, خائف, واضح, أمى, موجود, علم, يعد, أبحث,
     الدخول, جين, امرأة, متأكدة, هيه, تخبرني, مدى, إلهى, احب, عما, نرى, بيننا, تعيش, قتلت, الأحمق, تشارلي,
     بيل,
      عليكم, سؤال, طلبت, الهواء, وهذه, صوت, انتم, ميلاد, ماكس,
       تعتقدين, الحديث, الجانب, صديقك, ذا, خطر, أطلق, الشارع, عملية, ببعض, تتكلم, مختلف, تحمل, مساعدة, 
       بضعة, المناسب, المنطقة, قم, بالداخل, البداية, لأجل, زوجتك, مقابل, يحب, هاري, ممتاز, قريبا, سنكون,
        فعلته, بتلك, التفكير, أسفل, للعمل, العجوز, امي, الكلب, انتظري, مازال, إننا, اشعر, الجيش, شرطة
"""

# Source: https://github.com/tkaitchuck/ahash/blob/7d5c661a74b12d5bc5448b0b83fdb429190db1a3/tests/map_tests.rs#L9
alias words_en: String = """
    a, ability, able, about, above, accept, according, account, across, act, action,
    activity, actually, add, address, administration, admit, adult, affect, after,
    again, against, age, agency, agent, ago, agree, agreement, ahead, air, all,
    allow, almost, alone, along, already, also, although, always, American, among,
    amount, analysis, and, animal, another, answer, any, anyone, anything, appear,
    apply, approach, area, argue, arm, around, arrive, art, article, artist, as,
    ask, assume, at, attack, attention, attorney, audience, author, authority,
    available, avoid, away, baby, back, bad, bag, ball, bank, bar, base, be, beat,
    beautiful, because, become, bed, before, begin, behavior, behind, believe,
    benefit, best, better, between, beyond, big, bill, billion, bit, black, blood,
    blue, board, body, book, born, both, box, boy, break, bring, brother, budget,
    build, building, business, but, buy, by, call, camera, campaign, can, cancer,
    candidate, capital, car, card, care, career, carry, case, catch, cause, cell,
    center, central, century, certain, certainly, chair, challenge, chance, change,
    character, charge, check, child, choice, choose, church, citizen, city, civil,
    claim, class, clear, clearly, close, coach, cold, collection, college, color,
    come, commercial, common, community, company, compare, computer, concern,
    condition, conference, Congress, consider, consumer, contain, continue, control,
    cost, could, country, couple, course, court, cover, create, crime, cultural,
    culture, cup, current, customer, cut, dark, data, daughter, day, dead, deal,
    death, debate, decade, decide, decision, deep, defense, degree, Democrat,
    democratic, describe, design, despite, detail, determine, develop, development,
    die, difference, different, difficult, dinner, direction, director, discover,
    discuss, discussion, disease, do, doctor, dog, door, down, draw, dream, drive,
    drop, drug, during, each, early, east, easy, eat, economic, economy, edge,
    education, effect, effort, eight, either, election, else, employee, end, energy,
    enjoy, enough, enter, entire, environment, environmental, especially, establish,
    even, evening, event, ever, every, everybody, everyone, everything, evidence,
    exactly, example, executive, exist, expect, experience, expert, explain, eye,
    face, fact, factor, fail, fall, family, far, fast, father, fear, federal, feel,
    feeling, few, field, fight, figure, fill, film, final, finally, financial, find,
    fine, finger, finish, fire, firm, first, fish, five, floor, fly, focus, follow,
    food, foot, for, force, foreign, forget, form, former, forward, four, free,
    friend, from, front, full, fund, future, game, garden, gas, general, generation,
    get, girl, give, glass, go, goal, good, government, great, green, ground, group,
    grow, growth, guess, gun, guy, hair, half, hand, hang, happen, happy, hard,
    have, he, head, health, hear, heart, heat, heavy, help, her, here, herself,
    high, him, himself, his, history, hit, hold, home, hope, hospital, hot, hotel,
    hour, house, how, however, huge, human, hundred, husband, I, idea, identify, if,
    image, imagine, impact, important, improve, in, include, including, increase,
    indeed, indicate, individual, industry, information, inside, instead,
    institution, interest, interesting, international, interview, into, investment,
    involve, issue, it, item, its, itself, job, join, just, keep, key, kid, kill,
    kind, kitchen, know, knowledge, land, language, large, last, late, later, laugh,
    law, lawyer, lay, lead, leader, learn, least, leave, left, leg, legal, less,
    let, letter, level, lie, life, light, like, likely, line, list, listen, little,
    live, local, long, look, lose, loss, lot, love, low, machine, magazine, main,
    maintain, major, majority, make, man, manage, management, manager, many, market,
    marriage, material, matter, may, maybe, me, mean, measure, media, medical, meet,
    meeting, member, memory, mention, message, method, middle, might, military,
    million, mind, minute, miss, mission, model, modern, moment, money, month, more,
    morning, most, mother, mouth, move, movement, movie, Mr, Mrs, much, music, must,
    my, myself, name, nation, national, natural, nature, near, nearly, necessary,
    need, network, never, new, news, newspaper, next, nice, night, no, none, nor,
    north, not, note, nothing, notice, now, n't, number, occur, of, off, offer,
    office, officer, official, often, oh, oil, ok, old, on, once, one, only, onto,
    open, operation, opportunity, option, or, order, organization, other, others,
    our, out, outside, over, own, owner, page, pain, painting, paper, parent, part,
    participant, particular, particularly, partner, party, pass, past, patient,
    pattern, pay, peace, people, per, perform, performance, perhaps, period, person,
    personal, phone, physical, pick, picture, piece, place, plan, plant, play,
    player, PM, point, police, policy, political, politics, poor, popular,
    population, position, positive, possible, power, practice, prepare, present,
    president, pressure, pretty, prevent, price, private, probably, problem,
    process, produce, product, production, professional, professor, program,
    project, property, protect, prove, provide, public, pull, purpose, push, put,
    quality, question, quickly, quite, race, radio, raise, range, rate, rather,
    reach, read, ready, real, reality, realize, really, reason, receive, recent,
    recently, recognize, record, red, reduce, reflect, region, relate, relationship,
    religious, remain, remember, remove, report, represent, Republican, require,
    research, resource, respond, response, responsibility, rest, result, return,
    reveal, rich, right, rise, risk, road, rock, role, room, rule, run, safe, same,
    save, say, scene, school, science, scientist, score, sea, season, seat, second,
    section, security, see, seek, seem, sell, send, senior, sense, series, serious,
    serve, service, set, seven, several, sex, sexual, shake, share, she, shoot,
    short, shot, should, shoulder, show, side, sign, significant, similar, simple,
    simply, since, sing, single, sister, sit, site, situation, six, size, skill,
    skin, small, smile, so, social, society, soldier, some, somebody, someone,
    something, sometimes, son, song, soon, sort, sound, source, south, southern,
    space, speak, special, specific, speech, spend, sport, spring, staff, stage,
    stand, standard, star, start, state, statement, station, stay, step, still,
    stock, stop, store, story, strategy, street, strong, structure, student, study,
    stuff, style, subject, success, successful, such, suddenly, suffer, suggest,
    summer, support, sure, surface, system, table, take, talk, task, tax, teach,
    teacher, team, technology, television, tell, ten, tend, term, test, than, thank,
    that, the, their, them, themselves, then, theory, there, these, they, thing,
    think, third, this, those, though, thought, thousand, threat, three, through,
    throughout, throw, thus, time, to, today, together, tonight, too, top, total,
    tough, toward, town, trade, traditional, training, travel, treat, treatment,
    tree, trial, trip, trouble, true, truth, try, turn, TV, two, type, under,
    understand, unit, until, up, upon, us, use, usually, value, various, very,
    victim, view, violence, visit, voice, vote, wait, walk, wall, want, war, watch,
    water, way, we, weapon, wear, week, weight, well, west, western, what, whatever,
    when, where, whether, which, while, white, who, whole, whom, whose, why, wide,
    wife, will, win, wind, window, wish, with, within, without, woman, wonder, word,
    work, worker, world, worry, would, write, writer, wrong, yard, yeah, year, yes,
    yet, you, young, your, yourself"""

# Source: https://www.101languages.net/hebrew/most-common-hebrew-words/
alias words_he = """
לא , את , אני , זה , אתה ,
 מה , הוא , לי, על, כן, לך, של, יש , בסדר , אבל , כל , שלי , טוב , עם, היא, אם, רוצה,
 שלך, היה, אנחנו, הם, אותך, יודע, אז, רק, אותו, יכול, אותי, יותר, הזה, אל, כאן, או,
 למה, שאני, כך, אחד, עכשיו, משהו, להיות, היי, תודה, כמו, אין, זאת, איך, נכון, מי, שם,
 לו, צריך, לעשות, קדימה, לנו, חושב, כמה, שאתה, זו, גם, יודעת, אותה, עוד, באמת, הייתי,
 שהוא, אולי, בבקשה, עושה, פשוט, שזה, דבר, מאוד, כבר, שלא, נראה, לעזאזל, אתם, כדי, ואני,
 פה, אלוהים, הנה, פעם, האם, בוא, שלו, איפה, הרבה, כי, יכולה, שלנו, אומר, יהיה, אותם, עד,
 קצת, לפני, זמן, הכל, ממש, אבא, הולך, מר, אדוני, לראות, ובכן, מישהו, חייב, עדיין, לה,
 אף, בכל, בדיוק, היום, אנשים, ללכת, מצטער, היית, שלום, קרה, שוב, אוהב, אחת, הייתה, אמא,
 חשבתי, בן, איזה, יום, לדבר, תמיד, צריכה, לזה, חושבת, להם, היו, שאת, שיש, רואה, אפילו,
 בטח, כולם, בגלל, שום, שהיא, אחר, חבר, בשביל, קורה, איתך, הזאת, אמרתי, אדם, תן, צריכים,
 הזמן, יכולים, ואז, כלום, רגע, האלה, אחרי, מבין, תראה, בטוח, שהם, לכם, בו, אמר, מדי,
 ג, טובה, אותנו, תהיה, אנו, אך, ככה, בזה, יפה, כזה, אוכל, מותק, מספיק, בואו, אפשר, שלה,
 דברים, הכי, מזה, מקום, בואי, לכאן, אה, בבית, דרך, איתי, מתכוון, ביותר, הביתה, לומר, אחי,
 מת, הזו, ה, הדבר, מדבר, שאנחנו, לעזור, לעולם, זהו, לדעת, כאילו, גדול, אוהבת, שנים, בי,
 מכאן, יודעים, לקחת, ראיתי, בלי, נהדר, די, כסף, הו, היתה, מהר, עליי, י, מצטערת, וזה, ילד,
 לשם, קשה, חכה, לאן, ואתה, ממני, ו, תגיד, רוצים, שני, הלילה, עליך, כמובן, עליו, נחמד, לכל,
 להגיד, סליחה, אמרת, ל, מוכן, מחר, בא, ולא, והוא, אוקיי, אומרת, גברת, בך, נלך, בית, מעולם,
 שלהם, אי, הבחור, עבודה, למצוא, נמצא, חייבים, מכיר, מנסה, ב, ואת, מתי, תעשה, בשבילך, מספר,
 כדאי, דקות, שלכם, האמת, עושים, אלה, חייבת, דולר, הכסף, כעת, לילה, איש, עלי, לצאת, רציתי,
 לתת, בחור, בכלל, איתו, רע, עשית, מרגיש, הכול, בעיה, עבור, אמור, לקבל, עובד, בנאדם, הולכים,
 החיים, נוכל, מאמין, סוף, ידעתי, הולכת, לב, בחייך, היכן, שנה, זוכר, ממך, הגיע, קטן, החוצה,
 תוכל, בזמן, הן, ילדים, נשמע, חיים, בדרך, אכפת, נעשה, הבית, ש, ידי, בוקר, עשה, לחזור, המקום,
 הבא, מקווה, קח, עשיתי, חשוב, הי, אלו, בחיים, גבר, ללא, במקום, משנה, הלו, להרוג, שמעתי, העולם,
 ספר, ר, זונה, עצמך, האנשים, למעלה, בני, לגבי, מאוחר, כמעט, תראי, לספר, בקשר, שמח, להתראות,
 לבד, הדרך, האלו, שלוש, יופי, לגמרי, מדוע, אליך, מפה, קודם, ראית, למטה, בה, להגיע, חלק, מגיע,
 מ, בין, לבוא, אתן, היינו, חרא, הדברים, תפסיק, אחרת, לשמוע, מזל, המפקד, בחוץ, אהיה, הספר, אליי,
 ערב, תקשיב, אישה, השני, לחשוב, הערב, מבינה, מיד, בשבילי, למעשה, אוי, הראשון, אלי, תני, א,
 חברים, בטוחה, רבה, ומה, מאז, ביום, במשך, בהחלט, עלינו, ון, לעבוד, השם, כולנו, לאחר, הראשונה,
 למען, מניח, מוזר, בתוך, איזו, חזק, העבודה, מהם, לפחות, אמרה, האיש, ביחד, כנראה, שיהיה, בת,
 אלא, הבעיה, כאשר, ימים, ואם, אימא, קטנה, ברגע, אתכם, אעשה, איני, הדלת, משחק, חדש, בעוד, סתם,
 לבית, בחזרה, לאכול, להביא, אתמול, נורא, תראו, אדירים, יחד, גדולה, בעולם, חתיכת, לפעמים, מקרה,
 בפנים, נו, עומד, ברור, אבי, הפעם, ממנו, שעות, שלומך, בלילה, אומרים, מתכוונת, אינך, הילד, כרגע,
 האחרון, ביי, הא, טובים, העיר, חיי, הילדים, נראית, חכי, יהיו, למות, מצחיק, חוץ, תורגם, שהיה,
 חזרה, מאד, הראש, רעיון, הרגע, אהבה, לשאול, להישאר, שאתם, יקרה, מושג, בלתי, איתה, להשיג, ראש,
 יכולתי, תחת, עצמי, מכל, קוראים, אליו, שמעת, כ, להיכנס, אמיתי, הבן, תסתכל, כלומר, חברה, עליה,
 והיא, ילדה, לשחק, העניין, ועכשיו, קשר, הגדול, לעזוב, החבר, נפלא, האחרונה, חמש, בפעם, עצור, כפי,
 לישון, שתי, צודק, וגם, שלושה, ליד, לחיות, קרוב, רב, נגמר, לקרוא, שאין, תא, תדאג, יוצא, האדם,
 והם, גמור, בבוקר, קל, מתה, באופן, בשם, מעט, הבאה, יורק, מוכנה, היחיד, לכן, תגידי, חי, חצי, איי,
 לוקח, נעים, נהיה, לעצור, גרוע, לפגוש, נשים, שעה, נגד, הטוב, מים, חושבים, ממה, פנימה, מרגישה,
 לפה, שתיים, וכל, אסור, פנים, שכל, מתחיל, אשר, אותן, היחידה, שומע, כמוך, בקרוב, לנצח, המכונית,
 מחוץ, בחורה, חוזר, להבין, הבנתי, עלייך, לעבור, בערך, פעמים, בחדר, וואו, כאלה, קיבלתי, לכי, חודשים,
 אלך, ארוחת, סוג, מדברת, מכירה, מאמינה, כה, זקוק, הקטן, אידיוט, מדהים, מצאתי, הסיבה, אינני, בנות,
 בתור, לשמור, החברה, להמשיך, התחת, כשאני, שמי, נתראה, ימי, הסיפור, שכן, סיפור, בצורה, אקח, סלח,
 לזוז, רציני, הכבוד, שמישהו, פחות, מדברים, שאלה, סיכוי, מתחת, אחרים, הללו, נכנס, תביא, התכוונתי,
 שהייתי, הבוקר, ראשון, באותו, בחורים, בהם, לנסות, להשתמש, אדון, במה, שווה, זוכרת, טיפש, מיליון,
 מוכנים, להתחיל, להראות, אנג, אראה, מלא, המשפחה, לפי, מאשר, פרנק, מטורף, לכך, שבוע, מגניב, צא,
 ואנחנו, לתוך, לחכות, מאיפה, איתנו, מחדש, דעתך, שונה, חסר, תוך, נותן, לקנות, להכיר, הבחורה,
 ינא, מחפש, שבו, ישר, חדשות, חדר, אש, אפשרי, מצוין, הלא, ים, מתוקה, עזרה, שאם, מעולה, כדור,
 תירגע, אמרו, באה, החדש, בעיות, שאנו, המצב, הלך, לשתות, מעבר, מעל, טעות, כשאתה, עבר, עליהם,
 נשאר, ויש, שב, מייקל, אלף, לקח, ארבע, סיבה, מצב, מן, מסוגל, מידי, להרגיש, בעל, משפחה, שזו,
 שוטר, בחיי, מעניין, ההוא, קפה, הזדמנות, כלב, כלל, מקבל, שונא, מחכה, מפני, זין, תחזור, שקט,
 באתי, מוצא, אביך, ניסיתי, תקשיבי, חן, מצוות, רוח, מוקדם, קפטן, תהיי, מאיתנו, מבטיח, מושלם,
 ידעת, עניין, כוח, המון, פי, חולה, אוהבים, אינו, דם, הנשיא, משם, למרות, גורם, לאט, כבוד, ס,
 בעבר, להתקשר, אלייך, משוגע, עשר, ללמוד, שטויות, בנוגע, צוחק, לבדוק, בצד, להאמין, חדשה, עצמו,
 לגרום, המשחק, שרה, לעצמך, במיוחד, המשטרה, צוות, אחזור, שאמרתי, גברים, קורא, בראש, רחוק,
 למקום, לשלם, להפסיק, מיוחד, הז, שמו, שמחה, כיף, אגיד, למי, ניתן, מאחורי, תמשיך, כיצד,
 להוציא, מתים, כולכם, אצל, חבל, האישה, לעצמי, גברתי, תוכלי, רואים, דוד, להציל, שצריך,
 בעלי, דוקטור, חג, לעבודה, בוודאי, תעשי, הוד, מילה, ברצינות, הארץ, עשינו, לאנשים, רצה, 
 עזוב, יצא, נתן,
  שניות, בעיר, סי, חשבת, שאלות, אלינו, ידע, תנו, לשים, שאולי, בכך, יכולת, אן, היד, שאוכל,
  מין, דקה, לדאוג, שמה, תרצה, ראה, הצילו, נוסף, החרא, אופן, כשהוא, צעיר, הפה, עולה, עובדת,
  שמך, לתפוס, נמצאת, כלבה, האקדח, עדיף, הטלפון, טום, פול, חכו, קר, תלך, במקרה, יעשה, שניכם,
  הארי, זוז, יקירתי, בהצלחה, לשבת, אנא, דין, מכיוון, יד, הקטנה, לבן, בנו, בעצמי, יין, תוריד,
  למישהו, מייק, מול, נזוז, ככל, הלוואי, בעצמך, לרגע, קשור, בשקט, האל, ישנה, מעמד, כזאת, 
  רד, אחורה, איכפת, איתם, ממנה, חם, מבקש, שש, מידע, השנה,
   אכן, אהבתי, בשעה, בסוף, שקרה, לכו, אליה, לבחור, תחשוב, ספק, המים, הפנים, לכולם, תדאגי,
   קחי, שתוק, לברוח, מתוק, ארלי, התיק, שים, מישהי, לקרות, לטפל, לחפש, הידיים, ח, במצב, ואל
"""

# Source: https://www.101languages.net/latvian/most-common-latvian-words/
alias words_lv = """
    ir, es, un, tu, tas, ka, man, to, vai, ko, ar, kas, par, tā, kā, viņš, uz, no, tev, 
    mēs, nav, jūs, bet, labi, jā, lai, nē, mani, ja, bija, viņa, esmu, viņu, tevi, esi, 
    mums, tad, tikai, ne, viņi, kad, jums, arī, viss, nu, kur, pie, jau, tik, tur, te, vēl, 
    būs, visu, šeit, tagad, kaut, ļoti, pēc, viņam, taču, savu, gan, paldies, būtu, mūsu, 
    šo, lūdzu, mans, kāpēc, kungs, kāds, varbūt, tās, jūsu, cik, ak, daudz, jo, esam, 
    zinu, mana, zini, visi, būt, tam, šī, var, līdz, viens, pa, pat, esat, nekad, domāju, 
    nezinu, vairs, tiešām, tie, vien, kurš, varētu, dievs, neesmu, prom, tieši, kādu, aiziet, 
    šis, manu, protams, vajag, neko, vienkārši, tāpēc, gribu, varu, nāc, atpakaļ, mūs, 
    kārtībā, iet, kopā, viņiem, pats, pirms, domā, vienmēr, gribi, nekas, bez, tava, 
    vienu, ej, viņai, vairāk, notiek, nevaru, pret, tavs, teica, tavu, biju, dēļ, viņas, 
    laiku, neviens, kādēļ, vari, labāk, patīk, dari, mājās, nebija, cilvēki, ārā, viņus, 
    ejam, kāda, piedod, laikam, atkal, šķiet, trīs, sevi, ser, laiks, laika, nekā, manis, 
    iekšā, labs, tāds, darīt, harij, nevar, viena, lieliski, kuru, šīs, sauc, šurp, teicu, 
    laikā, tos, pagaidi, neesi, tevis, draugs, pārāk, tēvs, šodien, teikt, dienu, visiem, 
    tātad, notika, hei, zināt, bijis, sveiks, atvainojiet, tika, naudu, varam, savas, citu, 
    tādu, manas, redzi, šajā, kam, tajā, jābūt, vecīt, tiem, runā, cilvēku, taisnība, saka, 
    visus, mīlu, lietas, grib, tēt, izskatās, tiek, noteikti, nozīmē, kamēr, divi, it, tāpat, 
    tāda, ilgi, katru, dēls, noticis, jauki, redzēt, pareizi, lūk, kundze, aiz, iespējams, 
    pateikt, nebūtu, gandrīz, vīrs, cilvēks, ātri, žēl, pasaules, rokas, liekas, palīdzēt, 
    līdzi, visas, saki, negribu, vietā, gadus, starp, skaties, tomēr, tūlīt, džek, nevajag, 
    sev, vajadzētu, būšu, dzīvi, droši, gadu, priekšu, skaidrs, gribēju, nāk, paskaties, mazliet, 
    tikko, nebūs, augšā, ceru, joprojām, nevis, ātrāk, ļauj, gribētu, liels, zina, vārdu, reizi, 
    pasaulē, savā, sveiki, dienas, miris, dod, priekšā, galā, klau, cilvēkiem, tavas, patiesībā, 
    visa, vārds, gatavs, durvis, velns, nedaudz, naudas, redzēju, velna, manā, drīz, pāri, dzīve, 
    vēlies, nemaz, priekš, bērni, vieta, pāris, darbu, vajadzīgs, tālāk, rīt, roku, klāt, grūti, 
    beidz, laba, klausies, dara, varat, sveika, biji, vismaz, kopš, redzu, saproti, kura, draugi, 
    zemes, šovakar, patiešām, kaa, vietu, dieva, vajadzēja, mašīnu, lejā, saku, ceļu, gada, tādēļ, 
    cauri, runāt, ņem, oh, divas, lieta, tikt, šie, teici, vēlāk, vaļā, nogalināt, redzējis, jāiet, 
    nespēju, savus, atceries, ūdens, šejienes, labu, diena, mīļā, atvaino, doties, atrast, saprotu, 
    abi, reiz, jādara, nesaprotu, meitene, darbs, nevari, tai, nedomāju, pilnīgi, nakti, nekādu, 
    pati, gadiem, vēlos, taa, kādas, cits, ejiet, pirmais, a, būsi, mamma, lietu, slikti, pašu, 
    acis, diezgan, pasaki, gadā, puiši, asv, sava, nost, cilvēkus, džeks, manuprāt, mājas, o, 
    bērns, leo, otru, nopietni, vecais, laukā, caur, dzīves, izdarīt, sieviete, vienalga, 
    nogalināja, dzīvo, kādreiz, čau, sirds, paliec, gribat, vēlreiz, kuras, mazais, vietas, 
    piedodiet, laipni, palikt, brauc, ei, the, paliek, apkārt, sievietes, tālu, garām, pirmo, 
    dzīvot, nāciet, runāju, kuri, tiks, jüs, ceļā, nauda, nevienam, māja, vienīgais, īsti, 
    sapratu, gluži, svarīgi, atvainojos, i, sen, iespēja, tavā, pavisam, nāves, māte, citi, 
    viegli, zem, notiks, darba, nepatīk, daži, galvu, dienā, hallo, bērnu, neesam, kungi, beidzot, 
    nedrīkst, vajadzēs, māju, sieva, kādam, puika, kļūst, prieks, esot, iesim, daļa, pasaule, 
    pietiek, visā, saviem, rīta, pagaidiet, tētis, mājā, mieru, vīru, palīdzību, dzirdēju, 
    tādas, dzīvs, strādā, tām, vēlas, nakts, īpaši, jūtos, nolādēts, meitenes, pusi, mammu, mees, 
    aizveries, vispār, dzīvību, kurā, kādā, vārdā, mašīna, būsim, vispirms, vinji, nevienu, šos, 
    tiksimies, džeik, vinjsh, vaina, turpini, kādi, jaunu, tuvu, atradu, vēlu, varēja, citādi, šim, 
    satikt, neuztraucies, pārliecināts, liec, diez, liela, doktor, nevaram, palīdzi, uzmanīgi, dažas, 
    šiem, atgriezies, gribēja, priecājos, parasti, valsts, asinis, tēti, you, mierā, piemēram, 
    jautājums, atā, bijām, zemē, pasauli, spēlē, blakus, izskaties, pirmā, nomira, paši, šobrīd, 
    daru, gaida, tādi, iešu, labākais, jauks, maz, pieder, jauns, nezināju, uzmanību, skaista, 
    prātā, brālis, patiesību, mierīgi, šai, dr, patiesi, jēzus, mārtij, zināju, suns, juus, sievu, 
    dzirdi, tepat, mamm, tēvu, tēva, frodo, sasodīts, desmit, stundas, tavi, mazā, džon, cita, 
    vajadzīga, forši, minūtes, mīlestība, nebiju, saprast, izbeidz, šoreiz, labā, dāmas, kurienes, 
    problēma, šādi, spēj, gadījumā, tiesa, kuģi, pēdējā, tici, esiet, atceros, katrs, nee, palīgā, 
    mister, liek, likās, domāt, vīri, pēdējo, traks, reizes, vienīgā, tiesības, skolā, turies, beigas, 
    karš, pīter, uguni, pietiks, vienam, vienā, pakaļ, jauna, zemi, puisis, ziniet, negribi, labrīt, 
    ap, cilvēka, draugu, atver, nezini, sāra, vēlaties, gadi, dažreiz, rokās, dabūt, nomierinies, 
    istabā, agrāk, ieroci, savām, meiteni, paņem, meklē, pār, seju, ziņu, dzirdējis, zinām, gatavi, 
    braukt, sāka, sāk, dievam, neesat, dzirdēt, spēle, bērniem, izdarīja, muļķības, doma, pēdējais, 
    dīvaini, atdod, ziņas, bankas, darāt, vakar, ceļš, neviena, brāli, otrā, atgriezties, galvas, 
    pietiekami, gulēt, uzreiz, iespēju, bijusi, karalis, bobij, šrek, tikpat, palīdziet, durvīm, 
    vecāki, atrodas, smieklīgi, kuģa, bail, godīgi, pēkšņi, nedēļas, māsa, skrien, ceļa, džeims, gars, 
    lielu, mašīnā, bojā, kurieni, ļaudis, dārgais, vecs, ūdeni, kūper, eju, mašīnas, ideja, kājas, 
    spēles, galvenais, citiem, jātiek, skaisti, nāvi, vinju, problēmas, vērts, drīkstu, domājat, visur, 
    bieži, manai, citas, apsolu, zelta, strādāju, dzimšanas, jūtu, naktī, dārgā, atbildi, noticēt, 
    klājas, izdevās, dok, redzat, gana, divus, ģimene, runa, stāsts, braucam, brīnišķīgi, ģimenes, 
    kuģis, čārlij, hey, kä, sheit, ved, atrada, mirusi, meita, paklau, nevēlos, bērnus, boss, kaptein, 
    nekāda, roze, nespēj, vīrietis, brīdi, īsts, dzīvē, tādā, manī, jūras, jaunkundz, iemesls, sakot, 
    manam, daudzi, varēsi, pateicos, jaunais, policija, pilnībā, nekur, jauka, nedari, kurus, zināms, 
    jautājumu, seko, re, padomā, pusē, visām, mīļais, dolāru, gadžet, katram, izdarīji, šīm, vienīgi, 
    mirt, apmēram, spēku, jauno, mr, celies, iepriekš, prātu, vēlētos, četri, lietām, redzēji, nevajadzētu, 
    donna, jaa, ticu, minūtēm, sievieti, nāve, jūties, nezina, parādi, malā, redz, uh, gredzenu, uzmanies, 
    kara, drošībā, sapnis, bijāt, grāmatu, slepkava, vinja, paga, pieci, pilsētā, drošs, pateikšu, gāja, 
    spēli, beigās, hanna, princese, jebkad, dakter, veids, palīdzība, stāstu, izmantot, spēlēt, gaisā, 
    darīšu, došos, dodas, kreisi, negribēju, mazāk, pastāsti, tak, devās, sirdi, misis, vis, patiesība, 
    veidā, harijs, cenšos, tuvāk, kurp, klausieties, sāp, ļaujiet, neticami, kungu, sīkais, iedomāties, 
    daļu, mazs, iedod, mazo, meklēju, parunāt, jādodas, sevis, pārējie, veicas, otra, mīlestību, zēns, 
    dodies, galam, sem, bīstami, zvēru, iespējas, maza, ellē, virs, nekādas, maniem, skatieties, šonakt, 
    svēto, kapteinis, iepazīties, pazīstu, turp, gredzens, nepareizi, lieliska, īstais, pagaidām, kājām, 
    mirklīti, pašlaik, d, poter, saprati, aprunāties, paša, šejieni, interesanti, nevarētu, pašā, paskat, 
    bailes, skolas, vārdus, aizmirsti, gaismas, kāp, zēni, darīsim, pašam, beidzies, sauca, māti, akmens, 
    grāmatas, diemžēl, tevī, kļūt, endij, patika, nabaga, tuvojas, tēvoci, dienām, plāns
"""

# Source: https://www.101languages.net/polish/most-common-polish-words/
alias words_pl = """
nie, to, się, w, na, i, z, co, jest, że, do, tak, jak, o, mnie, a, ale, mi, za, ja, ci, tu, ty, czy, 
tym, go, tego, tylko, jestem, po, cię, ma, już, mam, jesteś, może, pan, dla, coś, dobrze, wiem, jeśli, 
teraz, proszę, od, wszystko, tam, więc, masz, nic, on, być, gdzie, będzie, są, ten, mogę, ciebie, 
bardzo, sobie, kiedy, ze, wiesz, no, jej, jeszcze, pani, był, mój, chcę, było, dlaczego, by, przez, 
nas, tutaj, chcesz, jego, ją, ich, nigdy, żeby, też, kto, naprawdę, przepraszam, bo, mamy, porządku, 
możesz, dobra, mu, dziękuję, ona, domu, panie, muszę, nawet, chyba, hej, właśnie, prawda, zrobić, te, 
zawsze, będę, moja, gdy, je, trochę, nam, moje, cześć, bez, nim, była, tej, jesteśmy, dalej, pana, 
dzięki, wszyscy, musisz, twój, lat, tobą, więcej, ktoś, czas, ta, który, chce, powiedzieć, chodź, dobry, 
mną, niech, sam, razem, chodzi, czego, boże, stało, musimy, raz, albo, prostu, będziesz, dzień, możemy, 
was, myślę, czym, daj, lepiej, czemu, ludzie, ok, przed, życie, ludzi, robisz, my, niż, tych, kim, rzeczy, 
myślisz, powiedz, przy, twoja, oni, oczywiście, nikt, siebie, stąd, niego, twoje, miał, jeden, mówi, 
powiedział, moim, czasu, u, dziś, im, które, musi, wtedy, taki, aby, pod, dwa, temu, pewnie, takie, cóż, 
wszystkie, mojego, dużo, cholera, kurwa, wie, znaczy, wygląda, dzieje, mieć, ile, iść, potem, będziemy, 
dzieci, dlatego, cały, byłem, moją, skąd, szybko, jako, kochanie, stary, trzeba, miejsce, myśli, można, 
sie, jasne, mojej, wam, swoje, zaraz, wiele, nią, rozumiem, nich, wszystkich, jakieś, jakiś, kocham, idź, 
tę, mają, mówię, mówisz, dzisiaj, nad, pomóc, takiego, przestań, tobie, jutro, robić, jaki, mamo, kilka, 
przykro, wiedzieć, ojciec, widzisz, zbyt, zobaczyć, która, ani, tyle, trzy, tą, sposób, miałem, tato, niej, 
później, pieniądze, robi, kogoś, kiedyś, zanim, widzę, pracy, świetnie, pewno, myślałem, będą, bardziej, 
życia, długo, och, sir, ponieważ, aż, dni, nocy, każdy, dnia, znowu, oh, chciałem, taka, swoją, twoim, 
widziałem, stanie, powiem, imię, wy, żebyś, nadzieję, twojej, panu, spokój, słuchaj, rację, spójrz, razie, 
znam, pierwszy, koniec, chciałbym, we, nami, jakie, posłuchaj, problem, przecież, dobre, nasz, dziecko, drzwi, 
nasze, miło, czuję, mógł, żyje, jeżeli, człowiek, powiedziałem, gdyby, roku, dom, sama, potrzebuję, 
wszystkim, zostać, wciąż, dokładnie, mama, którzy, mówić, zamknij, mów, twoją, chwilę, zrobił, samo, idziemy, 
nadal, jesteście, zabić, były, sobą, kogo, lub, lubię, the, podoba, minut, bym, chciał, bądź, czegoś, gdzieś, 
mówiłem, chodźmy, znaleźć, poza, spokojnie, wcześniej, został, rozumiesz, mogą, prawie, wydaje, miała, mały, 
byłeś, facet, zrobię, macie, żadnych, razy, noc, ciągle, broń, moich, twojego, końcu, pomocy, czekaj, znasz, 
oczy, weź, idę, halo, dość, innego, pomysł, jakby, trzymaj, jedno, ojca, porozmawiać, pamiętasz, lata, 
powinieneś, którą, powodu, takim, niczego, powinniśmy, oto, napisy, jednak, świat, pokoju, żebym, sprawy, 
dwie, samochód, swój, wystarczy, pewien, źle, pozwól, numer, jedną, miejscu, you, drogi, byłam, dokąd, miłość, 
panowie, pieniędzy, którego, matka, rano, dwóch, całe, patrz, rzecz, nowy, idzie, wyglądasz, bóg, byś, życiu, 
nimi, nikogo, całą, swojego, świecie, sprawa, dziewczyna, prawo, byli, zostaw, wiedziałem, jedna, widzieć, 
swoim, kobiety, uważaj, najpierw, właściwie, dam, również, diabła, chcą, którym, zrób, da, jednego, dać, 
musiał, ręce, powinienem, których, znów, powiedziała, wczoraj, czujesz, zaczekaj, sądzę, śmierć, mówił, 
podczas, której, całkiem, pracę, żona, pójdę, pamiętam, powiedziałeś, mówią, wiemy, jezu, witam, cholery, 
swoich, telefon, wielu, także, poważnie, skoro, miejsca, robię, śmierci, słyszałem, wina, zrobiłem, dobranoc, 
parę, prawdę, swojej, serce, inaczej, dziewczyny, kobieta, powiesz, martw, rób, pytanie, pięć, innych, one, 
gra, natychmiast, wrócić, szybciej, jednym, cokolwiek, wierzę, wcale, wieczór, ważne, człowieka, wielki, nowa, 
dopiero, ziemi, gdybym, tata, poznać, stać, jack, myślałam, witaj, słowa, zrobiłeś, gówno, john, dolarów, 
sprawę, inne, idziesz, miałam, wiecie, chciałam, zobaczenia, widziałeś, żyć, każdym, nasza, panią, wspaniale, 
chwili, każdego, nowego, nieźle, takich, między, dostać, powinien, dawaj, dopóki, naszych, naszej, świata, 
chłopaki, chcemy, poczekaj, jaką, człowieku, czasem, żadnego, inny, przynajmniej, nazywa, super, naszego, 
szczęście, potrzebuje, godziny, zabrać, powrotem, syn, lecz, słucham, twoich, udało, boga, pokój, działa, 
ogóle, naszym, szkoły, możliwe, wiedział, wyjść, wszystkiego, byłoby, daleko, wieczorem, skarbie, jaka, 
mógłbym, ostatni, możecie, cztery, doktorze, zrobimy, mąż, przeciwko, zgadza, zrobisz, czasie, czasami, 
brzmi, raczej, ciało, należy, miasta, miałeś, taką, brat, cieszę, rozmawiać, cała, czymś, wybacz, twarz, 
mała, chcecie, dr, pojęcia, lubisz, głowę, najbardziej, dziwne, głowy, wody, pół, wiadomość, policja, 
strony, l, pl, mogłem, mieli, widzenia, pewna, ruszaj, wracaj, ode, popatrz, końca, plan, kiedykolwiek, 
wejść, została, rok, syna, uda, wrócę, zewnątrz, droga, uwierzyć, późno, zostało, zostanie, zły, kapitanie, 
potrzebujemy, byliśmy, zobaczymy, gotowy, obchodzi, jechać, rodziny, widziałam, drodze, czeka, środku, film, 
spać, człowiekiem, zupełnie, taa, pomóż, mieliśmy, pomoc, słowo, innym, ostatnio, and, zna, mogła, pójść, 
chłopcy, wziąć, mógłbyś, tłumaczenie, potrzebujesz, słyszysz, blisko, godzin, miłości, góry, zabił, piękna, 
napisów, pokaż, moi, lubi, robota, prawa, ciężko, kimś, dół, rękę, nazywam, wielkie, część, wkrótce, naszą, 
jedziemy, zapomnij, prosto, radę, robimy, powinnaś, gdybyś, chociaż, zależy, stronie, wypadek, tydzień, byłaś, 
nowe, małe, praca, drogę, chłopak, zrobi, widział, mieście, synu, oznacza, krew, mógłby, krwi, górę, joe, wasza, 
robią, tędy, wszędzie, temat, pierwsze, zobacz, ponad, kraju, mało, racja, tymi, cicho, chciała, powiedziałam, 
leci, powinno, mówiąc, serca, chciałabym, miasto, george, spotkać, mniej, e, przyjaciel, mówiłeś, kłopoty, 
miesięcy, jakąś, żaden, zostań, roboty, zatrzymać, frank, nieważne, głupi, pa, koleś, sprawie, spotkanie, ojcze, 
pewnego, spróbuj, drugi, znalazłem, pracować, całym, zostały, złe, niemożliwe, jakoś, zdjęcia, stronę, wiedzą, it, 
dziewczynę, zaczyna, mogli, samego, sądzisz, rodzina, razu, trudno, samochodu, okay, boję, szkoda, wami, charlie, 
dał, środka, ojcem, piękne, dawno, choć, panem, przykład, nagle, bracie, żadnej, drugiej, przyjaciół, otwórz, 
myśleć, doktor, chwileczkę, pracuje, najlepszy, brata, czyż, często, http, powinnam, odejść, trzech, chodźcie, 
nazwisko, szansę, ciała, policji, szkole, prawdopodobnie, serio, matki, org, wolno, sami, muszą, zabierz, 
słyszałeś, siostra, uspokój, wystarczająco, początku, faceta, problemy, szefie, broni, me, zostawić, czuje, 
będziecie, przyszedł, wiedziałam, kilku, inni, b, głowie, historia, według, www, wezmę, nowym, czekać, stój, 
mężczyzna, mówiłam, pokazać, około, wracam, wieku, jakaś, pierwsza, niczym, zabiję, zdjęcie, zabawne, rodzice, 
musiałem, całkowicie, sprawdzić, mike, przyjdzie, sześć, kupić, dobrym, żonę, dasz, pomoże, nogi, obok, ruszać, 
trzymać, zadzwonić, panno, godzinę, boli, oraz, spokoju, walczyć, wróci, tom, wspólnego, zmienić, ostatnie, uwagę, 
znać, jednej, dłużej, powie, pogadać, łatwo, większość, nikomu, michael, córka, niedługo, powodzenia, tygodniu, 
włosy, niestety, górze, kochasz, prawdziwy, historii, ulicy, musicie, gotowi, chwila, samym, grać, zadzwonię, 
strasznie, mieszka, kocha, rady, tyłu, jakim, obiecuję, tysięcy, pomyślałem, pracuję, jedynie, pozwolić, uwaga, 
proste, zacząć, myśl, wstawaj, rany, prawdziwe, takiej, jakiegoś, umrzeć, złego, okazji
"""

# Source: https://www.101languages.net/greek/most-common-greek-words/
alias words_el = """
    να, το, δεν, θα, είναι, και, μου, με, ο, για, την, σου, τα, τον, η, τι, σε, που, του, αυτό, στο, ότι, 
    από, τη, της, ναι, σας, ένα, εδώ, τους, αν, όχι, μια, μας, είσαι, αλλά, κι, οι, πρέπει, είμαι, ήταν, 
    πολύ, στην, δε, γιατί, εγώ, τώρα, πως, εντάξει, τις, κάτι, ξέρω, μην, έχει, έχω, εσύ, θέλω, καλά, 
    έτσι, στη, στον, αυτή, ξέρεις, κάνεις, εκεί, σαν, μόνο, μπορώ, όταν, έχεις, μαζί, πώς, τίποτα, 
    ευχαριστώ, όλα, κάνω, πάμε, ή, ποτέ, τόσο, πού, αυτά, έλα, στα, μέσα, κάνει, των, μπορεί, κύριε, πιο, 
    σπίτι, παρακαλώ, λοιπόν, μπορείς, αυτός, υπάρχει, ακόμα, πίσω, λίγο, πάντα, είμαστε, γεια, τότε, 
    ειναι, μετά, πω, έχουμε, μη, ένας, ποιος, νομίζω, πριν, απλά, δω, δουλειά, παιδιά, οχι, αλήθεια, 
    όλοι, ίσως, λες, όπως, ας, θέλεις, μα, άλλο, είπε, ζωή, πάω, δύο, ωραία, έναν, καλό, απο, κάνουμε, 
    έξω, κοίτα, είχε, στις, πάνω, είπα, πες, χρόνια, ούτε, κάτω, είστε, ώρα, θες, σένα, έχουν, γυναίκα, 
    μένα, μέρα, καλή, φορά, όμως, κανείς, κάθε, ε, οτι, αρέσει, ήμουν, μέχρι, δυο, είχα, μαμά, χωρίς, 
    καλύτερα, πας, πράγματα, πάει, σήμερα, κάποιος, ήθελα, θέλει, θεέ, έπρεπε, λέει, μία, σωστά, αυτόν, 
    μπορούμε, συμβαίνει, ακριβώς, έγινε, πόσο, επειδή, λεφτά, πολλά, μόλις, εμένα, λένε, πεις, συγγνώμη, 
    γρήγορα, ω, έκανε, λυπάμαι, γίνει, παιδί, περίμενε, έκανα, φίλε, βλέπω, μέρος, στιγμή, φαίνεται, 
    πρόβλημα, άλλη, είπες, φυσικά, κάποιον, όσο, πήγαινε, πάλι, λάθος, ως, έχετε, εσένα, πράγμα, κυρία,
    χρόνο, στους, πάρω, μπαμπά, δικό, απ, γίνεται, εσείς, λέω, συγνώμη, όλο, μητέρα, έκανες, πιστεύω, 
    ήσουν, κάποια, σίγουρα, υπάρχουν, όλη, ενα, αυτο, ξέρει, μωρό, ιδέα, δει, μάλλον, ίδιο, πάρε, είδα, 
    αύριο, βλέπεις, νέα, κόσμο, νομίζεις, τί, εμείς, σταμάτα, πάρει, αγάπη, πατέρας, όλους, αρκετά, 
    χρειάζεται, καιρό, φορές, κάνουν, ακόμη, α, πατέρα, προς, αμέσως, πια, ηταν, χαρά, απόψε, όνομα, 
    μάλιστα, μόνος, μεγάλη, κανένα, ελα, πραγματικά, αυτοί, πει, πότε, εχω, βράδυ, αυτές, θέλετε, κάνετε, 
    σημαίνει, πρώτη, ποιο, πόλη, μπορούσα, ποια, γαμώτο, ήδη, τελευταία, άνθρωποι, τέλος, απλώς, νόμιζα, 
    ξέρετε, μέρες, δεις, θέση, αυτούς, καταλαβαίνω, φύγε, χέρια, εκτός, ήξερα, οπότε, λεπτά, μακριά, 
    κάνε, αμάξι, δική, λεπτό, μεγάλο, μήπως, κορίτσι, μάτια, ελάτε, πρόκειται, πόρτα, δίκιο, βοήθεια, 
    ήρθε, μιλήσω, δρόμο, εαυτό, καθόλου, ορίστε, βρω, πειράζει, μπορείτε, καλός, πέρα, κοντά, εννοώ, 
    τέτοιο, μπροστά, έρθει, χρειάζομαι, χέρι, ελπίζω, δώσε, διάολο, φύγω, ιστορία, όπλο, αφού, πρωί, 
    νύχτα, ωραίο, τύπος, ξανά, θυμάσαι, δούμε, κατά, εννοείς, αγαπώ, κακό, θέμα, εδω, αυτήν, τρόπο, 
    κεφάλι, είχες, μερικές, μιλάς, φίλος, άνθρωπος, φύγουμε, όλες, σκατά, ανθρώπους, βέβαια, άντρας, 
    κάποιο, πάνε, αστυνομία, αλλιώς, συνέβη, χαίρομαι, άλλα, περισσότερο, καλύτερο, εκείνη, πάρεις, τo, 
    νερό, ώρες, σίγουρος, vα, τρεις, εχεις, πρώτα, μπορούσε, σ, οταν, δρ, πιστεύεις, μόνη, ποιός, καμιά, 
    κανέναν, πέθανε, εχει, ετσι, αγόρι, ανησυχείς, άντρες, δωμάτιο, ομάδα, ίδια, εμπρός, βρούμε, βοηθήσω, 
    τέτοια, πήρε, τρία, λόγο, μικρό, αντίο, o, πέντε, πήγε, καν, ευκαιρία, είδες, έρχεται, δηλαδή, 
    αργότερα, ήθελε, πούμε, λέμε, όπου, αλλα, κόρη, κόσμος, γυναίκες, τηλέφωνο, εάν, δώσω, καρδιά, βρήκα, 
    γραφείο, επίσης, νιώθω, σχέση, θέλουν, ισως, τέλεια, είχαμε, κάπου, μυαλό, ώστε, καλημέρα, σχολείο, 
    θεός, μικρή, τρέχει, ψέματα, ξέρουμε, οικογένεια, εισαι, θυμάμαι, κ, ενός, φίλοι, πρόσεχε, 
    καταλαβαίνεις, αργά, ντε, θέλουμε, σύντομα, πήρα, σχεδόν, παιχνίδι, κύριοι, γειά, μήνες, μπαμπάς, 
    σοβαρά, δολάρια, τουλάχιστον, χρήματα, πείτε, πόδια, αίμα, κοπέλα, φαγητό, ειμαι, ποιον, μερικά, 
    δύσκολο, μπορούν, βρεις, όμορφη, φύγεις, τύχη, πλάκα, έρθεις, άντρα, κορίτσια, μείνε, αστείο, καμία, 
    είχαν, χάρη, άλλος, πρεπει, σημασία, φυλακή, νεκρός, συγχωρείτε, φοβάμαι, μπράβο, γύρω, κανένας, μεταξύ, 
    τ, χθες, πολλές, όνομά, τζακ, ρε, καληνύχτα, πολυ, φύγει, αφήσω, ήθελες, tι, ήρθες, ακούς, πρώτο, γιατι, 
    ηρέμησε, γι, πάρουμε, πάρα, άλλους, κατάλαβα, έρθω, συνέχεια, έλεγα, γλυκιά, νοιάζει, χριστέ, βιβλίο, 
    κύριος, μ, χώρα, αρχή, ήρθα, πεθάνει, γη, έτοιμος, εγω, άσχημα, συμβεί, αυτοκίνητο, ζωής, τελικά, φέρω, 
    τρόπος, κατάσταση, www, περιμένω, σημαντικό, όσα, σκέφτηκα, μιλήσουμε, αφήστε, τωρα, ακούω, γιος, σκοτώσω, 
    δύναμη, κα, κε, εκείνο, γονείς, μιλάω, σκοτώσει, ολα, μείνει, μείνω, αρέσουν, δεv, υπόθεση, φίλους, όπλα, 
    υποθέτω, εμάς, ενώ, έξι, σχέδιο, άρεσε, καφέ, σκότωσε, χρειαζόμαστε, φίλο, σωστό, προσπαθώ, κάναμε, 
    κοιτάξτε, μoυ, κου, ποτό, εσάς, έι, έφυγε, ταινία, μοιάζει, κρεβάτι, εχουμε, περιμένει, νέο, μπορούσες, 
    μάθω, αφήσεις, περιμένετε, χρειάζεσαι, υπήρχε, μισό, δέκα, αφεντικό, περίπου, άλλοι, λόγος, ξέρουν, κάποτε, 
    βρήκες, καλύτερη, υπέροχο, τζον, δίπλα, σκάσε, θεού, άκουσα, φύγετε, λέξη, παρά, επόμενη, λέτε, περάσει, 
    πόσα, γίνεις, σώμα, ν, πήρες, τελείωσε, γιο, ρούχα, σκέφτομαι, εσυ, άλλες, γυρίσω, βάλω, μουσική, ραντεβού, 
    φωτιά, έδωσε, πάτε, φοβάσαι, βρει, δείξω, γίνω, βοηθήσει, τύπο, σειρά, αξίζει, μείνεις, είπαν, άλλον, 
    κυρίες, λίγη, πέρασε, κάτσε, πήγα, δείτε, μιας, βδομάδα, έρχομαι, προσοχή, εύκολο, ερώτηση, υπέροχα, 
    σίγουρη, νοσοκομείο, τρελός, ενας, βάλε, πόλεμο, φέρε, δικά, τιμή, κατάλαβες, ταξίδι, οποίο, δουλεύει, θεό, 
    μικρέ, μάθεις, βρίσκεται, πολλοί, δες, πάρτε, παντού, πρόσωπο, μήνυμα, αδερφή, μιλάει, παλιά, πουθενά, 
    κράτα, περίπτωση, φως, επάνω, έλεγε, συμφωνία, οπως, ολοι, πρώτος, δεσποινίς, γιατρός, γνωρίζω, σαμ, 
    σκέφτεσαι, ει, φίλη, σεξ, έκαναν, προβλήματα, κάπως, ό, τελευταίο, ακούσει, τζο, καλώς, επιλογή, 
    σταματήστε, τόσα, οτιδήποτε, περισσότερα, άδεια, πάρτι, περίμενα, ακούγεται, gmteam, ήξερες, καιρός, 
    μαλλιά, καλύτερος, κανεις, φρανκ, μέση, συνέχισε, τίποτε, φωτογραφία, κατι, μεγάλος, περιοχή, άσε, καθώς, 
    είδε, λόγια, μήνα, μαλακίες, όμορφο, δώρο, στόμα, χάλια, εντελώς, μακάρι, τελειώσει, γνώμη, γιατρέ, ξερω, 
    πλευρά, μέλλον, θάνατο, νιώθεις, έτοιμοι, κομμάτι, μάθει, μιλάμε, ψηλά, αέρα, ερωτήσεις, αυτού, δώσει, 
    φεύγω, σημείο, τηλεόραση, κυριε, πραγματικότητα, ανάγκη, βοηθήσεις, προσπάθησε, γύρνα, άφησε, λίγα, κάντε, 
    είvαι, βλέπετε, αυτη, δείπνο, επιτέλους, κέντρο, περίεργο, ακούστε, πλοίο, κάποιες, δικός, σoυ, οικογένειά, 
    μιλήσει, πλέον, υπόσχομαι, περιμένεις, ήξερε, σκοτώσεις, ενταξει, δώσεις, εκει, ήμασταν, έρχονται, κώλο, 
    ρωτήσω, παίρνει, σιγά, σήκω, στοιχεία, αδελφή, βασικά, μένει, άκρη, πηγαίνετε, παίρνεις, tο, περιμένουμε, 
    συγχωρείς, μικρός, πόδι, δίνει, εκατομμύρια, ξενοδοχείο, αποστολή, ενδιαφέρον, χάρηκα, αεροπλάνο, γάμο, 
    χιλιάδες, υόρκη, οκ, ευχαριστούμε, καλα, κοιτάς, σα, π, χρόνος, ησυχία, ασφάλεια, εκείνος, a, βρήκε, 
    τέσσερα, βγάλω, μπες, συχνά, ημέρα, μάνα, εν, αγαπάς, άνθρωπο, γραμμή, φωτογραφίες, προσέχεις, ύπνο, 
    μυστικό, σχετικά, είδους, σκέψου, χριστούγεννα, κόσμου, τομ, μισώ, σύστημα, δουλειές, τελείως, πεθάνω, 
    αλλάξει, δεξιά, συνήθως, δουλεύω, μάικλ, εβδομάδα, νούμερο, λείπει, έτοιμη, τμήμα, βγει, ψυχή, έπεσε, 
    κάθαρμα, ματιά, οποία, πληροφορίες, μονο, κρίμα, τραγούδι, μαγαζί, δουλεύεις, μαζι, τέλειο, κύριο, 
    λέγεται, τσάρλι, πεθάνεις, σκεφτόμουν, καλησπέρα, συγχαρητήρια, φωνή, εκ, άτομο, παίζεις, σκάφος, 
    φαίνεσαι, ξαφνικά, παραπάνω, ατύχημα, θελω, ξέχνα, ήρθατε, εναντίον, τραπέζι, γράμμα, μείνετε, αμερική, 
    βασιλιάς, υπό, μπάνιο, ποτε, ίδιος, προφανώς, μαλάκα, αδερφός, άνδρες, nαι, χρονών, ναί, κλειδί, δις, 
    γιαγιά, παράξενο, πτώμα, βρήκαμε, μιλήσεις, υποτίθεται, ορκίζομαι, δυνατά, ποιό, θάλασσα, παίρνω, άκουσες, 
    παρέα, αριστερά, έμαθα, μάχη, μηχανή, σάρα, ζωντανός, όνειρο, παλιό, μπορούσαμε, πάντως, ανάμεσα, έχασα, 
    νωρίς, κάποιοι, άκου, παίζει, φτάνει, δίνω, βγες, υπέροχη, νόημα, έλεγχο, μέτρα, ξερεις, ζει, δείχνει, 
    βρες, τού
"""

# Source: https://www.101languages.net/russian/most-common-russian-words/
alias words_ru = """
я, не, что, в, и, ты, это, на, с, он, вы, как, мы, да, а, мне, меня, у, нет, так, но, то, все, тебя, его, 
за, о, она, тебе, если, они, бы, же, ну, здесь, к, из, есть, чтобы, для, хорошо, когда, вас, только, по, 
вот, просто, был, знаю, нас, всё, было, от, может, кто, вам, очень, их, там, будет, уже, почему, еще, 
быть, где, спасибо, ничего, сейчас, или, могу, хочу, нам, чем, мой, до, надо, этого, ее, теперь, давай, 
знаешь, нужно, больше, этом, нибудь, раз, со, была, этот, ему, ладно, эй, время, тоже, даже, хочешь, 
сказал, ли, себя, думаю, пока, должен, потому, никогда, ни, тут, ещё, её, пожалуйста, сюда, привет, 
тогда, конечно, моя, него, сегодня, один, тобой, правда, лучше, об, были, того, можно, мной, всегда, 
сказать, день, сэр, без, можешь, чего, эти, дело, значит, лет, много, во, делать, буду, порядке, должны, 
такой, ведь, ним, всего, сделать, хотел, твой, жизнь, ей, мистер, потом, через, себе, них, всех, такое, 
им, куда, том, мама, после, человек, люди, слишком, иди, зачем, этим, немного, сколько, этой, знаете, 
боже, ней, эту, который, отец, свою, деньги, два, под, твоя, мои, никто, моей, думаешь, друг, жизни, 
эта, назад, видел, кажется, точно, вместе, люблю, мог, случилось, сам, нравится, черт, какой, людей, 
папа, домой, тот, скажи, которые, должна, три, всем, сделал, возможно, прошу, будем, дома, парень, 
снова, говорит, место, отсюда, можем, будешь, пошли, делаешь, совсем, говорил, понимаю, завтра, хочет, 
простите, разве, давайте, хотите, отлично, сказала, туда, прямо, времени, вами, лишь, своей, хватит, 
думал, можете, дом, дела, знать, дай, понял, помочь, говорить, слушай, свои, поэтому, прости, знает, 
именно, знал, тем, кого, смотри, каждый, ваш, похоже, найти, моего, наш, мать, одна, имя, про, говорю, 
будут, оно, свой, нельзя, извините, стоит, действительно, зовут, поговорить, доктор, перед, несколько, 
нужен, происходит, ко, господи, возьми, мою, тех, нами, вижу, должно, наверное, откуда, понимаешь, верно, 
скоро, уж, деле, твои, пусть, всю, хотела, при, более, ребята, нее, быстро, подожди, идти, надеюсь, чём, 
работу, видеть, такая, этих, уверен, нужна, года, раньше, такие, руки, видишь, какая, посмотри, сын, 
самом, ваша, послушай, равно, наши, другой, ага, мир, извини, минут, против, твоей, пор, жить, ж, жаль, 
вообще, могли, хотя, человека, пора, ради, говорят, почти, твою, могут, над, весь, первый, чёрт, слышал, 
собой, брат, вещи, дня, скажу, говоришь, нормально, своего, мое, ваше, итак, будь, ночь, хоть, ясно, 
плохо, дверь, вопрос, господин, давно, денег, ваши, ка, мисс, одну, глаза, пять, будто, между, пойду, 
опять, работа, самое, иногда, детей, этому, рад, здорово, бог, одного, ночи, готов, номер, которая, 
машину, любовь, дорогая, виду, одно, прекрасно, вон, своих, быстрее, отца, женщина, достаточно, рядом, 
убить, таким, пойдем, смерти, дети, такого, правильно, месте, никаких, сказали, здравствуйте, пару, две, 
видела, долго, хороший, ах, кроме, алло, нашей, прав, вчера, вечером, жена, миссис, чтоб, друга, нужны, 
кем, какие, те, увидеть, утро, смогу, неё, сама, моему, большой, сразу, работать, сердце, стал, своим, 
сначала, могла, вроде, ними, говори, голову, дальше, помнишь, либо, ума, одной, вечер, случае, взять, 
проблемы, помощь, добрый, год, думала, делает, скорее, слова, капитан, последний, важно, дней, помню, 
ночью, утром, моих, произошло, которую, боюсь, также, вашей, ой, стой, твоего, никого, дорогой, убил, 
насчет, друзья, самый, проблема, видели, вперед, дерьмо, понятно, чувствую, наша, будете, тому, имею, 
вернуться, придется, пришел, спать, стать, столько, говорила, пойти, иначе, работает, девушка, час, 
момент, моим, умер, думаете, доброе, слово, новый, часов, мире, знаем, твое, мальчик, однажды, интересно, 
конец, играть, a, заткнись, сделали, посмотреть, идет, узнать, свое, права, хорошая, город, джон, 
долларов, парни, идем, говорите, уйти, понять, знала, поздно, нашли, работы, скажите, сделаю, увидимся, 
какого, другие, идея, пошел, доме, дочь, имеет, приятно, лицо, наших, обо, понимаете, руку, часть, 
смотрите, вся, собираюсь, четыре, прежде, хотят, скажешь, чувак, дайте, сделала, кофе, джек, верю, 
ждать, затем, большое, сами, неужели, моё, любит, мужчина, дать, господа, таких, осталось, которой, 
далеко, вернусь, сильно, ох, сможешь, кому, вашего, посмотрим, машина, подождите, свет, чуть, серьезно, 
пришли, оружие, решил, смысле, видите, тихо, нашел, свидания, путь, той, совершенно, следующий, которого, 
места, парня, вдруг, пути, мадам, какое, шанс, сестра, нашего, ужасно, минуту, вокруг, другом, иду, 
других, хотели, нем, смерть, подумал, фильм, оставь, делаете, уверена, кровь, говорили, внимание, 
помогите, идите, держи, получить, оба, взял, спокойно, обычно, мало, забыл, странно, смотреть, поехали, 
дал, часа, прекрати, посмотрите, готовы, вернулся, поверить, позже, милая, женщины, любишь, довольно, 
обратно, остаться, думать, та, стороны, полиция, тело, тысяч, делал, машины, угодно, муж, году, неплохо, 
бога, некоторые, конце, милый, the, рождения, трудно, добро, любви, больно, невозможно, спокойной, 
слышишь, типа, получил, которое, приятель, хуже, никому, честь, успокойся, вашу, маленький, выглядит, 
чарли, сына, неделю, i, девочка, делаю, шесть, ноги, история, рассказать, послушайте, часто, кстати, 
двух, забудь, которых, следует, знают, пришла, семья, станет, матери, ребенок, план, проблем, например, 
сделай, воды, немедленно, мира, сэм, телефон, перестань, правду, второй, прощения, ту, наше, уходи, твоих, 
помоги, пол, внутри, нему, смог, десять, нашу, около, бывает, самого, большая, леди, сможем, вниз, легко, 
делай, единственный, рада, меньше, волнуйся, хотим, полагаю, мам, иметь, своими, мере, наконец, начала, 
минутку, работе, пожаловать, другого, двое, никакого, честно, школе, лучший, умереть, дам, насколько, 
всей, малыш, оставить, безопасности, ненавижу, школу, осторожно, сынок, джо, таки, пытался, другое, б, 
клянусь, машине, недели, стало, истории, пришлось, выглядишь, чему, сможет, купить, слышала, знали, 
настоящий, сих, выйти, людям, замечательно, полиции, огонь, пойдём, спросить, дядя, детка, среди, особенно, 
твоим, комнате, шоу, выпить, постоянно, делают, позвольте, родители, письмо, городе, случай, месяцев, мужик, 
благодарю, o, ребенка, смешно, ответ, города, образом, любой, полностью, увидел, еду, имени, вместо, 
абсолютно, обязательно, улице, твоё, убили, ваших, ехать, крови, решение, вина, поможет, своё, секунду, 
обещаю, начать, голос, вещь, друзей, показать, нечего, э, месяц, подарок, приехал, самая, молодец, сделаем, 
крайней, женщин, собираешься, конца, страшно, новости, идиот, потерял, спасти, вернуть, узнал, слушайте, 
хотелось, сон, поняла, прошло, комнату, семь, погоди, главное, рано, корабль, пытаюсь, игра, умерла, 
повезло, всему, возьму, таком, моем, глаз, настолько, идём, удачи, готова, семьи, садись, гарри, держись, 
звучит, мило, война, человеком, право, такую, вопросы, представить, работаю, имеешь, красивая, идёт, никакой, 
профессор, думает, войны, стала, стали, оттуда, известно, слышу, начал, подумать, позвонить, старый, придётся, 
историю, вести, твоему, последнее, хочется, миллионов, нашла, способ, отношения, земле, фрэнк, получится, 
говоря, связи, многие, пошёл, пистолет, убью, руках, получилось, президент, остановить, тьi, оставил, одним, 
you, утра, боль, хорошие, пришёл, открой, брось, вставай, находится, поговорим, кино, людьми, полицию, покажу, 
волосы, последние, брата, месяца
"""


fn gen_word_pairs[words: String = words_en]() -> List[String]:
    var result = List[String]()
    try:
        var list = words.split(",")
        for w in list:
            var w1 = w[].strip()
            for w in list:
                var w2 = w[].strip()
                result.append(w1 + " " + w2)
    except:
        pass
    return result


def dif_bits(i1: UInt64, i2: UInt64) -> Int:
    return int(pop_count(i1 ^ i2))


def assert_dif_hashes(hashes: List[UInt64], upper_bound: Int):
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            var diff = dif_bits(hashes[i], hashes[j])
            assert_true(
                diff > upper_bound,
                str("Index: {}:{}, diff between: {} and {} is: {}").format(
                    i, j, hashes[i], hashes[j], diff
                ),
            )


alias hasher0 = AHasher[SIMD[DType.uint64, 4](0, 0, 0, 0)]
alias hasher1 = AHasher[SIMD[DType.uint64, 4](1, 0, 0, 0)]


def test_hash_byte_array():
    assert_equal(hash[HasherType=hasher0]("a"), hash[HasherType=hasher0]("a"))
    assert_equal(hash[HasherType=hasher1]("a"), hash[HasherType=hasher1]("a"))
    assert_not_equal(
        hash[HasherType=hasher0]("a"), hash[HasherType=hasher1]("a")
    )
    assert_equal(hash[HasherType=hasher0]("b"), hash[HasherType=hasher0]("b"))
    assert_equal(hash[HasherType=hasher1]("b"), hash[HasherType=hasher1]("b"))
    assert_not_equal(
        hash[HasherType=hasher0]("b"), hash[HasherType=hasher1]("b")
    )
    assert_equal(hash[HasherType=hasher0]("c"), hash[HasherType=hasher0]("c"))
    assert_equal(hash[HasherType=hasher1]("c"), hash[HasherType=hasher1]("c"))
    assert_not_equal(
        hash[HasherType=hasher0]("c"), hash[HasherType=hasher1]("c")
    )
    assert_equal(hash[HasherType=hasher0]("d"), hash[HasherType=hasher0]("d"))
    assert_equal(hash[HasherType=hasher1]("d"), hash[HasherType=hasher1]("d"))
    assert_not_equal(
        hash[HasherType=hasher0]("d"),
        hash[HasherType=hasher1]("d"),
    )


def test_avalanche():
    # test that values which differ just in one bit,
    # produce significatly different hash values
    var data = UnsafePointer[UInt8].alloc(256)
    memset_zero(data, 256)
    var hashes0 = List[UInt64]()
    var hashes1 = List[UInt64]()
    hashes0.append(hash[HasherType=hasher0](data, 256))
    hashes1.append(hash[HasherType=hasher1](data, 256))

    for i in range(256):
        memset_zero(data, 256)
        var v = 1 << (i & 7)
        data[i >> 3] = v
        hashes0.append(hash[HasherType=hasher0](data, 256))
        hashes1.append(hash[HasherType=hasher1](data, 256))

    for i in range(len(hashes0)):
        var diff = dif_bits(hashes0[i], hashes1[i])
        assert_true(
            diff > 16,
            str("Index: {}, diff between: {} and {} is: {}").format(
                i, hashes0[i], hashes1[i], diff
            ),
        )

    assert_dif_hashes(hashes0, 14)
    assert_dif_hashes(hashes1, 15)


def test_trailing_zeros():
    # checks that a value with different amount of trailing zeros,
    # results in significantly different hash values
    var data = UnsafePointer[UInt8].alloc(8)
    memset_zero(data, 8)
    data[0] = 23
    var hashes0 = List[UInt64]()
    var hashes1 = List[UInt64]()
    for i in range(1, 9):
        hashes0.append(hash[HasherType=hasher0](data, i))
        hashes1.append(hash[HasherType=hasher1](data, i))

    for i in range(len(hashes0)):
        var diff = dif_bits(hashes0[i], hashes1[i])
        assert_true(
            diff > 18,
            str("Index: {}, diff between: {} and {} is: {}").format(
                i, hashes0[i], hashes1[i], diff
            ),
        )

    assert_dif_hashes(hashes0, 18)
    assert_dif_hashes(hashes1, 18)


def assert_fill_factor[
    label: String
](words: List[String], num_buckets: Int, lower_bound: Float64):
    # A perfect hash function is when the number of buckets is equal to number of words
    # and the fill factor results in 1.0
    var buckets = List[Int](0) * num_buckets
    for w in words:
        var h = hash[HasherType=hasher0](w[])
        buckets[int(h) % num_buckets] += 1
    var unfilled = 0
    for v in buckets:
        if v[] == 0:
            unfilled += 1

    var fill_factor = 1 - unfilled / num_buckets
    assert_true(
        fill_factor >= lower_bound,
        str("Fill factor for {} is {}, provided lower boound was {}").format(
            label, fill_factor, lower_bound
        ),
    )


def assert_fill_factor_old_hash[
    label: String
](words: List[String], num_buckets: Int, lower_bound: Float64):
    # A perfect hash function is when the number of buckets is equal to number of words
    # and the fill factor results in 1.0
    var buckets = List[Int](0) * num_buckets
    for w in words:
        var h = old_hash(w[].unsafe_ptr(), w[].byte_length())
        buckets[h % num_buckets] += 1
    var unfilled = 0
    for v in buckets:
        if v[] == 0:
            unfilled += 1

    var fill_factor = 1 - unfilled / num_buckets
    assert_true(
        fill_factor >= lower_bound,
        str("Fill factor for {} is {}, provided lower bound was {}").format(
            label, fill_factor, lower_bound
        ),
    )


def test_fill_factor():
    var words = List[String]()

    words = gen_word_pairs[words_ar]()
    assert_fill_factor["AR"](words, len(words), 0.63)
    assert_fill_factor["AR"](words, len(words) // 2, 0.86)
    assert_fill_factor["AR"](words, len(words) // 4, 0.98)
    assert_fill_factor["AR"](words, len(words) // 13, 1.0)

    assert_fill_factor_old_hash["AR"](words, len(words), 0.59)

    words = gen_word_pairs[words_el]()
    assert_fill_factor["EL"](words, len(words), 0.63)
    assert_fill_factor["EL"](words, len(words) // 2, 0.86)
    assert_fill_factor["EL"](words, len(words) // 4, 0.98)
    assert_fill_factor["EL"](words, len(words) // 13, 1.0)

    assert_fill_factor_old_hash["EL"](words, len(words), 0.015)

    words = gen_word_pairs[words_en]()
    assert_fill_factor["EN"](words, len(words), 0.63)
    assert_fill_factor["EN"](words, len(words) // 2, 0.85)
    assert_fill_factor["EN"](words, len(words) // 4, 0.98)
    assert_fill_factor["EN"](words, len(words) // 14, 1.0)

    assert_fill_factor_old_hash["EN"](words, len(words), 0.015)

    words = gen_word_pairs[words_he]()
    assert_fill_factor["HE"](words, len(words), 0.63)
    assert_fill_factor["HE"](words, len(words) // 2, 0.86)
    assert_fill_factor["HE"](words, len(words) // 4, 0.98)
    assert_fill_factor["HE"](words, len(words) // 14, 1.0)

    assert_fill_factor_old_hash["HE"](words, len(words), 0.2)

    words = gen_word_pairs[words_lv]()
    assert_fill_factor["LV"](words, len(words), 0.63)
    assert_fill_factor["LV"](words, len(words) // 2, 0.86)
    assert_fill_factor["LV"](words, len(words) // 4, 0.98)
    assert_fill_factor["LV"](words, len(words) // 12, 1.0)

    assert_fill_factor_old_hash["LV"](words, len(words), 0.015)

    words = gen_word_pairs[words_pl]()
    assert_fill_factor["PL"](words, len(words), 0.63)
    assert_fill_factor["PL"](words, len(words) // 2, 0.86)
    assert_fill_factor["PL"](words, len(words) // 4, 0.98)
    assert_fill_factor["PL"](words, len(words) // 13, 1.0)

    assert_fill_factor_old_hash["PL"](words, len(words), 0.015)

    words = gen_word_pairs[words_ru]()
    assert_fill_factor["RU"](words, len(words), 0.63)
    assert_fill_factor["RU"](words, len(words) // 2, 0.86)
    assert_fill_factor["RU"](words, len(words) // 4, 0.98)
    assert_fill_factor["RU"](words, len(words) // 13, 1.0)

    assert_fill_factor_old_hash["RU"](words, len(words), 0.015)


def test_hash_simd_values():
    fn hash(value: SIMD) -> UInt64:
        var hasher = AHasher[SIMD[DType.uint64, 4](0)]()
        hasher._update_with_simd(value)
        return hasher^.finish()

    assert_equal(hash(SIMD[DType.float16, 1](1.5)), 18058966248987367737)
    assert_equal(hash(SIMD[DType.float32, 1](1.5)), 13467270117196531127)
    assert_equal(hash(SIMD[DType.float64, 1](1.5)), 719560574162820089)
    assert_equal(hash(SIMD[DType.float16, 1](1)), 1206414632147291024)
    assert_equal(hash(SIMD[DType.float32, 1](1)), 9557262614467209093)
    assert_equal(hash(SIMD[DType.float64, 1](1)), 7961842588256067709)

    assert_equal(hash(SIMD[DType.int8, 1](1)), 4759877148789019546)
    assert_equal(hash(SIMD[DType.int16, 1](1)), 4759877148789019546)
    assert_equal(hash(SIMD[DType.int32, 1](1)), 4759877148789019546)
    assert_equal(hash(SIMD[DType.int64, 1](1)), 4759877148789019546)
    assert_equal(hash(SIMD[DType.bool, 1](True)), 4759877148789019546)

    assert_equal(hash(SIMD[DType.int8, 1](-1)), 7301741325190448010)
    assert_equal(hash(SIMD[DType.int16, 1](-1)), 7301741325190448010)
    assert_equal(hash(SIMD[DType.int32, 1](-1)), 7301741325190448010)
    assert_equal(hash(SIMD[DType.int64, 1](-1)), 7301741325190448010)

    assert_equal(hash(SIMD[DType.int8, 1](0)), 16659764227661506736)
    assert_equal(hash(SIMD[DType.int8, 2](0)), 1562284133626399299)
    assert_equal(hash(SIMD[DType.int8, 4](0)), 17902233708981521127)
    assert_equal(hash(SIMD[DType.int8, 8](0)), 632562262308536351)
    assert_equal(hash(SIMD[DType.int8, 16](0)), 7298276873920245913)
    assert_equal(hash(SIMD[DType.int8, 32](0)), 7079057015559465054)
    assert_equal(hash(SIMD[DType.int8, 64](0)), 11911213625103275990)

    assert_equal(hash(SIMD[DType.int32, 1](0)), 16659764227661506736)
    assert_equal(hash(SIMD[DType.int32, 2](0)), 1562284133626399299)
    assert_equal(hash(SIMD[DType.int32, 4](0)), 17902233708981521127)
    assert_equal(hash(SIMD[DType.int32, 8](0)), 632562262308536351)
    assert_equal(hash(SIMD[DType.int32, 16](0)), 7298276873920245913)
    assert_equal(hash(SIMD[DType.int32, 32](0)), 7079057015559465054)
    assert_equal(hash(SIMD[DType.int32, 64](0)), 11911213625103275990)


def main():
    test_hash_byte_array()
    test_avalanche()
    test_trailing_zeros()
    test_fill_factor()
    test_hash_simd_values()
