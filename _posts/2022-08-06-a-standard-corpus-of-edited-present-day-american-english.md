---
layout: ML-note
title: "A Standard Corpus of Edited Present-Day American English"
date: 2022-08-06 02:07:00 +0800
categories: [
  Dataset,
]
tags: [
  Brown Corpus,
  part of speech,
]
author: [
  W. Nelson Francis,
]
---

|-|-|
|目標|建立大型文字標記資料集|
|作者|W. Nelson Francis|
|隸屬單位|University of Brown|
|期刊/會議名稱|College English|
|發表時間|1964|
|論文連結|<https://www.jstor.org/stable/373638>|
|參考手冊|<http://korpus.uib.no/icame/manuals/BROWN/INDEX.HTM>|

## 重點

- 建立大型文字標記資料集，後續標記都以 Brown Corpus 為參考基準
- 蒐集文字對象為 edited American English，prepared for print
- 所有詞都被標記詞性與語法結構

## 蒐集文字

<a name="paper-fig-1"></a>

圖 1：蒐集來源與編碼
圖片來源：[論文][論文]。

![圖 1](https://i.imgur.com/FkZo2e9.png)

- 起始時間：1961
- 蒐集對象：編輯後準備發表的美語文字（edited American English，prepared for print）
  - 這代表蒐集文字中不包含口語（spoken English）、信件（personal letters）、student themes、油印稿（ephemeral mimeographed material）
  - 所有蒐集資料都是來自 USA 出版物，假設都是由美國人寫的，無法保證假設為真
- 特定文字會被濾除，包含
  - 詩（verse），但如果是教材則納入蒐集
  - 戲劇（drama）
  - 包含大量數學與科學符號的文字（mathematical and scientific writing so laden with formulas as to be hardly linguistic discourse at all）
  - 對話數量超過 50% 以上的小說（fiction that contains more than 50 per cent dialogue）
  - 色情文學（outright pornography）
  - 備註（footnotes）
  - 參考書目（bibliographies）
  - 圖片標題（picture captions）
  - 表格（tables）
  - 意會型與偶發性用語（illustrative and incidental linguistic material）
- 目標蒐集總詞數為 1M，拆成 500 個範例，每個範例中包含 2000 詞
- 蒐集來源共分 15 個種類，主要種類會給予一個編碼（code letter），種類與細節見[圖 1](#paper-fig-1)
  - 絕大部分樣本母體來自兩個圖書館 Brown Univarsity Library 與 the Providence Athenaeum 館中的收藏
  - 針對 daily press 樣本母體來自於 New York Public Library
  - 針對類別 E. Skills and Hobbies 樣本母體來自於紐約數一數二大的二手雜誌書店
- 每個子類別的抽樣方法為
  1. 對子類別的總文章數作 uniform sampling 選出文章
  2. 對選出文章的總頁碼進行 uniform sampling 選出起始頁面
  3. 從起始頁面中的第一個完整句子開始紀錄連續文字，紀錄總詞數控制在約為 2000 次，當滿足 2000 詞後看到完整的句子結尾便停止紀錄
- 詞的定義為前後有空白出現的連續字母序列
  - 數學與化學公式會用特殊符號代替，並當成一個詞
- 句子的定義為
  - 由一連串的詞與結尾所組成
  - 第一個詞中的第一個字母為大寫
  - 結尾的定義為符號 `.!?` 加上空白加上大寫字母
  - 縮寫不會被當成結尾
- 版權（copyright）問題
  - 除了政府資料外，所有資料都有版權所屬者
  - 作者寄信給所有版權擁有者，絕大部分的人都迅速表示同意授權甚至對研究很有興趣
  - 由於希望不需給付版權費，因此部份樣本有重新抽樣
  - 少部份人表示 computer tape 的版權狀態當前無法評估，作者與版權專家都表示無奈（哇塞超級版權砲）
- 針對資料的編碼採用 U.S. Patent Office 提出的標準 "Notation System for Transliterating Technical and Scientific Texts for Use in Data Processing Systems"
  - 由於該標準是針對專利文件，因此作者提出了一些修改方便使用
  - 所有修改細節請見[參考手冊][參考手冊]
  - 所有複製資料都會附贈參考手冊
- 作者有額外撰寫移除標點符號的程式，目的是方便計算詞頻
  - 移除的標點符號不包含 internal hyphen、apostrophes、diacritics 等
  - 提供有包含程式碼與不包含程式碼的磁帶

## 標記

原始版本並不包含標記，後續版本（版本 C）才額外補上標記。

- 專有名詞的大寫保留
- 包有語法結構的標點符號保留，其他一律捨棄
- 標記類別數為 81

標記分成 6 大類別：

- Mayor form-classes（主要詞彙）
  - 具體上就是指 open lexcial classes（開放式詞彙），open 的意思是指常有新的詞彙加入
  - 此類別包含
    - Noun（名詞），可細分為 common noun（一般名詞）或 proper noun（專有名詞）
    - Verb（動詞）
    - Adjective（形容詞）
    - Adverb（副詞）
- Function words（功能詞）
  - 範圍包含 closed lexcial classes（封閉式詞彙）與 grammatical classes（語法詞彙），closed 的意思是指不常有新的詞彙加入
  - 此類別包含
    - Determiners（限定詞）
    - Prepositions（介系詞）
    - Conjunctions（連接詞）
    - Pronouns（代名詞）
- Certain important individual words（特殊詞）
  - （反義副詞）*not*
  - Existential（存在副詞） *there*
  - Infinitival（不定式）*to*
  - *do*、*be*、*have*
- Punctuation marks of syntactic significance（含有重要語法結構標點符號）
- Inflectional morphemes（詞根不變的變形）
  - 屬於此類的 noun 包含 noun plural（複數名詞）與 possessive（所有格）
  - 屬於此類的 verb 包含 past tense（過去式）、present participle（現在分詞）、past participle（過去分詞）與 3rd person singular concord marker（第三人稱單數型）
  - 屬於此類的 adjective 包含 comparative adjective（比較級）與 superlative adjective（最高級）
  - 屬於此類的 adverb 包含 adverb suffix（副詞後綴）
  - Brown Corpus 對此類別進行以下編碼
    - `$` 代表 possessive（所有格）
    - `D` 代表 past tense（過去式）
    - `G` 代表 present participle or gerund（現在分詞）
    - `S` 代表 plural（複數）
    - `N` 代表 past participle（過去分詞）
    - `O` 代表 objective case of pronoun（代名詞受格）
    - `R` 代表 comparative（比較級）
    - `T` 代表 superlative（最高級）
    - `Z` 代表 3rd singular verb（動詞第三人稱單數型）
- 外語及引用詞
  - 用 `FM` 代表 foreign word（外語）
  - 用 `NC` 代表 cited word（引用詞）
  - 兩者會用 hyphen 與其他標籤結合
  - 當詞出現在 headline（標題）時會補上 `-HL`
  - 當詞出現在 title（頭銜）時會補上 `-TL`

完整標記流程需參考 Automatic Grammatical Tagging of English, by Barbara B. Greene and Gerald M. Rubin (Providence: Brown Univ., 1971.)。
我找不到該份文件的連結，可能沒被數位化。

### Noun Phrase

Noun pharse（名詞片語）由 determiner sector（限定詞片段）+ modifier sector（修飾詞片段）+ head（頭）所組成。

> The (noun phrase) model for this consists of a head preceded by a determiner sector and a modifier sector.

#### Determiner Sector

Determiner sector 的中心為 determiner，主要包含三個種類：

- Article（冠詞）
  - 例如：*a*、*an*、*the*
  - 標記為 `AT`
- Deictics（指事詞）
  - 例如：*this*、*that*、*another*、*each*
  - 標記為 `DT`
  - Deictics with plurals（複數指事詞）
    - 例如：*these*、*those*
    - 標記為 `DTS`
  - Dual deictics（雙數指事詞）
    - 例如：*either*、*neither*
    - 標記為 `DTX`
    - Dual detictics 也常常作為 correlative conjunctions（相關連接詞）使用
- Quantifiers not marked for number（不具體描述數量的量詞）
  - 例如：*some*、*any*
  - 標記為 `DTI`

出現在 determiner 之前的詞包含：

- Pre-quantifiers（限定詞前置量詞）
  - 例如：*all*、*half*
  - 標記為 `ABN`
- Both（描述雙數）
  - 此類別只有 *both* 一個詞
  - 標記為 `ABX`
  - 有時作為 correlative conjunctions 使用

出現在 determinier 之後，modifier 之前的詞包含：

- Post-determinier（限定詞後置詞）
  - 主要都是 quantifier（量詞），例如：*many*、*more*、*most*、*several*、*single*
  - 也包含 particularizers，例如：*past*、*next*、*some*、*only*
  - 標記為 `AP`
- Cardinal numerals（基數）
  - 例如：*one*、*two*、*three*
  - 標記為 `CD`
- Ordinal numerals（序數）
  - 例如：*first*、*second*、*third*
  - 標記為 `OD`
- Possessive nouns and pronouns（所有格名詞）
  - 例如：*cat\'s*、*his*、*mine*
  - 標記結尾加上 `$`

#### Modifier Sector

Modifier sector 最簡單的形式是以三種類別構成：

- Adjectives（由形容詞組成修飾詞），包含
  - Positive adjectives（一般形容詞），標記為 `JJ`
  - Comparative adjectives（比較級形容詞），標記為 `JJR`
  - Superlative adjectives（最高級容詞），標記為 `JJT`
- Participles（由分詞組成修飾詞），包含
  - Present participles（現在分詞），標記為 `VBG`
  - Past participles（過去分詞），標記為 `VBN`
- Nominals（由名稱組成修飾詞），標記與 head 規則相同
- Compounding（由多個詞複合組成修飾詞），標記規則複雜

由 Adjective 組成的 modifier 前後可以加上以下內容：

- Adjective may be modified by qualifiers（在形容詞前面加上程度詞修飾）
  - 例如：*rather*、*very*、*too*
  - 標記為 `QL`
- Adjective may be followd by the post-qualifiers（在形容詞後面加上程度詞修飾）
  - 例如：*enough*、*indeed*
  - 標記為 `QLP`
- In general, adverbs in *-ly* immediately preceding and clearly qualifying an adjective or adverb are commonly tagged `QL` rather than the general adverb tag `RB`.
  - 例如：*exceedingly*、*sufficiently*、*terribly*、*unusually*
- Certain adjectives which are semantically superlative and thus never compared are given the tag `JJS`
  - 例如：*chief*、*head*、*main*、*prime*、*principal*、*single*、*top*

Compound words（複合詞）作為修飾詞的規則超級複雜，原因是複合詞的結構多變，因此規則也伴隨結構進行探討。

首先區分複合詞的組成方法：

- Open compound（多個詞以空格相隔組成一個片語）
- Hyphened compound（ hyphen 串接多個詞組成一個片語）
- Closed compound（多個詞去除空格組成一個片語）
- Adjunction（多個詞以空格相隔組成一個片語，但彼此之間沒有關聯）
- Affixation（加上綴字）
  - 英文包含 Preffixation（加上前綴）與 Suffixation（加上後綴）
  - 英文沒有 Inffixation（任意位置加上綴字）

以下列出複合詞作為修飾詞的標記方法：

1. 如果複合詞是 hyphened compound，且在去除 hyphen 後剩餘的所有詞是一個合法 noun phrase，則標記為 `NN`
   - 例如：*long-range*、*high-energy*
2. 如果複合詞是由單詞 + *-ed* 組成，且去除 *-ed* 後為 verb，則標記為 `VBN`
   - 例如：*united*
3. 如果複合詞是由多個詞 + *-ed* 組成，且去除 *-ed* 後為 verb，則標記為 `VBN`
   - 例如：*downgraded*
4. 如果複合詞是由單詞 + *-ing* 組成，且去除 *-ing* 後為 verb，則標記為 `VBG`
   - 例如：*outdistancing*
5. 如果複合詞是由多個詞 + *-ing* 組成，且去除 *-ing* 後為 verb，則標記為 `VBG`
   - 例如：*double-crossing*
6. 在 2345 的規則下有例外，當詞由 qualifier 修飾，則被標記為 `JJ`
   - 例如：*very tired*、*rather entertaining*
7. Words normally nouns appearing in the immediate prenomial position（名稱之前的詞）會被當成 noun-adjunct，每個詞都標記為 `NN`
   - 例如：*army officer*、*weather report*
8. 不符合上述規則的都被標記為 `JJ`，因此 `JJ` 類別會包含超大量且規則複雜的內容，以下舉例
   - Words ending in *-type*：*sandwich-type*
   - Noun-Adjective combinations：*fancy-free*、*screw-loose*、*shoulder-high*
   - Noun-Present Participle constructions：*run-scoring*、*sales-building*、*law-abiding*
   - Noun-Past Participle constructions：*home-made*、*rock-strewn*
   - Noun-Noun+*-ed* combinations：*shirt-sleeeved*
   - Adjective-Noun+*-ed* combinations：*short-skirted*、*slim-waisted*
   - Miscellaneous combinations：*show-offy*、*signal-to-noise*、*smash-\'em-down*、*snob-clannish*、*topsy-turvy*、*to-the-death*、*tongue-in-cheek*、*too-simple-to-be-true*、*unique-ingrown-screwedup*、*round-the-clock*、*day-after-day*

#### Head

- Singular noun（名詞單數型）被標記為 `NN`
- Plural noun（名詞複數型）被標記為 `NNS`
- Possessive noun（名詞所有格）被標記為 `NN$`
- Possessive plural noun（名詞所有格複數型）被標記為 `NNS$`
- Proper noun（專有名詞）被標記為 `NP`
- Plural proper noun（專有名詞複數型）被標記為 `NPS`
- Possessive proper noun（專有名詞所有格）被標記為 `NP$`
- Possessive plural proper noun（專有名詞所有格複數型）被標記為 `NPS$`

### Verbal Phrase

- Verbs in the base form（動詞原型）被標記為 `VB`
- Verbs in the 3rd person singular inflected form（動詞第三人稱單數型）被標記為 `VBZ`
- Verbs in the past tense（動詞過去式）被標記為 `VBD`
- Verbs in the past participle（動詞過去分詞）被標記為 `VBN`
- Verbs in the present participle（動詞現在分詞）被標記為 `VBG`
- Modal auxiliary verbs（情態助動詞）被標記為 `MD`
  - 不論時態都標記成 `MD`
  - 例如：*can*、*could*、*may*、*might*、*shall*、*should*
- *be* 被標記為 `BE`
  - 不論是作為 auxiliary verb（助動詞）或 full verb（一般動詞）都標記為 `BE`
  - *were* 被標記為 `BED`
  - *was* 被標記為 `BEDZ`
  - *being* 被標記為 `BEG`
  - *am* 被標記為 `BEM`
  - *been* 被標記為 `BEN`
  - *are*、*art* 被標記為 `BER`
  - *is* 被標記為 `BEZ`
- *have* 被標記為 `HV`
  - 不論是作為 auxiliary verb（助動詞）或 full verb（一般動詞）都標記為 `HV`
  - *had*（past tense）被標記為 `HVD`
  - *having* 被標記為 `HVG`
  - *had*（past participle）被標記為 `HVN`
  - *has* 被標記為 `HVZ`
- *do* 被標記為 `DO`
  - 不論是作為 auxiliary verb（助動詞）或 full verb（一般動詞）都標記為 `DO`
  - *did* 被標記為 `DOD`
  - *does* 被標記為 `DOZ`
  - *doing* 被標記為 `VBG`
  - *done* 被標記為 `VBN`
- Contracted forms of auxiliaries（助動詞縮寫）會與 subject（主詞）一起標記，標記的方式為 subject + auxiliary
  - 例如：I\'m 被標記為 `PPSS+BEM`
  - 例如：you\'ve 被標記為 `PPSS+HV`
  - 例如：he\'d 被標記為 `PPS+MD`
- Contracted negatives（反義縮寫）標記方式為 auxiliary + `*`
  - 例如：can\'t 被標記為 `MD*`
- Condensed forms in dialogue（對話縮寫）會以原始結構進行標記
  - 例如：gonna 被標記為 `VBG+TO`

### Pronoun

Personal pronouns（人身代名詞）的標記皆由 `PP` 開頭，額外接上一個字母代表不同的情況。
額外的字母可以是 case（主格、所有格、受格）、concord（與動詞結合的變化）或 number（第一、二、三人稱）。

- 3rd person singular nominative pronoun（第三人稱單數代名詞主格）
  - 例如：*he*、*she*、*it*、*one*
  - 標記為 `PPS`
- Nominative personal pronoun other than 3rd person singular（非第三人稱單數代名詞主格）
  - 例如：*I*、*we*、*they*、*you*
  - 標記為 `PPSS`
- Objective personal pronoun（任意人稱代名詞受格）
  - 例如：*me*、*him*、*it*、*them*
  - 標記為 `PPO`
- Possessive personal pronoun（第一人稱所有格）
  - 例如：*my*、*our*
  - 標記為 `PP$`
- Second (nominal) possessive pronoun
  - 例如：*mine*、*ours*
  - 標記為 `PP$$`
- Singular reflexive / intensive personal pronoun（單數反身代名詞）
  - 例如：*myself*、*yourself*
  - 標記為 `PPL`
- Plural reflexive / intensive personal pronoun（複數反身代名詞）
  - 例如：*ourselves*、*themselves*
  - 標記為 `PPLS`
- Interrogative pronoun（疑問代詞）與 relative pronoun（相對代詞）都由 `WP` 作為標記開頭
  - 作為主詞則標記為 `WPS`
  - 作為受詞則標記為 `WPO`
- Indefinite pronouns（不定代名詞）：由 *any-*、*every-*、*no-*、*some-* 組合而成的複合詞
  - 例如：*anyone*、*everyone\'s*、*nobody*、*somebody\'s*
  - 沒有 *\'s* 標記為 `PN`
  - 有 *\'s* 標記為 `PN$`
- Demonstrative pronouns（指事代詞）被當成 determiners（限定詞）
  - 例如：*this*、*that*
  - Singular determiner 標記為 `DT`
  - Singular or plural determiner 標記為 `DTI`
  - Plural determiner 標記為 `DTS`

### Adverbials

Adverbials（副詞片語）可以由單個 adverb（副詞）組成或是由多個詞組成，用以描述或修飾 verbs、adjectives or clauses。

- Adverb（副詞）
  - 例如：he swam *fast*
  - 標記為 `RB`
- Inflectional comparative adverb
  - 例如：he swam *faster* than another swimmer
  - 標記為 `RBR`
- Inflectional superlative adverb
  - 例如：he swim *fastest* among others
  - 標記為 `RBT`
- Nominal adverb（名詞性副詞），主要與時間或地點相關，本身為 adverb 卻常常作為 nominals 使用
  - 例如：*here*、*then*、*indoors*
  - 標記為 `RN`
- Advervial nouns（副詞性名詞），主要與時間或地點相關，本身為 noun 卻常常作為 adverbial 使用
  - 例如：*home*、*east*、*Tuesday*
  - 標記為 `NR`

在區分 article 與 particle（助詞）時，作者認為需要同時考量語法與語意才有辦法正確標記，因此以合成詞（portmanteau）的方式創造標記 `RP`，含意為 adverb or particle，用來標記可為兩者的詞。
此類別包含十個詞 *about*、*across*、*down*、*in*、*off*、*on*、*out*、*over*、*through*、*up*。
當這些詞作為 preposition 使用時則標記為 `IN`。

### Connectives

- Coordinating conjunction（對等連接詞）
  - 例如：*and*、*or*
  - 標記為 `CC`
- Subordinators（從屬子句連接詞）
  - 例如：*since*、*because*、*if*
  - 標記為 `CS`
- Prepositions（介系詞）
  - 例如：*in*、*on*
  - 標記為 `IN`
- *to* 當成 infinitive marker 時標記為 `TO`

### Miscellaneous Items

- The existential subject *there*
  - 標記為 `EX`
  - 與作為副詞時使用進行區分
- Exclamations（驚嘆詞）
  - 大部份只出現在對話
  - 標記為 `UH`
- *not*
  - 標記為 `*`
  - 在與動詞合成時標記也會合成

### Capitalized Words, Titles, and Proper Nouns

- 所有句子開頭的大寫都轉換成小寫，除了本來就是大寫的詞保留
- 在句子中的某些詞有可能因為作者的強調或其他因素而採用大寫，出現位置為隨機，因此保留大寫但不改變標記結果
- Proper nouns（專有名詞）的大小寫保留
  - 單數標記為 `NP`
  - 複數標記為 `NPS`
  - 單數所有格標記為 `NP$`
  - 複數所有格標記為 `NPS$`
- 外語會以 `FW-` 作為標記開頭
- 大部份出現在頭銜的詞都會在原始標記上加入 `-TL` 的標記
  - 出現在頭銜的詞幾乎都是大寫開頭，除了 prepositions、conjunctions 與 pronouns
  - 有些外語（例如法文）在頭銜不會大寫
- 專有名詞可以由多個詞合成，標記的規則如下
  1. 只要是人名都標記為 `NP`
  2. 組成地理詞彙的名詞在給予基本標記後補上 `-TL`
  3. 組成地理詞彙的專有名詞標記為 `NP-TL`
  4. 頭銜如 *Mr.*、*Mrs.*、*Ms.*、*Miss*、*Sir* 標記為 `NP`
  5. 同時擁有作為名詞、形容詞等功能的其他人類頭銜會在標記加上 `-TL`
  6. 不常見的外語頭銜標記為 `NP`

## 標記總表

| Tag    | Description                                               | Examples                    |
|--------|-----------------------------------------------------------|-----------------------------|
| `.`    | sentence closer                                           | . ; ? !                     |
| `(`    | left parenthesis                                          |                             |
| `)`    | right parenthesis                                         |                             |
| `*`    | *not*, *n\'t*                                             |                             |
| `--`   | dash                                                      |                             |
| `,`    | comma                                                     |                             |
| `:`    | colon                                                     |                             |
| `ABL`  | pre-qualifier                                             | *quite*, *rather*           |
| `ABN`  | pre-quantifier                                            | *half*, *all*               |
| `ABX`  | pre-quantifier                                            | *both*                      |
| `AP`   | post-determiner                                           | *many*, *several*, *next*   |
| `AT`   | article                                                   | *a*, *the*, *no*            |
| `BE`   | *be*                                                      |                             |
| `BED`  | *were*                                                    |                             |
| `BEDZ` | *was*                                                     |                             |
| `BEG`  | *being*                                                   |                             |
| `BEM`  | *am*                                                      |                             |
| `BEN`  | *been*                                                    |                             |
| `BER`  | *are*, *art*                                              |                             |
| `BEZ`  | *is*                                                      |                             |
| `CC`   | coordinating conjunction                                  | *and*, *or*                 |
| `CD`   | cardinal numeral                                          | *one*, *two*, *2*, etc.     |
| `CS`   | subordinating conjunction                                 | *if*, *although*            |
| `DO`   | *do*                                                      |                             |
| `DOD`  | *did*                                                     |                             |
| `DOZ`  | *does*                                                    |                             |
| `DT`   | singular determiner                                       | *this*, *that*              |
| `DTI`  | singular or plural determiner / quantifier                | *some*, *any*               |
| `DTS`  | plural determiner                                         | *these*, *those*            |
| `DTX`  | determiner / double conjunction                           | *either*                    |
| `EX`   | existentil *there*                                        |                             |
| `FW`   | foreign word (hyphenated before regular tag)              |                             |
| `HL`   | word occurring in headline (hyphenated after regular tag) |                             |
| `HV`   | *have*                                                    |                             |
| `HVD`  | *had* (past tense)                                        |                             |
| `HVG`  | *having*                                                  |                             |
| `HVN`  | *had* (past participle)                                   |                             |
| `HVZ`  | *has*                                                     |                             |
| `IN`   | preposition                                               |                             |
| `JJ`   | adjective                                                 |                             |
| `JJR`  | comparative adjective                                     |                             |
| `JJS`  | semantically superlative adjective                        | *chief*, *top*              |
| `JJT`  | morphologically superlative adjective                     | *biggest*                   |
| `MD`   | modal auxiliary                                           | *can*, *should*, *will*     |
| `NC`   | cited word (hyphenated after regular tag)                 |                             |
| `NN`   | singular or mass noun                                     |                             |
| `NN$`  | possessive singular noun                                  |                             |
| `NNS`  | plural noun                                               |                             |
| `NNS$` | possessive plural noun                                    |                             |
| `NP`   | proper noun or part of name phrase                        |                             |
| `NP$`  | possessive proper noun                                    |                             |
| `NPS`  | plural proper noun                                        |                             |
| `NPS$` | possessive plural proper noun                             |                             |
| `NR`   | adverbial noun                                            | *home*, *today*, *west*     |
| `NRS`  | plural adverbial noun                                     |                             |
| `OD`   | ordinal numeral                                           | *first*, *2nd*              |
| `PN`   | nominal pronoun                                           | *everybody*, *nothing*      |
| `PN$`  | possessive nominal pronoun                                |                             |
| `PP$`  | possessive personal pronoun                               | *my*, *our*                 |
| `PP$$` | second (nominal) possessive pronoun                       | *mine*, *ours*              |
| `PPL`  | singular reflexive/intensive personal pronoun             | *myself*                    |
| `PPLS` | plural reflexive/intensive personal pronoun               | *ourselves*                 |
| `PPO`  | objective personal pronoun                                | *me*, *him*, *it*, *them*   |
| `PPS`  | 3rd. singular nominative pronoun                          | *he*, *she*, *it*, *one*    |
| `PPSS` | other nominative personal pronoun                         | *I*, *we*, *they*, *you*    |
| `QL`   | qualifier                                                 | *very*, *fairly*            |
| `QLP`  | post-qualifier                                            | *enough*, *indeed*          |
| `RB`   | adverb                                                    |                             |
| `RBR`  | comparative adverb                                        |                             |
| `RBT`  | superlative adverb                                        |                             |
| `RN`   | nominal adverb                                            | *here*, *then*, *indoors*   |
| `RP`   | adverb / particle                                         | *about*, *off*, *up*        |
| `TL`   | word occurring in title (hyphenated after regular tag)    |                             |
| `TO`   | infinitive marker *to*                                    |                             |
| `UH`   | interjection, exclamation                                 |                             |
| `VB`   | verb, base form                                           |                             |
| `VBD`  | verb, past tense                                          |                             |
| `VBG`  | verb, present participle / gerund                         |                             |
| `VBN`  | verb, past participle                                     |                             |
| `VBZ`  | verb, 3rd. singular present                               |                             |
| `WDT`  | *wh-* determiner                                          | *what*, *which*             |
| `WP$`  | possessive *wh-* pronoun                                  | *whose*                     |
| `WPO`  | objective *wh-* pronoun                                   | *whom*, *which*, *that*     |
| `WPS`  | nominative *wh-* pronoun                                  | *who*, *which*, *that*      |
| `WQL`  | *wh-* qualifier                                           | *how*                       |
| `WRB`  | *wh-* adverb                                              | *how*, *where*, *when*      |


[論文]: https://www.jstor.org/stable/373638
[參考手冊]: http://korpus.uib.no/icame/manuals/BROWN/INDEX.HTM
