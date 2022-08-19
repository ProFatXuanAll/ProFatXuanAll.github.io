---
layout: ML-note
title: "Building a Large Annotated Corpus of English : The Penn Treebank"
date: 2022-08-12 15:24:00 +0800
categories: [
  Dataset,
]
tags: [
  Penn Treebank,
  part of speech,
  constituent parse,
]
author: [
  Mitchell Marcus,
  Beatrice Santorini,
  Mary Ann Marcinkiewicz,
]
---

|-|-|
|目標|建立大型文字標記資料集，基於 Brown Corpus 的詞性標記作修改|
|作者|Mitchell Marcus, Beatrice Santorini, Mary Ann Marcinkiewicz|
|隸屬單位|University of Pennsylvania|
|期刊/會議名稱|Computational Linguistics|
|發表時間|1993|
|論文連結|<https://alliance.seas.upenn.edu/~nlp/publications/pdf/marcus1993.pdf>|
|詞性標記參考手冊|<https://repository.upenn.edu/cgi/viewcontent.cgi?article=1603&context=cis_reports>|
|語法成份分析標記參考手冊|<http://languagelog.ldc.upenn.edu/myl/PennTreebank1995.pdf>（找不到原版）|

## 重點

- 包含超過 4.5M 個詞的英文（American）資料集
- 標記資料執行了三年（1989--1992），標記了詞性（part-of-speech，POS）與語法成份分析（constituent parse）
- 資料可以在 [Linguistic Data Consortium（LDC）][LDC]付費下載
- 資料是先採用自動化標記，後由人工校正的標記流程
  - 此流程速度較快、一致性高
  - 當不採用自動化標記的流程時，速度較慢，一致性降低
- Penn Treebank 詞性標記受到 Brown Corpus 的啟發
  - Brown Corpus 共有 87 種基礎詞性標記
  - Brown Corpus 允許透過基礎詞性標記進行組合，因此共有 187 種基礎型 + 組合型詞性標記
  - Brown Corpus 後續的研究通常都是將詞性標記分的更細，目的是為了將每個詞在不同文法下的功能進行區分（The ideal of providing distinct codings for all classes of words having distinct grammatical behaviour）
    - Lancaster-Oslo / Bergen（LOB）Corpus 擁有 135 種詞性標記
    - Lancaster UCREL group 擁有 165 種詞性標記
    - London-Lund Corpus of Spoken English 擁有 197 種詞性標記
- Penn Treebank 的詞性標記是將 Brown Corpus 的詞性標記種類減少後得到的結構
  - 作者宣稱這是基於**統計**且考量詞彙（lexical）與語法（syntactic）的資訊下進行標記簡化
  - 在 Brown Corpus 中隸屬於特定詞彙的標籤都在 Penn Treebank 中被視為多餘資訊並移除
  - Penn Treebank 的去除標籤範例見[圖 1](#paper-fig-1)
  - 標記總詞數約為 4.88M
- Penn Treebank 的語法成份分析標記將 Lancaster UCREL Treebank Project 中的結構精減（skeletal syntactic structure）
  - 以 **context free grammar** 為前提進行標記
  - 標記總詞數約為 2.88M

## 詞性標記

作者認為 Penn Treebank 的出發點為**統計**，因此在考量**詞彙**（**lexical**）與**語法**（**syntactic**）的資訊下將 Brown Corpus 中的 POS 標記種類種類減少。

### 與 Brown Corpus 的差別

<a name="paper-fig-1"></a>

圖 1：Penn Treebank 動詞標記。
圖片來源：[論文][論文]。

![圖 1](https://i.imgur.com/CMZtL3B.png)

- Penn Treebank 基於**詞彙還原度**（**lexical recoverability**）減少標記種類
  - 在 Brown Corpus 中許多類似的標記被進一步細分，作者認為這些區分只要**觀察詞彙本身**就能夠區分差異，不需要在標記進行區別
  - 例如：Brown Corpus 中的**助動詞**（**auxiliary verb**）有獨立標記，在 Penn Treebank 中採用一致的**動詞**（**verb**）標記（見[圖 1](#paper-fig-1)）
  - 例如：Brown Corpus 中**限定詞前置程度詞**（**pre-qualifiers**，標記為 `ABL`）、**限定詞前置量詞**（**pre-quantifiers**，標記為 `ABN`）或是 *both*（標記為 `ABX`）都有不同的標記，Penn Treebank 統一標記為**限定詞前置詞**（**predeterminer**，標記為 `PDT`）
  - 例如：Brown Corpus 中**單數反身人稱代名詞**（**singular reflexive personal pronoun**，標記為 `PPL`）與**複數反身人稱代名詞**（**plural reflexive personal pronoun**，標記為 `PPLS`）各有不同標記，在 Penn Treebank 中都合併進入**人稱代名詞**（**personal pronoun**，標記為 `PRP`）
  - 例如：Brown Corpus 中**名詞性副詞**（**nominal adverb**，標記為 `RN`）有獨立標記，在 Penn Treebank 中合併進入**副詞**（**adverb**，標記為 `RB`）
- Penn Treebank 基於**語法還原度**（**syntactic structure recoverability**）減少標記種類
  - 在 Brown Corpus 中許多名詞因為語法結構不同被賦與特殊標記，作者認為只要**觀察語法結構**就能夠區分差異，不需要在標記上進行區別
  - 例如：Brown Corpus 中**主詞代名詞**（**subject pronoun**，標記為 `PPSS`）與**受詞代名詞**（**object pronoun**，標記為 `PPO`）有不同標記，在 Penn Treebank 中合併成**人身代名詞**（**personal pronoun**，標記為 `PRP`）
  - 例如：Brown Corpus 中**從屬子句連接詞**（**subordinating conjunction**，標記為 `CS`）有獨立標記，在 Penn Treebank 中合併進入**介系詞**（**preposition**，標記為 `IN`）
- Penn Treebank 具有**一致性**（**consistency**）
  - 作者宣稱在同時考量詞彙還原度與語法還原度後能夠減少標記不一致的問題
  - 例如：在 Brown Corpus 中 *there*、*now* 都被標記為**副詞**（**adverb**，標記為 `RB`），但在完全相同的語法結構下 *here* 與 *then* 有可能被標記為**副詞**或**名詞性副詞**（**nominal adverb**，標記為 `RN`）；而 Penn Treebank 都一致標記為**副詞**
- Penn Treebank 標記有融入**語法功能**（**syntactic function**）
  - 例如：Brown Corpus 在進行**名詞片語**（**noun phrase）標記**時，**中心語**（**head**）不一定會被標記成**名詞**（**noun**，標記為 `NN`）；而 Penn Treebank 會考量中心語的**語法功能**進行標記，因此**中心語**會標記成**名詞**
  - 例如：Brown Corpus 永遠將 *both* 標記成 `ABX`；而 Penn Treebank 會依照 *both* 出現的位置給予不同標記
    - *both the boys* 中的 *both* 為冠詞 *the* 的前置詞，標記為 `PDT`
    - *the boys both* 中的 *both* 為名詞片語 *the boys* 的後置修飾詞，標記為 `RB`
    - *both of the boys* 中的 *both* 為名詞片語中心語且為複數，標記為 `NNS`
    - *both boys and girls* 中的 *both* 為名詞 *boys* 與 *girls* 的連接詞，標記為 `CC`
  - 唯一的例外是 existential *there*，Penn Treebank 中維持標記 `EX`
- Penn Treebank 針對**非第三人稱單數型動詞**（**non-3rd person singular present tense**）新增一個標籤 `VBP`
  - Brown Corpus 使用動詞原型（verb，標記為 `VB`）標記不定式（infinitive）、命令語句（imperative）與非第三人稱單數型動詞（non-3rd person singular present tense）
  - Penn Treebank 對不定式原型與命令語句都採用 `VB`
- Penn Treebank 能夠容忍**不確定性**（**indeterminacy**）
  - 在沒有足夠資訊的前提下，並不是所有標籤都是唯一的
    - 例如：*Grant can be outspoken--but not by anyone I know*，一般來說 *outspoken* 可以標記為形容詞（adjective，`JJ`），但後面的 *by* 告訴我們 *outspoken* 在這裡其實是過去分詞（past participle）
  - 因此作者認為可以容忍多個標記答案，不同標記間以 `|` 進行區隔
  - 實際上出現的多標記組合種類並不多
    - 形容詞或名詞，標記為 `JJ|NN`
    - 形容詞或現在分詞（present participle），標記為 `JJ|VBG`
    - 形容詞或過去分詞，標記為 `JJ|VBN`
    - 名詞或現在分詞，標記為 `NN|VBG`
    - 副詞或助詞，標記為 `RB|RP`

### 標記總表

<a name="paper-fig-2"></a>

圖 2：Penn Treebank 詞性標記總表。
圖片來源：[論文][論文]。

![圖 2](https://i.imgur.com/2Gggpbf.png)

- 總共包含了 36 個詞性標籤與 12 個其他標籤（主要是給標點符號與貨幣符號）
- 與 Brown Corpus 相比
  - 少了
    - 限定詞前置程度詞（pre-qualifier），標記為 `ABL`
    - 限定詞前置量詞（pre-quantifier），標記為 `ABN`
    - 單詞 *both*，標記為 `ABX`
    - 限定詞後置詞（post-determiner），標記為 `AP`
    - 冠詞（article），標記為 `AT`
    - 助動詞 *be*，標記為 `BE`
    - 助動詞 *were*，標記為 `BED`
    - 助動詞 *was*，標記為 `BEDZ`
    - 助動詞 *being*，標記為 `BEG`
    - 助動詞 *am*，標記為 `BEM`
    - 助動詞 *been*，標記為 `BEN`
    - 助動詞 *are*，標記為 `BER`
    - 助動詞 *is*，標記為 `BEZ`
    - 關係子句連接詞（subordinating conjunction），標記為 `CS`
    - 助動詞 *do*，標記為 `DO`
    - 助動詞 *did*，標記為 `DOD`
    - 助動詞 *does*，標記為 `DOZ`
    - 單或複數限定詞 / 量詞（singular or plural determiner / quantifier），標記為 `DTI`
    - 複數限定詞（plural determiner），標記為 `DTS`
    - 雙數限定詞（determiner / double conjunction），標記為 `DTX`
    - 標題（word occurring in headline），標記為 `HL`
    - 助動詞 *have*，標記為 `HV`
    - 助動詞 *have* 的過去式 *had*，標記為 `HVD`
    - 助動詞 *having*，標記為 `HVG`
    - 助動詞 *have* 的過去分詞 *had*，標記為 `HVN`
    - 助動詞 *has*，標記為 `HVZ`
    - 最高級形容詞（morphologically superlative adjective），標記為 `JJT`
    - 語意最高級形容詞（semantically superlative adjective），標記為 `JJS`
    - 單數名詞所有格（possessive singular noun），標記為 `NN$`
    - 複數名詞所有格（possessive plural noun），標記為 `NNS$`
    - 單數專有名詞（proper noun or part of name phrase），標記為 `NP`
    - 單數專有名詞所有格（possessive proper noun），標記為 `NP$`
    - 複數專有名詞（plural proper noun or part of name phrase），標記為 `NPS`
    - 複數專有名詞所有格（possessive plural proper noun），標記為 `NPS$`
    - 單數副詞性名詞（adverbial noun），標記為 `NR`
    - 複數副詞性名詞（plural adverbial noun），標記為 `NRS`
    - 序數（ordinal number），標記為 `OD`
    - 名詞性代詞（nominal pronoun），標記為 `PN`
    - 名詞性代詞 + *\'s*（possessive nominal pronoun），標記為 `PN$`
    - 人稱代名詞受格（possessive personal pronoun），標記為 `PP$$`
    - 單數反身人稱代名詞（singular reflexive/intensive personal pronoun），標記為 `PPL`
    - 複數反身人稱代名詞（plural reflexive/intensive personal pronoun），標記為 `PPLS`
    - 人稱代名詞受格（objective personal pronoun），標記為 `PPO`
    - 第三人稱單數型代名詞主格（3rd. singular nominative pronoun），標記為 `PPS`
    - 非第三人稱代名詞主格（other nominative personal pronoun），標記為 `PPSS`
    - 程度詞（qualifier），標記為 `QL`
    - 形容詞後置程度詞（post-qualifier），標記為 `QLP`
    - 副詞最高級（superlative adverb），標記為 `RBT`
    - 名詞性副詞（nominal adverb），標記為 `RN`
    - 頭銜（word occurring in title），標記為 `TL`
    - *wh-* 代名詞受格（objective *wh-* pronoun），標記為 `WPO`
    - *wh-* 代名詞主格（nominative *wh-* pronoun），標記為 `WPS`
    - *wh-* 程度詞（wh- qualifier），標記為 `WQL`
    - *not*、*n\'t*，標記為 `*`
    - 刪節號（dash），標記為 `--`
  - 多了
    - 形容詞最高級（adjective，superlative），標記為 `JJS`
    - 列表項目符號（list item marker），標記為 `LS`
    - 單數專有名詞（proper noun，singular），標記為 `NNP`
    - 複數專有名詞（proper noun，plural），標記為 `NNPS`
    - 限定詞前置詞（predeterminer），標記為 `PDT`
    - 所有格縮寫 *\'s*（possessive ending），標記為 `POS`
    - 人稱代名詞（personal pronoun），標記為 `PRP`
    - 副詞最高級（adverb、superlative），標記為 `RBS`
    - 科學符號（symbol，mathematical or scientific），標記為 `SYM`
    - 非第三人稱、單數動詞現在式（verb，non-3rd person singular present tense），標記為 `VBP`
    - *wh-* 代名詞（*wh-* pronoun），標記為 `WP`
    - 英鎊（pound sign），標記為 `#`
    - 美金（dollar sign），標記為 `$`
    - 對稱雙引號（stright double quote），標記為 `"`
    - 左雙引號（left open double quote），標記為 `“`
    - 右雙引號（right closed double quote），標記為 `”`
    - 左單引號（left open single quote），標記為 `'`
    - 右單引號（right closed single quote），標記為 `'`

### 詞性標記參考手冊

[詞性標記參考手冊][詞性標記參考手冊]中額外描述的細節如下：

- 對等連接詞（conjunction，coordinating）標記為 `CC`
  - 包含 *and*、*but*、*nor*、*or*、*yet*
  - 包含數學運算 *plus*、*minus*、*less*、*times*、*over*
  - 當 *for* 作為 *because* 使用時標記為 `CC`
    - 例如 *He asked to be transferred, for he was unhappy.* 中的 *for* 標記為 `CC`
  - 當 *both ... and* 配合出現且作為雙連接詞（double conjunction）的開頭時標記 *both* 為 `CC`
    - 例如：*both boys and girls are happy* 中的 *both* 標記為 `CC`
  - 當 *either ... or* 配合出現且作為雙連接詞（double conjunction）的開頭時標記 *either* 為 `CC`
    - 例如：*Either a boy could sing or a girl could dance.* 中的 *Either* 標記為 `CC`
    - 例如：*Either a boy or a girl could sing.* 中的 *Either* 標記為 `CC`
    - 例如：*Either a boy or girl could sing.* 中的 *Either* 標記為 `CC`
    - 例如：*Either child could sing.* 中的 *Either* 標記為 `DT`
    - 例如：*Either boy or girl could sing.* 中的 *Either* 標記為 `DT`
  - 當 *neither ... nor* 配合出現且作為雙連接詞（double conjunction）的開頭時標記 *neither* 為 `CC`
- 基數（cardinal number）標記為 `CD`
  - 當數字以 *number-number* 的方式出現在形容詞的位置時則標記為 `JJ`
    - 例如：*a 50-3 victory* 中的 *50-3* 標記為 `JJ`
  - 當數字以 *number-number* 的方式出現在副詞的位置時則標記為 `RB`
    - 例如：*They won 50-3.* 中的 *50-3* 標記為 `RB`
  - 當分數（hyphenated fraction）出現在類名詞之前進行修飾（prenominal modifier）時標記為 `JJ`，但如果可以用 *double* 或 *twice* 替換時則標記為 `RB`
    - 例如：*one-half cup* 中的 *one-half* 標記為 `JJ`
    - 例如：*one-half the amount* 中的 *one-half* 標記為 `RB`
  - 當 *one* 出現時有可能是基數或名詞，但在不確定是否描述基數的狀態下應標記為 `CD`
    - 例如：*one of the best reasons* 中的 *one* 標記為 `CD`
  - 當 *one* 出現且可以被複數化（pluralized）時標記為 `NN`
    - 例如：*the only one of its kind* 中的 *one* 標記為 `NN`
    - 例如：*the only ones of its kind* 中的 *ones* 標記為 `NNS`
  - 當 *one* 出現且可以被形容詞修飾時標記為 `NN`
    - 例如：*the good one of its kind* 中的 *one* 標記為 `NN`
    - 例如：*the good ones of its kind* 中的 *ones* 標記為 `NNS`
  - 當 *another one* 一起出現時 *one* 標記為 `NN`
- 限定詞（determiner）標記為 `DT`
  - 包含冠詞（article）
    - 例如：*a*、*an*、*every*、*no*、*the* 都標記為 `DT`
  - 包含不定限定詞（indefinite determiner）
    - 例如：*another*、*any*、*some*、*each* 都標記為 `DT`
    - 例如：*either way* 中的 *either* 標記為 `DT`
    - 例如：*neither decision* 中的 *neither* 標記為 `DT`
    - 例如：*that*、*these*、*this*、*those* 都標記為 `DT`
  - 當 *all* 與 *both* 不是出現在其他限定詞之前，也不是出現在代名詞所有格之前時，標記為 `DT`，否則標記為 `PDT`
    - 例如：*all girls* 中的 *all* 標記為 `DT`
    - 例如：*all the girls* 中的 *all* 標記為 `PDT`
    - 例如：*both little boys* 中的 *both* 標記為 `DT`
    - 例如：*both the little boys* 中的 *both* 標記為 `PDT`
  - 由於一個名詞片語（noun phrase）中只能出現一個限定詞，而 *such* 可以同時與限定詞一起出現，因此 *such* 應該要標記為 `JJ`
    - 例如：*the only such case* 中的 *such* 標記為 `JJ`
  - 當 *such* 出現在限定詞之前時標記為 `PDT`
    - 例如：*such a good time* 中的 *such* 標記為 `PDT`
  - 當限定詞作為代名詞使用時應標記為 `DT`
    - 例如：*I can\'t stand this.* 中的 *this* 標記為 `DT`
    - 例如：*I\'ll take both.* 中的 *both* 標記為 `DT`
    - 例如：*Either would be fine.* 中的 *Either* 標記為 `DT`
- Existential there 標記為 `EX`
  - 通常作為句子開頭銜接 be 動詞（be verb）或情態動詞（modal），發音上無重音（unstressed），並造成主詞（subject）與動詞（verb）順序掉換
    - 例如：*There was a party in progress.* 中的 *There* 標記為 `EX`
    - 例如：*There ensued a melee.* 中的 *There* 標記為 `EX`
  - 當作為副詞使用時發音會有重音（stress），主詞（subject）與動詞（verb）不會掉換順序
    - 例如：*There, a party was in progress.* 中的 *There* 標記為 `RB`
    - 例如：*There, a melee ensued.* 中的 *There* 標記為 `RB`
  - 一個句子中可以同時出現 existential *there* 與 adverbial *there*
    - 例如：*There was a party in progress there.* 中的 *There* 與 *there* 分別標記為 `EX` 與 `RB`
- 外語（foreign word）標記為 `FW`
  - 作者認為外語判別標準較為寬鬆
    - 例如：*yoga* 標記為 `NN`
    - 例如：*bête noire* 標記為 `FW FW`
    - 例如：*persona non grata* 標記為 `FW FW FW`
- 介系詞（prepositions）標記為 `IN`
  - *to* 擁有專屬的標記 `TO`
  - 包含從屬連接詞（subordinating conjunction）
    - 例如：*so that* 中的 *so* 標記為 `IN`
  - 從屬連接詞會放在子句（clause）之前，而介系詞會放在名詞片語（noun phrase）或介系詞片語（prepositional phrase）之前
  - 當 *that* 作為名詞補語（complements of noun）使用時功能為從屬連接詞，標記為 `IN`
    - 例如：*the fact that you\'re here* 中的 *that* 標記為 `IN`
    - 例如：*the claim that angels have wings* 中的 *that* 標記為 `IN`
  - 當 *that* 作為關係從句（relative clause，子句作為形容詞使用）時標記為 `WDT`
    - 例如：*a man that I know* 中的 *that* 標記為 `WDT`
  - 大部分的情況下介系詞會放在名詞片語之前，但有時候會因為強調語法導致介系詞與名詞片語位置錯開
    - 例如：*the credit card you won\'t want to do without* 中的 *without* 標記為 `IN`
    - 例如：*you won\'t want to do without the credit card* 中的 *without* 標記為 `IN`
    - 例如：*the picture which we will look at next* 中的 *at* 標記為 `IN`
    - 例如：*we will look at the picture next* 中的 *at* 標記為 `IN`
    - 例如：*He doesn\'t know what he is up against.* 中的 *against* 標記為 `IN`
    - 例如：*He is up against what he doesn\'t know.* 中的 *against* 標記為 `IN`
  - 當句子中的介系詞並沒有連接任何用語時標記為 `RB` 或 `RP`
    - 例如：*We\'ll just have to do without.* 中的 *without* 標記為 `RB`
    - 例如：*We\'ll just have to do without it.* 中的 *without* 標記為 `IN`
  - 介系詞可以放在介系詞片語之前，這表示可以出現連續兩個介系詞
    - 例如：*blaze out into space* 中 *out* 與 *into* 都標記為 `IN`
    - 例如：*come out of the woodwork* 中 *out* 與 *of* 都標記為 `IN`
    - 例如：*look up to someone* 中 *up* 與 *to* 分別標記為 `IN` 與 `TO`
    - 例如：*because of her late arrival* 中 *because* 與 *of* 都標記為 `IN`
    - 例如：*to plant on into spring* 中 *on* 與 *into* 都標記為 `IN`
  - 當介系詞只能出現在名詞片語之前，卻不能出現在之後時標記為 `IN`
    - 例如：*She stepped off the train.* 中的 *off* 標記為 `IN`
  - 將介系詞後出現的名詞片語替換成代名詞後，代名詞不能出現在介系詞之前，則該介系詞標記為 `IN`
    - 例如：*She has been into it for a year.* 中的 *into* 標記為 `IN`
  - 當介系詞為單音節（monosyllabic）、出現在句尾且不表重音（stress）則標記為 `IN`
    - 例如：*Real bargains are hard to come by.* 中的 *by* 標記為 `IN`
    - 例如：*Why don\'t you come by?* 中的 *by* 標記為 `RB`
  - 以 *-ed* 或 *-ing* 的分詞（participles）作為介系詞使用時應該標記為 `VBN` 或 `VBG`
    - 例如：*Granted that he is coming* 中的 *Granted* 標記為 `VBN`
    - 例如：*Provided that he comes* 中的 *Provided* 標記為 `VBN`
    - 例如：*According to reliable sources* 中的 *According* 標記為 `VBG`
    - 例如：*Concerning your request of last week* 中的 *Concerning* 標記為 `VBG`
- 形容詞（adjective）標記為 `JJ`
  - 包含以 hyphen 組成的複合修飾詞（hyphenated compounds used as modifier）
    - 例如：*happy-go-lucky*、*one-of-a-kind*、*run-of-the-mill* 都標記為 `JJ`
    - 例如：*income-tax return* 中的 *income-tax* 標記為 `JJ`
    - 例如：*income tax return* 中的 *income tax* 標記為 `NN NN`
    - 例如：*value-added tax* 中的 *value-added* 標記為 `JJ`
    - 例如：*value added tax* 中的 *value added* 標記為 `NN VBN`
  - 包含序數（ordinal numbers）
    - 例如：*first*、*2nd* 都標記為 `JJ`
  - 不論是單一名詞或是多個名詞一起作為修飾詞使用時應標記為 `NN`
    - 例如：*wool sweater* 中的 *wool* 標記為 `NN`
    - 例如：*woollen sweater* 中的 *woollen* 標記為 `JJ`
    - 例如：*terminal type* 中的 *terminal* 標記為 `NN`
    - 例如：*terminal cancer* 中的 *terminal* 標記為 `JJ`
    - 例如：*life insurance company* 標記為 `NN NN NN`
  - 顏色（color）作為形容詞使用時標記為 `JJ`，作為名詞使用時標記為 `NN`
    - 例如：*These plants are dark green.* 中的 *green* 標記為 `JJ`
    - 例如：*These plants are a dark green.* 中的 *green* 標記為 `NN`
  - 當形容詞作為名詞使用，且該形容詞可被副詞修飾時應被標記為 `JJ`，不論有無觸發主謂一致（subject-verb agreement）現象
    - 例如：*The rich in this country pay far too few tax.* 中的 *rich* 標記為 `JJ`
    - 例如：*The very rich in this country pay far too few tax.* 中的 *rich* 標記為 `JJ`
    - 例如：*The handicapped* 中的 *handicapped* 標記為 `JJ`
    - 例如：*The multiply handicapped* 中的 *handicapped* 標記為 `JJ`
  - 當形容詞作為名詞使用，但該形容詞不能被副詞修飾時應被標記為 `NN`
    - 例如：*Little good will come of it.* 中的 *good* 標記為 `NN`
  - 方位名詞作為類名詞前置修飾詞（prenominal modifier）時標記為 `JJ`
    - 例如：*top*、*side*、*bottom*、*front*、*back*
  - 開放式複合詞作為名詞使用時應該一起標記 `JJ`
    - 例如：*mild flavored* 標記為 `JJ JJ`
  - 語言名稱或國家名稱可以作為形容詞使用，標記為 `JJ`
    - 例如：*English cuisine tends to be uninspired.* 中的 *English* 標記為 `JJ`
    - 例如：*The English tends to be uninspired cooks.* 中的 *English* 標記為 `NPS`
  - 國家以開放式複合詞（open compound）型態出現時標記一致
    - 例如：*the West German mark* 標記為 `DT JJ JJ NN`
    - 例如：*He\'s a West German .* 標記為 `PRP POS DT NP NP .`
  - 標記謂詞形容詞（predicate adjective）時應標記為 `JJ` 而非 `RB`
    - 例如：*make life simple* 中的 *simple* 標記為 `JJ`
  - 如果 *-ing* 結尾的詞能夠區分程度（gradable），則標記為 `JJ` 而非 `VBG`
    - 例如：*Her talk was interesting.* 中的 *interesting* 標記為 `JJ`
    - 例如：*Her talk was very interesting.* 中的 *interesting* 標記為 `JJ`
    - 例如：*Her talk was more interesting than theirs.* 中的 *interesting* 標記為 `JJ`
  - 以 *-ing* 結尾的詞如果存在以 *un-* 開頭的反義詞，則標記為 `JJ` 而非 `VBG`
    - 例如：*an interesting conversation* 中的 *interesting* 標記為 `JJ`
    - 例如：*an uninteresting conversation* 中的 *uninteresting* 標記為 `JJ`
  - 以 *-ing* 結尾的詞如果與 *be* 動詞一起出現，且 *be* 動詞可以替換成 *become*、*feel*、*look*、*remain*、*seem*、*sound*，則標記為 `JJ` 而非 `VBG`
    - 例如：*The conversation is depressing.* 中的 *depressing* 標記為 `JJ`
    - 例如：*The conversation became depressing.* 中的 *depressing* 標記為 `JJ`
    - 例如：*That place feels depressing.* 中的 *depressing* 標記為 `JJ`
    - 例如：*That place looks depressing.* 中的 *depressing* 標記為 `JJ`
    - 例如：*That place remains depressing.* 中的 *depressing* 標記為 `JJ`
    - 例如：*That place seems depressing.* 中的 *depressing* 標記為 `JJ`
    - 例如：*That place sounds depressing.* 中的 *depressing* 標記為 `JJ`
  - 以 *-ing* 結尾的詞如果出現在名詞前，且該 *-ing* 結尾詞的動詞原型是不及物動詞（intransitive），則標記為 `JJ` 而非 `VBG`
    - 例如：*an appealing face* 中的 *appealing* 標記為 `JJ`
    - 例如：*an appetizing dish* 中的 *appetizing* 標記為 `JJ`
    - 例如：*a revolving fund* 中的 *revolving* 標記為 `JJ`
    - 例如：*the existing safeguards* 中的 *existing* 標記為 `VBG`
    - 例如：*a holding company* 中的 *holding* 標記為 `VBG`
    - 例如：*a managing director* 中的 *managing* 標記為 `VBG`
    - 例如：*a ruling class* 中的 *ruling* 標記為 `VBG`
  - 以 *-ing* 結尾的詞如果出現在名詞前，且該 *-ing* 結尾詞改為動詞原型時表達的意思不同，則標記為 `JJ` 而非 `VBG`
    - 例如：*a winning smile* 中的 *winning* 標記為 `JJ`
    - 例如：*a striking hat* 中的 *striking* 標記為 `JJ`
    - 例如：*the striking teachers* 中的 *striking* 標記為 `VBG`
  - 以 *-ing* 結尾的詞如果沒有對應的動詞原型則標記為 `JJ`
    - 例如：*a thoroughgoing investigation* 中的 *thoroughgoing* 標記為 `JJ`
    - 例如：*the outgoing president* 中的 *outgoing* 標記為 `JJ`
    - 例如：*a outgoing type of guy* 中的 *outgoing* 標記為 `JJ`
    - 例如：*and outstanding record* 中的 *outstanding* 標記為 `JJ`
    - 例如：*outstanding debts* 中的 *outstanding* 標記為 `JJ`
  - 如果以 `VBN` 形式出現的詞能夠區分程度（gradable），則標記為 `JJ` 而非 `VBN`
    - 例如：*He was superised.* 中的 *superised* 標記為 `JJ`
    - 例如：*He was very superised.* 中的 *superised* 標記為 `JJ`
    - 例如：*He was more superised than she was.* 中的 *superised* 標記為 `JJ`
  - 以 `VBN` 形式出現的詞如果存在以 *un-* 開頭的反義形容詞（不可為動詞），則標記為 `JJ` 而非 `VBN`
    - 例如：*a hurried meeting* 中的 *hurried* 標記為 `JJ`
    - 例如：*an unhurried meeting* 中的 *unhurried* 標記為 `JJ`
    - 例如：*Your shoelace has been untied ever since we started.* 中的 *untied* 標記為 `JJ`
    - 例如：*It got untied by accident.* 中的 *untied* 標記為 `VBN`
    - 例如：*We need an armed guard.* 中的 *armed* 標記為 `JJ`
    - 例如：*We need an unarmed guard.* 中的 *unarmed* 標記為 `JJ`
    - 例如：*Armed with only a knife.* 中的 *Armed* 標記為 `VBN`
  - 以 `VBN` 形式出現的詞如果與 *be* 動詞一起出現，且 *be* 動詞可以替換成 *become*、*feel*、*look*、*remain*、*seem*、*sound*，則標記為 `JJ` 而非 `VBN`
    - 例如：*He is interested.* 中的 *interested* 標記為 `JJ`
    - 例如：*He became interested.* 中的 *interested* 標記為 `JJ`
    - 例如：*He felt interested.* 中的 *interested* 標記為 `JJ`
    - 例如：*He looked surprised.* 中的 *surprised* 標記為 `JJ`
    - 例如：*He remained surprised.* 中的 *surprised* 標記為 `JJ`
    - 例如：*He seemed surprised.* 中的 *surprised* 標記為 `JJ`
    - 例如：*He sounded surprised.* 中的 *surprised* 標記為 `JJ`
  - 以 `VBN` 形式出現的詞如果與 *be* 動詞一起出現，且 *be* 動詞可以替換成 *become*、*feel*、*look*、*remain*、*seem*、*sound*，且後續有 *by* 的出現則標記為 `VBN` 而非 `JJ`
    - 例如：*He remains guided by these principles.* 中的 *guided* 標記為 `VBN`
  - 以 `VBN` 形式出現的詞與 *keep* 一起出現時標記為 `JJ` 而非 `VBN`
    - 例如：*They should be kept well watered.* 中的 *watered* 標記為 `JJ`
  - 以 `VBN` 形式出現的詞表達結果或狀態（state or resultant state）而不是事件（event）時標記為 `JJ` 而非 `VBN`
    - 例如：*At the time, I was married.* 中的 *married* 標記為 `JJ`
    - 例如：*I was mistaken the other day.* 中的 *mistaken* 標記為 `JJ`
    - 例如：*a mistaken decision.* 中的 *mistaken* 標記為 `JJ`
  - 如果 *X-ed N* 不能被改寫成 *N that has been X-ed*，則 *X-ed* 標記為 `JJ` 而非 `VBN`
    - 例如：*a decided advantage* 中的 *decided* 標記為 `JJ`
    - 例如：*a grown woman* 中的 *grown* 標記為 `JJ`
    - 例如：*a married life* 中的 *married* 標記為 `JJ`
    - 例如：*a worried faces* 中的 *worried* 標記為 `JJ`
- 形容詞比較級（adjective，comparative）標記為 `JJR`
  - 包含以 *-er* 結尾的形容詞
  - 當 *more*、*less* 作為形容詞使用時標記為 `JJR`
    - 例如：*more or less mail* 中的 *more* 與 *less* 都標記為 `JJR`
  - 具有比較意義的形容詞但並不是以 *-er* 結尾被標記為 `JJ`
    - 例如：*superior* 標記為 `JJ`
  - 以 *-er* 結尾但嚴格上來說沒有進行比較標記為 `JJ`
    - 例如：*further details* 中的 *further* 標記為 `JJ`
- 形容詞最高級（adjective，superlative）標記為 `JJS`
  - 包含以 *-est* 結尾的形容詞
  - 包含 *worst*
  - 當 *most*、*least* 作為形容詞使用時標記為 `JJS`
  - 具有最高級意義的形容詞但並不是以 *-est* 結尾則標記為 `JJ`
    - 例如：*first*、*last*、*unsurpassed* 都標記為 `JJ`
- 列表項目符號（list item marker）標記為 `LS`
  - 包含作為項目符號的字母（letter）
  - 包含作為項目符號的數字（numerals）
- 情態動詞（modal verb）標記為 `MD`
  - 包含所有在第三人稱單數型現在式（3rd person singular present）不加上 *-s* 的動詞
    - 例如：*can*、*could*、*dare*、*may*、*might*、*must*、*ought*、*shall*、*should*、*will*、*would* 都標記為 `MD`
- 名詞單數型 / 不可數名詞（noun，singular or mass）標記為 `NN`
  - 顏色（color）作為名詞使用時標記為 `NN`
    - 例如：*That\'s a nice red.* 中的 *red* 標記為 `NN`
    - 例如：*Not too many reds go with that purple.* 中的 *reds* 標記為 `NNS`
  - 若名詞結尾為 *-s*，但觸發動詞單數型（singular agreement on a verb），則標記為 `NN` 而不是 `NNS`
    - 例如：*Linguistics is a difficult field.* 中的 *Linguistics* 標記為 `NN`
  - 語意上為複數，但觸發動詞單數型，則標記為 `NN` 而不是 `NNS`
    - 例如：*The group has disbanded.* 中的 *group* 標記為 `NN`
    - 例如：*The jury is deliberating.* 中的 *group* 標記為 `NN`
  - 大寫引用詞應標記為 `NN` 而不是 `NNP`
    - 例如：*Chapter*、*Exhibit*、*Figure*、*Table* 都標記為 `NN`
  - 不定代詞（indefinite pronoun）應該標記為 `NN` 而不是 `NNP`
    - 例如：*naught*、*none* 都標記為 `NN`
    - 例如：開頭為 *any-*、*every-*、*no-*、*some-* 的詞都標記為 `NN`
    - 例如：結尾為 *-one*、*-thing* 的詞都標記為 `NN`
    - 例如：*no one* 標記為 `DT NN`
  - *no-one* 標記為 `NN`
  - 作為副詞使用的名詞標記為名詞 而非 `RB`
    - 例如：*He comes by Sundays and holidays.* 中 *Sundays* 與 *holidays* 分別被標記為 `NPS` 與 `NNS`
  - 方位可以被標記為 `NN` 或 `RB`，判斷方法為前面有沒有冠詞出現
    - 例如：*The nearest shopping center is two miles to the north of here.* 中 *north* 標記為 `NN`
    - 例如：*The nearest shopping center is two miles north of here.* 中 *north* 標記為 `RB`
  - *yesterday*、*today*、*tomorrow* 標記為 `NN` 而不是 `RB`，理由是他們都可以補上所有格結尾
  - 以 *-ing* 結尾的詞如果允許複數型則標記為 `NN` 而不是 `VBG`
    - 例如：*The reading for this class is difficult.* 中的 *reading* 標記為 `NN`
    - 例如：*The readings for this class is difficult.* 中的 *readings* 標記為 `NNP`
  - 以 *-ing* 結尾的詞作為名詞使用，且可由形容詞修飾，則標記為 `NN` 而不是 `VBG`
    - 例如：*Good cooking is something to enjoy.* 中的 *Good cooking* 標記為 `JJ NN`
    - 例如：*Cooking well is a useful skill.* 中的 *Cooking well* 標記為 `VBG RB`
  - 以 *-ing* 結尾的詞作為名詞使用，且由 *of* 片語進行修飾，則標記為 `NN` 而不是 `VBG`
    - 例如：*GM\'s closing of the plant* 中的 *closing* 標記為 `NN`
    - 例如：*GM\'s closing the plant* 中的 *closing* 標記為 `VBG`
  - 以 *-ing* 結尾的詞出現在名詞之後時標記為 `NN` 而不是 `VBG`
    - 例如：*the plant closing* 中的 *plant closing* 標記為 `NN NN`
    - 例如：*unsavory plant closing tactics* 中的 *plant closing tactics* 標記為 `NN NN NNS`
  - 當 *X-ing N* 不含有 *N X-es* 的語意時，*X-ing* 標記為 `NN` 而不是 `VBG`
    - 例如：*spending reductions* 中的 *spending* 標記為 `NN`
    - 例如：*the mating season* 中的 *mating* 標記為 `NN`
    - 例如：*a holding pattern* 中的 *holding* 標記為 `NN`
- 專有名詞單數型（proper noun，singular）標記為 `NNP`
  - 原本標記為 `NP`，但為了避免與語法標記中的名詞片語（noun phrase）混淆改為 `NNP`
  - 縮寫應該標記成如同沒有縮寫的內容
    - 例如：*U.S.* 標記為 `NNP`
  - 當 hyphenated compound proper noun 作為修飾詞時應標記為 `NNP` 而非 `JJ`
    - 例如：*Gramm-Rudman Act* 中的 *Gramm-Rudman* 標記為 `NNP`
  - 當 hyphenated compound 中第二個位置為專有名詞時應標記為 `NNP` 而非 `JJ`
    - 例如：*mid-March* 標記為 `NNP`
    - 例如：*non-NATO* 標記為 `NNP`
- 專有名詞複數型（proper noun，plural）標記為 `NNPS`
  - 原本標記為 `NPS`，但為了避免與語法標記中的名詞片語（noun phrase）混淆改為 `NNPS`
- 名詞複數型（noun，plural）標記為 `NNS`
  - 如果名詞結尾不為 *-s*，但觸發動詞雙數型（plural agreement on a verb），則標記為 `NNS` 而不是 `NN`
    - 例如：*The faculty are on strike.* 中的 *faculty* 標記為 `NNS`
    - 例如：*The police have arrived on the scene.* 中的 *police* 標記為 `NNS`
  - 當量詞結尾為 *-s*，但觸發動詞單數型（singular agreement on a verb），則標記為 `NNS` 而不是 `NN`
    - 例如：*Three years is a long time.* 中的 *years* 標記為 `NNS`
    - 例如：*Twelve inches is a foot.* 中的 *inches* 標記為 `NNS`
- 限定詞前置詞（predeterminer）標記為 `PDT`
  - 包含所有出現在冠詞（article）之前類似於限定詞的詞
    - 例如：*both the girls* 中的 *both* 標記為 `PDT`
    - 例如：*all his marbles* 中的 *all* 標記為 `PDT`
    - 例如：*many a moon* 中的 *many* 標記為 `PDT`
    - 例如：*nary a soul* 中的 *nary* 標記為 `PDT`
    - 例如：*quite a mess* 中的 *quite* 標記為 `PDT`
    - 例如：*rather a nuisance* 中的 *rather* 標記為 `PDT`
    - 例如：*such a good time* 中的 *such* 標記為 `PDT`
  - 包含所有出現在代名詞所有格（possessive pronoun）之前類似於限定詞的詞
    - 例如：*all his marbles* 中的 *all* 標記為 `PDT`
    - 例如：*half his time* 中的 *half* 標記為 `PDT`
- 所有格結尾（possessive ending）標記為 `POS`
  - 包含單數名詞所有格結尾 *\'s*
    - 例如：*John \'s idea* 標記為 `NNP POS NN`
  - 包含複數名詞所有格結尾 *\'*
    - 例如：*the parents \' distress* 標記為 `DT NNS POS NN`
- 代名詞所有格（possessive pronoun）標記為 `PP$`
  - 包含物主形容詞（adjectival possessive pronoun）
    - 例如：*my*、*your*、*his*、*her*、*its*、*one\'s*、*our*、*their* 都標記為 `PP$`
  - 名詞性物主代詞（nominal possessive pronouns）標記為 `PRP`
- 人稱代名詞（personal pronoun）標記為 `PRP`
  - 原本標記為 `PP`，但為了避免與語法標記中的介系詞片語（prepositional phrase）混淆改為 `PRP`
  - 包含主格（subject）代名詞與受格（object）代名詞
    - 例如：*I*、*me*、*you*、*he*、*him* 都標記為 `PRP`
  - 包含以 *-self* 或 *-selves* 結尾的反身代名詞（reflexive pronoun）
  - 包含名詞性物主代詞（nominal possessive pronouns）
    - 例如：*mine*、*yours*、*his*、*hers*、*ours*、*theirs* 都標記為 `PRP`
  - 物主形容詞（adjectival possessive pronoun）標記為 `PP$`
- 副詞（adverb）標記為 `RB`
  - 包含 *-ly* 結尾的副詞
  - 包含程度詞（degree word）
    - 例如：*quite*、*too*、*very* 都標記為 `RB`
  - 包含中心詞後置修飾詞（posthead modifiers）
    - 例如：*good enough* 中的 *enough* 標記為 `RB`
    - 例如：*very well indeed* 中的 *indeed* 標記為 `RB`
  - 包含反義詞（negative markers）
    - 例如：*not*、*n\'t*、*never* 都標記為 `RB`
  - 有些形容詞會當成副詞使用且沒有 *-ly* 結尾，標記為 `RB`
    - 例如：*rapid growing plants* 標記為 `RB VBG NNS`
    - 例如：*rapid growth* 標記為 `JJ NN`
  - 地點單獨出現時應標記為 `RB` 而不是 `NN`
    - 例如：*Call me when you get home.* 中的 *home* 標記為 `RB`
    - 例如：*Call me when you are at home.* 中的 *home* 標記為 `NN`
  - 當動詞與介系詞之間能夠插入方式副詞（manner adverb）時，標記為 `RB` 而不是 `RP`
    - 例如：*to sit calmly by* 中的 *calmly by* 標記為 `RB RB`
- 副詞比較級（adverb，comparative）標記為 `RBR`
  - 副詞但結尾為 *-er*，且嚴格上來說沒有進行比較則被標記為 `RB`
    - 例如：*come by later* 中的 *later* 標記為 `RB`
- 副詞最高級（adverb，superlative）標記為 `RBS`
  - *most* 標記為 `RBS`
  - *most every-* 中的 *most* 標記為 `RB` 而不是 `RBS`
- 助詞（particle）標記為 `RP`
  - 主要由單音節（monosyllabic）的詞組成，有可能會作為方向副詞（directional adverb）或介系詞（preposition）使用
  - 如果一個介系詞能夠任意出現在名詞片語（noun phrase）之前或之後，則標記為 `RB`
    - 例如：*She told off her friends.* 中的 *off* 標記為 `RB`
    - 例如：*She told her friends off.* 中的 *off* 標記為 `RB`
  - 將介系詞後出現的名詞片語替換成代名詞後，代名詞必須改為出現在介系詞之前，則該介系詞標記為 `RB`
    - 例如：*She told them off.* 中的 *off* 標記為 `RB`
    - 例如：*He peeled it off.* 中的 *off* 標記為 `RB`
    - 註：此規則與前一規則衝突時以此規則為主
  - 如果動詞與介系詞合併後能夠成為名詞，則介系詞標記為 `RB`
    - 例如：*to break down* 中的 *down* 標記為 `RB`，因為 *breakdown* 是名詞
    - 例如：*to break through* 中的 *through* 標記為 `RB`，因為 *breakthrough* 是名詞
    - 例如：*to be left over* 中的 *over* 標記為 `RB`，因為 *leftover* 是名詞
    - 例如：*to push over* 中的 *over* 標記為 `RB`，因為 *pushover* 是名詞
    - 例如：*to put down* 中的 *down* 標記為 `RB`，因為 *putdown* 是名詞
  - 當介系詞為單音節（monosyllabic）、出現在句尾且表重音（stress）則標記為 `RB`
    - 例如：*Why don\'t you come by?* 中的 *by* 標記為 `RB`
    - 例如：*Real bargains are hard to come by.* 中的 *by* 標記為 `IN`
  - 雖然助詞與動詞經常一起處現，但也可以與動詞的變形一起出現
    - 例如：*the cutting off of the top* 中 *cutting off* 標記為 `NN RB`
    - 例如：*the setting up of the problem* 中 *setting up* 標記為 `NN RB`
    - 例如：*He looks worn out* 中 *worn out* 標記為 `JJ RP`
  - 當 *about* 與 *around* 表達 *approximately* 的語意時標記為 `RB`
  - *close to* 標記為 `RB TO`
  - *closer to* 標記為 `RB TO`
  - *near to* 標記為 `RB TO`
  - *nearer to* 標記為 `RB TO`
  - *badly off*、*better off*、*well off*、*worse off* 中的 *off* 都標記為 `RP`
- 通用科學符號（symbols）標記為 `SYM`
  - 包含數學符號、科學記號、科技符號、非英語公式等
  - 化學名稱應該被標記為 `NN`
  - 測量單位應該被標記為 `NN`
- *to* 標記為 `TO`
  - 不論是作為不定式（infinitive）或介系詞（prepositional）都採用相同標記
- 感嘆詞（interjection）標記為 `UH`
  - 包含驚嘆詞（Exclamation）
    - 例如：*oh*、*please*、*uh*、*well*、*yes* 都標記為 `UH`
  - *my* 作為感嘆詞時標記為 `UH`
    - 例如：*My, what a gorgeous day* 中的 *My* 標記為 `UH`
  - *see* 作為感嘆詞時標記為 `UH`
    - 例如：*See, it\'s like this* 中的 *See* 標記為 `UH`
- 動詞原型（verb，base form）標記為 `VB`
  - 包含命令句（imperative）的動詞
    - 例如：*Do it.* 中的 *Do* 標記為 `VB`
  - 包含接在不定式（infinitive）後的動詞
    - 例如：*You should do it.* 中的 *do* 標記為 `VB`
    - 例如：*We want them to do it.* 中的 *do* 標記為 `VB`
    - 例如：*We made them do it.* 中的 *do* 標記為 `VB`
  - 包含現在式假設語氣（subjunctive）中的動詞
    - 例如：*We suggest that he do it.* 中的 *do* 標記為 `VB`
- 動詞過去式（verb，past tense）標記為 `VBD`
  - 包含過去式假設語氣（be verb，conditional form）
    - 例如：*If I were rich, ...* 中的 *were* 標記為 `VBD`
    - 例如：*If I were to win the lottery, ...* 中的 *were* 標記為 `VBD`
- 現在分詞（gerund）標記為 `VBG`
  - 當 *X-ing N* 含有 *N X-es* 的語意時，*X-ing* 標記為 `VBG` 而不是 `NN`
    - 例如：*the declining productivity of U.S. industry* 中的 *declining* 標記為 `VBG`
    - 例如：*the acting vice president* 中的 *acting* 標記為 `VBG`
- 過去分詞（past participle）標記為 `VBN`
  - 如果以 `VBN` 形式出現的詞可以接著 *by*，且不能區分程度（gradable），則標記為 `VBN` 而非 `JJ`
    - 例如：*He was invited by some friends of hers.* 中的 *invited* 標記為 `VBN`
    - 例如：*He was very surprised by her remarks.* 中的 *surprised* 標記為 `JJ`
  - 以 `VBN` 形式出現的詞表達事件（event）而不是結果或狀態（state or resultant state）時標記為 `VBN` 而非 `JJ`
    - 例如：*I was married on a Sunday.* 中的 *married* 標記為 `VBN`
    - 例如：*I was mistaken for you the other day.* 中的 *mistaken* 標記為 `VBN`
    - 例如：*a case of mistaken identity.* 中的 *mistaken* 標記為 `VBN`
  - 以 `VBN` 形式出現的詞如果與 *be* 動詞一起出現，且 *be* 動詞可以替換成 *get*，但 *be* 動詞不能替換成 *become*，則標記為 `VBN` 而非 `JJ`
    - 例如：*I was married on a Sunday.* 中的 *married* 標記為 `VBN`
    - 例如：*I got married.* 中的 *married* 標記為 `VBN`
    - 例如：*I became married.* 中的 *married* 標記為 `JJ`
- 動詞非第三人稱單數現在式（verb，present tense，other than 3rd person singular）標記為 `VBP`
- 動詞第三人稱單數現在式（verb，present tense，3rd person singular）標記為 `VBZ`
- *wh-* 限定詞（*wh-* determiner）標記為 `WDT`
  - 出現在名詞中心語（head noun）之前的 *wh-* 詞標記為 `WDT`
    - 例如：*What kind do you want?* 中的 *What* 標記為 `WDT`
    - 例如：*I don\'t know what kind do you want.* 中的 *what* 標記為 `WDT`
    - 例如：*Be sure to wash whatever fruit you buy.* 中的 *whatever* 標記為 `WDT`
    - 例如：*Which book do you like better?* 中的 *Which* 標記為 `WDT`
    - 例如：*I don\'t know which book you like better.* 中的 *which* 標記為 `WDT`
    - 例如：*Which one do you like better?* 中的 *Which* 標記為 `WDT`
    - 例如：*I don\'t know which one you like better.* 中的 *which* 標記為 `WDT`
  - 當 *which* 或 *whichever* 沒有出現在名詞中心語之前時仍然標記為 `WDT`
    - 例如：*Which do you like better?* 中的 *Which* 標記為 `WDT`
    - 例如：*I don\'t know which you like better.* 中的 *which* 標記為 `WDT`
    - 例如：*I\'ll get you whichever you want.* 中的 *whichever* 標記為 `WDT`
  - 當 *that* 作為關係從句（relative pronoun）使用時標記為 `WDT`
- *wh-* 代名詞（*wh-* pronoun）標記為 `WP`
  - 當 *what* 或 *whatever* 沒有出現在名詞中心語（head noun）之前時標記為 `WP`
    - 例如：*Tell me what you would like to eat.* 中的 *what* 標記為 `WP`
    - 例如：*I\'ll get you whatever you want.* 中的 *whatever* 標記為 `WP`
  - 包含 *who*、*whom*
- *wh-* 代名詞所有格（possessive *wh-* pronoun）標記為 `WP$`
  - 包含 *whose*
- *wh-* 副詞（*wh-* adverb）標記為 `WRB`
  - 包含 *how*、*where*、*why*
  - 當 *when* 描述時間時標記為 `WRB`
    - 例如：*When he finally arrived, I was on my way out.* 中的 *When* 標記為 `WRB`
  - 當 *when* 作為 *if* 使用時標記為 `IN`
    - 例如：*I like it when you make dinner for me.* 中的 *when* 標記為 `IN`

### 標記流程

標記詞性總共分兩個階段，分別為自動化詞性標記（automatic POS assignment）與人工校正（manual correction）。

- 採用 PARTS 進行自動化詞性標記
  - 由 AT & T Bell Labs 開發
  - 是一種 stochastic algorithm，error rate 為 $3--5\%$
  - 基於 Brown Corpus 的標記做點修改，與 Penn Treebank 的標記風格類似
  - 輸出結果進行 tokenization 且轉換成 Penn Treebank 標記，此過程約產生 4% 誤差
    - 誤差來源包含 `VBP`
    - 誤差來源包含 adverb 與 particle
- 撰寫規則進行自動化詞性標記
  - 全靠作者的經驗進行規則編寫（沒錯就是大量的 `if-else`）
  - 將 PARTS 自動化標記過程產生的 $4\%$ 誤差修正為 $2--6\%$
- 採用圖形化介面幫助人工校正
  - 圖形化介面是由 GNU Emacs Lisp 撰寫而成，內嵌在 GNU Emacs editor
  - 使用者透過滑鼠點擊不正確的標記，並輸入正確版本（一或多個標籤）
  - 使用者的更正結果會與[圖 2](#paper-fig-2) 中的標籤進行確認，避免使用者輸入錯誤
  - 使用者的更正結果會補在原本的註記上，並標記 `*` 幫助作者分析 error rate
  - 釋出的版本會以 `*` 標記答案為主，並移除所有錯誤答案
  - 校正人員經過一個月（15 hrs per week）的時間熟悉校正流程，之後每個月的工作效率約為 3000 words per hr
- 在不使用自動化標記進行輔助時，人工標記時間比校正時間多了 2 倍，標記結果的不一致多了 2 倍，錯誤率多了 50%
  - 此實驗的參與人員共有 4 位語言學研究生，校正受訓 15 hrs，標記受訓 6 hrs
  - 所有標記人員皆熟悉 GNU Emacs
  - 標記 Brown Corpus 中 8 個 samples，每個 sample 有 2000 words
  - 從 4 個 genres（2 fictions、2 nonfictions）中各抽兩個 samples，samples 必須為標記人員沒在受試過程中接觸過
  - 標記人員首先標記 4 個 samples（順序隨機），再修正 4 個自動標記的 samples（順序隨機）
  - 不同標記人員的速度沒有顯著差異
  - 不同 genres 的標記速度沒有顯著差異
  - 標記與校正速度上有顯著差異（$\alpha = 0.05$）：校正速度中位數為 22 mins per 1k words、平均為 20 mins per 1k words；標記中位數為 42 mins per 1k words、平均為 44 mins per 1k words
  - 分析標記不一致的細節如下
    - 假設有 $k$ 個標記詞，有 $n$ 個標記者，任取 $2$ 個標記者比較所有標記結果，總共有 $\binom{n}{2}$ 種取法
    - 將 $k$ 個數取平均得到標記不一致的比例，總共有 $\binom{n}{2}$ 個比例
    - 標記任務不一致的比例平均值為 $7.2\%$，中位數為 $7.2\%$；校正任務上不一致的比例平均值為 $4.1\%$，中位數為 $3.6\%$
    - 檢視結果發現不一致的主因為化學符號，在缺乏明確的指示下標記人員隨意進行標記
    - 將化學符號造成的不一致去除後，校正任務不一致的比例平均值降為 $3.5\%$，中位數仍為 $3.6\%$
  - 分析標記答案與官方答案的差異
    - 套用分析標記不一致的方法，只是比較的對象改為官方答案
    - 標記任務不一致的比例平均值為 $5.4\%$，中位數為 $5.7\%$；校正任務上不一致的比例平均值為 $4.0\%$，中位數為 $3.4\%$
    - 將化學符號造成的不一致去除後，校正任務不一致的比例平均值降為 $3.4\%$，中位數仍為 $3.4\%$
  - 分析自動化標記與官方答案的差異
    - 套用分析標記答案與官方答案的方法，只是標記答案是自動化產生
    - 標記不一致的比例平均值為 $9.6\%$
    - 人工校正能夠減少 $4.2\%$ 的標記錯誤，說明人工校正是必要的

## 語法成份分析標記

與詞性標記平行進行的語法成份分析（constituent parse）標記流程為 Fidditch 自動化標記加上人工校正而得。

### Fidditch

首先使用 Donald Hindle 在 University of Pennsylvania 與 AT & T Bell Labs 開發的 Fidditch parser 進行自動化標記。
Fidditch 的特性如下：

- Fidditch 永遠只會給出一種答案
  - 因此校正過程不需要分析多個答案
- 當 Fidditch 無法在非常確定的前提下，判斷句子中的部份 constituent 在更大的 constituent 中的定位時，不會給出標記
  - 這代表 Fidditch 的輸出可能不是完整 constituent
  - 校正人員需要把已經分析完的 constituent 組成更大更完整的 constituent，作者說這就好像在把constituent 「黏」（glue）在一起的感覺
  - Fidditch 的表現不錯，因此只要是在非常確定的前提下產出的 constituent 幾乎都是正確的
- 無法判斷的結果 Fidditch 會以 `?` 標記
  - Fidditch 為 syntatic parser，沒有考慮 semantic or pragmatic information
  - 由於判斷 prepositional phrases、relative clauses、adverbial modifiers 時需要考慮資訊遠超過語法（extrasyntatic information），因此 Fidditch 會將這些 constituent 不連結至任何上層 constituent（leaving such constituents unattached），而校正工作就是正確連結未標記的 constituent

### 標記總表

<a name="paper-fig-3"></a>

圖 3：Penn Treebank 語法成份分析標記總表。
圖片來源：[論文][論文]。

![圖 3](https://i.imgur.com/czVasbA.png)

- 與 Lancaster Treebank Project 標記種類相似，但 Penn Treebank 允許 null element
- Null elements 包含了 Fidditch 產出的標記
- 作者認為保留 null element 能夠分析 predicate-argument structure 與 verb transitivity
- 作者認為 Penn Treebank 是以 **context-free grammar** 作為前提進行標記，因此必須保留 null element

### 標記流程

根據觀察發現

- 語法成份分析標記所需的時間遠大於詞性標記時間
- 進行 Fidditch 輸出校正所需時間為
  - 每小時約 375 個詞（受訓 3 星期）
  - 每小時約 475 個詞（受訓 6 星期）
- 將 Fidditch 的輸出進一步化簡成類似於 Lancaster UCREL Treebank Project 的風格讓校正速度變快（每小時校正增加約 100-200 個詞）
  - 移除詞性標記
  - 移除 nonbranching lexical nodes
  - 移除特定 phrase nodes（主要為 `NBAR`）
  - 當 `ADJP` 出現在 `NP` 內，且 `ADJP` 不是 coordinate structure 的一部份時，移除 `ADJP`
- 由於 verb\'s arguments 與 verb\'s adjuncts 很難區分，當允許校正人員忽略差異時讓校正速度變快（每小時校正增加約 150-200 個詞）

因此作者決定標記採用**化簡風格**（**skeletal syntactic structure**），且不強制區分 verb\'s arguments 與 verb\'s adjuncts。

- 共 5 位兼職標記人員，平均每人每小時校正約 750 個詞
- 最有效率的標記人員每小時校正約 1500 個詞，且中間還穿插短暫休息時間
- 每天工作 3 小時，估算一年能標記約 2.5M 個詞
- 有經驗的標記者能夠非常快速的驗證標記結果，每人每小時約可驗證 4000 個詞

為了增加標記效率，Penn Treebank 額外使用了兩種標記：

- `X` constituent
  - 當標記者認為某個片段為 constituent，但不確定應該標記成什麼，則可以暫時寫為 `X` constituent
- Pseudo-attachment
  - 在給予前後文的情況下仍然無法確定一個 constituent 應該被連結到哪一個上層的結構時稱為 permanent predictable ambiguities，並標記為 pseudo-attachment
  - 有些情況中一個 constituent 修飾的對象不只一個，此情況出現時可以使用 pseudo-attachment 標記，但絕大多數情況標記都是以 context-free grammar 出發，即一個 constituent 只能連接一個上層 constituent

## 標記結果

<a name="paper-fig-4"></a>

圖 4：Penn Treebank 標記結果。
圖片來源：[論文][論文]。

![圖 4](https://i.imgur.com/JVj3svh.png)

- 所有資料都可由 [LDC][LDC] 取得
- Department of Energy abstracts 是科學研究的摘要
- The Department of Agriculture materials 包含種花的時機、如何將蔬菜水果做成罐頭等的文章
- The Library of America texts 的文章段落約為 5000-10000 詞，主要都是書本的章節內容
- MUC-3 都是 Federal News Service 的新聞內容，主題為南美的恐怖份子行動，包含部份西班牙文新聞翻譯
- IBM Manual sentences 是 IBM 電腦的參考手冊，字典大小約為 3000
- ATIS sentences 是由 DARPA Air Travel Information System project 轉錄而得
- Brown Corpus 完全依照 Penn Treebank 的規則重新進行標記
- 作者估算 POS 標記誤差為 $3\%$

[LDC]: https://catalog.ldc.upenn.edu/LDC93T1
[論文]: https://alliance.seas.upenn.edu/~nlp/publications/pdf/marcus1993.pdf
[詞性標記參考手冊]: https://repository.upenn.edu/cgi/viewcontent.cgi?article=1603&context=cis_reports
[語法成份分析標記參考手冊]: http://languagelog.ldc.upenn.edu/myl/PennTreebank1995.pdf
