---
layout: ML-note
title: "The Penn Treebank: Annotating Predicate Argument Structure"
date: 2022-08-19 22:26:00 +0800
categories: [
  Dataset,
]
tags: [
  Penn Treebank,
  part of speech,
  constituent parse,
  predicate argument structure,
  semantic role labeling,
]
author: [
  Mitchell Marcus,
  Grace Kim,
  Mary Ann Marcinkiewicz,
  Robert MacIntyre,
  Ann Bies,
  Mark Ferguson,
  Karen Katz,
  Britta Schasberger,
]
---

|-|-|
|目標|在 Penn Treebank 上額外標記 text categories、predicate argument structure、grammatical functions、semantic roles|
|作者|Mitchell Marcus, Grace Kim, Mary Ann Marcinkiewicz, Robert MacIntyre, Ann Bies, Mark Ferguson, Karen Katz, Britta Schasberger|
|隸屬單位|University of Pennsylvania|
|期刊/會議名稱|Human Language Technology|
|發表時間|1994|
|論文連結|<https://aclanthology.org/H94-1020/>|
|參考手冊|<https://www.ldc.upenn.edu/sites/www.ldc.upenn.edu/files/penn-etb-2-style-guidelines.pdf>|

## 重點

- 此論文是 Penn Treebank project 的第二階段
  - [第一階段][PTB1]已經標記 part of speech 與 constituent
  - 第一階段已經蒐集大量的資料（over 1 million words）
  - 新的標記資料可在 [LCD][PTB2] 下載
- 由於 Penn Treebank 在研究上的成效卓越，成功替代了 Brown Corpus 作為英文計算語言學主要研究資料集，同時進一步擴充標記內容
  - 增加 predicate-argument structure 標記
  - 新增加的標記啟蒙了後續的 SemEval 比賽內容
- 為了有效的標記 predicate argument structure，且不破壞 treebank context-free 的結構，作者提出以下手段進行標記
  - Null elements
  - Co-index
  - Pseudo-attach

## 新的標記準則

- 作者花了非常大量的時間撰寫新的標記準則，並編寫成參考手冊（style-book）
  - 主要目的是讓所有標記人員在標記資料上達成共識，增加標記的一致性
  - 希望標記人員能夠透過討論標記手冊的內容更加了解標記流程與規則，並藉此提高標記資料單位時間的產出（throughput）
  - 不同標記人員取得的資料中擁有 10% 的重疊，以此確保標記的一致性（之前 POS 與 constituent 的標記並沒有此規則）
  - 標記時間共花了 8 個月，每個禮拜標記工時為 10 小時
- 新的標記準則希望能夠達成 4 個目標
  - 讓標記具有一致性
  - 描述 null element 的使用方法
  - 提供 non-context free 標記方法
  - 清楚的區分 verb arguments and adjuncts

## Predicate Argument Structure

原始 Penn Treebank 每個 constituent 只有一個標記，並只與語法有關。
明顯的缺點就是當一個 constituent 語法上有明確分類，但語意上卻作為其他功能使用（例如：logical subject）時無法清楚標記。
新版的標記允許同一個 constituent 最多同時擁有 4 個不同的標記，包含 standard syntactic labels、text categories、grammatical functions 與 semantic roles。

### Text Categories

|tag|meaning|
|-|-|
|`-HLN`|headlines and datelines|
|`-LST`|list markers|
|`-TTL`|titles|

### Grammatical Functions

|tag|meaning|
|-|-|
|`-CLF`|true clefs|
|`-NOM`|non `NP`s that function as `NP`s|
|`-ADV`|clausal and `NP` adverbials|
|`-LGS`|logical subjects in passives|
|`-PRD`|non `VP` predicates|
|`-SBJ`|surface subject|
|`-TPC`|topicalized and fronted constituents|
|`-CLR`|closed related|

- 當 `NP` 或 `S` 明確的作為 predicate argument 使用時則不需任何標記
  - 常見的 predicate argument 有兩種
    - Predicate argument 是 lowest（right most branching）`VP`
    - Predicate argument 是 copular BE 的 immediately subconstituent
  - 如果不是以上狀況則額外使用 `-PRD` 進行標記
- 當 constituent 明確的作為 predication adjunction 使用時標記為 `-CLR`
- 作者說在實際上區隔 argument 與 adjunction 是非常困難的

例如：

{% highlight text %}
I am happy .

(S
  (NP-SBJ I)
  (VP am
    (ADJP happy)
  )
  .
)

Predicate Argument Structure:
be(I, happy)
{% endhighlight %}

{% highlight text %}
I consider Kris a fool .

(S
  (NP-SBJ I)
  (VP consider
    (S
      (NP-SBJ Kris)
      (NP-PRD a fool)
    )
  )
  .
)

Predicate Argument Structure:
consider(I, a fool)
{% endhighlight %}

{% highlight text %}
Was he ever successful ?

(SQ Was
  (NP-SBJ he)
  (ADVP-TMP ever)
  (ADJP-PRD successful)
  ?
)

Predicate Argument Structure:
be(he, successful)
{% endhighlight %}

### Semantic Roles

|tag|meaning|
|-|-|
|`-VOC`|vocatives|
|`-DIR`|direction & trajectory|
|`-LOC`|location|
|`-MNR`|manner|
|`-PRP`|purpose and reason|
|`-TMP`|temporal phrase|


## Null Elements + Co-indexing

- 使用 `*T*` 標記 wh-movement 與 topicalization
- 使用 `*` 標記其他 null elements
- 所有的 null elements 都會補上數字（稱為 co-indexing）連結替代的 constituents

例如：

{% highlight text %}
What is Tim eating ?

(SBARQ
  (WHNP-1 What)
  (SQ is
    (NP-SBJ Tim)
    (VP eating
      (NP *T*-1)
    )
  )
  ?
)

Predicate Argument Structure:
eat(Tim, what)
{% endhighlight %}

表達 passives（被動語氣）時，surface subject 會標記為 `-SBJ`，並加入提示 `(NP *)`，且與 surface subject 共享數字（co-indexing）。
而 logical subject 會標記為 `-LGS`。
例如：

{% highlight text %}
The ball was thrown by Chris .

(S
  (NP-SBJ-1 The ball)
  (VP was
    (VP thrown
      (NP *-1)
      (PP by
        (NP-LGS Chris)
      )
    )
  )
  .
)

Predicate Argument Structure:
throw(Chris, ball)
{% endhighlight %}

{% highlight text %}
Who was believed to have been shot ?

(SBARQ
  (WHNP-1 Who)
  (SQ was
    (NP-SBJ-2 *T*-1)
    (VP believed
      (S
        (NP-SBJ-3 *-2)
        (VP to
          (VP have
            (VP been
              (VP shot
                (NP *-3)
              )
            )
          )
        )
      )
    )
  )
  ?
)

Predicate Argument Structure:
believe(*someone*, shoot(*someone*, Who))
{% endhighlight %}

Null elements 也用來作為 null subject of infinitive complement clause。
例如：

{% highlight text %}
Chris wants to throw the ball .

(S
  (NP-SBJ-1 Chris)
  (VP wants
    (S
      (NP-SBJ *-1)
      (VP to
        (VP throw
          (NP the ball)
        )
      )
    )
  )
  .
)

Predicate Argument Structure:
want(Chris, throw(Chris, ball))
{% endhighlight %}

當 argument 被強調（topicalized）導致順序掉換時使用 null element 表達原本的結構，而 adjunct 被強調時則不使用 null element 做任何標記。
例如：

{% highlight text %}
This every man contains within him .

(S
  (NP-TPC-1 This)
  (NP-SBJ every man)
  (VP contains
    (NP *T*-1)
    (PP-LOC within
      (NP him)
    )
  )
  .
)

Predicate Argument Structure:
contains(every man, this)
{% endhighlight %}

當 argument 是由 `VP` 組成，且因 topicalized 導致順序掉換時，使用 null element 表達原本的結構。
例如：

{% highlight text %}
Marching past the reviewing stand were 500 musicians .

(SINV
  (VP-TPC-1 Marching
    (PP-CLR past
      (NP the reviewing stand)
    )
  )
  (VP were
    (VP *T*-1)
  )
  (NP-SBJ 500 musicians)
  .
)

Predicate Argument Structure:
be(500 musicians, marching)
{% endhighlight %}

## Pseudo-Attach

由於 Penn Treebank 的標記方法屬於 context-free，因此部份資訊無法在此標記架構下清楚表達。
主要的例子是出現在 verb + sentential adverb 之後，做為補語使用的 constituents（constituents which serve as complements to the verb occur after a sentential level adverb）。
這類型的 constituents 有兩種標記方法：

- 將 adverb 留在 `VP` 內，導致 constituents 一起留在 `VP` 之中
- 將 `VP` 結束，獨立出 adverb，導致 constituents 與 `VP` 平行

此類現象通稱為 discontinuous constituents 或 trapping problems。
作者認為可以透過 co-index + null element 進行標記解決 discontinuous constituents 的現象。
由於該現象的特殊性，作者決定給予獨特的標記，稱為 pseudo-attach。

|tag|meaning|
|-|-|
|`*ICH*`|interpret constituent here|
|`*PPA*`|permanent predictable ambiguity|
|`*RNR*`|right node raising|
|`*EXP*`|expletive|

`*ICH*` 就是專門用來解決簡單版 discontinuous constituents 的標記。
例如：

{% highlight text %}
Chris knew yesterday that Terry would catch the ball .

(S
  (NP-SBJ Chris)
  (VP knew
    (SBAR *ICH*-1)
    (NP-TMP yesterday)
    (SBAR-1 that
      (S
        (NP-SBJ Terry)
        (VP would
          (VP catch
            (NP the ball)
          )
        )
      )
    )
  )
  .
)

Predicate Argument Structure:
know(Chris, that)
{% endhighlight %}

`*PPA*` 是專門用來表達永遠無法判斷 subconstituent 要連結到哪個 constituent 的狀態。
例如：

{% highlight text %}
I saw the man with the telescope .

(S
  (NP-SBJ I)
  (VP saw
    (NP
      (NP the man)
      (PP *PPA*-1)
    )
    (PP-CLR-1 with
      (NP the telescope)
    )
  )
  .
)

Predicate Argument Structure:
see(I, the man)
{% endhighlight %}

當使用連接詞（conjunction）連接不同片段（conjuncts），且將共同的 constituent 提出時，使用 `*RNR*` 進行標記。
例如：

{% highlight text %}
But our outlook has been , and continues to be , defensive .

(S But
  (NP-SBJ-2 our outlook)
  (VP
    (VP has
      (VP been
        (ADJP *RNR*-1)
      )
    )
    ,
    and
    (VP continues
      (S
        (NP-SBJ *-2)
        (VP to
          (VP be
            (ADJP *RNR*-1)
          )
        )
      )
    )
    ,
    (ADJP-1 defensive)
  )
  .
)

Predicate Argument Structure:
be(our outlook, defensive)
continues(our outlook, be(our outlook, defensive))
{% endhighlight %}

由於用 It 作為句子開頭進行強調的用法很常見，作者決定給與特別標籤 `*EXP*`。
例如：

{% highlight text %}
It is a pleasure to teach her .

(S
  (NP-SBJ
    (NP It)
    (S *EXP*-1)
  )
  (VP is
    (NP a pleasure)
  )
  (S-1
    (NP-SBJ *)
    (VP to
      (VP teach
        (NP her)
      )
    )
  )
  .
)

Predicate Argument Structure:
pleasure(teach(*someone*, her))
{% endhighlight %}

## Conjunction and Gapping

作者採用 Chomsky adjunction structure 標記 coordination。
Word level conjuntion 不給予標記，兩個字以上的 conjunction 才會給予明確標記。

{% highlight text %}
Terry knew the person who threw the ball and who caught it .

(S
  (NP-SBJ Terry)
  (VP knew
    (NP
      (NP the person)
      (SBAR
        (SBAR
          (WHNP-1 who)
          (S
            (NP-SBJ T-1)
            (VP threw
              (NP the ball)
            )
          )
        )
        and
        (SBAR
          (WHNP-2 who)
          (S
            (NP-SBJ T-2)
            (VP caught
              (NP it)
            )
          )
        )
      )
    )
  )
  .
)

Predicate Argument Structure:
know(Terry, person(
  and(
    throw(*who*, ball),
    catch(*who*, it)
  )
))
{% endhighlight %}

當遇到 conjuntion 共享 predicate 時採用 `=` 進行標記。
例如：

{% highlight text %}
John gave Mary a book and Bill a pencil .

(S
  (S
    (NP-SBJ John)
    (VP gave
      (NP-1 Mary)
      (NP-2 a book)
    )
  )
  and
  (S
    (NP=1 Bill)
    (NP=2 a pencil)
  )
  .
)

Predicate Argument Structure:
give(John, Mary, a book) and give(John, Bill, a pencil)
{% endhighlight %}

{% highlight text %}
I eat breakfast in the morning and lunch in the afternoon .

(S
  (S
    (NP-SBJ I)
    (VP eat
      (NP-1 breakfast)
      (PP-TMP-2 in
        (NP the morning)
      )
    )
  )
  and
  (S
    (NP=1 lunch)
    (PP-TMP=2 in
      (NP the afternoon)
    )
  )
  .
)

Predicate Argument Structure:
eat(I, breakfast) and eat(I, lunch)
{% endhighlight %}

在口語上的共享詞時常省略且必須依賴前後文還原。
作者認為在此狀態下不需進行標記還原，並給予特殊標記 `FRAG` 表示無法取得 predicate argument structure。
例如：

{% highlight text %}
Who threw the ball ? Chris , yesterday .

(FRAG
  (NP Chris)
  ,
  (NP-TMP yesterday)
  .
)
{% endhighlight %}

[論文]: https://aclanthology.org/H94-1020/
[PTB1]: https://catalog.ldc.upenn.edu/LDC93T1
[PTB2]: https://catalog.ldc.upenn.edu/LDC95T7
