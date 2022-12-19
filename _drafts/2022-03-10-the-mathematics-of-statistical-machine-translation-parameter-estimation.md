---
layout: ML-note
title: "The Mathematics of Statistical Machine Translation: Parameter Estimation"
date: 2022-03-09 16:46:00 +0800
categories: [
  Text Modeling,
]
tags: [
  alignment,
  machine translation,
  model architecture,
]
author: [
  Peter F. Brown,
  Vincent J. Della Pietra,
  Stephen A. Della Pietra,
  Robert L. Mercer,
]
---

|-|-|
|目標|提出以 word alignment 作為特徵進行機器翻譯|
|作者|Peter F. Brown, Vincent J. Della Pietra, Stephen A. Della Pietra, Robert L. Mercer|
|隸屬單位|IBM T.J. Watson Research Center|
|期刊/會議名稱|Computational Linguistics|
|發表時間|1993|
|論文連結|<https://aclanthology.org/J93-2003/>|

<!--
  Define LaTeX command which will be used through out the writing.

  Each command must be wrapped with $ signs.
  We use "display: none;" to avoid redudant whitespaces.
 -->

<p style="display: none;">

  <!-- e but bold. -->
  $\newcommand{\eb}{\mathbf{e}}$
  <!-- f but bold. -->
  $\newcommand{\fb}{\mathbf{f}}$
  <!-- Probability of length of f. -->
  $\newcommand{\PrLenF}{\Pr_{f\operatorname{-length}}}$
  <!-- Probability of position alignment. -->
  $\newcommand{\PrPosAl}{\Pr_{\operatorname{pos-align}}}$

</p>

<!-- End LaTeX command define section. -->

## 重點

- 只實驗英文與法文之間的機器翻譯
  - 受限於資料集
  - 作者認為因為沒有使用太多的語言學特徵，所以在有資料時可以輕易的套用到其他語系

## 想法

- SMT 作法為列舉所有可能的英文序列，並找出機率最高的作為答案；而研究人員認為應該要先真正了解一個語言後再進行精確翻譯（p3）
- Language model + Translation model + searching algorithm
- $\Pr_{\eb \text{ must be well-formed}}(\eb \vert \fb) \propto \Pr(\eb) \times \Pr_{\eb \text{ can be ill-formed}}(\fb \vert \eb)$

### Model 1

#### 訓練

給予翻譯配對 $(\eb, \fb)$，目標為最大化 $\fb$ 翻譯成 $\eb$ 的機率值 $\Pr(\eb \vert \fb)$，達成的方法為最大化語言模型機率值 $\Pr(e)$ 以及翻譯模型機率值 $\Pr(\fb \vert \eb)$，即

$$
\Pr(\eb \vert \fb) \propto \Pr(\eb) \times \Pr(\fb \vert \eb)
$$

令 $\eb$ 長度為 $\ell$，$\fb$ 長度為 $m$。

- 假設 $\fb$ 的長度 $m$ 只能落在事先定義的長度範圍 $1, \dots, M$，並賦與 $\fb$ 的長度 uniform distribution，即 $\PrLenF(m) = \dfrac{1}{M}$
- 假設 $\fb$ 中的位置 $j \in \set{1, \dots, m}$ 對應到 $\eb$ 中的任意位置 $i \in \set{1, \dots, \ell}$ 機率都相同，即 $\PrPosAl(i, j) = \dfrac{1}{\ell}$
  - 由此假設可知翻譯文字順序不重要
  - 此假設不夠實際，後續的方法會替換此假設
- 擁有 unique local maximum，將訓練後的參數用來初始化 Model 2
- 後續的模型都沒有 unique local maximum
- 後續的模型都基於改良的模型訓練後的參數進行初始化

### Model 2

基於 Model 1 提出以下更改

- 不再假設 $\PrPosAl(i, j)$ 為 uniform distribution

### Model 3

- 先決定 $\eb$ 中的每個英文字 $\eb_i$ 是由多少個 $\fb$ 中的文字翻譯而得，數值介於 $0, \dots, m$
  - $0$ 代表 $\fb$ 中沒有文字翻譯成 $\eb_i$，常見於功能字（function word）翻譯
  - $m$ 代表 $\fb$ 中的所有文字都翻譯成 $\eb_i$，此狀況不太可能發生
- 決定完 $\eb_i$ 是由多少個文字翻譯而得後，接著決定具體是由哪幾個 $\fb$ 中的文字翻譯而得
  - 有幾個跟哪幾個是不一樣的問題
  - 哪幾個又再細分成哪些字跟哪些位置，因為同一句話內可以出現多個相同的文字但對應到不同翻譯位置

## 模型

### 概念

機器翻譯模型的目標是在給予一個法文句子 $f$ 時找到機率值最大的英文句子 $e$，根據貝式定理可以轉換成以下公式：

$$
\begin{align*}
& \Pr(e \vert f) = \frac{\Pr(e) \Pr(f \vert e)}{\Pr(f)} \\
\implies & \argmax_e \Pr(e \vert f) = \argmax_e \frac{\Pr(e) \Pr(f \vert e)}{\Pr(f)} = \argmax_e \Pr(e) \Pr(f \vert e)
\end{align*} \tag{1}\label{1}
$$

因此翻譯任務取決於 $3$ 個任務的合作結果

- 尋找配對（searching problem）：給予法文 $f$，找到適合的英文配對 $e$
  - 由於可能的英文句子數量太多，因此必須有效率的找到配對
  - 此論文將以單字（word）為出發點，後續的論文改用片語（phrase）進行計算
- 翻譯模型（translation model）$\Pr(f \vert e)$：計算任意英法文的配對 $(e, f)$ 的機率值
  - 此論文的探討重點
  - 與語言模型不同，探討的句子 $e$ 可以不用受限於句法結構，這也是為什麼需要額外的語言模型幫忙檢視句法結構
  - 此論文將以單字（word）為出發點，後續的論文改用片語（phrase）進行計算
- 語言模型（language model）$\Pr(e)$：決定英文翻譯結果的可能性
  - 跟同時期的語音辨識研究採用的演算法相同
  - 基於 $n$-gram 進行建模

### Alignment

<a name="paper-fig-1"></a>

圖 1：Alignment 演算法示意圖。
圖片來源：[論文][論文]。

![圖 1](https://i.imgur.com/TlOV5H5.png)

<a name="paper-fig-2"></a>

圖 2：Alignment 演算法示意圖。
圖片來源：[論文][論文]。

![圖 2](https://i.imgur.com/yv2DDcv.png)

<a name="paper-fig-3"></a>

圖 3：Alignment 演算法示意圖。
圖片來源：[論文][論文]。

![圖 3](https://i.imgur.com/mwi6mgp.png)

給予一個英文單字序列 $e$ 與法文單字序列 $f$，alignment 演算法的主要精神是將 $e$ 中的各個單字配對給一個 $f$ 中的單字。

對應在一起的單字在視覺化時會以線條相連（稱為 connection），如[圖 1](#paper-fig-1) 中的（the, Le）與（has, a）。

不是所有的單字都可以進行配對，如[圖 1](#paper-fig-1) 中的範例 And 就沒有對應的單字。

配對並沒有如同函數限制，可以是一對一、一對多甚至多對多的配對，如[圖 3](#paper-fig-3) 中的（have, sont）、（have, demunis）、（any, sont）與（any, demunis）。

[論文]: https://aclanthology.org/J93-2003/
