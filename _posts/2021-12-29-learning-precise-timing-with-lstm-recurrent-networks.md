---
layout: ML-note
title: "Learning Precise Timing with LSTM Recurrent Networks"
date: 2021-12-29 16:28:00 +0800
categories: [
  Model Architecture,
  Neural Network,
]
tags: [
  RNN,
  LSTM,
]
author: [
  Felix A. Gers,
  Nicol N. Schraudolph,
  Jürgen Schmidhuber,
]
---

|-|-|
|目標|在 LSTM 上加入 peephole connections|
|作者|Felix A. Gers, Nicol N. Schraudolph, Jürgen Schmidhuber|
|隸屬單位|IDSIA|
|期刊/會議名稱|JMLR, Volume 3|
|發表時間|2002|
|論文連結|<https://www.jmlr.org/papers/v3/gers02a.html>|

<!--
  Define LaTeX command which will be used through out the writing.

  Each command must be wrapped with $ signs.
  We use "display: none;" to avoid redudant whitespaces.
 -->

<p style="display: none;">

  <!-- Operator in. -->
  $\providecommand{\opnet}{}$
  $\renewcommand{\opnet}{\operatorname{net}}$
  <!-- Operator in. -->
  $\providecommand{\opin}{}$
  $\renewcommand{\opin}{\operatorname{in}}$
  <!-- Operator out. -->
  $\providecommand{\opout}{}$
  $\renewcommand{\opout}{\operatorname{out}}$
  <!-- Operator cell block. -->
  $\providecommand{\opblk}{}$
  $\renewcommand{\opblk}{\operatorname{block}}$
  <!-- Operator cell multiplicative forget gate. -->
  $\providecommand{\opfg}{}$
  $\renewcommand{\opfg}{\operatorname{fg}}$
  <!-- Operator cell multiplicative input gate. -->
  $\providecommand{\opig}{}$
  $\renewcommand{\opig}{\operatorname{ig}}$
  <!-- Operator cell multiplicative output gate. -->
  $\providecommand{\opog}{}$
  $\renewcommand{\opog}{\operatorname{og}}$
  <!-- Operator sequence. -->
  $\providecommand{\opseq}{}$
  $\renewcommand{\opseq}{\operatorname{seq}}$
  <!-- Operator loss. -->
  $\providecommand{\oploss}{}$
  $\renewcommand{\oploss}{\operatorname{loss}}$
  <!-- Operator tri. -->
  $\providecommand{\optri}{}$
  $\renewcommand{\optri}{\operatorname{tri}}$
  <!-- Operator rect. -->
  $\providecommand{\oprect}{}$
  $\renewcommand{\oprect}{\operatorname{rect}}$
  <!-- Operator mod. -->
  $\providecommand{\opmod}{}$
  $\renewcommand{\opmod}{\operatorname{mod}}$

  <!-- Net input. -->
  $\providecommand{\net}{}$
  $\renewcommand{\net}[2]{\opnet_{#1}(#2)}$
  <!-- Net input with activatiton f. -->
  $\providecommand{\fnet}{}$
  $\renewcommand{\fnet}[2]{f_{#1}\big(\net{#1}{#2}\big)}$
  <!-- Derivative of f with respect to net input. -->
  $\providecommand{\dfnet}{}$
  $\renewcommand{\dfnet}[2]{f_{#1}'\big(\net{#1}{#2}\big)}$

  <!-- Input dimension. -->
  $\providecommand{\din}{}$
  $\renewcommand{\din}{d_{\opin}}$
  <!-- Output dimension. -->
  $\providecommand{\dout}{}$
  $\renewcommand{\dout}{d_{\opout}}$
  <!-- Cell block dimension. -->
  $\providecommand{\dblk}{}$
  $\renewcommand{\dblk}{d_{\opblk}}$

  <!-- Number of cell blocks. -->
  $\providecommand{\nblk}{}$
  $\renewcommand{\nblk}{n_{\opblk}}$

  <!-- Cell block k. -->
  $\providecommand{\blk}{}$
  $\renewcommand{\blk}[1]{\opblk^{#1}}$

  <!-- Weight of multiplicative forget gate. -->
  $\providecommand{\wfg}{}$
  $\renewcommand{\wfg}{w^{\opfg}}$
  $\providecommand{\ufg}{}$
  $\renewcommand{\ufg}{u^{\opfg}}$
  <!-- Weight of multiplicative input gate. -->
  $\providecommand{\wig}{}$
  $\renewcommand{\wig}{w^{\opig}}$
  $\providecommand{\uig}{}$
  $\renewcommand{\uig}{u^{\opig}}$
  <!-- Weight of multiplicative output gate. -->
  $\providecommand{\wog}{}$
  $\renewcommand{\wog}{w^{\opog}}$
  $\providecommand{\uog}{}$
  $\renewcommand{\uog}{u^{\opog}}$
  <!-- Weight of cell units. -->
  $\providecommand{\wblk}{}$
  $\renewcommand{\wblk}[1]{w^{\blk{#1}}}$
  <!-- Weight of output units. -->
  $\providecommand{\wout}{}$
  $\renewcommand{\wout}{w^{\opout}}$

  <!-- Net input of multiplicative forget gate. -->
  $\providecommand{\netfg}{}$
  $\renewcommand{\netfg}[2]{\opnet_{#1}^{\opfg}(#2)}$
  <!-- Net input of multiplicative forget gate with activatiton f. -->
  $\providecommand{\fnetfg}{}$
  $\renewcommand{\fnetfg}[2]{f_{#1}^{\opfg}\big(\netfg{#1}{#2}\big)}$
  <!-- Derivative of f with respect to net input of forget gate. -->
  $\providecommand{\dfnetfg}{}$
  $\renewcommand{\dfnetfg}[2]{f_{#1}^{\opfg}{'}\big(\netfg{#1}{#2}\big)}$
  <!-- Net input of multiplicative input gate. -->
  $\providecommand{\netig}{}$
  $\renewcommand{\netig}[2]{\opnet_{#1}^{\opig}(#2)}$
  <!-- Net input of multiplicative input gate with activatiton f. -->
  $\providecommand{\fnetig}{}$
  $\renewcommand{\fnetig}[2]{f_{#1}^{\opig}\big(\netig{#1}{#2}\big)}$
  <!-- Derivative of f with respect to net input of input gate. -->
  $\providecommand{\dfnetig}{}$
  $\renewcommand{\dfnetig}[2]{f_{#1}^{\opig}{'}\big(\netig{#1}{#2}\big)}$
  <!-- Net input of multiplicative output gate. -->
  $\providecommand{\netog}{}$
  $\renewcommand{\netog}[2]{\opnet_{#1}^{\opog}(#2)}$
  <!-- Net input of multiplicative output gate with activatiton f. -->
  $\providecommand{\fnetog}{}$
  $\renewcommand{\fnetog}[2]{f_{#1}^{\opog}\big(\netog{#1}{#2}\big)}$
  <!-- Derivative of f with respect to net input of output gate. -->
  $\providecommand{\dfnetog}{}$
  $\renewcommand{\dfnetog}[2]{f_{#1}^{\opog}{'}\big(\netog{#1}{#2}\big)}$
  <!-- Net input of output units. -->
  $\providecommand{\netout}{}$
  $\renewcommand{\netout}[2]{\opnet_{#1}^{\opout}(#2)}$
  <!-- Net input of output units with activatiton f. -->
  $\providecommand{\fnetout}{}$
  $\renewcommand{\fnetout}[2]{f_{#1}^{\opout}\big(\netout{#1}{#2}\big)}$
  <!-- Derivative of f with respect to net input of output units. -->
  $\providecommand{\dfnetout}{}$
  $\renewcommand{\dfnetout}[2]{f_{#1}^{\opout}{'}\big(\netout{#1}{#2}\big)}$

  <!-- Net input of cell unit. -->
  $\providecommand{\netblk}{}$
  $\renewcommand{\netblk}[3]{\opnet_{#1}^{\blk{#2}}(#3)}$
  <!-- Net input of cell unit with activatiton g. -->
  $\providecommand{\gnetblk}{}$
  $\renewcommand{\gnetblk}[3]{g_{#1}\big(\netblk{#1}{#2}{#3}\big)}$
  <!-- Derivative of g with respect to net input of cell unit. -->
  $\providecommand{\dgnetblk}{}$
  $\renewcommand{\dgnetblk}[3]{g_{#1}'\big(\netblk{#1}{#2}{#3}\big)}$
  <!-- Cell unit with activatiton h. -->
  $\providecommand{\hblk}{}$
  $\renewcommand{\hblk}[3]{h_{#1}\big(s_{#1}^{\blk{#2}}(#3)\big)}$

  <!-- Gradient approximation by truncating gradient. -->
  $\providecommand{\aptr}{}$
  $\renewcommand{\aptr}{\approx_{\operatorname{tr}}}$
</p>

<!-- End LaTeX command define section. -->

## 重點

- [原版 LSTM][LSTM1997] 與 [LSTM-2000][LSTM2000] 都沒有 peephole connections
  - 論文提議的 peephole connections 是只連接到相同的記憶單元
  - 現今常用的 LSTM 使用 peephole connections 的方法是全連接，例如 [PyTorch 實作的 LSTM][PyTorch-LSTM] 就是一個例子
  - [原版 LSTM][LSTM1997] 細節可以看[我的筆記][note-LSTM1997]
  - [LSTM-2000][LSTM2000] 細節可以看[我的筆記][note-LSTM2000]
- 這篇論文終於把過去兩篇論文寫錯的數學式改對了
- 作者認為在不給予 LSTM 模型任何的輸入時， LSTM 必須要能夠觀察記憶單元內部狀態的變化才能模擬週期函數
  - 例如音樂節奏辨識
  - LSTM + peephole connections 在實驗中能夠成功解決模擬週期函數的任務
- 從[圖 15](#paper-fig-15) 可以發現模型的初始計算狀態為 $0$，但開始計算後模型計算狀態再也不為 $0$
  - 這表示模型**初始計算狀態**應該也被當成**參數**一起訓練
- 作者認為 RNN 模型在記憶上仍然有問題，即使使用 LSTM 記憶的容量仍然被記憶單元的個數限制，並且無法只靠簡單增加記憶單元個數解決
  - 與現今的 transformers 想法不同，大家都在搞大型 pre-trained model
  - 作者認為有效解決記憶容量問題的模型架構仍然未被發現
- LSTM 所採用的 truncated BPTT 最佳化演算法會導致模型沒辦法有效的學習遞迴的資訊
  - 根據作者實驗，當序列資料有大量雜訊時不做特殊的前處理就無法進行訓練

## 原始 LSTM 架構

### 模型架構

根據 [LSTM-2000][LSTM2000] 提出的架構如下（這篇論文不使用額外的**隱藏單元**，因此我們也完全不列出隱藏單元相關的公式）（細節可以參考[我的筆記][note-LSTM2000]）

|符號|意義|備註|
|-|-|-|
|$\din$|**輸入層**的維度|數值範圍為 $\Z^+$|
|$\dblk$|**記憶單元**的維度|數值範圍為 $\Z^+$|
|$\nblk$|**記憶單元**的個數|數值範圍為 $\Z^+$|
|$\dout$|**輸出層**的維度|數值範圍為 $\Z^+$|
|$T$|輸入序列的長度|數值範圍為 $\Z^+$|

以下所有符號的時間 $t$ 範圍為 $t \in \set{1, \dots, T}$

|符號|意義|維度|備註|
|-|-|-|-|
|$x(t)$|第 $t$ 個時間點的**輸入**|$\din$||
|$y^{\opfg}(t - 1)$|第 $t - 1$ 個時間點的**遺忘閘門**|$\nblk$|$y^{\opfg}(0) = 0$，同一個記憶單元**共享遺忘閘門**|
|$y^{\opig}(t - 1)$|第 $t - 1$ 個時間點的**輸入閘門**|$\nblk$|$y^{\opig}(0) = 0$，同一個記憶單元**共享輸入閘門**|
|$y^{\opog}(t - 1)$|第 $t - 1$ 個時間點的**輸出閘門**|$\nblk$|$y^{\opog}(0) = 0$，同一個記憶單元**共享輸出閘門**|
|$s^{\blk{k}}(t - 1)$|第 $t - 1$ 個時間點的第 $k$ 個**記憶單元區塊內部狀態**|$\dblk$|$s^{\blk{k}}(0) = 0$ 且 $k \in \set{1, \dots, \nblk}$|
|$y^{\blk{k}}(t - 1)$|第 $t - 1$ 個時間點的第 $k$ 個**記憶單元區塊輸出**|$\dblk$|$y^{\blk{k}}(0) = 0$ 且 $k \in \set{1, \dots, \nblk}$|
|$y(t)$|第 $t$ 個時間點的**輸出**|$\dout$|由 $t$ 時間點的**輸入**與**記憶單元輸出**透過**全連接**產生，因此沒有 $y(0)$|
|$\hat{y}(t)$|第 $t$ 個時間點的**預測目標**|$\dout$||

|符號|意義|下標範圍|
|-|-|-|
|$x_j(t)$|第 $t$ 個時間點的第 $j$ 個**輸入**|$j \in \set{1, \dots, \din}$|
|$y_k^{\opfg}(t - 1)$|第 $t - 1$ 個時間點第 $k$ 個記憶單元區塊的**遺忘閘門**|$k \in \set{1, \dots, \nblk}$|
|$y_k^{\opig}(t - 1)$|第 $t - 1$ 個時間點第 $k$ 個記憶單元區塊的**輸入閘門**|$k \in \set{1, \dots, \nblk}$|
|$y_k^{\opog}(t - 1)$|第 $t - 1$ 個時間點第 $k$ 個記憶單元區塊的**輸出閘門**|$k \in \set{1, \dots, \nblk}$|
|$s_i^{\blk{k}}(t - 1)$|第 $t - 1$ 個時間點的第 $k$ 個**記憶單元區塊**的第 $i$ 個**記憶單元內部狀態**|$i \in \set{1, \dots, \dblk}$|
|$y_i^{\blk{k}}(t - 1)$|第 $t - 1$ 個時間點的第 $k$ 個**記憶單元區塊**的第 $i$ 個**記憶單元輸出**|$i \in \set{1, \dots, \dblk}$|
|$y_i(t)$|第 $t$ 個時間點的第 $i$ 個**輸出**|$i \in \set{1, \dots, \dout}$|
|$\hat{y}_i(t)$|第 $t$ 個時間點的第 $i$ 個**預測目標**|$i \in \set{1, \dots, \dout}$|

|參數|意義|輸出維度|輸入維度|
|-|-|-|-|
|$\wfg$|產生**遺忘閘門**的全連接參數|$\nblk$|$\din + \nblk \cdot (3 + \dblk)$|
|$\wig$|產生**輸入閘門**的全連接參數|$\nblk$|$\din + \nblk \cdot (3 + \dblk)$|
|$\wog$|產生**輸出閘門**的全連接參數|$\nblk$|$\din + \nblk \cdot (3 + \dblk)$|
|$\wblk{k}$|產生第 $k$ 個**記憶單元淨輸入**的全連接參數|$\dblk$|$\din + \nblk \cdot (3 + \dblk)$|
|$\wout$|產生**輸出**的全連接參數|$\dblk$|$\din + \nblk \cdot \dblk$|

定義 $\sigma$ 為 sigmoid 函數 $\sigma(x) = \frac{1}{1 + e^{-x}}$

|函數|意義|公式|range|
|-|-|-|-|
|$f_k^{\opfg}$|第 $k$ 個**遺忘閘門**的啟發函數|$\sigma$|$[0, 1]$|
|$f_k^{\opig}$|第 $k$ 個**輸入閘門**的啟發函數|$\sigma$|$[0, 1]$|
|$f_k^{\opog}$|第 $k$ 個**輸出閘門**的啟發函數|$\sigma$|$[0, 1]$|
|$g_i^{\blk{k}}$|第 $k$ 個**記憶單元**第 $i$ 個**內部狀態**的啟發函數|$4\sigma - 2$|$[-2, 2]$|
|$h_i^{\blk{k}}$|第 $k$ 個**記憶單元**第 $i$ 個**輸出**的啟發函數|$2\sigma - 1$|$[-1, 1]$|
|$f_i^{\opout}$|第 $i$ 個**輸出**的啟發函數|$\sigma$|$[0, 1]$|

在 $t$ 時間點時得到**輸入** $x(t)$，產生 $t$ 時間點**遺忘閘門** $y^{\opfg}(t)$、**輸入閘門** $y^{\opig}(t)$ 與**輸出閘門** $y^{\opog}(t)$ 的方法如下

$$
\begin{align*}
g & \in \set{\opfg, \opig, \opog} \\
\opnet^g(t) & = w^g \cdot \begin{pmatrix}
x(t) \\
y^{\opfg}(t - 1) \\
y^{\opig}(t - 1) \\
y^{\opog}(t - 1) \\
y^{\blk{1}}(t - 1) \\
\vdots \\
y^{\blk{\nblk}}(t - 1)
\end{pmatrix} \\
y^g(t) & = f^g(\opnet^g(t))
\end{align*} \tag{1}\label{1}
$$

- 注意與[以前的筆記][note-LSTM2000]不同，這裡是產生 $t$ 時間點的資訊而不是 $t + 1$
- 注意是以 $t$ 時間點的輸入（不是 $t - 1$）與 $t - 1$ 時間點的計算狀態產生 $t$ 時間點的計算狀態

利用 $\eqref{1}$ 產生 $t$ 時間點的**記憶單元內部狀態** $s^{\blk{k}}(t)$ 方法如下

$$
\begin{align*}
k & \in \set{1, \dots, \nblk} \\
\opnet^{\blk{k}}(t) & = \wblk{k} \cdot \begin{pmatrix}
x(t) \\
y^{\opfg}(t - 1) \\
y^{\opig}(t - 1) \\
y^{\opog}(t - 1) \\
y^{\blk{1}}(t - 1) \\
\vdots \\
y^{\blk{\nblk}}(t - 1)
\end{pmatrix} \\
s^{\blk{k}}(t) & = y_k^{\opfg}(t) \cdot s^{\blk{k}}(t - 1) + y_k^{\opig}(t) \cdot g^{\blk{k}}(\opnet^{\blk{k}}(t))
\end{align*} \tag{2}\label{2}
$$

注意第 $k$ 個記憶單元內部狀態**共享遺忘閘門** $y_k^{\opfg}(t)$ 與**輸入閘門** $y_k^{\opig}(t)$。

利用 $\eqref{1}\eqref{2}$ 產生 $t$ 時間點的**記憶單元輸出** $y^{\blk{k}}(t)$ 方法如下

$$
\begin{align*}
k & \in \set{1, \dots, \nblk} \\
y^{\blk{k}}(t) & = y_k^{\opog}(t) \cdot h^{\blk{k}}(s^{\blk{k}}(t))
\end{align*} \tag{3}\label{3}
$$

注意第 $k$ 個記憶單元輸出**共享輸出閘門** $y_k^{\opog}(t)$。
由於實驗結果作者認為 $h^{\blk{k}}$ 不是很重要，因此 $\eqref{3}$ 中的式子改為

$$
y^{\blk{k}}(t) = y_k^{\opog}(t) \cdot s^{\blk{k}}(t) \quad k = 1, \dots, \nblk \tag{4}\label{4}
$$

產生 $t$ 時間點的**輸出**是透過 $t$ 時間點的**輸入**與**記憶單元輸出**（見 $\eqref{4}$）而得（注意是 $t$ 時間點不是 $t - 1$，代表[原版 LSTM][LSTM1997] 與 [LSTM-2000][LSTM2000] 都寫錯了）

$$
\begin{align*}
\opnet^{\opout}(t) & = \wout \cdot \begin{pmatrix}
x(t) \\
y^{\blk{1}}(t) \\
\vdots \\
y^{\blk{\nblk}}(t)
\end{pmatrix} \\
y(t) & = f^{\opout}(\opnet^{\opout}(t))
\end{align*} \tag{5}\label{5}
$$

### 最佳化

[原始 LSTM][LSTM1997] 提出與 truncated BPTT 相似的概念，透過 RTRL 進行參數更新，並故意**丟棄流出記憶單元的所有梯度**，避免梯度爆炸或梯度消失的問題，同時節省更新所需的空間與時間（local in time and space）。（細節可見[我的筆記][note-LSTM2000]）

令 $t = 1, \dots, T$，最佳化的目標為每個時間點 $t$ 所產生的**平方誤差總和最小化**

$$
\begin{align*}
\oploss(t) & = \sum_{i = 1}^{\dout} \oploss_i(t) \\
& = \sum_{i = 1}^{\dout} \frac{1}{2} \big(y_i(t) - \hat{y}_i(t)\big)^2
\end{align*} \tag{6}\label{6}
$$

以下我們使用 $\aptr$ 代表**丟棄部份梯度後的剩餘梯度**。

注意：論文中的式子 7 與 8 互相矛盾，式子 8 應改為 $\triangle w_{k m}(t) = \alpha \delta_k(t) y_m(t)$

#### 輸出參數的剩餘梯度

$$
\begin{align*}
\pd{\oploss(t)}{\wout_{i, j}} & = \pd{\oploss(t)}{y_i(t)} \cdot \pd{y_i(t)}{\netout{i}{t}} \cdot \pd{\netout{i}{t}}{\wout_{i, j}} \\
& = \big(y_i(t) - \hat{y}_i(t)\big) \cdot \dfnetout{i}{t} \cdot \begin{pmatrix}
x(t) \\
y^{\blk{1}}(t) \\
\vdots \\
y^{\blk{\nblk}}(t)
\end{pmatrix}_j
\end{align*} \tag{7}\label{7}
$$

其中 $1 \leq i \leq \dout$ 且 $1 \leq j \leq \din + \nblk \cdot \dblk$。

#### 輸出閘門參數的剩餘梯度

$$
\begin{align*}
\pd{\oploss(t)}{\wog_{k, q}} & \aptr \sum_{i = 1}^{\dout} \Bigg[\pd{\oploss(t)}{y_i(t)} \cdot \pd{y_i(t)}{\netout{i}{t}} \cdot \\
& \quad \pa{\sum_{j = 1}^{\dblk} \pd{\netout{i}{t}}{y_j^{\blk{k}}(t)} \cdot \pd{y_j^{\blk{k}}(t)}{y_k^{\opog}(t)}} \cdot \pd{y_k^{\opog}(t)}{\netog{k}{t}} \cdot \pd{\netog{k}{t}}{\wog_{k, q}}\Bigg] \\
& \aptr \sum_{i = 1}^{\dout} \Bigg[\big(y_i(t) - \hat{y}_i(t)\big) \cdot \dfnetout{i}{t} \cdot \\
& \quad \pa{\sum_{j = 1}^{\dblk} \wout_{i, \din + (k - 1) \cdot \dblk + j} \cdot s_j^{\blk{k}}(t)} \cdot \dfnetog{k}{t} \cdot \begin{pmatrix}
x(t) \\
y^{\opfg}(t - 1) \\
y^{\opig}(t - 1) \\
y^{\opog}(t - 1) \\
y^{\blk{1}}(t - 1) \\
\vdots \\
y^{\blk{\nblk}}(t - 1)
\end{pmatrix}_q\Bigg]
\end{align*} \tag{8}\label{8}
$$

其中 $1 \leq k \leq \nblk$ 且 $1 \leq q \leq \din + \nblk \cdot (3 + \dblk)$。

#### 輸入閘門參數的剩餘梯度

$$
\begin{align*}
& \pd{\oploss(t)}{\wig_{k, q}} \\
& \aptr \sum_{i = 1}^{\dout} \Bigg[\pd{\oploss(t)}{y_i(t)} \cdot \pd{y_i(t)}{\netout{i}{t}} \cdot \pa{\sum_{j = 1}^{\dblk} \pd{\netout{i}{t}}{y_j^{\blk{k}}(t)} \cdot \pd{y_j^{\blk{k}}(t)}{s_j^{\blk{k}}(t)} \cdot \pd{s_j^{\blk{k}}(t)}{\wig_{k, q}}}\Bigg] \\
& \aptr \sum_{i = 1}^{\dout} \Bigg[\pd{\oploss(t)}{y_i(t)} \cdot \pd{y_i(t)}{\netout{i}{t}} \cdot \Bigg(\sum_{j = 1}^{\dblk} \pd{\netout{i}{t}}{y_j^{\blk{k}}(t)} \cdot \pd{y_j^{\blk{k}}(t)}{s_j^{\blk{k}}(t)} \cdot \\
& \quad \quad \br{y_k^{\opfg}(t) \cdot \pd{s_j^{\blk{k}}(t - 1)}{\wig_{k, q}} + \pd{s_j^{\blk{k}}(t)}{y_k^{\opig}(t)} \cdot \pd{y_k^{\opig}(t)}{\netig{k}{t}} \cdot \pd{\netig{k}{t}}{\wig_{k, q}}}\Bigg)\Bigg] \\
& \aptr \sum_{i = 1}^{\dout} \Bigg[\big(y_i(t) - \hat{y}_i(t)\big) \cdot \dfnetout{i}{t} \cdot \Bigg(\sum_{j = 1}^{\dblk} \wout_{i, \din + (k - 1) \cdot \dblk + j} \cdot y_k^{\opog}(t) \cdot \\
& \quad \quad \br{y_k^{\opfg}(t) \cdot \pd{s_j^{\blk{k}}(t - 1)}{\wig_{k, q}} + \gnetblk{j}{k}{t} \cdot \dfnetig{k}{t} \cdot \begin{pmatrix}
x(t) \\
y^{\opfg}(t - 1) \\
y^{\opig}(t - 1) \\
y^{\opog}(t - 1) \\
y^{\blk{1}}(t - 1) \\
\vdots \\
y^{\blk{\nblk}}(t - 1)
\end{pmatrix}_q}\Bigg)\Bigg]
\end{align*} \tag{9}\label{9}
$$

其中 $1 \leq k \leq \nblk$ 且 $1 \leq q \leq \din + \nblk \cdot (3 + \dblk)$。

#### 遺忘閘門參數的剩餘梯度

$$
\begin{align*}
& \pd{\oploss(t)}{\wfg_{k, q}} \\
& \aptr \sum_{i = 1}^{\dout} \Bigg[\pd{\oploss(t)}{y_i(t)} \cdot \pd{y_i(t)}{\netout{i}{t}} \cdot \pa{\sum_{j = 1}^{\dblk} \pd{\netout{i}{t}}{y_j^{\blk{k}}(t)} \cdot \pd{y_j^{\blk{k}}(t)}{s_j^{\blk{k}}(t)} \cdot \pd{s_j^{\blk{k}}(t)}{\wfg_{k, q}}}\Bigg] \\
& \aptr \sum_{i = 1}^{\dout} \Bigg[\pd{\oploss(t)}{y_i(t)} \cdot \pd{y_i(t)}{\netout{i}{t}} \cdot \Bigg(\sum_{j = 1}^{\dblk} \pd{\netout{i}{t}}{y_j^{\blk{k}}(t)} \cdot \pd{y_j^{\blk{k}}(t)}{s_j^{\blk{k}}(t)} \cdot \\
& \quad \quad \br{\pd{y_k^{\opfg}(t)}{\netfg{k}{t}} \cdot \pd{\netfg{k}{t}}{\wfg_{k, q}} \cdot s_j^{\blk{k}}(t - 1) + y_k^{\opfg}(t) \cdot \pd{s_j^{\blk{k}}(t - 1)}{\wfg_{k, q}}}\Bigg)\Bigg] \\
& \aptr \sum_{i = 1}^{\dout} \Bigg[\big(y_i(t) - \hat{y}_i(t)\big) \cdot \dfnetout{i}{t} \cdot \Bigg(\sum_{j = 1}^{\dblk} \wout_{i, \din + (k - 1) \cdot \dblk + j} \cdot y_k^{\opog}(t) \cdot \\
& \quad \quad \br{\dfnetfg{k}{t} \cdot \begin{pmatrix}
x(t) \\
y^{\opfg}(t - 1) \\
y^{\opig}(t - 1) \\
y^{\opog}(t - 1) \\
y^{\blk{1}}(t - 1) \\
\vdots \\
y^{\blk{\nblk}}(t - 1)
\end{pmatrix}_q \cdot s_j^{\blk{k}}(t - 1) + y_k^{\opfg}(t) \cdot \pd{s_j^{\blk{k}}(t - 1)}{\wfg_{k, q}}}\Bigg)\Bigg]
\end{align*} \tag{10}\label{10}
$$

其中 $1 \leq k \leq \nblk$ 且 $1 \leq q \leq \din + \nblk \cdot (3 + \dblk)$。

#### 記憶單元淨輸入參數的剩餘梯度

$$
\begin{align*}
& \pd{\oploss(t)}{\wblk{k}_{p, q}} \\
& \aptr \sum_{i = 1}^{\dout} \br{\pd{\oploss(t)}{y_i(t)} \cdot \pd{y_i(t)}{\netout{i}{t}} \cdot \pd{\netout{i}{t}}{y_p^{\blk{k}}(t)} \cdot \pd{y_p^{\blk{k}}(t)}{s_p^{\blk{k}}(t)} \cdot \pd{s_p^{\blk{k}}(t)}{\wblk{k}_{p, q}}} \\
& \aptr \sum_{i = 1}^{\dout} \Bigg[\pd{\oploss(t)}{y_i(t)} \cdot \pd{y_i(t)}{\netout{i}{t}} \cdot \pd{\netout{i}{t}}{y_p^{\blk{k}}(t)} \cdot \pd{y_p^{\blk{k}}(t)}{s_p^{\blk{k}}(t)} \cdot \\
& \quad \quad \pa{f_k^{\opfg}(t) \cdot \pd{s_p^{\blk{k}}(t - 1)}{\wblk{k}_{p, q}} + \pd{s_p^{\blk{k}}(t)}{\gnetblk{j}{k}{t}} \cdot \pd{\gnetblk{j}{k}{t}}{\netblk{j}{k}{t}} \cdot \pd{\netblk{j}{k}{t}}{\wblk{k}_{p, q}}}\Bigg] \\
& \aptr \sum_{i = 1}^{\dout} \Bigg[\big(y_i(t) - \hat{y}_i(t)\big) \cdot \dfnetout{i}{t} \cdot \wout_{i, \din + (k - 1) \cdot \dblk + j} \cdot y_k^{\opog}(t) \cdot \\
& \quad \br{f_k^{\opfg}(t) \cdot \pd{s_p^{\blk{k}}(t - 1)}{\wblk{k}_{p, q}} + y_k^{\opig}(t) \cdot \dgnetblk{p}{k}{t} \cdot \begin{pmatrix}
x(t) \\
y^{\opfg}(t - 1) \\
y^{\opig}(t - 1) \\
y^{\opog}(t - 1) \\
y^{\blk{1}}(t - 1) \\
\vdots \\
y^{\blk{\nblk}}(t - 1)
\end{pmatrix}_q}\Bigg]
\end{align*} \tag{11}\label{11}
$$

其中 $1 \leq k \leq \nblk$， $1 \leq p \leq \dblk$ 且 $1 \leq q \leq \din + \nblk \cdot (3 + \dblk)$。

#### 梯度下降

計算完上述所有參數後使用**梯度下降**（gradient descent）進行參數更新

$$
\begin{align*}
\wout_{i, j} & \leftarrow \wout_{i, j} - \alpha \cdot \pd{\oploss(t)}{\wout_{i, j}} \\
\wog_{k, q} & \leftarrow \wog_{k, q} - \alpha \cdot \pd{\oploss(t)}{\wog_{k, q}} \\
\wig_{k, q} & \leftarrow \wig_{k, q} - \alpha \cdot \pd{\oploss(t)}{\wig_{k, q}} \\
\wfg_{k, q} & \leftarrow \wig_{k, q} - \alpha \cdot \pd{\oploss(t)}{\wfg_{k, q}} \\
\wblk{k}_{p, q} & \leftarrow \wblk{k}_{p, q} - \alpha \cdot \pd{\oploss(t)}{\wblk{k}_{p, q}}
\end{align*} \tag{12}\label{12}
$$

其中 $\alpha$ 為**學習率**（**learning rate**）。

由於使用基於 RTRL 的最佳化演算法，因此每個時間點 $t$ 計算完誤差後就可以更新參數。

### 問題

由於**輸出閘門**為 $0$ 時記憶單元的輸出等同於 $0$，導致基於記憶單元輸出計算所得的閘門與記憶單元本身無法觀察到**記憶單元的內部狀態**，作者認為在後續提出的任務中此現象會影響模型的表現。

## LSTM + Peephole Connections

### 模型架構

<a name="paper-fig-1"></a>

圖 1：LSTM 加上 peephole connections。
圖片來源：[論文][論文]。

![圖 1](https://i.imgur.com/G7Pgl3D.png)

- 針對前述問題提出的解決方法為 peephole connections
  - 所有閘門與記憶單元內部狀態相接
  - 最佳化時梯度不會經由 peephole connections 傳播（手動將梯度設為 $0$）

因此 $\eqref{1}$ 中的**遺忘閘門**與**輸入閘門**計算方法改成如下：

$$
\begin{align*}
g & \in \set{\opfg, \opig} \\
\opnet_k^g(t) & = \sum_{q = 1}^{\din + \nblk \cdot (3 + \dblk)} w_{k, q}^g \cdot \begin{pmatrix}
x(t) \\
y^{\opfg}(t - 1) \\
y^{\opig}(t - 1) \\
y^{\opog}(t - 1) \\
y^{\blk{1}}(t - 1) \\
\vdots \\
y^{\blk{\nblk}}(t - 1)
\end{pmatrix}_q + u_k^g \odot s^{\blk{k}}(t - 1) \\
y^g(t) & = f^g(\opnet^g(t))
\end{align*} \tag{13}\label{13}
$$

其中 $\ufg_k, \uig_k$ 的維度為 $1 \times \dblk$，$k$ 的範圍為 $1, \dots, \nblk$。

$\eqref{13}$ 的計算表示 $t$ 時間點的**遺忘閘門**與**輸入閘門**會與 $t - 1$ 時間點的**記憶單元內部狀態相連**，並且閘門只會與對應的記憶單元連接。

$\eqref{2}$ 的計算方法不變，在完成 $\eqref{2}$ 的計算後以 $t$ 時間點的記憶單元內部狀態計算**輸出閘門**（注意不是 $t - 1$）：

$$
\begin{align*}
\opnet_k^{\opog}(t) & = \sum_{q = 1}^{\din + \nblk \cdot (3 + \dblk)} \wog_{k, q} \cdot \begin{pmatrix}
x(t) \\
y^{\opfg}(t - 1) \\
y^{\opig}(t - 1) \\
y^{\opog}(t - 1) \\
y^{\blk{1}}(t - 1) \\
\vdots \\
y^{\blk{\nblk}}(t - 1)
\end{pmatrix}_q + \uog_k \odot s^{\blk{k}}(t) \\
y^{\opog}(t) & = f^{\opog}(\opnet^{\opog}(t))
\end{align*} \tag{14}\label{14}
$$

其中 $u_k^{\opog}$ 的維度為 $1 \times \dblk$，$k$ 的範圍為 $1, \dots, \nblk$。

$\eqref{14}$ 的計算表示 $t$ 時間點的**輸出閘門**會與 $t$ 時間點的**記憶單元內部狀態相連**，並且閘門只會與對應的記憶單元連接。

剩餘的計算方法（$\eqref{4}, \eqref{5}$）不變。

### 最佳化

由於只有閘門的計算方法受到影響，而且梯度不會流出 peephole connections，因此 $\eqref{8} \eqref{9} \eqref{10}$ 都不受影響，只需探討 $\ufg, \uig, \uog$ 的更新方法。

#### 輸出閘門參數的剩餘梯度

$$
\begin{align*}
\pd{\oploss(t)}{\uog_{k, q}} & \aptr \sum_{i = 1}^{\dout} \Bigg[\pd{\oploss(t)}{y_i(t)} \cdot \pd{y_i(t)}{\netout{i}{t}} \cdot \\
& \quad \pa{\sum_{j = 1}^{\dblk} \pd{\netout{i}{t}}{y_j^{\blk{k}}(t)} \cdot \pd{y_j^{\blk{k}}(t)}{y_k^{\opog}(t)}} \cdot \pd{y_k^{\opog}(t)}{\netog{k}{t}} \cdot \pd{\netog{k}{t}}{\uog_{k, q}}\Bigg] \\
& \aptr \sum_{i = 1}^{\dout} \Bigg[\big(y_i(t) - \hat{y}_i(t)\big) \cdot \dfnetout{i}{t} \cdot \\
& \quad \pa{\sum_{j = 1}^{\dblk} \wout_{i, \din + (k - 1) \cdot \dblk + j} \cdot s_j^{\blk{k}}(t)} \cdot \dfnetog{k}{t} \cdot s_q^{\blk{k}}(t)\Bigg]
\end{align*} \tag{15}\label{15}
$$

其中 $1 \leq k \leq \nblk$ 且 $1 \leq q \leq \dblk$，$\eqref{15}$ 式就是論文的 24 式。

#### 輸入閘門參數的剩餘梯度

$$
\begin{align*}
& \pd{\oploss(t)}{\uig_{k, q}} \\
& \aptr \sum_{i = 1}^{\dout} \Bigg[\pd{\oploss(t)}{y_i(t)} \cdot \pd{y_i(t)}{\netout{i}{t}} \cdot \pa{\sum_{j = 1}^{\dblk} \pd{\netout{i}{t}}{y_j^{\blk{k}}(t)} \cdot \pd{y_j^{\blk{k}}(t)}{s_j^{\blk{k}}(t)} \cdot \pd{s_j^{\blk{k}}(t)}{\uig_{k, q}}}\Bigg] \\
& \aptr \sum_{i = 1}^{\dout} \Bigg[\pd{\oploss(t)}{y_i(t)} \cdot \pd{y_i(t)}{\netout{i}{t}} \cdot \Bigg(\sum_{j = 1}^{\dblk} \pd{\netout{i}{t}}{y_j^{\blk{k}}(t)} \cdot \pd{y_j^{\blk{k}}(t)}{s_j^{\blk{k}}(t)} \cdot \\
& \quad \quad \br{y_k^{\opfg}(t) \cdot \pd{s_j^{\blk{k}}(t - 1)}{\uig_{k, q}} + \pd{s_j^{\blk{k}}(t)}{y_k^{\opig}(t)} \cdot \pd{y_k^{\opig}(t)}{\netig{k}{t}} \cdot \pd{\netig{k}{t}}{\uig_{k, q}}}\Bigg)\Bigg] \\
& \aptr \sum_{i = 1}^{\dout} \Bigg[\big(y_i(t) - \hat{y}_i(t)\big) \cdot \dfnetout{i}{t} \cdot \Bigg(\sum_{j = 1}^{\dblk} \wout_{i, \din + (k - 1) \cdot \dblk + j} \cdot y_k^{\opog}(t) \cdot \\
& \quad \quad \br{y_k^{\opfg}(t) \cdot \pd{s_j^{\blk{k}}(t - 1)}{\uig_{k, q}} + \gnetblk{j}{k}{t} \cdot \dfnetig{k}{t} \cdot s_q^{\blk{k}}(t - 1)}\Bigg)\Bigg]
\end{align*} \tag{16}\label{16}
$$

其中 $1 \leq k \leq \nblk$ 且 $1 \leq q \leq \dblk$，$\eqref{16}$ 式就是論文的 22 式。

#### 遺忘閘門參數的剩餘梯度

$$
\begin{align*}
& \pd{\oploss(t)}{\ufg_{k, q}} \\
& \aptr \sum_{i = 1}^{\dout} \Bigg[\pd{\oploss(t)}{y_i(t)} \cdot \pd{y_i(t)}{\netout{i}{t}} \cdot \pa{\sum_{j = 1}^{\dblk} \pd{\netout{i}{t}}{y_j^{\blk{k}}(t)} \cdot \pd{y_j^{\blk{k}}(t)}{s_j^{\blk{k}}(t)} \cdot \pd{s_j^{\blk{k}}(t)}{\ufg_{k, q}}}\Bigg] \\
& \aptr \sum_{i = 1}^{\dout} \Bigg[\pd{\oploss(t)}{y_i(t)} \cdot \pd{y_i(t)}{\netout{i}{t}} \cdot \Bigg(\sum_{j = 1}^{\dblk} \pd{\netout{i}{t}}{y_j^{\blk{k}}(t)} \cdot \pd{y_j^{\blk{k}}(t)}{s_j^{\blk{k}}(t)} \cdot \\
& \quad \quad \br{\pd{y_k^{\opfg}(t)}{\netfg{k}{t}} \cdot \pd{\netfg{k}{t}}{\ufg_{k, q}} \cdot s_j^{\blk{k}}(t - 1) + y_k^{\opfg}(t) \cdot \pd{s_j^{\blk{k}}(t - 1)}{\ufg_{k, q}}}\Bigg)\Bigg] \\
& \aptr \sum_{i = 1}^{\dout} \Bigg[\big(y_i(t) - \hat{y}_i(t)\big) \cdot \dfnetout{i}{t} \cdot \Bigg(\sum_{j = 1}^{\dblk} \wout_{i, \din + (k - 1) \cdot \dblk + j} \cdot y_k^{\opog}(t) \cdot \\
& \quad \quad \br{\dfnetfg{k}{t} \cdot s_q^{\blk{k}}(t - 1) \cdot s_j^{\blk{k}}(t - 1) + y_k^{\opfg}(t) \cdot \pd{s_j^{\blk{k}}(t - 1)}{\ufg_{k, q}}}\Bigg)\Bigg]
\end{align*} \tag{17}\label{17}
$$

其中 $1 \leq k \leq \nblk$ 且 $1 \leq q \leq \dblk$，$\eqref{17}$ 式就是論文的 23 式。

#### 梯度下降

計算完上述所有參數後使用**梯度下降**（gradient descent）進行參數更新

$$
\begin{align*}
\uog_{k, q} & \leftarrow \uog_{k, q} - \alpha \cdot \pd{\oploss(t)}{\uog_{k, q}} \\
\uig_{k, q} & \leftarrow \uig_{k, q} - \alpha \cdot \pd{\oploss(t)}{\uig_{k, q}} \\
\ufg_{k, q} & \leftarrow \uig_{k, q} - \alpha \cdot \pd{\oploss(t)}{\ufg_{k, q}}
\end{align*} \tag{18}\label{18}
$$

由於使用基於 RTRL 的最佳化演算法，因此每個時間點 $t$ 計算完誤差後就可以更新參數。

## 實驗設計

### 模型架構

<a name="paper-fig-2"></a>

圖 2：實驗所採用的 LSTM 架構。
圖片來源：[論文][論文]。

![圖 2](https://i.imgur.com/l1IUgTV.png)

所有實驗都使用相同架構，根據實驗作者發現少量的參數就可以達成所有任務。

|參數|數值（或範圍）|備註|
|-|-|-|
|$\din$|$1$||
|$\nblk$|$1$||
|$\dblk$|$1$||
|$\dout$|$1$||
|$\dim(\wblk{1})$|$\dblk \times [\din + \nblk \cdot \dblk + 1]$|只與輸入和記憶單元輸出相接，有額外使用偏差項|
|$\dim(\wfg)$|$\nblk \times [\din + \nblk \cdot \dblk + 1]$|只與輸入和記憶單元輸出相接，有額外使用偏差項|
|$\dim(\wig)$|$\nblk \times [\din + \nblk \cdot \dblk + 1]$|只與輸入和記憶單元輸出相接，有額外使用偏差項|
|$\dim(\wog)$|$\nblk \times [\din + \nblk \cdot \dblk + 1]$|只與輸入和記憶單元輸出相接，有額外使用偏差項|
|$\dim(\ufg_k)$|$1 \times \dblk$||
|$\dim(\uig_k)$|$1 \times \dblk$||
|$\dim(\uog_k)$|$1 \times \dblk$||
|$\dim(\wout)$|$\dout \times [\nblk \cdot \dblk + 1]$|外部輸入沒有直接連接到總輸出，有額外使用偏差項|
|遺忘閘門偏差項初始值|$-2$|[LSTM-2000][LSTM2000] 採用的初始值為正數，這裡居然用負數|
|輸入閘門偏差項初始值|$0$|[原版 LSTM][LSTM1997] 採用的初始值為負數，這裡居然用 $0$|
|輸出閘門偏差項初始值|$2$|[原版 LSTM][LSTM1997] 採用的初始值為負數，這裡居然用正數|
|參數初始化範圍|$[-0.1, 0.1]$||
|$g^{\blk{k}}$|$g^{\blk{k}}(x) = x$|identity mapping|
|$f^{\opout}$|$\sigma$|只有在模擬週期函數任務中採用 identity mapping|
|Learning rate|$10^{-5}$||
|總參數量|$17$||

### 實驗細節

- 與預測目標相減的絕對值作為誤差進行評估
  - 在凸波延遲偵測與生成任務中誤差必須小於 $0.49$
  - 在模擬週期函數任務中誤差必須小於 $0.3$
- 連續輸入只會在以下其中一個條件發生時停止
  - 單一時間點模型預測誤差過大
  - 在訓練時成功連續預測 $100$ 個凸波延遲
  - 在測試時成功連續預測 $1000$ 個凸波延遲
- 一次實驗最多進行 $10^7$ 次訓練，每執行一次訓練就進行一次測試
  - 每次訓練模型都是接收連續輸入
  - 在凸波延遲偵測任務中模型最多訓練 $10^8$ 次
- 總共實驗 $10$ 次，呈現平均實驗結果
- 訓練資料與測試資料皆為隨機產生，產生方法完全相同
- 梯度下降而外使用動量（momentum，細節請看[我的筆記][bp]），動量超參數以 $\eta$ 表示
  - 在連續凸波延遲偵測中 $\eta = 0.9999$
  - 在非連續凸波延遲偵測中 $\eta = 0.99$
  - 在凸波生成中 $\eta = 0.999$
  - 在模擬週期函數中 $\eta = 0.99$
- 論文沒寫但我猜最佳化目標一樣是 MSE

## 實驗 1：凸波延遲偵測

### 任務定義

輸入只會是 $0$ 或 $1$，$1$ 代表凸波，輸入序列的產生方法如下：

- 第 $1$ 個凸波產生的時間點為 $T(1) = F + I(1)$
  - $F \in \N$ 代表凸波週期，是一個常數
  - $I(1) \in \N$ 代表第 $1$ 個週期的延遲時間
  - 因此 $T(1) \geq F$
- 令 $n \geq 2$，第 $n$ 個凸波產生的時間點為 $T(n) = T(n - 1) + F + I(n)$
  - $I(n) \in \N$ 代表第 $n$ 個週期的延遲時間
- 模型必須要預測每個凸波的延遲時間（Measuring Spike Delays，MSD）
  - 令 $n \in \N$，任務等同於在第 $T(n)$ 時間點輸出 $I(n)$
  - 已知週期 $F$，LSTM 必須在接收 $F - 1$ 個 $0$ 開始紀錄延遲的時間差
- 任務分為連續輸入（MSD）與非連續輸入（non MSD，NMSD）
  - NMSD 的版本一次訓練只有一筆資料，即 $n = 1$
  - MSD 的版本一次訓練有多筆資料串接，$n = 100$

### 實驗結果

<a name="paper-fig-3"></a>

圖 3：凸波偵測實驗結果。
圖片來源：[論文][論文]。

![圖 3](https://i.imgur.com/RmIcNfd.png)

<a name="paper-fig-4"></a>

圖 4：凸波偵測實驗結果，分析週期長度對於表現的影響。
圖片來源：[論文][論文]。

![圖 4](https://i.imgur.com/nEDzkG3.png)

<a name="paper-fig-5"></a>

圖 5：凸波偵測實驗結果，增加延遲可能範圍進行實驗。
圖片來源：[論文][論文]。

![圖 5](https://i.imgur.com/JlSE6y7.png)

- 在 NMSD 任務中根據[圖 3](#paper-fig-3) 與[圖 4](#paper-fig-4) 實驗結果說明週期愈長（$F$ 愈大）愈不容易偵測
  - 即使 $I(n) \in \set{0, 1}$ 在週期較長的狀況下偵測延遲仍然很困難
- 雖然 peephole connections 在這個任務中不重要，但仍然比 [LSTM-2000][LSTM2000] 表現還要好
- 作者進一步將 $I(n)$ 的範圍調大，並且將 $f^{\opout}$ 從 sigmoid 函數改成 identity mapping（因為 sigmoid 的數值範圍只能落在 $[0, 1]$）進行實驗（見[圖 5](#paper-fig-5)）
  - 令 $i \in \set{1, \dots, 10}$，$I(n)$ 可以是 $\set{0, i}$ 或 $\set{0, \dots, i}$
  - 週期 $F = 10$
- 在 NMSD 任務中根據[圖 5](#paper-fig-5) 可以得到以下結論
  - 在此時驗中 [LSTM-2000][LSTM2000] 表現比 peephole connection 好
  - 延遲範圍差異愈大 LSTM 收斂愈快，作者認為過大的延遲差異會有明顯的特徵（見[圖 5](#paper-fig-5) 下半）
  - 當預測範圍可能性變多時，當最大延遲不超過 $5$ 時容易收斂，一旦超過 $5$ 則收斂變慢（見[圖 5](#paper-fig-5) 上半）
  - 在 $I(n) \in \set{0, 1}$ 時，使用 identity mapping 作為輸出函數表現比使用 sigmoid（見[圖 3](#paper-fig-3)）還要好，作者認為 sigmoid 會讓 gradient 變小所以收斂較慢

### 分析

<a name="paper-fig-6"></a>

圖 6：凸波偵測實驗中 [LSTM-2000][LSTM2000] 的計算狀態。
圖片來源：[論文][論文]。

![圖 6-1](https://i.imgur.com/ma5loA3.png)
![圖 6-2](https://i.imgur.com/adTLK96.png)

<a name="paper-fig-7"></a>

圖 7：凸波偵測實驗中 [LSTM + peephole connections][論文] 的計算狀態。
圖片來源：[論文][論文]。

![圖 7-1](https://i.imgur.com/ZOukPCr.png)
![圖 7-2](https://i.imgur.com/4GoR9TE.png)

- 透過實驗觀察發現 [LSTM-2000][LSTM2000] 學會兩種不同的方法進行凸波延遲偵測
  - [LSTM-2000][LSTM2000] 可以在每個時間點都增加記憶單元內部狀態 $s^{\blk{1}}$ 一點點，而預測值可以靠累加結果轉換而得（見[圖 6](#paper-fig-6) 左半）
  - [LSTM-2000][LSTM2000] 可以學會模擬振盪器，並根據振盪的次數進行預測（見[圖 6](#paper-fig-6) 右半）
- 從[圖 6](#paper-fig-6) 的下半可以發現[LSTM-2000][LSTM2000] 的輸出閘門維持在 $1$ 的狀態
  - 作者認為由於預測行為很少發生，因此維持輸出並不會影響表現
  - 但當任務需要預測的頻率變高時，模型就必須只在適當的時間點開啟輸出閘門，而該行為在沒有 peephole connections 的狀況下無法達成（原始 LSTM 架構的輸出閘門只會獲得 $t - 1$ 時間點的計算狀態，並沒有 $t$ 時間點的記憶單元內部狀態）
- 從[圖 7](#paper-fig-7) 的下半可以發現加上 peephole connections 的 LSTM 會在大多數時間關閉輸出閘門
  - 由於新加入的機制比[LSTM-2000][LSTM2000] 更複雜，因此需要更多的時間才會收斂（見[圖 3](#paper-fig-3) 與[圖 5](#paper-fig-5)）

## 實驗 2：凸波生成

### 任務定義

將凸波延遲偵測任務的輸入與輸出互換，稱為凸波生成（Generating Timed Spikes，GTS）

- 論文沒說明確的輸入輸出結構，但我的猜測如下
  - 輸入是 $T(n) \in \N$ 時，接下來的模型輸入會是 $T(n) - 1$ 個 $0$
  - 輸出是 $T(n) - 1$ 個 $0$，尾巴跟著一個 $1$
- 由於 LSTM 在不直接觀察記憶單元內部狀態的情況下無法完成 GTS（絕大多數的輸入都是 $0$），因此只顯示 peephole connections 的實驗

### 實驗結果

<a name="paper-fig-8"></a>

圖 8：凸波生成實驗結果。
圖片來源：[論文][論文]。

![圖 8](https://i.imgur.com/arXExQj.png)

<a name="paper-fig-9"></a>

圖 9：凸波生成實驗結果。
圖片來源：[論文][論文]。

![圖 9-1](https://i.imgur.com/HqjloKX.png)
![圖 9-2](https://i.imgur.com/jMgR93f.png)

<a name="paper-fig-10"></a>

圖 10：凸波生成實驗分析。
圖片來源：[論文][論文]。

![圖 10-1](https://i.imgur.com/qruAa3O.png)
![圖 10-2](https://i.imgur.com/ZnmkEO0.png)

- 根據[圖 8](#paper-fig-8) 我們可以發現週期愈長收斂時間愈久，與[圖 3](#paper-fig-3) 觀察結果相同
  - LSTM + peephole connections 可以解決圖波生成任務
- 根據[圖 9](#paper-fig-9) 下半我們可以發現輸出閘門只在需要生成凸波時開啟，生成完畢後馬上關閉
  - [圖 9](#paper-fig-9) 左下顯示生成凸波的當下由於遺忘閘門與輸入閘門一起關閉，因此記憶單元內部狀態直接重設為 $0$
- 作者嘗試在訓練時將遺忘閘門移除，發現模型無法收斂，證實遺忘閘門的必須性
- 根據[圖 10](#paper-fig-10) 可以觀察到模型生成凸波的時間點跟記憶單元內部狀態的增減時間相同

## 實驗 3：模擬週期函數

### 任務定義

讓 LSTM 模型模擬週期函數（Periodic Function Generation，PFG），注意訓練過程不需要給模型輸入，只要有輸出能夠模擬誤差即可，在此任務中就不得不使用 peephole connection（因為沒有輸入）。

令抽樣頻率為 $F$，模擬的週期函數共有三種，分別是三角函數波 $f_{\cos}$、三角波 $f_{\optri}$ 與方波 $f_{\oprect}$

$$
\begin{align*}
f_{\cos}(t) & = \frac{1}{2} \pa{1 - \cos\pa{\frac{2\pi t}{F}}} \\
f_{\optri}(t) & = \begin{dcases}
\frac{2 (t \opmod F)}{F} & \text{if } (t \opmod F) > \frac{F}{2} \\
2 - \frac{2 (t \opmod F)}{F} & \text{otherwise}
\end{dcases} \\
f_{\oprect}(t) & = \begin{dcases}
1 & \text{if } (t \opmod F) > \frac{F}{2} \\
0 & \text{otherwise}
\end{dcases}
\end{align*}
$$

模擬週期函數的難度與函數本身的波型（shape）和週期有關，而波型本身可以用一次微分和二次微分進行描述，論文採用一二次微分函數的最大絕對值 $\max_t \abs{f'(t)}$ 與 $\max_t \abs{f'{}'(t)}$ 作為特徵代表。

由於離散的時間點無法微分，作者將不可微分的函數用以下公式模擬微分

$$
\begin{align*}
f'(t) & \coloneqq f(t + 1) - f(t) \\
f'{}'(t) & \coloneqq f'(t + 1) - f'(t) \\
& \coloneqq f(t + 2) - 2 f(t + 1) + f(t)
\end{align*}
$$

因此當 $t^{\star} = \frac{F}{4}$ 時我們可以得到

$$
\begin{align*}
\max_t \abs{f_{\cos}'(t)} & = \max_t \abs{\frac{1}{2} \sin\pa{\frac{2 \pi t}{F}} \frac{2 \pi}{F}} \\
& = \max_t \abs{\frac{\pi}{F} \sin\pa{\frac{2 \pi t}{F}}} \\
& = \abs{\frac{\pi}{F} \sin\pa{\frac{2 \pi t^{\star}}{F}}} \\
& = \frac{\pi}{F}
\end{align*}
$$

當 $t^{\star} = 0$ 時我們可以得到

$$
\begin{align*}
\max_t \abs{f'{}_{\cos}'(t)} & = \max_t \abs{\frac{\pi}{F} \cos\pa{\frac{2 \pi t}{F}} \frac{2 \pi}{F}} \\
& = \abs{\frac{\pi}{F} \cos\pa{\frac{2 \pi t^{\star}}{F}} \frac{2 \pi}{F}} \\
& = \frac{2 \pi^2}{F^2}
\end{align*}
$$

當 $((t^{\star} + 1) \opmod F) < \frac{F}{2}$ 時我們可以得到

$$
\begin{align*}
\max_t \abs{f_{\optri}'(t)} & = \abs{f_{\optri}(t^{\star} + 1) - f_{\optri}(t^{\star})} \\
& = \abs{2 - \frac{2 ((t^{\star} + 1) \opmod F)}{F} - 2 + \frac{2 (t^{\star} \opmod F)}{F}} \\
& = \abs{-\frac{2 (t^{\star} + 1)}{F} + \frac{2t^{\star}}{F}} \\
& = \frac{2}{F}
\end{align*}
$$

當 $((t^{\star} + 1) \opmod F) = \frac{F}{2}$ 時我們可以得到

$$
\begin{align*}
\max_t \abs{f'{}_{\optri}'(t^{\star})} & = \abs{f_{\optri}(t^{\star} + 2) - 2f_{\optri}(t^{\star} + 1) + f_{\optri}(t^{\star})} \\
& = \abs{\frac{2 ((t^{\star} + 2) \opmod F)}{F} - 4 + \frac{4 ((t^{\star} + 1) \opmod F)}{F} + 2 - \frac{2 (t^{\star} \opmod F)}{F}} \\
& = \abs{\frac{2(t^{\star} + 2)}{F} - 4 + \frac{4F}{2F} + 2 - \frac{2t^{\star}}{F}} \\
& = \frac{4}{F}
\end{align*}
$$

當 $(t^{\star} \opmod F) = \frac{F}{2}$ 時我們可以得到

$$
\begin{align*}
\max_t \abs{f_{\oprect}'(t)} & = \abs{f_{\oprect}(t^{\star} + 1) - f_{\oprect}(t^{\star})} \\
& = \abs{1 - 0 + 0} \\
& = 1
\end{align*}
$$

當 $((t^{\star} + 1) \opmod F) = \frac{F}{2}$ 時我們可以得到

$$
\begin{align*}
\max_t \abs{f'{}_{\oprect}'(t)} & = \abs{f_{\oprect}(t^{\star} + 2) - 2f_{\oprect}(t^{\star} + 1) + f_{\oprect}(t^{\star})} \\
& = \abs{1 - 0} \\
& = 1
\end{align*}
$$

一般來說 $\max_t \abs{f'(t)}$ 與 $\max_t \abs{f'{}'(t)}$ 愈大代表波型變化愈大，因此愈難模擬。

而 $F$ 愈大代表同一個週期內的波型變化較多，因此 $F$ 愈大愈難模擬，此實驗的 $F \in \set{10, 25}$。

### 實驗結果

<a name="paper-fig-11"></a>

圖 11：模擬週期函數實驗結果。
圖片來源：[論文][論文]。

![圖 11](https://i.imgur.com/zIALWJF.png)

<a name="paper-fig-12"></a>

圖 12：模擬週期函數實驗結果。
圖片來源：[論文][論文]。

![圖 12-1](https://i.imgur.com/v8wFmJ2.png)
![圖 12-2](https://i.imgur.com/ctHi291.png)

- [LSTM-2000][LSTM2000] 只能模擬 $F = 10$ 的 $f_{\cos}$，且收斂時間長（見[圖 11](#paper-fig-11)）
- 不使用遺忘閘門的[原版 LSTM][LSTM1997] 無法模擬超過兩個以上的週期
- 將評估標準提生成誤差低於 $0.15$ 時，模型要花更長的時間收斂
  - 模擬的週期函數為 $f_{\cos}$，$F = 25$
  - RMSE 的表現從 $0.17 \pm 0.019$ （見[圖 11](#paper-fig-11)） 降至 $0.086 \pm 0.002$
  - 產生完美表現（$100\%$ 預測正確）的時間點為 $(2704 \pm 49) \cdot 10^3$

### 分析

<a name="paper-fig-13"></a>

圖 13：模擬週期函數實驗分析。
圖片來源：[論文][論文]。

![圖 13-1](https://i.imgur.com/SJ43cWb.png)
![圖 13-2](https://i.imgur.com/pRKxTpM.png)

<a name="paper-fig-14"></a>

圖 14：模擬週期函數實驗分析。
圖片來源：[論文][論文]。

![圖 14-1](https://i.imgur.com/gW5bmcu.png)
![圖 14-2](https://i.imgur.com/7TIXIqV.png)

<a name="paper-fig-15"></a>

圖 15：模擬週期函數實驗分析。
圖片來源：[論文][論文]。

![圖 15](https://i.imgur.com/biv8smX.png)

- 由於模型沒有收到任何輸入，完全只能依賴記憶單元內部狀態進行模擬，因此記憶單元內部狀態的數值變化應該要與模擬目標擁有類似的曲線（見[圖 13](#paper-fig-13)）
- 作者認為此任務可以在不使用 peephole connections 的狀態下完成任務，但流經閘門的梯度被手動丟棄，因此 [LSTM-2000][LSTM2000] 的架構很難最佳化，導致實驗表現不佳（見[圖 11](#paper-fig-11)）
  - LSTM + peephole connections 收斂速度比 [LSTM-2000][LSTM2000] 快，見[圖 11](#paper-fig-11)
  - 觀察 peephole connections 的參數數值，作者發現數值與記憶單元輸出連接到閘門的參數數量級相同，說明 peephole connections 真的有被用來協助模擬週期函數
- 從[圖 14](#paper-fig-14) 可以觀察到以下現象
  - 方波值為 $1$ 時
    - 記憶單元輸出與記憶單元內部狀態的數值相同
    - 輸出閘門維持開啟
    - 模型內部狀態逐漸遞減（趨向 $0$）
    - 由於記憶單元輸出與記憶單元輸入連結的參數數值為負，因此模型有辦法遞減記憶單元內部狀態
  - 方波值為 $0$ 時
    - 記憶單元輸出為 $0$
    - 輸出閘門維持關閉
    - 模型內部狀態逐漸遞增（趨向 $1$）
- 從[圖 15](#paper-fig-15) 可以發現模型的初始計算狀態為 $0$，但開始計算後模型計算狀態再也不為 $0$
  - 這表示模型**初始計算狀態**應該也被當成**參數**一起訓練

[LSTM1997]: https://ieeexplore.ieee.org/abstract/document/6795963
[LSTM2000]: https://direct.mit.edu/neco/article-abstract/12/10/2451/6415/Learning-to-Forget-Continual-Prediction-with-LSTM
[note-LSTM1997]: {% link _posts/2021-11-14-long-short-term-memory.md %}
[note-LSTM2000]: {% link _posts/2021-12-13-learning-to-forget-continual-prediction-with-lstm.md %}
[論文]: https://www.jmlr.org/papers/v3/gers02a.html
[bp]: {% link _posts/2021-12-07-learning-representations-by-backpropagating-errors.md %}
[PyTorch-LSTM]: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
