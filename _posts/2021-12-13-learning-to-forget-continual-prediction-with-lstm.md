---
layout: ML-note
title:  "Learning to Forget: Continual Prediction with LSTM"
date:   2021-12-13 14:09:00 +0800
categories: [
  Deep Learning,
  Model Architecture,
]
tags: [
  RNN,
  LSTM,
]
author: [
  Felix A. Gers,
  Jürgen Schmidhuber,
  Fred Cummins,
]
---

|-|-|
|目標|提出在 LSTM 上增加 forget gate|
|作者|Felix A. Gers, Jürgen Schmidhuber, Fred Cummins|
|期刊/會議名稱|Neural Computation|
|發表時間|2000|
|論文連結|<https://direct.mit.edu/neco/article-abstract/12/10/2451/6415/Learning-to-Forget-Continual-Prediction-with-LSTM>|

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
  <!-- Operator cell. -->
  $\providecommand{\opcell}{}$
  $\renewcommand{\opcell}{\operatorname{cell}}$
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
  <!-- Cell dimension. -->
  $\providecommand{\dcell}{}$
  $\renewcommand{\dcell}{d_{\opcell}}$

  <!-- Number of cells. -->
  $\providecommand{\ncell}{}$
  $\renewcommand{\ncell}{n_{\opcell}}$

  <!-- Cell block k. -->
  $\providecommand{\cell}{}$
  $\renewcommand{\cell}[1]{\opcell^{#1}}$

  <!-- Weight of multiplicative forget gate. -->
  $\providecommand{\wfg}{}$
  $\renewcommand{\wfg}{w^{\opfg}}$
  <!-- Weight of multiplicative input gate. -->
  $\providecommand{\wig}{}$
  $\renewcommand{\wig}{w^{\opig}}$
  <!-- Weight of multiplicative output gate. -->
  $\providecommand{\wog}{}$
  $\renewcommand{\wog}{w^{\opog}}$
  <!-- Weight of cell units. -->
  $\providecommand{\wcell}{}$
  $\renewcommand{\wcell}[1]{w^{\cell{#1}}}$
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
  $\providecommand{\netcell}{}$
  $\renewcommand{\netcell}[3]{\opnet_{#1}^{\cell{#2}}(#3)}$
  <!-- Net input of cell unit with activatiton g. -->
  $\providecommand{\gnetcell}{}$
  $\renewcommand{\gnetcell}[3]{g_{#1}^{\cell{#2}}\big(\netcell{#1}{#2}{#3}\big)}$
  <!-- Derivative of g with respect to net input of cell unit. -->
  $\providecommand{\dgnetcell}{}$
  $\renewcommand{\dgnetcell}[3]{g_{#1}^{\cell{#2}}{'}\big(\netcell{#1}{#2}{#3}\big)}$
  <!-- Cell unit with activatiton h. -->
  $\providecommand{\hcell}{}$
  $\renewcommand{\hcell}[3]{h_{#1}^{\cell{k}}\big(s_{#1}^{\cell{#2}}(#3)\big)}$
  <!-- Derivative of h with respect to cell unit. -->
  $\providecommand{\dhcell}{}$
  $\renewcommand{\dhcell}[3]{h_{#1}^{\cell{k}}{'}\big(s_{#1}^{\cell{#2}}(#3)\big)}$

  <!-- Gradient approximation by truncating gradient. -->
  $\providecommand{\aptr}{}$
  $\renewcommand{\aptr}{\approx_{\operatorname{tr}}}$
</p>

<!-- End LaTeX command define section. -->

- [此篇論文][論文]與[原版 LSTM][LSTM1997] 都寫錯自己的數學公式，但我的筆記內容主要以正確版本為主，原版 LSTM 可以參考[我的筆記][note-LSTM1997]
- [原版 LSTM][LSTM1997] 沒有遺忘閘門，現今常用的 LSTM 都有遺忘閘門，概念由這篇論文提出
- 包含多個子序列的**連續輸入**會讓 LSTM 的記憶單元內部狀態沒有上下界
  - 現實中的大多數資料並不存在好的分割序列演算法，導致輸入給模型的資料通常都包含多個子序列
  - 根據實驗 1 的分析發現記憶單元內部狀態的累積導致預測結果完全錯誤
- 使用遺忘閘門讓模型學會適當的忘記已經處理過的子序列資訊
  - 當遺忘閘門的**偏差項**初始化為**正數**時會保持記憶單元內部狀態，等同於使用原版的 LSTM
  - 因此使用遺忘閘門的 LSTM 能夠達成原版 LSTM 的功能，並額外擁有自動重設記憶單元的機制
- 這篇模型的理論背景較少，實驗為主的描述居多

## 原始 LSTM

### 模型架構

根據[原始論文][LSTM1997]提出的架構如下（這篇論文不使用額外的**隱藏單元**，因此我們也完全不列出隱藏單元相關的公式）（細節可以參考[我的筆記][note-LSTM1997]）

|符號|意義|備註|
|-|-|-|
|$\din$|**輸入層**的維度|數值範圍為 $\Z^+$|
|$\dcell$|**記憶單元**的維度|數值範圍為 $\Z^+$|
|$\ncell$|**記憶單元**的個數|數值範圍為 $\Z^+$|
|$\dout$|**輸出層**的維度|數值範圍為 $\Z^+$|
|$T$|輸入序列的長度|數值範圍為 $\Z^+$|

以下所有符號的時間 $t$ 範圍為 $t = 0, \dots, T - 1$

|符號|意義|維度|備註|
|-|-|-|-|
|$x(t)$|第 $t$ 個時間點的**輸入**|$\din$||
|$y^{\opig}(t)$|第 $t$ 個時間點的**輸入閘門**|$\ncell$|$y^{\opig}(0) = 0$，同一個記憶單元**共享輸入閘門**|
|$y^{\opog}(t)$|第 $t$ 個時間點的**輸出閘門**|$\ncell$|$y^{\opog}(0) = 0$，同一個記憶單元**共享輸出閘門**|
|$s^{\cell{k}}(t)$|第 $t$ 個時間點的第 $k$ 個**記憶單元內部狀態**|$\dcell$|$s^{\cell{k}}(0) = 0$，$k$ 的範圍為 $k = 1, \dots, \ncell$|
|$y^{\cell{k}}(t)$|第 $t$ 個時間點的第 $k$ 個**記憶單元輸出**|$\dcell$|$y^{\cell{k}}(0) = 0$，$k$ 的範圍為 $k = 1, \dots, \ncell$|
|$y(t + 1)$|第 $t + 1$ 個時間點的**輸出**|$\dout$|由 $t$ 時間點的**輸入**與**記憶單元輸出**透過**全連接**產生，因此沒有 $y(0)$|
|$\hat{y}(t + 1)$|第 $t + 1$ 個時間點的**預測目標**|$\dout$||

|符號|意義|下標範圍|
|-|-|-|
|$x_j(t)$|第 $t$ 個時間點的第 $j$ 個**輸入**|$j = 1, \dots, \din$|
|$y_k^{\opig}(t)$|第 $t$ 個時間點第 $k$ 個記憶單元的**輸入閘門**|$k = 1, \dots, \ncell$|
|$y_k^{\opog}(t)$|第 $t$ 個時間點第 $k$ 個記憶單元的**輸出閘門**|$k = 1, \dots, \ncell$|
|$s_i^{\cell{k}}(t)$|第 $t$ 個時間點的第 $k$ 個**記憶單元**的第 $i$ 個**內部狀態**|$i = 1, \dots, \dcell$|
|$y_i^{\cell{k}}(t)$|第 $t$ 個時間點的第 $k$ 個**記憶單元**的第 $i$ 個**輸出**|$i = 1, \dots, \dcell$|
|$y_i(t + 1)$|第 $t + 1$ 個時間點的第 $i$ 個**輸出**|$i = 1, \dots, \dout$|
|$\hat{y}_i(t + 1)$|第 $t + 1$ 個時間點的第 $i$ 個**預測目標**|$i = 1, \dots, \dout$|

|參數|意義|輸出維度|輸入維度|
|-|-|-|-|
|$\wig$|產生**輸入閘門**的全連接參數|$\ncell$|$\din + \ncell \cdot (2 + \dcell)$|
|$\wog$|產生**輸出閘門**的全連接參數|$\ncell$|$\din + \ncell \cdot (2 + \dcell)$|
|$\wcell{k}$|產生第 $k$ 個**記憶單元淨輸入**的全連接參數|$\dcell$|$\din + \ncell \cdot (2 + \dcell)$|
|$\wout$|產生**輸出**的全連接參數|$\dcell$|$\din + \ncell \cdot \dcell$|

定義 $\sigma$ 為 sigmoid 函數 $\sigma(x) = \frac{1}{1 + e^{-x}}$

|函數|意義|公式|range|
|-|-|-|-|
|$f_k^{\opig}$|第 $k$ 個**輸入閘門**的啟發函數|$\sigma$|$[0, 1]$|
|$f_k^{\opog}$|第 $k$ 個**輸出閘門**的啟發函數|$\sigma$|$[0, 1]$|
|$g_i^{\cell{k}}$|第 $k$ 個**記憶單元**第 $i$ 個**內部狀態**的啟發函數|$4\sigma - 2$|$[-2, 2]$|
|$h_i^{\cell{k}}$|第 $k$ 個**記憶單元**第 $i$ 個**輸出**的啟發函數|$2\sigma - 1$|$[-1, 1]$|
|$f_i^{\opout}$|第 $i$ 個**輸出**的啟發函數|$\sigma$|$[0, 1]$|

在 $t$ 時間點時得到**輸入** $x(t)$，產生 $t + 1$ 時間點**輸入閘門** $y^{\opig}(t + 1)$ 與**輸出閘門** $y^{\opog}(t + 1)$ 的方法如下

$$
\begin{align*}
\opnet^{\opig}(t + 1) & = \wig \cdot \begin{pmatrix}
x(t) \\
y^{\opig}(t) \\
y^{\opog}(t) \\
y^{\cell{1}}(t) \\
\vdots \\
y^{\cell{\ncell}}(t)
\end{pmatrix} \\
y^{\opig}(t + 1) & = f^{\opig}(\opnet^{\opig}(t + 1)) = \begin{pmatrix}
\fnetig{1}{t + 1} \\
\fnetig{2}{t + 1} \\
\vdots \\
\fnetig{\ncell}{t + 1}
\end{pmatrix} \\
\opnet^{\opog}(t + 1) & = \wog \cdot \begin{pmatrix}
x(t) \\
y^{\opig}(t) \\
y^{\opog}(t) \\
y^{\cell{1}}(t) \\
\vdots \\
y^{\cell{\ncell}}(t)
\end{pmatrix} \\
y^{\opog}(t + 1) & = f^{\opog}(\opnet^{\opog}(t + 1)) = \begin{pmatrix}
\fnetog{1}{t + 1} \\
\fnetog{2}{t + 1} \\
\vdots \\
\fnetog{\ncell}{t + 1}
\end{pmatrix}
\end{align*} \tag{1}\label{1}
$$

利用 $\eqref{1}$ 產生 $t + 1$ 時間點的**記憶單元內部狀態** $s^{\cell{k}}(t + 1)$ 方法如下

$$
\begin{align*}
\opnet^{\cell{k}}(t + 1) & = \wcell{k} \cdot \begin{pmatrix}
x(t) \\
y^{\opig}(t) \\
y^{\opog}(t) \\
y^{\cell{1}}(t) \\
\vdots \\
y^{\cell{\ncell}}(t)
\end{pmatrix} && k = 1, \dots, \ncell \\
s^{\cell{k}}(t + 1) & = s^{\cell{k}}(t) + y_k^{\opig}(t + 1) \cdot g^{\cell{k}}(\opnet^{\cell{k}}(t + 1)) && k = 1, \dots, \ncell \\
& = \begin{pmatrix}
s_1^{\cell{k}}(t) + y_k^{\opig}(t + 1) \cdot \gnetcell{1}{k}{t + 1} \\
s_2^{\cell{k}}(t) + y_k^{\opig}(t + 1) \cdot \gnetcell{2}{k}{t + 1} \\
\vdots \\
s_{\dcell}^{\cell{k}}(t) + y_k^{\opig}(t + 1) \cdot \gnetcell{\dcell}{k}{t + 1}
\end{pmatrix}
\end{align*} \tag{2}\label{2}
$$

注意第 $k$ 個記憶單元內部狀態**共享輸入閘門** $y_k^{\opig}(t + 1)$。

利用 $\eqref{1}\eqref{2}$ 產生 $t + 1$ 時間點的**記憶單元輸出** $y^{\cell{k}}(t + 1)$ 方法如下

$$
\begin{align*}
y^{\cell{k}}(t + 1) & = y_k^{\opog}(t + 1) \cdot h^{\cell{k}}(s^{\cell{k}}(t + 1)) && k = 1, \dots, \ncell \\
& = \begin{pmatrix}
y_k^{\opog}(t + 1) \cdot \hcell{1}{k}{t + 1} \\
y_k^{\opog}(t + 1) \cdot \hcell{2}{k}{t + 1} \\
\vdots \\
y_k^{\opog}(t + 1) \cdot \hcell{\dcell}{k}{t + 1}
\end{pmatrix}
\end{align*} \tag{3}\label{3}
$$

注意第 $k$ 個記憶單元輸出**共享輸出閘門** $y_k^{\opog}(t + 1)$。

產生 $t + 1$ 時間點的**輸出**是透過 $t$ 時間點的**輸入**與 $t + 1$ 時間點的**記憶單元輸出**（見 $\eqref{3}$）而得

$$
\begin{align*}
\opnet^{\opout}(t + 1) & = \wout \cdot \begin{pmatrix}
x(t) \\
y^{\cell{1}}(t + 1) \\
\vdots \\
y^{\cell{\ncell}}(t + 1)
\end{pmatrix} \\
y(t + 1) & = f^{\opout}(\opnet^{\opout}(t + 1)) = \begin{pmatrix}
\fnetout{1}{t + 1} \\
\fnetout{2}{t + 1} \\
\vdots \\
\fnetout{\dout}{t + 1}
\end{pmatrix}
\end{align*} \tag{4}\label{4}
$$

[這篇論文][論文]與[原版 LSTM 的論文][LSTM1997] 都不小心寫成 $t$ 時間點的記憶單元輸出，在 [LSTM-2002][LSTM2002] 才終於寫對。

### 最佳化

[原始 LSTM][LSTM1997] 提出與 truncated BPTT 相似的概念，透過 RTRL 進行參數更新，並故意**丟棄流出記憶單元的所有梯度**，避免梯度爆炸或梯度消失的問題，同時節省更新所需的空間與時間（local in time and space）。（細節可見[我的筆記][note-LSTM1997]）

令 $t = 0, \dots, T - 1$，最佳化的目標為每個時間點 $t + 1$ 所產生的**平方誤差總和最小化**

$$
\begin{align*}
\oploss(t + 1) & = \sum_{i = 1}^{\dout} \oploss_i(t + 1) \\
& = \sum_{i = 1}^{\dout} \frac{1}{2} \big(y_i(t + 1) - \hat{y}_i(t + 1)\big)^2
\end{align*} \tag{5}\label{5}
$$

以下我們使用 $\aptr$ 代表**丟棄部份梯度後的剩餘梯度**。

輸出參數的剩餘梯度為

$$
\begin{align*}
\pd{\oploss(t + 1)}{\wout_{i, j}} & = \pd{\oploss(t + 1)}{y_i(t + 1)} \cdot \pd{y_i(t + 1)}{\netout{i}{t + 1}} \cdot \pd{\netout{i}{t + 1}}{\wout_{i, j}} \\
& = \big(y_i(t + 1) - \hat{y}_i(t + 1)\big) \cdot \dfnetout{i}{t + 1} \cdot \begin{pmatrix}
x(t) \\
y^{\cell{1}}(t + 1) \\
\vdots \\
y^{\cell{\ncell}}(t + 1)
\end{pmatrix}_j
\end{align*} \tag{6}\label{6}
$$

其中 $1 \leq i \leq \dout$ 且 $1 \leq j \leq \din + \ncell \cdot \dcell$。

輸出閘門參數的剩餘梯度為

$$
\begin{align*}
& \pd{\oploss(t + 1)}{\wog_{k, q}} \\
& \aptr \sum_{i = 1}^{\dout} \Bigg[\pd{\oploss(t + 1)}{y_i(t + 1)} \cdot \pd{y_i(t + 1)}{\netout{i}{t + 1}} \cdot \\
& \quad \pa{\sum_{j = 1}^{\dcell} \pd{\netout{i}{t + 1}}{y_j^{\cell{k}}(t + 1)} \cdot \pd{y_j^{\cell{k}}(t + 1)}{y_k^{\opog}(t + 1)}} \cdot \pd{y_k^{\opog}(t + 1)}{\netog{k}{t + 1}} \cdot \pd{\netog{k}{t + 1}}{\wog_{k, q}}\Bigg] \\
& \aptr \sum_{i = 1}^{\dout} \Bigg[\big(y_i(t + 1) - \hat{y}_i(t + 1)\big) \cdot \dfnetout{i}{t + 1} \cdot \\
& \quad \pa{\sum_{j = 1}^{\dcell} \wout_{i, \din + (k - 1) \cdot \dcell + j} \cdot \hcell{j}{k}{t + 1}} \cdot \dfnetog{k}{t + 1} \cdot \begin{pmatrix}
x(t) \\
y^{\opig}(t) \\
y^{\opog}(t) \\
y^{\cell{1}}(t) \\
\vdots \\
y^{\cell{\ncell}}(t)
\end{pmatrix}_q\Bigg]
\end{align*} \tag{7}\label{7}
$$

其中 $1 \leq k \leq \ncell$ 且 $1 \leq q \leq \din + \ncell \cdot (2 + \dcell)$。

輸入閘門參數的剩餘梯度為

$$
\begin{align*}
& \pd{\oploss(t + 1)}{\wig_{k, q}} \\
& \aptr \sum_{i = 1}^{\dout} \Bigg[\pd{\oploss(t + 1)}{y_i(t + 1)} \cdot \pd{y_i(t + 1)}{\netout{i}{t + 1}} \cdot \\
& \quad \pa{\sum_{j = 1}^{\dcell} \pd{\netout{i}{t + 1}}{y_j^{\cell{k}}(t + 1)} \cdot \pd{y_j^{\cell{k}}(t + 1)}{s_j^{\cell{k}}(t + 1)} \cdot \pd{s_j^{\cell{k}}(t + 1)}{\wig_{k, q}}}\Bigg] \\
& \aptr \sum_{i = 1}^{\dout} \Bigg[\pd{\oploss(t + 1)}{y_i(t + 1)} \cdot \pd{y_i(t + 1)}{\netout{i}{t + 1}} \cdot \Bigg(\sum_{j = 1}^{\dcell} \pd{\netout{i}{t + 1}}{y_j^{\cell{k}}(t + 1)} \cdot \pd{y_j^{\cell{k}}(t + 1)}{s_j^{\cell{k}}(t + 1)} \cdot \\
& \quad \quad \br{\pd{s_j^{\cell{k}}(t)}{\wig_{k, q}} + \gnetcell{j}{k}{t + 1} \cdot \pd{y_k^{\opig}(t + 1)}{\netig{k}{t + 1}} \cdot \pd{\netig{k}{t + 1}}{\wig_{k, q}}}\Bigg)\Bigg] \\
& \aptr \sum_{i = 1}^{\dout} \Bigg[\big(y_i(t + 1) - \hat{y}_i(t + 1)\big) \cdot \dfnetout{i}{t + 1} \cdot \\
& \quad \Bigg(\sum_{j = 1}^{\dcell} \wout_{i, \din + (k - 1) \cdot \dcell + j} \cdot y_k^{\opog}(t + 1) \cdot \dhcell{j}{k}{t + 1} \cdot \\
& \quad \quad \br{\pd{s_j^{\cell{k}}(t)}{\wig_{k, q}} + \gnetcell{j}{k}{t + 1} \cdot \dfnetig{k}{t + 1} \cdot \begin{pmatrix}
x(t) \\
y^{\opig}(t) \\
y^{\opog}(t) \\
y^{\cell{1}}(t) \\
\vdots \\
y^{\cell{\ncell}}(t)
\end{pmatrix}_q}\Bigg)\Bigg]
\end{align*} \tag{8}\label{8}
$$

其中 $1 \leq k \leq \ncell$ 且 $1 \leq q \leq \din + \ncell \cdot (2 + \dcell)$。

記憶單元淨輸入參數的剩餘梯度為

$$
\begin{align*}
& \pd{\oploss(t + 1)}{\wcell{k}_{p, q}} \\
& \aptr \sum_{i = 1}^{\dout} \br{\pd{\oploss(t + 1)}{y_i(t + 1)} \cdot \pd{y_i(t + 1)}{\netout{i}{t + 1}} \cdot \pd{\netout{i}{t + 1}}{y_p^{\cell{k}}(t + 1)} \cdot \pd{y_p^{\cell{k}}(t + 1)}{s_p^{\cell{k}}(t + 1)} \cdot \pd{s_p^{\cell{k}}(t + 1)}{\wcell{k}_{p, q}}} \\
& \aptr \sum_{i = 1}^{\dout} \Bigg[\pd{\oploss(t + 1)}{y_i(t + 1)} \cdot \pd{y_i(t + 1)}{\netout{i}{t + 1}} \cdot \pd{\netout{i}{t + 1}}{y_p^{\cell{k}}(t + 1)} \cdot \pd{y_p^{\cell{k}}(t + 1)}{s_p^{\cell{k}}(t + 1)} \cdot \\
& \quad \quad \pa{\pd{s_p^{\cell{k}}(t)}{\wcell{k}_{p, q}} + y_k^{\opig}(t + 1) \cdot \pd{\gnetcell{j}{k}{t + 1}}{\netcell{j}{k}{t + 1}} \cdot \pd{\netcell{j}{k}{t + 1}}{\wcell{k}_{p, q}}}\Bigg] \\
& \aptr \sum_{i = 1}^{\dout} \Bigg[\big(y_i(t + 1) - \hat{y}_i(t + 1)\big) \cdot \dfnetout{i}{t + 1} \cdot \wout_{i, \din + (k - 1) \cdot \dcell + j} \cdot \\
& \quad y_k^{\opog}(t + 1) \cdot \dhcell{j}{k}{t + 1} \cdot \\
& \quad \br{\pd{s_p^{\cell{k}}(t)}{\wcell{k}_{p, q}} + y_k^{\opig}(t + 1) \cdot \dgnetcell{p}{k}{t + 1} \cdot \begin{pmatrix}
x(t) \\
y^{\opig}(t) \\
y^{\opog}(t) \\
y^{\cell{1}}(t) \\
\vdots \\
y^{\cell{\ncell}}(t)
\end{pmatrix}_q}\Bigg]
\end{align*} \tag{9}\label{9}
$$

其中 $1 \leq k \leq \ncell$， $1 \leq p \leq \dcell$ 且 $1 \leq q \leq \din + \ncell \cdot (2 + \dcell)$。

計算完上述所有參數後使用**梯度下降**（gradient descent）進行參數更新

$$
\begin{align*}
\wout_{i, j} & \leftarrow \wout_{i, j} - \alpha \cdot \pd{\oploss(t + 1)}{\wout_{i, j}} \\
\wog_{k, q} & \leftarrow \wog_{k, q} - \alpha \cdot \pd{\oploss(t + 1)}{\wog_{k, q}} \\
\wig_{k, q} & \leftarrow \wig_{k, q} - \alpha \cdot \pd{\oploss(t + 1)}{\wig_{k, q}} \\
\wcell{k}_{p, q} & \leftarrow \wcell{k}_{p, q} - \alpha \cdot \pd{\oploss(t + 1)}{\wcell{k}_{p, q}}
\end{align*} \tag{10}\label{10}
$$

其中 $\alpha$ 為**學習率**（**learning rate**）。

由於使用基於 RTRL 的最佳化演算法，因此每個時間點 $t + 1$ 計算完誤差後就可以更新參數。

### 問題

當一個輸入序列中包含多個獨立的子序列（例如一個文章段落有多個句子），則模型無法知道不同獨立子序列的起始點在哪裡（除非有明確的切斷序列演算法，但實際上不一定存在）。

[原始 LSTM][LSTM1997] 架構假設任意輸入序列都是由單一獨立序列組成，不會包含多個獨立的序列，因此會在每次序列**輸入時重設模型的計算狀態** $y^{\opig}(0), y^{\opog}(0), s^{\cell{k}}(0), y^{\cell{k}}(0)$，沒有**需要在計算過程中重設計算狀態的需求**。

但當輸入包含多個獨立的子序列時，且沒有明確的方法辨識不同獨立子序列的起始點時，LSTM 模型就必須要擁有能夠在任意時間點 $t$ **重設計算狀態** $y^{\opig}(t), y^{\opog}(t), s^{\cell{k}}(t), y^{\cell{k}}(t)$ 的功能。

## 遺忘閘門

### 模型架構

<a name="paper-fig-1"></a>

圖 1：在原始 LSTM 架構上增加遺忘閘門。
圖片來源：[論文][論文]。

![圖 1](https://i.imgur.com/ILRsaEU.png)

作者提出在模型中加入**遺忘閘門**（**forget gate**），概念是讓**記憶單元內部狀態**能夠進行重設。

首先需要計算**遺忘閘門** $y^{\opfg}(t)$，定義如下

$$
\begin{align*}
\opnet^{\opfg}(t + 1) & = \wfg \cdot \begin{pmatrix}
x(t) \\
y^{\opfg}(t) \\
y^{\opig}(t) \\
y^{\opog}(t) \\
y^{\cell{1}}(t) \\
\vdots \\
y^{\cell{\ncell}}(t)
\end{pmatrix} \\
y^{\opfg}(0) & = 0 \\
y^{\opfg}(t + 1) & = f^{\opfg}(\opnet^{\opfg}(t + 1)) = \begin{pmatrix}
\fnetfg{1}{t + 1} \\
\fnetfg{2}{t + 1} \\
\vdots \\
\fnetfg{\ncell}{t + 1}
\end{pmatrix}
\end{align*} \tag{11}\label{11}
$$

計算方法與輸入閘門和輸出閘門相同。

而計算過程需要做以下修改

- $\eqref{1}\eqref{2}$ 中的淨輸入需要加上 $y^{\opfg}(t)$
- 參數 $\wig, \wog, \wcell{k}$ 的輸入維度都改成 $\din + \ncell \cdot (3 + \dcell)$
- $\wfg$ 的維度與 $\wig$ 完全相同
- $f^{\opfg}$ 與 $f^{\opig}$ 的定義完全相同

所謂的遺忘並不是直接設定成 $0$，而是以乘法閘門的形式進行數值重設，因此 $\eqref{2}$ 的計算改成

$$
s^{\cell{k}}(t + 1) = y_k^{\opfg}(t + 1) \cdot s^{\cell{k}}(t) + y_k^{\opig}(t + 1) \cdot g^{\cell{k}}(\opnet^{\cell{k}}(t + 1)) \tag{12}\label{12}
$$

### 偏差項

如同[原始 LSTM][LSTM1997]，**輸入閘門**與**輸出閘門**可以使用**偏差項**（bias term），將偏差項初始化成**負數**可以讓輸入閘門與輸出閘門在需要的時候才被啟用（細節可以看[我的筆記][note-LSTM1997]）。

而**遺忘閘門**也可以使用偏差項，但初始化的數值應該為**正數**，理由是在模型計算前期應該要讓遺忘閘門開啟（$y^{\opfg} \approx 1$），讓記憶單元內部狀態的數值能夠進行改變。

注意遺忘閘門只有在**關閉**（$y^{\opfg} \approx 0$）時才能進行遺忘，這個名字取得不是很好。

### 最佳化

基於[原始 LSTM][LSTM1997] 的最佳化演算法，將流出遺忘閘門的梯度也一起**丟棄**

$$
\begin{align*}
\pd{\netfg{k}{t + 1}}{y_{k^{\star}}^{\opfg}(t)} & \aptr 0 && k = 1, \dots, \ncell \\
\pd{\netfg{k}{t + 1}}{y_{k^{\star}}^{\opig}(t)} & \aptr 0 && k^{\star} = 1, \dots, \ncell \\
\pd{\netfg{k}{t + 1}}{y_{k^{\star}}^{\opog}(t)} & \aptr 0 \\
\pd{\netfg{k}{t + 1}}{y_i^{\cell{k^{\star}}}(t)} & \aptr 0 && i = 1, \dots, \dcell
\end{align*} \tag{13}\label{13}
$$

因此**遺忘閘門的參數**剩餘梯度為

$$
\begin{align*}
& \pd{\oploss(t + 1)}{\wfg_{k, q}} \\
& \aptr \sum_{i = 1}^{\dout} \Bigg[\pd{\oploss(t + 1)}{y_i(t + 1)} \cdot \pd{y_i(t + 1)}{\netout{i}{t + 1}} \cdot \\
& \quad \pa{\sum_{j = 1}^{\dcell} \pd{\netout{i}{t + 1}}{y_j^{\cell{k}}(t + 1)} \cdot \pd{y_j^{\cell{k}}(t + 1)}{s_j^{\cell{k}}(t + 1)} \cdot \pd{s_j^{\cell{k}}(t + 1)}{\wfg_{k, q}}}\Bigg] \\
& \aptr \sum_{i = 1}^{\dout} \Bigg[\pd{\oploss(t + 1)}{y_i(t + 1)} \cdot \pd{y_i(t + 1)}{\netout{i}{t + 1}} \cdot \Bigg(\sum_{j = 1}^{\dcell} \pd{\netout{i}{t + 1}}{y_j^{\cell{k}}(t + 1)} \cdot \pd{y_j^{\cell{k}}(t + 1)}{s_j^{\cell{k}}(t + 1)} \cdot \\
& \quad \quad \br{y_k^{\opfg}(t + 1) \cdot \pd{s_j^{\cell{k}}(t)}{\wfg_{k, q}} + s_j^{\cell{k}}(t) \cdot \pd{y_k^{\opfg}(t + 1)}{\netfg{k}{t + 1}} \cdot \pd{\netfg{k}{t + 1}}{\wfg_{k, q}}}\Bigg)\Bigg] \\
& \aptr \sum_{i = 1}^{\dout} \Bigg[\big(y_i(t + 1) - \hat{y}_i(t + 1)\big) \cdot \dfnetout{i}{t + 1} \cdot \\
& \quad \Bigg(\sum_{j = 1}^{\dcell} \wout_{i, \din + (k - 1) \cdot \dcell + j} \cdot y_k^{\opog}(t + 1) \cdot \dhcell{j}{k}{t + 1} \cdot \\
& \quad \quad \br{y_k^{\opfg}(t + 1) \cdot \pd{s_j^{\cell{k}}(t)}{\wfg_{k, q}}  + s_j^{\cell{k}}(t) \cdot \dfnetog{k}{t + 1} \cdot \begin{pmatrix}
x(t) \\
y^{\opfg}(t) \\
y^{\opig}(t) \\
y^{\opog}(t) \\
y^{\cell{1}}(t) \\
\vdots \\
y^{\cell{\ncell}}(t)
\end{pmatrix}_q}\Bigg)\Bigg]
\end{align*} \tag{14}\label{14}
$$

$\eqref{14}$ 式就是論文的 3.12 式，其中 $1 \leq k \leq \ncell$ 且 $1 \leq q \leq \din + \ncell \cdot (3 + \dcell)$。

由於 $\eqref{12}$ 的修改，$\eqref{9} \eqref{10}$ 最佳化的過程也需要跟著修改。

輸入閘門的參數剩餘梯度改為

$$
\begin{align*}
& \pd{\oploss(t + 1)}{\wig_{k, q}} \\
& \aptr \sum_{i = 1}^{\dout} \Bigg[\big(y_i(t + 1) - \hat{y}_i(t + 1)\big) \cdot \dfnetout{i}{t + 1} \cdot \\
& \quad \Bigg(\sum_{j = 1}^{\dcell} \wout_{i, \din + (k - 1) \cdot \dcell + j} \cdot y_k^{\opog}(t + 1) \cdot \dhcell{j}{k}{t + 1} \cdot \\
& \quad \quad \br{y_k^{\opfg}(t + 1) \cdot \pd{s_j^{\cell{k}}(t)}{\wig_{k, q}} + \gnetcell{j}{k}{t + 1} \cdot \dfnetig{k}{t + 1} \cdot \begin{pmatrix}
x(t) \\
y^{\opfg}(t) \\
y^{\opig}(t) \\
y^{\opog}(t) \\
y^{\cell{1}}(t) \\
\vdots \\
y^{\cell{\ncell}}(t)
\end{pmatrix}_q}\Bigg)\Bigg]
\end{align*} \tag{15}\label{15}
$$

$\eqref{14}$ 式就是論文的 3.11 式，其中 $1 \leq k \leq \ncell$ 且 $1 \leq q \leq \din + \ncell \cdot (3 + \dcell)$。

記憶單元淨輸入參數的剩餘梯度改為

$$
\begin{align*}
& \pd{\oploss(t + 1)}{\wcell{k}_{p, q}} \\
& \aptr \sum_{i = 1}^{\dout} \Bigg[\big(y_i(t + 1) - \hat{y}_i(t + 1)\big) \cdot \dfnetout{i}{t + 1} \cdot \wout_{i, \din + (k - 1) \cdot \dcell + j} \cdot \\
& \quad y_k^{\opog}(t + 1) \cdot \dhcell{j}{k}{t + 1} \cdot \\
& \quad \br{y_k^{\opfg}(t + 1) \cdot \pd{s_p^{\cell{k}}(t)}{\wcell{k}_{p, q}} + y_k^{\opig}(t + 1) \cdot \dgnetcell{p}{k}{t + 1} \cdot \begin{pmatrix}
x(t) \\
y^{\opfg}(t) \\
y^{\opig}(t) \\
y^{\opog}(t) \\
y^{\cell{1}}(t) \\
\vdots \\
y^{\cell{\ncell}}(t)
\end{pmatrix}_q}\Bigg]
\end{align*} \tag{16}\label{16}
$$

$\eqref{14}$ 式就是論文的 3.10 式，其中 $1 \leq k \leq \ncell$， $1 \leq p \leq \dcell$ 且 $1 \leq q \leq \din + \ncell \cdot (3 + \dcell)$。

**注意錯誤**：根據論文中的 3.4 式，論文 2.5 式的 $t - 1$ 應該改成 $t$。

根據 $\eqref{14}\eqref{15}\eqref{16}$，當遺忘閘門 $y_k^{\opfg}(t + 1) \approx 0$ （關閉）時，不只記憶單元 $s^{\cell{k}}(t + 1)$ 會重設，與其相關的梯度也會重設，因此更新時需要額外紀錄以下的項次

$$
\pd{s_i^{\cell{k}}(t + 1)}{\wfg_{k, q}}, \pd{s_i^{\cell{k}}(t + 1)}{\wig_{k, q}}, \pd{s_i^{\cell{k}}(t + 1)}{\wcell{k}_{p, q}}
$$

同樣的概念在[原始 LSTM][LSTM1997] 中也有出現，細節可以看[我的筆記][note-LSTM1997]。

## 實驗 1：Continual Embedded Reber Grammar

<a name="paper-fig-2"></a>

圖 2：Continual Embedded Reber Grammar。
圖片來源：[論文][論文]。

![圖 2](https://i.imgur.com/rhHtVRN.png)

### 任務定義

- 根據[原始 LSTM 論文][LSTM1997]中的實驗 1（Embedded Reber Grammar）進行修改，輸入為連續序列，連續序列的定義是由多個 Embedded Reber Grammar 產生的序列組合而成（細節可以看[我的筆記][note-LSTM1997]）
- 每個分支的生成機率值為 $0.5$
- 當所有輸出單元的平方誤差低於 $0.49$ 時就當成預測正確
- 在一次的訓練過程中，給予模型的輸入只會在以下兩種狀況之一發生時停止
  - 當模型產生一次的預測錯誤
  - 模型連續接收 $10^6$ 個輸入
- 每次訓練停止就進行一次測試
  - 一次測試會執行 $10$ 次的連續輸入
  - 評估結果是 $10$ 次連續輸入的平均值
- 每輸入一個訊號就進行更新（RTRL）
- 訓練最多執行 $30000$ 次，實驗結果由 $100$ 個訓練模型實驗進行平均

### LSTM 架構

<a name="paper-fig-3"></a>

圖 3：LSTM 架構。
圖片來源：[論文][論文]。

![圖 3](https://i.imgur.com/uUJjmSz.png)

|參數|數值（或範圍）|備註|
|-|-|-|
|$\din$|$7$||
|$\ncell$|$4$||
|$\dcell$|$2$||
|$\dout$|$7$||
|$\dim(\wcell{k})$|$\dcell \times [\din + \ncell \cdot \dcell]$|訊號來源為外部輸入與記憶單元|
|$\dim(\wfg)$|$\ncell \times [\din + \ncell \cdot \dcell + 1]$|訊號來源為外部輸入與記憶單元，有額外使用偏差項|
|$\dim(\wig)$|$\ncell \times [\din + \ncell \cdot \dcell + 1]$|訊號來源為外部輸入與記憶單元，有額外使用偏差項|
|$\dim(\wog)$|$\ncell \times [\din + \ncell \cdot \dcell + 1]$|訊號來源為外部輸入與記憶單元，有額外使用偏差項|
|$\dim(\wout)$|$\dout \times [\din + \ncell \cdot \dcell + 1]$|訊號來源為外部輸入與記憶單元，有額外使用偏差項|
|總參數量|$424$||
|參數初始化|$[-0.2, 0.2]$|平均分佈|
|輸入閘門偏差項初始化|$\set{-0.5, -1.0, -1.5, -2.0}$|依序初始化成不同數值|
|輸出閘門偏差項初始化|$\set{-0.5, -1.0, -1.5, -2.0}$|依序初始化成不同數值|
|遺忘閘門偏差項初始化|$\set{0.5, 1.0, 1.5, 2.0}$|依序初始化成不同數值|
|Learning rate $\alpha$|$0.5$|訓練過程可以固定 $\alpha$，或是以 $0.99$ 的 decay factor 在每次更新後進行衰減|

### 實驗結果

<a name="paper-fig-4"></a>

圖 4：Continual Embedded Reber Grammar 實驗結果。
圖片來源：[論文][論文]。

![圖 4](https://i.imgur.com/uu9Nccj.png)

- [原始 LSTM][LSTM1997] 在有手動進行計算狀態的重置時表現非常好，但當沒有手動重置時完全無法執行任務
  - 就算讓記憶單元內部狀態進行 decay 也無濟於事
- 使用遺忘閘門的 LSTM 不需要手動重置計算狀態也能達成完美預測
  - 完美預測指的是連續 $10^6$ 輸入都預測正確
- 有嘗試使用 $\alpha / t$ 或 $\alpha / \sqrt{T}$ 作為 learning rate，實驗發現不論是哪種最佳化的方法使用遺忘閘門的 LSTM 都表現的不錯
  - 在其他模型架構上（包含原版 LSTM）就算使用這些最佳化演算法也無法解決任務
- 額外實驗在將 Embedded Reber Grammar 開頭的 `B` 與結尾的 `E` 去除的狀態下，使用遺忘閘門的 LSTM 仍然表現不錯

### 分析

<a name="paper-fig-5"></a>

圖 5：[原版 LSTM][LSTM1997] 記憶單元內部狀態的累加值。
圖片來源：[論文][論文]。

![圖 5](https://i.imgur.com/qwU4pnG.png)

<a name="paper-fig-6"></a>

圖 6：LSTM 加上遺忘閘門後第三個記憶單元內部狀態。
圖片來源：[論文][論文]。

![圖 6](https://i.imgur.com/jtLnfu2.png)

<a name="paper-fig-7"></a>

圖 7：LSTM 加上遺忘閘門後第一個記憶單元內部狀態。
圖片來源：[論文][論文]。

![圖 7](https://i.imgur.com/K1mp9rg.png)

- 觀察[原版 LSTM][LSTM1997] 的記憶單元內部狀態，可以發現在不進行手動重設的狀態下，記憶單元內部狀態的數值只會不斷的累加（朝向極正或極負前進）
- 觀察架上遺忘閘門後 LSTM 的記憶單元內部狀態，可以發現模型學會自動重設
  - 在第三個記憶單元中展現了長期記憶重設的能力
  - 在第一個記憶單元中展現了短期記憶重設的能力

## 實驗 2：Noisy Temporal Order Problem

### 任務定義

- 就是[原始 LSTM 論文][LSTM1997]中的實驗 6b，細節可以看[我的筆記][note-LSTM1997]
- 由於此任務需要讓記憶維持一段不短的時間，因此遺忘資訊對於這個任務可能有害，透過這個任務想要驗證是否有任務是只能使用原版 LSTM 可以解決但增加遺忘閘門後不能解決

### LSTM 架構

與實驗 1 大致相同，只做以下修改

- $\din = \dout = 8$
- 將遺忘閘門的偏差項初始化成較大的正數（論文使用 $5$），讓遺忘閘門很難被關閉，藉此達到跟原本 LSTM 幾乎相同的計算能力

### 實驗結果

- 使用遺忘閘門的 LSTM 仍然能夠解決 Noisy Temporal Order Problem
  - 當偏差項初始化成較大的正數（例如 $5$）時，收斂速度與原版 LSTM 一樣快
  - 當偏差項初始化成較小的正數（例如 $1$）時，收斂速度約為原版 LSTM 的 $3$ 倍
- 因此根據實驗沒有什麼任務是原版 LSTM 可以解決但加上遺忘閘門後不能解決的

## 實驗 3：Continual Noisy Temporal Order Problem

### 任務定義

- 根據[原始 LSTM 論文][LSTM1997]中的實驗 6b 進行修改，輸入為連續序列，連續序列的定義是由 $100$ 筆 Noisy Temporal Order 序列所組成
- 在一次的訓練過程中，給予模型的輸入只會在以下兩種狀況之一發生時停止
  - 當模型產生一次的預測錯誤
  - 模型連續接收 $100$ 個 Noisy Temporal Order 序列
- 每次訓練停止就進行一次測試
  - 一次測試會執行 $10$ 次的連續輸入
  - 評估結果是 $10$ 次連續輸入中預測正確的序列個數平均值
- 論文沒有講怎麼計算誤差與更新，我猜變成每個非預測時間點必須輸出 $0$，預測時間點時輸出預測結果
- 訓練最多執行 $10^5$ 次，實驗結果由 $100$ 個訓練模型實驗進行平均

### LSTM 架構

與實驗 2 相同。

### 實驗結果

<a name="paper-fig-8"></a>

圖 8：Continual Noisy Temporal Order Problem 實驗結果。
圖片來源：[論文][論文]。

![圖 8](https://i.imgur.com/VV5wQVG.png)

- [圖 8](#paper-fig-8) 中的註解 a 應該寫錯了，應該改為 correct classification of 100 successive NTO sequences
- 實驗再次驗證原版 LSTM 無法解決連續輸入，但使用輸入閘門後就能夠解決問題
- 將 learning rate 使用 decay factor $0.9$ 逐漸下降可以讓模型表現變更好，但作者認為這不重要

[LSTM1997]: https://ieeexplore.ieee.org/abstract/document/6795963
[note-LSTM1997]: /deep%20learning/model%20architecture/optimization/2021/11/14/long-short-term-memory.html
[論文]: https://direct.mit.edu/neco/article-abstract/12/10/2451/6415/Learning-to-Forget-Continual-Prediction-with-LSTM
[LSTM2002]: https://www.jmlr.org/papers/v3/gers02a.html
[note-LSTM2002]: /deep%20learning/model%20architecture/2021/12/29/learning-precise-timing-with-lstm-recurrent-networks.html
