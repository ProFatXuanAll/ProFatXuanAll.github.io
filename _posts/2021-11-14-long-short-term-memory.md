---
layout: post
title:  "Long Short-Term Memory"
date:   2021-11-14 15:28:00 +0800
categories: [
  Deep Learning,
  Model Architecture,
  Optimization,
]
tags: [
  LSTM,
  BPTT,
  Gradient Explosion,
  Gradient Vanishing,
]
author: [
  Sepp Hochreiter, Jürgen Schmidhuber
]
---

|-|-|
|論文連結|<https://ieeexplore.ieee.org/abstract/document/6795963>|
|書本連結|<https://link.springer.com/chapter/10.1007/978-3-642-24797-2_4>|
|期刊/會議名稱|Neural Computation|
|發表時間|1997|
|作者|Sepp Hochreiter, Jürgen Schmidhuber|
|目標|提出 RNN 使用 BPTT 進行最佳化時遇到的問題，並提出 LSTM 架構進行修正|

<!--
  Define LaTeX command which will be used through out the writing.

  First we need to include `tools/math` which setup auto rendering.
  Each command must be wrapped with $ signs.
  We use "display: none;" to avoid redudant whitespaces.
 -->

{% include tools/math.html %}

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
  <!-- Operator hid. -->
  $\providecommand{\ophid}{}$
  $\renewcommand{\ophid}{\operatorname{hid}}$
  <!-- Operator cell. -->
  $\providecommand{\opcell}{}$
  $\renewcommand{\opcell}{\operatorname{cell}}$
  <!-- Operator cell multiplicative input gate. -->
  $\providecommand{\opig}{}$
  $\renewcommand{\opig}{\operatorname{ig}}$
  <!-- Operator cell multiplicative output gate. -->
  $\providecommand{\opog}{}$
  $\renewcommand{\opog}{\operatorname{og}}$

  <!-- Total loss. -->
  $\providecommand{\Loss}{}$
  $\renewcommand{\Loss}[1]{\operatorname{loss}(#1)}$
  <!-- Partial loss. -->
  $\providecommand{\loss}{}$
  $\renewcommand{\loss}[2]{\operatorname{loss}_{#1}(#2)}$

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
  <!-- Hidden dimension. -->
  $\providecommand{\dhid}{}$
  $\renewcommand{\dhid}{d_{\ophid}}$
  <!-- Cell dimension. -->
  $\providecommand{\dcell}{}$
  $\renewcommand{\dcell}{d_{\opcell}}$

  <!-- Number of cells. -->
  $\providecommand{\ncell}{}$
  $\renewcommand{\ncell}{n_{\opcell}}$

  <!-- Past and Future time -->
  $\providecommand{\tp}{}$
  $\renewcommand{\tp}{t_{\operatorname{past}}}$
  $\providecommand{\tf}{}$
  $\renewcommand{\tf}{t_{\operatorname{future}}}$
  <!-- Graident of loss(t_2) with respect to net k_0 at time t_1. -->
  $\providecommand{\dv}{}$
  $\renewcommand{\dv}[3]{\vartheta_{#1}^{#2}[#3]}$

  <!-- Cell block k. -->
  $\providecommand{\cell}{}$
  $\renewcommand{\cell}[1]{\opcell^{#1}}$

  <!-- Weight of multiplicative input gate. -->
  $\providecommand{\wig}{}$
  $\renewcommand{\wig}{w^{\opig}}$
  <!-- Weight of multiplicative output gate. -->
  $\providecommand{\wog}{}$
  $\renewcommand{\wog}{w^{\opog}}$
  <!-- Weight of hidden units. -->
  $\providecommand{\whid}{}$
  $\renewcommand{\whid}{w^{\ophid}}$
  <!-- Weight of cell units. -->
  $\providecommand{\wcell}{}$
  $\renewcommand{\wcell}[1]{w^{\cell{#1}}}$
  <!-- Weight of output units. -->
  $\providecommand{\wout}{}$
  $\renewcommand{\wout}{w^{\opout}}$

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
  <!-- Net input of hidden unit. -->
  $\providecommand{\nethid}{}$
  $\renewcommand{\nethid}[2]{\opnet_{#1}^{\ophid}(#2)}$
  <!-- Net input of hidden unit with activatiton f. -->
  $\providecommand{\fnethid}{}$
  $\renewcommand{\fnethid}[2]{f_{#1}^{\ophid}\big(\nethid{#1}{#2}\big)}$
  <!-- Derivative of f with respect to net input of hidden units. -->
  $\providecommand{\dfnethid}{}$
  $\renewcommand{\dfnethid}[2]{f_{#1}^{\ophid}{'}\big(\nethid{#1}{#2}\big)}$
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
  $\renewcommand{\hcell}[3]{h_{#1}\big(\cell{#2}_{#1}(#3)\big)}$
  <!-- Derivative of h with respect to cell unit. -->
  $\providecommand{\dhcell}{}$
  $\renewcommand{\dhcell}[3]{h_{#1}'\big(\cell{#2}_{#1}(#3)\big)}$

  <!-- Gradient approximation by truncating gradient. -->
  $\providecommand{\aptr}{}$
  $\renewcommand{\aptr}{\approx_{\operatorname{tr}}}$
</p>

<!-- End LaTeX command define section. -->

## 重點

- 對 RNN 進行梯度反向傳播的演算法通常稱為 **BPTT** （**B**ack-**P**ropagation **T**hrought **T**ime） 或 real-time recurrent learning
  - RNN 透過 back-propagation 學習**效率差**
  - 梯度會**爆炸**或**消失**
    - 梯度爆炸造成神經網路的**權重劇烈振盪**
    - 梯度消失造成**訓練時間慢長**
  - 無法解決輸入與輸出訊號**間隔較長**（long time lag）的問題
- 論文提出 **LSTM + RTRL** 能夠解決上述問題
  - 能夠處理 time lag 間隔為 $1000$ 的問題
  - 甚至輸入訊號含有雜訊時也能處理
  - 同時能夠保有處理 short time lag 問題的能力
- 使用 mulitplicative gate 學習開啟與關閉記憶 hidden state 的機制
  - Forward pass 演算法複雜度為 $O(1)$
  - Backward pass 演算法複雜度為 $O(w)$，$w$ 代表權重
- 與 [Pytorch](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html?highlight=lstm#torch.nn.LSTM) 實作的 LSTM 完全不同
  - 本篇論文的架構定義更為**廣義**
  - 本篇論文只有**輸入閘門（input gate）**跟**輸出閘門（output gate）**，並沒有使用**失憶閘門（forget gate）**

## 傳統的 RNN

### BPTT

BPTT = **B**ack **P**ropagation **T**hrough **T**ime，是專門用來計算 RNN 神經網路模型的**梯度反向傳播演算法**。
一個 RNN 模型的**輸入**來源共有兩種：

- **外部輸入（external input）** $x(t)$
  - 輸入維度為 $\din$
  - 使用下標 $x_{j}(t)$ 代表不同的輸入訊號，$j = 1, \dots, \din$
- **前次輸出（previous output）** $y(t)$
  - 輸出維度為 $\dout$
  - 使用下標 $y_{j}(t)$ 代表不同的輸入訊號，$j = \din + 1, \dots, \dout$
  - 注意這裡是使用 $t$ 不是 $t - 1$
- $t$ 的起始值為 $0$，結束值為 $T$，每次遞增 $1$
  - 時間為離散狀態
  - 方便起見令 $y(0) = 0$

令 RNN 模型的參數為 $w \in \R^{\dout \times (\din + \dout)}$，如果我們已經取得 $t$ 時間點的**外部輸入** $x(t)$ 與**前次輸出** $y(t)$，則我們可以定義 $t + 1$ 時間點的第 $i$ 個**模型內部節點** $\net{i}{t}$

$$
\begin{align*}
  \net{i}{t + 1} & = \sum_{j = 1}^{\din} w_{i j} \cdot x_{j}(t) + \sum_{j = \din + 1}^{\din + \dout} w_{i j} \cdot y_{j}(t) \\
  & = \sum_{j = 1}^{\din + \dout} w_{i j} \cdot [x ; y]_{j}(t)
\end{align*} \tag{1}\label{eq:1}
$$

- $\net{i}{t + 1}$ 代表第 $t + 1$ 時間的**模型內部節點** $i$ 所收到的**淨輸入（total input）**
  - 注意 $t$ 時間點的輸入訊號變成 $t + 1$ 時間點的輸出結果
  - 這是早年常見的 RNN 公式表達法
- $w_{i j}$ 代表**輸入節點** $j$與**模型內部節點** $i$ 所連接的權重
  - 輸入節點可以是**外部輸入** $x_{j}(t)$ 或是**前次輸出** $y_{j}(t)$
  - 總共有 $\din + \dout$ 個輸入節點，因此 $1 \leq j \leq \din + \dout$
  - 總共有 $\dout$ 個內部節點，因此 $1 \leq i \leq \dout$
- $[x ; y]$ 代表將外部輸入與前次輸出**串接**在一起

令模型使用的**啟發函數**（activation function）為 $f : \R^{\dout} \to \R^{\dout}$，並且內部節點之間無法直接溝通（elementwise activation function），則我們可以得到 $t + 1$ 時間的輸出

$$
y_{i}(t + 1) = \fnet{i}{t + 1} \tag{2}\label{eq:2}
$$

- 使用下標 $f_{i}$ 是因為每個維度所使用的啟發函數可以**不同**
- $f$ 必須要可以**微分**
- 當時幾乎都是使用 sigmoid 函數 $\sigma(x) = 1 / (1 + e^{-x})$

如果 $t + 1$ 時間點的**輸出目標**為 $\hat{y}(t + 1) \in \R^{\dout}$，則**目標函數**為**最小平方差**（Mean Square Error）：

$$
\loss{i}{t + 1} = \frac{1}{2} \big(y_{i}(t + 1) - \hat{y}_{i}(t + 1)\big)^2 \tag{3}\label{eq:3}
$$

因此 $t + 1$ 時間點的總體目標函數（總誤差）為

$$
\Loss{t + 1} = \sum_{i = 1}^{\dout} \loss{i}{t + 1} \tag{4}\label{eq:4}
$$

根據 $\eqref{eq:3} \eqref{eq:4}$ 我們可以輕易的計算 $\loss{i}{t + 1}$ 對 $\Loss{t + 1}$ 所得梯度

$$
\pd{\Loss{t + 1}}{\loss{i}{t + 1}} = 1 \tag{5}\label{eq:5}
$$

而透過 $\eqref{eq:3}$ 我們知道 $y_{i}(t + 1)$ 對 $\Loss{t + 1}$ 所得梯度為

$$
\pd{\loss{i}{t + 1}}{y_{i}(t + 1)} = y_{i}(t + 1) - \hat{y}_{i}(t + 1) \tag{6}\label{eq:6}
$$

根據 $\eqref{eq:5} \eqref{eq:6}$ 我們可以推得

$$
\begin{align*}
\pd{\Loss{t + 1}}{y_{i}(t + 1)} & = \pd{\Loss{t + 1}}{\loss{i}{t + 1}} \cdot \pd{\loss{i}{t + 1}}{y_{i}(t + 1)} \\
& = 1 \cdot \big(y_{i}(t + 1) - \hat{y}_{i}(t + 1)\big) \\
& = y_{i}(t + 1) - \hat{y}_{i}(t + 1)
\end{align*} \tag{7}\label{eq:7}
$$

根據 $\eqref{eq:2}$ 我們知道 $\net{i}{t + 1}$ 對 $y_{i}(t + 1)$ 所得梯度為

$$
\pd{y_{i}(t + 1)}{\net{i}{t + 1}} = \dfnet{i}{t + 1} \tag{8}\label{eq:8}
$$

根據 $\eqref{eq:7} \eqref{eq:8}$ 我們可以推得 $\net{i}{t + 1}$ 對 $\Loss{t + 1}$ 所得梯度

$$
\begin{align*}
\pd{\Loss{t + 1}}{\net{i}{t + 1}} & = \pd{\Loss{t + 1}}{y_{i}(t + 1)} \cdot \pd{y_{i}(t + 1)}{\net{i}{t + 1}} \\
& = \dfnet{i}{t + 1} \cdot \big(y_{i}(t + 1) - \hat{y}_{i}(t + 1)\big)
\end{align*} \tag{9}\label{eq:9}
$$

式子 $\eqref{eq:9}$ 就是論文 3.1.1 節的第一條公式。
根據 $\eqref{eq:9}$ 我們可以推得 $x_{j}(t)$ 對 $\Loss{t + 1}$ 所得梯度

$$
\begin{align*}
\pd{\Loss{t + 1}}{x_{j}(t)} & = \sum_{i = 1}^{\dout} \bigg[\pd{\Loss{t + 1}}{\net{i}{t + 1}} \cdot \pd{\net{i}{t + 1}}{x_{j}(t)}\bigg] \\
& = \sum_{i = 1}^{\dout} \bigg[\dfnet{i}{t + 1} \cdot \big(y_{i}(t + 1) - \hat{y}_{i}(t + 1)\big) \cdot w_{i j}\bigg]
\end{align*} \tag{10}\label{eq:10}
$$

同樣的根據 $\eqref{eq:9}$ 我們可以推得 $y_{j}(t)$ 對 $\Loss{t + 1}$ 所得梯度為

$$
\begin{align*}
\pd{\Loss{t + 1}}{y_{j}(t)} & = \sum_{i = 1}^{\dout} \bigg[\pd{\Loss{t + 1}}{\net{i}{t + 1}} \cdot \pd{\net{i}{t + 1}}{y_{j}(t)}\bigg] \\
& = \sum_{i = 1}^{\dout} \bigg[\dfnet{i}{t + 1} \cdot \big(y_{i}(t + 1) - \hat{y}_{i}(t + 1)\big) \cdot w_{i j}\bigg]
\end{align*} \tag{11}\label{eq:11}
$$

由於第 $t$ 時間點的輸出 $y_{j}(t)$ 的計算是由 $\net{j}{t}$ 而來（請見 $\eqref{eq:2}$），所以我們也利用 $\eqref{eq:8} \eqref{eq:11}$ 計算 $\net{j}{t}$ 對 $\Loss{t + 1}$ 所得梯度（注意是 $t$ 不是 $t + 1$）

$$
\begin{align*}
& \pd{\Loss{t + 1}}{\net{j}{t}} \\
& = \pd{\Loss{t + 1}}{y_{j}(t)} \cdot \pd{y_{j}(t)}{\net{j}{t}} \\
& = \sum_{i = 1}^{\dout} \bigg[\dfnet{i}{t + 1} \cdot \big(y_{i}(t + 1) - \hat{y}_{i}(t + 1)\big) \cdot w_{i j} \cdot \dfnet{j}{t}\bigg] \\
& = \dfnet{j}{t} \cdot \sum_{i = 1}^{\dout} \bigg[w_{i j} \cdot \dfnet{i}{t + 1} \cdot \big(y_{i}(t + 1) - \hat{y}_{i}(t + 1)\big)\bigg] \\
& = \dfnet{j}{t} \cdot \sum_{i = 1}^{\dout} \bigg[w_{i j} \cdot \pd{\Loss{t + 1}}{\net{i}{t + 1}}\bigg]
\end{align*} \tag{12}\label{eq:12}
$$

式子 $\eqref{eq:12}$ 就是論文 3.1.1 節的最後一條公式。
模型參數 $w_{i j}$ 對於 $\Loss{t + 1}$ 所得梯度為

$$
\begin{align*}
& \pd{\Loss{t + 1}}{w_{i j}} \\
& = \pd{\Loss{t + 1}}{\net{i}{t + 1}} \cdot \pd{\net{i}{t + 1}}{w_{i j}} \\
& = \dfnet{i}{t + 1} \cdot \big(y_{i}(t + 1) - \hat{y}_{i}(t + 1)\big) \cdot [x ; y]_{j}(t) && \text{(by \eqref{eq:9})}
\end{align*} \tag{13}\label{eq:13}
$$

注意 $\eqref{eq:13}$ 中最後一行等式取決於 $w_{i j}$ 與哪個輸入相接。
而在時間點 $t + 1$ 進行參數更新的方法為

$$
w_{i j} \leftarrow w_{i j} - \alpha \pd{\Loss{t + 1}}{w_{i j}} \tag{14}\label{eq:14}
$$

$\eqref{eq:14}$ 就是最常用來最佳化神經網路的**梯度下降演算法**（Gradient Descent），$\alpha$ 代表**學習率**（Learning Rate）。

### 梯度爆炸 / 消失

從 $\eqref{eq:12}$ 式我們可以進一步推得 $t$ 時間點造成的梯度與前次時間點 ($t - 1, t - 2, \dots$) 所得的梯度**變化關係**。
注意這裡的變化關係指的是梯度與梯度之間的**變化率**，意即用時間點 $t - 1$ 的梯度對時間點 $t$ 的梯度算微分。

為了方便計算，我們定義新的符號

$$
\dv{k}{\tf}{\tp} \tag{15}\label{eq:15}
$$

意思是從**未來**時間點 $\tf$ 開始往回走到**過去**時間點 $\tp$，在**過去**時間點 $\tp$ 的第 $k$ 個**模型內部節點** $\net{k}{\tp}$ 對於**未來**時間點 $\tf$ 貢獻的**總誤差** $\Loss{\tf}$ 計算所得之**梯度**。

- 注意是貢獻總誤差所得之**梯度**
- 根據時間的限制我們有不等式 $0 \leq \tp \leq \tf$
- 節點 $k$ 的數值範圍為 $k = 1, \dots, \dout$，見式子 $\eqref{eq:1}$

因此下式如同 $\eqref{eq:9}$ 式

$$
\dv{k_{0}}{t}{t} = \pd{\Loss{t}}{\net{k_{0}}{t}} = \dfnet{k_{0}}{t} \cdot \big(y_{k_{0}}(t) - \hat{y}_{k_{0}}(t)\big) \tag{16}\label{eq:16}
$$

由 $\eqref{eq:12}$ 與 $\eqref{eq:16}$ 我們可以往回推 1 個時間點

$$
\begin{align*}
\dv{k_{1}}{t}{t - 1} & = \pd{\Loss{t}}{\net{k_{1}}{t - 1}} \\
& = \dfnet{k_{1}}{t - 1} \cdot \sum_{k_{0} = 1}^{\dout} \bigg[w_{k_{0} k_{1}} \cdot \pd{\Loss{t}}{\net{k_{0}}{t}}\bigg] \\
& = \sum_{k_{0} = 1}^{\dout} \bigg[w_{k_{0} k_{1}} \cdot \dfnet{k_{1}}{t - 1} \cdot \dv{k_{0}}{t}{t}\bigg]
\end{align*} \tag{17}\label{eq:17}
$$

由 $\eqref{eq:17}$ 我們可以往回推 2 個時間點

$$
\begin{align*}
& \dv{k_{2}}{t}{t - 2} \\
& = \pd{\Loss{t}}{\net{k_{2}}{t - 2}} \\
& = \sum_{k_{1} = 1}^{\dout} \bigg[\pd{\Loss{t}}{\net{k_{1}}{t - 1}} \cdot \pd{\net{k_{1}}{t - 1}}{\net{k_{2}}{t - 2}}\bigg] \\
& = \sum_{k_{1} = 1}^{\dout} \bigg[\dv{k_{1}}{t}{t - 1} \cdot \pd{\net{k_{1}}{t - 1}}{y_{k_{2}}(t - 2)} \cdot \pd{y_{k_{2}}(t - 2)}{\net{k_{2}}{t - 2}}\bigg] \\
& = \sum_{k_{1} = 1}^{\dout} \bigg[\dv{k_{1}}{t}{t - 1} \cdot w_{k_{1} k_{2}} \cdot \dfnet{k_{2}}{t - 2}\bigg] \\
& = \sum_{k_{1} = 1}^{\dout} \Bigg[\dfnet{k_{1}}{t - 1} \cdot \sum_{k_{0} = 1}^{\dout} \bigg(w_{k_{0} k_{1}} \cdot \dv{k_{0}}{t}{t}\bigg) \cdot w_{k_{1} k_{2}} \cdot \dfnet{k_{2}}{t - 2}\Bigg] \\
& = \sum_{k_{1} = 1}^{\dout} \sum_{k_{0} = 1}^{\dout} \bigg[w_{k_{0} k_{1}} \cdot w_{k_{1} k_{2}} \cdot \dfnet{k_{1}}{t - 1} \cdot \dfnet{k_{2}}{t - 2} \cdot \dv{k_{0}}{t}{t}\bigg]
\end{align*} \tag{18}\label{eq:18}
$$

由 $\eqref{eq:18}$ 我們可以往回推 3 個時間點

$$
\begin{align*}
& \dv{k_{3}}{t}{t - 3} \\
& = \pd{\Loss{t}}{\net{k_{3}}{t - 3}} \\
& = \sum_{k_{2} = 1}^{\dout} \bigg[\pd{\Loss{t}}{\net{k_{2}}{t - 2}} \cdot \pd{\net{k_{2}}{t - 2}}{\net{k_{3}}{t - 3}}\bigg] \\
& = \sum_{k_{2} = 1}^{\dout} \bigg[\dv{k_{2}}{t}{t - 2} \cdot \pd{\net{k_{2}}{t - 2}}{y_{k_{3}}(t - 3)} \cdot \pd{y_{k_{3}}(t - 3)}{\net{k_{3}}{t - 3}}\bigg] \\
& = \sum_{k_{2} = 1}^{\dout} \bigg[\dv{k_{2}}{t}{t - 2} \cdot w_{k_{2} k_{3}} \cdot \dfnet{k_{3}}{t - 3}\bigg] \\
& = \sum_{k_{2} = 1}^{\dout} \Bigg[\sum_{k_{1} = 1}^{\dout} \sum_{k_{0} = 1}^{\dout} \bigg[w_{k_{0} k_{1}} \cdot w_{k_{1} k_{2}} \cdot \dfnet{k_{1}}{t - 1} \cdot \dfnet{k_{2}}{t - 2} \cdot \dv{k_{0}}{t}{t}\bigg] \\
& \quad \cdot w_{k_{2} k_{3}} \cdot \dfnet{k_{3}}{t - 3}\Bigg] \\
& = \sum_{k_{2} = 1}^{\dout} \sum_{k_{1} = 1}^{\dout} \sum_{k_{0} = 1}^{\dout} \bigg[w_{k_{0} k_{1}} \cdot w_{k_{1} k_{2}} \cdot w_{k_{2} k_{3}} \cdot \\
& \quad \dfnet{k_{1}}{t - 1} \cdot \dfnet{k_{2}}{t - 2} \cdot \dfnet{k_{3}}{t - 3} \cdot \dv{k_{0}}{t}{t}\bigg] \\
& = \sum_{k_{2} = 1}^{\dout} \sum_{k_{1} = 1}^{\dout} \sum_{k_{0} = 1}^{\dout} \Bigg[\bigg[\prod_{q = 1}^{3} w_{k_{q - 1} k_{q}} \cdot \dfnet{k_{q}}{t - q}\bigg] \cdot \dv{k_{0}}{t}{t}\Bigg]
\end{align*} \tag{19}\label{eq:19}
$$

由 $\eqref{eq:17} \eqref{eq:18} \eqref{eq:19}$ 我們可以歸納以下結論：
若 $n \geq 1$，則往回推 $n$ 個時間點的公式為

$$
\dv{k_{n}}{t}{t - n} = \sum_{k_{n - 1} = 1}^{\dout} \cdots \sum_{k_{0} = 1}^{\dout} \Bigg[\bigg[\prod_{q = 1}^{n} w_{k_{q - 1} k_{q}} \cdot \dfnet{k_{q}}{t - q}\bigg] \cdot \dv{k_{0}}{t}{t}\Bigg] \tag{20}\label{eq:20}
$$

由 $\eqref{eq:20}$ 我們可以看出所有的 $\dv{k_{n}}{t}{t - n}$ 都與 $\dv{k_{0}}{t}{t}$ 相關，因此我們將 $\dv{k_{n}}{t}{t - n}$ 想成由 $\dv{k_{0}}{t}{t}$ 構成的函數。

現在讓我們固定 $k_{0}^* \in \set{1, \dots, \dout}$，我們可以計算 $\dv{k_{0}^*}{t}{t}$ 對於 $\dv{k_{n}}{t}{t - n}$ 的微分

- 當 $n = 1$ 時，根據 $\eqref{eq:17}$ 我們可以推得論文中的 (3.1) 式

  $$
  \pd{\dv{k_{n}}{t}{t - n}}{\dv{k_{0}^*}{t}{t}} = w_{k_{0}^* k_{1}} \cdot \dfnet{k_{1}}{t - 1} \tag{21}\label{eq:21}
  $$

- 當 $n > 1$ 時，根據 $\eqref{eq:20}$ 我們可以推得論文中的 (3.2) 式

  $$
  \pd{\dv{k_{n}}{t}{t - n}}{\dv{k_{0}^*}{t}{t}} = \sum_{k_{n - 1} = 1}^{\dout} \cdots \sum_{k_{1} = 1}^{\dout} \sum_{k_{0} \in \set{k_{0}^*}} \bigg[\prod_{q = 1}^{n} w_{k_{q - 1} k_{q}} \cdot \dfnet{k_{q}}{t - q}\bigg] \tag{22}\label{eq:22}
  $$

**注意錯誤**：論文中的 (3.2) 式不小心把 $w_{l_{m - 1} l_{m}}$ 寫成 $w_{l_{m} l_{m - 1}}$。

因此根據 $\eqref{eq:22}$，共有 $(\dout)^{n - 1}$ 個連乘積項次進行加總，所得結果會以 $\eqref{eq:13} \eqref{eq:14}$ 直接影響權種更新 $w$。

根據 $\eqref{eq:21} \eqref{eq:22}$，如果

$$
\abs{w_{k_{q - 1} k_{q}} \cdot \dfnet{k_{q}}{t - q}} > 1.0 \quad \forall q = 1, \dots, n \tag{23}\label{eq:23}
$$

則 $w$ 的梯度會以指數 $n$ 增加，直接導致**梯度爆炸**，參數會進行**劇烈的振盪**，無法進行順利更新。

而如果

$$
\abs{w_{k_{q - 1} k_{q}} \cdot \dfnet{k_{q}}{t - q}} < 1.0 \quad \forall q = 1, \dots, n \tag{24}\label{eq:24}
$$

則 $w$ 的梯度會以指數 $n$ 縮小，直接導致**梯度消失**，誤差**收斂速度**會變得**非常緩慢**。

如果 $f_{k_{q}}$ 是 sigmoid function $\sigma$，則 $\sigma'$ 最大值為 $0.25$，理由是

$$
\begin{align*}
\sigma(x) & = \frac{1}{1 + e^{-x}} \\
\sigma'(x) & = \frac{e^{-x}}{(1 + e^{-x})^2} = \frac{1}{1 + e^{-x}} \cdot \frac{e^{-x}}{1 + e^{-x}} \\
& = \frac{1}{1 + e^{-x}} \cdot \frac{1 + e^{-x} - 1}{1 + e^{-x}} = \sigma(x) \cdot \big(1 - \sigma(x)\big) \\
\sigma(\R) & = (0, 1) \\
\forall x \in \R, \max \sigma'(x) & = \sigma(0) * \big(1 - \sigma(0)\big) = 0.5 * 0.5 = 0.25
\end{align*} \tag{25}\label{eq:25}
$$

因此當 $\abs{w_{k_{q - 1} k_{q}}} < 4.0$ 時我們可以發現

$$
\abs{w_{k_{q - 1} k_{q}} \cdot \dfnet{k_{q}}{t - q}} < 4.0 * 0.25 = 1.0 \tag{26}\label{eq:26}
$$

所以 $\eqref{eq:26}$ 與 $\eqref{eq:24}$ 的結論相輔相成：當 $w_{k_{q - 1} k_{q}}$ 的絕對值小於 $4.0$ 會造成梯度消失。

而 $\abs{w_{k_{q - 1} k_{q}}} \to \infty$ 我們可以得到

$$
\begin{align*}
& \abs{\net{k_{q - 1}}{t - q - 1}} \to \infty \\
\implies & \begin{cases}
\fnet{k_{q - 1}}{t - q - 1} \to 1 & \text{if } \net{k_{q - 1}}{t - q - 1} \to \infty \\
\fnet{k_{q - 1}}{t - q - 1} \to 0 & \text{if } \net{k_{q - 1}}{t - q - 1} \to -\infty
\end{cases} \\
\implies & \abs{\dfnet{k_{q - 1}}{t - q - 1}} \to 0 && \text{(by \eqref{eq:25})} \\
\implies & \abs{\prod_{q = 1}^{n} w_{k_{q - 1} k_{q}} \cdot \dfnet{k_{q}}{t - q}} \to 0
\end{align*} \tag{27}\label{eq:27}
$$

**注意錯誤**：論文中的推論

$$
\abs{w_{k_{q - 1} k_{q}} \cdot \dfnet{k_{q}}{t - q}} \to 0
$$

是**錯誤**的，理由是 $w_{k_{q - 1} k_{q}}$ 無法對 $\net{k_{q}}{t - q}$ 造成影響，作者不小心把**時間順序寫反**了，但是**最後的邏輯仍然正確**，理由如 $\eqref{eq:27}$ 所示。

**注意錯誤**：論文中進行了以下**函數最大值**的推論

$$
\begin{align*}
& \dfnet{l_{m}}{t - m}\big) \cdot w_{l_{m} l_{m - 1}} \\
& = \sigma\big(\net{l_{m}}{t - m}\big) \cdot \Big(1 - \sigma\big(\net{l_{m}}{t - m}\big)\Big) \cdot w_{l_{m} l{m - l}}
\end{align*}
$$

最大值發生於微分值為 $0$ 的點，即我們想求出滿足以下式子的 $w_{l_{m} l_{m - 1}}$

$$
\pd{\Big[\sigma\big(\net{l_{m}}{t - m}\big) \cdot \Big(1 - \sigma\big(\net{l_{m}}{t - m}\big)\Big) \cdot w_{l_{m} l{m - l}}\Big]}{w_{l_{m} l_{m - 1}}} = 0
$$

拆解微分式可得

$$
\begin{align*}
& \pd{\Big[\sigma\big(\net{l_{m}}{t - m}\big) \cdot \Big(1 - \sigma\big(\net{l_{m}}{t - m}\big)\Big) \cdot w_{l_{m} l{m - l}}\Big]}{w_{l_{m} l_{m - 1}}} \\
& = \pd{\sigma\big(\net{l_{m}}{t - m}\big)}{\net{l_{m}}{t - m}} \cdot \pd{\net{l_{m}}{t - m}}{w_{l_{m} l_{m - 1}}} \cdot \Big(1 - \sigma\big(\net{l_{m}}{t - m}\big)\Big) \cdot w_{l_{m} l{m - l}} \\
& \quad + \sigma\big(\net{l_{m}}{t - m}\big) \cdot \pd{\Big(1 - \sigma\big(\net{l_{m}}{t - m}\big)\Big)}{\net{l_{m}}{t - m}} \cdot \pd{\net{l_{m}}{t - m}}{w_{l_{m} l_{m - 1}}} \cdot w_{l_{m} l{m - l}} \\
& \quad + \sigma\big(\net{l_{m}}{t - m}\big) \cdot \Big(1 - \sigma\big(\net{l_{m}}{t - m}\big)\Big) \cdot \pd{w_{l_{m} l_{m - 1}}}{w_{l_{m} l_{m - 1}}} \\
& = \sigma\big(\net{l_{m}}{t - m}\big) \cdot \Big(1 - \sigma\big(\net{l_{m}}{t - m}\big)\Big)^2 \cdot y_{l_{m - 1}}(t - m - 1) \cdot w_{l_{m} l_{m - 1}} \\
& \quad - \Big(\sigma\big(\net{l_{m}}{t - m}\big)\Big)^2 \cdot \Big(1 - \sigma\big(\net{l_{m}}{t - m}\big)\Big) \cdot y_{l_{m - 1}}(t - m - 1) \cdot w_{l_{m} l_{m - 1}} \\
& \quad + \sigma\big(\net{l_{m}}{t - m}\big) \cdot \Big(1 - \sigma\big(\net{l_{m}}{t - m}\big)\Big) \\
& = \Big[2 \Big(\sigma\big(\net{l_{m}}{t - m}\big)\Big)^3 - 3 \Big(\sigma\big(\net{l_{m}}{t - m}\big)\Big)^2 + \sigma\big(\net{l_{m}}{t - m}\big)\Big] \cdot \\
& \quad \quad y_{l_{m - 1}}(t - m - 1) \cdot w_{l_{m} l_{m - 1}} \\
& \quad + \sigma\big(\net{l_{m}}{t - m}\big) \cdot \Big(1 - \sigma\big(\net{l_{m}}{t - m}\big)\Big) \\
& = \sigma\big(\net{l_{m}}{t - m}\big) \cdot \Big(2 \sigma\big(\net{l_{m}}{t - m}\big) - 1\Big) \cdot \Big(\sigma\big(\net{l_{m}}{t - m}\big) - 1\Big) \cdot \\
& \quad \quad y_{l_{m - 1}}(t - m - 1) \cdot w_{l_{m} l_{m - 1}} \\
& \quad + \sigma\big(\net{l_{m}}{t - m}\big) \cdot \Big(1 - \sigma\big(\net{l_{m}}{t - m}\big)\Big) \\
& = 0
\end{align*}
$$

移項後可以得到

$$
\begin{align*}
& \sigma\big(\net{l_{m}}{t - m}\big) \cdot \Big(2 \sigma\big(\net{l_{m}}{t - m}\big) - 1\Big) \cdot \Big(1 - \sigma\big(\net{l_{m}}{t - m}\big)\Big) \cdot \\
& \quad \quad y_{l_{m - 1}}(t - m - 1) \cdot w_{l_{m} l_{m - 1}} = \sigma\big(\net{l_{m}}{t - m}\big) \cdot \Big(1 - \sigma\big(\net{l_{m}}{t - m}\big)\Big) \\
\implies & \Big(2 \sigma\big(\net{l_{m}}{t - m}\big) - 1\Big) \cdot y_{l_{m - 1}}(t - m - 1) \cdot w_{l_{m} l_{m - 1}} = 1 \\
\implies & w_{l_{m} l_{m - 1}} = \frac{1}{y_{l_{m - 1}}(t - m - 1)} \cdot \frac{1}{2 \sigma\big(\net{l_{m}}{t - m}\big) - 1} \\
\implies & w_{l_{m} l_{m - 1}} = \frac{1}{y_{l_{m - 1}}(t - m - 1)} \cdot \coth\bigg(\frac{\net{l_{m}}{t - m}}{2}\bigg)
\end{align*}
$$

註：推論中使用了以下公式

$$
\begin{align*}
\tanh(x) & = 2 \sigma(2x) - 1 \\
\tanh(\frac{x}{2}) & = 2 \sigma(x) - 1 \\
\coth(\frac{x}{2}) & = \frac{1}{\tanh(\frac{x}{2})} = \frac{1}{2 \sigma(x) - 1}
\end{align*}
$$

但公式的前提不對，理由是 $w_{l_{m} l_{m - 1}}$ 根本不存在，應該改為 $w_{l_{m - 1} l_{m}}$（同 $\eqref{eq:22}$）。

接著我們推導時間點 $t - n$ 的節點 $\net{k_{n}}{t - n}$ 針對 $t$ 時間點造成的**總誤差**梯度**變化**：

$$
\sum_{k_{0}^* = 1}^{\dout} \pd{\dv{k_{n}}{t}{t - n}}{\dv{k_{0}^*}{t}{t}} \tag{28}\label{eq:28}
$$

由於**每個項次**都能遭遇**梯度消失**，因此**總和**也會遭遇**梯度消失**。

## 梯度常數

**梯度常數（Constant Error Flow）**的概念是控制**部份梯度**為**常數**。

- 透過 $\eqref{eq:31}$ 的想法讓梯度的**連乘積項**為 $1.0$
  - 不會像 $\eqref{eq:23}$ 導致梯度**爆炸**
  - 不會像 $\eqref{eq:24}$ 導致梯度**消失**
- 要達成 $\eqref{eq:31}$，就必須讓 $f_j$ 是**線性函數**

### 情境 1：模型輸出與內部節點 1-1 對應

假設模型輸出節點 $y_{j}(t - 1)$ 只與 $\net{j}{t}$ 相連，即

$$
\net{j}{t} = w_{j j} y_{j}(t - 1) \tag{29}\label{eq:29}
$$

（$\eqref{eq:29}$ 假設實際上不可能發生）則根據式子 $\eqref{eq:17}$ 我們可以推得

$$
\dv{j}{t}{t - 1} = w_{j j} \cdot \dfnet{j}{t - 1} \cdot \dv{j}{t}{t} \tag{30}\label{eq:30}
$$

為了強制讓梯度 $\dv{j}{t}{t}$ 不消失，作者認為需要強制達成

$$
w_{j j} \cdot \dfnet{j}{t - 1} = 1.0 \tag{31}\label{eq:31}
$$

如果 $\eqref{eq:31}$ 能夠達成，則積分 $\eqref{eq:31}$ 可以得到

$$
\begin{align*}
& \int w_{j j} \cdot \dfnet{j}{t - 1} \; d \big[\net{j}{t - 1}\big] = \int 1.0 \; d \big[\net{j}{t - 1}\big] \\
\implies & w_{j j} \cdot \fnet{j}{t - 1} = \net{j}{t - 1} \\
\implies & y_{j}(t - 1) = \fnet{j}{t - 1} = \frac{\net{j}{t - 1}}{w_{j j}}
\end{align*} \tag{32}\label{eq:32}
$$

觀察 $\eqref{eq:32}$ 我們可以發現

- 輸入 $\net{j}{t - 1}$ 與輸出 $\fnet{j}{t - 1}$ 之間的關係是乘上一個常數項 $w_{j j}$
- 代表函數 $f_{j}$ 其實是一個**線性函數**
- **每個時間點**的**輸出**居然**完全相同**，這個現象稱為 **Constant Error Carousel** (請見 $\eqref{eq:33}$)

$$
\begin{align*}
y_{j}(t) & = \fnet{j}{t} = f_{j}\big(w_{j j} y_{j}(t - 1)\big) \\
& = f_{j}\big(w_{j j} \frac{\net{j}{t - 1}}{w_{j j}}\big) = \fnet{j}{t - 1} = y_{j}(t - 1) \tag{33}\label{eq:33}
\end{align*}
$$

### 情境 2：增加外部輸入

將 $\eqref{eq:29}$ 的假設改成每個模型內部節點可以額外接收一個外部輸入

$$
\net{j}{t} = \sum_{i = 1}^{\din} w_{j i} x_{i}(t - 1) + w_{j j} y_{j}(t - 1) \tag{34}\label{eq:34}
$$

由於 $y_{j}(t - 1)$ 的設計功能是保留過去計算所擁有的資訊，在 $\eqref{eq:34}$ 的假設中唯一能夠**更新**資訊的方法只有透過 $x_{i}(t - 1)$ 配合 $w_{j i}$ 將新資訊合併進入 $\net{j}{t}$。

但作者認為，在計算的過程中，部份時間點的**輸入**資訊 $x_{i}(\cdot)$ 可以(甚至必須)被**忽略**，但這代表 $w_{j i}$ 需要**同時**達成**兩種**任務就必須要有**兩種不同的數值**：

- **加入新資訊**：代表 $\abs{w_{j i}} \neq 0$
- **忽略新資訊**：代表 $\abs{w_{j i}} \approx 0$

因此**無法只靠一個** $w_{j i}$ 決定**輸入**的影響，必須有**額外**能夠**理解當前內容 (context-sensitive)** 的功能模組幫忙**寫入** $x_{i}(\cdot)$

### 情境 3：輸出回饋到多個節點

將 $\eqref{eq:29} \eqref{eq:34}$ 的假設改回正常的模型架構

$$
\begin{align*}
\net{j}{t} & = \sum_{i = 1}^{\din} w_{j i} x_{i}(t - 1) + \sum_{i = \din + 1}^{\din + \dout} w_{j i} y_{i}(t - 1) \\
& = \sum_{i = 1}^{\din} w_{j i} x_{i}(t - 1) + \sum_{i = \din + 1}^{\din + \dout} w_{j i} \fnet{i}{t - 1}
\end{align*} \tag{35}\label{eq:35}
$$

由於 $y_{j}(t - 1)$ 的設計功能是保留過去計算所擁有的資訊，在 $\eqref{eq:35}$ 的假設中唯一能夠讓**過去**資訊**影響未來**計算結果的方法只有透過 $y_{i}(t - 1)$ 配合 $w_{j i}$ 將新資訊合併進入 $\net{j}{t}$。

但作者認為，在計算的過程中，部份時間點的**輸出**資訊 $y_i(*)$ 可以(甚至必須)被**忽略**，但這代表 $w_{j i}$ 需要**同時**達成**兩種**任務就必須要有**兩種不同的數值**：

- **保留過去資訊**：代表 $\abs{w_{j i}} \neq 0$
- **忽略過去資訊**：代表 $\abs{w_{j i}} \approx 0$

因此**無法只靠一個** $w_{j i}$ 決定**輸出**的影響，必須有**額外**能夠**理解當前內容 (context-sensitive)** 的功能模組幫忙**讀取** $y_i(*)$

值得一提的是，上述的假設是基於以下的事實觀察：
已知 RNN 能夠學習解決多個記憶時間較短 (short-time-lag) 的任務，但如果要能夠同時解決記憶時間較長 (long-time-lag) 的任務，則模型應該依照以下順序執行：

1. 記住短期資訊 $t_{0} \sim t_{1}$ (需要寫入功能)
2. 解決需要短期資訊 $t_{0} \sim t_{1}$ 的任務 (需要讀取功能)
3. 忘記短期資訊 $t_{0} \sim t_{1}$ (需要忽略功能)
4. 記住短期資訊 $t_{1} \sim t_{2}$ (需要寫入功能)
5. 解決需要短期資訊 $t_{1} \sim t_{2}$ 的任務 (需要讀取功能)
6. 忘記短期資訊 $t_{1} \sim t_{2}$ (需要忽略功能)
7. 為了解決與短期資訊 $t_{0} \sim t_{1}$ 相關的任務，突然又需要回憶起短期資訊 $t_{0} \sim t_{1}$ (需要寫入 + 讀取功能)

## LSTM 架構

<a name="paper-fig-1"></a>
![paper-fig:1](https://i.imgur.com/uhS4AgH.png)

<a name="paper-fig-2"></a>
![paper-fig:2](https://i.imgur.com/UQ5LAu8.png)

為了解決**梯度爆炸 / 消失**問題，作者決定以 Constant Error Carousel 為出發點（見 $\eqref{eq:33}$），提出 **3** 個主要的機制，並將這些機制的合體稱為**記憶單元（Memory Cell）**（見[圖 1](#paper-fig-1)）：

- **乘法輸入閘門（Multiplicative Input Gate）**
  - 用於決定是否**更新**記憶單元的**內部狀態** $\cell{k}(t + 1)$
  - 細節請見 $\eqref{eq:36} \eqref{eq:38}$
- **乘法輸出閘門（Multiplicative Output Gate）**
  - 用於決定是否**輸出**記憶單元的**輸出訊號** $\hcell{i}{k}{t + 1}$
  - 細節請見 $\eqref{eq:40} \eqref{eq:41}$
- **自連接線性單元（Central Linear Unit with Fixed Self-connection）**
  - 概念來自於 $\eqref{eq:33}$，希望能夠讓 $\cell{k}(t)$ 與 $\cell{k}(t + 1)$ 相同，藉此保障**梯度不會消失**
  - 如果 $\cell{k}(t)$ 與 $\cell{k}(t + 1)$ 相同，則我們可以確保達成 $\eqref{eq:31}$
  - 細節請見 $\eqref{eq:39}$

### 初始狀態

我們將 $\eqref{eq:1}$ 中的計算重新定義，並新增幾個符號：

|符號|意義|數值範圍|
|-|-|-|
|$\dhid$|**隱藏單元**的維度|$\N_{\geq 0}$|
|$\dcell$|**記憶單元**的**維度**|$\N_{\geq 1}$|
|$\ncell$|**記憶單元**的**個數**|$\N_{\geq 1}$|

- 因為論文 4.3 節有提到可以完全沒有**隱藏單元**，因此隱藏單元的維度可以為 $0$。
- 根據論文 4.4 節，可以**同時**擁有 $\ncell$ 個不同的**記憶單元**，因此 $\ncell$ 可以大於 $1$

接著我們定義 $t$ 時間點的模型計算狀態：

|符號|意義|數值範圍|
|-|-|-|
|$y^{\ophid}(t)$|**隱藏單元（Hidden Units）**|$\R^{\dhid}$|
|$y^{\opig}(t)$|**輸入閘門單元（Input Gate Units）**|$\R^{\dcell}$|
|$y^{\opog}(t)$|**輸出閘門單元（Output Gate Units）**|$\R^{\dcell}$|
|$y^{\cell{k}}(t)$|**記憶單元** $k$ 的**輸出**|$\R^{\dcell}$|
|$\cell{k}(t)$|**記憶單元** $k$ 的**內部狀態**|$\R^{\dcell}$|
|$y(t)$|**模型總輸出**|$\R^{\dout}$|

- 以上所有向量全部都**初始化**成各自維度的**零向量**，也就是 $t = 0$ 時模型**所有節點**（除了**輸入**）都是 $0$
- 根據論文 4.4 節，可以**同時**擁有 $\ncell$ 個不同的**記憶單元**
  - [圖 2](#paper-fig-2)模型共有 $2$ 個不同的記憶單元
  - **記憶單元**上標 $k$ 的數值範圍為 $k = 1, \dots, \ncell$
  - **所有**記憶單元**共享閘門單元**
- 根據論文 4.3 節，**記憶單元**、**閘門單元**與**隱藏單元**都算是**隱藏層（Hidden Layer）**的一部份

### 輸入閘門單元

當我們得到 $t$ 時間點的外部輸入 $x(t)$ 時，我們使用如同 $\eqref{eq:1} \eqref{eq:2}$ 的方式計算模型 $t + 1$ 時間點的**輸入閘門單元（Input Gate Units）** $y_i^{\opig}(t + 1)$

$$
\begin{align*}
\netig{i}{t + 1} & = \bigg[\sum_{j = 1}^{\din} \wig_{i j} \cdot x_j(t)\bigg] + \bigg[\sum_{j = \din + 1}^{\din + \dout} \wig_{i j} \cdot y_j(t)\bigg] + \bigg[\sum_{j = \din + \dout + 1}^{\din + \dout + \dhid} \wig_{i j} \cdot y_j^{\ophid}(t)\bigg] \\
& \quad + \bigg[\sum_{j = \din + \dout + \dhid + 1}^{\din + \dout + \dhid + \dcell} \wig_{i j} \cdot y_j^{\opig}(t)\bigg] + \bigg[\sum_{j = \din + \dout + \dhid + \dcell + 1}^{\din + \dout + \dhid + 2\dcell} \wig_{i j} \cdot y_j^{\opog}(t)\bigg] \\
& \quad + \bigg[\sum_{k = 1}^{\ncell} \sum_{j = \din + \dout + \dhid + (1 + k) \cdot \dcell + 1}^{\din + \dout + \dhid + (2 + k) \cdot \dcell} \wig_{i j} \cdot y_j^{\cell{k}}(t)\bigg] \\
& = \sum_{j = 1}^{\din + \dout + \dhid + (2 + \ncell) \cdot \dcell} \wig_{i j} \cdot [x ; y ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t) \\
y_i^{\opig}(t + 1) & = \fnetig{i}{t + 1}
\end{align*} \tag{36}\label{eq:36}
$$

- **所有** $t$ 時間點的**模型節點**都參與了**輸入閘門單元**的計算
- 因為有 $\ncell$ 個**不同**的**記憶單元內部狀態**，所以 $\eqref{eq:36}$ 中加法的最後一個項次必須有兩個 $\sum$
- $\wig$ 為**連接輸入閘門單元**的**參數**
  - 我們可以將所有模型節點**串接**，**一次做完矩陣乘法**（如同$\eqref{eq:36}$ 中的第二個等式）
  - $\wig$ 的輸入維度為 $\din + \dout + \dhid + (2 + \ncell) \cdot \dcell$
  - $\wig$ 的輸出維度為 $\dcell$，因此 $i$ 的數值範圍為 $i = 1, \dots, \dcell$
  - $\wig$ 的輸出維度設計成 $\dcell$ 的理由是所有**記憶單元內部狀態** $\cell{k}(t + 1)$ 會**共享**輸入閘門單元，因此與**記憶單元內部狀態**的**維度相同**，細節請見 $\eqref{eq:38}$
- $f_i^{\opig} : \R \to [0, 1]$ 必須要是**可微分函數**，具有**數值範圍限制**
- 之後我們會將 $y_i^{\opig}(t + 1)$ 用來決定是否**更新** $t + 1$ 時間點的**記憶單元內部狀態** $\cell{k}(t + 1)$，請見 $\eqref{eq:38} \eqref{eq:39}$

### 乘法輸入閘門

首先我們使用與 $\eqref{eq:36}$ 相同想法，在得到 $t$ 時間點的外部輸入 $x(t)$ 時計算模型 $t + 1$ 時間點第 $k$ 個**記憶單元淨輸入** $\netcell{i}{k}{t + 1}$

$$
\begin{align*}
\netcell{i}{k}{t + 1} & = \bigg[\sum_{j = 1}^{\din} \wcell{k}_{i j} \cdot x_j(t)\bigg] + \bigg[\sum_{j = \din + 1}^{\din + \dout} \wcell{k}_{i j} \cdot y_j(t)\bigg] \\
& \quad + \bigg[\sum_{j = \din + \dout + 1}^{\din + \dout + \dhid} \wcell{k}_{i j} \cdot y_j^{\ophid}(t)\bigg] + \bigg[\sum_{j = \din + \dout + \dhid + 1}^{\din + \dout + \dhid + \dcell} \wcell{k}_{i j} \cdot y_j^{\opig}(t)\bigg] \\
& \quad + \bigg[\sum_{j = \din + \dout + \dhid + \dcell + 1}^{\din + \dout + \dhid + 2\dcell} \wcell{k}_{i j} \cdot y_j^{\opog}(t)\bigg] \\
& \quad + \bigg[\sum_{k' = 1}^{\ncell} \sum_{j = \din + \dout + \dhid + (1 + k') \cdot \dcell + 1}^{\din + \dout + \dhid + (2 + k') \cdot \dcell} \wcell{k}_{i j} \cdot y_j^{\cell{k'}}(t)\bigg] \\
& = \sum_{j = 1}^{\din + \dout + \dhid + (2 + \ncell) \cdot \dcell} \wcell{k}_{i j} \cdot [x ; y ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)
\end{align*} \tag{37}\label{eq:37}
$$

- 運算架構與 $\eqref{eq:36}$ **完全相同**
  - **所有** $t$ 時間點的**模型節點**都參與了**記憶單元淨輸入**的計算
  - $\ncell$ 個**不同**的**記憶單元內部狀態**導致 $\eqref{eq:37}$ 中加法的最後一個項次必須有兩個 $\sum$
- 共有 $\ncell$ 個**不同**的**參數** $\wcell{k}$
  - 我們可以將所有模型節點**串接**，**一次做完矩陣乘法**（如同$\eqref{eq:37}$ 中的第二個等式）
  - $\wcell{k}$ 的輸入維度為 $\din + \dout + \dhid + (2 + \ncell) \cdot \dcell$
  - $\wcell{k}$ 的輸出維度事先定義的 $\dcell$，因此 $i$ 的數值範圍為 $i = 1, \dots, \dcell$
  - 計算總共得出 $\ncell \cdot \dcell$ 個數字

定義**可微分**啟發函數 $g_i^{\cell{k}} : \R \to \R$，我們將 $\eqref{eq:37}$ 轉換成第 $k$ 個**記憶單元內部狀態** $\cell{k}$ 在 $t + 1$ 時間點可以收到的**輸入訊號** $\gnetcell{i}{k}{t + 1}$。

接著我們將 $\eqref{eq:36}$ 所得**輸入閘門單元** $y_i^{\opig}(t + 1)$ 與 $\gnetcell{i}{k}{t + 1}$ 進行**相乘**

$$
y_i^{\opig}(t + 1) \cdot \gnetcell{i}{k}{t + 1} \tag{38}\label{eq:38}
$$

- $y_i^{\opig}(t + 1)$ 扮演**輸入閘門**的角色
  - 由於**記憶單元內部狀態**的**輸入訊號**與 $\eqref{eq:36}$ 是以**相乘**進行結合，因此被稱為**乘法輸入閘門（Multiplicative Input Gate）**
  - 當模型認為**輸入訊號** $\gnetcell{i}{k}{t + 1}$ **不重要**時，模型應該要**關閉輸入閘門**，即 $y_i^{\opig}(t + 1) \approx 0$
    - 丟棄**當前**輸入訊號
    - 只以**過去資訊**進行決策
  - 當模型認為**輸入訊號** $\gnetcell{i}{k}{t + 1}$ **重要**時，模型應該要**開啟輸入閘門**，即 $y_i^{\opig}(t + 1) \approx 1$
    - 同時考慮**過去**與**當前**資訊
    - 但以**當前**資訊為主
- 不論**輸入訊號** $\gnetcell{i}{k}{t + 1}$ 的大小，只要 $y_i^{\opig}(t + 1) \approx 0$，則輸入訊號**完全無法影響**接下來的所有計算
  - 以此設計避免 $\eqref{eq:34}$ 所遇到的困境
  - 由 $\wcell{k}$ 決定**寫入**的**數值**，函數 $g$ 可以**沒有數值範圍限制**
  - 由 $\wig$ 根據**當前模型計算狀態**控制**寫入**（Context-Sensitive）
- 所有的 $\cell{k}$ 都**共享**相同的乘法輸入閘門 $y_i^{\opig}(t + 1)$
  - 有時候只需要寫入**部份**維度，不需要**同時寫入** $\dcell$ 個數值
  - 一旦需要寫入維度 $i$，則**必須同時**寫入 $\ncell$ 個數值
- 此設計有點**瑕疵**，邏輯應該修正成像是**圖靈機（Turing Machine）**的概念
  - 有時候只需要寫入**部份記憶單元**中的**部份維度**，不需要**同時寫入** $\ncell \cdot \dcell$ 個數值
  - 應該要有 $\ncell$ 個不同的 $\dcell$ 維度乘法輸入閘門

### 自連接線性單元

接著我們將 $\eqref{eq:38}$ 的計算結果用來計算 $t + 1$ 時間點的**記憶單元內部狀態** $\cell{k}(t + 1)$

$$
\cell{k}(t + 1) = \cell{k}(t) + y_i^{\opig}(t + 1) \cdot \gnetcell{i}{k}{t + 1} \tag{39}\label{eq:39}
$$

- 根據 $\eqref{eq:38}$ 我們知道 $y_i^{\opig}(t + 1)$ 能夠**控制輸入訊號的開關**
  - 當輸入訊號**完全關閉**時，$t + 1$ 時間點的**記憶單元內部狀態**與 $t$ 時間點**完全相同**，即 $\cell{k}(t + 1) = \cell{k}(t)$，達成 $\eqref{eq:31}$
  - 當輸入訊號開啟時，$t + 1$ 時間點的**記憶單元內部狀態**會被**更新**
- 由於 $t + 1$ 時間點的資訊有加上 $t$ 時間點的資訊，因此稱為**自連接線性單元（Central Linear Unit with Fixed Self-connection）**
  - **加法**是**線性**運算
  - **加上自己**是**自連接**
  - 概念與 $\eqref{eq:33}$ 相同，藉此保障**梯度不會消失**

### 輸出閘門單元

想法同 $\eqref{eq:36}$，當我們得到 $t$ 時間點的外部輸入 $x(t)$ 時便可以計算模型 $t + 1$ 時間點的**輸出閘門單元（Output Gate Units）** $y_i^{\opog}(t + 1)$

$$
\begin{align*}
\netog{i}{t + 1} & = \bigg[\sum_{j = 1}^{\din} \wog_{i j} \cdot x_j(t)\bigg] + \bigg[\sum_{j = \din + 1}^{\din + \dout} \wog_{i j} \cdot y_j(t)\bigg] + \bigg[\sum_{j = \din + \dout + 1}^{\din + \dout + \dhid} \wog_{i j} \cdot y_j^{\ophid}(t)\bigg] \\
& \quad + \bigg[\sum_{j = \din + \dout + \dhid + 1}^{\din + \dout + \dhid + \dcell} \wog_{i j} \cdot y_j^{\opig}(t)\bigg] + \bigg[\sum_{j = \din + \dout + \dhid + \dcell + 1}^{\din + \dout + \dhid + 2\dcell} \wog_{i j} \cdot y_j^{\opog}(t)\bigg] \\
& \quad + \bigg[\sum_{k = 1}^{\ncell} \sum_{j = \din + \dout + \dhid + (1 + k) \cdot \dcell + 1}^{\din + \dout + \dhid + (2 + k) \cdot \dcell} \wog_{i j} \cdot y_j^{\cell{k}}(t)\bigg] \\
& = \sum_{j = 1}^{\din + \dout + \dhid + (2 + \ncell) \cdot \dcell} \wog_{i j} \cdot [x ; y ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t) \\
y_i^{\opog}(t + 1) & = \fnetog{i}{t + 1} \tag{40}\label{eq:40}
\end{align*}
$$

- 運算架構與 $\eqref{eq:36}$ **完全相同**
  - **所有** $t$ 時間點的**模型節點**都參與了**輸出閘門單元**的計算
  - $\ncell$ 個**不同**的**記憶單元內部狀態**導致 $\eqref{eq:40}$ 中加法的最後一個項次必須有兩個 $\sum$
- $\wog$ 為**連接輸出閘門單元**的**參數**
  - 我們可以將所有模型節點**串接**，**一次做完矩陣乘法**（如同$\eqref{eq:40}$ 中的第二個等式）
  - $\wog$ 的輸入維度為 $\din + \dout + \dhid + (2 + \ncell) \cdot \dcell$
  - $\wog$ 的輸出維度為 $\dcell$，因此 $i$ 的數值範圍為 $i = 1, \dots, \dcell$
  - $\wog$ 的輸出維度設計成 $\dcell$ 的理由是所有**記憶單元內部狀態** $\cell{k}(t + 1)$ 會**共享**輸出閘門單元，因此與**記憶單元內部狀態**的**維度相同**，細節請見 $\eqref{eq:41}$
- $f_i^{\opog} : \R \to [0, 1]$ 必須要是**可微分函數**，具有**數值範圍限制**
- 之後我們會將 $y_i^{\opog}(t + 1)$ 用來決定是否**輸出** $t + 1$ 時間點的**記憶單元啟發值** $\hcell{i}{k}{t + 1}$

### 乘法輸出閘門

定義**可微分**啟發函數 $h_i^{\cell{k}} : \R \to \R$，我們將 $\eqref{eq:40}$ 轉換成第 $k$ 個**記憶單元內部狀態** $\cell{k}$ 在 $t + 1$ 時間點的**輸出訊號** $\hcell{i}{k}{t + 1}$。
注意不是 $\netcell{i}{k}{t + 1}$ 而是使用 $\cell{k}_i(t + 1)$。
接著我們將 $\eqref{eq:40}$ 所得**輸出閘門單元** $y_i^{\opog}(t + 1)$ 與 $\hcell{i}{k}{t + 1}$ 進行**相乘**得到記憶單元 $k$ 的**輸出訊號** $y_i^{\cell{k}}(t + 1)$

$$
y_i^{\cell{k}}(t + 1) = y_i^{\opog}(t + 1) \cdot \hcell{i}{k}{t + 1} \tag{41}\label{eq:41}
$$

- $y_i^{\opog}(t + 1)$ 扮演**輸出閘門**的角色
  - 由於**記憶單元內部狀態**的**輸出訊號**與 $\eqref{eq:40}$ 是以**相乘**進行結合，因此被稱為**乘法輸出閘門（Multiplicative Output Gate）**
  - 當模型認為**輸出訊號** $\hcell{i}{k}{t + 1}$ 會導致**當前計算錯誤**時，模型應該**關閉輸出閘門**，即 $y_i^{\opog}(t + 1) \approx 0$
    - 在**輸入**閘門**開啟**的狀況下，**關閉輸出**閘門代表不讓**現在**時間點的資訊影響當前計算
    - 在**輸入**閘門**關閉**的狀況下，**關閉輸出**閘門代表不讓**過去**時間點的資訊影響當前計算
  - 當模型認為**輸出訊號** $\hcell{i}{k}{t + 1}$ **包含重要資訊**時，模型應該要開啟**輸出閘門**，即 $y_i^{\opog}(t + 1) \approx 1$
    - 在**輸入**閘門**開啟**的狀況下，**開啟輸出**閘門代表讓**現在**時間點的資訊影響當前計算
    - 在**輸入**閘門**關閉**的狀況下，**開啟輸出**閘門代表不讓**過去**時間點的資訊影響當前計算
- 不論**輸出訊號** $\hcell{i}{k}{t + 1}$ 的大小，只要 $y_i^{\opog}(t + 1) \approx 0$，則輸出訊號**完全無法影響**接下來的所有計算
  - 以此設計避免 $\eqref{eq:34} \eqref{eq:35}$ 所遇到的困境
  - 由 $\cell{k}(t + 1)$ 決定**讀取**的**數值**，函數 $h$ 可以**沒有數值範圍限制**
  - 由 $\wog$ 根據**當前模型計算狀態**控制**輸出**（Context-sensitive）
- 所有的 $\cell{k}$ 都**共享**相同的乘法輸出閘門 $y_i^{\opog}(t + 1)$
  - 有時候只需要讀取**部份**維度，不需要**同時讀取** $\dcell$ 個數值
  - 一旦需要讀取維度 $i$，則**必須同時**讀取 $\ncell$ 個數值
- 此設計有點**瑕疵**，邏輯應該修正成像是**圖靈機（Turing Machine）**的概念
  - 有時候只需要讀取**部份記憶單元**中的**部份維度**，不需要**同時讀取** $\ncell \cdot \dcell$ 個數值
  - 應該要有 $\ncell$ 個不同的 $\dcell$ 維度乘法輸出閘門

### 總輸出

經過 $\eqref{eq:36} \eqref{eq:37} \eqref{eq:39} \eqref{eq:40} \eqref{eq:41}$ 後我們可以計算 $t + 1$ 時間點的**總輸出** $y_i(t + 1)$。

但是！！！

$t + 1$ 時間點的**總輸出**只與 $t$ 時間點的**模型狀態**（**不含閘門**）有關係，所以 $\eqref{eq:36} \eqref{eq:37} \eqref{eq:39} \eqref{eq:40} \eqref{eq:41}$ 的所有計算都只是在幫助 $t + 2$ 時間點的計算狀態**鋪陳**。

$$
\begin{align*}
\netout{i}{t + 1} & = \bigg[\sum_{j = 1}^{\din} \wout_{i j} \cdot x_j(t)\bigg] + \bigg[\sum_{j = \din + 1}^{\din + \dout} \wout_{i j} \cdot y_j(t)\bigg] \\
& \quad + \bigg[\sum_{j = \din + \dout + 1}^{\din + \dout + \dhid} \wout_{i j} \cdot y_j^{\ophid}(t)\bigg] + \bigg[\sum_{k = 1}^{\ncell} \sum_{j = \din + \dout + \dhid + (k - 1) \cdot \dcell + 1}^{\din + \dout + \dhid + k \dcell} \wout_{i j} \cdot y_j^{\cell{k}}(t)\bigg] \\
& = \sum_{j = 1}^{\din + \dout + \dhid + \ncell \cdot \dcell} \wout_{i j} \cdot [x ; y ; y^{\ophid} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t) \\
y_i(t + 1) & = \fnetout{i}{t + 1}
\end{align*} \tag{42}\label{eq:42}
$$

- $\wout$ 為**連接總輸出**的**參數**
  - 我們可以將所有模型節點**串接**，**一次做完矩陣乘法**（如同$\eqref{eq:42}$ 中的第二個等式）
  - $\wout$ 的輸入維度為 $\din + \dout + \dhid + \ncell \cdot \dcell$（注意不包含**閘門**）
  - $\wout$ 的輸出維度為 $\dout$，因此 $i$ 的數值範圍為 $i = 1, \dots, \dout$
- $f_i^{\opout} : \R \to \R$ 必須要是**可微分函數**，可以**沒有數值範圍限制**
- 注意 $y_i(t + 1)$ 與 $y_i^{\opog}$ 不同
  - $y_i(t + 1)$ 是**總輸出**，我的 $y_i(t + 1)$ 是論文中的 $y^k(t + 1)$
  - $y_i^{\opog}(t + 1)$ 是**記憶單元**的**輸出閘門**，我的 $y_i^{\opog}(t + 1)$ 是論文中的 $y^{\opout_i}(t + 1)$
- 接著就可以拿 $y_i(t + 1)$ 去做 $\eqref{eq:3} \eqref{eq:4}$ 的誤差計算，取得梯度並進行模型最佳化

### 隱藏單元

論文 4.3 節有提到可以完全沒有**隱藏單元**，因此這個段落可以完全不存在。
但如果**允許隱藏單元**出現，這就跟論文的出發點有點**矛盾**。
我會說矛盾是因為作者提出新架構的同時又保留原始架構，而且讓新舊架構**平行執行**。
可是從數學上來看**平行執行舊架構**應該會遭遇**梯度爆炸**的問題，就如 $\eqref{eq:23} \eqref{eq:24}$。

想法與 $\eqref{eq:36} \eqref{eq:37} \eqref{eq:40}$ 完全相同，以 $t$ 時間點的外部輸入 $x(t)$ 計算模型 $t + 1$ 時間點的**隱藏單元** $y_i^{\ophid}(t + 1)$

$$
\begin{align*}
\nethid{i}{t + 1} & = \bigg[\sum_{j = 1}^{\din} \whid_{i j} \cdot x_j(t)\bigg] + \bigg[\sum_{j = \din + 1}^{\din + \dout} \whid_{i j} \cdot y_j(t)\bigg] + \bigg[\sum_{j = \din + \dout + 1}^{\din + \dout + \dhid} \whid_{i j} \cdot y_j^{\ophid}(t)\bigg] \\
& \quad + \bigg[\sum_{j = \din + \dout + \dhid + 1}^{\din + \dout + \dhid + \dcell} \whid_{i j} \cdot y_j^{\opig}(t)\bigg] + \bigg[\sum_{j = \din + \dout + \dhid + \dcell + 1}^{\din + \dout + \dhid + 2\dcell} \whid_{i j} \cdot y_j^{\opog}(t)\bigg] \\
& \quad + \bigg[\sum_{k = 1}^{\ncell} \sum_{j = \din + \dout + \dhid + (k + 1) \cdot \dcell + 1}^{\din + \dout + \dhid + (k + 2) \cdot \dcell} \whid_{i j} \cdot y_j^{\cell{k}}(t)\bigg] \\
& = \sum_{j = 1}^{\din + \dout + \dhid + (2 + \ncell) \cdot \dcell} \whid_{i j} \cdot [x ; y ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t) \\
y_i^{\ophid}(t + 1) & = \fnethid{i}{t + 1} \tag{43}\label{eq:43}
\end{align*}
$$

- **所有** $t$ 時間點的**模型節點**都參與了**隱藏單元**的計算
- 因為有 $\ncell$ 個**不同**的**記憶單元內部狀態**，所以 $\eqref{eq:43}$ 中加法的最後一個項次必須有兩個 $\sum$
- $\whid$ 為**連接隱藏單元**的**參數**
  - 我們可以將所有模型節點**串接**，**一次做完矩陣乘法**（如同$\eqref{eq:43}$ 中的第二個等式）
  - $\whid$ 的輸入維度為 $\din + \dout + \dhid + (2 + \ncell) \cdot \dcell$
  - $\whid$ 得輸出維度為 $\dhid$，因此 $i$ 的數值範圍為 $i = 1, \dots, \dhid$
- $f_i^{\ophid} : \R \to \R$ 必須要是**可微分函數**，可以**沒有數值範圍限制**
- 之後我們會將 $y_i^{\opig}(t + 1)$ 用於計算**所有** $t + 2$ 時間點的**模型計算狀態**

## 以近似梯度最佳化 LSTM

過去的論文中提出以**修改最佳化過程**避免 RNN 訓練遇到**梯度爆炸 / 消失**的方法有

- Truncated BPTT
- **RTRL**（**R**eal **T**ime **R**ecurrent **L**earning）

論文 4.5 節提到最佳化 LSTM 的方法為 RTRL 的變種，但 A.1.2 節說論文採用 Truncated BPTT，所以我也不知道他使用的是哪個。

最佳化的核心思想是確保能夠達成 **CEC** （見 $\eqref{eq:33}$），而使用的手段是要求在**記憶單元**中計算所的的梯度，一旦經過**輸入閘門**流出**記憶單元**，便**不可以**再透過**輸出閘門**進入**記憶單元**。

### 丟棄部份模型單元的梯度

首先我們定義新的符號 $\aptr$，代表計算**梯度**的過程會有**部份梯度**故意被**丟棄**（設定為 $0$），並以丟棄結果**近似**最後**全微分**的概念。

$$
\pd{[\opnet^{\opig} ; \opnet^{\opog} ; \opnet^{\cell{1}} ; \dots ; \opnet^{\cell{\ncell}}]_i(t + 1)}{[x ; y ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)} \aptr 0 \tag{44}\label{eq:44}
$$

- 所有與**輸入閘門** $\netig{i}{t + 1}$、**輸出閘門** $\netog{i}{t + 1}$、**記憶單元** $\netcell{i}{k}{t + 1}$ **直接相連**的 $t$ 時間點的**單元**，一律**丟棄梯度**
- **丟棄梯度**的意思是，即使計算結果的梯度不為 $0$，仍然將梯度**手動設成** $0$
- 直接相連的**單元**包含**外部輸入** $x(t)$、**前次輸出** $y(t)$、**隱藏單元** $y^{\ophid}(t)$、**輸入閘門** $y^{\opig}(t)$、**輸出閘門** $y^{\opog}(t)$ 與**記憶單元** $y^{\cell{k}}$（見 $\eqref{eq:36}, \eqref{eq:37}, \eqref{eq:40}$）

根據 $\eqref{eq:44}$ 結合 $\eqref{eq:36}, \eqref{eq:40}$，我們可以進一步推得

$$
\begin{align*}
& \pd{[y^{\opig} ; y^{\opog}]_i(t + 1)}{[x ; y ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)} \\
& = \pd{[y^{\opig} ; y^{\opog}]_i(t + 1)}{[\opnet^{\opig} ; \opnet^{\opog}]_i(t + 1)} \cdot \cancelto{0}{\pd{[\opnet^{\opig} ; \opnet^{\opog}]_i(t + 1)}{[x ; y ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)}} \\
& \aptr 0
\end{align*} \tag{45}\label{eq:45}
$$

接著以 $\eqref{eq:40} \eqref{eq:45}$ 加上 $\eqref{eq:37} \eqref{eq:39} \eqref{eq:40} \eqref{eq:41}$ 我們可以得到

$$
\begin{align*}
& \pd{y_i^{\cell{k}}(t + 1)}{[x ; y ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)} \\
& = \pd{y_i^{\cell{k}}(t + 1)}{y_i^{\opig}(t + 1)} \cdot \cancelto{0}{\pd{y_i^{\opig}(t + 1)}{[x ; y ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)}} \\
& \quad + \pd{y_i^{\cell{k}}(t + 1)}{\netcell{i}{k}{t + 1}} \cdot \cancelto{0}{\pd{\netcell{i}{k}{t + 1}}{[x ; y ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)}} \\
& \quad + \pd{y_i^{\cell{k}}(t + 1)}{y_i^{\opog}(t + 1)} \cdot \cancelto{0}{\pd{y_i^{\opog}(t + 1)}{[x ; y ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)}} \\
& \aptr 0
\end{align*} \tag{46}\label{eq:46}
$$

由於**參數** $w^{\ophid}, w^{\opout}$ **沒有直接**與 $t + 1$ 時間點的 $y^{\opig}(t + 1), y^{\opog}(t + 1), \opnet^{\cell{k}}(t + 1)$ **相連**，因此 $w^{\ophid}, w^{\opout}$ 只能透過參與 $t$ 時間點**以前**的計算**間接**對 $t + 1$ 時間點計算造成影響（見 $\eqref{eq:42} \eqref{eq:43}$），這也代表在 $\eqref{eq:45} \eqref{eq:46}$ 作用的情況下 $w^{\ophid}, w^{\opout}$ **無法**從 $y^{\opig}(t + 1), y^{\opog}(t + 1), \opnet^{\cell{k}}(t + 1)$ 收到任何的**梯度**：

$$
\begin{align*}
& \pd{[y^{\opig} ; y^{\opog} ; y^{\cell{k}}]_i(t + 1)}{\whid_{p q}} \\
& = \sum_{j = 1}^{\din + \dout + \dhid + (2 + \ncell) \cdot \dcell} \Bigg[\cancelto{0}{\pd{[y^{\opig} ; y^{\opog} ; y^{\cell{k}}]_i(t + 1)}{[x ; y ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)}} \cdot \\
& \quad \sum_{t^{\star} = 1}^t \bigg[\pd{[x ; y ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)}{\nethid{p}{t^{\star}}} \cdot \pd{\nethid{p}{t^{\star}}}{\whid_{p q}}\bigg]\Bigg] \\
& \aptr 0 \\
& \pd{[y^{\opig} ; y^{\opog} ; y^{\cell{k}}]_i(t + 1)}{\wout_{p q}} \\
& = \sum_{j = 1}^{\din + \dout + \dhid + (2 + \ncell) \cdot \dcell} \Bigg[\cancelto{0}{\pd{[y^{\opig} ; y^{\opog} ; y^{\cell{k}}]_i(t + 1)}{[x ; y ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)}} \cdot \\
& \quad \sum_{t^{\star} = 1}^t \bigg[\pd{[x ; y ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)}{\netout{p}{t^{\star}}} \cdot \pd{\netout{p}{t^{\star}}}{\wout_{p q}}\bigg]\Bigg] \\
& \aptr 0
\end{align*} \tag{47}\label{eq:47}
$$

注意 $t = 0$ 時模型的**計算狀態**與 $\wout$ **無關**，因此 $t^{\star}$ 從 $1$ 開始。
不過從 $t^{\star}$ 從 $0$ 開始也沒差，反正前面的乘法項直接讓整個梯度 $\aptr 0$。

### 剩餘梯度

令 $\delta_{a b}$ 為 **Kronecker delta**，i.e.，

$$
\delta_{a b} = \begin{dcases}
1 & \text{if } a = b \\
0 & \text{otherwise}
\end{dcases} \tag{48}\label{eq:48}
$$

在 $\eqref{eq:44} \eqref{eq:45} \eqref{eq:46} \eqref{eq:47}$ 的作用下，我們可以求得 $\wout$ 的**丟棄**部份梯度後對於**總輸出**計算所得的**剩餘梯度**

$$
\begin{align*}
& \pd{y_i(t + 1)}{\wout_{p q}} \\
& = \pd{y_i(t + 1)}{\netout{i}{t + 1}} \cdot \pd{\netout{i}{t + 1}}{\wout{p q}} \\
& = \dfnetout{i}{t + 1} \cdot \Bigg(\delta_{i p} \cdot \pd{\netout{i}{t + 1}}{\wout_{i q}} + \\
& \quad \sum_{j = \din + 1}^{\din + \dout + \dhid + (2 + \ncell) \cdot \dcell} \bigg[\pd{\netout{i}{t + 1}}{[y ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)} \cdot \\
& \quad \pd{[y ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)}{\wout_{p q}}\bigg]\Bigg) \\
& = \dfnetout{i}{t + 1} \cdot \Bigg(\delta_{i p} \cdot [x ; y ; y^{\ophid} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_q(t) + \\
& \quad \sum_{j = \din + 1}^{\din + \dout + \dhid + (2 + \ncell) \cdot \dcell} \bigg[\wout_{i j} \cdot \pd{[y ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)}{\wout_{p q}}\bigg]\Bigg) \\
& = \dfnetout{i}{t + 1} \cdot \Bigg(\delta_{i p} \cdot [x ; y ; y^{\ophid} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_q(t) + \\
& \quad \sum_{j = \din + 1}^{\din + \dout + \dhid + (2 + \ncell) \cdot \dcell} \Bigg[\wout_{i j} \cdot \\
& \quad \sum_{t^{\star} = 1}^{t - 1} \bigg[\pd{[y ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)}{y_p(t^{\star})} \cdot \pd{y_p(t^{\star})}{\wout_{p q}}\bigg]\Bigg]\Bigg) \\
& \aptr \dfnetout{i}{t + 1} \cdot \Bigg(\delta_{i p} \cdot [x ; y ; y^{\ophid} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_q(t) + \\
& \quad \sum_{j = \din + 1}^{\din + \dout + \dhid + (2 + \ncell) \cdot \dcell} \Bigg[\wout_{i j} \cdot \sum_{t^{\star} = 1}^{t - 1} \bigg[\pd{[y ; y^{\ophid}]_j(t)}{y_p(t^{\star})} \cdot \pd{y_p(t^{\star})}{\wout_{p q}}\bigg]\Bigg]\Bigg)
\end{align*} \tag{49}\label{eq:49}
$$

- $\wout_{p q}$ 對於**總輸出** $y_i(t + 1)$ 的影響方法有兩種
  - 這也是為什麼 $\eqref{eq:49}$ 中的第二個等式有**兩個加法**項次
  - 當 $p = i$ 時，代表 $y_i(t + 1)$ 與 $\wout_{i q}$ **直接相連**
    - 在此情況下**梯度**為與 $y_i(t + 1)$ **相連**的 $x_q(t)$、$y_q(t)$、$y_q^{\ophid}(t)$、$y_q^{\cell{k}}(t)$，細節請見 $\eqref{eq:42}$
    - 注意**輸入閘門**與**輸出閘門沒有**直接與**總輸出**相連接，細節請見第三個等式與 $\eqref{eq:42}$
    - 但如果 $p \neq i$，則 $\wout_{p q}$ 無法直接影響 $y_i(t + 1)$，這也是為什麼需要**乘上** $\delta_{i p}$ 的原因
  - 當 $p \neq i$ 時，$\wout_{p q}$ 只能透過**間接**的方式（過去時間點的計算結果）影響 $y_i(t + 1)$
    - 根據論文 4.3 節，$t - 1$ 時間點的**總輸出**會影響 $t$ 時間點的**除了外部輸入**的**所有計算狀態**
    - 這也是為什麼在第二個等式中**除了外部輸入**的**所有計算狀態**都要參與**梯度**的計算
    - 細節請見 $\eqref{eq:36} \eqref{eq:37} \eqref{eq:40} \eqref{eq:41} \eqref{eq:42} \eqref{eq:43}$

<!--
在 $\eqref{eq:44} \eqref{eq:45} \eqref{eq:46} \eqref{eq:47}$ 的作用下，我們可以求得 $\wout$ 的**丟棄**部份梯度後所**剩餘可取得**的梯度

一但 $\eqref{eq:44}$ 中的梯度為 $0$，$t + 1$ 時間點**以後**與**輸入閘門**或**輸出閘門**有關的**所有**計算，其所得**梯度**便**無法**傳回 $t$ 時間點**以前**的**所有單元**與**所有參數**（見 $\eqref{eq:45}$）

一但 $\eqref{eq:44}$ 中的梯度為 $0$，$t + 1$ 時間點**以後**來自**記憶單元**的**所有梯度**便**無法**傳回 $t$ 時間點**以前**的**所有單元**與**所有參數**（見 $\eqref{eq:45} \eqref{eq:46}$）

### 更新模型參數

只針對來自記憶單元 $c_i(t + 1)$ 的梯度進行跟新

$$
\pd{c_i(t + 1)}{w_{j k}}
$$

- local in space: if update complexity per time stemp and weight does not depend on network size
- local in time: if its storage requirements do not depend on input sequeqnce length
- RTRL is local in time but not in space
- BPTT is local in space but not in time
- LSTM local in space and time
  - there is no need to store activation values observed during sequence processing in a stack with potentially unlimited size

### 訓練早期發生異常

在訓練的早期，LSTM 模型可能會學到維持輸出閘門開啟，使得

- sequential network construction
- output gate bias

### Internal state drift

輸入維持正或維持負

$h'(c_i(t))$ 得到較小的值，造成梯度消失

- 挑比較好的 $h$
- 但如果 $h(c_i(t)) = c_i(t)$，則輸出範為可能不受限制
- 解決方法為 at the beginning of learning is inititally to bias the input gate toward zero
  - 這個方法等同於改變 $y^{\opin}$ 的數值範圍
  - 雖然對計算有影響，但 internal state drift 影響更大，因此值得
- 根據實驗，採用 sigmoid 函數就不需要進行 bias 的調整

## 實驗

- 要測試較長的時間差
- 資料集不可以出現短時間差
- 任務要夠難
  - 不可以只靠 random weight guessing 解決
  - 需要比較多的參數或是高計算精度 (sparse in weight space)

### 實驗設計

- 使用 online learning
- 使用 sigmoid 作為啟發函數
- 參數初始化範圍為 $[-0.1, 0.1]$
  - 只有實驗 1 與 2 的初始化範圍為 $[-0.2, 0.2]$
- 資料的訓練順序為隨機
- 在每個時間點 $t$ 的計算順序為
  1. 將外部輸入 $x(t)$ 丟入模型
  2. 計算輸入閘門、輸出閘門、記憶單元
  3. 計算淨輸出

$h : \R \to [-1, 1]$ 函數的定義為

$$
h(x) = \frac{2}{1 + \exp(-x)} - 1 = 2 \sigma(x) - 1
$$

$g : \R \to [-2, 2]$ 函數的定義為

$$
g(x) = \frac{4}{1 + \exp(-x)} - 2 = 4 \sigma(x) - 2
$$

-->
