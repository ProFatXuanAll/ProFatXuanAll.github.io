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

<!-- Define LaTeX command which will be used through out the writing.
  Each command must be wrapped with $ signs.
  Commands must be separated by empty line.
 -->

<p style="display: none;">

  <!-- Real field. -->
  $\newcommand{\R}{\mathbf{R}}$
  <!-- Set. -->
  $\newcommand{\set}[1]{\left\lbrace #1 \right\rbrace}$
  <!-- Absolute value. -->
  $\newcommand{\abs}[1]{\left\lvert #1 \right\rvert}$
  <!-- Model weight. -->
  $\newcommand{\w}[2]{w_{#1 #2}}$
  <!-- Input x. -->
  $\newcommand{\x}[2]{x_{#1}(#2)}$
  <!-- Output y. -->
  $\newcommand{\y}[2]{y_{#1}(#2)}$
  <!-- Concate xy. -->
  $\newcommand{\xy}[2]{[x ; y]_{#1}(#2)}$
  <!-- Target y. -->
  $\newcommand{\hy}[2]{\hat{y}_{#1}(#2)}$
  <!-- Summation. -->
  $\newcommand{\S}[2]{\sum_{#1}^{#2}}$
  <!-- Product. -->
  $\newcommand{\P}[2]{\prod_{#1}^{#2}}$
  <!-- Net input. -->
  $\newcommand{\net}[2]{\operatorname{net}_{#1}(#2)}$
  <!-- Net input with activatiton. -->
  $\newcommand{\fnet}[2]{f_{#1}\big(\operatorname{net}_{#1}(#2)\big)}$
  <!-- Derivative of with respect to net input. -->
  $\newcommand{\dfnet}[2]{f_{#1}'\big(\operatorname{net}_{#1}(#2)\big)}$
  <!-- Input dimension. -->
  $\newcommand{\din}{d_{\operatorname{in}}}$
  <!-- Output dimension. -->
  $\newcommand{\dout}{d_{\operatorname{out}}}$
  <!-- Total loss. -->
  $\newcommand{\Loss}[1]{\operatorname{loss}(#1)}$
  <!-- Partial loss. -->
  $\newcommand{\loss}[2]{\operatorname{loss}_{#1}(#2)}$
  <!-- Partial derivative. -->
  $\newcommand{\pd}[2]{\frac{\partial #1}{\partial #2}}$
  <!-- Graident of loss(t_2) with respect to net k_0 at time t_1. -->
  $\newcommand{\dv}[3]{\vartheta_{#1}^{#2}[#3]}$
  <!-- Index k with time. -->
  $\newcommand{\k}[1]{k_{#1}}$
  <!-- Index l with time. -->
  $\newcommand{\l}[1]{l_{#1}}$
  <!-- Index t with time. -->
  $\newcommand{\t}[1]{t_{#1}}$
  <!-- f with index. -->
  $\newcommand{\f}[1]{f_{#1}}$

</p>

<!-- End LaTeX command define section. -->

## 重點

- 對 RNN 進行梯度反向傳播的演算法通常稱為 **BPTT** （**B**ack-**P**ropagation **T**hrought **T**ime） 或 real-time recurrent learning
  - RNN 透過 back-propagation 學習**效率差**
  - 梯度會**爆炸**或**消失**
    - 梯度爆炸造成神經網路的**權重劇烈振盪**
    - 梯度消失造成**訓練時間慢長**
  - 無法解決輸入與輸出訊號**間隔較長**（long time lag）的問題
- LSTM 能夠解決上述問題
  - 能夠處理 time lag 間隔為 $1000$ 的問題
  - 甚至輸入訊號含有雜訊時也能處理
  - 同時能夠保有處理 short time lag 問題的能力
- 使用 mulitplicative gate 學習開啟與關閉記憶 hidden state 的機制
  - Forward pass 演算法複雜度為 $O(1)$
  - Backward pass 演算法複雜度為 $O(w)$，$w$ 代表權重

## 傳統的 RNN

### BPTT

BPTT = **B**ack **P**ropagation **T**hrough **T**ime，是專門用來計算 RNN 神經網路模型的**梯度反向傳播演算法**。
一個 RNN 模型的**輸入**來源共有兩種：

- **外部輸入** $x(t)$
  - 輸入維度為 $\din$
  - 使用下標 $\x{j}{t}$ 代表不同的輸入訊號，$j = 1, \dots, \din$
- **前一次**的**輸出** $y(t)$
  - 輸出維度為 $\dout$
  - 使用下標 $\y{j}{t}$ 代表不同的輸入訊號，$j = \din + 1, \dots, \dout$
  - 注意這裡是使用 $t$ 不是 $t - 1$
- $t$ 的起始值為 $0$，結束值為 $T$，每次遞增 $1$
  - 時間為離散狀態
  - 方便起見令 $y(0) = 0$

令 RNN 模型的參數為 $w \in \R^{\dout \times (\din + \dout)}$，則我們可以定義第 $i$ 個外部節點

$$
\begin{align}
  \net{i}{t + 1} & = \S{j = 1}{\din} \w{i}{j} \x{j}{t} + \S{j = \din + 1}{\dout} \w{i}{j} \y{j}{t} \newline
  & = \S{j = 1}{\din + \dout} \w{i}{j} \xy{j}{t}
\end{align} \tag{1}\label{eq:1}
$$

- $\net{i}{t + 1}$ 代表第 $t + 1$ 時間的**模型內部節點** $i$ 所收到的**總輸入**（net input）
  - 注意 $t$ 時間點的輸入訊號變成 $t + 1$ 時間點的輸出結果
  - 這是早年常見的 RNN 公式表達法
- $\w{i}{j}$ 代表**輸入節點** $j$與**模型內部節點** $i$ 所連接的權重
  - 輸入節點可以是**外部輸入** $\x{j}{t}$ 或是**前次輸出** $\y{j}{t}$
  - 總共有 $\din + \dout$ 個輸入節點，因此 $1 \leq j \leq \din + \dout$
  - 總共有 $\dout$ 個內部節點，因此 $1 \leq i \leq \dout$
- $[x;y]$ 代表將外部輸入與前次輸出串接在一起

令模型使用的**啟發函數**（activation function）為 $f : \R^{\dout} \to \R^{\dout}$，並且內部節點之間無法直接溝通（elementwise activation function），則我們可以得到 $t + 1$ 時間的輸出

$$
\y{i}{t + 1} = \fnet{i}{t + 1} \tag{2}\label{eq:2}
$$

- 使用下標 $\f{i}$ 是因為每個維度所使用的啟發函數可以**不同**
- $f$ 必須要可以**微分**
- 當時幾乎都是使用 sigmoid 函數 $\sigma(x) = 1 / (1 + e^{-x})$

如果 $t + 1$ 時間點的**輸出目標**為 $\hat{y}(t + 1) \in \R^{\dout}$，則**目標函數**為**最小平方差**（Mean Square Error）：

$$
\loss{i}{t + 1} = \frac{1}{2} \big(\y{i}{t + 1} - \hy{i}{t + 1}\big)^2 \tag{3}\label{eq:3}
$$

因此 $t + 1$ 時間點的總體目標函數（總誤差）為

$$
\Loss{t + 1} = \S{i = 1}{\dout} \loss{i}{t + 1} \tag{4}\label{eq:4}
$$

根據 $\eqref{eq:3} \eqref{eq:4}$ 我們可以輕易的計算 $\loss{i}{t + 1}$ 對 $\Loss{t + 1}$ 所得梯度

$$
\pd{\Loss{t + 1}}{\loss{i}{t + 1}} = 1 \tag{5}\label{eq:5}
$$

而透過 $\eqref{eq:3}$ 我們知道 $\y{i}{t + 1}$ 對 $\Loss{t + 1}$ 所得梯度為

$$
\pd{\loss{i}{t + 1}}{\y{i}{t + 1}} = \y{i}{t + 1} - \hy{i}{t + 1} \tag{6}\label{eq:6}
$$

根據 $\eqref{eq:5} \eqref{eq:6}$ 我們可以推得

$$
\begin{align}
\pd{\Loss{t + 1}}{\y{i}{t + 1}} & = \pd{\Loss{t + 1}}{\loss{i}{t + 1}} \cdot \pd{\loss{i}{t + 1}}{\y{i}{t + 1}} \newline
& = 1 \cdot \big(\y{i}{t + 1} - \hy{i}{t + 1}\big) \newline
& = \y{i}{t + 1} - \hy{i}{t + 1}
\end{align} \tag{7}\label{eq:7}
$$

根據 $\eqref{eq:2}$ 我們知道 $\net{i}{t + 1}$ 對 $\y{i}{t + 1}$ 所得梯度為

$$
\pd{\y{i}{t + 1}}{\net{i}{t + 1}} = \dfnet{i}{t + 1} \tag{8}\label{eq:8}
$$

根據 $\eqref{eq:7} \eqref{eq:8}$ 我們可以推得 $\net{i}{t + 1}$ 對 $\Loss{t + 1}$ 所得梯度

$$
\begin{align}
\pd{\Loss{t + 1}}{\net{i}{t + 1}} & = \pd{\Loss{t + 1}}{\y{i}{t + 1}} \cdot \pd{\y{i}{t + 1}}{\net{i}{t + 1}} \newline
& = \dfnet{i}{t + 1} \cdot \big(\y{i}{t + 1} - \hy{i}{t + 1}\big)
\end{align} \tag{9}\label{eq:9}
$$

式子 $\eqref{eq:9}$ 就是論文 3.1.1 節的第一條公式。
根據 $\eqref{eq:9}$ 我們可以推得 $\x{j}{t}$ 對 $\Loss{t + 1}$ 所得梯度

$$
\begin{align}
\pd{\Loss{t + 1}}{\x{j}{t}} & = \S{i = 1}{\dout} \bigg[\pd{\Loss{t + 1}}{\net{i}{t + 1}} \cdot \pd{\net{i}{t + 1}}{\x{j}{t}}\bigg] \newline
& = \S{i = 1}{\dout} \bigg[\dfnet{i}{t + 1} \cdot \big(\y{i}{t + 1} - \hy{i}{t + 1}\big) \cdot \w{i}{j}\bigg]
\end{align} \tag{10}\label{eq:10}
$$

同樣的根據 $\eqref{eq:9}$ 我們可以推得 $\y{j}{t}$ 對 $\Loss{t + 1}$ 所得梯度為

$$
\begin{align}
\pd{\Loss{t + 1}}{\y{j}{t}} & = \S{i = 1}{\dout} \bigg[\pd{\Loss{t + 1}}{\net{i}{t + 1}} \cdot \pd{\net{i}{t + 1}}{\y{j}{t}}\bigg] \newline
& = \S{i = 1}{\dout} \bigg[\dfnet{i}{t + 1} \cdot \big(\y{i}{t + 1} - \hy{i}{t + 1}\big) \cdot \w{i}{j}\bigg]
\end{align} \tag{11}\label{eq:11}
$$

由於第 $t$ 時間點的輸出 $\y{j}{t}$ 的計算是由 $\net{j}{t}$ 而來（請見 $\eqref{eq:2}$），所以我們也利用 $\eqref{eq:8} \eqref{eq:11}$ 計算 $\net{j}{t}$ 對 $\Loss{t + 1}$ 所得梯度（注意是 $t$ 不是 $t + 1$）

$$
\begin{align}
& \pd{\Loss{t + 1}}{\net{j}{t}} \newline
& = \pd{\Loss{t + 1}}{\y{j}{t}} \cdot \pd{\y{j}{t}}{\net{j}{t}} \newline
& = \S{i = 1}{\dout} \bigg[\dfnet{i}{t + 1} \cdot \big(\y{i}{t + 1} - \hy{i}{t + 1}\big) \cdot \w{i}{j} \cdot \dfnet{j}{t}\bigg] \newline
& = \dfnet{j}{t} \cdot \S{i = 1}{\dout} \bigg[\w{i}{j} \cdot \dfnet{i}{t + 1} \cdot \big(\y{i}{t + 1} - \hy{i}{t + 1}\big)\bigg] \newline
& = \dfnet{j}{t} \cdot \S{i = 1}{\dout} \bigg[\w{i}{j} \cdot \pd{\Loss{t + 1}}{\net{i}{t + 1}}\bigg]
\end{align} \tag{12}\label{eq:12}
$$

式子 $\eqref{eq:12}$ 就是論文 3.1.1 節的最後一條公式。
模型參數 $\w{i}{j}$ 對於 $\Loss{t + 1}$ 所得梯度為

$$
\begin{align}
& \pd{\Loss{t + 1}}{\w{i}{j}} \newline
& = \pd{\Loss{t + 1}}{\net{i}{t + 1}} \cdot \pd{\net{i}{t + 1}}{\w{i}{j}} \newline
& = \dfnet{i}{t + 1} \cdot \big(\y{i}{t + 1} - \hy{i}{t + 1}\big) \cdot \xy{j}{t} && \text{(by \eqref{eq:9})}
\end{align} \tag{13}\label{eq:13}
$$

注意 $\eqref{eq:13}$ 中最後一行等式取決於 $\w{i}{j}$ 與哪個輸入相接。
而在時間點 $t + 1$ 進行參數更新的方法為

$$
\w{i}{j} \leftarrow \w{i}{j} - \alpha \pd{\Loss{t + 1}}{\w{i}{j}} \tag{14}\label{eq:14}
$$

$\eqref{eq:14}$ 就是最常用來最佳化神經網路的**梯度下降演算法**（Gradient Descent），$\alpha$ 代表**學習率**（Learning Rate）。

### 梯度爆炸 / 消失

從 $\eqref{eq:12}$ 式我們可以進一步推得 $t$ 時間點造成的梯度與前次時間點 ($t - 1, t - 2, \dots$) 所得的梯度**變化關係**。
注意這裡的變化關係指的是梯度與梯度之間的**變化率**，意即用時間點 $t - 1$ 的梯度對時間點 $t$ 的梯度算微分。

為了方便計算，我們定義新的符號

$$
{\large \dv{k}{\t{\text{future}}}{\t{\text{past}}}} \tag{15}\label{eq:15}
$$

意思是從**未來**時間點 $\t{\text{future}}$ 開始往回走到**過去**時間點 $\t{\text{past}}$，在**過去**時間點 $\t{\text{past}}$ 的第 $k$ 個**模型內部節點** $\net{k}{\t{\text{past}}}$ 對於**未來**時間點 $\t{\text{future}}$ 貢獻的**總誤差** $\Loss{\t{\text{future}}}$ 計算所得之**梯度**。

- 注意是貢獻總誤差所得之**梯度**
- 根據時間的限制我們有不等式 $0 \leq \t{\text{past}} \leq \t{\text{future}}$
- 節點 $k$ 的數值範圍為 $k = 1, \dots, \dout$，見式子 $\eqref{eq:1}$

因此下式如同 $\eqref{eq:9}$ 式

$$
\dv{\k{0}}{t}{t} = \pd{\Loss{t}}{\net{\k{0}}{t}} = \dfnet{\k{0}}{t} \cdot \big(\y{\k{0}}{t} - \hy{\k{0}}{t}\big) \tag{16}\label{eq:16}
$$

由 $\eqref{eq:12}$ 與 $\eqref{eq:16}$ 我們可以往回推 1 個時間點

$$
\begin{align}
\dv{\k{1}}{t}{t - 1} & = \pd{\Loss{t}}{\net{\k{1}}{t - 1}} \newline
& = \dfnet{\k{1}}{t - 1} \cdot \S{\k{0} = 1}{\dout} \bigg[\w{\k{0}}{\k{1}} \cdot \pd{\Loss{t}}{\net{\k{0}}{t}}\bigg] \newline
& = \S{\k{0} = 1}{\dout} \bigg[\w{\k{0}}{\k{1}} \cdot \dfnet{\k{1}}{t - 1} \cdot \dv{\k{0}}{t}{t}\bigg]
\end{align} \tag{17}\label{eq:17}
$$

由 $\eqref{eq:17}$ 我們可以往回推 2 個時間點

$$
\begin{align}
& \dv{\k{2}}{t}{t - 2} \newline
& = \pd{\Loss{t}}{\net{\k{2}}{t - 2}} \newline
& = \S{\k{1} = 1}{\dout} \bigg[\pd{\Loss{t}}{\net{\k{1}}{t - 1}} \cdot \pd{\net{\k{1}}{t - 1}}{\net{\k{2}}{t - 2}}\bigg] \newline
& = \S{\k{1} = 1}{\dout} \bigg[\dv{\k{1}}{t}{t - 1} \cdot \pd{\net{\k{1}}{t - 1}}{\y{\k{2}}{t - 2}} \cdot \pd{\y{\k{2}}{t - 2}}{\net{\k{2}}{t - 2}}\bigg] \newline
& = \S{\k{1} = 1}{\dout} \bigg[\dv{\k{1}}{t}{t - 1} \cdot \w{\k{1}}{\k{2}} \cdot \dfnet{\k{2}}{t - 2}\bigg] \newline
& = \S{\k{1} = 1}{\dout} \Bigg[\dfnet{\k{1}}{t - 1} \cdot \S{\k{0} = 1}{\dout} \bigg(\w{\k{0}}{\k{1}} \cdot \dv{\k{0}}{t}{t}\bigg) \cdot \w{\k{1}}{\k{2}} \cdot \dfnet{\k{2}}{t - 2}\Bigg] \newline
& = \S{\k{1} = 1}{\dout} \S{\k{0} = 1}{\dout} \bigg[\w{\k{0}}{\k{1}} \cdot \w{\k{1}}{\k{2}} \cdot \dfnet{\k{1}}{t - 1} \cdot \dfnet{\k{2}}{t - 2} \cdot \dv{\k{0}}{t}{t}\bigg]
\end{align} \tag{18}\label{eq:18}
$$

由 $\eqref{eq:18}$ 我們可以往回推 3 個時間點

$$
\begin{align}
& \dv{\k{3}}{t}{t - 3} \newline
& = \pd{\Loss{t}}{\net{\k{3}}{t - 3}} \newline
& = \S{\k{2} = 1}{\dout} \bigg[\pd{\Loss{t}}{\net{\k{2}}{t - 2}} \cdot \pd{\net{\k{2}}{t - 2}}{\net{\k{3}}{t - 3}}\bigg] \newline
& = \S{\k{2} = 1}{\dout} \bigg[\dv{\k{2}}{t}{t - 2} \cdot \pd{\net{\k{2}}{t - 2}}{\y{\k{3}}{t - 3}} \cdot \pd{\y{\k{3}}{t - 3}}{\net{\k{3}}{t - 3}}\bigg] \newline
& = \S{\k{2} = 1}{\dout} \bigg[\dv{\k{2}}{t}{t - 2} \cdot \w{\k{2}}{\k{3}} \cdot \dfnet{\k{3}}{t - 3}\bigg] \newline
& = \S{\k{2} = 1}{\dout} \Bigg[\S{\k{1} = 1}{\dout} \S{\k{0} = 1}{\dout} \bigg[\w{\k{0}}{\k{1}} \cdot \w{\k{1}}{\k{2}} \cdot \dfnet{\k{1}}{t - 1} \cdot \dfnet{\k{2}}{t - 2} \cdot \dv{\k{0}}{t}{t}\bigg] \newline
& \quad \cdot \w{\k{2}}{\k{3}} \cdot \dfnet{\k{3}}{t - 3}\Bigg] \newline
& = \S{\k{2} = 1}{\dout} \S{\k{1} = 1}{\dout} \S{\k{0} = 1}{\dout} \bigg[\w{\k{0}}{\k{1}} \cdot \w{\k{1}}{\k{2}} \cdot \w{\k{2}}{\k{3}} \cdot \newline
& \quad \dfnet{\k{1}}{t - 1} \cdot \dfnet{\k{2}}{t - 2} \cdot \dfnet{\k{3}}{t - 3} \cdot \dv{\k{0}}{t}{t}\bigg] \newline
& = \S{\k{2} = 1}{\dout} \S{\k{1} = 1}{\dout} \S{\k{0} = 1}{\dout} \Bigg[\bigg[\P{q = 1}{3} \w{\k{q - 1}}{\k{q}} \cdot \dfnet{\k{q}}{t - q}\bigg] \cdot \dv{\k{0}}{t}{t}\Bigg]
\end{align} \tag{19}\label{eq:19}
$$

由 $\eqref{eq:17} \eqref{eq:18} \eqref{eq:19}$ 我們可以歸納以下結論：
若 $n \geq 1$，則往回推 $n$ 個時間點的公式為

$$
\dv{\k{n}}{t}{t - n} = \S{\k{n - 1} = 1}{\dout} \cdots \S{\k{0} = 1}{\dout} \Bigg[\bigg[\P{q = 1}{n} \w{\k{q - 1}}{\k{q}} \cdot \dfnet{\k{q}}{t - q}\bigg] \cdot \dv{\k{0}}{t}{t}\Bigg] \tag{20}\label{eq:20}
$$

由 $\eqref{eq:20}$ 我們可以看出所有的 $\dv{\k{n}}{t}{t - n}$ 都與 $\dv{\k{0}}{t}{t}$ 相關，因此我們將 $\dv{\k{n}}{t}{t - n}$ 想成由 $\dv{\k{0}}{t}{t}$ 構成的函數。

現在讓我們固定 $\bar{\k{0}} \in \set{1, \dots, \dout}$，我們可以計算 $\dv{\bar{\k{0}}}{t}{t}$ 對於 $\dv{\k{n}}{t}{t - n}$ 的微分

- 當 $n = 1$ 時，根據 $\eqref{eq:17}$ 我們可以推得論文中的 (3.1) 式

  $$
  \pd{\dv{\k{n}}{t}{t - n}}{\dv{\bar{\k{0}}}{t}{t}} = \w{\bar{\k{0}}}{\k{1}} \cdot \dfnet{\k{1}}{t - 1} \tag{21}\label{eq:21}
  $$

- 當 $n > 1$ 時，根據 $\eqref{eq:20}$ 我們可以推得論文中的 (3.2) 式

  $$
  \pd{\dv{\k{n}}{t}{t - n}}{\dv{\bar{\k{0}}}{t}{t}} = \S{\k{n - 1} = 1}{\dout} \cdots \S{\k{1} = 1}{\dout} \S{\k{0} \in \set{\bar{\k{0}}}}{} \bigg[\P{q = 1}{n} \w{\k{q - 1}}{\k{q}} \cdot \dfnet{\k{q}}{t - q}\bigg] \tag{22}\label{eq:22}
  $$

**注意錯誤**：論文中的 (3.2) 式不小心把 $\w{\l{m - 1}}{\l{m}}$ 寫成 $\w{\l{m}}{\l{m - 1}}$。

因此根據 $\eqref{eq:22}$，共有 $(\dout)^{n - 1}$ 個連乘積項次進行加總，所得結果會以 $\eqref{eq:13} \eqref{eq:14}$ 直接影響權種更新 $w$。

根據 $\eqref{eq:21} \eqref{eq:22}$，如果

$$
\abs{\w{\k{q - 1}}{\k{q}} \cdot \dfnet{\k{q}}{t - q}} > 1.0 \quad \forall q = 1, \dots, n \tag{23}\label{eq:23}
$$

則 $w$ 的梯度會以指數 $n$ 增加，直接導致**梯度爆炸**，參數會進行**劇烈的振盪**，無法進行順利更新。

而如果

$$
\abs{\w{\k{q - 1}}{\k{q}} \cdot \dfnet{\k{q}}{t - q}} < 1.0 \quad \forall q = 1, \dots, n \tag{24}\label{eq:24}
$$

則 $w$ 的梯度會以指數 $n$ 縮小，直接導致**梯度消失**，誤差**收斂速度**會變得**非常緩慢**。

如果 $\f{\k{q}}$ 是 sigmoid function $\sigma$，則 $\sigma'$ 最大值為 $0.25$，理由是

$$
\begin{align}
\sigma(x) & = \frac{1}{1 + e^{-x}} \newline
\sigma'(x) & = \frac{e^{-x}}{(1 + e^{-x})^2} = \frac{1}{1 + e^{-x}} \cdot \frac{e^{-x}}{1 + e^{-x}} \newline
& = \frac{1}{1 + e^{-x}} \cdot \frac{1 + e^{-x} - 1}{1 + e^{-x}} = \sigma(x) \cdot \big(1 - \sigma(x)\big) \newline
\sigma(\R) & = (0, 1) \newline
\forall x \in \R, \max \sigma'(x) & = \sigma(0) * \big(1 - \sigma(0)\big) = 0.5 * 0.5 = 0.25
\end{align} \tag{25}\label{eq:25}
$$

因此當 $\abs{\w{\k{q - 1}}{\k{q}}} < 4.0$ 時我們可以發現

$$
\abs{\w{\k{q - 1}}{\k{q}} \cdot \dfnet{\k{q}}{t - q}} < 4.0 * 0.25 = 1.0 \tag{26}\label{eq:26}
$$

所以 $\eqref{eq:26}$ 與 $\eqref{eq:24}$ 的結論相輔相成：當 $\w{\k{q - 1}}{\k{q}}$ 的絕對值小於 $4.0$ 會造成梯度消失。

而 $\abs{\w{\k{q - 1}}{\k{q}}} \to \infty$ 我們可以得到

$$
\begin{align}
& \abs{\net{\k{q - 1}}{t - q - 1}} \to \infty \newline
\implies & \begin{cases}
\fnet{\k{q - 1}}{t - q - 1} \to 1 & \text{if } \net{\k{q - 1}}{t - q - 1} \to \infty \newline
\fnet{\k{q - 1}}{t - q - 1} \to 0 & \text{if } \net{\k{q - 1}}{t - q - 1} \to -\infty
\end{cases} \newline
\implies & \abs{\dfnet{\k{q - 1}}{t - q - 1}} \to 0 && \text{(by \eqref{eq:25})} \newline
\implies & \abs{\P{q = 1}{n} \w{\k{q - 1}}{\k{q}} \cdot \dfnet{\k{q}}{t - q}} \to 0
\end{align} \tag{27}\label{eq:27}
$$

**注意錯誤**：論文中的推論

$$
\abs{\w{\k{q - 1}}{\k{q}} \cdot \dfnet{\k{q}}{t - q}} \to 0
$$

是**錯誤**的，理由是 $\w{\k{q - 1}}{\k{q}}$ 無法對 $\net{\k{q}}{t - q}$ 造成影響，作者不小心把**時間順序寫反**了，但是**最後的邏輯仍然正確**，理由如 $\eqref{eq:27}$ 所示。

**注意錯誤**：論文中進行了以下**函數最大值**的推論

$$
\begin{align}
& \dfnet{\l{m}}{t - m}\big) \cdot \w{\l{m}}{\l{m - 1}} \newline
& = \sigma\big(\net{\l{m}}{t - m}\big) \cdot \Big(1 - \sigma\big(\net{\l{m}}{t - m}\big)\Big) \cdot \w{\l{m}}{\l{m - l}}
\end{align}
$$

最大值發生於微分值為 $0$ 的點，即我們想求出滿足以下式子的 $\w{\l{m}}{\l{m - 1}}$

$$
\pd{\Big[\sigma\big(\net{\l{m}}{t - m}\big) \cdot \Big(1 - \sigma\big(\net{\l{m}}{t - m}\big)\Big) \cdot \w{\l{m}}{\l{m - l}}\Big]}{\w{\l{m}}{\l{m - 1}}} = 0
$$

拆解微分式可得

$$
\begin{align}
& \pd{\Big[\sigma\big(\net{\l{m}}{t - m}\big) \cdot \Big(1 - \sigma\big(\net{\l{m}}{t - m}\big)\Big) \cdot \w{\l{m}}{\l{m - l}}\Big]}{\w{\l{m}}{\l{m - 1}}} \newline
& = \pd{\sigma\big(\net{\l{m}}{t - m}\big)}{\net{\l{m}}{t - m}} \cdot \pd{\net{\l{m}}{t - m}}{\w{\l{m}}{\l{m - 1}}} \cdot \Big(1 - \sigma\big(\net{\l{m}}{t - m}\big)\Big) \cdot \w{\l{m}}{\l{m - l}} \newline
& \quad + \sigma\big(\net{\l{m}}{t - m}\big) \cdot \pd{\Big(1 - \sigma\big(\net{\l{m}}{t - m}\big)\Big)}{\net{\l{m}}{t - m}} \cdot \pd{\net{\l{m}}{t - m}}{\w{\l{m}}{\l{m - 1}}} \cdot \w{\l{m}}{\l{m - l}} \newline
& \quad + \sigma\big(\net{\l{m}}{t - m}\big) \cdot \Big(1 - \sigma\big(\net{\l{m}}{t - m}\big)\Big) \cdot \pd{\w{\l{m}}{\l{m - 1}}}{\w{\l{m}}{\l{m - 1}}} \newline
& = \sigma\big(\net{\l{m}}{t - m}\big) \cdot \Big(1 - \sigma\big(\net{\l{m}}{t - m}\big)\Big)^2 \cdot \y{\l{m - 1}}{t - m - 1} \cdot \w{\l{m}}{\l{m - 1}} \newline
& \quad - \Big(\sigma\big(\net{\l{m}}{t - m}\big)\Big)^2 \cdot \Big(1 - \sigma\big(\net{\l{m}}{t - m}\big)\Big) \cdot \y{\l{m - 1}}{t - m - 1} \cdot \w{\l{m}}{\l{m - 1}} \newline
& \quad + \sigma\big(\net{\l{m}}{t - m}\big) \cdot \Big(1 - \sigma\big(\net{\l{m}}{t - m}\big)\Big) \newline
& = \Big[2 \Big(\sigma\big(\net{\l{m}}{t - m}\big)\Big)^3 - 3 \Big(\sigma\big(\net{\l{m}}{t - m}\big)\Big)^2 + \sigma\big(\net{\l{m}}{t - m}\big)\Big] \cdot \newline
& \quad \quad \y{\l{m - 1}}{t - m - 1} \cdot \w{\l{m}}{\l{m - 1}} \newline
& \quad + \sigma\big(\net{\l{m}}{t - m}\big) \cdot \Big(1 - \sigma\big(\net{\l{m}}{t - m}\big)\Big) \newline
& = \sigma\big(\net{\l{m}}{t - m}\big) \cdot \Big(2 \sigma\big(\net{\l{m}}{t - m}\big) - 1\Big) \cdot \Big(\sigma\big(\net{\l{m}}{t - m}\big) - 1\Big) \cdot \newline
& \quad \quad \y{\l{m - 1}}{t - m - 1} \cdot \w{\l{m}}{\l{m - 1}} \newline
& \quad + \sigma\big(\net{\l{m}}{t - m}\big) \cdot \Big(1 - \sigma\big(\net{\l{m}}{t - m}\big)\Big) \newline
& = 0
\end{align}
$$

移項後可以得到

$$
\begin{align}
& \sigma\big(\net{\l{m}}{t - m}\big) \cdot \Big(2 \sigma\big(\net{\l{m}}{t - m}\big) - 1\Big) \cdot \Big(1 - \sigma\big(\net{\l{m}}{t - m}\big)\Big) \cdot \newline
& \quad \quad \y{\l{m - 1}}{t - m - 1} \cdot \w{\l{m}}{\l{m - 1}} = \sigma\big(\net{\l{m}}{t - m}\big) \cdot \Big(1 - \sigma\big(\net{\l{m}}{t - m}\big)\Big) \newline
\implies & \Big(2 \sigma\big(\net{\l{m}}{t - m}\big) - 1\Big) \cdot \y{\l{m - 1}}{t - m - 1} \cdot \w{\l{m}}{\l{m - 1}} = 1 \newline
\implies & \w{\l{m}}{\l{m - 1}} = \frac{1}{\y{\l{m - 1}}{t - m - 1}} \cdot \frac{1}{2 \sigma\big(\net{\l{m}}{t - m}\big) - 1} \newline
\implies & \w{\l{m}}{\l{m - 1}} = \frac{1}{\y{\l{m - 1}}{t - m - 1}} \cdot \coth\bigg(\frac{\net{\l{m}}{t - m}}{2}\bigg)
\end{align}
$$

註：推論中使用了以下公式

$$
\begin{align}
\tanh(x) & = 2 \sigma(2x) - 1 \newline
\tanh(\frac{x}{2}) & = 2 \sigma(x) - 1 \newline
\coth(\frac{x}{2}) & = \frac{1}{\tanh(\frac{x}{2})} = \frac{1}{2 \sigma(x) - 1}
\end{align}
$$

但公式的前提不對，理由是 $\w{\l{m}}{\l{m - 1}}$ 根本不存在，應該改為 $\w{\l{m - 1}}{\l{m}}$（同 $\eqref{eq:22}$）。

接著我們推導時間點 $t - n$ 的節點 $\net{\k{n}}{t - n}$ 針對 $t$ 時間點造成的**總誤差**梯度**變化**：

$$
\S{\bar{\k{0}} = 1}{\dout} \pd{\dv{\k{n}}{t}{t - n}}{\dv{\bar{\k{0}}}{t}{t}} \tag{28}\label{eq:28}
$$

由於**每個項次**都能遭遇**梯度消失**，因此**總和**也會遭遇**梯度消失**。

## 梯度常數 (Constant Error Flow)

將**部份梯度**定為**常數**

- 想法有點矛盾：
  - 梯度應該隨著最佳化 (梯度下降) 的過程逐漸縮小數值
  - 但遇到了梯度消失的問題，因此要求梯度維持常數
  - 需要讓梯度隨著時間變小，卻又要求梯度維持常數，看起來互相**矛盾**

### 情境 1：模型輸出與內部節點 1-1 對應

假設模型輸出節點 $\y{j}{t - 1}$ 只與 $\net{j}{t}$ 相連，即

$$
\net{j}{t} = \w{j}{j} \y{j}{t - 1} \tag{29}\label{eq:29}
$$

（$\eqref{eq:29}$ 假設實際上不可能發生）則根據式子 $\eqref{eq:17}$ 我們可以推得

$$
\dv{j}{t}{t - 1} = \w{j}{j} \cdot \dfnet{j}{t - 1} \cdot \dv{j}{t}{t} \tag{30}\label{eq:30}
$$

為了強制讓梯度 $\dv{j}{t}{t}$ 不消失，作者認為需要強制達成

$$
\w{j}{j} \cdot \dfnet{j}{t - 1} = 1.0 \tag{31}\label{eq:31}
$$

如果 $\eqref{eq:31}$ 能夠達成，則積分 $\eqref{eq:31}$ 可以得到

$$
\begin{align}
& \int \w{j}{j} \cdot \dfnet{j}{t - 1} \; d \big[\net{j}{t - 1}\big] = \int 1.0 \; d \big[\net{j}{t - 1}\big] \newline
\implies & \w{j}{j} \cdot \fnet{j}{t - 1} = \net{j}{t - 1} \newline
\implies & \y{j}{t - 1} = \fnet{j}{t - 1} = \frac{\net{j}{t - 1}}{\w{j}{j}}
\end{align} \tag{32}\label{eq:32}
$$

觀察 $\eqref{eq:32}$ 我們可以發現

- 輸入 $\net{j}{t - 1}$ 與輸出 $\fnet{j}{t - 1}$ 之間的關係是乘上一個常數項 $\w{j}{j}$
- 代表函數 $\f{j}$ 其實是一個**線性函數**
- **每個時間點**的**輸出**居然**完全相同**，這個現象稱為 **Constant Error Carousel** (請見 $\eqref{eq:33}$)

$$
\begin{align}
\y{j}{t} & = \fnet{j}{t} = \f{j}\big(\w{j}{j} \y{j}{t - 1}\big) \newline
& = \f{j}\big(\w{j}{j} \frac{\net{j}{t - 1}}{\w{j}{j}}\big) = \fnet{j}{t - 1} = \y{j}{t - 1} \tag{33}\label{eq:33}
\end{align}
$$

### 情境 2：增加外部輸入

將 $\eqref{eq:29}$ 的假設改成每個模型內部節點可以額外接收一個外部輸入

$$
\net{j}{t} = \S{i = 1}{\din} \w{j}{i} \x{i}{t - 1} + \w{j}{j} \y{j}{t - 1} \tag{34}\label{eq:34}
$$

由於 $\y{j}{t - 1}$ 的設計功能是保留過去計算所擁有的資訊，在 $\eqref{eq:34}$ 的假設中唯一能夠**更新**資訊的方法只有透過 $\x{i}{t - 1}$ 配合 $\w{j}{i}$ 將新資訊合併進入 $\net{j}{t}$。

但作者認為，在計算的過程中，部份時間點的**輸入**資訊 $\x{i}{\cdot}$ 可以(甚至必須)被**忽略**，但這代表 $\w{j}{i}$ 需要**同時**達成**兩種**任務就必須要有**兩種不同的數值**：

- **加入新資訊**：代表 $\abs{\w{j}{i}} \neq 0$
- **忽略新資訊**：代表 $\abs{\w{j}{i}} \approx 0$

因此**無法只靠一個** $\w{j}{i}$ 決定**輸入**的影響，必須有**額外**能夠**理解當前內容 (context-sensitive)** 的功能模組幫忙**寫入** $\x{i}{\cdot}$

### 情境 3：輸出回饋到多個節點

將 $\eqref{eq:29} \eqref{eq:34}$ 的假設改回正常的模型架構

$$
\begin{align}
\net{j}{t} & = \S{i = 1}{\din} \w{j}{i} \x{i}{t - 1} + \S{i = 1}{\dout} \w{j}{i} \y{i}{t - 1} \newline
& = \S{i = 1}{\din} \w{j}{i} \x{i}{t - 1} + \S{i = 1}{\dout} \w{j}{i} f_i\big(\text{net}_i(t - 1)\big)
\end{align} \tag{35}\label{eq:35}
$$

由於 $\y{j}{t - 1}$ 的設計功能是保留過去計算所擁有的資訊，在 $\eqref{eq:35}$ 的假設中唯一能夠讓**過去**資訊**影響未來**計算結果的方法只有透過 $\y{i}{t - 1}$ 配合 $\w{j}{i}$ 將新資訊合併進入 $\net{j}{t}$。

但作者認為，在計算的過程中，部份時間點的**輸出**資訊 $y_i(*)$ 可以(甚至必須)被**忽略**，但這代表 $\w{j}{i}$ 需要**同時**達成**兩種**任務就必須要有**兩種不同的數值**：

- **保留過去資訊**：代表 $\abs{\w{j}{i}} \neq 0$
- **忽略過去資訊**：代表 $\abs{\w{j}{i}} \approx 0$

因此**無法只靠一個** $\w{j}{i}$ 決定**輸出**的影響，必須有**額外**能夠**理解當前內容 (context-sensitive)** 的功能模組幫忙**讀取** $y_i(*)$

值得一提的是，上述的假設是基於以下的事實觀察：
已知 RNN 能夠學習解決多個記憶時間較短 (short-time-lag) 的任務，但如果要能夠同時解決記憶時間較長 (long-time-lag) 的任務，則模型應該依照以下順序執行：

1. 記住短期資訊 $\t{0} \sim \t{1}$ (需要寫入功能)
2. 解決需要短期資訊 $\t{0} \sim \t{1}$ 的任務 (需要讀取功能)
3. 忘記短期資訊 $\t{0} \sim \t{1}$ (需要忽略功能)
4. 記住短期資訊 $\t{1} \sim \t{2}$ (需要寫入功能)
5. 解決需要短期資訊 $\t{1} \sim \t{2}$ 的任務 (需要讀取功能)
6. 忘記短期資訊 $\t{1} \sim \t{2}$ (需要忽略功能)
7. 為了解決與短期資訊 $\t{0} \sim \t{1}$ 相關的任務，突然又需要回憶起短期資訊 $\t{0} \sim \t{1}$ (需要寫入 + 讀取功能)
