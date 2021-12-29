---
layout: ML-note
title:  "Long Short-Term Memory"
date:   2021-11-14 15:28:00 +0800
categories: [
  Deep Learning,
  Model Architecture,
  Optimization,
]
tags: [
  LSTM,
  RTRL,
  BPTT,
  Gradient Explosion,
  Gradient Vanishing,
]
author: [
  Sepp Hochreiter,
  Jürgen Schmidhuber,
]
---

|-|-|
|目標|提出 RNN 使用 BPTT 進行最佳化時遇到的問題，並提出 LSTM 架構進行修正|
|作者|Sepp Hochreiter, Jürgen Schmidhuber|
|期刊/會議名稱|Neural Computation|
|發表時間|1997|
|論文連結|<https://ieeexplore.ieee.org/abstract/document/6795963>|
|書本連結|<https://link.springer.com/chapter/10.1007/978-3-642-24797-2_4>|

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
  <!-- Operator sequence. -->
  $\providecommand{\opseq}{}$
  $\renewcommand{\opseq}{\operatorname{seq}}$

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
  $\renewcommand{\hcell}[3]{h_{#1}^{\cell{k}}\big(s_{#1}^{\cell{#2}}(#3)\big)}$
  <!-- Derivative of h with respect to cell unit. -->
  $\providecommand{\dhcell}{}$
  $\renewcommand{\dhcell}[3]{h_{#1}^{\cell{k}}{'}\big(s_{#1}^{\cell{#2}}(#3)\big)}$

  <!-- Gradient approximation by truncating gradient. -->
  $\providecommand{\aptr}{}$
  $\renewcommand{\aptr}{\approx_{\operatorname{tr}}}$
</p>

<!-- End LaTeX command define section. -->

## 重點

- [此篇論文][論文]與 [LSTM-2000][LSTM2000] 都寫錯自己的數學公式，但我讓筆記內容儘量與論文原始內容相同，因此所有筆記都是以論文原始（錯誤）的數學公式為基礎，正確的數學公式可以看 [LSTM-2002][LSTM2002] 的論文或[我的筆記][note-LSTM2002]
- 計算 **RNN** 梯度反向傳播的演算法包含 **BPTT** 或 **RTRL**
  - BPTT 全名為 **B**ack-**P**ropagation **T**hrought **T**ime
  - RTRL 全名為 **R**eal **T**ime **R**ecurrent **L**earning
- 不論使用 BPTT 或 RTRL，RNN 的梯度都會面臨**爆炸**或**消失**的問題
  - 梯度**爆炸**造成神經網路的**權重劇烈振盪**
  - 梯度**消失**造成**訓練時間慢長**
  - 無法解決**時間差較長**的問題
- 論文提出 **LSTM + RTRL** 能夠解決上述問題
  - Backward pass 演算法**時間複雜度**為 $O(w)$，$w$ 代表權重
  - Backward pass 演算法**空間複雜度**也為 $O(w)$，因此**沒有輸入長度的限制**
  - 此結論必須依靠**丟棄部份梯度**並使用 **RTRL** 才能以有**效率**的辦法解決梯度**爆炸**或**消失**
- 使用**乘法閘門**（**Multiplicative Gate**）學習**開啟** / **關閉**模型記憶**寫入** / **讀取**機制
- LSTM 的**閘門單元參數**應該讓**偏差項**（bias term）初始化成**負數**
  - 輸**入**閘門偏差項初始化成負數能夠解決**內部狀態偏差行為**（**Internal State Drift**）
  - 輸**出**閘門偏差項初始化成負數能夠避免模型**濫用記憶單元初始值**與**訓練初期梯度過大**
  - 如果沒有輸出閘門，則**收斂速度會變慢**
- 根據實驗 LSTM 能夠達成以下任務
  - 擁有處理**短時間差**（**Short Time Lag**）任務的能力
  - 擁有處理**長時間差**（**Long Time Lag**）任務的能力
  - 能夠處理最長時間差長達 $1000$ 個單位的任務
  - 輸入訊號含有雜訊時也能處理
- LSTM 的缺點
  - 仍然無法解決 delayed XOR 問題
    - 改成 BPTT 可能可以解決，但計算複雜度變高
    - CEC 在使用 BPTT 後有可能無效，但根據實驗使用 BPTT 時誤差傳遞的過程中很快就消失
  - 在部份任務上無法比 random weight guessing 最佳化速度還要快
    - 例如 500-bit parity
    - 使用 CEC 才導致此後果
    - 但計算效率高，最佳化過程也比較穩定
  - 無法精確的判斷重要訊號的輸入時間
    - 所有使用梯度下降作為最佳演算法的模型都有相同問題
    - 如果精確判斷是很重要的功能，則作者認為需要幫模型引入計數器的功能
- 當單一字元的**出現次數期望值增加**時，**學習速度會下降**
  - 作者認為是常見字詞的出現導致參數開始振盪
- 與 [PyTorch][Pytorch-LSTM] 實作的 LSTM 完全不同
  - 本篇論文的架構定義更為**廣義**
  - 本篇論文只有**輸入閘門**（**Input Gate**）跟**輸出閘門**（**Output Gate**），並沒有使用**失憶閘門**（**Forget Gate**）

## 傳統的 RNN

### 模型輸入

一個 RNN 模型的**輸入**來源共有兩種：

- **外部輸入**（**External Input**） $x(t)$
  - 輸入維度為 $\din$
  - 使用下標 $x_{j}(t)$ 代表不同的輸入訊號，$j = 1, \dots, \din$
- **總輸出**（**Total Output**） $y(t)$
  - 輸出維度為 $\dout$
  - 使用下標 $y_{j}(t)$ 代表不同的輸入訊號，$j = \din + 1, \dots, \din + \dout$
  - 注意這裡是使用 $t$ 不是 $t - 1$
- $t$ 的起始值為 $0$，結束值為 $T$，每次遞增 $1$
  - 時間為離散狀態
  - 方便起見令 $y(0) = 0$

### 模型輸出

令 RNN 模型的**參數**為 $w \in \R^{\dout \times (\din + \dout)}$，如果我們已經取得 $t$ 時間點的**外部輸入** $x(t)$ 與**總輸出** $y(t)$，則我們可以定義 $t + 1$ 時間點的第 $i$ 個**模型內部節點** $\net{i}{t}$

$$
\begin{align*}
  \net{i}{t + 1} & = \sum_{j = 1}^{\din} w_{i, j} \cdot x_{j}(t) + \sum_{j = \din + 1}^{\din + \dout} w_{i, j} \cdot y_{j}(t) \\
  & = \sum_{j = 1}^{\din + \dout} w_{i, j} \cdot [x ; y]_{j}(t)
\end{align*} \tag{1}\label{eq:1}
$$

- $\net{i}{t + 1}$ 代表第 $t + 1$ 時間的**模型內部節點** $i$ 所收到的**淨輸入（total input）**
  - 注意 $t$ 時間點的輸入訊號變成 $t + 1$ 時間點的輸出結果
  - 這是早年常見的 RNN 公式表達法
- $w_{i, j}$ 代表**輸入節點** $j$與**模型內部節點** $i$ 所連接的權重
  - 輸入節點可以是**外部輸入** $x_{j}(t)$ 或是**總輸出** $y_{j}(t)$
  - 總共有 $\din + \dout$ 個輸入節點，因此 $1 \leq j \leq \din + \dout$
  - 總共有 $\dout$ 個內部節點，因此 $1 \leq i \leq \dout$
- $[x ; y]$ 代表將外部輸入與總輸出**串接**在一起

令模型使用的**啟發函數**（**Activation Function**）為 $f : \R^{\dout} \to \R^{\dout}$，並且內部節點之間無法直接溝通（**Element-wise** Activation Function），則我們可以得到 $t + 1$ 時間的輸出

$$
y_{i}(t + 1) = \fnet{i}{t + 1} \tag{2}\label{eq:2}
$$

- 使用下標 $f_{i}$ 是因為每個維度所使用的啟發函數可以**不同**
- $f$ 必須要可以**微分**
- 當時幾乎都是使用 sigmoid 函數 $\sigma(x) = 1 / (1 + e^{-x})$

### 計算誤差

如果 $t + 1$ 時間點的**輸出目標**為 $\hat{y}(t + 1) \in \R^{\dout}$，則**目標函數**為**最小平方差**（Mean Square Error）：

$$
\loss{i}{t + 1} = \frac{1}{2} \big(y_{i}(t + 1) - \hat{y}_{i}(t + 1)\big)^2 \tag{3}\label{eq:3}
$$

因此 $t + 1$ 時間點的總體目標函數（總誤差）為

$$
\Loss{t + 1} = \sum_{i = 1}^{\dout} \loss{i}{t + 1} \tag{4}\label{eq:4}
$$

### 反向傳播

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
& = \sum_{i = 1}^{\dout} \bigg[\dfnet{i}{t + 1} \cdot \big(y_{i}(t + 1) - \hat{y}_{i}(t + 1)\big) \cdot w_{i, j}\bigg]
\end{align*} \tag{10}\label{eq:10}
$$

同樣的根據 $\eqref{eq:9}$ 我們可以推得 $y_{j}(t)$ 對 $\Loss{t + 1}$ 所得梯度為

$$
\begin{align*}
\pd{\Loss{t + 1}}{y_{j}(t)} & = \sum_{i = 1}^{\dout} \bigg[\pd{\Loss{t + 1}}{\net{i}{t + 1}} \cdot \pd{\net{i}{t + 1}}{y_{j}(t)}\bigg] \\
& = \sum_{i = 1}^{\dout} \bigg[\dfnet{i}{t + 1} \cdot \big(y_{i}(t + 1) - \hat{y}_{i}(t + 1)\big) \cdot w_{i, j}\bigg]
\end{align*} \tag{11}\label{eq:11}
$$

由於第 $t$ 時間點的輸出 $y_{j}(t)$ 的計算是由 $\net{j}{t}$ 而來（請見 $\eqref{eq:2}$），所以我們也利用 $\eqref{eq:8} \eqref{eq:11}$ 計算 $\net{j}{t}$ 對 $\Loss{t + 1}$ 所得梯度（注意是 $t$ 不是 $t + 1$）

$$
\begin{align*}
& \pd{\Loss{t + 1}}{\net{j}{t}} \\
& = \pd{\Loss{t + 1}}{y_{j}(t)} \cdot \pd{y_{j}(t)}{\net{j}{t}} \\
& = \sum_{i = 1}^{\dout} \bigg[\dfnet{i}{t + 1} \cdot \big(y_{i}(t + 1) - \hat{y}_{i}(t + 1)\big) \cdot w_{i, j} \cdot \dfnet{j}{t}\bigg] \\
& = \dfnet{j}{t} \cdot \sum_{i = 1}^{\dout} \bigg[w_{i, j} \cdot \dfnet{i}{t + 1} \cdot \big(y_{i}(t + 1) - \hat{y}_{i}(t + 1)\big)\bigg] \\
& = \dfnet{j}{t} \cdot \sum_{i = 1}^{\dout} \bigg[w_{i, j} \cdot \pd{\Loss{t + 1}}{\net{i}{t + 1}}\bigg]
\end{align*} \tag{12}\label{eq:12}
$$

式子 $\eqref{eq:12}$ 就是論文 3.1.1 節的最後一條公式。
模型參數 $w_{i, j}$ 對於 $\Loss{t + 1}$ 所得梯度為

$$
\begin{align*}
& \pd{\Loss{t + 1}}{w_{i, j}} \\
& = \pd{\Loss{t + 1}}{\net{i}{t + 1}} \cdot \pd{\net{i}{t + 1}}{w_{i, j}} \\
& = \dfnet{i}{t + 1} \cdot \big(y_{i}(t + 1) - \hat{y}_{i}(t + 1)\big) \cdot [x ; y]_{j}(t) && \text{(by \eqref{eq:9})}
\end{align*} \tag{13}\label{eq:13}
$$

注意 $\eqref{eq:13}$ 中最後一行等式取決於 $w_{i, j}$ 與哪個輸入相接。
而在時間點 $t + 1$ 進行參數更新的方法為

$$
w_{i, j} \leftarrow w_{i, j} - \alpha \pd{\Loss{t + 1}}{w_{i, j}} \tag{14}\label{eq:14}
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
& = \dfnet{k_{1}}{t - 1} \cdot \sum_{k_{0} = 1}^{\dout} \bigg[w_{k_{0}, k_{1}} \cdot \pd{\Loss{t}}{\net{k_{0}}{t}}\bigg] \\
& = \sum_{k_{0} = 1}^{\dout} \bigg[w_{k_{0}, k_{1}} \cdot \dfnet{k_{1}}{t - 1} \cdot \dv{k_{0}}{t}{t}\bigg]
\end{align*} \tag{17}\label{eq:17}
$$

由 $\eqref{eq:17}$ 我們可以往回推 2 個時間點

$$
\begin{align*}
& \dv{k_{2}}{t}{t - 2} \\
& = \pd{\Loss{t}}{\net{k_{2}}{t - 2}} \\
& = \sum_{k_{1} = 1}^{\dout} \bigg[\pd{\Loss{t}}{\net{k_{1}}{t - 1}} \cdot \pd{\net{k_{1}}{t - 1}}{\net{k_{2}}{t - 2}}\bigg] \\
& = \sum_{k_{1} = 1}^{\dout} \bigg[\dv{k_{1}}{t}{t - 1} \cdot \pd{\net{k_{1}}{t - 1}}{y_{k_{2}}(t - 2)} \cdot \pd{y_{k_{2}}(t - 2)}{\net{k_{2}}{t - 2}}\bigg] \\
& = \sum_{k_{1} = 1}^{\dout} \bigg[\dv{k_{1}}{t}{t - 1} \cdot w_{k_{1}, k_{2}} \cdot \dfnet{k_{2}}{t - 2}\bigg] \\
& = \sum_{k_{1} = 1}^{\dout} \Bigg[\dfnet{k_{1}}{t - 1} \cdot \sum_{k_{0} = 1}^{\dout} \bigg(w_{k_{0}, k_{1}} \cdot \dv{k_{0}}{t}{t}\bigg) \cdot w_{k_{1}, k_{2}} \cdot \dfnet{k_{2}}{t - 2}\Bigg] \\
& = \sum_{k_{1} = 1}^{\dout} \sum_{k_{0} = 1}^{\dout} \bigg[w_{k_{0}, k_{1}} \cdot w_{k_{1}, k_{2}} \cdot \dfnet{k_{1}}{t - 1} \cdot \dfnet{k_{2}}{t - 2} \cdot \dv{k_{0}}{t}{t}\bigg]
\end{align*} \tag{18}\label{eq:18}
$$

由 $\eqref{eq:18}$ 我們可以往回推 3 個時間點

$$
\begin{align*}
& \dv{k_{3}}{t}{t - 3} \\
& = \pd{\Loss{t}}{\net{k_{3}}{t - 3}} \\
& = \sum_{k_{2} = 1}^{\dout} \bigg[\pd{\Loss{t}}{\net{k_{2}}{t - 2}} \cdot \pd{\net{k_{2}}{t - 2}}{\net{k_{3}}{t - 3}}\bigg] \\
& = \sum_{k_{2} = 1}^{\dout} \bigg[\dv{k_{2}}{t}{t - 2} \cdot \pd{\net{k_{2}}{t - 2}}{y_{k_{3}}(t - 3)} \cdot \pd{y_{k_{3}}(t - 3)}{\net{k_{3}}{t - 3}}\bigg] \\
& = \sum_{k_{2} = 1}^{\dout} \bigg[\dv{k_{2}}{t}{t - 2} \cdot w_{k_{2}, k_{3}} \cdot \dfnet{k_{3}}{t - 3}\bigg] \\
& = \sum_{k_{2} = 1}^{\dout} \Bigg[\sum_{k_{1} = 1}^{\dout} \sum_{k_{0} = 1}^{\dout} \bigg[w_{k_{0}, k_{1}} \cdot w_{k_{1}, k_{2}} \cdot \dfnet{k_{1}}{t - 1} \cdot \dfnet{k_{2}}{t - 2} \cdot \dv{k_{0}}{t}{t}\bigg] \\
& \quad \cdot w_{k_{2}, k_{3}} \cdot \dfnet{k_{3}}{t - 3}\Bigg] \\
& = \sum_{k_{2} = 1}^{\dout} \sum_{k_{1} = 1}^{\dout} \sum_{k_{0} = 1}^{\dout} \bigg[w_{k_{0}, k_{1}} \cdot w_{k_{1}, k_{2}} \cdot w_{k_{2}, k_{3}} \cdot \\
& \quad \dfnet{k_{1}}{t - 1} \cdot \dfnet{k_{2}}{t - 2} \cdot \dfnet{k_{3}}{t - 3} \cdot \dv{k_{0}}{t}{t}\bigg] \\
& = \sum_{k_{2} = 1}^{\dout} \sum_{k_{1} = 1}^{\dout} \sum_{k_{0} = 1}^{\dout} \Bigg[\bigg[\prod_{q = 1}^{3} w_{k_{q - 1}, k_{q}} \cdot \dfnet{k_{q}}{t - q}\bigg] \cdot \dv{k_{0}}{t}{t}\Bigg]
\end{align*} \tag{19}\label{eq:19}
$$

由 $\eqref{eq:17} \eqref{eq:18} \eqref{eq:19}$ 我們可以歸納以下結論：
若 $n \geq 1$，則往回推 $n$ 個時間點的公式為

$$
\dv{k_{n}}{t}{t - n} = \sum_{k_{n - 1} = 1}^{\dout} \cdots \sum_{k_{0} = 1}^{\dout} \Bigg[\bigg[\prod_{q = 1}^{n} w_{k_{q - 1}, k_{q}} \cdot \dfnet{k_{q}}{t - q}\bigg] \cdot \dv{k_{0}}{t}{t}\Bigg] \tag{20}\label{eq:20}
$$

由 $\eqref{eq:20}$ 我們可以看出所有的 $\dv{k_{n}}{t}{t - n}$ 都與 $\dv{k_{0}}{t}{t}$ 相關，因此我們將 $\dv{k_{n}}{t}{t - n}$ 想成由 $\dv{k_{0}}{t}{t}$ 構成的函數。

現在讓我們固定 $k_{0}^{\star} \in \set{1, \dots, \dout}$，我們可以計算 $\dv{k_{0}^{\star}}{t}{t}$ 對於 $\dv{k_{n}}{t}{t - n}$ 的微分

- 當 $n = 1$ 時，根據 $\eqref{eq:17}$ 我們可以推得論文中的 (3.1) 式

  $$
  \pd{\dv{k_{n}}{t}{t - n}}{\dv{k_{0}^{\star}}{t}{t}} = w_{k_{0}^{\star}, k_{1}} \cdot \dfnet{k_{1}}{t - 1} \tag{21}\label{eq:21}
  $$

- 當 $n > 1$ 時，根據 $\eqref{eq:20}$ 我們可以推得論文中的 (3.2) 式

  $$
  \pd{\dv{k_{n}}{t}{t - n}}{\dv{k_{0}^{\star}}{t}{t}} = \sum_{k_{n - 1} = 1}^{\dout} \cdots \sum_{k_{1} = 1}^{\dout} \sum_{k_{0} \in \set{k_{0}^{\star}}} \bigg[\prod_{q = 1}^{n} w_{k_{q - 1}, k_{q}} \cdot \dfnet{k_{q}}{t - q}\bigg] \tag{22}\label{eq:22}
  $$

**注意錯誤**：論文中的 (3.2) 式不小心把 $w_{l_{m - 1} l_{m}}$ 寫成 $w_{l_{m} l_{m - 1}}$。

因此根據 $\eqref{eq:22}$，共有 $(\dout)^{n - 1}$ 個連乘積項次進行加總，所得結果會以 $\eqref{eq:13} \eqref{eq:14}$ 直接影響權種更新 $w$。

根據 $\eqref{eq:21} \eqref{eq:22}$，如果

$$
\abs{w_{k_{q - 1}, k_{q}} \cdot \dfnet{k_{q}}{t - q}} > 1.0 \quad \forall q = 1, \dots, n \tag{23}\label{eq:23}
$$

則 $w$ 的梯度會以指數 $n$ 增加，直接導致**梯度爆炸**，參數會進行**劇烈的振盪**，無法進行順利更新。

而如果

$$
\abs{w_{k_{q - 1}, k_{q}} \cdot \dfnet{k_{q}}{t - q}} < 1.0 \quad \forall q = 1, \dots, n \tag{24}\label{eq:24}
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

因此當 $\abs{w_{k_{q - 1}, k_{q}}} < 4.0$ 時我們可以發現

$$
\abs{w_{k_{q - 1}, k_{q}} \cdot \dfnet{k_{q}}{t - q}} < 4.0 * 0.25 = 1.0 \tag{26}\label{eq:26}
$$

所以 $\eqref{eq:26}$ 與 $\eqref{eq:24}$ 的結論相輔相成：當 $w_{k_{q - 1}, k_{q}}$ 的絕對值小於 $4.0$ 會造成梯度消失。

而 $\abs{w_{k_{q - 1}, k_{q}}} \to \infty$ 我們可以使用 $\eqref{eq:25}$ 得到

$$
\begin{align*}
& \abs{\net{k_{q - 1}}{t - q + 1}} \to \infty \\
\implies & \begin{dcases}
\fnet{k_{q - 1}}{t - q + 1} \to 1 & \text{if } \net{k_{q - 1}}{t - q + 1} \to \infty \\
\fnet{k_{q - 1}}{t - q + 1} \to 0 & \text{if } \net{k_{q - 1}}{t - q + 1} \to -\infty
\end{dcases} \\
\implies & \abs{\dfnet{k_{q - 1}}{t - q + 1}} \to 0 \\
\implies & \abs{\prod_{q = 1}^{n} w_{k_{q - 1}, k_{q}} \cdot \dfnet{k_{q}}{t - q}} \\
& = \abs{w_{k_0, k_1} \cdot \prod_{q = 2}^{n} \bigg[\dfnet{k_{q - 1}}{t - q + 1} \cdot w_{k_{q - 1}, k_{q}}\bigg] \cdot \dfnet{k_{n}}{t - n}} \\
& \to 0
\end{align*} \tag{27}\label{eq:27}
$$

最後一個推論的原理是**指數收斂的速度比線性快**。

**注意錯誤**：論文中的推論

$$
\abs{w_{k_{q - 1}, k_{q}} \cdot \dfnet{k_{q}}{t - q}} \to 0
$$

是**錯誤**的，理由是 $w_{k_{q - 1}, k_{q}}$ 無法對 $\net{k_{q}}{t - q}$ 造成影響，作者不小心把**時間順序寫反**了，但是**最後的邏輯仍然正確**，理由如 $\eqref{eq:27}$ 所示。

**注意錯誤**：論文中進行了以下**函數最大值**的推論

$$
\begin{align*}
& \dfnet{l_{m}}{t - m}\big) \cdot w_{l_{m} l_{m - 1}} \\
& = \sigma\big(\net{l_{m}}{t - m}\big) \cdot \Big(1 - \sigma\big(\net{l_{m}}{t - m}\big)\Big) \cdot w_{l_{m} l_{m - l}}
\end{align*}
$$

最大值發生於微分值為 $0$ 的點，即我們想求出滿足以下式子的 $w_{l_{m} l_{m - 1}}$

$$
\pd{\Big[\sigma\big(\net{l_{m}}{t - m}\big) \cdot \Big(1 - \sigma\big(\net{l_{m}}{t - m}\big)\Big) \cdot w_{l_{m} l_{m - l}}\Big]}{w_{l_{m} l_{m - 1}}} = 0
$$

拆解微分式可得

$$
\begin{align*}
& \pd{\Big[\sigma\big(\net{l_{m}}{t - m}\big) \cdot \Big(1 - \sigma\big(\net{l_{m}}{t - m}\big)\Big) \cdot w_{l_{m} l_{m - l}}\Big]}{w_{l_{m} l_{m - 1}}} \\
& = \pd{\sigma\big(\net{l_{m}}{t - m}\big)}{\net{l_{m}}{t - m}} \cdot \pd{\net{l_{m}}{t - m}}{w_{l_{m} l_{m - 1}}} \cdot \Big(1 - \sigma\big(\net{l_{m}}{t - m}\big)\Big) \cdot w_{l_{m} l_{m - l}} \\
& \quad + \sigma\big(\net{l_{m}}{t - m}\big) \cdot \pd{\Big(1 - \sigma\big(\net{l_{m}}{t - m}\big)\Big)}{\net{l_{m}}{t - m}} \cdot \pd{\net{l_{m}}{t - m}}{w_{l_{m} l_{m - 1}}} \cdot w_{l_{m} l_{m - l}} \\
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

接著我們可以計算 $t$ 時間點 $\dout$ 個**不同**節點 $\net{k_0^{\star}}{t}$ 對於**同一個** $t - n$ 時間點的 $\net{k_{n}}{t - n}$ 節點所貢獻的**梯度變化總和**：

$$
\sum_{k_{0}^{\star} = 1}^{\dout} \pd{\dv{k_{n}}{t}{t - n}}{\dv{k_{0}^{\star}}{t}{t}} \tag{28}\label{eq:28}
$$

由於**每個項次**都能遭遇**梯度消失**，因此**總和**也會遭遇**梯度消失**。

## 問題觀察

### 情境 1：模型輸出與內部節點 1-1 對應

**梯度常數（Constant Error Flow）**的概念是控制**部份梯度**為**常數**。

- 透過 $\eqref{eq:31}$ 的想法讓梯度的**連乘積項**為 $1.0$
  - 不會像 $\eqref{eq:23}$ 導致梯度**爆炸**
  - 不會像 $\eqref{eq:24}$ 導致梯度**消失**
- 要達成 $\eqref{eq:31}$，就必須讓 $f_j$ 是**線性函數**

假設模型輸出節點 $y_{j}(t - 1)$ 只與 $\net{j}{t}$ 相連，即

$$
\net{j}{t} = w_{j, j} y_{j}(t - 1) \tag{29}\label{eq:29}
$$

（$\eqref{eq:29}$ 假設實際上不可能發生）則根據式子 $\eqref{eq:17}$ 我們可以推得

$$
\dv{j}{t}{t - 1} = w_{j, j} \cdot \dfnet{j}{t - 1} \cdot \dv{j}{t}{t} \tag{30}\label{eq:30}
$$

為了強制讓梯度 $\dv{j}{t}{t}$ 不消失，作者認為需要強制達成

$$
w_{j, j} \cdot \dfnet{j}{t - 1} = 1.0 \tag{31}\label{eq:31}
$$

如果 $\eqref{eq:31}$ 能夠達成，則積分 $\eqref{eq:31}$ 可以得到

$$
\begin{align*}
& \int w_{j, j} \cdot \dfnet{j}{t - 1} \; d \big[\net{j}{t - 1}\big] = \int 1.0 \; d \big[\net{j}{t - 1}\big] \\
\iff & w_{j, j} \cdot \fnet{j}{t - 1} = \net{j}{t - 1} \\
\iff & y_{j}(t - 1) = \fnet{j}{t - 1} = \frac{\net{j}{t - 1}}{w_{j, j}}
\end{align*} \tag{32}\label{eq:32}
$$

觀察 $\eqref{eq:32}$ 我們可以發現

- 輸入 $\net{j}{t - 1}$ 與輸出 $\fnet{j}{t - 1}$ 之間的關係是乘上一個常數項 $w_{j, j}$
- 代表函數 $f_{j}$ 其實是一個**線性函數**
- **每個時間點**的**輸出**居然**完全相同**，這個現象稱為 **Constant Error Carousel** (請見 $\eqref{eq:33}$)

$$
\begin{align*}
y_{j}(t) & = \fnet{j}{t} = f_{j}\big(w_{j, j} y_{j}(t - 1)\big) \\
& = f_{j}\big(w_{j, j} \frac{\net{j}{t - 1}}{w_{j, j}}\big) = \fnet{j}{t - 1} = y_{j}(t - 1) \tag{33}\label{eq:33}
\end{align*}
$$

### 情境 2：增加外部輸入

將 $\eqref{eq:29}$ 的假設改成每個模型內部節點可以額外接收**外部輸入**

$$
\net{j}{t} = \sum_{i = 1}^{\din} w_{j, i} x_{i}(t - 1) + w_{j, j} y_{j}(t - 1) \tag{34}\label{eq:34}
$$

由於 $y_{j}(t - 1)$ 的設計功能是保留過去計算所擁有的資訊，在 $\eqref{eq:34}$ 的假設中唯一能夠**更新**資訊的方法只有透過 $x_{i}(t - 1)$ 配合 $w_{j, i}$ 將新資訊合併進入 $\net{j}{t}$。

但作者認為，在計算的過程中，部份時間點的**輸入**資訊 $x_{i}(\cdot)$ 可以(甚至必須)被**忽略**，但這代表 $w_{j, i}$ 需要**同時**達成**兩種**任務就必須要有**兩種不同的數值**：

- **加入新資訊**：代表 $\abs{w_{j, i}} \neq 0$
- **忽略新資訊**：代表 $\abs{w_{j, i}} \approx 0$

因此**無法只靠一個** $w_{j, i}$ 決定**輸入**的影響，必須有**額外**能夠**理解當前內容 (context-sensitive)** 的功能模組幫忙**寫入** $x_{i}(\cdot)$

### 情境 3：輸出回饋到多個節點

將 $\eqref{eq:29} \eqref{eq:34}$ 的假設改回正常的模型架構

$$
\begin{align*}
\net{j}{t} & = \sum_{i = 1}^{\din} w_{j, i} x_{i}(t - 1) + \sum_{i = \din + 1}^{\din + \dout} w_{j, i} y_{i}(t - 1) \\
& = \sum_{i = 1}^{\din} w_{j, i} x_{i}(t - 1) + \sum_{i = \din + 1}^{\din + \dout} w_{j, i} \fnet{i}{t - 1}
\end{align*} \tag{35}\label{eq:35}
$$

由於 $y_{j}(t - 1)$ 的設計功能是保留過去計算所擁有的資訊，在 $\eqref{eq:35}$ 的假設中唯一能夠讓**過去**資訊**影響未來**計算結果的方法只有透過 $y_{i}(t - 1)$ 配合 $w_{j, i}$ 將新資訊合併進入 $\net{j}{t}$。

但作者認為，在計算的過程中，部份時間點的**輸出**資訊 $y_i(*)$ 可以(甚至必須)被**忽略**，但這代表 $w_{j, i}$ 需要**同時**達成**兩種**任務就必須要有**兩種不同的數值**：

- **保留過去資訊**：代表 $\abs{w_{j, i}} \neq 0$
- **忽略過去資訊**：代表 $\abs{w_{j, i}} \approx 0$

因此**無法只靠一個** $w_{j, i}$ 決定**輸出**的影響，必須有**額外**能夠**理解當前內容 (context-sensitive)** 的功能模組幫忙**讀取** $y_i(*)$

## LSTM 架構

<a name="paper-fig-1"></a>

圖 1：記憶單元內部架構。
符號對應請見下個小節。
圖片來源：[論文][論文]。

![圖 1](https://i.imgur.com/uhS4AgH.png)

<a name="paper-fig-2"></a>

圖 2：LSTM 全連接架構範例。
線條真的多到讓人看不懂，看我整理過的公式比較好理解。
圖片來源：[論文][論文]。

![圖 2](https://i.imgur.com/UQ5LAu8.png)

為了解決**梯度爆炸 / 消失**問題，作者決定以 Constant Error Carousel 為出發點（見 $\eqref{eq:33}$），提出 **3** 個主要的機制，並將這些機制的合體稱為**記憶單元（Memory Cell）**（見[圖 1](#paper-fig-1)）：

- **乘法輸入閘門（Multiplicative Input Gate）**
  - 用於決定是否**更新**記憶單元的**內部狀態** $s^{\cell{k}}(t + 1)$
  - 細節請見 $\eqref{eq:36} \eqref{eq:38}$
- **乘法輸出閘門（Multiplicative Output Gate）**
  - 用於決定是否**輸出**記憶單元的**輸出訊號** $\hcell{i}{k}{t + 1}$
  - 細節請見 $\eqref{eq:40} \eqref{eq:41}$
- **自連接線性單元（Central Linear Unit with Fixed Self-connection）**
  - 概念來自於 $\eqref{eq:33}$，希望能夠讓 $s^{\cell{k}}(t)$ 與 $s^{\cell{k}}(t + 1)$ 相同，藉此保障**梯度不會消失**
  - 如果 $s^{\cell{k}}(t)$ 與 $s^{\cell{k}}(t + 1)$ 相同，則我們可以確保達成 $\eqref{eq:31}$
  - 細節請見 $\eqref{eq:39}$

### 初始狀態

我們將 $\eqref{eq:1}$ 中的計算重新定義，並新增幾個符號：

|符號|意義|數值範圍|
|-|-|-|
|$\dhid$|**隱藏單元**的維度|$\set{n \in \N : n \geq 0}$|
|$\dcell$|**記憶單元**的**維度**|$\set{n \in \N : n \geq 1}$|
|$\ncell$|**記憶單元**的**個數**|$\set{n \in \N : n \geq 1}$|

- 因為論文 4.3 節有提到可以完全沒有**隱藏單元**，因此隱藏單元的維度可以為 $0$。
- 根據論文 4.4 節，可以**同時**擁有 $\ncell$ 個不同的**記憶單元**，因此 $\ncell$ 可以大於 $1$

接著我們定義 $t$ 時間點的模型計算狀態：

|符號|意義|數值範圍|
|-|-|-|
|$y^{\ophid}(t)$|**隱藏單元（Hidden Units）**|$\R^{\dhid}$|
|$y^{\opig}(t)$|**輸入閘門單元（Input Gate Units）**|$\R^{\ncell}$|
|$y^{\opog}(t)$|**輸出閘門單元（Output Gate Units）**|$\R^{\ncell}$|
|$y^{\cell{k}}(t)$|**記憶單元** $k$ 的**輸出**|$\R^{\dcell}$|
|$s^{\cell{k}}(t)$|**記憶單元** $k$ 的**內部狀態**|$\R^{\dcell}$|
|$y(t)$|**模型總輸出**|$\R^{\dout}$|

- 以上所有向量全部都**初始化**成各自維度的**零向量**，也就是 $t = 0$ 時模型**所有節點**（除了**輸入**）都是 $0$
- 根據論文 4.4 節，可以**同時**擁有 $\ncell$ 個不同的**記憶單元**
  - [圖 2](#paper-fig-2)模型共有 $2$ 個不同的記憶單元
  - **記憶單元**上標 $k$ 的數值範圍為 $k = 1, \dots, \ncell$
- **同一個**記憶單元**共享閘門單元**
  - 包含**輸入閘門**與**輸出閘門**
  - 因此 $y^{\opig}(t), y^{\opog}(t) \in \R^{\ncell}$
- 根據論文 4.3 節，**記憶單元**、**閘門單元**與**隱藏單元**都算是**隱藏層（Hidden Layer）**的一部份
  - **外部輸入**會與**隱藏層**和**總輸出**連接
  - **隱藏層**會與**總輸出**連接（但**閘門**不會）

> **All units** (except for gate units) in all layers have **directed** connections (serve as input) to **all units** in the **layer above** (or to **all higher layers**; see experiments 2a and 2b)

### 輸入閘門單元

當我們得到 $t$ 時間點的外部輸入 $x(t)$ 時，我們使用如同 $\eqref{eq:1} \eqref{eq:2}$ 的方式計算模型 $t + 1$ 時間點的**輸入閘門單元（Input Gate Units）** $y^{\opig}(t + 1)$

$$
\begin{align*}
\netig{k}{t + 1} & = \br{\sum_{j = 1}^{\din} \wig_{k, j} \cdot x_j(t)} + \br{\sum_{j = \din + 1}^{\din + \dhid} \wig_{k, j} \cdot y_j^{\ophid}(t)} \\
& \quad + \br{\sum_{j = \din + \dhid + 1}^{\din + \dhid + \ncell} \wig_{k, j} \cdot y_j^{\opig}(t)} + \br{\sum_{j = \din + \dhid + \ncell + 1}^{\din + \dhid + 2\ncell} \wig_{k, j} \cdot y_j^{\opog}(t)} \\
& \quad + \br{\sum_{k^{\star} = 1}^{\ncell} \sum_{j = \din + \dhid + 2\ncell + (k^{\star} - 1) \cdot \dcell + 1}^{\din + \dhid + 2\ncell + k^{\star} \cdot \dcell} \wig_{k, j} \cdot y_j^{\cell{k^{\star}}}(t)} \\
& = \sum_{j = 1}^{\din + \dhid + \ncell \cdot (2 + \dcell)} \wig_{k, j} \cdot [x ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t); \\
y_k^{\opig}(t + 1) & = \fnetig{k}{t + 1}.
\end{align*} \tag{36}\label{eq:36}
$$

- **所有** $t$ 時間點的**模型節點**（除了**總輸出**）都參與了**輸入閘門單元**的計算
- 因為有 $\ncell$ 個**不同**的**記憶單元**，所以 $\eqref{eq:36}$ 第一個等式中加法的**最後一個**項次必須有兩個 $\sum$
- $\wig$ 為**連接輸入閘門單元**的**參數**
  - 我們可以將所有模型節點**串接**，**一次做完矩陣乘法**（如同$\eqref{eq:36}$ 中的第二個等式）
  - $\wig$ 的輸入維度為 $\din + \dhid + \ncell \cdot (2 + \dcell)$
  - $\wig$ 的輸出維度為 $\ncell$，因此 $k$ 的數值範圍為 $k = 1, \dots, \ncell$
  - $\wig$ 的輸出維度設計成 $\ncell$ 的理由是**同一個記憶單元內部狀態** $s^{\cell{k}}(t + 1)$ 會**共享**輸入閘門單元，因此與**記憶單元**的**個數相同**，細節請見 $\eqref{eq:38}$
- $f_k^{\opig} : \R \to [0, 1]$ 必須要是**可微分函數**，具有**數值範圍限制**
  - 通常選擇 sigmoid 函數
- 之後我們會將 $y_k^{\opig}(t + 1)$ 用來決定是否**更新** $t + 1$ 時間點的**記憶單元內部狀態** $s^{\cell{k}}(t + 1)$，請見 $\eqref{eq:38} \eqref{eq:39}$
- $\eqref{eq:36}$ 沒有使用偏差項（bias term），但後續的分析會提到可以使用偏差項進行計算缺陷的修正

### 乘法輸入閘門

首先我們使用與 $\eqref{eq:36}$ 相同想法，在得到 $t$ 時間點的外部輸入 $x(t)$ 時計算模型 $t + 1$ 時間點第 $k$ 個**記憶單元淨輸入** $\opnet^{\cell{k}}(t + 1)$

$$
\begin{align*}
\netcell{i}{k}{t + 1} & = \br{\sum_{j = 1}^{\din} \wcell{k}_{i, j} \cdot x_j(t)} + \br{\sum_{j = \din + 1}^{\din + \dhid} \wcell{k}_{i, j} \cdot y_j^{\ophid}(t)} \\
& \quad + \br{\sum_{j = \din + \dhid + 1}^{\din + \dhid + \ncell} \wcell{k}_{i, j} \cdot y_j^{\opig}(t)} + \br{\sum_{j = \din + \dhid + \ncell + 1}^{\din + \dhid + 2\ncell} \wcell{k}_{i, j} \cdot y_j^{\opog}(t)} \\
& \quad + \br{\sum_{k^{\star} = 1}^{\ncell} \sum_{j = \din + \dhid + 2\ncell + (k^{\star} - 1) \cdot \dcell + 1}^{\din + \dhid + 2\ncell + k^{\star} \cdot \dcell} \wcell{k}_{i, j} \cdot y_j^{\cell{k^{\star}}}(t)} \\
& = \sum_{j = 1}^{\din + \dhid + \ncell \cdot (2 + \dcell)} \wcell{k}_{i, j} \cdot [x ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)
\end{align*} \tag{37}\label{eq:37}
$$

- 運算架構與 $\eqref{eq:36}$ **完全相同**
  - **所有** $t$ 時間點的**模型節點**（除了**總輸出**）都參與了**記憶單元淨輸入**的計算
  - $\ncell$ 個**不同**的**記憶單元**導致 $\eqref{eq:37}$ 第一個等式中加法的**最後一個**項次必須有兩個 $\sum$
- 共有 $\ncell$ 個**不同**的**參數** $\wcell{k}$
  - 我們可以將所有模型節點**串接**，**一次做完矩陣乘法**（如同$\eqref{eq:37}$ 中的第二個等式）
  - $\wcell{k}$ 的輸入維度為 $\din + \dhid + \ncell \cdot (2 + \dcell)$
  - $\wcell{k}$ 的輸出維度事先定義的 $\dcell$，因此 $i$ 的數值範圍為 $i = 1, \dots, \dcell$
  - 計算總共得出 $\ncell \cdot \dcell$ 個數字

定義**可微分**啟發函數 $g_i^{\cell{k}} : \R \to \R$，我們將 $\eqref{eq:37}$ 轉換成第 $k$ 個**記憶單元** $\cell{k}$ 在 $t + 1$ 時間點可以收到的**輸入訊號** $\gnetcell{i}{k}{t + 1}$。

接著我們將 $\eqref{eq:36}$ 所得**輸入閘門單元** $y_k^{\opig}(t + 1)$ 與 $\gnetcell{i}{k}{t + 1}$ 進行**相乘**

$$
y_k^{\opig}(t + 1) \cdot \gnetcell{i}{k}{t + 1} \tag{38}\label{eq:38}
$$

- $y_k^{\opig}(t + 1)$ 扮演**輸入閘門**的角色
  - 由於**記憶單元淨輸入**與 $\eqref{eq:36}$ 是以**相乘**進行結合，因此被稱為**乘法輸入閘門（Multiplicative Input Gate）**
  - 當模型認為**輸入訊號** $\gnetcell{i}{k}{t + 1}$ **不重要**時，模型應該要**關閉輸入閘門**，即 $y_k^{\opig}(t + 1) \approx 0$
    - 丟棄**當前**輸入訊號
    - 只以**過去資訊**進行決策
  - 當模型認為**輸入訊號** $\gnetcell{i}{k}{t + 1}$ **重要**時，模型應該要**開啟輸入閘門**，即 $y_k^{\opig}(t + 1) \approx 1$
    - 同時考慮**過去**與**當前**資訊
    - 但以**當前**資訊為主
- 不論**輸入訊號** $\gnetcell{i}{k}{t + 1}$ 的大小，只要 $y_k^{\opig}(t + 1) \approx 0$，則輸入訊號**完全無法影響**接下來的所有計算
  - 以此設計避免 $\eqref{eq:34}$ 所遇到的困境
  - 由 $\wcell{k}$ 決定**寫入**的**數值**，函數 $g$ 可以**沒有數值範圍限制**
  - 由 $\wig$ 根據**當前模型計算狀態**控制**寫入**（Context-Sensitive）
- 單一記憶單元 $\cell{k}$ 中的所有維度都**共享**相同的乘法輸入閘門 $y_k^{\opig}(t + 1)$
  - 一旦需要寫入記憶單元 $k$，則必須**同時寫入** $\dcell$ 個數值
- 有時候只需要寫入**部份**記憶單元，不需要同時寫入所有記憶單元，因此 $y^{\opig}(t + 1)$ 的維度為 $\ncell$

### 自連接線性單元

接著我們將 $\eqref{eq:38}$ 的計算結果用來計算 $t + 1$ 時間點的**記憶單元內部狀態** $s^{\cell{k}}(t + 1)$

$$
s_i^{\cell{k}}(t + 1) = s_i^{\cell{k}}(t) + y_k^{\opig}(t + 1) \cdot \gnetcell{i}{k}{t + 1} \tag{39}\label{eq:39}
$$

- 根據 $\eqref{eq:38}$ 我們知道 $y_k^{\opig}(t + 1)$ 能夠**控制輸入訊號的開關**
  - 當輸入訊號**完全關閉**時，$t + 1$ 時間點的**記憶單元內部狀態**與 $t$ 時間點**完全相同**，即 $s^{\cell{k}}(t + 1) = s^{\cell{k}}(t)$，達成 $\eqref{eq:31}$
  - 當輸入訊號開啟時，$t + 1$ 時間點的**記憶單元內部狀態**會被**更新**
- 由於 $t + 1$ 時間點的資訊有加上 $t$ 時間點的資訊，因此稱為**自連接線性單元（Central Linear Unit with Fixed Self-connection）**
  - **加法**是**線性**運算
  - **加上自己**是**自連接**
  - 概念與 $\eqref{eq:33}$ 相同，藉此保障**梯度不會消失**

### 輸出閘門單元

想法同 $\eqref{eq:36}$，當我們得到 $t$ 時間點的外部輸入 $x(t)$ 時便可以計算模型 $t + 1$ 時間點的**輸出閘門單元（Output Gate Units）** $y^{\opog}(t + 1)$

$$
\begin{align*}
\netog{k}{t + 1} & = \bigg[\sum_{j = 1}^{\din} \wog_{k, j} \cdot x_j(t)\bigg] + \bigg[\sum_{j = \din + 1}^{\din + \dhid} \wog_{k, j} \cdot y_j^{\ophid}(t)\bigg] \\
& \quad + \bigg[\sum_{j = \din + \dhid + 1}^{\din + \dhid + \ncell} \wog_{k, j} \cdot y_j^{\opig}(t)\bigg] + \bigg[\sum_{j = \din + \dhid + \ncell + 1}^{\din + \dhid + 2\ncell} \wog_{k, j} \cdot y_j^{\opog}(t)\bigg] \\
& \quad + \bigg[\sum_{k^{\star} = 1}^{\ncell} \sum_{j = \din + \dhid + 2\ncell + (k^{\star} - 1) \cdot \dcell + 1}^{\din + \dhid + 2\ncell + k^{\star} \cdot \dcell} \wog_{k, j} \cdot y_j^{\cell{k^{\star}}}(t)\bigg] \\
& = \sum_{j = 1}^{\din + \dhid + \ncell \cdot (2 + \dcell)} \wog_{k, j} \cdot [x ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t) \\
y_k^{\opog}(t + 1) & = \fnetog{k}{t + 1}
\end{align*} \tag{40}\label{eq:40}
$$

- 運算架構與 $\eqref{eq:36}$ **完全相同**
  - **所有** $t$ 時間點的**模型節點**（除了**總輸出**）都參與了**輸出閘門單元**的計算
  - $\ncell$ 個**不同**的**記憶單元**導致 $\eqref{eq:40}$ 第一個等式中加法的**最後一個**項次必須有兩個 $\sum$
- $\wog$ 為**連接輸出閘門單元**的**參數**
  - 我們可以將所有模型節點**串接**，**一次做完矩陣乘法**（如同$\eqref{eq:40}$ 中的第二個等式）
  - $\wog$ 的輸入維度為 $\din + \dhid + \ncell \cdot (2 + \dcell)$
  - $\wog$ 的輸出維度為 $\ncell$，因此 $k$ 的數值範圍為 $k = 1, \dots, \ncell$
  - $\wog$ 的輸出維度設計成 $\ncell$ 的理由是**同一個記憶單元內部狀態** $s^{\cell{k}}(t + 1)$ 會**共享**輸出閘門單元，因此與**記憶單元**的**個數相同**，細節請見 $\eqref{eq:41}$
- $f_k^{\opog} : \R \to [0, 1]$ 必須要是**可微分函數**，具有**數值範圍限制**
  - 通常選擇 sigmoid 函數
- 之後我們會將 $y_k^{\opog}(t + 1)$ 用來決定是否**輸出** $t + 1$ 時間點的**記憶單元啟發值** $\hcell{i}{k}{t + 1}$

### 乘法輸出閘門

定義**可微分**啟發函數 $h_i^{\cell{k}} : \R \to \R$，我們將 $\eqref{eq:40}$ 轉換成第 $k$ 個**記憶單元內部狀態** $s^{\cell{k}}$ 在 $t + 1$ 時間點的**輸出訊號** $h^{\cell{k}}(s^{\cell{k}}(t + 1))$。

注意不是 $\opnet^{\cell{k}}(t + 1)$ 而是使用 $s^{\cell{k}}(t + 1)$。

接著我們將 $\eqref{eq:40}$ 所得**輸出閘門單元** $y_k^{\opog}(t + 1)$ 與 $h^{\cell{k}}(s^{\cell{k}}(t + 1))$ 進行**相乘**得到記憶單元 $k$ 的**輸出訊號** $y^{\cell{k}}(t + 1)$

$$
y_i^{\cell{k}}(t + 1) = y_k^{\opog}(t + 1) \cdot \hcell{i}{k}{t + 1} \tag{41}\label{eq:41}
$$

- $y_k^{\opog}(t + 1)$ 扮演**輸出閘門**的角色
  - 由於**記憶單元內部狀態**的**輸出訊號**與 $\eqref{eq:40}$ 是以**相乘**進行結合，因此被稱為**乘法輸出閘門（Multiplicative Output Gate）**
  - 當模型認為**輸出訊號** $h^{\cell{k}}(s^{\cell{k}}(t + 1))$ 會導致**當前計算錯誤**時，模型應該**關閉輸出閘門**，即 $y_k^{\opog}(t + 1) \approx 0$
    - 在**輸入**閘門**開啟**的狀況下，**關閉輸出**閘門代表不讓**現在**時間點的資訊影響當前計算
    - 在**輸入**閘門**關閉**的狀況下，**關閉輸出**閘門代表不讓**過去**時間點的資訊影響當前計算
  - 當模型認為**輸出訊號** $h^{\cell{k}}(s^{\cell{k}}(t + 1))$ **包含重要資訊**時，模型應該要開啟**輸出閘門**，即 $y_k^{\opog}(t + 1) \approx 1$
    - 在**輸入**閘門**開啟**的狀況下，**開啟輸出**閘門代表讓**現在**時間點的資訊影響當前計算
    - 在**輸入**閘門**關閉**的狀況下，**開啟輸出**閘門代表不讓**過去**時間點的資訊影響當前計算
- 不論**輸出訊號** $h^{\cell{k}}(s^{\cell{k}}(t + 1))$ 的大小，只要 $y_k^{\opog}(t + 1) \approx 0$，則輸出訊號**完全無法影響**接下來的所有計算
  - 以此設計避免 $\eqref{eq:34} \eqref{eq:35}$ 所遇到的困境
  - 由 $s^{\cell{k}}(t + 1)$ 決定**讀取**的**數值**，函數 $h$ 可以**沒有數值範圍限制**
  - 由 $\wog$ 根據**當前模型計算狀態**控制**輸出**（Context-sensitive）
- 單一記憶單元 $\cell{k}$ 中的所有維度都**共享**相同的乘法輸出閘門 $y_k^{\opog}(t + 1)$
  - 一旦需要讀取記憶單元 $k$，則必須**同時讀取** $\dcell$ 個數值
- 有時候只需要讀取**部份**記憶單元，不需要**同時讀取**所有記憶單元，因此 $y^{\opog}(t + 1)$ 的維度為 $\ncell$

### 總輸出

經過 $\eqref{eq:36} \eqref{eq:37} \eqref{eq:39} \eqref{eq:40} \eqref{eq:41}$ 後我們可以計算 $t + 1$ 時間點的**總輸出** $y(t + 1)$。

但是！！！

$t + 1$ 時間點的**總輸出**只與 $t$ 時間點的**模型狀態**（**不含閘門與總輸出**）有關係，所以 $\eqref{eq:36} \eqref{eq:37} \eqref{eq:39} \eqref{eq:40} \eqref{eq:41}$ 的所有計算都只是在幫助 $t + 2$ 時間點的計算狀態**鋪陳**。

$$
\begin{align*}
\netout{i}{t + 1} & = \bigg[\sum_{j = 1}^{\din} \wout_{i, j} \cdot x_j(t)\bigg] + \bigg[\sum_{j = \din + 1}^{\din + \dhid} \wout_{i, j} \cdot y_j^{\ophid}(t)\bigg] \\
& \quad + \bigg[\sum_{k = 1}^{\ncell} \sum_{j = \din + \dhid + (k - 1) \cdot \dcell + 1}^{\din + \dhid + k \dcell} \wout_{i, j} \cdot y_j^{\cell{k}}(t)\bigg] \\
& = \sum_{j = 1}^{\din + \dhid + \ncell \cdot \dcell} \wout_{i, j} \cdot [x ; y^{\ophid} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t) \\
y_i(t + 1) & = \fnetout{i}{t + 1}
\end{align*} \tag{42}\label{eq:42}
$$

- $\wout$ 為**連接總輸出**的**參數**
  - 我們可以將所有模型節點**串接**，**一次做完矩陣乘法**（如同$\eqref{eq:42}$ 中的第二個等式）
  - $\wout$ 的輸入維度為 $\din + \dhid + \ncell \cdot \dcell$（注意不包含**閘門**）
  - $\wout$ 的輸出維度為 $\dout$，因此 $i$ 的數值範圍為 $i = 1, \dots, \dout$
- $f_i^{\opout} : \R \to \R$ 必須要是**可微分函數**，可以**沒有數值範圍限制**
- 注意 $y_i(t + 1)$ 與 $y_i^{\opog}$ 不同
  - $y_i(t + 1)$ 是**總輸出**，我的 $y_i(t + 1)$ 是論文中的 $y^k(t + 1)$
  - $y_i^{\opog}(t + 1)$ 是**記憶單元**的**輸出閘門**，我的 $y_i^{\opog}(t + 1)$ 是論文中的 $y^{\opout_i}(t + 1)$
- 接著就可以拿 $y_i(t + 1)$ 去做 $\eqref{eq:3} \eqref{eq:4}$ 的誤差計算，取得梯度並進行模型最佳化
- 與 [PyTorch 實作的 LSTM][Pytorch-LSTM] **不同**，$t + 1$ 時間點的**總輸出**並不是拿 $y^{\ophid}(t + 1)$ 與 $y^{\cell{k}}(t + 1)$ 計算 $y(t + 1)$，導致 $y(1)$ 只能**完全**依靠 $x(0)$ 的訊號（細節可見論文 A.1 節）
  - **直接**讓**輸入與輸出相接**看起來等同於**保留** $\eqref{eq:1} \eqref{eq:2}$ 的架構
  - 雖然**總輸出**再也沒有辦法**直接**與**總輸出**相連接，但仍透過 $y^{\ophid}, \cell{k}$ **間接**影響**總輸出**
  - 因此論文後續在**最佳化**的過程中動了手腳，詳細請見 $\eqref{eq:44} \eqref{eq:45} \eqref{eq:46} \eqref{eq:47}$

### 隱藏單元

論文 4.3 節有提到可以完全沒有**隱藏單元**，因此這個段落可以完全不存在。
計算**隱藏單元**想法與 $\eqref{eq:36} \eqref{eq:37} \eqref{eq:40}$ 完全相同，在得到 $t$ 時間點的外部輸入 $x(t)$ 時計算模型 $t + 1$ 時間點的**隱藏單元** $y^{\ophid}(t + 1)$

$$
\begin{align*}
\nethid{i}{t + 1} & = \bigg[\sum_{j = 1}^{\din} \whid_{i, j} \cdot x_j(t)\bigg] + \bigg[\sum_{j = \din + 1}^{\din + \dhid} \whid_{i, j} \cdot y_j^{\ophid}(t)\bigg] \\
& \quad + \bigg[\sum_{j = \din + \dhid + 1}^{\din + \dhid + \ncell} \whid_{i, j} \cdot y_j^{\opig}(t)\bigg] + \bigg[\sum_{j = \din + \dhid + \ncell + 1}^{\din + \dhid + 2\ncell} \whid_{i, j} \cdot y_j^{\opog}(t)\bigg] \\
& \quad + \bigg[\sum_{k = 1}^{\ncell} \sum_{j = \din + \dhid + 2\ncell + (k - 1) \cdot \dcell + 1}^{\din + \dhid + 2\ncell + k \cdot \dcell} \whid_{i, j} \cdot y_j^{\cell{k}}(t)\bigg] \\
& = \sum_{j = 1}^{\din + \dhid + \ncell \cdot (2 + \dcell)} \whid_{i, j} \cdot [x ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t) \\
y_i^{\ophid}(t + 1) & = \fnethid{i}{t + 1}
\end{align*} \tag{43}\label{eq:43}
$$

- **所有** $t$ 時間點的**模型節點**（除了**總輸出**）都參與了**隱藏單元**的計算
- 因為有 $\ncell$ 個**不同**的**記憶單元**，所以 $\eqref{eq:43}$ 第一個等式中加法的**最後一個**項次必須有兩個 $\sum$
- $\whid$ 為**連接隱藏單元**的**參數**
  - 我們可以將所有模型節點**串接**，**一次做完矩陣乘法**（如同$\eqref{eq:43}$ 中的第二個等式）
  - $\whid$ 的輸入維度為 $\din + \dhid + \ncell \cdot (2 + \dcell)$
  - $\whid$ 得輸出維度為 $\dhid$，因此 $i$ 的數值範圍為 $i = 1, \dots, \dhid$
- $f_i^{\ophid} : \R \to \R$ 必須要是**可微分函數**，可以**沒有數值範圍限制**
- 之後我們會將 $y^{\ophid}(t + 1)$ 用於計算**所有** $t + 2$ 時間點的**模型計算狀態**

## 丟棄部份模型單元的梯度

過去的論文中提出以**修改最佳化過程**避免 RNN 訓練遇到**梯度爆炸 / 消失**的問題（例如 Truncated BPTT）。

論文 4.5 節提到**最佳化** LSTM 的方法為 **RTRL 的變種**，主要精神如下

- 最佳化的核心思想是確保能夠達成 **CEC** （見 $\eqref{eq:33}$）
- 使用的手段是要求所有梯度**反向傳播**的過程在經過**記憶單元**與**隱藏單元**後便**不**可以繼續傳播
- 丟棄部份梯度後導致梯度計算可以在完成 $t + 1$ 時間點的 forward pass 便可馬上完成（real time 的精神便是來自於此）

首先我們定義新的符號 $\aptr$，代表計算**梯度**的過程會有**部份梯度**故意被**丟棄**（設定為 $0$），並以丟棄結果**近似**最後**全微分**的概念。

$$
\pd{[\opnet^{\ophid} ; \opnet^{\opig} ; \opnet^{\opog} ; \opnet^{\cell{1}} ; \dots ; \opnet^{\cell{\ncell}}]_i(t + 1)}{[x ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)} \aptr 0 \tag{44}\label{eq:44}
$$

- 所有與**隱藏單元淨輸入** $\nethid{i}{t + 1}$、**輸入閘門淨輸入** $\netig{i}{t + 1}$、**輸出閘門淨輸入** $\netog{i}{t + 1}$、**記憶單元淨輸入** $\netcell{i}{k}{t + 1}$ **直接相連**的 $t$ 時間點的**單元**，一律**丟棄梯度**
  - 注意論文在 A.1.2 節的開頭只提到**輸入閘門**、**輸出閘門**、**記憶單元**要**丟棄梯度**
  - 但論文在 A.9 式描述可以將**隱藏單元**的梯度一起**丟棄**，害我白白推敲公式好幾天

> Here it would be possible to use the full gradient without affecting constant error flow through internal states of memory cells.

- **丟棄梯度**的意思是，即使計算結果的梯度不為 $0$，仍然將梯度**手動設成** $0$
- 直接相連的**單元**包含**外部輸入** $x(t)$、、**隱藏單元** $y^{\ophid}(t)$、**輸入閘門** $y^{\opig}(t)$、**輸出閘門** $y^{\opog}(t)$ 與**記憶單元** $y^{\cell{k}}$（見 $\eqref{eq:36}, \eqref{eq:37}, \eqref{eq:40}$）

根據 $\eqref{eq:44}$ 結合 $\eqref{eq:36} \eqref{eq:40} \eqref{eq:43}$，我們可以進一步推得

$$
\begin{align*}
& \pd{[y^{\ophid} ; y^{\opig} ; y^{\opog}]_i(t + 1)}{[x ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)} \\
& = \pd{[y^{\ophid} ; y^{\opig} ; y^{\opog}]_i(t + 1)}{[\opnet^{\ophid} ; \opnet^{\opig} ; \opnet^{\opog}]_i(t + 1)} \cdot \cancelto{0}{\pd{[\opnet^{\ophid} ; \opnet^{\opig} ; \opnet^{\opog}]_i(t + 1)}{[x ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)}} \\
& \aptr 0
\end{align*} \tag{45}\label{eq:45}
$$

接著以 $\eqref{eq:40} \eqref{eq:45}$ 加上 $\eqref{eq:37} \eqref{eq:39} \eqref{eq:40} \eqref{eq:41}$ 我們可以得到

$$
\begin{align*}
& \pd{y_i^{\cell{k}}(t + 1)}{[x ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)} \\
& = \pd{y_i^{\cell{k}}(t + 1)}{y_k^{\opig}(t + 1)} \cdot \cancelto{0}{\pd{y_k^{\opig}(t + 1)}{[x ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)}} \\
& \quad + \pd{y_i^{\cell{k}}(t + 1)}{\netcell{i}{k}{t + 1}} \cdot \cancelto{0}{\pd{\netcell{i}{k}{t + 1}}{[x ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)}} \\
& \quad + \pd{y_i^{\cell{k}}(t + 1)}{y_k^{\opog}(t + 1)} \cdot \cancelto{0}{\pd{y_k^{\opog}(t + 1)}{[x ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)}} \\
& \aptr 0
\end{align*} \tag{46}\label{eq:46}
$$

由於 $y^{\opig}(t + 1), y^{\opog}(t + 1), \opnet^{\cell{k}}(t + 1)$ 並不是**直接**透過 $w^{\ophid}$ 產生，因此 $w^{\ophid}$ 只能透過參與 $t$ 時間點**以前**的計算**間接**對 $t + 1$ 時間點的計算造成影響（見 $\eqref{eq:43}$），這也代表在 $\eqref{eq:45} \eqref{eq:46}$ 作用的情況下 $w^{\ophid}$ **無法**從 $y^{\opig}(t + 1), y^{\opog}(t + 1), \opnet^{\cell{k}}(t + 1)$ 收到任何的**梯度**：

$$
\begin{align*}
& \pd{[y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_i(t + 1)}{\whid_{p, q}} \\
& = \sum_{j = \din + 1}^{\din + \dhid + \ncell \cdot (2 + \dcell)} \bigg[\cancelto{0}{\pd{[y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_i(t + 1)}{[y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)}} \cdot \\
& \quad \pd{[y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)}{\whid_{p, q}}\bigg] \\
& \aptr 0
\end{align*} \tag{47}\label{eq:47}
$$

## 相對於總輸出所得剩餘梯度

我們將論文的 A.8 式拆解成 $\eqref{eq:49} \eqref{eq:50} \eqref{eq:51} \eqref{eq:52}$。

### 總輸出參數

令 $\delta_{a, b}$ 為 **Kronecker delta**，i.e.，

$$
\delta_{a, b} = \begin{dcases}
1 & \text{if } a = b \\
0 & \text{otherwise}
\end{dcases} \tag{48}\label{eq:48}
$$

由於**總輸出** $y(t + 1)$ 不會像是 $\eqref{eq:1} \eqref{eq:2}$ 的方式**回饋**到模型的計算狀態中，因此**總輸出參數** $\wout$ 對**總輸出** $y(t + 1)$ 計算所得的**梯度**為

$$
\begin{align*}
\pd{y_i(t + 1)}{\wout_{p, q}} & = \pd{y_i(t + 1)}{\netout{i}{t + 1}} \cdot \pd{\netout{i}{t + 1}}{\wout_{p, q}} \\
& = \dfnetout{i}{t + 1} \cdot \delta_{i, p} \cdot [x ; y^{\ophid} ; y^{\cell{1}} ; \dots, y^{\cell{\ncell}}]_q(t)
\end{align*} \tag{49}\label{eq:49}
$$

- $\eqref{eq:49}$ 就是論文中 A.8 式的第一個 case
- 由於 $p$ 可以是**任意**的輸出節點，因此只有 $i = p$ 的時候計算 $\wout_{p, q}$ 對於 $y_i(t + 1)$ 造成的梯度才有意義
  - $i$ 的數值範圍為 $i = 1, \dots, \dout$
  - $p$ 的數值範圍為 $p = 1, \dots, \dout$
- 我們使用 $\delta_{i, p}$ 確保梯度只有在 $i = p$ 才會造成作用
  - 與 $y_i(t + 1)$ 透過 $\wout_{i, q}$ 相連的節點只有**外部輸入** $x_q(t)$、**隱藏單元** $y_q^{\ophid}(t)$ 以及**記憶單元輸出** $y_q^{\cell{k}}(t)$
  - 閘門並不會與總輸出相連，請見 $\eqref{eq:42}$

### 隱藏單元參數

在 $\eqref{eq:44} \eqref{eq:45} \eqref{eq:46} \eqref{eq:47}$ 的作用下，我們可以求得**隱藏單元參數** $\whid$ 在**丟棄**部份梯度後對於**總輸出** $y(t + 1)$ 計算所得的**剩餘梯度**

$$
\begin{align*}
& \pd{y_i(t + 1)}{\whid_{p, q}} \\
& = \pd{y_i(t + 1)}{\netout{i}{t + 1}} \cdot \pd{\netout{i}{t + 1}}{\whid_{p, q}} \\
& = \dfnetout{i}{t + 1} \cdot \\
& \quad \sum_{j = \din + 1}^{\din + \dhid + \ncell \cdot \dcell} \br{\pd{\netout{i}{t + 1}}{[y^{\ophid} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)} \cdot \pd{[y^{\ophid} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)}{\whid_{p, q}}} \\
& = \dfnetout{i}{t + 1} \cdot \\
& \quad \sum_{j = \din + 1}^{\din + \dhid + \ncell \cdot \dcell} \br{\wout_{i, j} \cdot \pd{[y^{\ophid} ; \cancelto{0}{y^{\cell{1}}} ; \dots ; \cancelto{0}{y^{\cell{\ncell}}}]_j(t)}{\whid_{p, q}}} \\
& \aptr \dfnetout{i}{t + 1} \cdot \sum_{j = \din + 1}^{\din + \dhid} \br{\wout_{i, j} \cdot \pd{y_j^{\ophid}(t)}{\whid_{p, q}}} \\
& \aptr \dfnetout{i}{t + 1} \cdot \delta_{j, p} \cdot \wout_{i, j} \cdot \pd{y_j^{\ophid}(t)}{\whid_{j, q}}
\end{align*} \tag{50}\label{eq:50}
$$

- $\eqref{eq:50}$ 的第二個等式中只有**隱藏單元**與**記憶單元輸出**參與微分的理由請見 $\eqref{eq:42}$
  - 沒有列出**外部輸入** $x(t)$ 的理由是 $x(t)$ 並不是**經由** $\whid$ **產生**，因此微分為 $0$
  - 公式中的加法項次從 $\din + 1$ 開始代表**跳過** $x(t)$
- $\eqref{eq:50}$ 的**近似**結果是來自 $\eqref{eq:47}$
  - $\eqref{eq:50}$ 就是論文中 A.8 式的最後一個 case
  - 根據近似， $\whid_{p, q}$ 對於**總輸出** $y_i(t + 1)$ 的影響都是來自 $t$ 時間點的**隱藏單元** $y^{\ophid}(t)$

### 閘門單元參數

同 $\eqref{eq:50}$，我們可以計算**閘門單元參數** $\wig, \wog$ 對**總輸出** $y(t + 1)$ 計算所得的**剩餘梯度**

$$
\begin{align*}
& \pd{y_i(t + 1)}{[\wig ; \wog]_{p, q}} \\
& = \pd{y_i(t + 1)}{\netout{i}{t + 1}} \cdot \pd{\netout{i}{t + 1}}{[\wig ; \wog]_{p, q}} \\
& = \dfnetout{i}{t + 1} \cdot \\
& \quad \sum_{j = \din + 1}^{\din + \dhid + \ncell \cdot \dcell} \bigg[\pd{\netout{i}{t + 1}}{[y^{\ophid} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)} \cdot \pd{[y^{\ophid} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)}{[\wig ; \wog]_{p, q}}\bigg] \\
& = \dfnetout{i}{t + 1} \cdot \sum_{j = \din + 1}^{\din + \dhid + \ncell \cdot \dcell} \bigg[\wout_{i, j} \cdot \pd{[y^{\ophid} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)}{[\wig ; \wog]_{p, q}}\bigg] \\
& = \dfnetout{i}{t + 1} \cdot \Bigg[\sum_{j = \din + 1}^{\din + \dhid} \bigg[\wout_{i, j} \cdot \pd{y_j^{\ophid}(t)}{[\wig ; \wog]_{p, q}}\bigg] + \\
& \quad \sum_{j = \din + \dhid + 1}^{\din + \dhid + \ncell \cdot \dcell} \bigg[\wout_{i, j} \cdot \pd{[y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)}{[\wig ; \wog]_{p, q}}\bigg]\Bigg] \\
& = \dfnetout{i}{t + 1} \cdot \Bigg[ \\
& \quad \sum_{j = \din + 1}^{\din + \dhid} \bigg[\wout_{i, j} \cdot \\
& \quad \quad \sum_{\ell = \din + 1}^{\din + \dhid + \ncell \cdot (2 + \dcell)} \bigg(\cancelto{0}{\pd{y_j^{\ophid}(t)}{[y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_{\ell}(t - 1)}} \cdot \\
& \quad \quad \pd{[y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_{\ell}(t - 1)}{[\wig ; \wog]_{p, q}}\bigg) \\
& \quad \bigg] + \cancelto{\dcell}{\sum_{j = \din + \dhid + 1}^{\din + \dhid + \ncell \cdot \dcell}} \bigg[\wout_{i, j} \cdot \pd{[y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)}{[\wig ; \wog]_{p, q}}\bigg]\Bigg] \\
& \aptr \dfnetout{i}{t + 1} \cdot \sum_{j = 1}^{\dcell} \bigg[\wout_{i, \din + \dhid + (p - 1) \cdot \dcell + j} \cdot \pd{y_j^{\cell{p}}(t)}{[\wig ; \wog]_{p, q}}\bigg]
\end{align*} \tag{51}\label{eq:51}
$$

- $\eqref{eq:51}$ 的第二個等式中只有**隱藏單元**與**記憶單元輸出**參與微分的理由請見 $\eqref{eq:42}$
  - 沒有列出**外部輸入** $x(t)$ 的理由是 $x(t)$ 並不是**經由** $\wig, \wog$ **產生**，因此微分為 $0$
  - 公式中的加法項次從 $\din + 1$ 開始代表**跳過** $x(t)$
- $\eqref{eq:51}$ 的**近似**結果是來自 $\eqref{eq:45}$
  - $\eqref{eq:51}$ 就是論文中 A.8 式的第三個 case
  - 根據近似， $\wig_{p, q}, \wog_{p, q}$ 對於**總輸出** $y_i(t + 1)$ 的影響都是來自 $t$ 時間點的第 $p$ 個**記憶單元輸出** $y^{\cell{p}}(t)$

### 記憶單元淨輸入參數

**記憶單元淨輸入參數** $\wcell{k}$ 對**總輸出** $y(t + 1)$ 計算所得的**剩餘梯度**與 $\eqref{eq:51}$ 幾乎**相同**

$$
\begin{align*}
& \pd{y_i(t + 1)}{\wcell{k}_{p, q}} \\
& = \pd{y_i(t + 1)}{\netout{i}{t + 1}} \cdot \pd{\netout{i}{t + 1}}{\wcell{k}_{p, q}} \\
& = \dfnetout{i}{t + 1} \cdot \\
& \quad \sum_{j = \din + 1}^{\din + \dhid + \ncell \cdot \dcell} \bigg[\pd{\netout{i}{t + 1}}{[y^{\ophid} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)} \cdot \pd{[y^{\ophid} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)}{\wcell{k}_{p, q}}\bigg] \\
& = \dfnetout{i}{t + 1} \cdot \sum_{j = \din + 1}^{\din + \dhid + \ncell \cdot \dcell} \bigg[\wout_{i, j} \cdot \pd{[y^{\ophid} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)}{\wcell{k}_{p, q}}\bigg] \\
& = \dfnetout{i}{t + 1} \cdot \\
& \quad \Bigg[\sum_{j = \din + 1}^{\din + \dhid} \bigg[\wout_{i, j} \cdot \cancelto{0}{\pd{y_j^{\ophid}(t)}{\wcell{k}_{p, q}}}\bigg] + \\
& \quad \cancelto{\din + \dhid + (k - 1) \cdot \dcell + p}{\sum_{j = \din + \dhid + 1}^{\din + \dhid + \ncell \cdot \dcell}} \bigg[\wout_{i, j} \cdot \pd{[y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)}{\wcell{k}_{p, q}}\bigg]\Bigg] \\
& \aptr \dfnetout{i}{t + 1} \cdot \wout_{i, \din + \dhid + (k - 1) \cdot \dcell + p} \cdot \pd{y_p^{\cell{k}}(t)}{\wcell{k}_{p, q}}
\end{align*} \tag{52}\label{eq:52}
$$

- $\eqref{eq:52}$ 的第二個等式中只有**隱藏單元**與**記憶單元輸出**參與微分的理由請見 $\eqref{eq:42}$
- $\eqref{eq:52}$ 的**近似**結果是來自 $\eqref{eq:45} \eqref{eq:46}$
  - $\eqref{eq:52}$ 就是論文中 A.8 式的第二個 case
  - 根據近似， $\wcell{k}_{p, q}$ 對於**總輸出** $y_i(t + 1)$ 的影響都是來自 $t$ 時間點的第 $k$ 個**記憶單元輸出** $y^{\cell{k}}(t)$

## 相對於隱藏單元所得剩餘梯度

我們將論文的 A.9 式拆解成 $\eqref{eq:53} \eqref{eq:54} \eqref{eq:55} \eqref{eq:56}$。

### 總輸出參數

由於**隱藏單元** $y^{\ophid}(t + 1)$ 並不是透過**總輸出參數** $\wout$ 產生，因此 $\wout$ 對於 $y^{\ophid}(t + 1)$ 所得梯度為 $0$

$$
\pd{y_i^{\ophid}(t + 1)}{\wout_{p, q}} = 0 \tag{53}\label{eq:53}
$$

### 隱藏單元參數

根據 $\eqref{eq:44} \eqref{eq:45}$ 我們可以得到**隱藏單元參數** $\whid$ 對於**隱藏單元** $y^{\ophid}(t + 1)$ 計算所得**剩餘梯度**

$$
\begin{align*}
& \pd{y_i^{\ophid}(t + 1)}{\whid_{p, q}} \\
& = \pd{y_i^{\ophid}(t + 1)}{\nethid{i}{t + 1}} \cdot \pd{\nethid{i}{t + 1}}{\whid_{p, q}} \\
& = \delta_{i, p} \cdot \dfnethid{i}{t + 1} \cdot [x ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_q(t) + \\
& \quad \sum_{j = \din + 1}^{\din + \dhid + \ncell \cdot (2 + \dcell)} \bigg[\dfnethid{i}{t + 1} \cdot \\
& \quad \cancelto{0}{\pd{\nethid{i}{t + 1}}{[y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)}} \cdot \\
& \quad \pd{[y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)}{\whid_{p, q}}\bigg] \\
& \aptr \delta_{i, p} \cdot \dfnethid{i}{t + 1} \cdot [x ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_q(t)
\end{align*} \tag{54}\label{eq:54}
$$

### 閘門單元參數

由於**隱藏單元** $y^{\ophid}(t + 1)$ 並不是**直接**透過**閘門參數** $\wig, \wog$ 產生，因此根據 $\eqref{eq:44}$ 我們可以推得 $\wig, \wog$ 對於 $y^{\ophid}(t + 1)$ **剩餘梯度**為 $0$

$$
\begin{align*}
& \pd{y_i^{\ophid}(t + 1)}{[\wig ; \wog]_{p, q}} \\
& = \pd{y_i^{\ophid}(t + 1)}{\nethid{i}{t + 1}} \cdot \sum_{j = \din + 1}^{\din + \dhid + \ncell \cdot (2 + \dcell)} \bigg[ \\
& \quad \cancelto{0}{\pd{\nethid{i}{t + 1}}{[y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)}} \cdot \\
& \quad \pd{[y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)}{[\wig ; \wog]_{p, q}}\bigg] \\
& \aptr 0
\end{align*} \tag{55}\label{eq:55}
$$

### 記憶單元淨輸入參數

同 $\eqref{eq:55}$，由於**隱藏單元** $y^{\ophid}(t + 1)$ 並不是**直接**透過**記憶單元淨輸入參數** $\wcell{k}$ 產生，因此根據 $\eqref{eq:44}$ 我們可以推得 $\wcell{k}$ 對於 $y^{\ophid}(t + 1)$ **剩餘梯度**為 $0$

$$
\begin{align*}
& \pd{y_i^{\ophid}(t + 1)}{\wcell{k}_{p, q}} \\
& = \pd{y_i^{\ophid}(t + 1)}{\nethid{i}{t + 1}} \cdot \sum_{j = \din + 1}^{\din + \dhid + \ncell \cdot (2 + \dcell)} \bigg[ \\
& \quad \cancelto{0}{\pd{\nethid{i}{t + 1}}{[y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)}} \cdot \\
& \quad \pd{[y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)}{\wcell{k}_{p, q}}\bigg] \\
& \aptr 0
\end{align*} \tag{56}\label{eq:56}
$$

## 相對於記憶單元輸出所得剩餘梯度

我們將論文的 A.13 式拆解成 $\eqref{eq:57} \eqref{eq:58} \eqref{eq:59} \eqref{eq:60}$。

### 總輸出參數

由於**記憶單元輸出** $y^{\cell{k}}(t + 1)$ 並不是透過**總輸出參數** $\wout$ 產生，因此 $\wout$ 對於 $y^{\cell{k}}(t + 1)$ 所得梯度為 $0$

$$
\pd{y_i^{\cell{k}}(t + 1)}{\wout_{p, q}} = 0 \tag{57}\label{eq:57}
$$

### 隱藏單元參數

根據 $\eqref{eq:47}$ 我們知道**隱藏單元參數** $\whid$ 對於**記憶單元輸出** $y^{\cell{k}}(t + 1)$ 計算所得**剩餘梯度**為 $0$。

### 閘門單元參數

根據 $\eqref{eq:46}$ 我們可以推得**閘門單元參數** $\wig, \wog$ 對於**記憶單元輸出** $y^{\cell{k}}(t + 1)$ 計算所得**剩餘梯度**

$$
\begin{align*}
& \pd{y_i^{\cell{k}}(t + 1)}{\wog_{p, q}} \\
& = \pd{y_i^{\cell{k}}(t + 1)}{y_k^{\opog}(t + 1)} \cdot \pd{y_k^{\opog}(t + 1)}{\wog_{p, q}} + \pd{y_i^{\cell{k}}(t + 1)}{s_i^{\cell{k}}(t + 1)} \cdot \cancelto{0}{\pd{s_i^{\cell{k}}(t + 1)}{\wog_{p, q}}} \\
& \aptr \delta_{k, p} \cdot \hcell{i}{k}{t + 1} \cdot \pd{y_k^{\opog}(t + 1)}{\wog_{k, q}}
\end{align*} \tag{58}\label{eq:58}
$$

$$
\begin{align*}
& \pd{y_i^{\cell{k}}(t + 1)}{\wig_{p, q}} \\
& = \pd{y_i^{\cell{k}}(t + 1)}{y_k^{\opog}(t + 1)} \cdot \cancelto{0}{\pd{y_k^{\opog}(t + 1)}{\wig_{p, q}}} + \pd{y_i^{\cell{k}}(t + 1)}{s_i^{\cell{k}}(t + 1)} \cdot \pd{s_i^{\cell{k}}(t + 1)}{\wig_{p, q}} \\
& \aptr \delta_{k, p} \cdot y_k^{\opog}(t + 1) \cdot \dhcell{i}{k}{t + 1} \cdot \pd{s_i^{\cell{k}}(t + 1)}{\wig_{k, q}}
\end{align*} \tag{59}\label{eq:59}
$$

### 記憶單元淨輸入參數

同 $\eqref{eq:59}$，使用 $\eqref{eq:46}$ 推得**記憶單元淨輸入參數** $\wcell{k^{\star}}$ 對於**記憶單元輸出** $y^{\cell{k}}(t + 1)$ 計算所得**剩餘梯度**（注意 $k^{\star}$ 可以**不等於** $k$）

$$
\begin{align*}
& \pd{y_i^{\cell{k}}(t + 1)}{\wcell{k^{\star}}_{p, q}} \\
& = \pd{y_i^{\cell{k}}(t + 1)}{y_k^{\opog}(t + 1)} \cdot \cancelto{0}{\pd{y_k^{\opog}(t + 1)}{\wcell{k^{\star}}_{p, q}}} + \pd{y_i^{\cell{k}}(t + 1)}{s_i^{\cell{k}}(t + 1)} \cdot \pd{s_i^{\cell{k}}(t + 1)}{\wcell{k^{\star}}_{p, q}} \\
& \aptr \delta_{k, k^{\star}} \cdot \delta_{i, p} \cdot y_k^{\opog}(t + 1) \cdot \dhcell{i}{k}{t + 1} \cdot \pd{s_i^{\cell{k}}(t + 1)}{\wcell{k}_{i, q}}
\end{align*} \tag{60}\label{eq:60}
$$

**注意錯誤**：論文 A.13 式最後使用**加法** $\delta_{\opin_j l} + \delta_{c_j^v l}$，可能會導致梯度**乘上常數** $2$，因此應該修正成**乘法** $\delta_{\opin_j l} \cdot \delta_{c_j^v l}$

## 相對於閘門單元所得剩餘梯度

我們將論文的 A.10, A.11 式拆解成 $\eqref{eq:61} \eqref{eq:62} \eqref{eq:63} \eqref{eq:64}$。

### 總輸出參數

由於**閘門單元** $y^{\opig}(t + 1), y^{\opog}(t + 1)$ 並不是透過**總輸出參數** $\wout$ 產生，因此 $\wout$ 對於 $y^{\opig}(t + 1), y^{\opog}(t + 1)$ 所得梯度為 $0$

$$
\pd{[y^{\opig} ; y^{\opog}]_k(t + 1)}{\wout_{p, q}} = 0 \tag{61}\label{eq:61}
$$

### 隱藏單元參數

根據 $\eqref{eq:47}$ 我們知道**隱藏單元參數** $\whid$ 對於**閘門單元** $y^{\opig}(t + 1), y^{\opog}(t + 1)$ 計算所得**剩餘梯度**為 $0$。

### 閘門單元參數

根據 $\eqref{eq:44} \eqref{eq:45}$ 我們可以得到**閘門單元參數** $\wig, \wog$ 對於**閘門單元** $y^{\opig}(t + 1), y^{\opog}(t + 1)$ 計算所得**剩餘梯度**

$$
\begin{align*}
& \pd{y_k^{\opig}(t + 1)}{[\wig ; \wog]_{p, q}} \\
& = \pd{y_k^{\opig}(t + 1)}{\netig{k}{t + 1}} \cdot \pd{\netig{k}{t + 1}}{[\wig ; \wog]_{p, q}} \\
& = \delta_{k, p} \cdot \dfnetig{k}{t + 1} \cdot [x ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_q(t) + \\
& \quad \sum_{j = \din + 1}^{\din + \dhid + \ncell \cdot (2 + \dcell)} \bigg[\pd{y_k^{\opig}(t + 1)}{\netig{k}{t + 1}} \cdot \\
& \quad \cancelto{0}{\pd{\netig{k}{t + 1}}{[y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)}} \cdot \\
& \quad \pd{[y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)}{[\wig ; \wog]_{p, q}}\bigg] \\
& \aptr \delta_{k, p} \cdot \dfnetig{k}{t + 1} \cdot [x ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_q(t) \\
& \\
& \pd{y_k^{\opog}(t + 1)}{[\wig ; \wog]_{p, q}} \\
& \aptr \delta_{k, p} \cdot \dfnetog{k}{t + 1} \cdot [x ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_q(t)
\end{align*} \tag{62}\label{eq:62}
$$

### 記憶單元淨輸入參數

由於**閘門單元** $y^{\opig}(t + 1), y^{\opog}(t + 1)$ 並不是**直接**透過**記憶單元淨輸入參數** $\wcell{k}$ 產生，因此根據 $\eqref{eq:44}$ 我們可以推得 $\wcell{k}$ 對於 $y^{\opig}(t + 1), y^{\opog}(t + 1)$ **剩餘梯度**為 $0$

$$
\begin{align*}
\pd{y_k^{\opig}(t + 1)}{\wcell{k}_{p, q}} & = \pd{y_k^{\opig}(t + 1)}{\netig{k}{t + 1}} \cdot \sum_{j = \din + 1}^{\din + \dhid + \ncell \cdot (2 + \dcell)} \bigg[ \\
& \quad \cancelto{0}{\pd{\netig{k}{t + 1}}{[y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)}} \cdot \\
& \quad \pd{[y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)}{\wcell{k}_{p, q}}\bigg] \\
& \aptr 0 \\
& \\
\pd{y_k^{\opog}(t + 1)}{\wcell{k}_{p, q}} & \aptr 0
\end{align*} \tag{63}\label{eq:63}
$$

## 相對於記憶單元內部狀態所得剩餘梯度

我們將論文的 A.12 式拆解成 $\eqref{eq:64} \eqref{eq:65} \eqref{eq:66} \eqref{eq:67} \eqref{eq:68}$。

### 總輸出參數

由於**記憶單元內部狀態** $s^{\cell{k}}(t + 1)$ 並不是透過**總輸出參數** $\wout$ 產生，因此 $\wout$ 對於 $s^{\cell{k}}(t + 1)$ 所得梯度為 $0$

$$
\pd{s_i^{\cell{k}}(t + 1)}{\wout_{p, q}} = 0 \tag{64}\label{eq:64}
$$

### 隱藏單元參數

根據 $\eqref{eq:47}$ **隱藏單元參數** $\whid$ 對於**記憶單元內部狀態** $s^{\cell{k}}(t + 1)$ 計算所得**剩餘梯度**為 $0$

$$
\begin{align*}
& \pd{s_i^{\cell{k}}(t + 1)}{\whid_{p, q}} \\
& = \pd{s_i^{\cell{k}}(t + 1)}{s_i^{\cell{k}}(t)} \cdot \cancelto{0}{\pd{s_i^{\cell{k}}(t)}{\whid_{p, q}}} + \pd{s_i^{\cell{k}}(t + 1)}{y_k^{\opig}(t + 1)} \cdot \cancelto{0}{\pd{y_k^{\opig}(t + 1)}{\whid_{p, q}}} \\
& \quad + \pd{s_i^{\cell{k}}(t + 1)}{\netcell{i}{k}{t + 1}} \cdot \cancelto{0}{\pd{\netcell{i}{k}{t + 1}}{\whid_{p, q}}} \\
& \aptr 0
\end{align*} \tag{65}\label{eq:65}
$$

### 閘門單元參數

將 $\eqref{eq:44}$ 結合 $\eqref{eq:62}$ 我們可以推得**閘門單元參數** $\wig, \wog$ 對於**記憶單元內部狀態** $s^{\cell{k}}(t + 1)$ 計算所得**剩餘梯度**

$$
\begin{align*}
& \pd{s_i^{\cell{k}}(t + 1)}{\wog_{p, q}} \\
& = \pd{s_i^{\cell{k}}(t + 1)}{s_i^{\cell{k}}(t)} \cdot \cancelto{0}{\pd{s_i^{\cell{k}}(t)}{\wog_{p, q}}} + \pd{s_i^{\cell{k}}(t + 1)}{y_k^{\opig}(t + 1)} \cdot \cancelto{0}{\pd{y_k^{\opig}(t + 1)}{\wog_{p, q}}} \\
& \quad + \pd{s_i^{\cell{k}}(t + 1)}{\netcell{i}{k}{t + 1}} \cdot \cancelto{0}{\pd{\netcell{i}{k}{t + 1}}{\wog_{p, q}}} \\
& \aptr 0
\end{align*} \tag{66}\label{eq:66}
$$

$$
\begin{align*}
& \pd{s_i^{\cell{k}}(t + 1)}{\wig_{p, q}} \\
& = \pd{s_i^{\cell{k}}(t + 1)}{s_i^{\cell{k}}(t)} \cdot \pd{s_i^{\cell{k}}(t)}{\wig_{p, q}} + \pd{s_i^{\cell{k}}(t + 1)}{y_k^{\opig}(t + 1)} \cdot \pd{y_k^{\opig}(t + 1)}{\wig_{p, q}} \\
& \quad + \pd{s_i^{\cell{k}}(t + 1)}{\netcell{i}{k}{t + 1}} \cdot \cancelto{0}{\pd{\netcell{i}{k}{t + 1}}{\wig_{p, q}}} \\
& \aptr \delta_{k, p} \cdot 1 \cdot \pd{s_i^{\cell{k}}(t)}{\wig_{k, q}} + \delta_{k, p} \cdot \gnetcell{i}{k}{t + 1} \cdot \pd{y_k^{\opig}(t + 1)}{\wig_{k, q}} \\
& \aptr \delta_{k, p} \cdot \bigg[\pd{s_i^{\cell{k}}(t)}{\wig_{k, q}} + \gnetcell{i}{k}{t + 1} \cdot \dfnetig{k}{t + 1} \cdot \\
& \quad [x ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_q(t)\bigg]
\end{align*} \tag{67}\label{eq:67}
$$

### 記憶單元淨輸入參數

使用 $\eqref{eq:44}$ 推得**記憶單元淨輸入參數** $\wcell{k^{\star}}$ 對於**記憶單元內部狀態** $s^{\cell{k}}(t + 1)$ 計算所得**剩餘梯度**（注意 $k^{\star}$ 可以**不等於** $k$）

$$
\begin{align*}
\pd{s_i^{\cell{k}}(t + 1)}{\wcell{k^{\star}}_{p, q}} & = \pd{s_i^{\cell{k}}(t + 1)}{s_i^{\cell{k}}(t)} \cdot \pd{s_i^{\cell{k}}(t)}{\wcell{k^{\star}}_{p, q}} + \pd{s_i^{\cell{k}}(t + 1)}{y_k^{\opig}(t + 1)} \cdot \cancelto{0}{\pd{y_k^{\opig}(t + 1)}{\wcell{k^{\star}}_{p, q}}} \\
& \quad + \pd{s_i^{\cell{k}}(t + 1)}{\netcell{i}{k}{t + 1}} \cdot \pd{\netcell{i}{k}{t + 1}}{\wcell{k^{\star}}_{p, q}} \\
& \aptr \delta_{k, k^{\star}} \cdot \delta_{i, p} \cdot 1 \cdot \pd{s_i^{\cell{k}}(t)}{\wcell{k}_{i, q}} + \delta_{k, k^{\star}} \cdot \delta_{i, p} \cdot y_k^{\opig}(t + 1) \cdot \\
& \quad \dgnetcell{i}{k}{t + 1} \cdot [x ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_q(t) \\
& = \delta_{k, k^{\star}} \cdot \delta_{i, p} \cdot \bigg[\pd{s_i^{\cell{k}}(t)}{\wcell{k}_{i, q}} + y_k^{\opig}(t + 1) \cdot \dgnetcell{i}{k}{t + 1} \cdot \\
& \quad [x ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_q(t)\bigg]
\end{align*} \tag{68}\label{eq:68}
$$

**注意錯誤**：論文 A.12 式最後使用**加法** $\delta_{\opin_j l} + \delta_{c_j^v l}$，可能會導致梯度**乘上常數** $2$，因此應該修正成**乘法** $\delta_{\opin_j l} \cdot \delta_{c_j^v l}$

## 更新模型參數

### 總輸出參數

從 $\eqref{eq:5} \eqref{eq:6} \eqref{eq:7} \eqref{eq:49} \eqref{eq:53} \eqref{eq:57} \eqref{eq:61} \eqref{eq:64}$ 我們可以觀察出以下結論

$$
\begin{align*}
& \sum_{t = 0}^{T} \pd{\Loss{t + 1}}{\wout_{i, j}} \\
& = \sum_{t = 0}^{T} \bigg[\pd{\Loss{t + 1}}{\loss{i}{t + 1}} \cdot \pd{\loss{i}{t + 1}}{y_i(t + 1)} \cdot \pd{y_i(t + 1)}{\wout_{i, j}}\bigg] \\
& = \sum_{t = 0}^{T} \bigg[1 \cdot \big(y_i(t + 1) - \hat{y}_i(t + 1)\big) \cdot \pd{y_i(t + 1)}{\wout_{i, j}}\bigg] \\
& = \sum_{t = 0}^{T} \bigg[\big(y_i(t + 1) - \hat{y}_i(t + 1)\big) \cdot \dfnetout{i}{t + 1} \cdot \\
& \quad [x ; y^{\ophid} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_j(t)\bigg]
\end{align*} \tag{69}\label{eq:69}
$$

### 隱藏單元參數

從 $\eqref{eq:5} \eqref{eq:6} \eqref{eq:7} \eqref{eq:47} \eqref{eq:50} \eqref{eq:54}$ 我們可以觀察出以下結論

$$
\begin{align*}
& \sum_{t = 0}^{T} \pd{\Loss{t + 1}}{\whid_{p, q}} \\
& = \sum_{t = 0}^{T} \sum_{i = 1}^{\dout} \bigg[\pd{\Loss{t + 1}}{\loss{i}{t + 1}} \cdot \pd{\loss{i}{t + 1}}{y_i(t + 1)} \cdot \pd{y_i(t + 1)}{\whid_{p, q}}\bigg] \\
& \aptr \sum_{t = 0}^{T} \sum_{i = 1}^{\dout} \bigg[1 \cdot \big(y_i(t + 1) - \hat{y}_i(t + 1)\big) \cdot \dfnetout{i}{t + 1} \cdot \wout_{i, p} \cdot \\
& \quad \pd{y_p^{\ophid}(t)}{\whid_{p, q}}\bigg] \\
& \aptr \sum_{t = 0}^{T} \sum_{i = 1}^{\dout} \bigg[\big(y_i(t + 1) - \hat{y}_i(t + 1)\big) \cdot \dfnetout{i}{t + 1} \cdot \wout_{i, p} \cdot \\
& \quad \dfnethid{p}{t} \cdot [x ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_q(t - 1)\bigg] \\
& = \sum_{t = 0}^{T} \Bigg[\bigg(\sum_{i = 1}^{\dout} \big(y_i(t + 1) - \hat{y}_i(t + 1)\big) \cdot \dfnetout{i}{t + 1} \cdot \wout_{i, p}\bigg) \cdot \\
& \quad \dfnethid{p}{t} \cdot [x ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_q(t - 1)\Bigg]
\end{align*} \tag{70}\label{eq:70}
$$

### 輸出閘門單元參數

從 $\eqref{eq:5} \eqref{eq:6} \eqref{eq:7} \eqref{eq:51} \eqref{eq:55} \eqref{eq:58} \eqref{eq:62} \eqref{eq:66}$ 我們可以觀察出以下結論

$$
\begin{align*}
& \sum_{t = 0}^{T} \pd{\Loss{t + 1}}{\wog_{k, q}} \\
& = \sum_{t = 0}^{T} \sum_{i = 1}^{\dout} \bigg[\pd{\Loss{t + 1}}{\loss{i}{t + 1}} \cdot \pd{\loss{i}{t + 1}}{y_i(t + 1)} \cdot \pd{y_i(t + 1)}{\wog_{k, q}}\bigg] \\
& \aptr \sum_{t = 0}^{T} \sum_{i = 1}^{\dout} \Bigg[1 \cdot \big(y_i(t + 1) - \hat{y}_i(t + 1)\big) \cdot \dfnetout{i}{t + 1} \cdot \\
& \quad \sum_{j = 1}^{\dcell} \bigg(\wout_{i, \din + \dhid + (k - 1) \cdot \dcell + j} \cdot \pd{y_j^{\cell{k}}(t)}{\wog_{k, q}}\bigg)\Bigg] \\
& \aptr \sum_{t = 0}^{T} \sum_{i = 1}^{\dout} \Bigg[\big(y_i(t + 1) - \hat{y}_i(t + 1)\big) \cdot \dfnetout{i}{t + 1} \cdot \\
& \quad \sum_{j = 1}^{\dcell} \bigg(\wout_{i, \din + \dhid + (k - 1) \cdot \dcell + j} \cdot \hcell{j}{k}{t} \cdot \pd{y_k^{\opog}(t)}{\wog_{k, q}}\bigg)\Bigg] \\
& = \sum_{t = 0}^{T} \Bigg[\bigg[\sum_{i = 1}^{\dout} \big(y_i(t + 1) - \hat{y}_i(t + 1)\big) \cdot \dfnetout{i}{t + 1} \cdot \\
& \quad \bigg(\sum_{j = 1}^{\dcell} \wout_{i, \din + \dhid + (k - 1) \cdot \dcell + j} \cdot \hcell{j}{k}{t}\bigg)\bigg] \cdot \pd{y_k^{\opog}(t)}{\wog_{k, q}}\Bigg] \\
& \aptr \sum_{t = 0}^{T} \Bigg[\bigg[\sum_{i = 1}^{\dout} \big(y_i(t + 1) - \hat{y}_i(t + 1)\big) \cdot \dfnetout{i}{t + 1} \cdot \\
& \quad \bigg(\sum_{j = 1}^{\dcell} \wout_{i, \din + \dhid + (k - 1) \cdot \dcell + j} \cdot \hcell{j}{k}{t}\bigg)\bigg] \cdot \dfnetog{k}{t} \cdot \\
& \quad [x ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_q(t - 1)\Bigg]
\end{align*} \tag{71}\label{eq:71}
$$

### 輸入閘門單元參數

從 $\eqref{eq:5} \eqref{eq:6} \eqref{eq:7} \eqref{eq:51} \eqref{eq:55} \eqref{eq:59} \eqref{eq:62} \eqref{eq:67}$ 我們可以觀察出以下結論

$$
\begin{align*}
& \sum_{t = 0}^{T} \pd{\Loss{t + 1}}{\wig_{k, q}} \\
& = \sum_{t = 0}^{T} \sum_{i = 1}^{\dout} \bigg[\pd{\Loss{t + 1}}{\loss{i}{t + 1}} \cdot \pd{\loss{i}{t + 1}}{y_i(t + 1)} \cdot \pd{y_i(t + 1)}{\wig_{k, q}}\bigg] \\
& \aptr \sum_{t = 0}^{T} \sum_{i = 1}^{\dout} \Bigg[1 \cdot \big(y_i(t + 1) - \hat{y}_i(t + 1)\big) \cdot \dfnetout{i}{t + 1} \cdot \\
& \quad \sum_{j = 1}^{\dcell} \bigg(\wout_{i, \din + \dhid + (k - 1) \cdot \dcell + j} \cdot \pd{y_j^{\cell{k}}(t)}{\wig_{k, q}}\bigg)\Bigg] \\
& \aptr \sum_{t = 0}^{T} \sum_{i = 1}^{\dout} \Bigg[\big(y_i(t + 1) - \hat{y}_i(t + 1)\big) \cdot \dfnetout{i}{t + 1} \cdot \\
& \quad \sum_{j = 1}^{\dcell} \bigg(\wout_{i, \din + \dhid + (k - 1) \cdot \dcell + j} \cdot y_k^{\opog}(t) \cdot \dhcell{j}{k}{t} \cdot \pd{s_j^{\cell{k}}(t)}{\wig_{k, q}}\bigg)\Bigg] \\
& = \sum_{t = 0}^{T} \Bigg[\Bigg(\sum_{i = 1}^{\dout} \big(y_i(t + 1) - \hat{y}_i(t + 1)\big) \cdot \dfnetout{i}{t + 1} \cdot \\
& \quad \bigg[\sum_{j = 1}^{\dcell} \wout_{i, \din + \dhid + (k - 1) \cdot \dcell + j} \cdot \dhcell{j}{k}{t} \cdot \pd{s_j^{\cell{k}}(t)}{\wig_{k, q}}\bigg]\Bigg) \cdot y_k^{\opog}(t)\Bigg] \\
& \aptr \sum_{t = 0}^{T} \Bigg[\Bigg(\sum_{i = 1}^{\dout} \big(y_i(t + 1) - \hat{y}_i(t + 1)\big) \cdot \dfnetout{i}{t + 1} \cdot \\
& \quad \bigg[\sum_{j = 1}^{\dcell} \wout_{i, \din + \dhid + (k - 1) \cdot \dcell + j} \cdot \dhcell{j}{k}{t} \cdot \bigg(\pd{s_j^{\cell{k}}(t - 1)}{\wig_{k, q}} + \\
& \quad \gnetcell{j}{k}{t} \cdot \dfnetig{k}{t} \cdot \\
& \quad [x ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_q(t - 1)\bigg)\bigg]\Bigg) \cdot y_k^{\opog}(t)\Bigg]
\end{align*} \tag{72}\label{eq:72}
$$

### 記憶單元淨輸入參數

從 $\eqref{eq:5} \eqref{eq:6} \eqref{eq:7} \eqref{eq:52} \eqref{eq:56} \eqref{eq:60} \eqref{eq:63} \eqref{eq:68}$ 我們可以觀察出以下結論

$$
\begin{align*}
& \sum_{t = 0}^{T} \pd{\Loss{t + 1}}{\wcell{k}_{p, q}} \\
& = \sum_{t = 0}^{T} \sum_{i = 1}^{\dout} \bigg[\pd{\Loss{t + 1}}{\loss{i}{t + 1}} \cdot \pd{\loss{i}{t + 1}}{y_i(t + 1)} \cdot \pd{y_i(t + 1)}{\wcell{k}_{p, q}}\bigg] \\
& \aptr \sum_{t = 0}^{T} \sum_{i = 1}^{\dout} \bigg[1 \cdot \big(y_i(t + 1) - \hat{y}_i(t + 1)\big) \cdot \dfnetout{i}{t + 1} \cdot \\
& \quad \wout_{i, \din + \dhid + (k - 1) \cdot \dcell + p} \cdot \pd{y^{\cell{k}}_p(t)}{\wcell{k}_{p, q}}\bigg] \\
& = \sum_{t = 0}^{T} \Bigg[\bigg(\sum_{i = 1}^{\dout} \big(y_i(t + 1) - \hat{y}_i(t + 1)\big) \cdot \dfnetout{i}{t + 1} \cdot \\
& \quad \wout_{i, \din + \dhid + (k - 1) \cdot \dcell + p}\bigg) \cdot \pd{y^{\cell{k}}_p(t)}{\wcell{k}_{p, q}}\Bigg] \\
& \aptr \sum_{t = 0}^{T} \Bigg[\bigg(\sum_{i = 1}^{\dout} \big(y_i(t + 1) - \hat{y}_i(t + 1)\big) \cdot \dfnetout{i}{t + 1} \cdot \\
& \quad \wout_{i, \din + \dhid + (k - 1) \cdot \dcell + p}\bigg) \cdot y_k^{\opog}(t) \cdot \dhcell{p}{k}{t} \cdot \pd{s_p^{\cell{k}}(t)}{\wcell{k}_{p, q}}\Bigg] \\
& \aptr \sum_{t = 0}^{T} \Bigg[\bigg(\sum_{i = 1}^{\dout} \big(y_i(t + 1) - \hat{y}_i(t + 1)\big) \cdot \dfnetout{i}{t + 1} \cdot \\
& \quad \wout_{i, \din + \dhid + (k - 1) \cdot \dcell + p}\bigg) \cdot y_k^{\opog}(t) \cdot \dhcell{p}{k}{t} \cdot \bigg(\pd{s_p^{\cell{k}}(t - 1)}{\wcell{k}_{p, q}} + \\
& \quad y_k^{\opig}(t) \cdot \dgnetcell{p}{k}{t} \cdot [x ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_q(t - 1)\bigg)\Bigg]
\end{align*} \tag{73}\label{eq:73}
$$

## 架構分析

### 時間複雜度

假設 $t + 1$ 時間點的 **forward pass** 已經執行完成，則我們推得**更新** $t + 1$ 時間點**所有參數**的**時間複雜度**

$$
O(\dim(\whid) + \dim(\wog) + \dim(\wig) + \ncell \cdot \dim(\wcell{1}) + \dim(\wout)) \tag{74}\label{eq:74}
$$

- $\eqref{eq:74}$ 就是論文中的 A.27 式
- 在 $t + 1$ 時間點**參數更新**需要考慮 $t - 1$ 時間點的**計算狀態**
  - 注意是 $t - 1$ 不是 $t$
  - 這也代表需要進行兩次以上的 **forward pass** （$t \geq 2$）**部份參數**才能收到梯度
  - **部份參數**指的是除了**總輸出參數**以外的所有參數，細節請見 $\eqref{eq:70} \eqref{eq:71} \eqref{eq:72} \eqref{eq:73}$
- 沒有如同 $\eqref{eq:22}$ 的**連乘積**項，因此不會有**梯度消失**問題
- 整個計算過程需要額外紀錄的**梯度**項次**只有** $\eqref{eq:72} \eqref{eq:73}$ 中的 $\pd{s_j^{\cell{k}}(t - 1)}{\wig_{k, q}}, \pd{s_p^{\cell{k}}(t - 1)}{\wcell{k}_{p, q}}$
  - 紀錄讓 LSTM 可以隨著 **forward pass** 的過程**即時更新**
  - **不需要**等到 $T$ 時間點的計算結束，因此不是採用 **BPTT** 的演算法
  - **即時更新**（意思是 $t + 1$ 時間點的 forward pass 完成後便可計算 $t + 1$ 時間點的誤差梯度）是 **RTRL** 的主要精神

總共會執行 $T + 1$ 個 **forward pass**，因此**更新所有參數**所需的**總時間複雜度**為

$$
O\big(T \cdot \big[\dim(\whid) + \dim(\wog) + \dim(\wig) + \ncell \cdot \dim(\wcell{1}) + \dim(\wout)\big]\big) \tag{75}\label{eq:75}
$$

### 空間複雜度

我們也可以推得在 $t + 1$ 時間點**更新所有參數**所需的**空間複雜度**

$$
O(\dim(\whid) + \dim(\wog) + \dim(\wig) + \ncell \cdot \dim(\wcell{1}) + \dim(\wout)) \tag{76}\label{eq:76}
$$

總共會執行 $T$ 個 **forward pass**，但**更新**所需的**總空間複雜度**仍然同 $\eqref{eq:76}$

- 依照**時間順序**計算梯度，計算完 $t + 1$ 時間點的梯度時 $t - 1$ 的資訊便可丟棄
- 這就是 **RTRL** 的最大優點

### 達成梯度常數

根據 $\eqref{eq:39} \eqref{eq:44} \eqref{eq:45} \eqref{eq:46}$ 我們可以推得

$$
\begin{align*}
\pd{s_i^{\cell{k}}(t + 1)}{s_i^{\cell{k}}(t)} & = \pd{s_i^{\cell{k}}(t)}{s_i^{\cell{k}}(t)} + \cancelto{0}{\pd{y_k^{\opig}(t + 1)}{s_i^{\cell{k}}(t)}} \cdot \gnetcell{i}{k}{t + 1} + \\
& \quad y_k^{\opig}(t + 1) \cdot \cancelto{0}{\pd{\gnetcell{i}{k}{t + 1}}{s_i^{\cell{k}}(t)}} \\
& \aptr 1
\end{align*} \tag{77}\label{eq:77}
$$

由於**丟棄部份梯度**的作用，$s^{\cell{k}}$ 的**梯度**是模型中**唯一**進行**遞迴**（跨過多個時間點的意思）的梯度。
透過丟棄部份梯度我們從 $\eqref{eq:77}$ 可以看出 LSTM 達成 $\eqref{eq:31}$ 所設想的情況。

### 內部狀態偏差行為

觀察 $\eqref{eq:67} \eqref{eq:72}$ 我們可以推得以下結果

$$
\begin{align*}
& \sum_{t = 0}^T \pd{\Loss{t + 1}}{\wig_{p, q}} \\
& \aptr \sum_{t = 0}^{T} \Bigg[\Bigg(\sum_{i = 1}^{\dout} \big(y_i(t + 1) - \hat{y}_i(t + 1)\big) \cdot \dfnetout{i}{t + 1} \cdot \\
& \quad \bigg[\sum_{j = 1}^{\dcell} \wout_{i, \din + \dhid + (k - 1) \cdot \dcell + j} \cdot \dhcell{j}{k}{t} \cdot \bigg(\pd{s_j^{\cell{k}}(t - 1)}{\wig_{k, q}} + \\
& \quad \gnetcell{j}{k}{t} \cdot \dfnetig{k}{t} \cdot \\
& \quad [x ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_q(t - 1)\bigg)\bigg]\Bigg) \cdot y_k^{\opog}(t)\Bigg] \\
& \aptr \sum_{t = 0}^{T} \Bigg[\Bigg(\sum_{i = 1}^{\dout} \big(y_i(t + 1) - \hat{y}_i(t + 1)\big) \cdot \dfnetout{i}{t + 1} \cdot \\
& \quad \bigg[\sum_{j = 1}^{\dcell} \wout_{i, \din + \dhid + (k - 1) \cdot \dcell + j} \cdot \dhcell{j}{k}{t} \cdot \\
& \quad \bigg(\sum_{t^{\star} = 1}^t \gnetcell{j}{k}{t^{\star}} \cdot \dfnetig{k}{t^{\star}} \cdot \\
& \quad [x ; y^{\ophid} ; y^{\opig} ; y^{\opog} ; y^{\cell{1}} ; \dots ; y^{\cell{\ncell}}]_q(t^{\star} - 1)\bigg)\bigg]\Bigg) \cdot y_k^{\opog}(t)\Bigg]
\end{align*} \tag{78}\label{eq:78}
$$

當 $h$ 是 sigmoid 函數時，我們可以發現

- 如果 $s^{\cell{k}}(t)$ 是一個**非常大**的**正數**，則 $\dhcell{j}{k}{t}$ 會變得**非常小**
- 如果 $s^{\cell{k}}(t)$ 是一個**非常小**的**負數**，則 $\dhcell{j}{k}{t}$ 也會變得**非常小**
- 在 $s^{\cell{k}}(t)$ 極正或極負的情況下，**輸入閘門參數** $\wig$ 的**梯度**會**消失**
- 此現象稱為**內部狀態偏差行為**（**Internal State Drift**）
- 同樣的現象也會發生在**記憶單元淨輸入參數** $\wcell{1}, \dots \wcell{\ncell}$ 身上，請見 $\eqref{eq:73}$
- 此分析就是論文的 A.39 式改寫而來

### 解決 Internal State Drift

作者提出可以在 $\opnet^{\opig}$ 加上偏差項，並在**訓練初期**將偏差項弄成很小的**負數**，邏輯如下

$$
\begin{align*}
& b^{\opig} \ll 0 \\
\implies & \opnet^{\opig}(1) \ll 0 \\
\implies & y^{\opig}(1) \approx 0 \\
\implies & s^{\wcell{k}}(1) = s^{\wcell{k}}(0) + y^{\opig}(1) \odot g\big(\opnet^{\wcell{k}}(1)\big) \\
& = y^{\opig}(1) \odot g\big(\opnet^{\wcell{k}}(1)\big) \approx 0 \\
\implies & \begin{dcases}
s^{\wcell{k}}(t + 1) \not\ll 0 \\
s^{\wcell{k}}(t + 1) \not\gg 0
\end{dcases} \quad \forall t = 0, \dots, T
\end{align*} \tag{79}\label{eq:79}
$$

根據 $\eqref{eq:79}$ 我們就不會得到 $s^{\cell{k}}(t)$ 極正或極負的情況，也就不會出現 Internal State Drift。

雖然這種作法是種**模型偏差**（**Model Bias**）而且會導致 $y^{\opig}(\star)$ 與 $\dfnetig{k}{\star}$ **變小**，但作者認為這些影響比起 Internal State Drift 一點都不重要。

### 輸出閘門初始化

論文 4.7 節表示，在訓練的初期模型有可能濫用**記憶單元的初始值**作為計算的常數項（細節請見 $\eqref{eq:41}$），導致模型在訓練的過程中學會完全**不紀錄資訊**。

因此可以將**輸出閘門**加上偏差項，並初始化成**較小的負數**（理由類似於 $\eqref{eq:79}$），讓記憶單元在**計算初期**輸出值為 $0$，迫使模型只在**需要**時指派記憶單元進行**記憶**。

如果有多個記憶單元，則可以給予**不同的負數**，讓模型能夠按照需要**依大小順序**取得記憶單元（**愈大的負數**愈容易被取得）。

### 輸出閘門的優點

在訓練的初期**誤差**通常比較**大**，導致**梯度**跟著變**大**，使得模型在訓練初期的參數劇烈振盪。

由於**輸出閘門**所使用的**啟發函數** $f^{\opog}$ 是 sigmoid，數值範圍是 $(0, 1)$，我們可以發現 $\eqref{eq:72} \eqref{eq:73}$ 的梯度乘積包含 $y^{\opog}$，可以避免**過大誤差**造成的**梯度變大**。

但這些說法並沒有辦法真的保證一定會實現，算是這篇論文說服力比較薄弱的點。

## 實驗

### 實驗設計

- 要測試較長的時間差
  - 資料集不可以出現短時間差
- 任務要夠難
  - 不可以只靠 random weight guessing 解決
  - 需要比較多的參數或是高計算精度 (sparse in weight space)

### 控制變因

- 使用 Online Learning 進行最佳化
  - 意思就是 batch size 為 1
  - 不要被 Online 這個字誤導
- 使用 sigmoid 作為啟發函數
  - 包含 $f^{\opout}, f^{\ophid}, f^{\opig}, f^{\opog}$
- 資料隨機性
  - 資料生成為隨機
  - 訓練順序為隨機
- 在每個時間點 $t$ 的計算順序為
  1. 將外部輸入 $x(t)$ 丟入模型
  2. 計算輸入閘門、輸出閘門、記憶單元、隱藏單元
  3. 計算總輸出
- 訓練初期只使用一個記憶單元，即 $\ncell = 1$
  - 如果訓練中發現最佳化做的不好，開始增加記憶單元，即 $\ncell = \ncell + 1$
  - 一旦記憶單元增加，輸入閘門與輸出閘門也需要跟著增加
  - 這個概念稱為 Sequential Network Construction
- $h^{\cell{k}}$ 與 $g^{\cell{k}}$ 函數如果沒有特別提及，就是使用 $\eqref{eq:80} \eqref{eq:81}$ 的定義

$h^{\cell{k}} : \R \to [-1, 1]$ 函數的定義為

$$
h^{\cell{k}}(x) = \frac{2}{1 + \exp(-x)} - 1 = 2 \sigma(x) - 1 \tag{80}\label{eq:80}
$$

$g^{\cell{k}} : \R \to [-2, 2]$ 函數的定義為

$$
g^{\cell{k}}(x) = \frac{4}{1 + \exp(-x)} - 2 = 4 \sigma(x) - 2 \tag{81}\label{eq:81}
$$

### 實驗 1：Embedded Reber Grammar

<a name="paper-fig-3"></a>

圖 3：Reber Grammar。
一個簡單的有限狀態機，能夠生成的字母包含 BEPSTVX。
圖片來源：[論文][論文]。

![圖 3](https://i.imgur.com/frOl0Tf.png)

<a name="paper-fig-4"></a>

圖 4：Embedded Reber Grammar。
一個簡單的有限狀態機，包含兩個完全相同的 Reber Grammar，開頭跟結尾只能是 BT...TE 與 BP...PE。
圖片來源：[論文][論文]。

![圖 4](https://i.imgur.com/SVfVbJN.png)

#### 任務定義

- Embedded Reber Grammar 是實驗 RNN 短時間差（Short Time Lag）的基準測試資料集
  - [圖 3](#paper-fig-3) 只是 Reber Grammar，真正的生成資料是使用[圖 4](#paper-fig-4) 的 Embedded Reber Grammar
  - Embedded Reber Grammar 時間差最短只有 $9$ 個單位
  - 傳統 RNN 在此資料集上仍然表現不錯
  - 資料生成為隨機，任何一個分支都有 $0.5$ 的機率被生成
- 根據[圖 3](#paper-fig-3) 的架構，生成的第一個字為 B，接著是 T 或 P
  - 因此前兩個字生成 BT 或 BP 的機率各為 $0.5$
  - 能夠生成的字母包含 BEPSTVX
  - 生成直到產生 E 結束，結尾一定是 SE 或 VE
  - 由於有限狀態機中有 Loop，因此 Reber Grammar 有可能產生**任意長度**的文字
- 根據[圖 4](#paper-fig-4) 的架構，生成的開頭為 BT 或 BP
  - 前兩個字生成 BT 或 BP 的機率各為 $0.5$
  - 如果生成 BT，則結尾一定要是 TE
  - 如果生成 BP，則結尾一定要是 PE
  - 因此 RNN 模型必須學會記住**開頭**的 T / P 與**結尾搭配**，判斷一個文字序列是否由 Embedded Reber Grammar 生成
- 模型會在每個時間點 $t$ 收到一個字元，並輸出下一個時間點 $t + 1$ 會收到的字元
  - 輸入與輸出都是 one-hot vector，維度為 $7$，每個維度各自代表 BEPSTVX 中的一個字元，取數值最大的維度作為預測結果
  - 模型必須根據 $0, 1, \dots t - 1, t$ 時間點收到的字元預測 $t + 1$ 時間點輸出的字元
  - 概念就是 Language Model
- 資料數
  - 訓練集：256 筆
  - 測試集：256 筆
  - 總共產生 3 組不同的訓練測試集
  - 每組資料集都跑 $10$ 次實驗，每次實驗模型都隨機初始化
  - 總共執行 $30$ 次實驗取平均
- 評估方法
  - Accuracy

#### LSTM 架構

|參數|數值（或範圍）|備註|
|-|-|-|
|$\din$|$7$||
|$\dhid$|$0$|沒有隱藏單元|
|$(\ncell, \dcell)$|$\set{(3, 2), (4, 1)}$|至少有 $3$ 個記憶單元|
|$\dout$|$7$||
|$\dim(\whid)$|$0$|沒有隱藏單元|
|$\dim(\wcell{k})$|$\dcell \times [\din + \ncell \cdot (2 + \dcell)]$|全連接隱藏層|
|$\dim(\wig)$|$\ncell \times [\din + \ncell \cdot (2 + \dcell) + 1]$|全連接隱藏層，有額外使用偏差項|
|$\dim(\wog)$|$\ncell \times [\din + \ncell \cdot (2 + \dcell) + 1]$|全連接隱藏層，有額外使用偏差項|
|$\dim(\wout)$|$\dout \times [\ncell \cdot \dcell]$|外部輸入沒有直接連接到總輸出|
|參數初始化範圍|$[-0.2, 0.2]$||
|輸出閘門偏差項初始化範圍|$\set{-1, -2, -3, -4}$|由大到小依序初始化不同記憶單元對應輸出閘門偏差項|
|Learning rate|$\set{0.1, 0.2, 0.5}$||
|總參數量|$\set{264, 276}$||

#### 實驗結果

<a name="paper-table-1"></a>

表格 1：Embedded Reber Grammar 實驗結果。
表格來源：[論文][論文]。

![表 1](https://i.imgur.com/51yPwmH.png)

- LSTM + 丟棄梯度 + RTRL 在不同的實驗架構中都能解決任務
  - RNN + RTRL 無法完成
  - Elman Net + ELM 無法完成
- LSTM 收斂速度比其他模型都還要快
- LSTM 使用的參數數量並沒有比其他的模型多太多
- 驗證**輸出閘門**的有效性
  - 當 LSTM 模型記住第二個輸入是 T / P 之後，輸出閘門就會讓後續運算的啟發值接近 $0$，不讓記憶單元內部狀態影響模型學習簡單的 Reber Grammar
  - 如果沒有輸出閘門，則**收斂速度會變慢**

### 實驗 2a：無雜訊長時間差任務

#### 任務定義

定義 $p + 1$ 種不同的字元，標記為 $V = \set{\alpha, \beta, c_1, c_2, \dots, c_{p - 1}}$。

定義 $2$ 種長度為 $p + 1$ 不同的序列 $\opseq_1, \opseq_2$，分別為

$$
\begin{align*}
\opseq_1 & = \alpha, c_1, c_2, \dots, c_{p - 2}, c_{p - 1}, \alpha \\
\opseq_2 & = \beta, c_1, c_2, \dots, c_{p - 2}, c_{p - 1}, \beta
\end{align*}
$$

令 $\opseq_{\star} \in \set{\opseq_1, \opseq_2}$，令 $\opseq_{\star}$ 第 $t$ 個時間點的字元為 $\opseq_{\star}(t) \in V$。

當給予模型 $\opseq_{\star}(t)$ 時，模型要能夠根據 $\opseq_{\star}(0), \opseq_{\star}(1), \dots \opseq_{\star}(t - 1), \opseq_{\star}(t)$ 預測 $\opseq_{\star}(t + 1)$。

- 模型需要記住 $c_1, \dots, c_{p - 1}$ 的順序
- 模型也需要記住開頭的 $\opseq_{\star}(0)$ 是 $\alpha$ 還是 $\beta$，並利用 $\opseq_{\star}(0)$ 的資訊預測 $\opseq_{\star}(p + 1)$
- 根據 $p$ 的大小這個任務可以是**短**時間差或**長**時間差
- 訓練資料
  - 每次以各 $0.5$ 的機率抽出 $\opseq_1, \opseq_2$ 作為輸入
  - 總共執行 $5000000$ 次抽樣與更新
- 測試資料
  - 每次以各 $0.5$ 的機率抽出 $\opseq_1, \opseq_2$ 作為輸入
  - 每次錯誤率在 $0.25$ 以下就是成功，反之失敗
  - 總共執行 $10000$ 次成功與失敗的判斷

#### LSTM 架構

|參數|數值（或範圍）|備註|
|-|-|-|
|$\din$|$p + 1$||
|$\dhid$|$0$|沒有隱藏單元|
|$\dcell$|$\dout$|總輸出就是記憶單元的輸出|
|$\ncell$|$1$|當誤差停止下降時，增加記憶單元|
|$\dout$|$p + 1$||
|$g$|$g(x) = \sigma(x)$|Sigmoid 函數|
|$h$|$h(x) = x$||
|$\dim(\whid)$|$0$|沒有隱藏單元|
|$\dim(\wcell{k})$|$\dcell \times [\din + (1 + \ncell) \cdot \dcell]$|全連接隱藏層|
|$\dim(\wig)$|$\ncell \times [\din + (1 + \ncell) \cdot \dcell]$|全連接隱藏層|
|$\dim(\wog)$|$0$|沒有輸出閘門|
|$\dim(\wout)$|$0$|總輸出就是記憶單元的輸出|
|參數初始化範圍|$[-0.2, 0.2]$||
|Learning rate|$1$||
|最大更新次數|$5000000$||

#### 實驗結果

<a name="paper-table-2"></a>

表格 2：無雜訊長時間差任務實驗結果。
表格來源：[論文][論文]。

![表 2](https://i.imgur.com/638FPkg.png)

- 在 $p = 4$ 時使用 RNN + RTRL 時部份實驗能夠預測序列
  - 序列很短時 RNN 還是有能力完成任務
- 在 $p \geq 10$ 時使用 RNN + RTRL 時直接失敗
- 在 $p = 100$ 時只剩 LSTM 能夠完全完成任務
- LSTM 收斂速度最快

### 實驗 2b：有雜訊長時間差任務

實驗設計和 LSTM 的架構與實驗 2a 完全相同，只是序列 $\opseq_1, \opseq_2$ 中除了頭尾之外的字元可以替換成 $V$ 中任意的文字，總長度維持 $p + 1$。

- 此設計目的是為了確保實驗 2a 中的順序性無法被順利壓縮
- 先創造訓練資料，測試使用與訓練完全相同的資料
- 仍然只有 LSTM 能夠完全完成任務
- LSTM 的誤差仍然很快就收斂
  - 當 $p = 100$ 時只需要 $5680$ 次更新就能完成任務
  - 代表 LSTM 能夠在有雜訊的情況下正常運作

### 實驗 2c：有雜訊超長時間差任務

#### 任務定義

實驗設計和 LSTM 的架構與實驗 2a 概念相同，只是 $V$ 增加了兩個字元 $b, e$，而序列長度可以不同。

生成一個序列的概念如下：

1. 固定一個正整數 $q$，代表序列基本長度
2. 從 $c_1, \dots, c_{p - 1}$ 中隨機抽樣生成長度為 $q$ 的序列 $\opseq$
3. 在序列的開頭補上 $b \alpha$ 或 $b \beta$（機率各為 $0.5$），讓序列長度變成 $q + 2$
4. 接著以 $0.9$ 的機率從 $c_1, \dots, c_{p - 1}$ 中挑一個字補在序列 $\opseq$ 的尾巴，或是以 $0.1$ 的機率補上 $e$
5. 如果生成 $e$ 就再補上 $\alpha$ 或 $\beta$（與開頭第二個字元相同）並結束
6. 如果不是生成 $e$ 則重複步驟 4

假設步驟 $4$ 執行了 $k + 1$ 次，則序列長度為 $2 + q + (k + 1) + 1 = q + k + 4$。
序列的最短長度為 $q + 4$，長度的期望值為

$$
\begin{align*}
& 4 + \sum_{k = 0}^\infty \frac{1}{10} \pa{\frac{9}{10}}^k (q + k) \\
& = 4 + \frac{q}{10} \br{\sum_{k = 0}^\infty \pa{\frac{9}{10}}^k} + \frac{1}{10} \br{\sum_{k = 0}^\infty \pa{\frac{9}{10}}^k \cdot k} \\
& = 4 + \frac{q}{10} \cdot 10 + \frac{1}{10} \cdot 100 \\
& = q + 14
\end{align*}
$$

其中

$$
\begin{align*}
& \br{\sum_{k = 0}^n k x^k} - x \br{\sum_{k = 0}^n k x^k} \\
& = (0x^0 + 1x^1 + 2x^2 + 3x^3 + \dots + nx^n) - \\
& \quad (0x^1 + 1x^2 + 2x^3 + 3x^4 + \dots + nx^{n + 1}) \\
& = 0x^0 + 1x^1 + 1x^2 + 1x^3 + \dots + 1x^n - nx^{n + 1} \\
& = \br{\sum_{k = 0}^n x^k} - nx^{n + 1} \\
& = \frac{1 - x^{n + 1}}{1 - x} - nx^{n + 1} \\
& = \frac{1 - x^{n + 1} - nx^{n + 1} + nx^{n + 2}}{1 - x}
\end{align*}
$$

因此

$$
\begin{align*}
& \br{\sum_{k = 0}^n k x^k} - x \br{\sum_{k = 0}^n k x^k} = \frac{1 - x^{n + 1} - nx^{n + 1} + nx^{n + 2}}{1 - x} \\
\implies & \sum_{k = 0}^n k x^k = \frac{1 - x^{n + 1} - nx^{n + 1} + nx^{n + 2}}{(1 - x)^2} \\
\implies & \sum_{k = 0}^\infty k x^k = \frac{1}{(1 - x)^2} \text{ when } 0 \leq x \lt 1
\end{align*}
$$

利用二項式分佈的期望值公式我們可以推得 $c_i \in V$ 出現次數的期望值

$$
\begin{align*}
& \sum_{k = 0}^\infty \frac{1}{10} \cdot \pa{\frac{9}{10}}^k \cdot \br{\sum_{i = 0}^{q + k} \binom{q + k}{i} \cdot \pa{\frac{1}{p - 1}}^i \cdot \pa{1 - \frac{1}{p - 1}}^{q + k - i}} \\
& = \sum_{k = 0}^\infty \frac{1}{10} \cdot \pa{\frac{9}{10}}^k \cdot \frac{q + k}{p - 1} \\
& = \frac{q}{10(p - 1)} \br{\sum_{k = 0}^\infty \pa{\frac{9}{10}}^k} + \frac{1}{10(p - 1)} \br{\sum_{k = 0}^\infty \pa{\frac{9}{10}}^k \cdot k} \\
& = \frac{q}{p - 1} + \frac{10}{p - 1} \\
& \approx \frac{q}{p - 1} \text{ when } q \gg 0
\end{align*}
$$

訓練誤差只考慮最後一個時間點 $\opseq(2 + q + k + 2)$ 的預測結果，必須要跟第 $\opseq(1)$ 個時間點的輸入相同（概念同實驗 2a）。

測試時會連續執行 $10000$ 次的實驗，預測誤差必須要永遠小於 $0.2$。
會以 $20$ 次的測試結果取平均。

#### LSTM 架構

|參數|數值（或範圍）|備註|
|-|-|-|
|$\din$|$p + 4$||
|$\dhid$|$0$|沒有隱藏單元|
|$\dcell$|$1$||
|$\ncell$|$2$|作者認為其實只要一個記憶單元就夠了|
|$\dout$|$2$|只考慮最後一個時間點的預測誤差，並且預測的可能結果只有 $2$ 種（$\alpha$ 或 $\beta$）|
|$\dim(\whid)$|$0$|沒有隱藏單元|
|$\dim(\wcell{k})$|$\dcell \times [\din + \ncell \cdot (2 + \dcell)]$|全連接隱藏層|
|$\dim(\wig)$|$\ncell \times [\din + \ncell \cdot (2 + \dcell)]$|全連接隱藏層|
|$\dim(\wog)$|$\ncell \times [\din + \ncell \cdot (2 + \dcell)]$|全連接隱藏層|
|$\dim(\wout)$|$\dout \times [\ncell \cdot \dcell]$|外部輸入沒有直接連接到總輸出|
|參數初始化範圍|$[-0.2, 0.2]$||
|Learning rate|$0.01$||

#### 實驗結果

<a name="paper-table-3"></a>

表格 3：有雜訊超長時間差任務實驗結果。
表格來源：[論文][論文]。

![表 3](https://i.imgur.com/j8e0W2U.png)

- 其他方法沒有辦法完成任務，因此不列入表格比較
- 輸入序列長度可到達 $1000$
- 當輸入字元種類與輸入長度一起增加時，訓練時間只會緩慢增加
- 當單一字元的**出現次數期望值增加**時，**學習速度會下降**
  - 作者認為是常見字詞的出現導致參數開始振盪

### 實驗 3a：Two-Sequence Problem

#### 任務定義

給予一個**實數**序列 $\opseq$，該序列可能隸屬於兩種類別 $C_1, C_2$，隸屬機率分別是 $0.5$。

如果 $\opseq \in C_1$，則該序列的前 $N$ 個數字都是 $1.0$，序列的最後一個數字為 $1.0$。
如果 $\opseq \in C_2$，則該序列的前 $N$ 個數字都是 $-1.0$，序列的最後一個數字為 $0.0$。

給定一個常數 $T$，並從 $[T, T + \frac{T}{10}]$ 的區間中隨機挑選一個整數作為序列 $\opseq$ 的長度 $L$。

當 $L \geq N$ 時，任何在 $\opseq(N + 1), \dots \opseq(L - 1)$ 中的數字都是由常態分佈隨機產生，常態分佈的平均為 $0$ 變異數為 $0.2$。

- 此任務由 Bengio 提出
- 作者發現只要用隨機權重猜測（Random Weight Guessing）就能解決，因此在實驗 3c 提出任務的改進版本
- 訓練分成兩個階段
  - ST1：事先隨機抽取的 $256$ 筆測試資料完全分類正確
  - ST2：達成 ST1 後在 $2560$ 筆測試資料上平均錯誤低於 $0.01$
- 實驗結果是執行 $10$ 次實驗的平均值

#### LSTM 架構

|參數|數值（或範圍）|備註|
|-|-|-|
|$\din$|$1$||
|$\dhid$|$0$|沒有隱藏單元|
|$\dcell$|$1$||
|$\ncell$|$3$||
|$\dout$|$1$||
|$\dim(\whid)$|$0$|沒有隱藏單元|
|$\dim(\wcell{k})$|$\dcell \times [\din + \ncell \cdot (2 + \dcell) + 1]$|全連接隱藏層，有額外使用偏差項|
|$\dim(\wig)$|$\ncell \times [\din + \ncell \cdot (2 + \dcell) + 1]$|全連接隱藏層，有額外使用偏差項|
|$\dim(\wog)$|$\ncell \times [\din + \ncell \cdot (2 + \dcell) + 1]$|全連接隱藏層，有額外使用偏差項|
|$\dim(\wout)$|$\dout \times [\ncell \cdot \dcell]$|外部輸入沒有直接連接到總輸出|
|參數初始化範圍|$[-0.1, 0.1]$||
|輸入閘門偏差項初始化範圍|$\set{-1, -3, -5}$|由大到小依序初始化不同記憶單元對應輸入閘門偏差項|
|輸出閘門偏差項初始化範圍|$\set{-2, -4, -6}$|由大到小依序初始化不同記憶單元對應輸出閘門偏差項|
|Learning rate|$1$||

#### 實驗結果

<a name="paper-table-4"></a>

表格 4：Two-Sequence Problem 實驗結果。
表格來源：[論文][論文]。

![表 4](https://i.imgur.com/e1OKDP5.png)

- 偏差項初始化的數值其實不需要這麼準確
- LSTM 能夠快速解決任務
- LSTM 在輸入有雜訊（高斯分佈）時仍然能夠正常表現

### 實驗 3b：Two-Sequence Problem + 雜訊

<a name="paper-table-5"></a>

表格 5：Two-Sequence Problem + 雜訊實驗結果。
表格來源：[論文][論文]。

![表 5](https://i.imgur.com/DEkS8ST.png)

實驗設計與 LSTM 完全與實驗 3a 相同，但對於序列 $\opseq$ 前 $N$ 個實數加上雜訊（與實驗 2a 相同的高斯分佈）。

- 兩階段訓練稍微做點修改
  - ST1：事先隨機抽取的 $256$ 筆測試資料少於 $6$ 筆資料分類錯誤
  - ST2：達成 ST1 後在 $2560$ 筆測試資料上平均錯誤低於 $0.04$
- 結論
  - 增加雜訊導致誤差收斂時間變長
  - 相較於實驗 3a，雖然分類錯誤率上升，但 LSTM 仍然能夠保持較低的分類錯誤率

### 實驗 3c：強化版 Two-Sequence Problem

<a name="paper-table-6"></a>

表格 6：強化版 Two-Sequence Problem 實驗結果。
表格來源：[論文][論文]。

![表 6](https://i.imgur.com/1eXhAr4.png)

實驗設計與 LSTM 完全與實驗 3b 相同，但進行以下修改

- $C_1$ 類別必須輸出 $0.2$，$C_2$ 類別必須輸出 $0.8$
- 高斯分佈變異數改為 $0.1$
- 預測結果與答案絕對誤差大於 $0.1$ 就算分類錯誤
- 任務目標是所有的預測絕對誤差平均值小於 $0.015$
- 兩階段訓練改為一階段
  - 事先隨機抽取的 $256$ 筆測試資料完全分類正確
  - $2560$ 筆測試資料上絕對誤差平均值小於 $0.015$
- Learning rate 改成 $0.1$
- 結論
  - 任務變困難導致收斂時間變更長
  - 相較於實驗 3a，雖然分類錯誤率上升，但 LSTM 仍然能夠保持較低的分類錯誤率

### 實驗 4：加法任務

#### 任務定義

定義一個序列 $\opseq$，序列的每個元素都是由兩個實數組合而成，具體的數值範圍如下

$$
\opseq(t) \in [-1, 1] \times \set{-1, 0, 1} \quad \forall t = 0, \dots, T
$$

每個時間點的元素的第一個數值都是隨機從 $[-1, 1]$ 中取出，第二個數值只能是 $-1, 0, 1$ 三個數值的其中一個。

令 $T$ 為序列的最小長度，則序列 $\opseq$ 的長度 $L$ 將會落在 $[T, T + T / 10]$ 之間。

決定每個時間點的元素的第二個數值的方法如下：

1. 首先將所有元素的第二個數值初始化成 $0$
2. 將 $t = 0$ 與 $t = L$ 的第二個數值初始化成 $-1$
3. 從 $t = 0, \dots, 9$ 隨機挑選一個時間點，並將該時間點的第二個數值加上 $1$
4. 如果前一個步驟剛好挑到 $t = 0$，則 $t = 0$ 的第二個數值將會是 $0$，否則為 $-1$
5. 從 $t = 0, \dots, T / 2 - 1$ 隨機挑選一個時間點，並只挑選第二個數值仍為 $0$ 的時間點，挑選後將該時間點的第二個數值設為 $1$

透過上述步驟 $\opseq$ 最少會包含一個元素其第二個數值為 $1$，最多會包含二個元素其第二個數值為 $1$。

模型在 $L + 1$ 時間點必須輸出所有元素中第二個數值為 $1$ 的元素，其第一個數值的總和，並轉換到 $[0, 1]$ 區間的數值，即

$$
\hat{y}(L + 1) = 0.5 + \frac{1}{4} \sum_{t = 0}^{L} \br{\mathbb{1}(\opseq_1(t) = 1) \cdot \opseq_2(t)}
$$

只考慮 $L + 1$ 時間點的誤差，誤差必須要低於 $0.04$ 才算預測正確。

- 模型必須要學會長時間關閉輸入閘門
- 在實驗中故意對所有參數加上偏差項，實驗**內部狀態偏差行為**造成的影響
- 當連續 $2000$ 次的誤差第於 $0.04$，且平均絕對誤差低於 $0.01$ 時停止訓練
- 測試資料集包含 $2560$ 筆資料

#### LSTM 架構

|參數|數值（或範圍）|備註|
|-|-|-|
|$\din$|$2$||
|$\dhid$|$0$|沒有隱藏單元|
|$\dcell$|$2$||
|$\ncell$|$2$||
|$\dout$|$1$||
|$\dim(\whid)$|$0$|沒有隱藏單元|
|$\dim(\wcell{k})$|$\dcell \times [\din + \ncell \cdot (2 + \dcell) + 1]$|全連接隱藏層，有額外使用偏差項|
|$\dim(\wig)$|$\ncell \times [\din + \ncell \cdot (2 + \dcell) + 1]$|全連接隱藏層，有額外使用偏差項|
|$\dim(\wog)$|$\ncell \times [\din + \ncell \cdot (2 + \dcell) + 1]$|全連接隱藏層，有額外使用偏差項|
|$\dim(\wout)$|$\dout \times [\ncell \cdot \dcell + 1]$|外部輸入沒有直接連接到總輸出，有額外使用偏差項|
|參數初始化範圍|$[-0.1, 0.1]$||
|輸入閘門偏差項初始化範圍|$\set{-3, -6}$|由大到小依序初始化不同記憶單元對應輸入閘門偏差項|
|Learning rate|$0.5$||

#### 實驗結果

<a name="paper-table-7"></a>

表格 7：加法任務實驗結果。
表格來源：[論文][論文]。

![表 7](https://i.imgur.com/pGuMKyt.png)

- LSTM 能夠達成任務目標
  - 不超過 $3$ 筆以上預測錯誤的資料
- LSTM 能夠摹擬加法器，具有作為 distributed representation 的能力
- 能夠儲存時間差至少有 $T / 2$ 以上的資訊，因此不會被**內部狀態偏差行為**影響

### 實驗 5：乘法任務

#### 任務定義

從 LSTM 的架構上來看實驗 4 的加法任務可以透過 $\eqref{eq:39}$ 輕鬆完成，因此實驗 5 的目標是確認模型是否能夠從加法上延伸出乘法的概念，確保實驗 4 並不只是單純因模型架構而解決。

概念與實驗 4 的任務幾乎相同，只做以下修改：

- 每個時間點的元素第一個數值改為 $[0, 1]$ 之間的隨機值
- $L + 1$ 時間點的輸出目標改成

$$
\hat{y}(L + 1) = 0.5 + \frac{1}{4} \prod_{t = 0}^{L} \br{\mathbb{1}(\opseq_1(t) = 1) \cdot \opseq_2(t)}
$$

- 當連續 $2000$ 筆訓練資料中，不超過 $n_{\opseq}$ 筆資料的絕對誤差小於 $0.04$ 就停止訓練
- $n_{\opseq} \in \set{13, 140}$
  - 選擇 $140$ 的理由是模型已經有能力記住資訊，但計算結果不夠精確
  - 選擇 $13$ 的理由是模型能夠精確達成任務

#### LSTM 架構

與實驗 4 完全相同，只做以下修改：

- 輸入閘門偏差項改成隨機初始化
- Learning rate 改為 $0.1$

#### 實驗結果

<a name="paper-table-8"></a>

表格 8：乘法任務實驗結果。
表格來源：[論文][論文]。

![表 8](https://i.imgur.com/bi9jJ3W.png)

- LSTM 能夠達成任務目標
  - 在 $n_{\opseq} = 140$ 時不超過 $170$ 筆以上預測錯誤的資料
  - 在 $n_{\opseq} = 13$ 時不超過 $15$ 筆以上預測錯誤的資料
- 如果額外使用隱藏單元，則收斂速度會更快
- LSTM 能夠摹擬乘法器，具有作為 distributed representation 的能力
- 能夠儲存時間差至少有 $T / 2$ 以上的資訊，因此不會被**內部狀態偏差行為**影響

### 實驗 6a：Temporal Order with 4 Classes

#### 任務定義

給予一個序列 $\opseq$，其長度 $L$ 會落在 $[100, 110]$ 之間，序列中的所有元素都來自於集合 $V = \set{a, b, c, d, B, E, X, Y}$。

序列 $\opseq$ 的開頭必定為 $B$，最後為 $E$，剩餘所有的元素都是 $a, b, c, d$，除了兩個時間點 $t_1, t_2$。

$t_1, t_2$ 時間點只能出現 $X$ 或 $Y$，$t_1$ 時間點會落在 $[10, 20]$，$t_2$ 時間點會落在 $[50, 60]$。

因此根據 $X, Y$ 出現的**次數**與**順序**共有 $4$ 種不同的類別

$$
\begin{align*}
C_1 & = XX \\
C_2 & = XY \\
C_3 & = YX \\
C_4 & = YY
\end{align*}
$$

模型必須要在 $L + 1$ 時間點進行類別預測，誤差只會出現在 $L + 1$ 時間點。

- $t_1, t_2$ 的最少時間差為 $30$
- 模型必須要記住資訊與**出現順序**
- 當模型成功預測連續 $2000$ 筆資料，並且預測平均誤差低於 $0.1$ 時便停止訓練
- 測試資料共有 $2560$ 筆

#### LSTM 架構

|參數|數值（或範圍）|備註|
|-|-|-|
|$\din$|$8$||
|$\dhid$|$0$|沒有隱藏單元|
|$\dcell$|$2$||
|$\ncell$|$2$||
|$\dout$|$4$||
|$\dim(\whid)$|$0$|沒有隱藏單元|
|$\dim(\wcell{k})$|$\dcell \times [\din + \ncell \cdot (2 + \dcell) + 1]$|全連接隱藏層，有額外使用偏差項|
|$\dim(\wig)$|$\ncell \times [\din + \ncell \cdot (2 + \dcell) + 1]$|全連接隱藏層，有額外使用偏差項|
|$\dim(\wog)$|$\ncell \times [\din + \ncell \cdot (2 + \dcell) + 1]$|全連接隱藏層，有額外使用偏差項|
|$\dim(\wout)$|$\dout \times [\ncell \cdot \dcell + 1]$|外部輸入沒有直接連接到總輸出，有額外使用偏差項|
|參數初始化範圍|$[-0.1, 0.1]$||
|輸入閘門偏差項初始化範圍|$\set{-2, -4}$|由大到小依序初始化不同記憶單元對應輸入閘門偏差項|
|Learning rate|$0.5$||

#### 實驗結果

<a name="paper-table-9"></a>

表格 9：Temporal Order with 4 Classes 任務實驗結果。
表格來源：[論文][論文]。

![表 9](https://i.imgur.com/ucyQoeQ.png)

- LSTM 的平均誤差低於 $0.1$
  - 沒有超過 $3$ 筆以上的預測錯誤
- LSTM 可能使用以下的方法進行解答
  - 擁有 $2$ 個記憶單元時，依照順序記住出現的資訊
  - 只有 $1$ 個記憶單元時，LSTM 可以改成記憶狀態的轉移

### 實驗 6b：Temporal Order with 8 Classes

#### 任務定義

與實驗 6a 完全相同，只是多了一個 $t_3$ 時間點可以出現 $X, Y$。

- $t_2$ 時間點改成落在 $[33, 43]$
- $t_3$ 時間點落在 $[66, 76]$
- 類別變成 $8$ 種

#### LSTM 架構

|參數|數值（或範圍）|備註|
|-|-|-|
|$\din$|$8$||
|$\dhid$|$0$|沒有隱藏單元|
|$\dcell$|$2$||
|$\ncell$|$3$||
|$\dout$|$8$||
|$\dim(\whid)$|$0$|沒有隱藏單元|
|$\dim(\wcell{k})$|$\dcell \times [\din + \ncell \cdot (2 + \dcell) + 1]$|全連接隱藏層，有額外使用偏差項|
|$\dim(\wig)$|$\ncell \times [\din + \ncell \cdot (2 + \dcell) + 1]$|全連接隱藏層，有額外使用偏差項|
|$\dim(\wog)$|$\ncell \times [\din + \ncell \cdot (2 + \dcell) + 1]$|全連接隱藏層，有額外使用偏差項|
|$\dim(\wout)$|$\dout \times [\ncell \cdot \dcell + 1]$|外部輸入沒有直接連接到總輸出，有額外使用偏差項|
|參數初始化範圍|$[-0.1, 0.1]$||
|輸入閘門偏差項初始化範圍|$\set{-2, -4, -6}$|由大到小依序初始化不同記憶單元對應輸入閘門偏差項|
|Learning rate|$0.1$||

#### 實驗結果

見[表格 9](#paper-table-9)。

[Pytorch-LSTM]: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html?highlight=lstm#torch.nn.LSTM
[論文]: https://ieeexplore.ieee.org/abstract/document/6795963
[LSTM2000]: https://direct.mit.edu/neco/article-abstract/12/10/2451/6415/Learning-to-Forget-Continual-Prediction-with-LSTM
[note-LSTM2000]: /deep%20learning/model%20architecture/2021/12/13/learning-to-forget-continual-prediction-with-lstm.html
[LSTM2002]: https://www.jmlr.org/papers/v3/gers02a.html
[note-LSTM2002]: /deep%20learning/model%20architecture/2021/12/28/learning-precise-timing-with-lstm-recurrent-networks.html
