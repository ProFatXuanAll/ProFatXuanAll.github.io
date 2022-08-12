---
layout: ML-note
title: "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"
date: 2022-03-08 14:39:00 +0800
categories: [
  Text Modeling,
]
tags: [
  GRU,
  RNN,
  machine translation,
  model architecture,
  neural network,
]
author: [
  Kyunghyun Cho,
  Bart van Merrienboer,
  Caglar Gulcehre,
  Dzmitry Bahdanau,
  Fethi Bougares,
  Holger Schwenk,
  Yoshua Bengio,
]
---

|-|-|
|目標|提出 RNN Encoder-Decoder 架構，並提出 GRU 取代 LSTM|
|作者|Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio|
|隸屬單位|Université de Montréal|
|期刊/會議名稱|EMNLP|
|發表時間|2014|
|論文連結|<https://aclanthology.org/D14-1179/>|


<!--
  Define LaTeX command which will be used through out the writing.

  Each command must be wrapped with $ signs.
  We use "display: none;" to avoid redudant whitespaces.
 -->

<p style="display: none;">

  <!-- Operator encoder. -->
  $\newcommand{\enc}{\operatorname{enc}}$
  <!-- Operator decoder. -->
  $\newcommand{\dec}{\operatorname{dec}}$

</p>

<!-- End LaTeX command define section. -->


## 重點

- 提出 RNN Encoder-Decoder 架構，幫助基於片語（phrase-based）進行統計機器翻譯（statistical machine translation，SMT）的模型表現更好
  - 作者認為 RNN Encoder-Decoder 架構比起 SMT 能夠更容易發現片語表（pharse table）中的語言規則（linguistic regularities）
  - 透過分析發現 RNN Encoder-Decoder 架構在經過訓練後，輸入向量空間具有連續的特性，且連續特性可以反映出語法（syntactic）與語意（semantic）的知識

## 架構

### RNN Encoder-Decoder

<a name="paper-fig-1"></a>

圖 1：RNN Encoder-Decoder 架構。
圖片來源：[論文][論文]。

![圖 1](https://i.imgur.com/QrpUUgQ.png)

#### 目標函數

令資料集為 $D$，令 $i \in \set{1, \dots, \abs{D}}$，資料集中的每一筆資料以 $(x^{(i)}, y^{(i)})$ 表示。

資料集 $D$ 中的每一筆資料都是一個成對的序列，在這篇論文中專指翻譯的配對文字，因此 $x^{(i)}$ 是翻譯輸入文字，$y^{(i)}$ 是翻譯結果，希望能夠透過訓練演算法找到一個模型參數 $\theta^{\star}$ 能夠滿足翻譯機率值最大化，即

$$
\begin{align*}
\theta^{\star} & = \argmax_{\theta} \pa{\prod_{i = 1}^{\abs{D}} P\pa{y^{(i)}|x^{(i)} ; \theta}}^{\frac{1}{\abs{D}}} \\
& = \argmax_{\theta} \pa{\log \pa{\prod_{i = 1}^{\abs{D}} P\pa{y^{(i)}|x^{(i)} ; \theta}}^{\frac{1}{\abs{D}}}} \\
& = \argmax_{\theta} \pa{\frac{1}{\abs{D}} \sum_{i = 1}^{\abs{D}} \log P\pa{y^{(i)}|x^{(i)} ; \theta}} \\
& = \argmin_{\theta} \pa{\frac{-1}{\abs{D}} \sum_{i = 1}^{\abs{D}} \log P\pa{y^{(i)}|x^{(i)} ; \theta}}.
\end{align*} \tag{1}\label{1}
$$

令 $T_x^{(i)}$ 為 $x^{(i)}$ 的文字長度（已斷詞），$T_y^{(i)}$ 為 $y^{(i)}$ 的文字長度（已斷詞），則 $x^{(i)}$ 與 $y^{(i)}$ 可以表示為

$$
\begin{align*}
x^{(i)} & = (x_1^{(i)}, x_2^{(i)}, \dots, x_{T_x^{(i)}}^{(i)}) \\
y^{(i)} & = (y_1^{(i)}, y_2^{(i)}, \dots, y_{T_y^{(i)}}^{(i)})
\end{align*}
$$

而最佳化目標 $\eqref{1}$ 可以改寫成

$$
\begin{align*}
\frac{-1}{\abs{D}} \sum_{i = 1}^{\abs{D}} \log P\pa{y^{(i)}|x^{(i)} ; \theta} & = \frac{-1}{\abs{D}} \sum_{i = 1}^{\abs{D}} \log \pa{\prod_{j = 1}^{T_y^{(i)}} P\pa{y_j^{(i)}|x_1^{(i)}, \dots, x_{T_x^{(i)}}^{(i)} ; \theta}}^{\frac{1}{T_y^{(i)}}} \\
& = \frac{-1}{\abs{D}} \sum_{i = 1}^{\abs{D}} \pa{\frac{1}{T_y^{(i)}} \sum_{j = 1}^{T_y^{(i)}} \log P\pa{y_j^{(i)}|x_1^{(i)}, \dots, x_{T_x^{(i)}}^{(i)} ; \theta}}.
\end{align*} \tag{2}\label{2}
$$

#### 運算方法

為了方便表達，我們以 $(x, y)$ 代表任意的 $(x^{(i)}, y^{(i)})$，並以 $T_x, T_y$ 代表 $T_x^{(i)}, T_y^{(i)}$。

首先定義 RNN Encoder，顧名思義是使用 RNN 的架構進行運算，因此我們定義以下算式

$$
\begin{align*}
h_0 & = \mathbf{0} \\
h_t & = f_{\enc}(h_{t - 1}, x_t) && t \in \set{1, \dots, T_x}
\end{align*} \tag{3}\label{3}
$$

其中 $h_0$ 是 RNN Encoder 的初始狀態，$h_t$ 是在輸入第 $t$ 個文字 $x_t$ 給模型時透過前次遞迴狀態 $h_{t - 1}$ 計算所得的新遞迴狀態。

注意這裡的假設並沒有限制 $x$ 的長度，也就是說 $T_x$ 可以是任意的正整數，任意長度的文字都可以被壓縮成一個向量 $h_{T_x}$ 進行表達，從計算與最佳化的角度來看非常不合理。

假設最後時間點 $T_x$ 計算所得的遞迴狀態 $h_{T_x}$ 可以用來代表 $x$ 的整體資訊，則我們可以定義 RNN Decoder 的計算方法

$$
\begin{align*}
y_0 & = \mathbf{0} \\
s_0 & = \mathbf{0} \\
s_t & = f_{\dec}(s_{t - 1}, y_{t - 1}, h_{T_x}) && t \in \set{1, \dots, T_y} \\
P(y_t = v | y_0, \dots, y_{t - 1}, x_1, \dots, x_{T_x}) & = \softmax(w_v \cdot s_t) \\
& = \frac{\exp(w_v \cdot s_t)}{\sum_{v' \in V} \exp(w_{v'} \cdot s_t)}
\end{align*} \tag{4}\label{4}
$$

其中 $s_0$ 是 RNN Decoder 的初始狀態，$s_t$ 是將 $t - 1$ 時間點預測所的文字 $y_{t - 1}$、前次遞迴狀態 $s_{t - 1}$ 與 RNN Encoder 的最終狀態 $h_{T_x}$ 計算所得的新遞迴狀態，最後會將 $s_t$ 與各個文字的 word embedding $w_v$ 進行內積與 softmax 作為預測 $y_t$ 的機率值。

訓練完畢後模型可以拿來進行以下兩個任務：

- 給予輸入序列 $x$ 後生成序列 $y$
  - 例如產生翻譯結果
  - 例如產生文章摘要
  - 需要配合 decoding algorithm，例如 beam search
- 給予輸入序列 $x$ 與目標序列 $y$，觀察 $y$ 的預測機率值
  - 例如計算翻譯結果的可能性
  - 例如計算摘要結果的可能性
  - 通常是用來評估訓練結果，不需要額外的 decoding algorithm

### GRU

<a name="paper-fig-2"></a>

圖 2：GRU 架構。
圖片來源：[論文][論文]。

![圖 2](https://i.imgur.com/60ejrGT.png)

當時比較常見的 RNN 都是採用 LSTM，而這篇論文提出 LSTM 的簡化版本，稱為 Gated Recurrent Units（GRU），在保留 LSTM 的閘門機制同時簡化計算的複雜度，作者認為 GRU 比 LSTM 容易實作。

新提出的架構想要保留 [LSTM-1997][LSTM1997] 中的 output gate，並重新命名為 reset gate，定義如下

$$
r_t = \sigma\pa{W^r \cdot \mathbf{x}_t + U^r \cdot h_{t - 1}} \tag{5}\label{5}
$$

- $\mathbf{x}_t$ 是文字 $x_t$ 的 word embedding
- $W^r$ 與 $U^r$ 是可訓練參數，後續我們會探討參數的維度
- 以 $r_t$ 代表時間點 $t$ 計算所得的 reset gate
- $\sigma$ 是 sigmoid 函數，這邊是 elementwise 的版本，意即向量中的每個維度各自進行 sigmoid
- 論文中沒有在 $\eqref{5}$ 中加入偏差向（bias），但實作上都會使用 bias

接著利用 $\eqref{5}$ 計算所得的 reset gate 計算 GRU 的**淨輸入**

$$
\tilde{h}_t = \phi\pa{W \cdot \mathbf{x}_t + U (r_t \odot h_{t - 1})} \tag{6}\label{6}
$$

- $\phi$ 是一個 elementwise activation function，通常設 $\phi = \tanh$
- $W$ 與 $U$ 是可訓練參數，後續我們會探討參數的維度
- Reset gate 與前次遞迴狀態進行 elementwise product，在計算上賦與的意義為**透過 reset gate 決定是否讓過去的資訊參與當前的計算**
  - 當過去的資訊不重要時，應該要讓 reset gate 關閉，即 $r_t \approx \mathbf{0}$
  - Reset gate 是針對過去遞迴資訊進行重設，這也是為什麼我認為 reset gate 在模擬 [LSTM-1997][LSTM1997] 中的 output gate 的原因（注意不是 [LSTM-2000][LSTM2000] 中的 forget gate）
- 論文中沒有在 $\eqref{6}$ 中加入偏差向（bias），但實作上都會使用 bias

得到 $\eqref{6}$ 中的淨輸入後，作者認為需要有一個**控制輸入內容的閘門機制**，因此定義以下符號

$$
\begin{align*}
z_t & = \sigma\pa{W^z \cdot \mathbf{x} + U^z \cdot h_{t - 1}} \\
h_t & = z_t \odot h_{t - 1} + (\mathbf{1} - z_t) \odot \tilde{h}_t
\end{align*} \tag{7}\label{7}
$$

- $\mathbf{1}$ 是一個向量，向量中的每個維度都是 $1$
- $W^z$ 與 $U^z$ 是可訓練參數，後續我們會探討參數的維度
- $z_t$ 稱為 update gate，與 [LSTM-1997][LSTM1997] 中的 input gate 概念不同，並不是決定是否讓 $\tilde{h}_t$ 參與遞迴過程，而是**決定參與的比例**
  - 當過去的資訊不重要時，應該要讓 update gate 關閉，即 $z_t \approx \mathbf{0}$ 且 $h_t \approx \tilde{h}_t$
  - 同時讓 update gate 扮演 [LSTM-1997][LSTM1997] 中的 input gate 與 [LSTM-2000][LSTM2000] 中的 forget gate
  - 在 [Highway Network][HighwayNetwork] 中也有使用類似的架構
- 觀察 $\eqref{7}$ 的第二個式子可以發現 GRU 中的 $h_t$ 同時扮演了 [LSTM-1997][LSTM1997] 中的 memory cell output 與 memory cell internal state
  - 計算上 GRU 架構比 LSTM 架構精減許多
  - 將 GRU 類比於 memory cell 時，可以發現每個 memory cell 的維度都為 $1$，概念比較接近 [LSTMP][LSTMP]
  - 從後續的參數推論也可以發現 GRU 架構的參數數量比 LSTM 架構少

令 $d_x$ 為 $\mathbf{x}_t$ 的維度，$d_h$ 為 $h_t$ 的維度，根據 $\eqref{5} \eqref{6} \eqref{7}$ 我們可以推得

|參數或節點|維度|理由|
|-|-|-|
|$z_t$|$\R^{d_h}$|$\odot$ is elementwise product and $z_t \odot h_{t - 1}$ is well-defined|
|$W^z$|$\R^{d_h \times d_x}$|$\sigma$ is elementwise activation, $z_t \in \R^{d_h}$ and $\mathbf{x}_t \in \R^{d_x}$|
|$U^z$|$\R^{d_h \times d_h}$|$\sigma$ is elementwise activation, $z_t \in \R^{d_h}$ and $h_{t - 1} \in \R^{d_h}$|
|$\tilde{h}_t$|$\R^{d_h}$|$\odot$ is elementwise product and $(\mathbf{1} - z_t) \odot \tilde{h}_t$ is well-defined|
|$r_t$|$\R^{d_h}$|$\odot$ is elementwise product and $r_t \odot h_{t - 1}$ is well-defined|
|$W^r$|$\R^{d_h \times d_x}$|$\sigma$ is elementwise activation, $r_t \in \R^{d_h}$ and $\mathbf{x}_t \in \R^{d_x}$|
|$U^r$|$\R^{d_h \times d_h}$|$\sigma$ is elementwise activation, $r_t \in \R^{d_h}$ and $h_{t - 1} \in \R^{d_h}$|
|$W$|$\R^{d_h \times d_x}$|$\phi$ is elementwise activation, $\tilde{h}_t \in \R^{d_h}$ and $\mathbf{x}_t \in \mathbf{R}^{d_x}$|
|$U$|$\R^{d_h \times d_h}$|$\phi$ is elementwise activation, $\tilde{h}_t \in \R^{d_h}$ and $r_t \in \mathbf{R}^{d_h}$|

作者說根據實驗當 $\phi = \tanh$ 時如果不採用任何的 gate 就無法得到任何有意義的計算結果。

[HighwayNetwork]: https://arxiv.org/abs/1505.00387
[LSTM1997]: https://ieeexplore.ieee.org/abstract/document/6795963
[LSTM2000]: https://direct.mit.edu/neco/article-abstract/12/10/2451/6415/Learning-to-Forget-Continual-Prediction-with-LSTM
[LSTM2002]: https://www.jmlr.org/papers/v3/gers02a.html
[LSTMP]: https://research.google/pubs/pub43905/
[論文]: https://aclanthology.org/D14-1179/

