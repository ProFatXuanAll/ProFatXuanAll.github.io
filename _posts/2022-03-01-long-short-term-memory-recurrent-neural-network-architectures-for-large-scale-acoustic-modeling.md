---
layout: ML-note
title: "Long Short-Term Memory Recurrent Neural Network Architectures for Large Scale Acoustic Modeling"
date: 2022-03-01 19:42:00 +0800
categories: [
  Model Architecture,
  Neural Network,
]
tags: [
  LSTM,
  LSTMP,
]
author: [
  Hasim Sak,
  Andrew W. Senior,
  Françoise Beaufays,
]
---

|-|-|
|目標|嘗試分散式平型化訓練 LSTM 進行字典範圍較大的語音辨識|
|作者|Hasim Sak, Andrew W. Senior, Françoise Beaufays|
|隸屬單位|Google|
|期刊/會議名稱|Interspeech|
|發表時間|2014|
|論文連結|<https://research.google/pubs/pub43905/>|

## 重點

- 這篇論文是 Google [前一篇][pub43895]論文的續作，補了更多實驗後終於投稿上 Interspeech
  - 所有實驗採用的架構都與[前一篇][pub43895]論文相同
  - 在這篇論文中幫提出的架構取名為 LSTMP（Long Short-Term Memory Projected）
  - 不再使用額外的 non-recurrent projection layer，因此[前一篇][pub43895]論文中的 $n_p = 0$
- 第一篇論文嘗試以大量叢集節點 + asynchronous stochastic gradient descent（ASGD）訓練 LSTM 進行語音辨識
  - 人家有錢
  - 兩層 LSTM 可以達到語音辨識的 SOTA
  - 比 RNN + feed-forward 架構表現還好
  - 比單純使用 feed-forward 架構的參數數量少快 10 倍
  - 比 [LSTM-2002][LSTM2002] 架構表現更好，雖然兩者在層數增加時表現接近，但 [LSTM-2002][LSTM2002] 更難訓練且訓練時間更長

## 架構

<a name="paper-fig-1"></a>

圖 1：LSTMP 架構。
圖片來源：[論文][論文]。

![圖 1](https://i.imgur.com/Thd51gv.png)

<a name="paper-fig-2"></a>

圖 2：多層 LSTM 與 LSTMP 架構。
圖片來源：[論文][論文]。

![圖 2](https://i.imgur.com/eeTLPV3.png)

參數定義與運算架構與[前一篇][pub43895]論文完全相同

- $n_i$：輸入單元個數
- $n_o$：輸出單元個數
- $n_c$：記憶單元區塊個數
- $n_r$：記憶單元輸出降維後的維度

|符號|意義|維度|備註|
|-|-|-|-|
|$T$|輸入序列的總長度||$T \in \N$|
|$t$|輸入序列的時間點||$t = 1, \dots, T$|
|$x_t$|第 $t$ 個時間點的**輸入**|$n_i$|$x = (x_1, \dots, x_T)$|
|$y_t$|第 $t$ 個時間點的**輸出**|$n_o$|$y = (y_1, \dots, y_T)$|
|$f_t$|第 $t$ 個時間點的**遺忘閘門**|$n_c$|$f_0 = 0$|
|$i_t$|第 $t$ 個時間點的**輸入閘門**|$n_c$|$i_0 = 0$|
|$o_t$|第 $t$ 個時間點的**輸出閘門**|$n_c$|$o_0 = 0$|
|$c_t$|第 $t$ 個時間點**記憶單元內部狀態**|$n_c$|$c_0 = 0$|
|$m_t$|第 $t$ 個時間點**記憶單元輸出**|$n_c$||
|$r_t$|第 $t$ 個時間點**記憶單元輸出**經過降維後的結果|$n_r$|$r_0 = 0$|
|$W_{g x}$|連接外部輸入與閘門 $g$ 的參數|$n_c \times n_i$|全連接，$g \in \set{i, f, o}$|
|$W_{g r}$|連接記憶單元輸出降維結果與閘門 $g$ 的參數|$n_c \times n_r$|全連接，$g \in \set{i, f, o}$|
|$W_{g c}$|連接記憶單元內部狀態與閘門 $g$ 的參數|$n_c$|peephole connections，$g \in \set{i, f, o}$|
|$b_g$|閘門 $g$ 的偏差項|$n_c$|$g \in \set{i, f, o}$|
|$W_{c x}$|連接外部輸入與記憶單元輸入的參數|$n_c \times n_i$|全連接|
|$W_{c r}$|連接記憶單元輸出降維結果與記憶單元輸入的參數|$n_c \times n_r$|全連接|
|$b_c$|記憶單元輸入的偏差項|$n_c$||
|$W_{y r}$|連接記憶單元輸出降維結果與總輸出的參數|$n_o \times n_r$|全連接|
|$b_y$|總輸出的偏差項|$n_o$||
|$\sigma$|sigmoid 函數|$\sigma(x) = \frac{1}{1 + e^{-x}}$||

計算公式定義如下

$$
\begin{align*}
i_t & = \sigma(W_{i x} \cdot x_t + W_{i r} \cdot r_{t - 1} + W_{i c} \odot c_{t - 1} + b_i) \\
f_t & = \sigma(W_{f x} \cdot x_t + W_{f r} \cdot r_{t - 1} + W_{f c} \odot c_{t - 1} + b_f) \\
c_t & = f_t \odot c_{t - 1} + i_t \odot \tanh(W_{c x} \cdot x_t + W_{c m} \cdot r_{t - 1} + b_c) \\
o_t & = \sigma(W_{o x} \cdot x_t + W_{o r} \cdot r_{t - 1} + W_{o c} \odot c_t + b_o) \\
m_t & = o_t \odot \tanh(c_t) \\
r_t & = W_{r m} \cdot m_t \\
y_t & = \operatorname{softmax}(W_{y r} r_t + b_y)
\end{align*} \tag{1}\label{1}
$$

## 最佳化

- 使用 CPU 叢集進行訓練
  - 共有 $500$ 個計算節點
  - 每個節點使用 $3$ 個 threads
  - 每個 thread 計算 $4$ 個訊號序列（batch size per thread = $4$）
- 使用 [Eigen][Eigen] 函式庫進行矩陣計算
  - 版本為 `v3`
  - 支援 C++
  - 支援 SIMD 平行化指令
- 採用 truncated BPTT 進行最佳化，truncated window size 為 $20$
- 使用 cross entropy loss 作為最佳化目標
- 使用非同步梯度下降（Asynchronous Stochastic Gradient Descent，ASGD）演算法進行最佳化
  - 擁有一個中央伺服器負責儲存參數
  - 單一計算節點完成 $3 \times 4 \times 20$ 的梯度計算後將梯度傳給中央伺服器
  - 中央伺服器收到梯度後進行更新，並回傳更新後的參數給計算節點
  - 由於 batch size 理論上變大了，作者將 learning rate 設定成較小的數值（這句話似乎完全是經驗談）

## 實驗設計

- 實驗資料集為 Google English Voice Search task，非公開資料集
- 實驗的比較對象為 DNN 與 RNN
  - 所有模型都訓練在 $3$ 百萬筆的語音資料集上，長度共 $1900$ 小時
  - 所有資料都有去識別化
- 資料前處理
  - 每筆資料共有 $25$ 毫秒
  - 每筆資料一幀為 $10$ 毫秒
  - 每幀都使用 log-filterbank 將頻率進行特徵提取（phonemes），取 40 個維度當作特徵
  - 額外訓練了一個共有 $90$ M 參數的 feed-forward neural network（FFNN）進行輸入特徵與狀態對齊（states alignment），總共定義了 $14247$ 個前後文相依狀態（context dependent states，CD）
- 標記資料為每個 $40$ 維的 feature 對應到的 phoneme state
  - 模型每個時間點的輸入至少為 $40$ 維（代表 $n_i = 40$）
  - 模型每個時間點的輸出為對應到的狀態（代表 $n_o = 14247$）
- 所有參數初始化範圍為 $(-0.02, 0.02)$
- 每個實驗設置都採用各自最適合的 learning rate（hyperparameter tuning），並對 learning rate 使用 expenentially decay
  - Learning rate 範圍大約落在 $[5 \times 10^{-6}, 1 \times 10^{-5}]$
- 額外限制 LSTM 中 $c_t$ 的數值範圍，落在 $[-50, 50]$
  - 概念如同 gradient clipping
- 評估方法
  - 驗證資料（development set）有 $200000$ 幀，針對每一幀中所有的 state 進行準確率（accuracy）的計算，稱為 frame accuracy
  - 測試資料（test set）有 $22500$ 幀，計算文字辨識錯誤率（word error rates）
  - 所有實驗共用相同的 $5$-gram language model
    - 這裡的假設為：當模型能夠將輸入特徵與狀態對齊成功時，後續的 language model 就會自然產出正確的辨識文字結果
    - 共有兩種不同的字典大小，分別為 $23$ M 與 $1$ B
    - Language model decoding 採用 beam search，beam width 設成較大的數字
- 由於未來時間的資訊有助於提升預測的準確度，因此模型預測會在延遲 $5$ 幀後開始輸出
  - ex: 第 $0$ 幀到第 $4$ 幀輸入完後，當第 $5$ 幀輸入時預測第 $0$ 幀的 $40$ 維特徵所對應到的狀態
  - 前 $5$ 幀不計算誤差，最後 $5$ 幀重複輸入讓模型可以預測 $T - 4$ 到 $T$ 的狀態

## 實驗結果

<a name="paper-fig-3"></a>

圖 3：LSTM 與 LSTMP 的表現對照。
圖片來源：[論文][論文]。

![圖 3](https://i.imgur.com/NlKdg0R.png)

<a name="paper-fig-4"></a>

圖 4：LSTM 與 LSTMP 的收斂速度對照。
圖片來源：[論文][論文]。

![圖 4](https://i.imgur.com/hB3iGDJ.png)

<a name="paper-fig-5"></a>

圖 5：LSTMP 不同參數組合實驗結果。
圖片來源：[論文][論文]。

![圖 5](https://i.imgur.com/cIPrLTD.png)

- 對 [LSTM-2002][LSTM2002] 架構進行分析（見[圖 3](#paper-fig-3) 上半部）
  - 在只使用 $1$ 層時表現不好
  - 改用 $2$ 層時表現有進步但仍然不夠好
  - 採用 $5$ 層時表現最佳
  - 採用 $7$ 層時很難收斂（作者 train 了一天以上才看到收斂），而且表現沒有比較好
- 對 LSTMP 進行分析（見[圖 3](#paper-fig-3) 下半部）
  - 只使用 $1$ 層且使用大量的 memory cell blocks（$n_c$ 較大）時容易導致 overfitting
  - 單純的將層數增加似乎就減少 overfitting 的現象
  - 多層 LSTMP 表現只比多層 LSTM 好一點點，與[前一篇][pub43895]論文的實驗結果差異蠻大的（理由是前一篇論文都只用一層進行實驗）
- 對收斂狀況進行分析（見[圖 4](#paper-fig-4)）
  - LSTMP 收斂速度比 [LSTM-2002][LSTM2002] 還要快
  - 層數愈多表現愈好，但愈難收斂
- 對 LSTMP 的參數數量進行分析（見[圖 5](#paper-fig-5)）
  - 參數數量大於 $13$ M 時並不會讓表現進步更多
  - 只有兩層時表現可以達到最佳
    - 訓練 $48$ 小時可以讓 WER 達到 $10.9\%$
    - 訓練 $100$ 小時可以讓 WER 達到 $10.7\%$
    - 訓練 $200$ 小時可以讓 WER 達到 $10.5\%$
  - 參數數量為 $85$ M 的 DNN 模型，最佳表現只能達到 $11.3\%$，並且需要訓練好幾個星期

[Eigen]: http://eigen.tuxfamily.org
[LSTM2002]: https://www.jmlr.org/papers/v3/gers02a.html
[pub43895]: https://research.google/pubs/pub43895/
[論文]: https://research.google/pubs/pub43905/
