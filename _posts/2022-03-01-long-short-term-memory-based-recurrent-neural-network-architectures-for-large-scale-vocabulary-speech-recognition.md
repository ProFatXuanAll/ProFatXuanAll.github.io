---
layout: ML-note
title: "Long Short-Term Memory Based Recurrent Neural Network Architectures for Large Scale Vocabulary Speech Recognition"
date: 2022-03-01 16:01:00 +0800
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
|目標|嘗試以 LSTM 進行字典範圍較大的語音辨識|
|作者|Hasim Sak, Andrew W. Senior, Françoise Beaufays|
|隸屬單位|Google|
|期刊/會議名稱|arXiv|
|發表時間|2014|
|論文連結|<https://research.google/pubs/pub43895/>|

## 重點

- 此篇論文被 ICASSP reject，因為頁數太少（含 reference 只有 5 頁）
  - 這篇論文真的就只跑兩個實驗
  - 後續作品為[這篇][pub43905]
- 此論文實驗結果說明 LSTM 可以套用到字典量大的語音辨識
  - 字典量大代表對應的 phoneme states 變多，真正的難題是如何將輸入特徵對應到 phoneme states
  - 過去使用傳統 RNN 模型的論文只能在字典量小的語音辨識資料集上表現不錯
  - 使用作者提出的 LSTM 架構可以達到語音辨識的 SOTA，原本的 LSTM 架構（這裡指的是 [LSTM-2002][LSTM2002]）則無法超越單純使用 DNN 的表現
  - 作者提出的 LSTM 架構主要在減少參數數量，參數數量比單純使用 feed-forward 架構的模型少 $2$ 到 $3$ 倍
- 為了使用 LSTM 進行大規模的平行化訓練，修改了 LSTM 架構讓訓練更有效率
  - 不需要使用 Connectionist Temporal Classifier （CTC） 或 RNN transducer 等架構
  - [PyTorch 實作的 LSTM][PyTorch-LSTM] 宣稱是參考此篇論文，但實際上實作的卻是 [LSTM-2000][LSTM2000] 的架構
  - 此篇論文是基於 [LSTM-2002][LSTM2002] 的架構進行改良

## 模型架構

### 原版 LSTM

假設 LSTM 的記憶單元（memory block）維度為 $1$（one cell in each memory block），共有 $n_c$ 個記憶單元，$n_i$ 個輸入單元，$n_o$ 個輸出單元，則 LSTM 總參數量（不含 bias）為（細節可見[我的筆記][LSTM2002]）

$$
W = n_c \times n_c \times 4 + n_i \times n_c \times 4 + n_c \times n_o + n_c \times 3 \tag{1}\label{1}
$$

- $n_c \times n_c \times 4$：記憶單元輸出以全連接的形式連接到記憶單元輸入、遺忘閘門、輸入閘門與輸出閘門
- $n_i \times n_c \times 4$：外部輸入以全連接的形式連接到記憶單元輸入、遺忘閘門、輸入閘門與輸出閘門
- $n_c \times n_o$：記憶單元輸出以全連接的形式連接到總輸出
- $n_c \times 3$：peephole connections

由於 LSTM 使用 truncated RTRL，因此每個時間點以隨機梯度下降法（stochastic gradient descent，SGD）進行參數最佳化的時間複雜度為 $O(W)$。

當輸入維度 $n_i$ 較小時，時間複雜度的主要貢獻來自於 $n_c \times (n_c + n_o)$。
在輸出預測範圍較大（字典範圍較大）或需要大量記憶容量（$n_c$ 較大時）的狀況下，模型的最佳化時間複雜度變高，計算成本大幅提升。
因此此論文提出想要將 LSTM 的複雜度降低成 $n_r \times (n_c + n_o)$，其中 $n_r \ll n_c$，$n_r$ 的定義在後面的文章段落中進行描述。

首先我們定義這篇論文使用的符號

|符號|意義|維度|備註|
|-|-|-|-|
|$T$|輸入序列的總長度||$T \in \N$|
|$t$|輸入序列的時間點||$t = 1, \dots, T$|
|$x_t$|第 $t$ 個時間點的**輸入**|$n_i$|$x = (x_1, \dots, x_T)$|
|$f_t$|第 $t$ 個時間點的**遺忘閘門**|$n_c$|$f_0 = 0$|
|$i_t$|第 $t$ 個時間點的**輸入閘門**|$n_c$|$i_0 = 0$|
|$o_t$|第 $t$ 個時間點的**輸出閘門**|$n_c$|$o_0 = 0$|
|$c_t$|第 $t$ 個時間點**記憶單元內部狀態**|$n_c$|$c_0 = 0$|
|$m_t$|第 $t$ 個時間點**記憶單元輸出**|$n_c$|$m_0 = 0$|
|$y_t$|第 $t$ 個時間點的**輸出**|$n_o$|$y = (y_1, \dots, y_T)$|
|$W_{i x}$|連接外部輸入與輸入閘門的參數|$n_c \times n_i$|全連接|
|$W_{i m}$|連接記憶單元輸出與輸入閘門的參數|$n_c \times n_c$|全連接|
|$W_{i c}$|連接記憶單元內部狀態與輸入閘門的參數|$n_c$|peephole connections|
|$b_i$|輸入閘門的偏差項|$n_c$||
|$W_{f x}$|連接外部輸入與遺忘閘門的參數|$n_c \times n_i$|全連接|
|$W_{f m}$|連接記憶單元輸出與遺忘閘門的參數|$n_c \times n_c$|全連接|
|$W_{f c}$|連接記憶單元內部狀態與遺忘閘門的參數|$n_c$|peephole connections|
|$b_f$|遺忘閘門的偏差項|$n_c$||
|$W_{o x}$|連接外部輸入與輸出閘門的參數|$n_c \times n_i$|全連接|
|$W_{o m}$|連接記憶單元輸出與輸出閘門的參數|$n_c \times n_c$|全連接|
|$W_{o c}$|連接記憶單元內部狀態與輸出閘門的參數|$n_c$|peephole connections|
|$b_o$|輸出閘門的偏差項|$n_c$||
|$W_{c x}$|連接外部輸入與記憶單元輸入的參數|$n_c \times n_i$|全連接|
|$W_{c m}$|連接記憶單元輸出與記憶單元輸入的參數|$n_c \times n_c$|全連接|
|$b_c$|記憶單元輸入的偏差項|$n_c$||
|$W_{y m}$|連接記憶單元輸出與總輸出的參數|$n_o \times n_c$|全連接|
|$b_y$|總輸出的偏差項|$n_o$||
|$\sigma$|sigmoid 函數|$\sigma(x) = \frac{1}{1 + e^{-x}}$||

得到 $t$ 時間點的外部輸入時可以計算 $t$ 時間點的遺忘閘門 $f_t$ 與輸入閘門 $i_t$

$$
\begin{align*}
i_t & = \sigma(W_{i x} \cdot x_t + W_{i m} \cdot m_{t - 1} + W_{i c} \odot c_{t - 1} + b_i) \\
f_t & = \sigma(W_{f x} \cdot x_t + W_{f m} \cdot m_{t - 1} + W_{f c} \odot c_{t - 1} + b_f)
\end{align*} \tag{2}\label{2}
$$

注意：論文不小心把 peephole connections 寫成全連接，因此 $W_{i c} \cdot c_{t - 1}$ 要改成 $W_{i c} \odot c_{t - 1}$，同理 $W_{f c} \cdot c_{t - 1}$ 要改成 $W_{f c} \odot c_{t - 1}$。

接著產生 $t$ 時間點的記憶單元內部狀態 $c_t$

$$
c_t = f_t \odot c_{t - 1} + i_t \odot \tanh(W_{c x} \cdot x_t + W_{c m} \cdot m_{t - 1} + b_c) \tag{3}\label{3}
$$

利用 $t - 1$ 時間點的記憶單元輸出 $m_{t - 1}$ 加上 $t$ 時間點的外部輸入 $x_t$ 與記憶單元內部狀態 $c_t$ 更新 $t$ 時間點的輸出閘門

$$
o_t = \sigma(W_{o x} \cdot x_t + W_{o m} \cdot m_{t - 1} + W_{o c} \odot c_t + b_o) \tag{4}\label{4}
$$

注意：論文不小心把 peephole connections 寫成全連接，因此 $W_{o c} \cdot c_t$ 要改成 $W_{o c} \odot c_t$。

接著可以計算 $t$ 時間點的記憶單元輸出 $m_t$

$$
m_t = o_t \odot \tanh(c_t) \tag{5}\label{5}
$$

最後利用 $t$ 時間點的記憶單元輸出 $m_t$ 計算 $t$ 時間點的總輸出 $y_t$

$$
y_t = W_{y m} \cdot m_t + b_y \tag{6}\label{6}
$$

注意 LSTM 的總輸出沒有使用啟發函數。

### 改版 LSTM

<a name="paper-fig-1"></a>

圖 1：改版 LSTM 架構。
圖片來源：[論文][論文]。

![圖 1](https://i.imgur.com/Oz7AHYQ.png)

為了降低計算的時間複雜度，作者提出了對 $m_t$ 進行降低維度的概念。

以 $r_t$ 代表降維後的 $m_t$，$r_t$ 的維度為 $n_r$，協助降維的參數為 $W_{r m}$，將 $\eqref{2} \eqref{3} \eqref{4}$ 中的 $m_t$ 改為 $r_t$

$$
\begin{align*}
i_t & = \sigma(W_{i x} \cdot x_t + W_{i r} \cdot r_{t - 1} + W_{i c} \odot c_{t - 1} + b_i) \\
f_t & = \sigma(W_{f x} \cdot x_t + W_{f r} \cdot r_{t - 1} + W_{f c} \odot c_{t - 1} + b_f) \\
c_t & = f_t \odot c_{t - 1} + i_t \odot \tanh(W_{c x} \cdot x_t + W_{c r} \cdot r_{t - 1} + b_c) \\
o_t & = \sigma(W_{o x} \cdot x_t + W_{o r} \cdot r_{t - 1} + W_{o c} \odot c_t + b_o)
\end{align*} \tag{7}\label{7}
$$

而 $\eqref{5}$ 的計算方法不變，得到 $\eqref{5}$ 我們使用 $W_{r m}$ 進行降維的動作

$$
r_t = W_{r m} \cdot m_t \tag{8}\label{8}
$$

最後計算總輸出的式子 $\eqref{6}$ 改為

$$
y_t = W_{y r} \cdot r_t + b_y \tag{9}\label{9}
$$

由於 $n_r < n_c$，將 $W_{\star m}$ 改成 $W_{\star r}$ 之後維度從 $n_c \times n_c$ 降維 $n_c \times n_r$，模型的總參數量（不含 bias）變成

$$
W = n_c \times n_i \times 4 + n_c \times n_r \times 4 + n_c \times 3 + n_r \times n_c + n_o \times n_r \tag{10}\label{10}
$$

當輸入維度 $n_i$ 較小時，時間複雜度的主要貢獻來自於 $n_r \times (n_c + n_o)$。

作者認為可以額外加上一些非遞迴單元 $p_t$，在不增加遞迴計算的維度下讓與輸出層相接的隱藏層維度稍微增加一些。

令 $p_t$ 的維度為 $n_p$，我們額外定義新的參數 $W_{p m}$，並使用記憶單元輸出 $m_t$ 計算 $p_t$

$$
p_t = W_{p m} \cdot m_t \tag{11}\label{11}
$$

最後將 $\eqref{9}$ 修改為

$$
y_t = W_{y r} \cdot r_t + W_{y p} \cdot p_t + b_y \tag{12}\label{12}
$$

注意 $r_t$ 與 $p_t$ 不同，$r_t$ 有參與遞迴的過程，$p_t$ 並沒有參與遞迴的過程。

在加入 $p_t$ 後參數的數量變成

$$
W = n_c \times n_i \times 4 + n_c \times n_r \times 4 + n_c \times 3 + (n_r + n_p) \times n_c + n_o \times (n_r + n_p) \tag{13}\label{13}
$$

### 實作

- 使用 CPU 而不是 GPU
  - 使用 CPU 方便 debug
  - 當時的環境是 Google 有大量 CPU 叢集節點（clustering node），但沒有 GPU 叢集節點
  - 這是 2014 年的論文，還沒有 tensorflow 可以用，所以這個選擇可以理解
- 使用 [Eigen][Eigen] 函式庫進行矩陣計算
  - 版本為 `v3`
  - 支援 C++
  - 支援 SIMD 平行化指令
- 使用非同步梯度下降（Asynchronous Stochastic Gradient Descent，ASGD）演算法進行最佳化
- 因為有多層 LSTM，使用 truncated BPTT 進行最佳化
  - 注意不是採用[原始 LSTM][LSTM1997] 論文中的 truncated RTRL
  - 每 $20$ 個時間點進行一次 BPTT
  - 每 $20$ 個時間點的計算狀態會保留給下一次 $20$ 個時間點當成計算初始狀態
  - 一個 batch 會由長度為 $20$ 個時間點的序列組成
  - 一個 batch 中比較短的序列會以 padding 補齊，並且在下一個 batch 中替換成其他輸入序列，對應的計算狀態都會初始化
- 最佳化目標為 cross entropy

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
- 將總共 $14247$ 個狀態減少成三種不同數量的狀態進行實驗
  - 前後文無關狀態（context independent states，CI）：共有 $126$ 個狀態，每 $3$ 個 phonemes 組成一個狀態，共有 $42$ 種不同的 phonemes
  - 使用事先定義好的 phonemes 等價關係將狀態分別減少至 $8000$ 與 $2000$，仍為 CD 狀態
- 標記資料為每個 $40$ 維的 feature 對應到的 phoneme state
  - 模型每個時間點的輸入至少為 $40$ 維（對應到 $n_i$）
  - 模型每個時間點的輸出為對應到的狀態（對應到 $n_o$）
- 每個實驗設置都採用各自最適合的 learning rate（hyperparameter tuning），並對 learning rate 使用 expenentially decay
- 評估方法
  - 驗證資料（development set）有 $200000$ 幀，針對每一幀中所有的 state 進行準確率（accuracy）的計算，稱為 frame accuracy
  - 測試資料（test set）有 $23000$ 幀，計算文字辨識錯誤率（word error rates）
    - 需要額外擁有一個 language model 進行狀態到文字的轉換
    - 所有實驗共用相同的 language model，字典大小為 $2.6$ M
    - 這裡的假設為：當模型能夠將輸入特徵與狀態對齊成功時，後續的 language model 就會自然產出正確的辨識文字結果

### DNN 實驗設計

- batch size = $200$ 幀
- 使用 GPU 進行訓練
- 模型採用全連接架構，隱藏層都使用 sigmoid 作為 activation function，輸出使用 softmax 進行 normalization
- 輸入共包含 $3$ 個部份
  - 當前幀數：$1$
  - 未來幀數：$5$
  - 過去幀數：$10$ 或 $16$，分別標記為 ``10w5`` 與 ``16w5``

### RNN 實驗設計

- 使用 ASGD 進行最佳化
- 使用 CPU 進行訓練，一個 CPU 使用 $24$ 的 threads，只使用一個 CPU
  - 使用 data parallel 的概念，每個 thread 計算 $4$ 到 $8$ 筆序列資料
  - 使用 truncated BPTT，一次只計算 $20$ 個 time steps
- RNN 的非遞迴隱藏層使用 sigmoid activation，遞迴的隱藏層不使用 activation function
- LSTM 架構請參考 $\eqref{7} \eqref{8} \eqref{11} \eqref{12}$
- 由於未來時間的資訊有助於提升預測的準確度，因此模型預測會在延遲 $5$ 幀後開始輸出
  - ex: 第 $0$ 幀到第 $4$ 幀輸入完後預測第 $0$ 幀的 $40$ 維特徵所對應到的狀態

## 實驗 1：驗證資料的表現結果

<a name="paper-fig-2"></a>

圖 2：在 $n_o = 126$ 時的驗證資料的表現結果。
圖片來源：[論文][論文]。

![圖 2](https://i.imgur.com/qjEOPv9.png)

<a name="paper-fig-3"></a>

圖 3：在 $n_o = 2000$ 時的驗證資料的表現結果。
圖片來源：[論文][論文]。

![圖 3](https://i.imgur.com/II30qkv.png)

<a name="paper-fig-4"></a>

圖 4：在 $n_o = 8000$ 時的驗證資料的表現結果。
圖片來源：[論文][論文]。

![圖 4](https://i.imgur.com/Fq8vYNQ.png)

- 圖中的實驗名稱包含架構資訊
  - ``c2048`` 代表 $n_c = 2048$
  - ``r512`` 代表 $n_r = 512$
  - ``p256`` 代表 $n_p = 256$
  - ``10w5_6_704`` 代表輸入包含過去 $10$ 幀與未來 $5$ 幀，隱藏層有 $6$ 層，每個隱藏層維度為 $704$
  - ``lr_256`` 代表故意降維成 $256$ 維的全連接層，目的是為了和 LSTM 公平的比較
  - 括號中的數字代表總參數量
- RNN 在 $n_o = 126$ 時表現已經比 DNN 與 LSTM 差，因此後續實驗不討論 RNN
  - 在訓練過程作者發現 RNN 非常不穩定，必須要額外進行 gradients clipping 確保不會產生 gradient explosion
- LSTM 表現最好而且收斂速度最快
  - 在採用作者提出的架構下，使用 projection 的 LSTM 比原本 LSTM 表現還要好，使用的參數也比較少
  - 在採用 $n_p > 0$ 的架構下，大部份實驗都比 $n_p = 0$ 的架構表現還要好，唯一的例外是[圖 3](#paper-fig-2) 的實驗，作者認為是 learning rate 不小心調的太小

## 實驗 2：測試資料的表現結果

<a name="paper-fig-5"></a>

圖 5：在 $n_o = 126$ 時的測試資料的表現結果。
圖片來源：[論文][論文]。

![圖 5](https://i.imgur.com/H4omLt0.png)

<a name="paper-fig-6"></a>

圖 6：在 $n_o = 2000$ 時的測試資料的表現結果。
圖片來源：[論文][論文]。

![圖 6](https://i.imgur.com/xJpYoZY.png)

<a name="paper-fig-7"></a>

圖 7：在 $n_o = 8000$ 時的測試資料的表現結果。
圖片來源：[論文][論文]。

![圖 7](https://i.imgur.com/3rG7mzq.png)

- 作者說有些模型還沒完全收斂，他會更新實驗結果，很顯然他忘記了
- 簡單來說作者提出的 LSTM 架構就是表現比較好
  - 如果使用 [LSTM-2002][LSTM2002] 而不是作者的架構，則表現會比 DNN 還差
  - 單純的增加 DNN 的層數也可以讓表現變好

[Eigen]: http://eigen.tuxfamily.org
[LSTM1997]: https://ieeexplore.ieee.org/abstract/document/6795963
[LSTM2000]: https://direct.mit.edu/neco/article-abstract/12/10/2451/6415/Learning-to-Forget-Continual-Prediction-with-LSTM
[LSTM2002]: https://www.jmlr.org/papers/v3/gers02a.html
[PyTorch-LSTM]: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
[pub43905]: https://research.google/pubs/pub43905/
[論文]: https://research.google/pubs/pub43895/
