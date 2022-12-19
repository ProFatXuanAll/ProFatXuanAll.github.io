---
layout: ML-note
title: "Generating Sequences with Recurrent Neural Networks"
date: 2022-08-29 15:30:00 +0800
categories: [
  Text Modeling,
]
tags: [
  BPTT,
  LSTM,
  RNN,
  language model,
  model architecture,
  neural network,
]
author: [
  Alex Graves,
]
---

|-|-|
|目標|使用 LSTM 進行 sequence generation|
|作者|Alex Graves|
|隸屬單位|University of Toronto|
|期刊/會議名稱|arXiv|
|發表時間|2013|
|論文連結|<https://arxiv.org/abs/1308.0850>|

<!--
  Define LaTeX command which will be used through out the writing.

  Each command must be wrapped with $ signs.
  We use "display: none;" to avoid redudant whitespaces.
 -->

<p style="display: none;">

  <!-- c but bold. -->
  $\providecommand{\cb}{}$
  $\renewcommand{\cb}{\mathbf{c}}$
  <!-- h but bold. -->
  $\providecommand{\hb}{}$
  $\renewcommand{\hb}{\mathbf{h}}$
  <!-- x but bold. -->
  $\providecommand{\xb}{}$
  $\renewcommand{\xb}{\mathbf{x}}$
  <!-- y but bold. -->
  $\providecommand{\yb}{}$
  $\renewcommand{\yb}{\mathbf{y}}$

  <!-- H but curly. -->
  $\providecommand{\Hc}{}$
  $\renewcommand{\Hc}{\mathcal{H}}$
  <!-- L but curly. -->
  $\providecommand{\Lc}{}$
  $\renewcommand{\Lc}{\mathcal{L}}$
  <!-- N but curly. -->
  $\providecommand{\Nc}{}$
  $\renewcommand{\Nc}{\mathcal{N}}$
  <!-- Y but curly. -->
  $\providecommand{\Yc}{}$
  $\renewcommand{\Yc}{\mathcal{Y}}$

  <!-- alpha with hat. -->
  $\providecommand{\alphah}{}$
  $\renewcommand{\alphah}{\hat{\alpha}}$
  <!-- beta with hat. -->
  $\providecommand{\betah}{}$
  $\renewcommand{\betah}{\hat{\beta}}$
  <!-- e with hat. -->
  $\providecommand{\eh}{}$
  $\renewcommand{\eh}{\hat{e}}$
  <!-- gamma with hat. -->
  $\providecommand{\gammah}{}$
  $\renewcommand{\gammah}{\hat{\gamma}}$
  <!-- kappa with hat. -->
  $\providecommand{\kappah}{}$
  $\renewcommand{\kappah}{\hat{\kappa}}$
  <!-- mu with hat. -->
  $\providecommand{\muh}{}$
  $\renewcommand{\muh}{\hat{\mu}}$
  <!-- pi with hat. -->
  $\providecommand{\pih}{}$
  $\renewcommand{\pih}{\hat{\pi}}$
  <!-- rho with hat. -->
  $\providecommand{\rhoh}{}$
  $\renewcommand{\rhoh}{\hat{\rho}}$
  <!-- sigma with hat. -->
  $\providecommand{\sigmah}{}$
  $\renewcommand{\sigmah}{\hat{\sigma}}$
  <!-- y with hat. -->
  $\providecommand{\yh}{}$
  $\renewcommand{\yh}{\hat{y}}$

</p>

<!-- End LaTeX command define section. -->

## 重點

- 採用全微分更新 LSTM 時要記得作 gradient clipping
- LSTM + residual connection 才能讓多層模型正常訓練
- 作者使用單詞中的平均字數估算 BPC 與 perplexity 的轉換公式
  - 早年的 BPC 與 perplexity 需要對不同的 n-gram 進行計算，由於神經網路可以直接模擬完整序列因此不需要呈現不同 n 對應的數據
  - 不論是 BPC 還是 perplexity 都是以 $2$ 為底，Pytorch 與 Tensorflow 中的 entropy 都是以自然對數為底，計算數值要小心
- 透過 Hutter prize 實驗證實 LSTM 擁有處理長距離資訊的能力
  - 能夠生成對稱的 XML tags
  - 能夠生成對稱的標點符號
- LSTM + Gaussian mixture models 可以模擬實數序列，例如生成手寫字跡
  - 同時提出 seq2seq 架構的雛型
  - 同時提出控制生成風格的方法雛型

## 模型

### 整體模型架構

<a name="paper-fig-1"></a>

圖 1：整體模型架構。
圖片來源：[論文][論文]。

![圖 1](https://i.imgur.com/Sv7n7wk.png)

令 $(x_1, \dots, x_T, x_{T+1})$ 為一向量序列：

- $\xb = (x_1, \dots, x_T)$ 會作為模型的輸入，得到輸出向量序列 $\yb = (y_1, \dots, y_T)$
- $(x_2, \dots, x_{T+1})$ 會作為模型的預測目標，對任意的 $t \in \set{1, \dots, T}$，模型在輸入 $x_t$ 之後必須要讓輸出 $x_{t+1}$ 的機率值最大化，即最大化 $\Pr(x_{t + 1} \vert y_t)$
- 令 $x_1$ 代表 begin-of-sequence token，作者將 $x_1$ 所有數值設成 $0$

令 $N$ 為序列模型中的 RNN 層數，令第 $n$ 層（$n \in \set{1, \dots, N}$）的輸出結果為 $\hb^n = (h_1^n, \dots, h_T^n)$：

- 將 RNN 模型以 $\Hc$ 表示
- 論文中每一層 RNN 模型都是採用 LSTM 模型（見[圖 2](#paper-fig-2)），但當前討論先假設是廣義的 RNN
- 根據[圖 1](#paper-fig-1)，每層模型都有 skip connections 將輸入 $x_t$ 與 $h_t^n$ 相連，作者說這是為了避免梯度消失（gradient vanishing）問題
- 根據[圖 1](#paper-fig-1)，每個時間點不同 RNN 隱藏層的輸出會進行轉換並疊加，計算結果作為最後的預測的來源

正式的數學描述如下：

$$
\begin{align*}
& \algoProc{SeqModel}(\xb, h_0^1, \dots, h_0^N) \\
& \hspace{1em} \algoFor{t \in \set{1, \dots, T}} \\
& \hspace{2em} h_t^1 \algoEq \Hc(W_{i h^1} x_t + W_{h^1 h^1} h_{t-1}^1 + b_h^1) && \tag{1}\label{1} \\
& \hspace{2em} \algoFor{n \in \set{2, \dots, N}} \\
& \hspace{3em} h_t^n \algoEq \Hc(W_{i h^n} x_t + W_{h^{n-1} h^n} h_t^{n-1} + W_{h^n h^n} h_{t-1}^n + b_h^n) && \tag{2}\label{2} \\
& \hspace{2em} \algoEndFor \\
& \hspace{2em} \yh_t \algoEq b_y + \sum_{n = 1}^N W_{h^n y} h_t^n && \tag{3}\label{3} \\
& \hspace{2em} y_t \algoEq \Yc(\yh_t) && \tag{4}\label{4} \\
& \hspace{1em} \algoEndFor \\
& \hspace{1em} \Pr(\xb) \algoEq \prod_{t = 1}^T \Pr(x_{t+1} \vert y_t) && \tag{5}\label{5} \\
& \hspace{1em} \Lc(\xb) \algoEq -\sum_{t = 1}^T \log \Pr(x_{t+1} \vert y_t) && \tag{6}\label{6} \\
& \hspace{1em} \algoReturn \Pr(\xb), \Lc(\xb) \\
& \algoEndProc
\end{align*}
$$

|參數|意義|節點|意義|
|-|-|-|-|
|$W_{i h^1}$|連接輸入與第 $1$ 層 RNN|$\xb$|輸入向量序列|
|$W_{h^n h^n}$|第 $n$ 層 RNN 的自連接層|$h_0^n$|第 $n$ 層 RNN 的起始狀態|
|$W_{h^{n-1} h^n}$|連接第 $n-1$ 層 RNN 與第 $n$ 層 RNN|$h_t^n$|第 $n$ 層 RNN 第 $t$ 時間點的狀態|
|$W_{h^n y}$|連接第 $N$ 層 RNN 的狀態與輸出層|$\yh_t$|所有 RNN 隱藏第 $t$ 時間點的狀態疊加結果|
|$b_h^n$|第 $n$ 層 RNN 的輸入 bias|$y_t$|$t$ 時間點的輸出|
|$\Yc$|輸出層|$\Pr(x_{t+1} \vert y_t)$|$y_t$ 預測答案 $x_{t+1}$ 的機率值|
|||$\Lc(\xb)$|Negative log-likelihood of $\xb$|

### LSTM 模型架構

<a name="paper-fig-2"></a>

圖 2：LSTM 模型架構。
圖片來源：[論文][論文]。
![圖 2](https://i.imgur.com/quKKMOh.png)

將[圖 1](#paper-fig-1) 中的 $\Hc$ 定義成 [LSTM-2002](LSTM2002)，正式定義如下

$$
\begin{align*}
& \algoProc{LSTM}(x_t, h_{t-1}, c_{t-1}) \\
& \hspace{1em} i_t \algoEq \sigma(W_{x i} x_t + W_{h i} h_{t-1} + W_{c i} \odot c_{t - 1} + b_i) && \tag{7}\label{7} \\
& \hspace{1em} f_t \algoEq \sigma(W_{x f} x_t + W_{h f} h_{t-1} + W_{c f} \odot c_{t - 1} + b_f) && \tag{8}\label{8} \\
& \hspace{1em} c_t \algoEq f_t \odot c_{t-1} + i_t \odot \tanh(W_{x c} x_t + W_{h c} h_{t-1} + b_c) && \tag{9}\label{9} \\
& \hspace{1em} o_t \algoEq \sigma(W_{x o} x_t + W_{h o} h_{t-1} + W_{c o} c_t + b_o) && \tag{10}\label{10} \\
& \hspace{1em} h_t \algoEq o_t \odot \tanh(c_t) && \tag{11}\label{11} \\
& \hspace{1em} \algoReturn h_t, c_t \\
& \algoEndProc
\end{align*}
$$

|參數|意義|節點|意義|
|-|-|-|-|
|$W_{x g}$|連接輸入與 gate $g \in \set{i, f, o}$|$x_t$|$t$ 時間點的輸入向量|
|$W_{h g}$|連接 LSTM hidden states 與 gate $g \in \set{i, f, o}$|$h_t$|$t$ 時間點的 LSTM hidden states|
|$W_{c g}$|連接 LSTM internal states 與 gate $g \in \set{i, f, o}$|$c_t$|$t$ 時間點的 LSTM internal states|
|$b_g$|bias of gate $g \in \set{i, f, o}$|$i_t$|$t$ 時間點的 input gate|
|$W_{x c}$|連接輸入與 memory cell|$f_t$|$t$ 時間點的 forget gate|
|$W_{h c}$|連接 LSTM hidden states 與 memory cell|$o_t$|$t$ 時間點的 output gate|
|$b_c$|memory cell bias|$c_t$|$t$ 時間點的 LSTM internal states|

在替換 $\eqref{1} \eqref{2}$ 中的 RNN $\mathcal{H}$ 為上述定義的 LSTM 後，仍然採用[圖 1](#paper-fig-1) 中的計算方法，可以想成把 LSTM 的輸入 $x_t$ 換成 $W_{i h^n} x_t + W_{h^{n-1} h^n} h_t^{n-1}$。

### LSTM 最佳化

[原版的 LSTM][LSTM1997] 使用 truncated RTRL 的變種計算梯度，而在此篇論文中作者採用 BPTT 全微分作為更新梯度。
然而原始的梯度計算方法是為了避免梯度爆炸（gradient explosion）問題，因此作者提出手動將超過數值範圍的梯度設為符合事先定義的範圍（gradient clipping）。
作者說他之前發表的研究其實都有作 gradient clipping，但他忘記講了，好壞。

## 預測文字

定義 $\eqref{4}$ 的計算方法為

$$
\forall k \in \set{1, \dots, K}, \Pr(x_{t+1} = k \vert y_t) = y_t^k = \dfrac{\exp(\yh_t^k)}{\sum_{k' = 1}^K \exp(\yh_t^{k'})}. \tag{12}\label{12}
$$

其中 $K$ 代表類別數量，在預測文字的任務中代表字典大小。
而輸入 $x_t \in \R^K$ 則對應到 one-hot encoding。
將 $\eqref{12}$ 代入 $\eqref{6}$ 可以推得

$$
\Lc(\xb) = -\sum_{t = 1}^T \log y_t^{x_{t + 1}}. \tag{13}\label{13}
$$

而 $\Lc(\xb)$ 相對於 $\yh_t^k$ 的梯度為

$$
\begin{align*}
\pd{\Lc(\xb)}{\yh_t^k} & = \pd{\Lc(\xb)}{y_t^{x_{t+1}}} \pd{y_t^{x_{t+1}}}{\yh_t^k} \\
& = -\pd{\log y_t^{x_{t+1}}}{y_t^{x_{t+1}}} \pd{y_t^{x_{t+1}}}{\yh_t^k} \\
& = \dfrac{-1}{y_t^{x_{t+1}}} \pd{y_t^{x_{t+1}}}{\yh_t^k} \\
& = \dfrac{-1}{y_t^{x_{t+1}}} \pa{\dfrac{\exp(\yh_t^{x_{t+1}})}{\sum_{k' = 1}^K \exp(\yh_t^{k'})} \delta_{k, x_{t+1}} - \dfrac{\exp(y_t^{x_{t+1}}) \exp(\yh_t^k)}{\pa{\sum_{k' = 1}^K \exp(\yh_t^k)}^2}} \\
& = \dfrac{-1}{y_t^{x_{t+1}}} \pa{y_t^{x_{t+1}} \delta_{k, x_{t+1}} - y_t^{x_{t+1}} y_t^k} \\
& = y_t^k - \delta_{k, x_{t + 1}}. && \tag{14}\label{14}
\end{align*}
$$

當 $K$ 很大時（常見的單詞數量 $> 100000$），計算 softmax 成本就會很高，因此不少論文提出減少 $K$ 的方法，其中 character-level language model 就是能夠有效減少 $K$ 的手段。

### 實驗 1：Penn Treebank - Wall Street Journal

<a name="paper-fig-3"></a>

圖 3：PTB-WSJ 實驗結果。
圖片來源：[論文][論文]。
![圖 3](https://i.imgur.com/n9115bj.png)

- Penn Treebank 中的 Wall Street Journal（WSJ）用來作為 language modeling benchmark
  - Training set：930000 words
  - Validation set：74000 words
  - Test set：82000 words
- 評估方法
  - BPC（bit-per-character）：average value of $-\log_2 \Pr(x_{t+1} \vert y_t)$ over the whole test set, i.e.,

    $$
    \dfrac{-1}{\hash\text{characters in dataset}} \sum_{\xb \text{ in dataset}} \log_2 \Pr(x_{t+1} \vert y_t)
    $$

  - Perplexity：two to the power of the average number of bits per word, i.e.,

    $$
    \dfrac{-1}{\hash\text{words in dataset}} \sum_{\xb \text{ in dataset}} \log_2 \Pr(x_{t+1} \vert y_t)
    $$

  - 當 token 設定成 word 時，BPC 無法直接計算，作者認為由於 WSJ 中的所有 word 平均擁有 5.6 個 characters，因此套用以下公式進行近似計算：

    $$
    \operatorname{perplexity} \approx 2^{5.6 \operatorname{BPC}}
    $$

  - 當 token 設定成 character 時，word 機率值為組成的 characters 的機率值連乘積，因此 perplexity 仍然可以正常計算
- 每個文字序列都有插入 end-of-sentence token，注意是 sentence 不是 sequence，代表單一文字序列中可能有多個 end-of-sentence token
  - End-of-sentence token 會參與誤差計算
  - 由於 begin-of-sequence token 已經定義，因此不需要定義 start-of-sentence token，可以由 end-of-sentence token 替代 start-of-sentence token
- 模型架構
  - 只使用 $1$ 層 LSTM（$N = 1$），隱藏層單元數為 $1000$（$h_t^1 \in \R^{1000}$）
  - 作者將模型字典大小設定為 $10000$ tokens，不在字典中的所有字都定義成 unknown token
  - 當 token 設定成 character 時
    - 字典大小為 $49$
    - 資料集中幾乎沒有任何的 unknown token
    - 參數總數約為 $4.3$ M
  - 當 token 設定成 word 時
    - 字典大小為 $10000$
    - 資料集中會包含不少的 unknown token
    - 參數總數約為 $54$ M
- 最佳化演算法為 SGD（stochastic gradient descent）
  - Learning rate 設為 $0.0001$
  - Momentum 設為 $0.99$
  - Gradient clipping 的範圍為 $[-1, 1]$
- 由於模型容易 overfitting，因此作者加入 2 種不同的 regularization 技巧
  - Weight noise with std set to $0.075$
  - Adaptive weight noise + minimum description length
- 參數初始化策略
  - 使用 weight noise 的模型參數初始化為不使用 weight noise 訓練的模型參數
  - 使用 adaptive weight noise 的模型參數初始化為使用 weight noise 訓練的模型參數
  - 作者發現使用訓練過的模型作為初始化的參數收斂速度快於從頭訓練 + regularization
  - Word-level language model 使用 adaptive weight noise 訓練太慢，因此只使用 fix weight noise
- 結論
  - Word-level 表現比 character-level 好，但使用 regularization 可讓兩者差異減少
  - 當 test set 也被用來更新模型（只更新 1 次，稱為 dynamic evaluation），模型表現更好
  - 作者與 Tomas Mikolov 的實驗結果比較，發現 LSTM + dynamic evaluation 表現比 RNN + dynamic evaluation 更好
  - 此階段的 NN 模型比 N-gram 統計模型表現還要差（perplexity 最低可到 $89.4$）

### 實驗 2：Hutter prize

<a name="paper-fig-4"></a>

圖 4：Hutter prize 實驗結果。
圖片來源：[論文][論文]。
![圖 4](https://i.imgur.com/ZK80PEG.png)

- First $100$ million bytes of English Wikipedia dumped at March 3rd 2006
  - 包含 XML、數字、中文等非英文文字
  - Hutter prize 是在比賽壓縮演算法，作者用 LSTM 來測試 BPC
- 作者自己定義 train-valid split
  - Training set：the first $96$ M bytes
  - Validation set：the remaining $4$ M bytes
  - 切成以 $100$ bytes 為單位的 sequence 進行計算
  - 以 $1$ byte 為單位來計算共有 $205$ 個不同的 unicode symbols
  - 以 character 為單位來計算遠超過 $205$ 個不同的 unicode characters
  - 作者決定以 $1$ byte 作為字典基礎單位，所以字典大小為 $205$
  - 由於以 $1$ byte 作為單位，因此計算 BPC 時每個 character 的機率是由組成的 bytes 的機率值的連乘積估算而得
- 由於資料中有 XML 結構的存在，模型需要能夠記憶長距離資訊才能讓 bit per byte 下降
  - 特別寫出結構與長距離問題其實是作者想要強調 LSTM 擁有該能力
  - 每隔 $100$ 個 sequences（共 $10000$ bytes）才手動重設一次 LSTM 的狀態（$h_t$ 與 $c_t$）
  - 每次的模型更新都只限制在 $1$ 個 sequence（共 $100$ bytes）內，以此加速訓練
  - 由於想要摹擬長距離的能力，因此資料集不進行隨機抽樣
- 模型架構
  - 使用 $7$ 層 LSTM（$N = 7$），隱藏層單元數為 $700$（$h_t^1 \in \R^{700}$）
  - 定義 token 為 $1$ byte，參數總量約為 $21.3$ M
- 最佳化演算法為 SGD
  - Learning rate 設為 $0.0001$
  - Momentum 設為 $0.9$
  - Gradient clipping 的範圍為 $[-1, 1]$
  - 需要訓練 $4$ 個 epoch 才會收斂
- 結論
  - 使用 dynamic evaluation 在 validation set 上表現比較好
    - 與 PTB-WSJ 實驗結果一致
    - 作者認為在 validation set 上的分佈與 training set 不同，因此 dynamic evaluation 表現較好是理所當然
  - 使用 dynamic evaluation 在 validation set 上表現比 training set 好
    - 作者認為可能是模型在 training set 上仍然 underfitting 所導致
    - 作者認為可能部份資料比較難 fitting（例如 plain text），而 validation set 都拿到剛好比較容易 fitting 的資料（例如 XML tag）
  - 此階段的 NN 模型比 N-gram 統計模型表現還要差，但 LSTM 在 NN 模型中表現最好
    - N-gram 模型 BPC 最低可到 $1.28$
    - zip 壓縮演算法 BPC 會高於 $2$
    - Character-level RNN，且只處理 plain text 不包含 XML，BPC 最低可到 $1.47$
  - 將 LSTM 模型用於生成文字內容，作者觀察發現 LSTM 有學到文字中包含的結構
    - 學到常見單詞與 sub-words，並且能夠任意組合 sub-words
    - 學到標點符號的用法
    - 學到對稱的 quotation marks 與 parentheses 證明模型擁有記憶長距離資訊的能力
    - 進一步學會對稱生成 `==` 符號（wiki 用來標記 header 的符號）、XML tags 與縮排結構（indentation）
  - 雖然生成的有模有樣，但作者認為從 phrase 結構來看生成的結果是無法作為正常文字使用的
    - 作者認為更好的模擬方法可以從加大與加深模型下手（與 2019--2022 的趨勢相同）
    - 作者認為增加訓練資料也是一個強化模型的方法（再度與 2019--2022 的趨勢相同）
    - 作者認為不論讓模型多有效的模擬文字生成，最後都無法模擬人類（想法不如 GPT 等研究樂觀）

## 手寫文字預測

模型將會接收到連續的數值，代表筆尖（pen-tip）的座標隨著時間移動的軌跡

- 作者把這類型的連續數值稱為 online handwriting data
- 相對於 online，offline 的意思是直接獲得最終的手寫圖片進行辨識
- 相較於 offline，online 的優勢為輸入數字每個時間點都只有 2 個數值，因此數值結構比起完整的圖片來說維度較低

沿用 $\eqref{1}--\eqref{11}$ 定義的 LSTM 模型，定義每個時間點的模型輸入如下

$$
x_t = (x_{t, 1}, x_{t, 2}, x_{t, 3}) \tag{15}\label{15}
$$

- $(x_{t, 1}, x_{t, 2}) \in \R^2$ 代表平面座標
- $x_{t, 3} \in \set{0, 1}$ 代表筆尖是否離開智慧白板（end-of-stroke）

模型預測目標共有兩個，分別為下個時間點的筆跡座標值與筆尖是否離開智慧白板。
作者將 LSTM 模型結合 mixture density network 使用，主要概念為將 LSTM 模型的輸出作為 Gaussian mixture models（GMM）的參數（parameters）使用，因此定義每個時間點的輸出如下

$$
y_t = \pa{e_t, \set{\pi_t^j, \mu_t^j, \sigma_t^j, \rho_t^j}_{j = 1}^M} \tag{16}\label{16}
$$

- 由於預測目標共有兩個，因此使用兩個不同的機率分佈進行模擬
- $e_t \in (0, 1)$ 代表 end-of-stroke 的機率值，因此採用 Bernoulli distribution 進行模擬
- 作者假設下個時間點的筆尖座標值符合 bivariate Gaussian distribution，因此使用 GMM 進行模擬
- $M$ 代表 bivariate Gaussian models 的數量
- $\mu_t^j \in \R^2$ 代表第 $j$ 個 bivariate Gaussian model 的平均值
- $\sigma_t^j \in \R^2$ 代表第 $j$ 個 bivariate Gaussian model 的標準差
- $\rho_t^j \in \R$ 代表第 $j$ 個 bivariate Gaussian model 的相關係數
- $\pi_t^j \in \R$ 代表第 $j$ 個 bivariate Gaussian model 的組合權重

根據 $\eqref{16}$，作者將 $\eqref{3}$ 的定義改為

$$
\yh_t = \pa{\eh_t, \set{\pih_t^j, \muh_t^j, \sigmah_t^j, \rhoh_t^j}_{j = 1}^M} = b_y + \Sum_{n = 1}^N W_{h^n y} h_t^n \tag{17}\label{17}
$$

接著定義 $\eqref{16}$ 的計算方法：

$$
\begin{align*}
e_t        & = \dfrac{1}{1 + \exp(\eh_t)}                                                              && \implies e_t \in (0, 1)                                 && \tag{18}\label{18} \\
\pi_t^j    & = \softmax\pa{\pih_t^j} = \dfrac{\exp\pa{\pih_t^j}}{\Sum_{j' = 1}^M \exp\pa{\pih_t^{j'}}} && \implies \pi_t^j \in (0, 1), \Sum_{j = 1}^M \pi_t^j = 1 && \tag{19}\label{19} \\
\mu_t^j    & = \muh_t^j                                                                                && \implies \muh_t^j \in \R^2                              && \tag{20}\label{20} \\
\sigma_t^j & = \exp\pa{\sigmah_t^j}                                                                    && \implies \sigma_t^j \in \R_+^2                          && \tag{21}\label{21} \\
\rho_t^j   & = \tanh\pa{\rhoh_t^j}                                                                     && \implies \rho_t^j \in (-1, 1)                           && \tag{22}\label{22}
\end{align*}
$$

注意 $\eqref{18}$ 與 sigmoid function 類似但不是 sigmoid function。
接著定義 GMM 的計算方法：

$$
\Pr(x_{t+1} \vert y_t) = \begin{dcases}
  \Sum_{j = 1}^M \pi_t^j \cdot \Nc\pa{x_{t+1} \vert \mu_t^j, \sigma_t^j, \rho_t^j} \cdot e_t       & \text{if } x_{t+1, 3} = 1 \\
  \Sum_{j = 1}^M \pi_t^j \cdot \Nc\pa{x_{t+1} \vert \mu_t^j, \sigma_t^j, \rho_t^j} \cdot (1 - e_t) & \text{otherwise}
\end{dcases} \tag{23}\label{23}
$$

其中 $\Nc$ 代表 bivariate Gaussian distribution，計算方法為

$$
\begin{align*}
& \Nc\pa{x_{t+1} \vert \mu_t^j, \sigma_t^j, \rho_t^j} = \dfrac{1}{2 \pi \sigma_{t, 1}^j \sigma_{t, 2}^j \sqrt{1 - (\rho_t^j)^2}} \cdot \exp\pa{\dfrac{-Z}{2 (1 - (\rho_t^j)^2)}} && \tag{24}\label{24} \\
& Z = \pa{\dfrac{x_{t+1, 1} - \mu_{t, 1}^j}{\sigma_{t, 1}^j}}^2 + \pa{\dfrac{x_{t+1, 2} - \mu_{t, 2}^j}{\sigma_{t, 2}^j}}^2 - \dfrac{2 \rho_t^j \pa{x_{t+1, 1} - \mu_{t, 1}^j} \pa{x_{t+1, 2} - \mu_{t, 2}^j}}{\sigma_{t, 1}^j \sigma_{t, 2}^j} && \tag{25}\label{25}
\end{align*}
$$

注意 $\eqref{23}$ 中的 $e_t$ 與 $\Sum_{j = 1}^M$ 無關，因此 $\eqref{6}$ 可以改寫為

$$
\begin{align*}
\Lc(\xb) & = \Sum_{t = 1}^T -\log \Pr(x_{t+1} \vert y_t) \\
& = \begin{dcases}
  \Sum_{t = 1}^T -\log \pa{\Sum_{j = 1}^M \pi_t^j \cdot \Nc\pa{x_{t+1} \vert \mu_t^j, \sigma_t^j, \rho_t^j} \cdot e_t}       & \text{if } x_{t+1, 3} = 1 \\
  \Sum_{t = 1}^T -\log \pa{\Sum_{j = 1}^M \pi_t^j \cdot \Nc\pa{x_{t+1} \vert \mu_t^j, \sigma_t^j, \rho_t^j} \cdot (1 - e_t)} & \text{otherwise}
\end{dcases} \\
& = \begin{dcases}
  \Sum_{t = 1}^T \pa{-\log \pa{\Sum_{j = 1}^M \pi_t^j \cdot \Nc\pa{x_{t+1} \vert \mu_t^j, \sigma_t^j, \rho_t^j}} - \log e_t}       & \text{if } x_{t+1, 3} = 1 \\
  \Sum_{t = 1}^T \pa{-\log \pa{\Sum_{j = 1}^M \pi_t^j \cdot \Nc\pa{x_{t+1} \vert \mu_t^j, \sigma_t^j, \rho_t^j}} - \log (1 - e_t)} & \text{otherwise}
\end{dcases} && \tag{26}\label{26}
\end{align*}
$$

作者接著推導微分，首先計算 $\pd{\Lc(\xb)}{\eh_t}$：

$$
\begin{align*}
\pd{\Lc(\xb)}{\eh_t} & = \begin{dcases}
  \pd{-\log(e_t)}{e_t} \pd{e_t}{\eh_t}     & \text{if } x_{t+1, 3} = 1 \\
  \pd{-\log(1 - e_t)}{e_t} \pd{e_t}{\eh_t} & \text{otherwise}
\end{dcases} \\
& = \begin{dcases}
  \dfrac{-1}{e_t} \dfrac{-\exp(\eh_t)}{(1 + \exp(\eh_t))^2}    & \text{if } x_{t+1, 3} = 1 \\
  \dfrac{1}{1 - e_t} \dfrac{-\exp(\eh_t)}{(1 + \exp(\eh_t))^2} & \text{otherwise}
\end{dcases} \\
& = \begin{dcases}
  1 - e_t & \text{if } x_{t+1, 3} = 1 \\
  -e_t    & \text{otherwise}
\end{dcases} \\
& = x_{t+1, 3} - e_t. && \tag{27}\label{27}
\end{align*}
$$

推導與 GMM 相關的微分比較複雜，首先定義 $\gammah_t^j$ 輔助推導過程

$$
\begin{align*}
\gammah_t^j & = \pi_t^j \cdot \Nc\pa{x_{t+1} \vert \mu_t^j, \sigma_t^j, \rho_t^j} && \tag{28}\label{28} \\
\gamma_t^j  & = \dfrac{\gammah_t^j}{\Sum_{j' = 1}^M \gammah_t^{j'}}               && \tag{29}\label{29}
\end{align*}
$$

接著推導 $\pd{\Lc(\xb)}{\pih_t^j}$：

$$
\begin{align*}
\pd{\Lc(\xb)}{\pih_t^j} & = \Sum_{i = 1}^M \pa{\pd{-\log \Sum_{j' = 1}^M \gammah_t^{j'}}{\pi_t^i} \pd{\pi_t^i}{\pih_t^j}} \\
& = \Sum_{i = 1}^M \pa{\dfrac{-1}{\Sum_{j' = 1}^M \gammah_t^{j'}} \cdot \pd{\gammah_t^i}{\pi_t^i} \cdot (\delta_{i, j} \pi_t^i - \pi_t^i \pi_t^j)} \\
& = \Sum_{i = 1}^M \pa{\dfrac{-\Nc\pa{x_{t+1} \vert \mu_t^i, \sigma_t^i, \rho_t^i}}{\Sum_{j' = 1}^M \gammah_t^{j'}} \cdot (\delta_{i, j} \pi_t^i - \pi_t^i \pi_t^j)} \\
& = \Sum_{i = 1}^M \pa{\dfrac{-\delta_{i, j} \pi_t^i \Nc\pa{x_{t+1} \vert \mu_t^i, \sigma_t^i, \rho_t^i}}{\Sum_{j' = 1}^M \gammah_t^{j'}}} + \Sum_{i = 1}^M \pa{\dfrac{\pi_t^i \pi_t^j \Nc\pa{x_{t+1} \vert \mu_t^i, \sigma_t^i, \rho_t^i}}{\Sum_{j' = 1}^M \gammah_t^{j'}}} \\
& = -\gamma_t^j + \pi_t^j \\
& = \pi_t^j - \gamma_t^j. && \tag{30}\label{30}
\end{align*}
$$

接著推導 $\Lc(\xb)$ 對於 $\muh_{t, 1}^j, \muh_{t, 2}^j, \sigmah_{t, 1}^j, \sigmah_{t, 2}^j, \rhoh_t^j$ 的微分，由於結構類似，為了方便一起表達成 $\pd{\Lc(\xb)}{(\muh_t^j, \sigmah_t^j, \rhoh_t^j)}$：

$$
\begin{align*}
& \pd{\Nc\pa{x_{t+1} \vert \mu_t^j, \sigma_t^j, \rho_t^j}}{(\muh_t^j, \sigmah_t^j, \rhoh_t^j)} \\
& = \pd{\dfrac{1}{2 \pi \sigma_{t, 1}^j \sigma_{t, 2}^j \sqrt{1 - (\rho_t^j)^2}}}{(\muh_t^j, \sigmah_t^j, \rhoh_t^j)} \cdot \exp\pa{\dfrac{-Z}{2 (1 - (\rho_t^j)^2)}} \\
& \quad + \dfrac{1}{2 \pi \sigma_{t, 1}^j \sigma_{t, 2}^j \sqrt{1 - (\rho_t^j)^2}} \cdot \pd{\exp\pa{\dfrac{-Z}{2 (1 - (\rho_t^j)^2)}}}{(\muh_t^j, \sigmah_t^j, \rhoh_t^j)} \\
& = \dfrac{-\Nc\pa{x_{t+1} \vert \mu_t^j, \sigma_t^j, \rho_t^j}}{2 \pi \sigma_{t, 1}^j \sigma_{t, 2}^j \sqrt{1 - (\rho_t^j)^2}} \cdot \pd{2 \pi \sigma_{t, 1}^j \sigma_{t, 2}^j \sqrt{1 - (\rho_t^j)^2}}{(\muh_t^j, \sigmah_t^j, \rhoh_t^j)} + \Nc\pa{x_{t+1} \vert \mu_t^j, \sigma_t^j, \rho_t^j} \cdot \pd{\dfrac{-Z}{2 (1 - (\rho_t^j)^2)}}{(\muh_t^j, \sigmah_t^j, \rhoh_t^j)} \\
& = -\Nc\pa{x_{t+1} \vert \mu_t^j, \sigma_t^j, \rho_t^j} \cdot \pa{\dfrac{1}{\sigma_{t, 1}^j \sigma_{t, 2}^j \sqrt{1 - (\rho_t^j)^2}} \cdot \pd{\sigma_{t, 1}^j \sigma_{t, 2}^j \sqrt{1 - (\rho_t^j)^2}}{(\muh_t^j, \sigmah_t^j, \rhoh_t^j)} + \pd{\dfrac{Z}{2 (1 - (\rho_t^j)^2)}}{(\muh_t^j, \sigmah_t^j, \rhoh_t^j)}}.
\end{align*}
$$

$$
\begin{align*}
& \pd{\Lc(\xb)}{(\muh_t^j, \sigmah_t^j, \rhoh_t^j)} \\
& = \pd{-\log \Sum_{j' = 1}^M \gammah_t^{j'}}{\gammah_t^j} \pd{\gammah_t^j}{\Nc\pa{x_{t+1} \vert \mu_t^j, \sigma_t^j, \rho_t^j}} \pd{\Nc\pa{x_{t+1} \vert \mu_t^j, \sigma_t^j, \rho_t^j}}{(\muh_t^j, \sigmah_t^j, \rhoh_t^j)} \\
& = \gamma_t^j \cdot \pa{\dfrac{1}{\sigma_{t, 1}^j \sigma_{t, 2}^j \sqrt{1 - (\rho_t^j)^2}} \cdot \pd{\sigma_{t, 1}^j \sigma_{t, 2}^j \sqrt{1 - (\rho_t^j)^2}}{(\muh_t^j, \sigmah_t^j, \rhoh_t^j)} + \pd{\dfrac{Z}{2 (1 - (\rho_t^j)^2)}}{(\muh_t^j, \sigmah_t^j, \rhoh_t^j)}} && \tag{31}\label{31}
\end{align*}
$$

各個微分項次各自展開可以得到以下結果：

$$
\begin{align*}
\pd{\Lc(\xb)}{\muh_{t, 1}^j} & = r_t^j \cdot \pd{\dfrac{Z}{2 (1 - (\rho_t^j)^2)}}{\muh_{t, 1}^j} \\
& = \dfrac{r_t^j}{2 (1 - (\rho_t^j)^2)} \cdot \pa{-2 \dfrac{x_{t+1, 1} - \mu_{t, 1}^j}{(\sigma_{t, 1}^j)^2} + \dfrac{2 \rho_t^j (x_{t+1, 2} - \mu_{t, 2}^j)}{\sigma_{t, 1}^j \sigma_{t, 2}^j}} \\
& = \dfrac{-r_t^j}{\sigma_{t, 1}^j (1 - (\rho_t^j)^2)} \cdot \pa{\dfrac{x_{t+1, 1} - \mu_{t, 1}^j}{\sigma_{t, 1}^j} - \dfrac{\rho_t^j (x_{t+1, 2} - \mu_{t, 2}^j)}{\sigma_{t, 2}^j}} && \tag{32}\label{32} \\
\pd{\Lc(\xb)}{\muh_{t, 2}^j} & = \dfrac{-r_t^j}{\sigma_{t, 2}^j (1 - (\rho_t^j)^2)} \cdot \pa{\dfrac{x_{t+1, 2} - \mu_{t, 2}^j}{\sigma_{t, 2}^j} - \dfrac{\rho_t^j (x_{t+1, 1} - \mu_{t, 2}^j)}{\sigma_{t, 1}^j}} && \tag{33}\label{33} \\
\pd{\sigma_{t, 1}^j \sigma_{t, 2}^j \sqrt{1 - (\rho_t^j)^2}}{\sigmah_{t, 1}^j} & = \pd{\sigma_{t, 1}^j \sigma_{t, 2}^j \sqrt{1 - (\rho_t^j)^2}}{\sigma_{t, 1}^j} \cdot \pd{\sigma_{t, 1}^j}{\sigmah_{t, 1}^j} \\
& = \sigma_{t, 1}^j \sigma_{t, 2}^j \sqrt{1 - (\rho_t^j)^2} \\
\pd{\dfrac{Z}{2 (1 - (\rho_t^j)^2)}}{\sigmah_{t, 1}^j} & = \dfrac{1}{2 (1 - (\rho_t^j)^2)} \pd{Z}{\sigma_{t, 1}^j} \pd{\sigma_{t, 1}^j}{\sigmah_{t, 1}^j} \\
& = \dfrac{1}{2 (1 - (\rho_t^j)^2)} \pa{\dfrac{-2 (x_{t+1, 1} - \mu_{t, 1}^j)^2}{(\sigma_{t, 1}^j)^2} + \dfrac{2 \rho_t^j \pa{x_{t+1, 1} - \mu_{t, 1}^j} \pa{x_{t+1, 2} - \mu_{t, 2}^j}}{\sigma_{t, 1}^j \sigma_{t, 2}^j}} \\
& = \dfrac{-\pa{x_{t+1, 1} - \mu_{t, 1}^j}}{\sigma_{t, 1}^j (1 - (\rho_t^j)^2)} \pa{\dfrac{x_{t+1, 1} - \mu_{t, 1}^j}{\sigma_{t, 1}^j} - \dfrac{\rho_t^j \pa{x_{t+1, 2} - \mu_{t, 2}^j}}{\sigma_{t, 2}^j}} \\
\pd{\Lc(\xb)}{\sigmah_{t, 1}^j} & = r_t^j \cdot \pa{\dfrac{1}{\sigma_{t, 1}^j \sigma_{t, 2}^j \sqrt{1 - (\rho_t^j)^2}} \cdot \pd{\sigma_{t, 1}^j \sigma_{t, 2}^j \sqrt{1 - (\rho_t^j)^2}}{\sigmah_{t, 1}^j} + \pd{\dfrac{Z}{2 (1 - (\rho_t^j)^2)}}{\sigmah_{t, 1}^j}} \\
& = r_t^j \pa{1 - \dfrac{x_{t+1, 1} - \mu_{t, 1}^j}{\sigma_{t, 1}^j (1 - (\rho_t^j)^2)} \pa{\dfrac{x_{t+1, 1} - \mu_{t, 1}^j}{\sigma_{t, 1}^j} - \dfrac{\rho_t^j \pa{x_{t+1, 2} - \mu_{t, 2}^j}}{\sigma_{t, 2}^j}}} && \tag{34}\label{34} \\
\pd{\Lc(\xb)}{\sigmah_{t, 2}^j} & = r_t^j \pa{1 - \dfrac{x_{t+1, 2} - \mu_{t, 2}^j}{\sigma_{t, 2}^j (1 - (\rho_t^j)^2)} \pa{\dfrac{x_{t+1, 2} - \mu_{t, 2}^j}{\sigma_{t, 2}^j} - \dfrac{\rho_t^j \pa{x_{t+1, 1} - \mu_{t, 1}^j}}{\sigma_{t, 1}^j}}} && \tag{35}\label{35} \\
\pd{\sigma_{t, 1}^j \sigma_{t, 2}^j \sqrt{1 - (\rho_t^j)^2}}{\rhoh_t^j} & = \sigma_{t, 1}^j \sigma_{t, 2}^j \pd{\sqrt{1 - (\rho_t^j)^2}}{\rho_t^j} \pd{\rho_t^j}{\rhoh_t^j} \\
& = \sigma_{t, 1}^j \sigma_{t, 2}^j \dfrac{-2 \rho_t^j}{2 \sqrt{1 - (\rho_t^j)^2}} (1 - (\rho_t^j)^2) \\
& = -\sigma_{t, 1}^j \sigma_{t, 2}^j \rho_t^j \sqrt{1 - (\rho_t^j)^2} \\
\pd{\dfrac{Z}{2 (1 - (\rho_t^j)^2)}}{\rhoh_t^j} & = \pd{Z}{\rho_t^j} \pd{\rho_t^j}{\rhoh_t^j} \dfrac{1}{2 (1 - (\rho_t^j)^2)} + \dfrac{Z}{2} \pd{\dfrac{1}{1 - (\rho_t^j)^2}}{\rho_t^j} \pd{\rho_t^j}{\rhoh_t^j} \\
& = -\dfrac{\pa{x_{t+1, 1} - \mu_{t, 1}^j} \pa{x_{t+1, 2} - \mu_{t, 2}^j}}{\sigma_{t, 1}^j \sigma_{t, 2}^j} + Z \dfrac{\rho_t^j}{1 - (\rho_t^j)^2} \\
\pd{\Lc(\xb)}{\rhoh_t^j} & = r_t^j \cdot \pa{\dfrac{1}{\sigma_{t, 1}^j \sigma_{t, 2}^j \sqrt{1 - (\rho_t^j)^2}} \cdot \pd{\sigma_{t, 1}^j \sigma_{t, 2}^j \sqrt{1 - (\rho_t^j)^2}}{\rhoh_t^j} + \pd{\dfrac{Z}{2 (1 - (\rho_t^j)^2)}}{\rhoh_t^j}} \\
& = r_t^j \pa{-\rho_t^j - \dfrac{\pa{x_{t+1, 1} - \mu_{t, 1}^j} \pa{x_{t+1, 2} - \mu_{t, 2}^j}}{\sigma_{t, 1}^j \sigma_{t, 2}^j} + Z \dfrac{\rho_t^j}{1 - (\rho_t^j)^2}} \\
& = r_t^j \pa{-\dfrac{\pa{x_{t+1, 1} - \mu_{t, 1}^j} \pa{x_{t+1, 2} - \mu_{t, 2}^j}}{\sigma_{t, 1}^j \sigma_{t, 2}^j} + \rho_t^j \pa{\dfrac{Z}{1 - (\rho_t^j)^2} - 1}} \tag{36}\label{36}
\end{align*}
$$

### 實驗 3：IAM-OnDB

<a name="paper-fig-5"></a>

圖 5：手寫辨識訓練結果。
圖片來源：[論文][論文]。
![圖 5](https://i.imgur.com/NEM0awl.png)

<a name="paper-fig-6"></a>

圖 6：手寫輸入為 *under* 時模型的預測分佈。
圖片來源：[論文][論文]。
![圖 6](https://i.imgur.com/LBVkors.png)

- 此論文所有與手寫文字相關的資料都是來自於 IAM-OnDB 資料集
  - 資料來源為 $221$ 位不同的寫作者在智慧白板（smart whiteboard）上的筆跡
  - 所有寫作者都被要求在智慧白板上寫下 Lancaster-Oslo-bergen text corpus
  - 所有的筆跡都是由智慧白板角落配備的紅外線感應裝置進行偵測與紀錄
  - 缺少的紀錄數值都用內插法（interpolation）進行填補
  - 超過特定長度的手寫數值序列會被從資料集中移除
  - 平均一個字母有 $25$ 個座標值，白板上的一行文字平均有 $700$ 個座標值
  - Training set 有 $5364$ 行
  - 共有兩個 validation sets，分別有 $1438$ 與 $1518$ 行
    - 作者把比較大的 validation set 作為 training data 使用
    - 作者把比較小的 validation set 作為 early-stopping 的評估手段
  - Test set 有 $3859$ 行
- 預測結果將會作為下個時間點的輸入，概念跟 language model 一樣
  - 此論文是第一篇論文採用 language model 的作法模擬與生成手寫文字
  - 一個字母的最後預測目標為 end-of-stroke
  - 與大部分的方法不同，LSTM 沒有任何特殊的前處理，只有簡單的將座標值 normalize 成平均值為 $0$ 標準差為 $1$
- 模型架構
  - 共有 $20$ 個 mixture models（$M = 20$）
    - 每個時間點包含 $20$ 個組合權重 $\pi_t^1, \dots, \pi_t^{20}$
    - 每個時間點包含 $40$ 個平均值 $\mu_{t, 1}^1, \mu_{t, 2}^1, \dots, \mu_{t, 1}^{20}, \mu_{t, 2}^{20}$
    - 每個時間點包含 $40$ 個標準差 $\sigma_{t, 1}^1, \sigma_{t, 2}^1, \dots, \sigma_{t, 1}^{20}, \sigma_{t, 2}^{20}$
    - 每個時間點包含 $20$ 個相關係數 $\rho_t^1, \dots, \rho_t^{20}$
    - 每個時間點共有 $120$ 個參數用於模擬座標值
  - 每個時間點都有 $1$ 個 end-of-stoke 機率值，因此每個時間點共有 $121$ 個參數用於模擬 $x_{t+1}$
  - 實驗兩種大小不同的 LSTM
    - 版本 1：$3$ 層 LSTM（$N = 3$），隱藏層單元數為 $400$（$h_t^n \in \R^{400}$）
    - 版本 2：$1$ 層 LSTM（$N = 1$），隱藏層單元數為 $900$（$h_t^1 \in \R^{900}$）
    - 兩個版本的總參數量約為 $3.4$ M
    - 版本 1 有額外使用 adaptive weight noise，std 初始化為 $0.075$
    - 實驗證實使用 fixed weight noise 對模型沒有幫助，因此不採用
- 最佳化演算法為 RMSProp，公式定義如下

  $$
  \begin{align*}
    n_i & = \aleph n_{i - 1} + (1 - \aleph) \pa{\pd{\Lc(\xb)}{w_i}}^2 \\
    g_i & = \aleph g_{i - 1} + (1 - \aleph) \pd{\Lc(\xb)}{w_i} \\
    \Delta_i & = \lambda \Delta_{i - 1} - \eta \dfrac{\pd{\Lc(\xb)}{w_i}}{\sqrt{n_i - g_i^2 + \epsilon}} \\
    w_i & = w_{i - 1} + \Delta_i
  \end{align*}
  $$

  - 定義 $w_i$ 為參數
  - $\aleph = 0.95$
  - $\lambda = 0.9$
  - $\eta = 0.0001$
  - $\epsilon = 0.0001$
  - Gradient clipping 的範圍為 $[-10, 10]$
  - 針對 $\pd{\Lc(\xb)}{\yh_t}$，gradient clipping 的範圍為 $[-100, 100]$，實驗證實此舉能夠穩定數值計算
  - 當模型開始 overfitting 時仍然會出現數值問題
- [圖 5](#paper-fig-5) 中紀錄了模型平均預測誤差（以對數呈現，愈低愈好），平均值的計算方法區分成以序列為資料點或以每個時間點的座標為資料點
  - 以序列為資料點時，$3$ 層 LSTM 模型的平均預測誤差低於 $1$ 層 LSTM 模型
  - 以每個時間點的座標為資料點時，$3$ 層 LSTM 模型的平均預測誤差高於 $1$ 層 LSTM 模型
  - 使用 adaptive weight noise 能夠降低以序列為資料點的平均預測誤差
  - 使用 adaptive weight noise 無法降低以每個時間點的座標為資料點的平均預測誤差
- [圖 6](#paper-fig-6) 的上半部是預測手寫文字 *under* 的筆跡時畫出的座標值機率分佈
  - 觀察發現預測分佈出現兩種不同的泡泡（blobs）
  - 小的泡泡是單一字母內的連續筆跡，由於下一個的座標有主要軌跡，造成標準差 $\sigma_{t, 1}^j, \sigma_{t, 2}^j$ 較小
  - 大的泡泡出現在字的結尾，由於下一個字母的起始座標比較沒有預測邏輯，造成標準差 $\sigma_{t, 1}^j, \sigma_{t, 2}^j$ 較大
- [圖 6](#paper-fig-6) 的下半部是預測手寫文字 *under* 的筆跡時 $\pi_t^j$ 的權重分佈
  - 觀察發現 end-of-stroke 所使用的權重分佈與單一字母中使用的權重分佈不同
- 從生成結果來看能夠生成出 end-of-stroke 以及一些常用字（*of*、*the* 等）
  - 由於一個字母平均由 $25$ 個座標點組合而成，因此作者認為這證實 LSTM 擁有模擬長距離資訊的能力
  - 作者沒有給出具體的生成方法，推測是將輸出的平均值 $\mu_{t, 1}^j \mu_{t, 2}^j$ 以 $\pi_t^j$ 進行加權平均當成下個時間點的座標點

## 透過文字序列生成手寫字跡

<a name="paper-fig-7"></a>

圖 7：文字需列生成手寫字跡模型架構。
圖片來源：[論文][論文]。
![圖 7](https://i.imgur.com/TENhiJm.png)

<a name="paper-fig-8"></a>

圖 8：驗證模型學會文字與筆跡間的對應關係（alignment）。
圖片來源：[論文][論文]。
![圖 8](https://i.imgur.com/bgJf57n.png)

作者嘗試讓模型在獲得文字輸入的狀態下生成手寫字跡。
當時並沒有 sequence-to-sequence 架構，並且資料集也沒有標記文字與筆跡之間的對應關係（alignment），因此作者基於 LSTM + GMM 提出了新架構。
從現在（2022）的角度來看，此架構就是 encoder + decoder，並且擁有類似 attention 的計算架構。

首先定義 window $w_t$，功能就是 encoder，目標為將文字序列的所有資訊輸入給 LSTM 幫助生成手寫字跡。
令 $\cb = (c_1, \dots, c_U)$ 為文字序列，每個 $c_u$ 代表一個文字對應的 one-hot-encoding。
令 $\xb = (x_1, \dots, x_T)$ 為座標值序列，每個 $x_t$ 定義如同 $\eqref{15}$。
則 $w_t$ 的計算方法如下

$$
\begin{align*}
\phi(t, u) & = \Sum_{k = 1}^K \alpha_t^k \exp\pa{-\beta_t^k \pa{\kappa_t^k - u}^2} && \tag{37}\label{37} \\
w_t & = \Sum_{u = 1}^U \phi(t, u) c_u && \tag{38}\label{38}
\end{align*}
$$

- 共有 $K$ 個 Gaussian distribution
- $\phi(t, u)$ 代表文字 $c_u$ 對於時間點 $t$ 的預測重要程度（window weight of $c_u$ at time step $t$）
  - 這代表 $\phi(t, u)$ 應該要是一個非負實數，甚至應該要加上 $\Sum_{u = 1}^U \phi(t, u) = 1$ 的限制
  - 作者在這篇論文中並沒有加上總和為 $1$ 的限制
- $\pa{\kappa_t^k - u}^2$ 代表距離（window location）
  - 距離愈大數值就愈大
  - 此項次的目標是希望距離愈遠的文字影響力愈小
  - 這代表 $\kappa_t^k$ 應該要是一個非負實數
- $\beta_t^k$ 代表距離的影響力（window width）
  - 數值愈大影響力愈小，理由是會將計算結果取負號後在取指數
  - 這代表 $\beta_t^k$ 應該要是一個非負實數
- $\alpha_t^k$ 代表 GMM 的結合權重
  - 這代表 $\alpha_t^k$ 應該要是一個非負實數
- $w_t$ 的維度與 $c_u$ 的維度相同

接著定義 $\eqref{37} \eqref{38}$ 的符號細節

$$
\begin{align*}
p & = \set{\alphah_t^k, \betah_t^k, \kappah_t^k}_{k = 1}^K = W_{h^1 p} h_t^1 + b_p && \tag{39}\label{39} \\
\alpha_t^k & = \exp(\alphah_t^k) && \tag{40}\label{40} \\
\beta_t^k & = \exp(\betah_t^k) && \tag{41}\label{41} \\
\kappa_t^k & = \kappa_{t-1}^k + \exp(\kappah_t^k) && \tag{42}\label{42} \\
h_t^1 & = \Hc\pa{W_{i h^1} x_t + W_{h^1 h^1} h_{t-1}^1 + W_{w h^1} w_{t-1} + b_h^1} && \tag{43}\label{43} \\
h_t^n & = \Hc\pa{W_{i h^n} x_t + W_{h^{n-1} h^n} h_t^{n-1} + W_{h^n h^n} h_{t-1}^n + W_{w h^n} w_t + b_h^n} && \tag{44}\label{44} \\
\Lc(\xb) & = -\log \Pr(\xb \vert \cb) && \tag{45}\label{45} \\
\Pr(\xb \vert \cb) & = \Prod_{t = 1}^T \Pr(x_{t+1} \vert y_t) && \tag{46}\label{46}
\end{align*}
$$

- $p \in \R^{3K}$，$y_t$ 的定義沿用 $\eqref{17} -- \eqref{22}$
- $\eqref{42}$ 的定義是為了讓模型能夠模擬 sliding window 的概念
- $\eqref{43} \eqref{44}$ 是修改 $\eqref{1} \eqref{2}$ 而得，架構見[圖 7](#paper-fig-7)
- 由於 $h_t^1$ 用於建立 $w_t$，因此 $h_t^1$ 會收到 $w_{t-1}$ 而不是 $w_t$
- 由於 $h_t^n$ 的輸出是基於 encoder 資訊，因此 $h_t^n$ 會收到 $w_t$ 而不是 $w_{t-1}$

接下來的微分推導我就跳過了，因為很懶，先相信他是對的。

### 實驗 4：IAM-OnDB

<a name="paper-fig-9"></a>

圖 9：給予文字預測手寫軌跡的訓練結果。
圖片來源：[論文][論文]。
![圖 9](https://i.imgur.com/h7VV4Zp.png)

<a name="paper-fig-10"></a>

圖 10：給予文字 *under* 時預測手寫軌跡的機率分佈。
圖片來源：[論文][論文]。
![圖 10](https://i.imgur.com/bbrhMg7.png)

- 資料集一樣是 IAM-OnDB
- 將 token 定義為 character
  - 區分大小寫，包含標點符號與數字
  - 共有 $80$ 個不同的符號，但作者只使用 $57$ 個符號，將大多數數字與標點符號換成 non-letter
- 模型架構
  - 預測座標一共使用 $20$ 個 mixture models（$M = 20$）
  - Encoder 一共使用 $10$ 個 mixture models（$K = 10$）
  - $w_t \in \R^{57}$
  - $3$ 層 LSTM（$N = 3$），隱藏層單元數為 $400$（$h_t^n \in \R^{400}$）
  - 總參數量約為 $3.7$ M
- 最佳化演算法為 RMSProp，設定與實驗 3 完全相同
- 根據[圖 9](#paper-fig-9)
  - 以序列為資料點時，使用 adaptive weight noise 表現較好
  - 以每個時間點的座標為資料點時，使用 adaptive weight noise 沒有差別
  - 相比於[圖 5](#paper-fig-5)，作者認為實驗 4 表現較好
- 比較[圖 10](#paper-fig-10) 與[圖 6](#paper-fig-6)，可以發現預測的信心度上升導致圓形泡泡都縮小

## Unbiased / Biased Sampling

首先定義 unbiased sampling 的概念：
生成一個座標序列 $x_1, x_2, \dots, x_T$ 時，最佳解為從機率分佈中挑選機率值最高的答案。
但由於排列組合數過高，因此實行上效率很低。
作者便採用每個時間點獨立挑選機率值最高的答案作為近似解。
而生成結束的判斷依據則是當生成 end-of-sequence（index 定義成 $U + 1$）的機率值高於其他 token 時則結束，即滿足以下條件

$$
\forall 1 \leq u \leq U, \phi(t, U + 1) > \phi(t, u).
$$

而作者認為在 inference 階段可以人為刻意讓生成的機率值偏向特定 Gaussian distribution，甚至讓 Gaussian distribution 的 std 變小確保生成差異性變小，概念就像是生成「平均手寫結果」。
作者稱此方法為 biased sampling，具體上採用的公式如下（修改 $\eqref{19} \eqref{21}$ 而得）：

$$
\begin{align*}
\sigma_t^j & = \exp\pa{\sigmah_j^t - b}; \\
\pi_t^j & = \dfrac{\exp(\pih_t^j (1 + b))}{\Sum_{j' = 1}^M \exp\pa{\pih_t^{j'} (1 + b)}}.
\end{align*}
$$

其中 $b \in [0, \infty)$.
當 $b = 0$ 時等同於採用 unbiased sampling，當 $b \to \infty$ 時等同於 std 為 0 且永遠採用一個 Gaussian distribution。

根據生成範例可以發現，使用 biased sampling 的結果真的能夠讓生成差異變小，而在 $b = 0.15$ 時能夠在維持生成多樣性與減少差異達到不錯的平衡點。

## Primed Sampling

作者認為如果想要指定生成特定寫作者的字跡，最簡單的作法就是讓模型只訓練在該作者的字跡資料上。
而作者額外提出以特殊符號代表寫作者，並加入至 encoder 的輸入 $\cb$ 之中，如此就可以訓練一個模型模擬所有寫作者的風格。
此方法稱為 primed sampling，根據生成範例證實此方法可行。

[LSTM1997]: https://ieeexplore.ieee.org/abstract/document/6795963
[LSTM2002]: https://www.jmlr.org/papers/v3/gers02a.html
[論文]: https://arxiv.org/abs/1308.0850
