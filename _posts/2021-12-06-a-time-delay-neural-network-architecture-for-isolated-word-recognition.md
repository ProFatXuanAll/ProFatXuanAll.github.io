---
layout: ML-note
title: "A time-delay neural network architecture for isolated word recognition"
date: 2021-12-06 12:28:00 +0800
categories: [
  Acoustic Modeling,
]
tags: [
  RNN,
  gradient descent,
  model architecture,
  neural network,
  note-is-under-construction,
]
author: [
  Kevin J.Lang,
  Alex H.Waibel,
  Geoffrey E.Hinton,
]
---

|-|-|
|目標|提出 Time-delay Neural Network，在 IBM `BDEV` 語音辨識資料集上達到最好表現，並且不需要多餘的前處理|
|作者|Kevin J.Lang, Alex H.Waibel, Geoffrey E.Hinton|
|期刊/會議名稱|Neural Networks|
|發表時間|1990|
|論文連結|<https://www.sciencedirect.com/science/article/pii/089360809090044L>|

<!--
  Define LaTeX command which will be used through out the writing.

  Each command must be wrapped with $ signs.
  We use "display: none;" to avoid redudant whitespaces.
 -->

<p style="display: none;">

  <!-- Sequence. -->
  $\providecommand{\seq}{}$
  $\renewcommand{\seq}[2]{u_{#1}, \dots, u_{#2}}$

</p>

<!-- End LaTeX command define section. -->

## 重點

- 認為 **HMM**（**Hidden Markov Model**）在執行語音辨識時有很多缺點
  - 必須要將代表語音的實數向量先轉換成特定的語音類別，再以類別作為 HMM 的輸入才能進行運算
  - Markov assumption 還有輸出之間互相獨立的假設過於簡單
- Peter Brown 針對 IBM 發表的 HMM 語音辨識模型提出以下結論
  - 如果輸入可以使用連續機率分佈進行模擬會更好
  - 針對語音輸入與輸出文字間的**相互知識**（**Mutual Information**）必須要最大化
  - 這代表需要擁有辨別輸入語音的功能，而不是直接把輸入規類成不同類別進行運算
  - 當語音的字母結尾是 E 時模型造成的誤差最高，因為這些語音的發音時間較短且聲音較小（short in duration and low in energy）
  - 最後 Peter Brown 使用 **GMM**（**Gaussian Mixure Models**） 並針對相互知識最佳化，將誤差降至 IBM 模型的一半以下
- 此論文認為上述的問題都可以使用 Neural Network 解決
  - 輸入是代表語音訊號的實數向量
  - 輸出是 $n$ 個文字的類別預測機率值
  - 預測目標是維度為 $n$ 的 one-hot vector
- 最後提出的 Time-delay Neural Network 能夠達到 IBM `BDEV` 語音辨識的最佳表現
  - 不使用如同 IBM HMM 的 Viterbi Alignment 進行前處理，而是直接輸入原始資料

## IBM 語音辨識資料集

- 由 IBM T. J. Watson Research Center 提供
- 蒐集過程
  - 使用 remote pressure-zone microphone 在辦公室錄音
  - 12 bits A/D converter running at 20000 Hz
  - 錄製每個字母的發音
  - 每個字母由 100 個不同的人講，每個人會唸兩次同一個字母，兩次發音分別用來訓練與測試
  - 錄音者需要念出三個句子，每個句子是由字母隨機組成，字母之間有空白，錄音者被要求在空白上停頓
- `BDEV` 這四個字母發音特別難分辨
  - 發音分別是 `bee, dee, ee, vee`
  - 共有 $372$ 筆訓練資料、$396$ 筆測試資料
  - 時間長度介於 $[0.3, 6.4]$ 秒，平均 $1.1$ 秒
- 雜訊比（signal-to-noise ratio）為 $16.4$ dB，計算方法如下
  - 使用 HMM 標記哪些聲音片段是發音，哪些是背景雜訊
  - 將發音（包含母音與子音）的平均訊號強度（分貝）除以背景雜訊的平均訊號強度
  - 常見的 lip-mike 語音辨識資料集的雜訊比是 $50$ dB
- 人類對於 `BDEV` 的辨識率為 $94\%$
  - 但經過 IBM 的訊號前處理與重建後降到 $75\%$
- 模型對於 `BDEV` 的辨識率
  - IBM 提出的 HMM 辨識率為 $80\%$
  - Peter Brown 提出的 GMM 辨識率為 $89\%$
- 使用 IBM HMM 加上 Viterbi Alignment 將每一幀訊號資料進行標記
  - 這裡的細節看不太懂
  - 每筆資料的時間長度為 $150$ ms
- 前處理
  - 將 20000 Hz 降低抽樣頻率成 16000 Hz
  - 使用 CMU 開發的 makedft 將聲音訊號轉成頻譜
  - 這裡的細節看不太懂
  - 每筆資料變成 $48$ 幀，每幀的時間長度為 $3$ ms，每幀共有 $128$ 個資料點
- 後處理
  - 由於輸入共有 $48 \cdot 128 = 6144$ 個數值，將所有輸入配合全連接模型所需參數至少大於 $6144$
  - 但訓練資料只有 $372$ 筆，因此必須要讓模型參數縮小
  - 使用 mel spetrogram 合併頻譜降低頻譜維度，每筆資料變成每幀 $16$ 個資料點，共有 $6$ 幀
  - 將低於 $-5$ dB 的數值設為 $-5$ dB
  - 將超過 $105$ dB 的數值設為 $105$ dB
  - 將數值 normalize 到 $[0, 1]$ 之間，採用了四種不同的數值轉換方法，分別為
    - 除以最大值
    - 使用 sigmoid 轉換
    - 使用 sigmoid 轉換後乘上 $1.4$
    - 什麼都不做
  - 輸入與輸出全連接模型配合後處理得到最好的表現為 $86\%$ 的預測準確度
    - melscaled frequency bands
    - global energy normalization
    - input values reshaped by squaring
- 針對幀數進行實驗
  - 幀數愈高，同時間的輸入愈多
  - 幀數落在 $\set{3, 6, 12, 24}$ ms
  - 當參數數量太多時，使用 weight decay 進行 regularization
  - 實驗證實一幀 $12$ ms 表現最好

## 最佳化

$$
\begin{align*}
\operatorname{loss} & = \operatorname{MSE} + \delta \cdot \norm{W(t)} \\
W(t + 1) & = W(t) - \varepsilon \cdot \pd{\operatorname{loss}}{W(t)} + \alpha \cdot \pd{\operatorname{loss}}{W(t - 1)}
\end{align*}
$$

- 使用 batch back-propagation 進行最佳化
  - 總梯度等於每筆資料的梯度進行相加
  - 不除以總資料數作 normalization
  - 總共 train 1000 個 epoch
- Weight decay factor $\delta = 0.001$
- 模型剛開始訓練時
  - Learning rate 為 $\varepsilon = 0.001$
  - First momentum $\alpha = 0.5$
- 模型訓練 $50$ 個 epochs 後
  - Learning rate 為 $\varepsilon = 0.001$
  - First momentum $\alpha = 0.9$
- 模型訓練 $100$ 個 epochs 後
  - Learning rate 為 $\varepsilon = 0.002$
  - First momentum $\alpha = 0.95$
- 模型訓練 $200$ 個 epochs 後
  - Learning rate 為 $\varepsilon = 0.005$
  - First momentum $\alpha = 0.95$
- 每 $200$ 個 epoch 就執行一次測試
  - 評估方法為 accuracy
  - 預測正確答案的機率必須大於 $0.5$ 才算預測正確
  - 取表現最好 checkpoint 的作為最終模型 checkpoint

## 模型架構

### 版本 1：輸入輸出全連接

- 輸出層維度為 $4$
  - 分別代表 `BDEV`
- 輸入層維度為 $12 \times 16 = 192$
  - 輸入層與輸出層全連接
  - 每個輸出節點都有使用 bias
  - 共有 $4 \times 192 + 4 = 772$ 個參數
- 在 Convex C-1 上執行需要花 $5$ 分鐘
- 準確度
  - 在訓練集上為 $93\%$
  - 在測試集上為 $86\%$
- 論文太舊了根本看不清楚圖片

### 版本 2：增加隱藏層

- 將 $25$ 筆測試資料變成訓練資料
  - 這個動作讓模型有更多資料可以訓練，可以探索比較複雜的模型
  - 但也讓前面的實驗無法比較
- 基於版本 1，加入隱藏層，維度為 $4$
  - 隱藏層與輸入層全連接
    - 有使用 bias
    - 共有 $4 \times 192 + 4 = 772$ 個參數
  - 輸出層與輸入層全連接
    - 有使用 bias
    - 共有 $4 \times 4 + 4 = 20$ 個參數
  - 共有 $772 + 20 = 792$ 個參數
- 與版本 1 的比較
  - 訓練時間更長
  - 在訓練資料上誤差較低
  - 在測試資料上雖然 MSE 較低，但準確度不變

### 版本 3：擴增隱藏層

- 基於版本 2，隱藏層維度變成 $8$
- 在訓練集上可以達到 $99\%$ 的準確度
  - 但在測試集上只有 $89\%$ 的準確度，作者認為模型 overfitting

### 版本 4：Receptive Fields

> According to the standard intuitive explanation of the behavior of multilayer feed-forward networks, hidden units are supposed to extract meaningful features from the input patterns.

要給 reference 的來源阿！！
該不會是你自己說的話自己稱為 standard？？

- 基於版本 3，但每個隱藏單元只會接收 $3$ 幀的輸入單元
  - 總共有 $12$ 幀，每幀 $16$ 個輸入
  - 一個隱藏單元會收到 $3 \times 16$ 個輸入
  - 總共有 $10$ 個 $3$ 幀的組合，因此至少要有 $10$ 個隱藏單元
  - 作者決定每個 $3$ 幀的組合要與 $3$ 個隱藏單元全連接，因此總共有 $30$ 個隱藏單元
  - 共有 $30 \times 3 \times 16 = 1440$ 個參數
- 隱藏層與輸出層全連接
  - 共有 $4 \times 30 = 120$ 個參數
- 總共有 $1440 + 120 = 1560$ 個參數
- 與版本 3 相比只有進步一點點
  - 但使用版本 4 的架構去做子音判斷任務很有效
  - 這個神來一筆我也是覺得？？？

### 版本 5：解決 Misalignment

- 由於前處理是靠 HMM 生成資料，有可能資料的標記順序有誤
- 版本 4 的模型強制隱藏單元接收相同位置的輸入
  - 如果輸入標記錯誤，則隱藏單元可能學到錯誤的資訊
- 基於版本 4，將接收不同輸入單元的 $10$ 個隱藏單元所連接的權重取平均值，並替換原本的權重
  - 以此減少標記錯誤導致的影響
  - 進行平均權重的步驟只在訓練結束後進行
  - 一樣產生 $30$ 個隱藏單元，計算方法如版本 4，只是所有權重共享
- 額外增加每 $3$ 幀對應到的隱藏單元數量
  - 版本 4 只有 $3$ 個隱藏單元
  - 增加到 $\set{4, 6, 8}$ 個
  - $8$ 個的版本表現最好

### 版本 6：多組輸出

- 基於版本 5，但不使用隱藏層，可以觀察**輸出隨著時間的變化**
  - 每 $5$ 幀輸入與輸出對應到 $4$ 個輸出單元，共產出 $8$ 組輸出，每組包含 $4$ 個輸出單元
  - 總輸出定義成 $8 \times 4$ 個數字各自取平方後沿著 $8$ 的維度加總，因此又回到只有 $4$ 個數出單元
  - 在訓練結束後進行進行平均權重
- 認為使用回饋的機制後，分析變得困難
  - 沒有學習目標就無法判斷成效
- 準確度
  - 在訓練集上為 $94\%$
  - 在測試集上為 $91\%$

### 版本 7：多組輸出加上隱藏層

- 基於版本 6，但使用隱藏層
  - 每 $3$ 幀輸入對應到 $8$ 個隱藏單元，共產生 $10 \times 8$ 個隱藏單元
  - 每 $5$ 組隱藏單元（一組 $8$ 個，共 $5\times 8$ 個隱藏單元)
  對應到 $4$ 個輸出，共有 $6 \times 4$ 組輸出
  - 總輸出同版本 6 定義
  - 在訓練結束後進行進行平均權重
- 比較難訓練
  - 需要跑 $20000$ 個 epochs
  - Learning rate $\varepsilon = 0.001$
  - First momentum $\alpha = 0.95$
- 準確度
  - 在訓練集上為 $93\%$
  - 在測試集上為 $93\%$

### 版本 8：Time-delay Neural Networks

- 維度
  - 輸入層維度為 $16$，一次讀取 $1$ 幀
  - 隱藏層維度為 $8$
  - 輸出層維度為 $4$
- 輸入層與隱藏層的連接方法需要考慮時間差
  - 每個輸入單元會與每個隱藏單元有 $3$ 個連接方式，每個連接方式代表時間差，時間差為 $\set{0, 1, 2}$
  - 連接架構
    - 時間點為第 $k$ 幀時，所有隱藏單元使用第 $1$ 組參數與第 $k$ 幀全連接，代表第 $k$ 幀輸入時間差為 $0$ 的貢獻
    - 時間點為第 $k$ 幀時，所有隱藏單元使用第 $2$ 組參數與第 $k - 1$ 幀全連接，代表第 $k - 1$ 幀輸入時間差為 $1$ 的貢獻
    - 時間點為第 $k$ 幀時，所有隱藏單元使用第 $3$ 組參數與第 $k - 2$ 幀全連接，代表第 $k - 2$ 幀輸入時間差為 $2$ 的貢獻
  - 舉例
    - 時間點為第 $1$ 幀時，所有隱藏單元使用第 $1$ 組參數與第 $1$ 幀全連接，代表第 $1$ 幀輸入時間差為 $0$ 的貢獻
    - 時間點為第 $2$ 幀時，所有隱藏單元使用第 $1$ 組參數與第 $2$ 幀全連接，代表第 $2$ 幀輸入時間差為 $0$ 的貢獻
    - 時間點為第 $2$ 幀時，所有隱藏單元使用第 $2$ 組參數與第 $1$ 幀全連接，代表第 $1$ 幀輸入時間差為 $1$ 的貢獻
    - 時間點為第 $3$ 幀時，所有隱藏單元使用第 $1$ 組參數與第 $3$ 幀全連接，代表第 $3$ 幀輸入時間差為 $0$ 的貢獻
    - 時間點為第 $3$ 幀時，所有隱藏單元使用第 $2$ 組參數與第 $2$ 幀全連接，代表第 $2$ 幀輸入時間差為 $1$ 的貢獻
    - 時間點為第 $3$ 幀時，所有隱藏單元使用第 $3$ 組參數與第 $1$ 幀全連接，代表第 $1$ 幀輸入時間差為 $2$ 的貢獻
- 隱藏層與輸出層的連接方法也需要考慮時間差
  - 每個隱藏單元會與每個輸出單元有 $5$ 個連接方式，每個連接方式代表時間差，時間差為 $\set{0, 1, 2, 3, 4}$
  - 連接架構
    - 時間點為第 $k$ 幀時，所有隱藏單元使用第 $1$ 組參數與第 $k$ 幀全連接，代表第 $k$ 幀輸入時間差為 $0$ 的貢獻
    - 時間點為第 $k$ 幀時，所有隱藏單元使用第 $2$ 組參數與第 $k - 1$ 幀全連接，代表第 $k - 1$ 幀輸入時間差為 $1$ 的貢獻
    - 時間點為第 $k$ 幀時，所有隱藏單元使用第 $3$ 組參數與第 $k - 2$ 幀全連接，代表第 $k - 2$ 幀輸入時間差為 $2$ 的貢獻
    - 時間點為第 $k$ 幀時，所有隱藏單元使用第 $4$ 組參數與第 $k - 3$ 幀全連接，代表第 $k - 3$ 幀輸入時間差為 $3$ 的貢獻
    - 時間點為第 $k$ 幀時，所有隱藏單元使用第 $5$ 組參數與第 $k - 4$ 幀全連接，代表第 $k - 4$ 幀輸入時間差為 $4$ 的貢獻
- 證實在日文語音辨識的任務表現最佳
  - 阿不給當前實驗的數據是怎樣

### 版本 9：Multiresolution Training

- 基於版本 8，進行參數調整
  - 時間差一律變成 $\set{0,1,2,3}$
  - 隱藏層維度改成 $6$
  - 隱藏層與輸出層有使用 bias
  - 總共有 $6 \times 4 \times 16 + 6 + 4 \times 4 \times 6 + 4 = 490$ 個參數
- 額外建構一幀為 $24$ ms 的模型
  - 輸入變成只有 $6$ 幀
  - 由於一幀包含 $2$ 個 $12$ ms，因此時間差變成 $\set{0,1}$
- 當訓練完一幀為 $24$ ms 的模型，將模型的參數用來初始化一幀為 $12$ ms 的模型
  - 訓練的停止條件為訓練資料準確度達到 $85\%$
  - 達成停止條件需要 $3000$ epochs
  - 起始參數
    - Learning rate $\varepsilon = 0.0001$
    - First momentum $\alpha = 0.05$
  - 第 $200$ 個 epochs
    - Learning rate $\varepsilon = 0.0001$
    - First momentum $\alpha = 0.9$
  - 第 $1000$ 個 epochs
    - Learning rate $\varepsilon = 0.0005$
    - First momentum $\alpha = 0.95$
  - 第 $2000$ 個 epochs
    - Learning rate $\varepsilon = 0.001$
    - First momentum $\alpha = 0.95$
- 所有參數初始化的數值都是落在 $(-0.01, 0.01)$
- 訓練資料從版本 2 的比例回到正常
- 使用 $\set{0, 1}$ 作為預測目標而不是 $\set{0.2, 0.8}$
- 模型最好表現落在第 $10000$ 個 epochs
  - 訓練準確率為 $95.4\%$
  - 測試準確率為 $91.4\%$
- 額外訓練 $10000$ 個 epochs（代表總共訓練了 $20000$ 個 epochs）造成 overfitting
  - 訓練準確率為 $98.1\%$
  - 測試準確率為 $88.1\%$

### 版本 10：更改前處理

- 不再使用 Viterbi alignment 前處理的版本，直接針對原始資料進行訓練
- 假設所有聲音的能量都集中在母音，而分類的主要依據是子音與母音之間的轉換方法
  - 訓練時使用資料必須要先找到與母音（最大能量）差距最大的能量（當成子音轉換成母音的過程）的位置
  - 每 $3$ ms 作為一個區間，區間中的所有數值減去區間最小值
  - 以 $150$ ms 作為一個區間進行 smoothing，方法為取中位數作為代表
  - Smoothing 後的結果取最大的駝峰代表母音，並將駝峰的前後 $150$ ms 一起納入分析
  - 在分析的範圍中找出最大值與最小值的差距，並找出最大值座落的時間點
  - 若最大值座落的時間點為 $d$，則訓練資料的時間區間為 $[d - 120, d - 120 + 216]$
  - 輸入資料長度為 $216$ ms，比起 $144$ ms 多了 $50\%$
- 為了讓模型能夠學會處理雜訊，額外在資料中加入了雜訊片斷
  - 每筆資料只包含雜訊（從每一筆不是前述資料片段的剩餘片段取出）
  - 每筆長度為 $216$ ms 的資料
  - 預測目標設定為所有類別都輸出為 $0$
- 測試時並不是像訓練時做複雜的前處理
  - 將每一筆測試資料以 sliding window 的方式切成多個長度為 $216$ ms 的片段
  - Sliding window 一次移動 $12$ ms
  - 所有輸出向量中，包含最大數值的向量作為最終預測的類別
  - 如果是使用 Time-delay Neural Network 則輸出會以版本 6 的形式計算
- 使用兩種不同模型架構進行實驗
  - 全連接模型
    - 輸入維度為 $18 \times 16$
    - 隱藏層維度為 $8$
    - 輸出維度為 $4$
    - 有使用 bias
    - 總共有 $18 \times 16 \times 8 + 8 + 8 \times 4 + 4 = 2348$ 個參數（論文寫 $2358$ 應該是 typo）
    - 總共訓練兩次
      - 第一次不使用雜訊進行訓練
      - 第二次使用雜訊進行訓練
    - 不管是哪次訓練，使用的訓練參數都相同
      - Learning rate $\varepsilon = 0.0005$
      - First momentum $\alpha = 0.95$
    - 在 $400$ 個 epochs 就達到最好表現
    - 在訓練資料集上的評估方法有兩種
      - 使用有前處理的資料，並以預測最大值作為答案
      - 使用原始資料，並使用同測試資料的評估方法
  - Time-delay Neural Network
    - 運算與最佳化方法請參考版本 9
    - 輸入以 $84$ ms 為一幀，一次 sliding window 移動為 $11$ ms
      - 例如第一幀為 $[0, 84]$ ms，第二幀為 $[11, 95]$ ms
      - 最後一幀為 $[205, 216]$ ms
    - 共產生 $12$ 組輸出，每組有 $4$ 個數值
    - 總共訓練兩次
      - 第一次不使用雜訊訓練小模型，並用小模型初始化大模型的參數
      - 第二次使用有包含雜訊的資料訓練大模型
- 實驗結論
  - 使用 Time-delay Neural Network 可以讓只用無雜訊、有前處理版的資料表現與無雜訊、不做任何處理的資料表現相似
    - 全連接模型無法做到
    - 同樣的結論也發生在有加入雜訊的資料集中
  - 在有加入雜訊時，Time-delay Neural Network 在沒有前處理的資料上表現比有前處理的資料還要好
  - 在 `BDEV` 上比 IBM HMM 表現還要好
