---
layout: ML-note
title:  "A time-delay neural network architecture for isolated word recognition"
date:   2021-11-30 12:28:00 +0800
categories: [
  Deep Learning,
  Model Architecture,
  Optimization,
]
tags: [
  RNN,
  note-is-under-construction,
]
author: [
  Kevin J.Lang,
  Alex H.Waibel,
  Geoffrey E.Hinton,
]
---

|-|-|
|目標||
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
- 使用 batch back-propagation 進行最佳化
  - 總梯度等於每筆資料的梯度進行相加
  - 不除以總資料數作 normalization
  - 總共 train 1000 個 epoch
- 針對幀數進行實驗
  - 幀數愈高，同時間的輸入愈多
  - 幀數落在 $\set{3, 6, 12, 24}$ ms
  - 當參數數量太多時，使用 weight decay 進行 regularization

## 最佳化

$$
\begin{align*}
\operatorname{loss} & = \operatorname{MSE} + \delta \cdot \norm{W(t)} \\
W(t + 1) & = W(t) - \varepsilon \cdot \pd{\operatorname{loss}}{W(t)} + \alpha \cdot \pd{\operatorname{loss}}{W(t - 1)}
\end{align*}
$$

- Learning rate 為 $\varepsilon = 0.005$
- First momentum $\alpha = 0.95$
- Weight decay factor $\delta = 0.001$
