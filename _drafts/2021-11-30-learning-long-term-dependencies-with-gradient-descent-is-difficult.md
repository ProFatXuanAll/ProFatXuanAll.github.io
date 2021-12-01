---
layout: post
title:  "Learning long-term dependencies with gradient descent is difficult"
date:   2021-11-30 12:28:00 +0800
categories: [
  Deep Learning,
  Optimization,
]
tags: [
  RNN,
  RTRL,
  BPTT,
  Gradient Explosion,
  Gradient Vanishing,
]
author: [
  Yoshua Bengio,
  Patrice Simard,
  Paolo Frasconi,
]
---

|-|-|
|目標|提出 RNN 使用 BPTT 進行最佳化時遇到的問題|
|作者|Yoshua Bengio, Patrice Simard, Paolo Frasconi|
|期刊/會議名稱|IEEE Transactions on Neural Networks|
|發表時間|1994|
|論文連結|<https://ieeexplore.ieee.org/abstract/document/279181>|

<!--
  Define LaTeX command which will be used through out the writing.

  First we need to include `tools/math` which setup auto rendering.
  Each command must be wrapped with $ signs.
  We use "display: none;" to avoid redudant whitespaces.
 -->

{% include tools/math.html %}

<p style="display: none;">

  <!-- Sequence. -->
  $\providecommand{\seq}{}$
  $\renewcommand{\seq}[2]{u_{#1}, \dots, u_{#2}}$

</p>

<!-- End LaTeX command define section. -->

## 重點

- 當 RNN 訓練在重要訊號時間差較長的任務上時會遇到問題
- 根據實驗發現透過 gradient descent 進行最佳化的 RNN 模型只能達成以下兩種結果之一，且不會同時發生
  - 模型很穩定，而且能過判斷與忽略雜訊
  - 模型能夠透過 graident descent 進行有效率的訓練
- 提出 gradient descent 以外的最佳化方法

## 目標

一個能夠在多個時間儲存重要資訊的模型要能達到以下標準

1. 要能夠儲存資訊，儲存的**時間**可以是**任意長度**
2. 要能夠判斷與忽略**雜訊**
3. 模型參數要能夠被**訓練**

根據實驗，當模型想要滿足條件 1、2 時，梯度會隨著時間的增加成指數遞減，導致無法達成條件 3。

## 範例任務：Two-Sequence Problem

定義 $L, T \in \N$，給予任意長度為 $T$ 的實數序列 $\seq{1}{T}$，其前 $L$ 個數字是有意義的，剩餘的 $T - L$ 個數字都是雜訊（產生方式是透過高斯分佈，平均值為 $0$）。

定義序列 $\seq{1}{T}$ 的類別 $C(\seq{1}{T}) \in \set{0, 1}$，也就是類別只能是 $0$ 或 $1$。
由於只有前 $L$ 個數字有意義，剩餘都是雜訊，因此

$$
C(\seq{1}{T}) = C(\seq{1}{L})
$$

模型必須要在收到序列 $\seq{1}{L}$ 儲存資訊，並在收到 $\seq{L + 1}{T}$ 時忽略資訊，最後在收到 $u_T$ 的同時輸出預測類別。

實驗採用 $T \gg L$，理由如下

- 模型在收到 $\seq{1}{L}$ 時儲存的資訊必須要保存足夠長的時間才會被採用，因此 $T \gg L$ 能夠測試模型是否能夠保存長時間差的資訊
- $\seq{L + 1}{T}$ 之間的所有資訊都是雜訊，因此 $T \gg L$ 能夠測試模型是否能夠偵測與忽略雜訊
- 模型要能夠將 $u_T$ 計算所得的梯度一路往回傳遞至 $\seq{1}{L}$，因此 $T \gg L$ 能夠測試模型是否能夠在時間差較長的情況下仍能有效訓練
