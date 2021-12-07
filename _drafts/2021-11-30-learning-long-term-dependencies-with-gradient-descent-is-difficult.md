---
layout: ML-note
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
  note-is-under-construction,
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

定義 $L, T \in \N$，給予任意長度為 $T$ 的實數序列 $\seq{1}{T}$， $T$ 中的前 $L$ 個數字是有意義的，剩餘的 $T - L$ 個數字都是雜訊（產生方式是透過高斯分佈，平均值為 $0$，變異數為 $s$）。

定義序列 $\seq{1}{T}$ 的類別 $C(\seq{1}{T}) \in \set{0, 1}$，也就是類別只能是 $0$ 或 $1$。
由於只有前 $L$ 個數字有意義，剩餘都是雜訊，因此

$$
C(\seq{1}{T}) = C(\seq{1}{L})
$$

模型必須要在收到序列 $\seq{1}{L}$ 儲存資訊，並在收到 $\seq{L + 1}{T}$ 時忽略資訊，最後在收到 $u_T$ 的同時輸出預測類別。

當預測類別為 $0$ 時模型必須輸出 $0.8$，預測類別為 $1$ 時模型必須輸出 $-0.8$。

實驗採用 $T \gg L$，理由如下

- 模型在收到 $\seq{1}{L}$ 時儲存的資訊必須要保存足夠長的時間才會被採用，因此 $T \gg L$ 能夠測試模型是否能夠保存長時間差的資訊
- $\seq{L + 1}{T}$ 之間的所有資訊都是雜訊，因此 $T \gg L$ 能夠測試模型是否能夠偵測與忽略雜訊
- 模型要能夠將 $u_T$ 計算所得的梯度一路往回傳遞至 $\seq{1}{L}$，因此 $T \gg L$ 能夠測試模型是否能夠在時間差較長的情況下仍能有效訓練

## 實驗 1：簡易 RNN

定義 $t \in \set{1,2,\dots,T}$ 時間點的輸入為 $x(t)$，輸出為 $y(t)$，模型參數為 $w$，計算方法如下

$$
\begin{align*}
y(0) & = 0 \\
y(t) & = f(h(t)) = \tanh(h(t)) \\
h(t) & = w y(t - 1) + x(t)
\end{align*}
$$

- 當 $t \leq L$ 時，輸入 $x(t)$ 是可以訓練的參數
  - $x(t)$ 只有兩種類別，分別是 $\set{0, 1}$
  - $x(t)$ 以平均分佈初始化成很小的數值
  - 想成在訓練 word embedding 的感覺
- 當 $t > L$ 時，$x(t)$ 是由高斯分佈產生的雜訊，平均值為 $0$，變異數為 $s$
- 根據實驗，當 $w$ 初始數值很小，且 $s$ 很大時，模型的輸出 $y(t)$ 都會變得很靠近 $0$
  - $T$ 設定為 $20$，$L$ 設定為 $3$
  - 實驗結果是 $18$ 次實驗的平均值
  - 輸出結果都很接近 $0$ 表示模型不太可能收斂
- 根據實驗，當輸入長度 $T$ 變長時，$x(t)$ 的收斂頻率變得愈來愈少
  - 收斂頻率指的是執行 $18$ 次實驗，$x(t)$ 的數值收斂的次數
  - 參數 $w$ 初始值設定為 $1.25$
  - 雜訊變異數 $s$ 設定為 $0.2$
  - 結論是即使是在簡單的任務下，如果輸入較長，則模型無法學會記住資訊

## 分析

接下來的分析與 differentiable manifold 有關，等我會了再回來看。
