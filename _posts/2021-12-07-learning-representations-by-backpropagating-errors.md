---
layout: ML-note
title: "Learning representations by back-propagating errors"
date: 2021-12-07 15:15:00 +0800
categories: [
  Gradient Descent,
  Neural Network,
]
tags: [
  Back-Propagation,
]
author: [
  David E. Rumelhart,
  Geoffrey E. Hinton,
  Ronald J. Williams,
]
---

|-|-|
|目標|提出 back-propagation|
|作者|David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams|
|期刊/會議名稱|Nature|
|發表時間|1986|
|論文連結|<https://www.nature.com/articles/323533a0>|

<!--
  Define LaTeX command which will be used through out the writing.

  Each command must be wrapped with $ signs.
  We use "display: none;" to avoid redudant whitespaces.
 -->

<p style="display: none;">

  $\newcommand{\opin}{\operatorname{in}}$
  $\newcommand{\opout}{\operatorname{out}}$
  $\newcommand{\din}{d_{\opin}}$
  $\newcommand{\dout}{d_{\opout}}$

</p>

<!-- End LaTeX command define section. -->

## 重點

- 提出 back-propagation，只需要 $1$ 階微分的最佳化演算法
  - 相較於 $2$ 階微分計算複雜度低
  - 不使用 $2$ 階微分無法保證收斂，但根據實驗大部份模型最佳化時都會透過 $1$ 階微分找到 global minimum
  - 作者說，當你遇到模型卡在 local minimum 時，多加一點參數他就會繞過去了（幹話王）
- 可以套用到 RNN 模型上，現在稱為 BPTT

## 模型架構

令模型有 $L$ 層，每一層以 $l$ 表示，因此 $l \in \set{1, \dots L}$。
定義第 $1$ 層為輸入層，第 $L$ 層為輸出層，剩餘的所有層都稱為隱藏層。
由於模型一定包含輸入與輸出，因此 $L \geq 2$。

定義第 $l$ 層的架構如下

- 輸入定義為 $x[l]$，維度為 $\din[l]$
  - 以下標 $j$ 表示第 $j$ 個輸入
  - $j$ 的範圍為 $j = 1, \dots \din[l]$
- 淨輸入定義為 $a[l]$，維度為 $\dout[l]$
  - 以下標 $i$ 表示第 $i$ 個淨輸入
  - $i$ 的範圍為 $i = 1, \dots \dout[l]$
- 輸出定義為 $y[l]$，維度為 $\dout[l]$
  - 以下標 $i$ 表示第 $i$ 個輸出
  - $i$ 的範圍為 $i = 1, \dots \dout[l]$
  - 第 $l$ 層的輸出會作為第 $l + 1$ 層的輸入，因此 $\din[l + 1] = \dout[l]$
- 參數為 $w[l]$，維度為 $\dout[l] \times \din[l]$
  - 輸入維度為 $\din[l]$
  - 輸出維度為 $\dout[l]$
  - 以下標 $w_{i, j}[l]$ 代表第 $j$ 個輸入 $x_j[l]$ 與第 $i$ 個淨輸入 $a_i[l]$ 相接的參數

計算方法如下

$$
\begin{align*}
a_i[l] & = \sum_{j = 1}^{\din[l]} w_{i, j}[l] \cdot x_j[l] && \forall i = 1, \dots, \dout[l] \\
a[l] & = w[l] \cdot x[l] \\
y_i[l] & = f\big(a_i[l]\big) = \frac{1}{1 + \exp(-a_i[l])} && \forall i = 1, \dots, \dout[l] \\
x[l + 1] & = y[l]
\end{align*} \tag{1}\label{1}
$$

- $f$ 定義成 sigmoid 函數 $\sigma$
  - 論文說可以不用是 sigmoid，只要是任何可微分且微分值有上下界即可
- 使用線性組合再配合非線性轉換讓學習的過程簡單許多

## 目標函數

若資料集有 $N$ 筆資料，則目標函數定義為

$$
\begin{align*}
E & = \frac{1}{2} \sum_{n = 1}^N E^n\big(x^n, \hat{y}^{n}\big) \\
& = \frac{1}{2} \sum_{n = 1}^N \sum_{i = 1}^{\dout[L]} \big(y_i^n[L] - \hat{y}_i^n\big)^2
\end{align*} \tag{2}\label{2}
$$

- 以 $E^n$ 代表第 $n$ 筆資料的誤差
- 以 $y^n[L]$ 代表第 $n$ 筆資料在第 $L$ 層的輸出
- 以 $\hat{y}^n$ 代表第 $n$ 筆資料的預測目標

目標為透過最佳化 $w$ 的過常達到 $E$ 的數值最小化。

## 最佳化

為了方便描述，假設 $N = 1$，唯一的資料為 $x = x[1]$，且模型在輸入 $x[1]$ 後得到的輸出為 $y[L]$。

首先計算模型輸出 $y[L]$ 對於 $\eqref{2}$ 中的 $E$ 的梯度

$$
\pd{E}{y_i[L]} = y_i[L] - \hat{y}_i \quad \forall i = 1, \dots, \dout[L] \tag{3}\label{3}
$$

令 $y[L], \hat{y}$ 為行向量（column vector），則全微分的寫法為

$$
\begin{align*}
\pd{E}{y[L]} & = \nabla E \big(y[L]\big) \\
& = \begin{pmatrix}
\pd{E}{y_1[L]} & \pd{E}{y_2[L]} & \cdots & \pd{E}{y_{\dout[L]}[L]}
\end{pmatrix} \\
& = \begin{pmatrix}
y_1[L] - \hat{y}_1 & y_2[L] - \hat{y}_2 & \cdots & y_{\dout[L]}[L] - \hat{y}_{\dout[L]}
\end{pmatrix} \\
& = \big(y[L] - \hat{y}\big)^T
\end{align*} \tag{4}\label{4}
$$

這裡我們採用 nominator-layout notation 與梯度為列向量（row vector）的習慣。

假設模型使用的啟發函數（activation function）為 sigmoid 函數 $\sigma$，則根據 chain rule 我們可以利用 $\eqref{1}\eqref{3}\eqref{4}$ 求得淨輸入 $a[L]$ 對於 $E$ 的梯度

$$
\begin{align*}
\pd{E}{a_i[L]} & = \pd{E}{y_i[L]} \cdot \pd{y_i[L]}{a_i[L]} && \forall i = 1, \dots, \dout[L] \\
& = \pd{E}{y_i[L]} \cdot \sigma'(a_i[L]) \\
& = \pd{E}{y_i[L]} \cdot \sigma(a_i[L]) \cdot \big(1 - \sigma(a_i[L])\big) \\
& = \pd{E}{y_i[L]} \cdot y_i[L] \cdot \big(1 - y_i[L]\big) \\
& = \big(y_i[L] - \hat{y}_i\big) \cdot y_i[L] \cdot \big(1 - y_i[L]\big) \\
\end{align*} \tag{5}\label{5}
$$

$$
\begin{align*}
\pd{E}{a[L]} & = \pd{E}{y[L]} \cdot \pd{y[L]}{a[L]} \\
& = \big(y[L] - \hat{y}\big)^T \cdot \begin{pmatrix}
\pd{y_1[L]}{a_1[L]} & \pd{y_1[L]}{a_2[L]} & \cdots & \pd{y_1[L]}{a_{\dout[L]}[L]} \\
\pd{y_2[L]}{a_1[L]} & \pd{y_2[L]}{a_2[L]} & \cdots & \pd{y_2[L]}{a_{\dout[L]}[L]} \\
\vdots & \vdots & \ddots & \vdots \\
\pd{y_{\dout[L]}[L]}{a_1[L]} & \pd{y_{\dout[L]}[L]}{a_2[L]} & \cdots & \pd{y_{\dout[L]}[L]}{a_{\dout[L]}[L]}
\end{pmatrix} \\
& = \big(y[L] - \hat{y}\big)^T \cdot \begin{pmatrix}
\pd{y_1[L]}{a_1[L]} & 0 & \cdots & 0 \\
0 & \pd{y_2[L]}{a_2[L]} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \pd{y_{\dout[L]}[L]}{a_{\dout[L]}[L]}
\end{pmatrix} \\
& = \big(y[L] - \hat{y}\big)^T \odot \begin{pmatrix}
\pd{y_1[L]}{a_1[L]} & \pd{y_2[L]}{a_2[L]} & \cdots & \pd{y_{\dout[L]}[L]}{a_{\dout[L]}[L]} \\
\end{pmatrix} \\
& = \big(y[L] - \hat{y}\big)^T \odot \begin{pmatrix}
y_1[L] \cdot \big(1 - y_1[L]\big) \\
y_2[L] \cdot \big(1 - y_2[L]\big) \\
\vdots \\
y_{\dout[L]}[L] \cdot \big(1 - y_{\dout[L]}[L]\big)
\end{pmatrix}^T \\
& = \big(y[L] - \hat{y}\big)^T \odot \Big(y[L] \odot \big(1 - y[L]\big)\Big)^T \\
& = \Big(\big(y[L] - \hat{y}\big) \odot y[L] \odot \big(1 - y[L]\big)\Big)^T
\end{align*} \tag{6}\label{6}
$$

其中 $\odot$ 代表 elementwise product。

接著利用 $\eqref{1}\eqref{5}\eqref{6}$ 我們可以算出第 $L$ 層的參數 $w[L]$ 對於 $E$ 的梯度

$$
\begin{align*}
\pd{E}{w_{i, j}[L]} & = \pd{E}{a_i[L]} \cdot \pd{a_i[L]}{w_{i, j}[L]} && \forall i = 1, \dots, \dout[L] \\
& = \pd{E}{a_i[L]} \cdot x_j[L] && \forall j = 1, \dots, \din[L] \\
& = \pd{E}{a_i[L]} \cdot y_j[L - 1] \\
& = \big(y_i[L] - \hat{y}_i\big) \cdot y_i[L] \cdot \big(1 - y_i[L]\big) \cdot y_j[L - 1]
\end{align*} \tag{7}\label{7}
$$

$$
\begin{align*}
\pd{E}{w_i[L]} & = \pd{E}{a[L]} \cdot \pd{a[L]}{w_i[L]} && \forall i = 1, \dots, \dout[L] \\
& = \pd{E}{a[L]} \cdot \begin{pmatrix}
\pd{a_1[L]}{w_{i, 1}[L]} & \pd{a_1[L]}{w_{i, 2}[L]} & \cdots & \pd{a_1[L]}{w_{i, \dout[L]}[L]} \\
\pd{a_2[L]}{w_{i, 1}[L]} & \pd{a_2[L]}{w_{i, 2}[L]} & \cdots & \pd{a_2[L]}{w_{i, \dout[L]}[L]} \\
\vdots & \vdots & \ddots & \vdots \\
\pd{a_{\dout[L]}[L]}{w_{i, 1}[L]} & \pd{a_{\dout[L]}[L]}{w_{i, 2}[L]} & \cdots & \pd{a_{\dout[L]}[L]}{w_{i, \din[L]}[L]}
\end{pmatrix} \\
& = \pd{E}{a[L]} \cdot \begin{pmatrix}
0 & 0 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
x_1[L] & x_2[L] & \cdots & x_{\din[L]} \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 0
\end{pmatrix} \\
& = \pd{E}{a_i[L]} \cdot x[L]^T
\end{align*} \tag{8}\label{8}
$$

其中 $w_i$ 代表參數 $w$ 的第 $i$ 列（$i$th row）。
由於 $w$ 的微分展開需要三個維度，無法使用二維空間表達，因此我按照每個列展開，最後再將展開的列合併成矩陣。

$$
\pd{E}{w} = \begin{pmatrix}
\pd{E}{w_1} \\
\pd{E}{w_2} \\
\vdots \\
\pd{E}{w_{\dout[L]}}
\end{pmatrix} = \begin{pmatrix}
\pd{E}{a_1[L]} \cdot x[L]^T \\
\pd{E}{a_2[L]} \cdot x[L]^T \\
\vdots \\
\pd{E}{a_{\dout[L]}[L]} \cdot x[L]^T
\end{pmatrix} \tag{9}\label{9}
$$

以 $\eqref{1}\eqref{5}\eqref{6}$ 我們也可以算出第 $L$ 層的輸入 $x[L]$ 對於 $E$ 的梯度

$$
\begin{align*}
\pd{E}{x_j[L]} & = \sum_{i = 1}^{\dout[L]} \pd{E}{a_i[L]} \cdot \pd{a_i[L]}{x_j[L]} && \forall j = 1, \dots, \din[L] \\
& = \sum_{i = 1}^{\dout[L]} \pd{E}{a_i[L]} \cdot w_{i, j}[L] \\
& = \sum_{i = 1}^{\dout[L]} \big(y_i[L] - \hat{y}_i\big) \cdot y_i[L] \cdot \big(1 - y_i[L]\big) \cdot w_{i, j}[L]
\end{align*} \tag{10}\label{10}
$$

$$
\begin{align*}
\pd{E}{x[L]} & = \pd{E}{a[L]} \cdot \pd{a[L]}{x[L]} \\
& = \pd{E}{a[L]} \cdot \begin{pmatrix}
\pd{a_1[L]}{x_1[L]} & \pd{a_1[L]}{x_2[L]} & \cdots & \pd{a_1[L]}{x_{\din[L]}[L]} \\
\pd{a_2[L]}{x_1[L]} & \pd{a_2[L]}{x_2[L]} & \cdots & \pd{a_2[L]}{x_{\din[L]}[L]} \\
\vdots & \vdots & \ddots & \vdots \\
\pd{a_{\dout[L]}[L]}{x_1[L]} & \pd{a_{\dout[L]}[L]}{x_2[L]} & \cdots & \pd{a_{\dout[L]}[L]}{x_{\din[L]}[L]}
\end{pmatrix} \\
& = \pd{E}{a[L]} \cdot \begin{pmatrix}
w_{1, 1}[L] & w_{1, 2}[L] & \cdots & w_{1, \din[L]}[L] \\
w_{2, 1}[L] & w_{2, 2}[L] & \cdots & w_{2, \din[L]}[L] \\
\vdots & \vdots & \ddots & \vdots \\
w_{\dout[L], 1}[L] & w_{\dout[L], 2}[L] & \cdots & w_{\dout[L], \din[L]}[L]
\end{pmatrix} \\
& = \pd{E}{a[L]} \cdot w[L] \\
& = \Big(\big(y[L] - \hat{y}\big) \odot y[L] \odot \big(1 - y[L]\big)\Big)^T \cdot w[L]
\end{align*} \tag{11}\label{11}
$$

由於 $x[L] = y[L - 1]$，我們根據 $\eqref{11}$ 可以推得

$$
\pd{E}{x[L]} = \pd{E}{y[L - 1]} = \Big(\big(y[L] - \hat{y}\big) \odot y[L] \odot \big(1 - y[L]\big)\Big)^T \cdot w[L] \tag{12}\label{12}
$$

依照 $\eqref{5} \eqref{7} \eqref{10}$ 的結果我們可以推得

$$
\begin{align*}
& \pd{E}{a_i[L - 1]} && \forall i = 1, \dots, \dout[L - 1] \\
& = \pd{E}{y_i[L - 1]} \cdot \pd{y_i[L - 1]}{a_i[L - 1]} \\
& = \pd{E}{x_i[L]} \cdot y_i[L - 1] \cdot \big(1 - y_i[L - 1]\big) \\
& = \bigg(\sum_{k = 1}^{\dout[L]} \big(y_k[L] - \hat{y}_k[L]\big) \cdot y_k[L] \cdot \big(1 - y_k[L]\big) \cdot w_{k, i}[L]\bigg) \\
& \quad \cdot y_i[L - 1] \cdot \big(1 - y_i[L - 1]\big)
\end{align*} \tag{13}\label{13}
$$

$$
\begin{align*}
& \pd{E}{w_{i, j}[L - 1]} && \forall i = 1, \dots, \dout[L - 1] \\
& = \pd{E}{a_i[L - 1]} \cdot \pd{a_i[L - 1]}{w_{i, j}} && \forall j = 1, \dots, \din[L - 1] \\
& = \bigg(\sum_{k = 1}^{\dout[L]} \big(y_k[L] - \hat{y}_k[L]\big) \cdot y_k[L] \cdot \big(1 - y_k[L]\big) \cdot w_{k, i}[L]\bigg) \\
& \quad \cdot y_i[L - 1] \cdot \big(1 - y_i[L - 1]\big) \cdot y_j[L - 2]
\end{align*} \tag{14}\label{14}
$$

$$
\begin{align*}
& \pd{E}{x_j[L - 1]} && \forall j = 1, \dots, \din[L - 1] \\
& = \sum_{i = 1}^{\dout[L - 1]} \pd{E}{a_i[L - 1]} \cdot \pd{a_i[L - 1]}{x_j[L - 1]} \\
& = \sum_{i = 1}^{\dout[L - 1]} \Bigg[\bigg(\sum_{k = 1}^{\dout[L]} \big(y_k[L] - \hat{y}_k[L]\big) \cdot y_k[L] \cdot \big(1 - y_k[L]\big) \cdot w_{k, i}[L]\bigg) \\
& \quad \cdot y_i[L - 1] \cdot \big(1 - y_i[L - 1]\big)\Bigg] \cdot w_{i j}[L - 1] \\
\end{align*} \tag{15}\label{15}
$$

依照 $\eqref{6} \eqref{9} \eqref{11} \eqref{12}$ 的結果我們可以推得

$$
\begin{align*}
& \pd{E}{a[L - 1]} \\
& = \pd{E}{y[L - 1]} \cdot \pd{y[L - 1]}{a[L - 1]} \\
& = \bigg[\Big(\big(y[L] - \hat{y}\big) \odot y[L] \odot \big(1 - y[L]\big)\Big)^T \cdot w\bigg] \odot \Big(y[L - 1] \odot \big(1 - y[L - 1]\big)\Big)^T
\end{align*} \tag{16}\label{16}
$$

$$
\begin{align*}
\pd{E}{w[L - 1]} & = \begin{pmatrix}
\pd{E}{a_1[L - 1]} \cdot x[L - 1]^T \\
\pd{E}{a_2[L - 1]} \cdot x[L - 1]^T \\
\vdots \\
\pd{E}{a_{\dout[L - 1]}[L - 1]} \cdot x[L - 1]^T
\end{pmatrix}
\end{align*} \tag{17}\label{17}
$$

$$
\begin{align*}
& \pd{E}{x[L - 1]} \\
& = \pd{E}{a[L - 1]} \cdot \pd{a[L - 1]}{x[L - 1]} \\
& = \Bigg[\bigg[\Big(\big(y[L] - \hat{y}\big) \odot y[L] \odot \big(1 - y[L]\big)\Big)^T \cdot w\bigg] \odot \Big(y[L - 1] \odot \big(1 - y[L - 1]\big)\Big)^T\Bigg] \\
& \quad \cdot w[L - 1]
\end{align*} \tag{18}\label{18}
$$

從 $\eqref{13}\eqref{14}\eqref{15}\eqref{16}\eqref{17}\eqref{18}$ 我們可以繼續往後推到任意隱藏層或輸入層的每個節點相對於誤差的微分

$$
\begin{align*}
\pd{E}{a_i[l]} & = \pd{E}{y_i[l]} \cdot \pd{y_i[l]}{a_i[l]} && \forall i = 1, \dots, \dout[l] \\
& = \pd{E}{y_i[l]} \cdot y_i[l] \cdot \big(1 - y_i[l]\big) \\
\pd{E}{w_{i j}[l]} & = \pd{E}{a_i[l]} \cdot \pd{a_i[l]}{w_{i j}[l]} && \forall i = 1, \dots, \dout[l] \\
& = \pd{E}{a_i[l]} \cdot y_j[l - 1] && \forall j = 1, \dots, \din[l] \\
& = \pd{E}{y_i[l]} \cdot y_i[l] \cdot \big(1 - y_i[l]\big) \cdot y_j[l - 1] \\
\pd{E}{x_j[l]} & = \sum_{i = 1}^{\dout[l]} \pd{E}{a_i[l]} \cdot \pd{a_i[l]}{x_j[l]} && \forall j = 1, \dots, \din[l] \\
& = \sum_{i = 1}^{\dout[l]} \pd{E}{y_i[l]} \cdot y_i[l] \cdot \big(1 - y_i[l]\big) \cdot w_{i j}[l] \\
\pd{E}{a[l]} & = \pd{E}{y[l]} \cdot \pd{y[l]}{a[l]} \\
& = \pd{E}{y[l]} \odot \Big(y[l] \odot \big(1 - y[l]\big)\Big)^T \\
\pd{E}{w[l]} & = \begin{pmatrix}
\pd{E}{a_1[l]} \cdot x[l]^T \\
\pd{E}{a_2[l]} \cdot x[l]^T \\
\vdots \\
\pd{E}{a_{\dout[l]}[l]} \cdot x[l]^T
\end{pmatrix} \\
\pd{E}{x[l]} & = \pd{E}{a[l]} \cdot \pd{a[l]}{x[l]} \\
& = \pd{E}{a[l]} \cdot w[l]
\end{align*} \tag{19}\label{19}
$$

接著使用梯度進行參數更新，論文中使用的方法是對每一筆資料（共 $N$ 筆）進行一次 $\eqref{19}$ 的計算，並將所有誤差所得梯度加總作為梯度更新的方向

$$
\pd{E}{w[l]} = \sum_{n = 1}^N \pd{E^n(x^n, \hat{y}^n)}{w[l]} \quad \forall l = 1, \dots, L \tag{20}\label{20}
$$

接著利用 $\eqref{20}$ 進行梯度下降（gradient descent）

$$
\begin{align*}
\triangle w[l]\pa{0} & = O \\
\triangle w[l]\pa{t} & = -\varepsilon \cdot \pd{E}{w[l]\pa{t}} + \alpha \cdot \triangle w[l]\pa{t - 1} \\
w[l]\pa{t + 1} & = w[l]\pa{t} + \triangle w[l]\pa{t}
\end{align*}\tag{21}\label{21}
$$

- 我們使用 $w[l]\pa{t}$ 代表第 $t$ 個 epochs 時第 $l$ 層的參數 $w[l]$
- 我們使用 $\triangle w[l]\pa{t}$ 代表第 $t$ 個 epochs 時第 $l$ 層的參數 $w[l]$ 更新的方向
  - 更新方向與 $t$ 時間點計算所得梯度方向相反
  - $\varepsilon$ 為 learning rate
  - 更新方向與 $t - 1$ 時間點計算所得更新方向相同
  - $\alpha$ 為 exponential decay factor，數值介於 $(0, 1)$
  - 參數減去 $t$ 時間點的誤差梯度，並加上 $t - 1$ 時間點的梯度所得
- $t + 1$ 時間點的參數是由 $t$ 時間點的參數往 $\triangle w[l]\pa{t}$ 更新
  - 總共訓練 $T$ 個 epochs
  - $t$ 的範圍為 $t = 1, \dots T$
- 收斂速度會比使用 $2$ 階微分的最佳化方法還要慢
  - 計算成本比 $2$ 階微分還要低許多
  - 計算概念簡單且容易實作
  - 但不像 $2$ 階微分保證收斂
- 所有參數隨機初始化成較小的數值

## 實驗 1：對稱性偵測

- 任務為給予 $6$ 個輸入數值，每 $3$ 個數值為一組，模型需要偵測兩組輸入數值是否對稱
  - 所有數值只能是 $0$ 或 $1$，因此共有 $2^6 = 64$ 種不同的輸入
  - 總共只有 $2^3 = 8$ 種對稱輸入
- 如果沒有使用隱藏層，則單純的把輸入相加得到輸出是無法偵測輸入的對稱性
- 根據實驗結果，模型成功學會讓參數數值對稱，確保對稱輸入能夠被偵測
  - 只使用一層隱藏層，只有兩個隱藏單元
  - 訓練花了 $1425$ 個 epochs
  - $\varepsilon = 0.1$
  - $\alpha = 0.9$
  - 參數使用平均分佈初始化，區間為 $[-0.3, 0.3]$
  - 連接第一組輸入和隱藏單元的參數與連接第二組輸入和隱藏單元的參數的數值正負號相反
    - 如果輸入對稱，則隱藏單元收到的淨輸入為 $0$
    - 數值呈現 $1 : 2 : 4$ 的比例，確保數值能夠正確對應
  - 隱藏單元的 bias 為負數
    - 如果輸入對稱，淨輸入加上 bias 得到負數，sigmoid 啟發值就會接近 $0$
  - 連接輸入與兩個隱藏單元的參數正負號相反
    - 當輸入不對稱時，會有一個隱藏單元輸出接近 $1$，另一個接近 $0$
  - 輸出單元參數
    - 連接隱藏單元的參數為負數
    - bias 為正數
    - 在兩個隱藏單元輸出接近 $0$ 時輸出接近 $1$
    - 在其中一個隱藏單元輸出接近 $1$ 時輸出接近 $0$

## 實驗 2：家族關係圖

- 每個家族關係定義為一個三元組（3-tuple），概念為：人-關係-人
  - 總共有 $24$ 個人，分屬兩個不同的家族，每個家族有 $12$ 個人
  - 關係共有 $12$ 個，分別為 `father`、`mother`、`husband`、`wife`、`son`、`daughter`、`uncle`、`aunt`、`brother`、`sister`、`nephew`、`niece`
- 給予 `人-人` 時需要預測關係
  - 輸入為一個 $12$ 維的向量，其中兩個數值為 $1$，剩餘為 $0$
  - 輸出為 $12$ 維的向量，其中一個數值為 $1$（代表關係），剩餘為 $0$
- 給予 `人-關係` 時需要預測人
  - 其中一個輸入為 $12$ 維的向量，只有一個數值為 $1$，剩餘為 $0$
  - 另外一個輸入為 $12$ 維的向量，只有一個數值為 $1$，剩餘為 $0$
  - 輸出為 $24$ 維的向量，可以有多個數值為 $1$（可能有多個人符合關係），剩餘為 $0$
- 架構為 $5$ 層，都是全連接
  - 第一層維度為 $12 \times 2 + 12 = 36$
  - 第二層維度為 $6 \times 2 = 12$，每個家族各自全連接到自己專屬的 $6$ 個隱藏單元
  - 第三層維度為 $12$
  - 第四層維度為 $6$
  - 第五層維度為 $12 \times 2 + 12 = 36$
- 總資料數有 $104$ 筆，訓練資料共有 $100$ 筆
  - 總共訓練 $1500$ 個 epochs
  - 前 $20$ 個 epochs 使用不同的參數
    - $\varepsilon = 0.005$
    - $\alpha = 0.5$
  - 剩餘的 epochs 使用相同的參數
    - $\varepsilon = 0.01$
    - $\alpha = 0.9$
  - 每次更新參數後，手動進行 $0.2\%$ 的 weight decay（將參數乘上 $0.998$）
  - 為了避免模型一定要輸出 $0$ 或 $1$，當模型輸出 $0.8$ 就當成 $1$，輸出 $0.2$ 就當成 $0$
    - 如果預測正確就不計算誤差（讓誤差為 $0$）
    - 如果預測失敗就按照原本方法計算誤差
- 實驗結果顯示在兩組家族上學到一樣的結構（isomorphism）
  - 這是廢話，因為本來就是用數值代表
