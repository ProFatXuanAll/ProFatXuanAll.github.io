---
layout: post
title:  "Local Feedback Multilayered Networks"
date:   2021-11-30 14:21:00 +0800
categories: [
  Deep Learning,
  Model Architecture,
  Optimization,
]
tags: [
  RNN,
  BPTT,
  note-is-under-construction,
]
author: [
  Paolo Frasconi,
  Marco Gori,
  Giovanni Soda,
]
---

|-|-|
|目標|分析 Local Feedback Multilayered Networks，證明該模型擁有遺忘行為，並認為遺忘行為是**好現象**|
|作者|Paolo Frasconi, Marco Gori, Giovanni Soda|
|期刊/會議名稱|Neural Computation|
|發表時間|1992|
|論文連結|<https://direct.mit.edu/neco/article-abstract/4/1/120/5626/Local-Feedback-Multilayered-Networks>|

<!--
  Define LaTeX command which will be used through out the writing.

  First we need to include `tools/math` which setup auto rendering.
  Each command must be wrapped with $ signs.
  We use "display: none;" to avoid redudant whitespaces.
 -->

{% include tools/math.html %}

<p style="display: none;">

  <!-- Operators. -->
  $\providecommand{\opin}{}$
  $\renewcommand{\opin}{\operatorname{in}}$
  $\providecommand{\ophid}{}$
  $\renewcommand{\ophid}{\operatorname{hid}}$
  $\providecommand{\opout}{}$
  $\renewcommand{\opout}{\operatorname{out}}$
  $\providecommand{\oploss}{}$
  $\renewcommand{\oploss}{\operatorname{loss}}$
  $\providecommand{\opnorm}{}$
  $\renewcommand{\opnorm}{\operatorname{normal}}$
  $\providecommand{\oprecur}{}$
  $\renewcommand{\oprecur}[1]{\operatorname{recur}_{#1}}$

  <!-- Dimensions. -->
  $\providecommand{\din}{}$
  $\renewcommand{\din}{d_{\opin}}$
  $\providecommand{\dhid}{}$
  $\renewcommand{\dhid}{d_{\ophid}}$
  $\providecommand{\dout}{}$
  $\renewcommand{\dout}{d_{\opout}}$

  <!-- Weights. -->
  $\providecommand{\whid}{}$
  $\renewcommand{\whid}{w^{\ophid}}$
  $\providecommand{\wout}{}$
  $\renewcommand{\wout}{w^{\opout}}$
  $\providecommand{\Wout}{}$
  $\renewcommand{\Wout}{W^{\opout}}$

  <!-- Self-connected weights. -->
  $\providecommand{\uhid}{}$
  $\renewcommand{\uhid}{u^{\ophid}}$
  $\providecommand{\uout}{}$
  $\renewcommand{\uout}{u^{\opout}}$
  $\providecommand{\vhid}{}$
  $\renewcommand{\vhid}{v^{\ophid}}$
  $\providecommand{\vout}{}$
  $\renewcommand{\vout}{v^{\opout}}$

</p>

<!-- End LaTeX command define section. -->

## 重點

- 分析 Local Feedback Multilayered Networks
  - 是 RNN 模型的其中一種
  - Feedback connection 只能接到相同的來源
  - 作者認為 RNN 架構不需要全連結
  - 作者證明 Local Feedback Multilayered Networks 在特定條件下擁有**遺忘行為**（**Forgetting Behavior**）
- 作者認為如果 RNN 模型擁有**遺忘行為**，則該 RNN 就等同於**簡單的 MLP**，但輸入是**固定範圍**的資訊
  - 作者認為遺忘行為能夠幫助模型忘記**較早期**已經處理過的資訊，在**語音辨識**中是很重要的功能
  - 從現在的角度來看，我覺得就像是使用 RNN 模擬一個簡單的 **CNN**
- 最佳化可以達成 local in both space and time
- 無法學習輸入較長的任務
  - 例如無法模擬計數器

## 模型架構

論文提出的架構包含一個輸入層、**多個**隱藏層與一個輸出層。
但論文中所有分析都是建立於只有**一個**隱藏層的前提，因此我們也只討論只有一層隱藏層的架構。

定義以下符號

- $T$ 代表輸入序列的**長度**
  - 模型會按照 $t = 1, \dots T$ 的順序得到輸入，並產出輸出
- $I$ 代表所有輸入單元所形成的集合
  - 第 $t$ 個時間點的所有輸入單元定義為 $x(t)$，維度為 $\din$
  - 使用腳標 $x_k(t) \in I$ 代表 $t$ 時間點的第 $k$ 個輸入單元，$k = 1, \dots, \din$
- $H$ 代表所有隱藏單元所形成的集合
  - 第 $t$ 個時間點的所有隱藏單元定義為 $h(t)$，維度為 $\dhid$
  - 使用腳標 $h_j(t) \in H$ 代表 $t$ 時間點的第 $j$ 個隱藏單元，$j = 1, \dots, \dhid$
- $O$ 代表所有輸出單元所形成的集合
  - 第 $t$ 個時間點的所有輸出單元定義為 $y(t)$，維度為 $\dout$
  - 使用腳標 $y_i(t) \in O$ 代表 $t$ 時間點的第 $i$ 個輸出單元，$i = 1, \dots, \dout$
- $D$ 代表所有**遞迴單元**所形成的集合
  - 論文定義所有遞迴單元只能是**隱藏**單元或是**輸出**單元，因此 $D \subseteq H \cup O$
  - 所有動態單元只能與**自己連接**還有**直接**和**輸入**連接
  - 與自己連接的方法有**兩種**
    - 與淨輸入（net input）相連，我們稱為**淨輸入遞迴單元**
    - 與啟發值（activation）相連，我們稱為**啟發值遞迴單元**
  - 論文又稱這些遞迴單元為**動態**單元
  - 相對於動態單元，所有沒有進行遞迴的單元則被稱為**靜態**單元，我們以**非遞迴**單元稱呼
- 隱藏層與輸出層使用的**啟發函數** $f : \R \to [-1, 1]$ 定義為 $f(a) = \tanh(a / 2)$

### 隱藏單元

若以遞迴與否加上遞迴方式進行區分，共有**三**種不同的隱藏單元，分別為

- 非遞迴隱藏單元，共有 $\opnorm(h)$ 個
- 淨輸入遞迴隱藏單元，共有 $\oprecur{1}(h)$ 個
- 啟發值遞迴隱藏單元，共有 $\oprecur{2}(h)$ 個

因此 $\dhid = \opnorm(h) + \oprecur{1}(h) + \oprecur{2}(h)$。

#### 非遞迴隱藏單元

$$
h_j(t) = f\pa{\whid_j \odot x(t)} = f\pa{\sum_{k = 1}^{\din} \whid_{j k} \cdot x_k(t)} \tag{1}\label{1}
$$

- 將 $t$ 時間點的輸入 $x(t)$ 透過**全連接**的方式計算 $t$ 時間點的第 $j$ 個**隱藏單元** $h_j(t)$
  - 當隱藏單元的架構為 $\eqref{1}$ 時，我們以 $h_j(t) \in H \setminus D$ 進行表達
  - $j$ 的範圍為 $j = 1, \dots, \opnorm(h)$
- 參數 $\whid$ 的維度為 $\dhid \times \din$，與 $\eqref{2} \eqref{3}$ 共享參數
- $t$ 時間點的**非遞迴隱藏單元**唯一的訊號來源就是 $t$ 時間點的輸入 $x(t)$
- $\eqref{1}$ 就是論文中的 2.1 式

#### 淨輸入遞迴隱藏單元

$$
\begin{align*}
\tilde{h}_j(0) & = 0 \\
\tilde{h}_j(t) & = \uhid_j \cdot \tilde{h}_j(t - 1) + \whid_j \odot x(t) \\
h_j(t) & = f\pa{\tilde{h}_j(t)}
\end{align*} \tag{2}\label{2}
$$

- 基於 $\eqref{1}$，但額外加上 $t - 1$ 時間點的第 $j$ 個**淨輸入遞迴隱藏單元的淨輸入** $\tilde{h}_j(t - 1)$
  - $\uhid$ 的維度為 $\oprecur{1}(h)$
  - $\uhid$ 的下標應該要從 $1$ 開始，但為了方便表達我們改成從 $\opnorm(h) + 1$ 開始
  - 與 $\eqref{3}$ 的差別在於不是使用**啟發值**進行遞迴
  - 與 $\eqref{1} \eqref{3}$ 共享參數 $\whid$
  - 當隱藏單元的架構為 $\eqref{2}$ 時，我們以 $h_j(t) \in H \cap D$ 進行表達
  - $j$ 的範圍為 $j = \opnorm(h) + 1, \dots, \opnorm(h) + \oprecur{1}(h)$
- 第 $j$ 個淨輸入遞迴隱藏單元的訊號來源共有兩種
  - $t$ 時間點的的輸入 $x(t)$
  - $t - 1$ 時間點的第 $j$ 個**淨輸入遞迴隱藏單元的淨輸入**
- **遞迴**的來源就是 $\tilde{h}_j(t - 1)$
- $\eqref{2}$ 就是論文中的 2.2 式

#### 啟發值遞迴隱藏單元

$$
\begin{align*}
h_j(0) & = 0 \\
\bar{h}_j(t) & = \vhid_j \cdot h_j(t - 1) + \whid_j \odot x(t) \\
h_j(t) & = f\pa{\bar{h}_j(t)}
\end{align*} \tag{3}\label{3}
$$

- 基於 $\eqref{1}$，但額外加上 $t - 1$ 時間點的第 $j$ 個隱藏單元 $h_j(t - 1)$
  - $\vhid$ 的維度為 $\oprecur{2}(h)$
  - $\vhid$ 的下標應該要從 $1$ 開始，但為了方便表達我們改成從 $\opnorm(h) + \oprecur{1}(h) + 1$ 開始
  - 與 $\eqref{2}$ 的差別在於不是使用**淨輸入**進行遞迴
  - 與 $\eqref{1} \eqref{2}$ 共享參數 $\whid$
  - 當隱藏單元的架構為 $\eqref{3}$ 時，我們以 $h_j(t) \in H \cap D$ 進行表達
  - $j$ 的範圍為 $j = \opnorm(h) + \oprecur{1}(h) + 1, \dots, \opnorm(h) + \oprecur{1}(h) + \oprecur{2}(h)$
- 第 $j$ 個啟發值遞迴隱藏單元的訊號來源共有兩種
  - $t$ 時間點的的輸入 $x(t)$
  - $t - 1$ 時間點的第 $j$ 個**隱藏單元**
- **遞迴**的來源就是 $h_j(t - 1)$
- $\eqref{3}$ 就是論文中的 2.3 式

### 輸出單元

若以遞迴與否加上遞迴方式進行區分，共有**三**種不同的輸出單元，分別為

- 非遞迴輸出單元，共有 $\opnorm(y)$ 個
- 淨輸入遞迴輸出單元，共有 $\oprecur{1}(y)$ 個
- 啟發值遞迴輸出單元，共有 $\oprecur{2}(y)$ 個

因此 $\dout = \opnorm(y) + \oprecur{1}(y) + \oprecur{2}(y)$。

與隱藏單元不同的是，遞迴輸出單元只能與**輸入**直接連接，而不是與隱藏單元進行連接。

#### 非遞迴輸出單元

$$
y_i(t) = f\pa{\wout_i \odot h(t)} = f\pa{\sum_{j = 1}^{\dhid} \wout_{i j} \cdot h_j(t)} \tag{4}\label{4}
$$

- 將 $t$ 時間點的隱藏層 $h(t)$ 透過**全連接**的方式計算 $t$ 時間點的第 $i$ 個**輸出單元** $y_i(t)$
  - 當輸出單元的架構為 $\eqref{4}$ 時，我們以 $y_i(t) \in O \setminus D$ 進行表達
  - $i$ 的範圍為 $i = 1, \dots, \opnorm(y)$
- 參數 $\wout$ 的維度為 $\opnorm(y) \times \dhid$
- $t$ 時間點的**非遞迴輸出單元**唯一的訊號來源就是 $t$ 時間點的隱藏單元 $h(t)$
- $\eqref{4}$ 就是論文中的 2.1 式

#### 淨輸入遞迴輸出單元

$$
\begin{align*}
\tilde{y}_i(0) & = 0 \\
\tilde{y}_i(t) & = \uout_i \cdot \tilde{y}_i(t - 1) + \Wout_i \odot x(t) \\
y_i(t) & = f\pa{\tilde{y}_i(t)}
\end{align*} \tag{5}\label{5}
$$

- 與 $\eqref{4}$ 完全不同
  - $\uout$ 的維度為 $\oprecur{1}(y)$
  - $\uout$ 的下標應該要從 $1$ 開始，但為了方便表達我們改成從 $\opnorm(y) + 1$ 開始
  - $\Wout$ 的維度為 $(\oprecur{1}(y) + \oprecur{2}(y)) \times \din$
  - $\Wout$ 的下標應該要從 $1$ 開始，但為了方便表達我們改成從 $\opnorm(y) + 1$ 開始
  - 與 $\eqref{6}$ 共享參數 $\Wout$
  - 與 $\eqref{5}$ 的差別在於不是使用**啟發值**進行遞迴
  - 注意 $\Wout \neq \wout$，我們以大小寫區分連接的來源
  - 當輸出單元的架構為 $\eqref{5}$ 時，我們以 $y_i(t) \in O \cap D$ 進行表達
  - $i$ 的範圍為 $i = \opnorm(y) + 1, \dots, \opnorm(y) + \oprecur{1}(y)$
- 第 $i$ 個淨輸入遞迴輸出單元的訊號來源共有兩種
  - $t$ 時間點的的輸入 $x(t)$
  - $t - 1$ 時間點的第 $i$ 個**淨輸入遞迴輸出單元的淨輸入**
- **遞迴**的來源就是 $\tilde{y}_i(t - 1)$
- $\eqref{5}$ 就是論文中的 2.2 式

#### 啟發值遞迴輸出單元

$$
\begin{align*}
y_i(0) & = 0 \\
\bar{y}_i(t) & = \vout_i \cdot y_i(t - 1) + \Wout_i \odot x(t) \\
y_i(t) & = f\pa{\bar{y}_i(t)}
\end{align*} \tag{6}\label{6}
$$

- 與 $\eqref{5}$ 概念相同，但是遞迴的項次是**輸出單元**
  - $\vout$ 的維度為 $\oprecur{2}(y)$
  - $\vout$ 的下標應該要從 $1$ 開始，但為了方便表達我們改成從 $\opnorm(y) + \oprecur{1}(y) + 1$ 開始
  - $\Wout$ 的維度為 $(\oprecur{1}(y) + \oprecur{2}(y)) \times \din$
  - $\Wout$ 的下標應該要從 $1$ 開始，但為了方便表達我們改成從 $\opnorm(y) + \oprecur{1}(y) + 1$ 開始
  - 與 $\eqref{5}$ 共享 $\Wout$
  - 與 $\eqref{5}$ 的差別在於不是使用**淨輸入**進行遞迴
  - 當輸出單元的架構為 $\eqref{6}$ 時，我們以 $y_i(t) \in O \cap D$ 進行表達
  - $i$ 的範圍為 $i = \opnorm(y) + \oprecur{1}(y) + 1, \dots, \opnorm(y) + \oprecur{1}(y) + \oprecur{2}(y)$
- 第 $i$ 個啟發值遞迴輸出單元的訊號來源共有兩種
  - $t$ 時間點的的輸入 $x(t)$
  - $t - 1$ 時間點的第 $i$ 個**輸出單元**
- **遞迴**的來源就是 $y_i(t - 1)$
- $\eqref{6}$ 就是論文中的 2.3 式

### 目標函數

使用最小平方差作為目標函數

$$
\begin{align*}
\oploss & = \sum_{t = 1}^T \sigma_t \cdot C(t) \\
& = \sum_{t = 1}^T \sigma_t \cdot \pa{\sum_{i = 1}^{\dout} \big(y_i(t) - d_i(t)\big)^2} \\
& = \sum_{t = 1}^T \sum_{i = 1}^{\dout} \sigma_t \cdot \big(y_i(t) - d_i(t)\big)^2
\end{align*} \tag{7}\label{7}
$$

- $C(t)$ 代表 $t$ 時間點最小平方差的總和
- $d_i(t)$ 是 $t$ 時間點第 $i$ 個輸出的**預測目標**
- 由於不是所有時間點都需要預測答案，因此以 $\sigma_t$ 表示是否需要考慮 $t$ 時間點的誤差
  - 如果需要考慮 $t$ 時間點的誤差，則 $\sigma_t = 1$
  - 如果不需要考慮 $t$ 時間點的誤差，則 $\sigma_t = 0$
- $\eqref{7}$ 就是論文中的 2.5 式

### 反向傳播

由於模型是透過梯度下降演算法（gradient descent）進行最佳化，因此必須透過反向傳播計算參數所得梯度。

針對不同的參數，我們可以求得不同種的梯度結構。
以下我們將會一一列舉各個參數的梯度。

#### 與非遞迴輸出單元相連參數

根據 $\eqref{4}$ 我們可以求得

$$
\begin{align*}
\pd{(\sigma_t \cdot C(t))}{\wout_{i j}} & = \sigma_t \cdot \pd{C(t)}{y_i(t)} \cdot \pd{y_i(t)}{\wout_{i j}} \\
& = \sigma_t \cdot 2\big(y_i(t) - d_i(t)\big) \cdot f'\pa{\wout_{i} \odot h(t)} \cdot h_j(t)
\end{align*} \tag{8}\label{8}
$$

- 由於 $\wout$ 沒有參與遞迴的過程，因此梯度不會傳遞給 $t - 1$ 時間點以前的節點
- $i$ 的範圍為 $i = 1 \dots \opnorm(y)$
- $j$ 的範圍為 $j = 1 \dots \dhid$

#### 與淨輸入遞迴輸出單元相連參數

根據 $\eqref{5}$ 我們可以求得

$$
\begin{align*}
\pd{(\sigma_t \cdot C(t))}{\uout_i} & = \sigma_t \cdot \pd{C(t)}{y_i(t)} \cdot \pd{y_i(t)}{\tilde{y}_i(t)} \cdot \pd{\tilde{y}_i(t)}{\uout_i} \\
& = \sigma_t \cdot 2\big(y_i(t) - d_i(t)\big) \cdot f'\pa{\tilde{y}_i(t)} \cdot \pd{\tilde{y}_i(t)}{\uout_i} \\
\pd{\tilde{y}_i(t)}{\uout_i} & = \tilde{y}_i(t - 1) + \uout_i \cdot \pd{\tilde{y}_i(t - 1)}{\uout_i}
\end{align*} \tag{9}\label{9}
$$

- 由於 $\uout$ 有參與遞迴的過程，因此梯度必須傳遞給 $t - 1$ 時間點以前的**淨輸入遞迴輸出單元**
- $i$ 的範圍為 $i = \opnorm(y) + 1 \dots \opnorm(y) + \oprecur{1}(y)$
- $\eqref{9}$ 就是論文中的 2.8 式

#### 與啟發值遞迴輸出單元相連參數

根據 $\eqref{6}$ 我們可以求得

$$
\begin{align*}
\pd{(\sigma_t \cdot C(t))}{\vout_i} & = \sigma_t \cdot \pd{C(t)}{y_i(t)} \cdot \pd{y_i(t)}{\bar{y}_i(t)} \cdot \pd{\bar{y}_i(t)}{\vout_i} \\
& = \sigma_t \cdot 2\big(y_i(t) - d_i(t)\big) \cdot f'\pa{\bar{y}_i(t)} \cdot \pd{\bar{y}_i(t)}{\vout_i} \\
\pd{\bar{y}_i(t)}{\vout_i} & = y_i(t - 1) + \vout_i \cdot \pd{y_i(t - 1)}{\bar{y}_i(t - 1)} \cdot \pd{\bar{y}_i(t - 1)}{\vout_i} \\
& = y_i(t - 1) + \vout_i \cdot f'\pa{\bar{y}_i(t - 1)} \cdot \pd{\bar{y}_i(t - 1)}{\vout_i}
\end{align*} \tag{10}\label{10}
$$

- 由於 $\vout$ 有參與遞迴的過程，因此梯度必須傳遞給 $t - 1$ 時間點以前的**啟發值遞迴輸出單元**
- $i$ 的範圍為 $i = \opnorm(y) + \oprecur{1}(y) + 1 \dots \opnorm(y) + \oprecur{1}(y) + \oprecur{2}(y)$
- $\eqref{10}$ 就是論文中的 2.7 式

#### 連接輸出單元與輸入單元的參數

根據 $\eqref{5}\eqref{6}$ 我們可以求得

$$
\begin{align*}
\pd{(\sigma_t \cdot C(t))}{\Wout_{i k}} & = \sigma_t \cdot \pd{C(t)}{y_i(t)} \cdot \br{\pd{y_i(t)}{\tilde{y}_i(t)} \cdot \pd{\tilde{y}_i(t)}{\Wout_{i k}} + \pd{y_i(t)}{\bar{y}_i(t)} \cdot \pd{\bar{y}_i(t)}{\Wout_{i k}}} \\
& = \sigma_t \cdot 2\big(y_i(t) - d_i(t)\big) \cdot \br{f'\pa{\tilde{y}_i(t)} \cdot \pd{\tilde{y}_i(t)}{\Wout_{i k}} + f'\pa{\bar{y}_i(t)} \cdot \pd{\bar{y}_i(t)}{\Wout_{i k}}} \\
\pd{\tilde{y}_i(t)}{\Wout_{i k}} & = \uout_i \cdot \pd{\tilde{y}_i(t - 1)}{\Wout_{i k}} + x_k(t) \\
\pd{\bar{y}_i(t)}{\Wout_{i k}} & = \vout_i \cdot f'\pa{\bar{y}_i(t - 1)} \cdot \pd{\bar{y}_i(t - 1)}{\Wout_{i k}} + x_k(t)
\end{align*} \tag{11}\label{11}
$$

- 由於 $\Wout$ 有參與 $\eqref{5}\eqref{6}$ 的遞迴過程，因此梯度必須傳遞給 $t - 1$ 時間點以前的**遞迴輸出單元**
- $i$ 的範圍為 $i = \opnorm(y) + 1 \dots \opnorm(y) + \oprecur{1}(y) + \oprecur{2}(y)$
- $k$ 的範圍為 $k = 1 \dots \din$
- $\eqref{11}$ 就是論文中的 2.7, 2.8 式

#### 與淨輸入遞迴隱藏單元相連參數

根據 $\eqref{2}\eqref{4}$ 我們可以求得

$$
\begin{align*}
\pd{(\sigma_t \cdot C(t))}{\uhid_j} & = \sigma_t \cdot \pa{\sum_{i = 1}^{\opnorm(y)} \pd{C(t)}{y_i(t)} \cdot \pd{y_i(t)}{h_j(t)}} \cdot \pd{h_j(t)}{\tilde{h}_j(t)} \cdot \pd{\tilde{h}_j(t)}{\uhid_j} \\
& = \sigma_t \cdot \pa{\sum_{i = 1}^{\opnorm(y)} 2\big(y_i(t) - d_i(t)\big) \cdot \wout_{i j}} \cdot f'\pa{\tilde{h}_j(t)} \cdot \pd{\tilde{h}_j(t)}{\uhid_j} \\
\pd{\tilde{h}_j(t)}{\uhid_j} & = \tilde{h}_j(t - 1) + \uhid_j \cdot \pd{\tilde{h}_j(t - 1)}{\uhid_j}
\end{align*} \tag{12}\label{12}
$$

- 由於 $h(t)$ 只會參與非遞迴輸出單元的計算，因此第一個等式中只有 $\opnorm(y)$ 個加法項次
- 由於 $\uhid$ 有參與遞迴的過程，因此梯度必須傳遞給 $t - 1$ 時間點以前的**淨輸入遞迴隱藏單元**
- $j$ 的範圍為 $j = \opnorm(h) + 1 \dots \opnorm(h) + \oprecur{1}(h)$
- $\eqref{12}$ 就是論文中的 2.8 式

#### 與啟發值遞迴隱藏單元相連參數

根據 $\eqref{3}\eqref{4}$ 我們可以求得

$$
\begin{align*}
\pd{(\sigma_t \cdot C(t))}{\vhid_j} & = \sigma_t \cdot \pa{\sum_{i = 1}^{\opnorm(y)} \pd{C(t)}{y_i(t)} \cdot \pd{y_i(t)}{h_j(t)}} \cdot \pd{h_j(t)}{\bar{h}_j(t)} \cdot \pd{\bar{h}_j(t)}{\vhid_j} \\
& = \sigma_t \cdot \pa{\sum_{i = 1}^{\opnorm(y)} 2\big(y_i(t) - d_i(t)\big) \cdot \wout_{i j}} \cdot f'\pa{\bar{h}_j(t)} \cdot \pd{\bar{h}_j(t)}{\vhid_j} \\
\pd{\bar{h}_j(t)}{\vhid_j} & = h_j(t - 1) + \vhid_j \cdot \pd{h_j(t - 1)}{\bar{h}_j(t - 1)} \cdot \pd{\bar{h}_j(t - 1)}{\vhid_j} \\
& = h_j(t - 1) + \vhid_j \cdot f'\pa{\bar{h}_j(t - 1)} \cdot \pd{\bar{h}_j(t - 1)}{\vhid_j}
\end{align*} \tag{13}\label{13}
$$

- 由於 $h(t)$ 只會參與非遞迴輸出單元的計算，因此第一個等式中只有 $\opnorm(y)$ 個加法項次
- 由於 $\vhid$ 有參與遞迴的過程，因此梯度必須傳遞給 $t - 1$ 時間點以前的**啟發值遞迴隱藏單元**
- $j$ 的範圍為 $j = \opnorm(h) + \oprecur{1}(h) + 1 \dots \opnorm(h) + \oprecur{1}(h) + \oprecur{2}(h)$
- $\eqref{13}$ 就是論文中的 2.7 式

#### 連接隱藏單元與輸入單元的參數

根據 $\eqref{1}\eqref{2}\eqref{3}\eqref{4}$ 我們可以求得

$$
\begin{align*}
\pd{(\sigma_t \cdot C(t))}{\whid_{j k}} & = \sigma_t \cdot \pa{\sum_{i = 1}^{\opnorm(y)} \pd{C(t)}{y_i(t)} \cdot \pd{y_i(t)}{h_j(t)}} \cdot \pd{h_j(t)}{\whid_{j k}} \\
& = \sigma_t \cdot \pa{\sum_{i = 1}^{\opnorm(y)} 2\big(y_i(t) - d_i(t)\big) \cdot \wout_{i j}} \cdot \pd{h_j(t)}{\whid_{j k}} \\
\pd{h_j(t)}{\whid_{j k}} & = \begin{dcases}
f'\pa{\whid_j \odot x(t)} \cdot x_k(t) & \text{if } j = 1, \dots, \opnorm(h) \\
f'\pa{\tilde{h}_j(t)} \cdot \pd{\tilde{h}_j(t)}{\whid_{j k}} & \text{if } j = \opnorm(h) + 1, \dots, \\
& \opnorm(h) + \oprecur{1}(h) \\
f'\pa{\bar{h}_j(t)} \cdot \pd{\bar{h}_j(t)}{\whid_{j k}} & \text{if } j = \opnorm(h) + \oprecur{1}(h) + 1, \dots, \\
& \opnorm(h) + \oprecur{1}(h) + \oprecur{2}(h)
\end{dcases} \\
\pd{\tilde{h}_j(t)}{\whid_{j k}} & = \uhid_j \cdot \pd{\tilde{h}_j(t - 1)}{\whid_{j k}} + x_k(t) \\
\pd{\bar{h}_j(t)}{\whid_{j k}} & = \vhid_j \cdot f'\pa{\bar{h}_j(t - 1)} \cdot \pd{\bar{h}_j(t - 1)}{\whid_{j k}} + x_k(t)
\end{align*} \tag{14}\label{14}
$$

- 由於 $h(t)$ 只會參與非遞迴輸出單元的計算，因此第一個等式中只有 $\opnorm(y)$ 個加法項次
- 由於 $\whid$ 參與不同的遞迴過程，因此梯度會分別傳遞給 $t - 1$ 時間點以前的淨輸入遞迴隱藏單元與啟發值遞迴隱藏單元。
- $j$ 的範圍為 $j = 1 \dots \opnorm(h) + \oprecur{1}(h) + \oprecur{2}(h)$
  - 倒數第二個等式中 $j$ 的範圍為 $j = \opnorm(h) + 1 \dots \opnorm(h) + \oprecur{1}(h)$
  - 最後一個等式中 $j$ 的範圍為 $j = \opnorm(h) + \oprecur{1}(h) + 1 \dots \opnorm(h) + \oprecur{1}(h) + \oprecur{2}(h)$
- $k$ 的範圍為 $k = 1 \dots \din$
- $\eqref{14}$ 就是論文中的 2.7, 2.8 式

## 遺忘行為

一個 RNN 模型如果滿足以下的行為

$$
\lim_{q \to \infty} \pd{y_i(t)}{x_k(t - q)} = 0 \tag{15}\label{15}
$$

則論文稱該模型擁有**遺忘行為**（**Forgetting Behavior**）。

- $i$ 的範圍為 $i = 1, \dots, \dout$
- $k$ 的範圍為 $k = 1, \dots, \din$
- $t$ 的範圍為 $t = 1, \dots, T$
- 作者認為如果 RNN 模型擁有**遺忘行為**，則該 RNN 就等同於**簡單的 MLP**，但輸入是**固定範圍**的資訊
  - 作者認為遺忘行為能夠幫助模型忘記**較早期**已經處理過的資訊，在**語音辨識**中是很重要的功能
  - 從現在的角度來看，我覺得就像是使用 RNN 模擬一個簡單的 **CNN**

作者接著證明 Local Feedback Multilayered Networks 在特定條件下能夠擁有遺忘行為。

### 引理：啟發值的最大值

令啟發函數 $f(z) = \tanh(z / 2)$。
則

$$
M = \max_{z \in \R} \pa{f'(z)} = 0.5. \tag{16}\label{16}
$$

理由如下

$$
\begin{align*}
& \tanh(z) = 2\sigma(2z) - 1 \\
\implies & \tanh'(z) = \frac{d \tanh(z)}{dz} = 2\frac{d \sigma(2z)}{z} \\
& = 2 \frac{d\sigma(2z)}{d2z} \cdot \frac{d2z}{dz} \\
& = 4 \big(\sigma(2z) \cdot [1 - \sigma(2z)]\big) \\
\implies & f'(z) = \frac{d\tanh(\frac{z}{2})}{dz} = \frac{d\tanh(\frac{z}{2})}{d\frac{z}{2}} \cdot \frac{d\frac{z}{2}}{dz} \\
& = 4 \big(\sigma(z) \cdot [1 - \sigma(z)]\big) \cdot \frac{1}{2} = 2\big(\sigma(z) \cdot [1 - \sigma(z)]\big) \\
\implies & M = \max_{z \in \R}\pa{f'(z)} = 2(0.5 \cdot 0.5) = 0.5.
\end{align*}
$$

### 證明

我們接著證明以下敘述：

令 $M$ 為 $\eqref{16}$ 中的數值。
當 Local Feedback Multilayered Networks 使用的啟發函數都是 $f(z) = \tanh(z / 2)$，且 $\uhid, \vhid$ 的數值範圍滿足

$$
\begin{align*}
\abs{\uhid_j} & < 1 \\
\abs{\vhid_j} & < \frac{1}{M}
\end{align*} \tag{17}\label{17}
$$

則 Local Feedback Multilayered Networks 擁有遺忘特行為。

理由如下：

當 $q = 0$ 時，我們可以推得

$$
\begin{align*}
\abs{\pd{\tilde{h}_j(t)}{\pa{\whid_j \odot x(t - q)}}} & = \abs{\pd{\tilde{h}_j(t)}{\pa{\whid_j \odot x(t)}}} = 1 \\
\abs{\pd{\bar{h}_j(t)}{\pa{\whid_j \odot x(t - q)}}} & = \abs{\pd{\bar{h}_j(t)}{\pa{\whid_j \odot x(t)}}} = 1
\end{align*} \tag{18}\label{18}
$$

當 $q = 1$ 時，根據 $\eqref{17}\eqref{18}$ 我們可以推得

$$
\begin{align*}
\abs{\pd{\tilde{h}_j(t)}{\pa{\whid_j \odot x(t - q)}}} & = \abs{\pd{\tilde{h}_j(t)}{\pa{\whid_j \odot x(t - 1)}}} \\
& = \abs{\uhid_j \cdot \pd{\tilde{h}_j(t - 1)}{\pa{\whid_j \odot x(t - 1)}}} \\
& = \abs{\uhid_j} = \abs{\uhid_j}^1  \\
\abs{\pd{\bar{h}_j(t)}{\pa{\whid_j \odot x(t - q)}}} & = \abs{\pd{\bar{h}_j(t)}{\pa{\whid_j \odot x(t - 1)}}} \\
& = \abs{\vhid_j \cdot f'\pa{\bar{h}(t - 1)} \cdot \pd{\bar{h}_j(t - 1)}{\pa{\whid_j \odot x(t - 1)}}} \\
& = \abs{\vhid_j \cdot f'\pa{\bar{h}(t - 1)}} \\
& \leq \abs{\vhid_j \cdot M} = \abs{\vhid_j \cdot M}^1
\end{align*} \tag{19}\label{19}
$$

當 $q = 2$ 時，根據 $\eqref{17}\eqref{19}$ 我們可以推得

$$
\begin{align*}
\abs{\pd{\tilde{h}_j(t)}{\pa{\whid_j \odot x(t - q)}}} & = \abs{\pd{\tilde{h}_j(t)}{\pa{\whid_j \odot x(t - 2)}}} \\
& = \abs{\pd{\tilde{h}_j(t)}{\tilde{h}_j(t - 1)} \cdot \pd{\tilde{h}_j(t - 1)}{\pa{\whid_j \odot x(t - 2)}}} \\
& = \abs{\uhid_j \cdot \uhid_j} = \abs{\uhid_j}^2  \\
\abs{\pd{\bar{h}_j(t)}{\pa{\whid_j \odot x(t - q)}}} & = \abs{\pd{\bar{h}_j(t)}{\pa{\whid_j \odot x(t - 2)}}} \\
& = \abs{\pd{\bar{h}_j(t)}{h_j(t - 1)} \cdot \pd{h_j(t - 1)}{\bar{h}_j(t - 1)} \cdot \pd{\bar{h}_j(t - 1)}{\pa{\whid_j \odot x(t - 2)}}} \\
& = \abs{\vhid_j \cdot f'\pa{\bar{h}(t - 1)} \cdot \pd{\bar{h}_j(t - 1)}{\pa{\whid_j \odot x(t - 2)}}} \\
& \leq \abs{\vhid_j \cdot M \cdot \vhid_j \cdot M} = \abs{\vhid_j \cdot M}^2
\end{align*} \tag{20}\label{20}
$$

根據歸納法，我們可以推得

$$
\begin{align*}
\abs{\pd{\tilde{h}_j(t)}{\pa{\whid_j \odot x(t - q)}}} & = \abs{\uhid_j}^q  \\
\abs{\pd{\bar{h}_j(t)}{\pa{\whid_j \odot x(t - q)}}} & \leq \abs{\vhid_j \cdot M}^q
\end{align*} \tag{21}\label{21}
$$

展開 $\eqref{15}$ 我們可以發現當 $i = 1, \dots, \opnorm(y)$ 時

$$
\begin{align*}
& \pd{y_i(t)}{x_k(t - q)} \\
& = \pd{y_i(t)}{(\wout_i \odot h(t))} \cdot \sum_{j = 1}^{\dhid} \br{\pd{(\wout_i \odot h(t))}{h_j(t)} \cdot \pd{h_j(t)}{x_k(t - q)}} \\
& = f'\pa{\wout_i \odot h(t)} \cdot \sum_{j = 1}^{\dhid} \br{\wout_{i j} \cdot \pd{h_j(t)}{x_k(t - q)}} \\
& = f'\pa{\wout_i \odot h(t)} \cdot \Bigg[\mathbb{1}[q = 0] \cdot \sum_{j = 1}^{\opnorm(h)} \pa{\wout_{i j} \cdot f'\pa{\whid_j \odot x(t)} \cdot x_k(t)} \\
& \quad + \sum_{j = \opnorm(h) + 1}^{\opnorm(h) + \oprecur{1}(h)} \pa{\wout_{i j} \cdot f'\pa{\tilde{h}_j(t)} \cdot \pd{\tilde{h}_j(t)}{(\whid_j \odot x(t - q))} \cdot \pd{(\whid_j \odot x(t - q))}{x_k(t - q)}} \\
& \quad + \sum_{j = \opnorm(h) + \oprecur{1}(h) + 1}^{\dhid} \pa{\wout_{i j} \cdot f'\pa{\bar{h}_j(t)} \cdot \pd{\bar{h}_j(t)}{(\whid_j \odot x(t - q))} \cdot \pd{(\whid_j \odot x(t - q))}{x_k(t - q)}}\Bigg] \\
& = f'\pa{\wout_i \odot h(t)} \cdot \Bigg[\mathbb{1}[q = 0] \cdot \sum_{j = 1}^{\opnorm(h)} \pa{\wout_{i j} \cdot f'\pa{\whid_j \odot x(t)} \cdot x_k(t)} \\
& \quad + \sum_{j = \opnorm(h) + 1}^{\opnorm(h) + \oprecur{1}(h)} \pa{\wout_{i j} \cdot f'\pa{\tilde{h}_j(t)} \cdot \pd{\tilde{h}_j(t)}{(\whid_j \odot x(t - q))} \cdot \whid_{j k}} \\
& \quad + \sum_{j = \opnorm(h) + \oprecur{1}(h) + 1}^{\dhid} \pa{\wout_{i j} \cdot f'\pa{\bar{h}_j(t)} \cdot \pd{\bar{h}_j(t)}{(\whid_j \odot x(t - q))} \cdot \whid_{j k}}\Bigg]
\end{align*} \tag{22}\label{22}
$$

因此結合 $\eqref{21}\eqref{22}$ 加上 $\eqref{17}$ 的假設我們可以推得 $\eqref{15}$ 的結論

$$
\begin{align*}
& q > 0 \\
\implies & \abs{\pd{y_i(t)}{x_k(t - q)}} \leq \abs{f'\pa{\wout_i \odot h(t)}} \cdot \Bigg[ \\
& \quad \abs{\sum_{j = \opnorm(h) + 1}^{\opnorm(h) + \oprecur{1}(h)} \pa{\wout_{i j} \cdot f'\pa{\tilde{h}_j(t)} \cdot \abs{\uhid_j}^q \cdot \whid_{j k}}} \\
& \quad + \abs{\sum_{j = \opnorm(h) + \oprecur{1}(h) + 1}^{\dhid} \pa{\wout_{i j} \cdot f'\pa{\bar{h}_j(t)} \cdot \abs{\vhid_j \cdot M}^q \cdot \whid_{j k}}}\Bigg] \\
\implies & \lim_{q \to \infty} \abs{\pd{y_i(t)}{x_k(t - q)}} = 0 \\
\implies & \lim_{q \to \infty} \pd{y_i(t)}{x_k(t - q)} = 0.
\end{align*}
$$

### 備註

同樣的推理也可以套用至 $\uout, \vout$。

- 輸入長度固定時，Local Feedback Multilayered Networks 使用固定輸入長度的 MLP 概念相同，但架構比 MLP 更廣義
- Local Feedback Multilayered Networks 可能不適合用來解決輸入長度不固定的問題
