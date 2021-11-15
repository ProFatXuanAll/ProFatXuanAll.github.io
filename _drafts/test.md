---
layout: post
title: Test
---

- [Paper Link](https://ieeexplore.ieee.org/abstract/document/6795963)
- [Book Link](https://link.springer.com/chapter/10.1007/978-3-642-24797-2_4)
- 時間：1997
- 期刊/會議名稱：Neural Computation
- 作者：Sepp Hochreiter, Jürgen Schmidhuber
- 目標：強化 RNN 架構

## 重點

- RNN 透過 backpropagation 學習速度慢
  - 這些演算法通常稱為 backpropagation throught time 或 real-time recurrent learning
  - 梯度會**爆炸**或**消失**
    - 梯度爆炸造成神經網路的**權重劇烈振盪**
    - 梯度消失造成**訓練時間慢長**
  - 無法解決輸入與輸出訊號間隔較長（long time lag）的問題
- LSTM 能夠解決上述問題
  - 能夠處理 time lag 間隔為 $1000$ 的問題
  - 甚至輸入訊號含有雜訊時也能處理
  - 同時能夠保有處理 short time lag 問題的能力
- 使用 mulitplicative gate 學習開啟與關閉記憶 hidden state 的機制
  - Forward pass 演算法複雜度為 $O(1)$
  - Backward pass 演算法複雜度為 $O(w)$，$w$ 是權重

## Keywords

- short-term memory
- long-term memory
- back propagation through time
- real-time recurrent learning
- recurrent cascade corrleation
- Elman nets
- neural sequence chunking
- incompressible input sequence
- constant error flow

## 傳統的 RNN

### BPTT

BPTT = Back Propagation Through Time，是用來計算 RNN 模型梯度的演算法。

模型的輸入來源共有兩種：

- 外部輸入 $x(t)$
  - 輸入維度 $d_{\text{in}}$
- 前一次的輸出 $y(t)$
  - 輸出維度 $d_{\text{out}}$
  - 注意這裡是使用 $t$ 不是 $t - 1$
- $t$ 的起始值為 $0$，結束值為 $T$，每次遞增 $1$
  - 時間為離散狀態
  - 方便起見令 $y(0) = 0$

令模型的參數為 $w$，則我們可以定義以下符號

$$
\begin{align}
\text{net}_i(t + 1) & = \sum_{j = 1}^{d_{\text{in}}} w_{ij} x_j(t) + \sum_{j = 1}^{d_{\text{out}}} w_{ij} y_j(t) \\
& = \sum_{j = 1}^{d_{\text{in}} + d_{\text{out}}} w_{ij} [x ; y]_j(t)
\end{align}
$$

- $\text{net}_i(t + 1)$ 代表第 $t + 1$ 時間的模型內部節點 $i$ 所收到的總輸入（net input）
- $w_{ij}$ 代表輸入節點 $j$（可以是外部輸入 $x_j(t)$ 或是前次輸出 $y_j(t)$）與模型內部節點 $i$ 所連接的權重
  - 總共有 $d_{\text{in}} + d_{\text{out}}$ 個輸入節點，因此 $1 \leq j \leq d_{\text{in}} + d_{\text{out}}$
  - 總共有 $d_{\text{out}}$ 個內部節點，因此 $1 \leq i \leq d_{\text{out}}$
- $w$ 的維度為 $\mathbf{R}^{d_{\text{out}} \times (d_{\text{in}} + d_{\text{out}})}$
- $[x;y]$ 代表將外部輸入與前次輸出串接在一起

如果模型使用的啟發函數（elementwise activation function）為 $f : \mathbf{R}^{d_{\text{out}}} \to \mathbf{R}^{d_{\text{out}}}$，則我們可以得到 $t + 1$ 時間的輸出

$$
y_i(t + 1) = f_i\big(\text{net}_i(t + 1)\big)
$$

- 使用 $f_i$ 的理由是每個維度所使用的啟發函數可以不同
- $f$ 必須要可以微分

如果 $t + 1$ 時間點的目標為 $\hat{y}(t + 1) \in \mathbf{R}^{d_{\text{out}}}$，則目標函數為最小平方差：

$$
\begin{align}
\text{loss}_i(t + 1) & = \frac{1}{2} \big(y_i(t + 1) - \hat{y}_i(t + 1)\big)^2 \\
\text{loss}(t + 1) & = \sum_{i = 1}^{d_{\text{out}}} \text{loss}_i(t + 1) \\
& = \frac{1}{2}\sum_{i = 1}^{d_{\text{out}}} \big(y_i(t + 1) - \hat{y}_i(t + 1)\big)^2
\end{align}
$$

我們可以輕易的計算 $\text{loss}_i(t + 1)$ 對 $\text{loss}(t + 1)$ 所得梯度

$$
\frac{\partial \text{loss}(t + 1)}{\partial \text{loss}_i(t + 1)} = 1 \tag{1}
$$

而 $\text{y}_i(t + 1)$ 對 $\text{loss}(t + 1)$ 所得梯度為

$$
\frac{\partial \text{loss}_i(t + 1)}{\partial y_i(t + 1)} = y_i(t + 1) - \hat{y}_i(t + 1) \tag{2}
$$

根據 (1)(2) 我們可以推得

$$
\begin{align}
\frac{\partial \text{loss}(t + 1)}{\partial y_i(t + 1)} & = \frac{\partial \text{loss}(t + 1)}{\partial \text{loss}_i(t + 1)} \frac{\partial \text{loss}_i(t + 1)}{\partial y_i(t + 1)} \\
& = 1 \times \big(y_i(t + 1) - \hat{y}_i(t + 1)\big) \\
& = y_i(t + 1) - \hat{y}_i(t + 1)
\end{align} \tag{3}
$$

$\text{net}_i(t + 1)$ 對 $\text{y}_i(t + 1)$ 所得梯度為

$$
\frac{\partial y_i(t + 1)}{\partial \text{net}_i(t + 1)} = f_i'\big(\text{net}_i(t + 1)\big) \tag{4}
$$

根據 (3)(4) 我們可以推得 $\text{net}_i(t + 1)$ 對 $\text{loss}(t + 1)$ 所得梯度

$$
\begin{align}
\frac{\partial \text{loss}(t + 1)}{\partial \text{net}_i(t + 1)} & = \frac{\partial \text{loss}(t + 1)}{\partial y_i(t + 1)} \frac{\partial y_i(t + 1)}{\partial \text{net}_i(t + 1)} \\
& = f_i'\big(\text{net}_i(t + 1)\big) \big(y_i(t + 1) - \hat{y}_i(t + 1)\big)
\end{align} \tag{5}
$$

根據 (5) 我們可以推得 $\text{x}_j(t)$ 對 $\text{loss}(t + 1)$ 所得梯度

$$
\begin{align}
& \frac{\partial \text{loss}(t + 1)}{\partial \text{x}_j(t)} \\
& = \sum_{i = 1}^{d_{\text{out}}} \frac{\partial \text{loss}(t + 1)}{\partial \text{net}_i(t + 1)} \frac{\partial \text{net}_i(t + 1)}{\partial x_j(t)} \\
& = \sum_{i = 1}^{d_{\text{out}}} f_i'\big(\text{net}_i(t + 1)\big) \big(y_i(t + 1) - \hat{y}_i(t + 1)\big) w_{i j}
\end{align} \tag{6}
$$

同樣的 $\text{y}_j(t)$ 對 $\text{loss}(t + 1)$ 所得梯度為

$$
\begin{align}
& \frac{\partial \text{loss}(t + 1)}{\partial \text{y}_j(t)} \\
& = \sum_{i = 1}^{d_{\text{out}}} \frac{\partial \text{loss}(t + 1)}{\partial \text{net}_i(t + 1)} \frac{\partial \text{net}_i(t + 1)}{\partial y_j(t)} \\
& = \sum_{i = 1}^{d_{\text{out}}} f_i'\big(\text{net}_i(t + 1)\big) \big(y_i(t + 1) - \hat{y}_i(t + 1)\big) w_{i j}
\end{align} \tag{7}
$$

由於 $y_j(t)$ 的計算是由 $\text{net}_j(t)$ 而來，所以我們也利用 (4)(7) 計算 $\text{net}_j(t)$ 對 $\text{loss}(t + 1)$ 所得梯度（注意是 $t$ 不是 $t + 1$）

$$
\begin{align}
& \frac{\partial \text{loss}(t + 1)}{\partial \text{net}_j(t)} \\
& = \frac{\partial \text{loss}(t + 1)}{\partial y_j(t)} \frac{\partial y_j(t)}{\partial \text{net}_j(t)} \\
& = \sum_{i = 1}^{d_{\text{out}}} f_i'\big(\text{net}_i(t + 1)\big) \big(y_i(t + 1) - \hat{y}_i(t + 1)\big) w_{i j} f_j'\big(\text{net}_j(t)\big) \\
& = f_j'\big(\text{net}_j(t)\big) \bigg(\sum_{i = 1}^{d_{\text{out}}} w_{i j} f_i'\big(\text{net}_i(t + 1)\big) \big(y_i(t + 1) - \hat{y}_i(t + 1)\big)\bigg) \\
& = f_j'\big(\text{net}_j(t)\big) \bigg(\sum_{i = 1}^{d_{\text{out}}} w_{i j} \frac{\partial \text{loss}(t + 1)}{\partial \text{net}_i(t + 1)}\bigg)
\end{align} \tag{8}
$$

模型參數 $w_{ij}$ 對於 $\text{loss}(t + 1)$ 所得梯度為

$$
\begin{align}
& \frac{\partial \text{loss}(t + 1)}{\partial w_{ij}} \\
& = \frac{\partial \text{loss}(t + 1)}{\partial \text{net}_{i}(t + 1)} \frac{\partial \text{net}_{i}(t + 1)}{\partial w_{ij}} \\
& = \begin{cases} f_i'\big(\text{net}_i(t + 1)\big) \big(y_i(t + 1) - \hat{y}_i(t + 1)\big) x_j(t) \\
f_i'\big(\text{net}_i(t + 1)\big) \big(y_i(t + 1) - \hat{y}_i(t + 1)\big) y_j(t)
\end{cases} && \text{(by (5))}
\end{align} \tag{9}
$$

注意最後一行等式取決於 $w_{ij}$ 與哪個輸入相接。
而在時間點 $t + 1$ 進行參數更新的方法為

$$
w_{ij} \leftarrow w_{ij} - \alpha \frac{\partial \text{loss}(t + 1)}{\partial w_{ij}} \tag{10}
$$

### 梯度爆炸 / 消失

從第 (8) 式我們可以進一步推得 $t$ 時間點造成的梯度與前次時間點 ($t - 1, t - 2, \dots$) 所得的梯度**變化關係**。

注意這裡的變化關係指的是梯度與梯度之間的**變化率**，意即用時間點 $t - 1$ 的梯度對時間點 $t$ 的梯度算微分。

為了方便計算，我們定義新的符號 $\vartheta_{k_0}^{t_1}[t_2]$，意思為以 $t_1$ 時間點開始往回走到 $t_2$ 時間點，在 $t_2$ 時間點的第 $k_0$ 個模型內部節點 $\text{net}_{k_0}(t_2)$ 對於 $t_1$ 時間點產生的 $\text{loss}(t_1)$ 計算所得之梯度。

因此下式如同 (5) 式

$$
\vartheta_{k_0}^t[t] = \frac{\partial \text{loss}(t)}{\partial \text{net}_{k_0}(t)} = f_{k_0}'\big(\text{net}_{k_0}(t)\big) \big(y_{k_0}(t) - \hat{y}_{k_0}(t)\big) \tag{11}
$$

由 (8) 與 (11) 我們可以往回推 1 個時間點

$$
\begin{align}
\vartheta_{k_1}^{t}[t - 1] & = \frac{\partial \text{loss}(t)}{\partial \text{net}_{k_1}(t - 1)} \\
& = f_{k_1}'\big(\text{net}_{k_1}(t - 1)\big) \cdot \bigg[\sum_{k_0 = 1}^{d_{\text{out}}} w_{k_0 k_1} \cdot \frac{\partial \text{loss}(t)}{\partial \text{net}_{k_0}(t)}\bigg] \\
& = \sum_{k_0 = 1}^{d_{\text{out}}} \bigg[w_{k_0 k_1} \cdot f_{k_1}'\big(\text{net}_{k_1}(t - 1)\big) \cdot \vartheta_{k_0}^t[t]\bigg]
\end{align} \tag{12}
$$

由 (12) 我們可以往回推 2 個時間點

$$
\begin{align}
& \vartheta_{k_2}^{t}[t - 2] \\
& = \frac{\partial \text{loss}(t)}{\partial \text{net}_{k_2}(t - 2)} \\
& = \sum_{k_1 = 1}^{d_{\text{out}}} \frac{\partial \text{loss}(t)}{\partial \text{net}_{k_1}(t - 1)} \cdot \frac{\partial \text{net}_{k_1}(t - 1)}{\partial \text{net}_{k_2}(t - 2)} \\
& = \sum_{k_1 = 1}^{d_{\text{out}}} \vartheta_{k_1}^t[t - 1] \cdot \frac{\partial \text{net}_{k_1}(t - 1)}{\partial y_{k_2}(t - 2)} \cdot \frac{\partial y_{k_2}(t - 2)}{\partial \text{net}_{k_2}(t - 2)} \\
& = \sum_{k_1 = 1}^{d_{\text{out}}} \vartheta_{k_1}^t[t - 1] \cdot w_{k_1 k_2} \cdot f_{k_2}'\big(\text{net}_{k_2}(t - 2)\big) \\
& = \sum_{k_1 = 1}^{d_{\text{out}}} \Bigg[f_{k_1}'\big(\text{net}_{k_1}(t - 1)\big) \cdot \bigg(\sum_{k_0 = 1}^{d_{\text{out}}} w_{k_0 k_1} \vartheta_{k_0}^t[t]\bigg)\Bigg] \cdot w_{k_1 k_2} \cdot f_{k_2}'\big(\text{net}_{k_2}(t - 2)\big) \\
& = \sum_{k_1 = 1}^{d_{\text{out}}} \sum_{k_0 = 1}^{d_{\text{out}}} \bigg[w_{k_0 k_1} \cdot w_{k_1 k_2} \cdot f_{k_1}'\big(\text{net}_{k_1}(t - 1)\big) \cdot f_{k_2}'\big(\text{net}_{k_2}(t - 2)\big) \cdot \vartheta_{k_0}^t[t]\bigg]
\end{align} \tag{13}
$$

由 (13) 我們可以往回推 3 個時間點

$$
\begin{align}
& \vartheta_{k_3}^{t}[t - 3] \\
& = \frac{\partial \text{loss}(t)}{\partial \text{net}_{k_3}(t - 3)} \\
& = \sum_{k_2 = 1}^{d_{\text{out}}} \frac{\partial \text{loss}(t)}{\partial \text{net}_{k_2}(t - 2)} \cdot \frac{\partial \text{net}_{k_2}(t - 2)}{\partial \text{net}_{k_3}(t - 3)} \\
& = \sum_{k_2 = 1}^{d_{\text{out}}} \vartheta_{k_2}^t[t - 2] \cdot \frac{\partial \text{net}_{k_2}(t - 2)}{\partial y_{k_3}(t - 3)} \cdot \frac{\partial y_{k_3}(t - 3)}{\partial \text{net}_{k_3}(t - 3)} \\
& = \sum_{k_2 = 1}^{d_{\text{out}}} \vartheta_{k_2}^t[t - 2] \cdot w_{k_2 k_3} \cdot f_{k_3}'\big(\text{net}_{k_3}(t - 3)\big) \\
& = \sum_{k_2 = 1}^{d_{\text{out}}} \bigg[\sum_{k_1 = 1}^{d_{\text{out}}} \sum_{k_0 = 1}^{d_{\text{out}}} w_{k_0 k_1} \cdot w_{k_1 k_2} \cdot f_{k_1}'\big(\text{net}_{k_1}(t - 1)\big) \cdot f_{k_2}'\big(\text{net}_{k_2}(t - 2)\big) \cdot \vartheta_{k_0}^t[t]\bigg] \\
& \quad \cdot w_{k_2 k_3} \cdot f_{k_3}'\big(\text{net}_{k_3}(t - 3)\big) \\
& = \sum_{k_2 = 1}^{d_{\text{out}}} \sum_{k_1 = 1}^{d_{\text{out}}} \sum_{k_0 = 1}^{d_{\text{out}}} \bigg[w_{k_0 k_1} \cdot w_{k_1 k_2} \cdot w_{k_2 k_3} \cdot \\
& \quad f_{k_1}'\big(\text{net}_{k_1}(t - 1)\big) \cdot f_{k_2}'\big(\text{net}_{k_2}(t - 2)\big) \cdot f_{k_3}'\big(\text{net}_{k_3}(t - 3)\big) \cdot \vartheta_{k_0}^t[t]\bigg] \\
& = \sum_{k_2 = 1}^{d_{\text{out}}} \sum_{k_1 = 1}^{d_{\text{out}}} \sum_{k_0 = 1}^{d_{\text{out}}} \Bigg[\bigg[\prod_{q = 1}^3 w_{k_{q - 1} k_q} \cdot f_{k_q}'\big(\text{net}_{k_q}(t - q)\big)\bigg] \cdot \vartheta_{k_0}^t[t]\Bigg]
\end{align} \tag{14}
$$

由 (12)(13)(14) 我們可以歸納以下結論：
若 $n \geq 1$，則往回推 $n$ 個時間點的公式為

$$
\begin{align}
& \vartheta_{k_n}^{t}[t - n] \\
& = \sum_{k_{n - 1} = 1}^{d_{\text{out}}} \cdots \sum_{k_0 = 1}^{d_{\text{out}}} \Bigg[\bigg[\prod_{q = 1}^n w_{k_{q - 1} k_q} \cdot f_{k_q}'\big(\text{net}_{k_q}(t - q)\big)\bigg] \cdot \vartheta_{k_0}^t[t]\Bigg] \tag{15}
\end{align}
$$

由 (15) 我們可以看出所有的 $\vartheta_{k_n}^{t}[t - n]$ 都與 $\vartheta_{k_0}^{t}[t]$ 相關，因此我們將 $\vartheta_{k_n}^{t}[t - n]$ 想成由 $\vartheta_{k_0}^{t}[t]$ 構成的函數。

現在讓我們固定 $k_0^{*} \in \{1, \dots, d_{\text{out}}\}$，我們可以計算 $\vartheta_{k_0^{*}}^{t}[t]$ 對於 $\vartheta_{k_n^{*}}^{t}[t - n]$ 的微分

當 $n = 1$ 時，根據 (12) 我們可以推得論文中的 (3.1) 式

$$
\frac{\partial \vartheta_{k_n}^{t}[t - n]}{\partial \vartheta_{k_0^{*}}^{t}[t]} = w_{k_0^{*} k_1} \cdot f_{k_1}'\big(\text{net}_{k_1}(t - 1)\big) \tag{16}
$$

當 $n > 1$ 時，根據 (15) 我們可以推得論文中的 (3.2) 式

$$
\frac{\partial \vartheta_{k_n}^{t}[t - n]}{\partial \vartheta_{k_0^{*}}^{t}[t]} = \sum_{k_{n - 1} = 1}^{d_{\text{out}}} \cdots \sum_{k_1 = 1}^{d_{\text{out}}} \sum_{k_0 \in \{k_0^*\}} \bigg[\prod_{q = 1}^n w_{k_{q - 1} k_q} \cdot f_{k_q}'\big(\text{net}_{k_q}(t - q)\big)\bigg] \tag{17}
$$

**注意錯誤**：論文中的 (3.2) 式不小心把 $w_{l_{m - 1} l_m}$ 寫成 $w_{l_m l_{m - 1}}$。

因此根據 (17)，共有 $(d_{\text{out}})^{n - 1}$ 個連乘積項次進行加總，所得結果會以 (9)(10) 直接影響權種更新 $w$。

根據 (16)(17)，如果

$$
\left| w_{k_{q - 1} k_q} \cdot f_{k_q}'\big(\text{net}_{k_q}(t - q)\big) \right| > 1.0 \quad \forall q = 1, \dots, n \tag{18}
$$

則 $w$ 的梯度會以指數 $n$ 增加，直接導致**梯度爆炸**，參數會進行**劇烈的振盪**，無法進行順利更新。

而如果

$$
\left| w_{k_{q - 1} k_q} \cdot f_{k_q}'\big(\text{net}_{k_q}(t - q)\big) \right| < 1.0 \quad \forall q = 1, \dots, n \tag{19}
$$

則 $w$ 的梯度會以指數 $n$ 縮小，直接導致**梯度消失**，誤差**收斂速度**會變得**非常緩慢**。

如果 $f_{k_q}$ 是 sigmoid function，則 $f_{k_q}'$ 最大值為 $0.25$，理由是

$$
\begin{align}
f_{k_q}(x) & = \sigma(x) = \frac{1}{1 + e^{-x}} \\
f_{k_q}'(x) & = \sigma'(x) = \frac{e^{-x}}{(1 + e^{-x})^2} = \frac{1}{1 + e^{-x}} \frac{e^{-x}}{1 + e^{-x}} \\
& = \frac{1}{1 + e^{-x}} \frac{1 + e^{-x} - 1}{1 + e^{-x}} = \sigma(x) (1 - \sigma(x)) \\
\sigma(\mathbf{R}) & = (0, 1) \\
\max_{x \in \mathbf{R}} f_{k_q}'(x) & = \sigma(0) * (1 - \sigma(0)) = 0.5 * 0.5 = 0.25
\end{align} \tag{20}
$$

因此當 $|w_{k_{q - 1} k_{q}}| < 4.0$ 時我們可以發現

$$
\left| w_{k_{q - 1} k_{q}} \cdot f_{k_q}'\big(\text{net}_{k_q}(t - q)\big) \right| < 4.0 * 0.25 = 1.0 \tag{21}
$$

所以 (21) 與 (19) 的結論相輔相成：當 $w_{k_{q - 1} k_{q}}$ 的絕對值小於 $4.0$ 會造成梯度消失。

而 $|w_{k_{q - 1} k_{q}}| \to \infty$ 我們可以得到

$$
\begin{align}
& \left| \text{net}_{k_{q - 1}}(t - q - 1) \right| \to \infty \\
\implies & \begin{cases}
f_{k_{q - 1}}\big(\text{net}_{k_{q - 1}}(t - q - 1)\big) \to 1 & \text{if } \text{net}_{k_{q - 1}}(t - q - 1) \to \infty \\
f_{k_{q - 1}}\big(\text{net}_{k_{q - 1}}(t - q - 1)\big) \to 0 & \text{if } \text{net}_{k_{q - 1}}(t - q - 1) \to -\infty
\end{cases} \\
\implies & \left| f_{k_{q - 1}}'\big(\text{net}_{k_{q - 1}}(t - q - 1)\big) \right| \to 0 \\
\implies & \left| \prod_{q = 1}^n w_{k_{q - 1} k_{q}} \cdot f_{k_q}'\big(\text{net}_{k_q}(t - q)\big) \right| \to 0
\end{align} \tag{22}
$$

**注意錯誤**：論文中的推論

$$
\left| w_{k_{q - 1} k_{q}} \cdot f_{k_q}'\big(\text{net}_{k_q}(t - q)\big) \right| \to 0
$$

是**錯誤**的，理由是 $w_{k_{q - 1} k_q}$ 無法對 $\text{net}_{k_q}(t - q)$ 造成影響，作者不小心把時間順序寫反了，但是最後的邏輯仍然正確，理由如 (22) 所示。

**注意錯誤**：論文中進行了以下函數最大值的推論

$$
\begin{align}
& f'_{l_m}\big(\text{net}_{l_m}(t - m)\big) w_{l_m l_{m - 1}} \\
& = \sigma\big(\text{net}_{l_m}(t - m)\big) \cdot \Big(1 - \sigma\big(\text{net}_{l_m}(t - m)\big)\Big) \cdot w_{l_m l_{m - l}}
\end{align}
$$

最大值發生於微分值為 $0$ 的點，即我們想求出滿足以下式子的 $w_{l_m l_{m - 1}}$

$$
\frac{\partial \Big[\sigma\big(\text{net}_{l_m}(t - m)\big) \cdot \Big(1 - \sigma\big(\text{net}_{l_m}(t - m)\big)\Big) \cdot w_{l_m l_{m - l}}\Big]}{\partial w_{l_m l_{m - 1}}} = 0
$$

拆解微分式可得

$$
\begin{align}
& \frac{\partial \Big[\sigma\big(\text{net}_{l_m}(t - m)\big) \cdot \Big(1 - \sigma\big(\text{net}_{l_m}(t - m)\big)\Big) \cdot w_{l_m l_{m - l}}\Big]}{\partial w_{l_m l_{m - 1}}} \\
& = \frac{\partial \sigma\big(\text{net}_{l_m}(t - m)\big)}{\partial \text{net}_{l_m}(t - m)} \cdot \frac{\partial \text{net}_{l_m}(t - m)}{\partial w_{l_m l_{m - 1}}} \cdot \Big(1 - \sigma\big(\text{net}_{l_m}(t - m)\big)\Big) \cdot w_{l_m l_{m - l}} \\
& \quad + \sigma\big(\text{net}_{l_m}(t - m)\big) \cdot \frac{\partial \Big(1 - \sigma\big(\text{net}_{l_m}(t - m)\big)\Big)}{\partial \text{net}_{l_m}(t - m)} \cdot \frac{\partial \text{net}_{l_m}(t - m)}{\partial w_{l_m l_{m - 1}}} \cdot w_{l_m l_{m - l}} \\
& \quad + \sigma\big(\text{net}_{l_m}(t - m)\big) \cdot \Big(1 - \sigma\big(\text{net}_{l_m}(t - m)\big)\Big) \cdot \frac{\partial w_{l_m l_{m - 1}}}{\partial w_{l_m l_{m - 1}}} \\
& = \sigma\big(\text{net}_{l_m}(t - m)\big) \cdot \Big(1 - \sigma\big(\text{net}_{l_m}(t - m)\big)\Big)^2 \cdot y_{l_{m - 1}}(t - m - 1) \cdot w_{l_m l_{m - 1}} \\
& \quad - \Big(\sigma\big(\text{net}_{l_m}(t - m)\big)\Big)^2 \cdot \Big(1 - \sigma\big(\text{net}_{l_m}(t - m)\big)\Big) \cdot y_{l_{m - 1}}(t - m - 1) \cdot w_{l_m l_{m - 1}} \\
& \quad + \sigma\big(\text{net}_{l_m}(t - m)\big) \cdot \Big(1 - \sigma\big(\text{net}_{l_m}(t - m)\big)\Big) \\
& = \Big[2 \Big(\sigma\big(\text{net}_{l_m}(t - m)\big)\Big)^3 - 3 \Big(\sigma\big(\text{net}_{l_m}(t - m)\big)\Big)^2 + \sigma\big(\text{net}_{l_m}(t - m)\big)\Big] \cdot \\
& \quad \quad y_{l_{m - 1}}(t - m - 1) \cdot w_{l_m l_{m - 1}} \\
& \quad + \sigma\big(\text{net}_{l_m}(t - m)\big) \cdot \Big(1 - \sigma\big(\text{net}_{l_m}(t - m)\big)\Big) \\
& = \sigma\big(\text{net}_{l_m}(t - m)\big) \cdot \Big(2 \sigma\big(\text{net}_{l_m}(t - m)\big) - 1\Big) \Big(\sigma\big(\text{net}_{l_m}(t - m)\big) - 1\Big) \cdot \\
& \quad \quad y_{l_{m - 1}}(t - m - 1) \cdot w_{l_m l_{m - 1}} \\
& \quad + \sigma\big(\text{net}_{l_m}(t - m)\big) \cdot \Big(1 - \sigma\big(\text{net}_{l_m}(t - m)\big)\Big) \\
& = 0
\end{align}
$$

移項後可以得到

$$
\begin{align}
& \sigma\big(\text{net}_{l_m}(t - m)\big) \cdot \Big(2 \sigma\big(\text{net}_{l_m}(t - m)\big) - 1\Big) \Big(1 - \sigma\big(\text{net}_{l_m}(t - m)\big)\Big) \cdot \\
& \quad \quad y_{l_{m - 1}}(t - m - 1) \cdot w_{l_m l_{m - 1}} = \sigma\big(\text{net}_{l_m}(t - m)\big) \cdot \Big(1 - \sigma\big(\text{net}_{l_m}(t - m)\big)\Big) \\
\implies & \Big(2 \sigma\big(\text{net}_{l_m}(t - m)\big) - 1\Big) \cdot y_{l_{m - 1}}(t - m - 1) \cdot w_{l_m l_{m - 1}} = 1 \\
\implies & w_{l_m l_{m - 1}} = \frac{1}{y_{l_{m - 1}}(t - m - 1)} \cdot \frac{1}{2 \sigma\big(\text{net}_{l_m}(t - m)\big) - 1} \\
\implies & w_{l_m l_{m - 1}} = \frac{1}{y_{l_{m - 1}}(t - m - 1)} \cdot \coth\bigg(\frac{\text{net}_{l_m}(t - m)}{2}\bigg)
\end{align}
$$

註：推論中使用了以下公式

$$
\begin{align}
\tanh(x) & = 2 \sigma(2x) - 1 \\
\tanh(\frac{x}{2}) & = 2 \sigma(x) - 1 \\
\coth(\frac{x}{2}) & = \frac{1}{\tanh(\frac{x}{2})} = \frac{1}{2 \sigma(x) - 1}
\end{align}
$$

但公式的前提不對，理由是 $w_{l_m l_{m - 1}}$ 根本不存在，應該改為 $w_{l_{m - 1} l_m}$ 同 (17)。

接著我們推導時間點 $t - n$ 的節點 $\text{net}_{k_n}(t - n)$ 針對 $t$ 時間點造成的總誤差梯度變化：

$$
\sum_{k_0^* = 1}^{d_{\text{out}}} \frac{\partial \vartheta_{k_n}^t[t - n]}{\partial \vartheta_{k_0^*}^t[t]} \tag{23}
$$

由於每個項次都能遭遇梯度消失，因此總和也會遭遇梯度消失。

## 梯度常數 (Constant Error Flow)

將**部份梯度**定為**常數**

- 想法有點矛盾：
  - 梯度應該隨著最佳化 (梯度下降) 的過程逐漸縮小數值
  - 但遇到了梯度消失的問題，因此要求梯度維持常數
  - 需要讓梯度隨著時間變小，卻又要求梯度維持常數，看起來互相**矛盾**

### 情境 1：模型輸出與內部節點 1-1 對應

假設模型輸出節點 $y_j(t - 1)$ 只與 $\text{net}_j(t)$ 相連，即

$$
\text{net}_j(t) = w_{j j} y_j(t - 1) \tag{24}
$$

((24) 假設實際上不可能發生) 則根據式子 (12) 我們可以推得

$$
\vartheta_j^t[t - 1] = w_{j j} \cdot f_j'\big(\text{net}_j(t - 1)\big) \cdot \vartheta_j^t[t] \tag{25}
$$

為了強制讓梯度 $\vartheta_j^t[t]$ 不消失，作者認為需要強制達成

$$
w_{j j} \cdot f_j'\big(\text{net}_j(t - 1)\big) = 1.0 \tag{26}
$$

如果 (26) 能夠達成，則積分 (26) 可以得到

$$
\begin{align}
& \int w_{j j} \cdot f_j'\big(\text{net}_j(t - 1)\big) \; d \text{net}_j(t - 1) = \int 1.0 \; d \text{net}_j(t - 1) \\
\implies & w_{j j} \cdot f_j\big(\text{net}_j(t - 1)\big) = \text{net}_j(t - 1) \\
\implies & y_j(t - 1) = f_j\big(\text{net}_j(t - 1)\big) = \frac{\text{net}_j(t - 1)}{w_{j j}}
\end{align} \tag{27}
$$

觀察 (27) 我們可以發現

- 輸入 $\text{net}_j(t - 1)$ 與輸出 $f_j\big(\text{net}_j(t - 1)\big)$ 之間的關係是乘上一個常數項 $w_{j j}$
- 代表函數 $f_j$ 其實是一個**線性函數**
- **每個時間點**的**輸出**居然**完全相同**，這個現象稱為 **Constant Error Carousel** (請見 (28))

$$
\begin{align}
y_j(t) & = f_j\big(\text{net}_j(t)\big) = f_j\big(w_{j j} y_j(t - 1)\big) \\
& = f_j\big(w_{j j} \frac{\text{net}_j(t - 1)}{w_{j j}}\big) = f_j\big(\text{net}_j(t - 1)\big) = y_j(t - 1) \tag{28}
\end{align}
$$

### 情境 2：增加外部輸入

將 (24) 的假設改成每個模型內部節點可以額外接收一個外部輸入

$$
\text{net}_j(t) = \sum_{i = 1}^{d_{\text{in}}} w_{j i} x_i(t - 1) + w_{j j} y_j(t - 1) \tag{29}
$$

由於 $y_j(t - 1)$ 的設計功能是保留過去計算所擁有的資訊，在 (29) 的假設中唯一能夠**更新**資訊的方法只有透過 $x_i(t - 1)$ 配合 $w_{j i}$ 將新資訊合併進入 $\text{net}_j(t)$。

但作者認為，在計算的過程中，部份時間點的**輸入**資訊 $x_i(*)$ 可以(甚至必須)被**忽略**，但這代表 $w_{j i}$ 需要**同時**達成**兩種**任務就必須要有**兩種不同的數值**：

- **加入新資訊**：代表 $|w_{j i}| \neq 0$
- **忽略新資訊**：代表 $|w_{j i}| \approx 0$

因此**無法只靠一個** $w_{j i}$ 決定**輸入**的影響，必須有**額外**能夠**理解當前內容 (context-sensitive)** 的功能模組幫忙**寫入** $x_i(*)$

### 情境 3：輸出回饋到多個節點

將 (24)(29) 的假設改回正常的模型架構

$$
\begin{align}
\text{net}_j(t) & = \sum_{i = 1}^{d_{\text{in}}} w_{j i} x_i(t - 1) + \sum_{i = 1}^{d_{\text{out}}} w_{j i} y_i(t - 1) \\
& = \sum_{i = 1}^{d_{\text{in}}} w_{j i} x_i(t - 1) + \sum_{i = 1}^{d_{\text{out}}} w_{j i} f_i\big(\text{net}_i(t - 1)\big)
\end{align} \tag{30}
$$

由於 $y_j(t - 1)$ 的設計功能是保留過去計算所擁有的資訊，在 (30) 的假設中唯一能夠讓**過去**資訊**影響未來**計算結果的方法只有透過 $y_i(t - 1)$ 配合 $w_{j i}$ 將新資訊合併進入 $\text{net}_j(t)$。

但作者認為，在計算的過程中，部份時間點的**輸出**資訊 $y_i(*)$ 可以(甚至必須)被**忽略**，但這代表 $w_{j i}$ 需要**同時**達成**兩種**任務就必須要有**兩種不同的數值**：

- **保留過去資訊**：代表 $|w_{j i}| \neq 0$
- **忽略過去資訊**：代表 $|w_{j i}| \approx 0$

因此**無法只靠一個** $w_{j i}$ 決定**輸出**的影響，必須有**額外**能夠**理解當前內容 (context-sensitive)** 的功能模組幫忙**讀取** $y_i(*)$

值得一提的是，上述的假設是基於以下的事實觀察：
已知 RNN 能夠學習解決多個記憶時間較短 (short-time-lag) 的任務，但如果要能夠同時解決記憶時間較長 (long-time-lag) 的任務，則模型應該依照以下順序執行：

1. 記住短期資訊 $t_0 \sim t_1$ (需要寫入功能)
2. 解決需要短期資訊 $t_0 \sim t_1$ 的任務 (需要讀取功能)
3. 忘記短期資訊 $t_0 \sim t_1$ (需要忽略功能)
4. 記住短期資訊 $t_1 \sim t_2$ (需要寫入功能)
5. 解決需要短期資訊 $t_1 \sim t_2$ 的任務 (需要讀取功能)
6. 忘記短期資訊 $t_1 \sim t_2$ (需要忽略功能)
7. 為了解決與短期資訊 $t_0 \sim t_1$ 相關的任務，突然又需要回憶起短期資訊 $t_0 \sim t_1$ (需要寫入 + 讀取功能)
