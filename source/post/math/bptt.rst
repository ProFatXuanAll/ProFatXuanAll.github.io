=============================
Back-Propagation Through Time
=============================

.. ====================================================================================================================
.. Setup SEO.
.. ====================================================================================================================

.. meta::
  :description:
    推導 BPTT
  :keywords:
    BPTT,
    Gradient Descent,
    Optimization,
    RNN

.. ====================================================================================================================
.. Setup front matter.
.. ====================================================================================================================

.. article-info::
  :author: ProFatXuanAll
  :date: 2023-08-16
  :class-container: sd-p-2 sd-outline-muted sd-rounded-1

.. ====================================================================================================================
.. Create visible tags from SEO keywords.
.. ====================================================================================================================

:bdg-secondary:`BPTT`
:bdg-secondary:`Gradient Descent`
:bdg-secondary:`Optimization`
:bdg-secondary:`RNN`

.. ====================================================================================================================
.. Define math macros.
.. ====================================================================================================================

.. math::
  :nowrap:

  \[
    % Vector notations.
    \newcommand{\vh}{\mathbf{h}}
    \newcommand{\vx}{\mathbf{x}}
    \newcommand{\vy}{\mathbf{y}}
    \newcommand{\vyh}{\hat{\mathbf{y}}}
    \newcommand{\vz}{\mathbf{z}}

    % Matrix notations.
    \newcommand{\vW}{\mathbf{W}}

    % Symbols in mathcal.
    \newcommand{\cL}{\mathcal{L}}
    \newcommand{\cM}{\mathcal{M}}
    \newcommand{\cT}{\mathcal{T}}

    % Symbols with subscripts.
    \newcommand{\vxj}{{\vx_j}}
    \newcommand{\vyi}{{\vy_i}}
    \newcommand{\vyhi}{{\vyh_i}}
    \newcommand{\vzi}{{\vz_i}}

    % Symbols with star.
    \newcommand{\ts}{{t^\star}}
    \newcommand{\vhs}{{\vh^\star}}
    \newcommand{\vxs}{{\vx^\star}}
    \newcommand{\vys}{{\vy^\star}}
    \newcommand{\vyis}{{\vy_i^\star}}
    \newcommand{\vyhs}{{\vyh^\star}}
    \newcommand{\vyhis}{{\vyh_i^\star}}
    \newcommand{\vzs}{{\vz^\star}}
    \newcommand{\vzis}{{\vz_i^\star}}
    \newcommand{\vWs}{{\vW^\star}}

    % Dimensions.
    \newcommand{\din}{{d_{\operatorname{in}}}}
    \newcommand{\dout}{{d_{\operatorname{out}}}}
  \]

基本符號定義
============

- 所有向量（vector）都以粗體表示，例如：:math:`\vx`
- 所有向量討論僅限於實數空間，例如 :math:`\R^3`
- The :math:`i`\-th coordinate of a vector 以下標表示，例如：:math:`\vx_i`
- 所有向量皆為行向量（column vector），除非特別表示
- 所有矩陣（matrix）都以大寫粗體表示，例如：:math:`\vW`
- The :math:`i`\-th row and the :math:`j`\-th column of a matrix 以下標表示，例如：:math:`\vW_{i, j}`
- 矩陣乘法以 :math:`\cdot` 表示，例如：:math:`\vW \cdot \vx`

RNN 計算定義
============

給定一 :term:`RNN` 模型與參數 :math:`w`，給定輸入序列 :math:`\vx` 與答案序列 :math:`\vyh`。
我們希望輸入序列 :math:`\vx` 與參數 :math:`w` 經由 RNN 演算法得到的輸出序列 :math:`\vy` 會近似於答案序列 :math:`\vyh`。

假定輸入序列 :math:`x` 的長度為 :math:`\cT`，則我們可定義 RNN 的 :term:`forward pass`：

.. math::
  :nowrap:

  \[
    \begin{align*}
      & \algoProc{\operatorname{RNN}}(\vx, \vW, \cT) \\
      & \indent{1} \vy(0) \algoEq \zv \\
      & \indent{1} \algoFor{t \in \Set{0, \dots, \cT - 1}} \\
      & \indent{2} \vz(t + 1) \algoEq \vW \cdot \mqty[\vx(t) \\ \vy(t)] \\
      & \indent{2} \vy(t + 1) \algoEq f\qty(\vz(t + 1)) \\
      & \indent{1} \algoEndFor \\
      & \indent{1} \algoReturn \vy(1), \dots, \vy(\cT) \\
      & \algoEndProc
    \end{align*}
  \]

上述演算法的符號定義如下：

- 定義 :math:`\vx(t)` 為輸入序列 :math:`\vx` 中，時間點 :math:`t` 所對應到的資料

  - 令 :math:`t \in \Set{0, 1, \dots, \cT - 1}`
  - 定義 :math:`\vx(t)` 為向量，由 :math:`\din` 個實數組成，即 :math:`\vx(t) \in \R^\din`

- 定義 :math:`\vyh(t)` 為答案序列 :math:`\vyh` 中，時間點 :math:`t` 所對應到的資料

  - 令 :math:`t \in \Set{1, 2, \dots, \cT}`，注意此處定義與 :math:`\vx(t)` 的 index 範圍不同
  - 定義 :math:`\vyh(t)` 為向量，由 :math:`\dout` 個實數組成，即 :math:`\vyh(t) \in \R^\dout`

- 定義 :math:`\vy(t)` 為 RNN 輸出序列 :math:`\vy` 中，時間點 :math:`t` 所對應到的資料

  - 由於目標是讓 :math:`\vy \approx \vyh`，因此 :math:`\vy(t) \in \R^\dout`
  - 定義 :math:`t \in \Set{1, 2, \dots, \cT}`

- 定義常數 :math:`\vy(0) = \zv`

  - :math:`\zv` 是由 :math:`\dout` 個零組成的零向量
  - 注意此定義並無與 :math:`\vy(1), \dots, \vy(\cT)` 衝突

- 定義 :math:`\vW` 為 RNN 模型的參數

  - 定義 :math:`\vW` 為一矩陣，由 :math:`\dout \times (\din + \dout)` 個實數組成，即 :math:`\vW \in \R^{\dout \times (\din + \dout)}`

- 定義 :math:`\vz(t)` 為 RNN 模型在時間點 :math:`t` 得到的 net input

  - 定義 :math:`t \in \Set{1, 2, \dots, \cT}`
  - RNN 模型的 net input 來源為輸入 :math:`\vx(t - 1)` 與前一次的模型輸出 :math:`\vy(t - 1)`

- 定義 :math:`f` 為 RNN 模型的 :term:`activation function`

  - 定義 :math:`f_i` 為 :math:`f` 的第 :math:`i` 個 real valued function，:math:`i \in \Set{1, \dots, \dout}`
  - :math:`f` 必須要可以\ **微分**，每個 :math:`f_i` 所使用的 activation function 可以\ **不同**，但都只用 :math:`\vzi(t + 1)` 作為輸入

透過以上符號我們可以拆解矩陣乘法：

.. math::
  :nowrap:

  \[
    \begin{align*}
      & \algoProc{\operatorname{RNN}}(x, \cT) \\
      & \indent{1} \vy(0) \algoEq \zv \\
      & \indent{1} \algoFor{t \in \Set{0, \dots, \cT - 1}} \\
      & \indent{2} \algoFor{i \in \Set{1, \dots, \dout}} \\
      & \indent{3} \vzi(t + 1) \algoEq \sum_{j = 1}^\din \vW_{i, j} \cdot \vx_j(t) + \sum_{j = \din + 1}^{\din + \dout} \vW_{i, j} \cdot \vy_j(t) \\
      & \indent{3} \vyi(t + 1) \algoEq f_i(\vzi(t + 1)) \\
      & \indent{2} \algoEndFor \\
      & \indent{1} \algoEndFor \\
      & \indent{1} \algoReturn \vy(1), \dots, \vy(\cT) \\
      & \algoEndProc
    \end{align*}
  \]

目標函數
=========

定義 :math:`L : \R^\dout \times \R^\dout \to \R` 代表\ **最小平方差**。
假設每個時間點的誤差計算法為最小平方差，則 :math:`t + 1` 時間點的誤差可以表達為

.. math::
  :nowrap:

  \[
    L(\vy(t + 1), \vyh(t + 1)) = \frac{1}{2} \sum_{i = 1}^\dout \qty[\vyi(t + 1) - \vyhi(t + 1)]^2. \tag{1}\label{1}
  \]

而目標函數（objective function） :math:`\cL : \R^\dout \times \R^\dout \to \R` 的定義如下

.. math::
  :nowrap:

  \[
    \cL(\vy, \vyh) = \sum_{t = 0}^{\cT - 1} L(\vy(t + 1), \vyh(t + 1)). \tag{2}\label{2}
  \]

對目標函數微分
==============

為了將 forward pass 中使用的符號與微分計算對象區隔，我們需要定義以下符號：

- 令 :math:`(\vxs, \vyhs)` 為一真實資料點
- 令 RNN 模型目前使用的參數為 :math:`\vWs`
- 令 :math:`t \in \Set{0, \dots, \cT - 1}`
- 令 :math:`i \in \Set{1, \dots, \dout}`
- 令 :math:`j \in \Set{1, \dots, \din}`
- 假設 RNN forward pass 演算法產生的 net inputs 為 :math:`\vzs(1), \dots, \vzs(\cT)`
- 假設 RNN forward pass 演算法產生的 輸出序列為 :math:`\vys(1), \dots, \vys(\cT)`

根據目標函數 :math:`\eqref{2}` 的定義，我們計算 :math:`\vyi(t + 1)` 對 :math:`\cL(\vy, \vyh)` 的微分可得：

.. math::
  :nowrap:

  \[
    \begin{align*}
      & \eval{\pdv{\cL(\vy, \vyh)}{\vyi(t + 1)}}_{\vy = \vys, \vyh = \vyhs} \\
      & = \qty[\sum_{s = 0}^{\cT - 1} \eval{\pdv{L(\vy(s + 1), \vyh(s + 1))}{\vyi(t + 1)}}_{\vy(s + 1) = \vys(s + 1), \vyh(s + 1) = \vyhs(s + 1)}] \\
      & = \eval{\pdv{L(\vy(t + 1), \vyh(t + 1))}{\vyi(t + 1)}}_{\vy(t + 1) = \vys(t + 1), \vyh(t + 1) = \vyhs(t + 1)} \\
      & = \qty[\frac{1}{2} \sum_{k = 1}^\dout \eval{\pdv{\qty[\vy_k(t + 1) - \vyh_k(t + 1)]^2}{\vyi(t + 1)}}_{\vy(t + 1) = \vys(t + 1), \vyh(t + 1) = \vyhs(t + 1)}] \\
      & = \qty[\frac{1}{2} \eval{\pdv{\qty[\vyi(t + 1) - \vyhi(t + 1)]^2}{\vyi(t + 1)}}_{\vy(t + 1) = \vys(t + 1), \vyh(t + 1) = \vyhs(t + 1)}] \\
      & = \eval{\qty[\vyi(t + 1) - \vyhi(t + 1)]}_{\vyi(t + 1) = \vyis(t + 1), \vyhi(t + 1) = \vyhis(t + 1)} \\
      & = \vyis(t + 1) - \vyhis(t + 1).
    \end{align*} \tag{3}\label{3}
  \]

由於 :math:`\vy(t + 1)` 是由 :math:`\vz(t + 1)` 產生，我們可以定義新的函數 :math:`L^z : \R^\dout \times \R^\dout \to \R` 用來描述 :math:`\vz(t + 1)` 對 :math:`\cL(\vy, \vyh)` 的貢獻
根據 :math:`\eqref{3}` 我們可以推得 :math:`\vzi(t + 1)` 對 :math:`\cL(\vy, \vyh)` 的微分：

.. math::
  :nowrap:

  \[
    \begin{align*}
      \eval{\pdv{\cL}{z_i(t + 1)}}_{z_i(t + 1), \vyhi(t + 1)}
      & = \eval{\pdv{\cL}{y_i(t + 1)}}_{y_i(t + 1), \vyhi(t + 1)}
        \cdot
        \eval{\dv{y_i(t + 1)}{z_i(t + 1)}}_{z_i(t + 1)} \\
      & = \qty[y_i(t + 1) - \vyhi(t + 1)] \cdot f'\qty(z_i(t + 1)) \\
      & = \sigma'\qty(z_i(t + 1)) \cdot \qty[y_i(t + 1) - \vyhi(t + 1)].
    \end{align*}
  \]

.. note::

  式子 :math:`\eqref{2}` 就是論文 3.1.1 節的第一條公式。

根據 :math:`\eqref{2}` 我們可以推得 :math:`y_j(t)` 對 :math:`\cL` 的微分（注意時間差）：

.. math::
  :nowrap:

  \[
    \begin{align*}
      \eval{\pdv{\cL}{y_j(t)}}_{y_j(t), w_{i, j}, \vyhi(t + 1)} & = \sum_{i = 1}^{\dout} \qty[\eval{\pdv{\cL}{z_i(t + 1)}}_{z_i(t + 1), \vyhi(t + 1)} \cdot \eval{\pdv{z_i(t + 1)}{y_j(t)}}_{y_j(t)}] \\
                                                                               & = \sum_{i = 1}^{\dout} \qty[\sigma'\qty(z_i(t + 1)) \cdot \qty(y_i(t + 1) - \vyhi(t + 1)) \cdot w_{i, j}].
    \end{align*}
  \]

由於 :math:`\vy(t)` 是由 :math:`\opnet(t)` 計算而來，所以我們也利用 :math:`\eqref{3}` 計算 :math:`\net{j}{t}` 對 :math:`\cL` 的微分：

.. math::
  :nowrap:

  \[
    \begin{align*}
    \eval{\pdv{\cL}{\net{j}{t}}}_{\net{j}{t}} & = \eval{\pdv{\cL}{y_j(t)}}_{y_j(t)} \cdot \eval{\dv{y_j(t)}{\net{j}{t}}}_{\net{j}{t}} \\
                                                        & = \qty[\sum_{i = 1}^{\dout} \pdv{\cL}{z_i(t + 1)} \cdot w_{i, j}] \cdot \sigma'\qty(\net{j}{t}) \\
                                                        & = \sigma'\qty(\net{j}{t}) \cdot \sum_{i = 1}^{\dout} \qty[w_{i, j} \cdot \pdv{\tloss(t + 1)}{z_i(t + 1)}].
    \end{align*}
  \]

.. note::

  式子 :math:`\eqref{4}` 就是論文 3.1.1 節的最後一條公式。

模型參數 :math:`w_{i, j}` 對於 :math:`\tloss(t + 1)` 微分可得：

.. math::
  :nowrap:

  \[
    \begin{align*}
      \pdv{\cL}{w_{i, j}}    & = \pdv{\cL}{z_i(t + 1)} \cdot \pdv{z_i(t + 1)}{w_{i, j}} \\
                                        & = \sigma'\qty(\net{j}{t + 1}) \cdot \qty(y_i(t + 1) - \vyhi(t + 1)) \cdot \begin{pmatrix}
                                              \vx(t) \\
                                              \vy(t)
                                            \end{pmatrix}_j; \\
      \pdv{\loss_{i'}(t + 1)}{w_{i, j}} & = \pdv{\loss_{i'}(t + 1)}{\net{i}{t}} \cdot \pdv{\net{i}{t}}{w_{i, j}} \\
                                        & = \sigma'\qty(\net{j}{t + 1}) \cdot \qty(y_i(t + 1) - \vyhi(t + 1)) \cdot \begin{pmatrix}
                                              \vx(t) \\
                                              \vy(t)
                                            \end{pmatrix}_j; \\
      \pdv{\tloss(t + 1)}{w_{i, j}}     & = \pdv{\tloss(t + 1)}{z_i(t + 1)} \cdot \pdv{z_i(t + 1)}{w_{i, j}} + \sum_{k = 1}^\dout \pdv{\tloss(t + 1)}{y_k(t)} \cdot \pdv{y_k(t)}{w_{i, j}} \\
                                        & = \sigma'\qty(\net{j}{t + 1}) \cdot \qty(y_i(t + 1) - \vyhi(t + 1)) \cdot \begin{pmatrix}
                                              \vx(t) \\
                                              \vy(t)
                                            \end{pmatrix}_j.
    \end{align*}
  \]

.. note::

  式子 :math:`\eqref{5}` 是論文 3.1.1 節最後一段文字中提到的參數更新演算法。

梯度爆炸 / 消失
---------------

從 :math:`\eqref{2}\eqref{4}` 式我們可以進一步推得對不同時間點 net input 對誤差的微分。
探討此微分公式的目的是為了後續對微分分析，推導產生\ **梯度爆炸**\與\ **梯度消失**\的原因。
為了方便討論，我們定義新的符號：

.. math::
  :nowrap:

  \[
    \vth{k}{\tf}{\tp} = \pdv{\tloss(\tf)}{\net{k}{\tp}}.
  \]

意思是 the :math:`k`\-th coordinate of :math:`\opnet(\tp)` 對於 :math:`\tloss(\tf)` 計算所得之\ **微分**。

- 根據時間的限制我們有不等式 :math:`0 \leq \tp \leq \tf \leq T`
- 節點 :math:`k` 的數值範圍為 :math:`k \in \Set{1, \dots, \dout}`，見 RNN 計算定義

因此

.. math::
  :nowrap:

  \[
    \begin{align*}
    \vth{k_0}{t}{t}     & = \pdv{\tloss(t)}{\net{k_0}{t}}; \\
    \vth{k_1}{t}{t - 1} & = \pdv{\tloss(t)}{\net{k_1}{t - 1}} \\
                        & = \sigma'\qty(\net{k_1}{t - 1}) \cdot \qty(\sum_{k_0 = 1}^{\dout} w_{k_0, k_1} \cdot \vth{k_0}{t}{t}); \\
    \vth{k_2}{t}{t - 2} & = \pdv{\tloss(t)}{\net{k_2}{t - 2}} \\
                        & = \sum_{k_1 = 1}^{\dout} \qty[\pdv{\tloss(t)}{\net{k_1}{t - 1}} \cdot \pdv{\net{k_1}{t - 1}}{y_{k_2}(t - 2)} \cdot \pdv{y_{k_2}(t - 2)}{\net{k_2}{t - 2}}] \\
                        & = \sum_{k_1 = 1}^{\dout} \qty[\vth{k_1}{t}{t - 1} \cdot w_{k_1, k_2} \cdot \sigma'\qty(\net{k_2}{t - 2})] \\
                        & = \sum_{k_1 = 1}^{\dout} \qty[\sigma'\qty(\net{k_1}{t - 1}) \cdot \qty(\sum_{k_0 = 1}^{\dout} w_{k_0, k_1} \cdot \vth{k_0}{t}{t}) \cdot w_{k_1, k_2} \cdot \sigma'\qty(\net{k_2}{t - 2})] \\
                        & = \sum_{k_1 = 1}^{\dout} \sum_{k_0 = 1}^{\dout} \qty[w_{k_0, k_1} \cdot w_{k_1, k_2} \cdot \sigma'\qty(\net{k_1}{t - 1}) \cdot \sigma'\qty(\net{k_2}{t - 2}) \cdot \vth{k_0}{t}{t}]; \\
    \vth{k_3}{t}{t - 3} & = \sum_{k_2 = 1}^{\dout} \qty[\pdv{\tloss(t)}{\net{k_2}{t - 2}} \cdot \pdv{\net{k_2}{t - 2}}{y_{k_3}(t - 3)} \cdot \pdv{y_{k_3}(t - 3)}{\net{k_3}{t - 3}}] \\
                        & = \sum_{k_2 = 1}^{\dout} \qty[\vth{k_2}{t}{t - 2} \cdot w_{k_2, k_3} \cdot \sigma'\qty(\net{k_3}{t - 3})] \\
                        & = \sum_{k_2 = 1}^{\dout} \qty[\sum_{k_1 = 1}^{\dout} \sum_{k_0 = 1}^{\dout} \qty[w_{k_0, k_1} \cdot w_{k_1, k_2} \cdot \sigma'\qty(\net{k_1}{t - 1}) \cdot \sigma'\qty(\net{k_2}{t - 2}) \cdot \vth{k_0}{t}{t}] \cdot w_{k_2, k_3} \cdot \sigma'\qty(\net{k_3}{t - 3})] \\
                        & = \sum_{k_2 = 1}^{\dout} \sum_{k_1 = 1}^{\dout} \sum_{k_0 = 1}^{\dout} \qty[w_{k_0, k_1} \cdot w_{k_1, k_2} \cdot w_{k_2, k_3} \cdot \sigma'\qty(\net{k_1}{t - 1}) \cdot \sigma'\qty(\net{k_2}{t - 2}) \cdot \sigma'\qty(\net{k_3}{t - 3}) \cdot \vth{k_0}{t}{t}] \\
                        & = \sum_{k_2 = 1}^{\dout} \sum_{k_1 = 1}^{\dout} \sum_{k_0 = 1}^{\dout} \qty[\qty[\prod_{q = 1}^3 w_{k_{q - 1}, k_q} \cdot \sigma'\qty(\net{k_q}{t - q})] \cdot \vth{k_0}{t}{t}]
    \end{align*} \tag{7}\label{7}
  \]

由 :math:`\eqref{7}` 我們可以歸納得出 :math:`n \geq 1` 時的公式

..
  $$
  \vth{k_{n}}{t}{t - n} = \sum_{k_{n - 1} = 1}^{\dout} \cdots \sum_{k_{0} = 1}^{\dout} \br{\br{\prod_{q = 1}^{n} w_{k_{q - 1}, k_{q}} \cdot \sigma'\pa{\net{k_{q}}{t - q}}} \cdot \vth{k_{0}}{t}{t}} \tag{12}\label{12}
  $$

  由 $\eqref{12}$ 我們可以看出 $\vth{k_{n}}{t}{t - n}$ 都與 $\vth{k_{0}}{t}{t}$ 相關，因此我們將 $\vth{k_{n}}{t}{t - n}$ 想成由 $\vth{k_{0}}{t}{t}$ 構成的函數。

  現在讓我們固定 $k_{0}^{\star} \in \set{1, \dots, \dout}$，我們可以計算 $\vth{k_{0}^{\star}}{t}{t}$ 對於 $\vth{k_{n}}{t}{t - n}$ 的微分，分析**梯度**在進行**反向傳遞過程**中的**變化率**

  - 當 $n = 1$ 時，根據 $\eqref{11}$ 我們可以推得論文中的 (3.1) 式

    $$
    \pdv{\vth{k_{n}}{t}{t - n}}{\vth{k_{0}^{\star}}{t}{t}} = w_{k_{0}^{\star}, k_{1}} \cdot \sigma'\pa{\net{k_{1}}{t - 1}} \tag{13}\label{13}
    $$

  - 當 $n > 1$ 時，根據 $\eqref{12}$ 我們可以推得論文中的 (3.2) 式

    $$
    \pdv{\vth{k_{n}}{t}{t - n}}{\vth{k_{0}^{\star}}{t}{t}} = \sum_{k_{n - 1} = 1}^{\dout} \cdots \sum_{k_{1} = 1}^{\dout} \sum_{k_{0} \in \set{k_{0}^{\star}}} \br{\prod_{q = 1}^{n} w_{k_{q - 1}, k_{q}} \cdot \sigma'\pa{\net{k_{q}}{t - q}}} \tag{14}\label{14}
    $$

  **注意錯誤**：論文中的 (3.2) 式不小心把 $w_{l_{m - 1} l_{m}}$ 寫成 $w_{l_{m} l_{m - 1}}$。

  因此根據 $\eqref{14}$，共有 $(\dout)^{n - 1}$ 個連乘積項次進行加總。

  根據 $\eqref{13} \eqref{14}$，如果

  $$
  \abs{w_{k_{q - 1}, k_{q}} \cdot \sigma'\pa{\net{k_{q}}{t - q}}} > 1.0 \quad \forall q = 1, \dots, n \tag{15}\label{15}
  $$

  則**梯度變化率**成指數 $n$ 增長，直接導致**梯度爆炸**，參數會進行**劇烈的振盪**，無法進行順利更新。

  而如果

  $$
  \abs{w_{k_{q - 1}, k_{q}} \cdot \sigma'\pa{\net{k_{q}}{t - q}}} < 1.0 \quad \forall q = 1, \dots, n \tag{16}\label{16}
  $$

  則**梯度變化率**成指數 $n$ 縮小，直接導致**梯度消失**，誤差**收斂速度**會變得**非常緩慢**。

  從 $\eqref{17}$ 我們知道 $\sigma'$ 最大值為 $0.25$

  $$
  \begin{align*}
  \sigma(x) & = \frac{1}{1 + e^{-x}} \\
  \sigma'(x) & = \frac{e^{-x}}{(1 + e^{-x})^2} = \frac{1}{1 + e^{-x}} \cdot \frac{e^{-x}}{1 + e^{-x}} \\
  & = \frac{1}{1 + e^{-x}} \cdot \frac{1 + e^{-x} - 1}{1 + e^{-x}} = \sigma(x) \cdot \big(1 - \sigma(x)\big) \\
  \sigma(\R) & = (0, 1) \\
  \max_{x \in \R} \sigma'(x) & = \sigma(0) \times \big(1 - \sigma(0)\big) = 0.5 \times 0.5 = 0.25
  \end{align*} \tag{17}\label{17}
  $$

  因此當 $\abs{w_{k_{q - 1}, k_{q}}} < 4.0$ 時我們可以發現

  $$
  \abs{w_{k_{q - 1}, k_{q}} \cdot \sigma'\pa{\net{k_{q}}{t - q}}} < 4.0 * 0.25 = 1.0 \tag{18}\label{18}
  $$

  所以 $\eqref{18}$ 與 $\eqref{16}$ 的結論相輔相成：當 $w_{k_{q - 1}, k_{q}}$ 的絕對值小於 $4.0$ 會造成**梯度消失**。

  而 $\abs{w_{k_{q - 1}, k_{q}}} \to \infty$ 我們可以使用 $\eqref{17}$ 得到

  $$
  \begin{align*}
  & \abs{\net{k_{q - 1}}{t - q + 1}} \to \infty \\
  \implies & \begin{dcases}
  \sigma\pa{\net{k_{q - 1}}{t - q + 1}} \to 1 & \text{if } \net{k_{q - 1}}{t - q + 1} \to \infty \\
  \sigma\pa{\net{k_{q - 1}}{t - q + 1}} \to 0 & \text{if } \net{k_{q - 1}}{t - q + 1} \to -\infty
  \end{dcases} \\
  \implies & \abs{\sigma'\pa{\net{k_{q - 1}}{t - q + 1}}} \to 0 \\
  \implies & \abs{\prod_{q = 1}^{n} w_{k_{q - 1}, k_{q}} \cdot \sigma'\pa{\net{k_{q}}{t - q}}} \\
  & = \abs{w_{k_0, k_1} \cdot \prod_{q = 2}^{n} \qty[\sigma'\pa{\net{k_{q - 1}}{t - q + 1}} \cdot w_{k_{q - 1}, k_{q}}] \cdot \sigma'\pa{\net{k_{n}}{t - n}}} \\
  & \to 0
  \end{align*} \tag{19}\label{19}
  $$

  最後一個推論的原理是**指數函數的收斂速度比線性函數快**。

  **注意錯誤**：論文中的推論

  $$
  \abs{w_{k_{q - 1}, k_{q}} \cdot \dfnet{k_{q}}{t - q}} \to 0
  $$

  是**錯誤**的，理由是 $w_{k_{q - 1}, k_{q}}$ 無法對 $\net{k_{q}}{t - q}$ 造成影響，作者不小心把**時間順序寫反**了，但是**最後的邏輯仍然正確**，理由如 $\eqref{19}$ 所示。

  **注意錯誤**：論文中進行了以下**函數最大值**的推論

  $$
  \begin{align*}
  & \dfnet{l_{m}}{t - m}\big) \cdot w_{l_{m} l_{m - 1}} \\
  & = \sigma\big(\net{l_{m}}{t - m}\big) \cdot \Big(1 - \sigma\big(\net{l_{m}}{t - m}\big)\Big) \cdot w_{l_{m} l_{m - l}}
  \end{align*}
  $$

  最大值發生於微分值為 $0$ 的點，即我們想求出滿足以下式子的 $w_{l_{m} l_{m - 1}}$

  $$
  \pdv{\Big[\sigma\big(\net{l_{m}}{t - m}\big) \cdot \Big(1 - \sigma\big(\net{l_{m}}{t - m}\big)\Big) \cdot w_{l_{m} l_{m - l}}\Big]}{w_{l_{m} l_{m - 1}}} = 0
  $$

  拆解微分式可得

  $$
  \begin{align*}
  & \pdv{\Big[\sigma\big(\net{l_{m}}{t - m}\big) \cdot \Big(1 - \sigma\big(\net{l_{m}}{t - m}\big)\Big) \cdot w_{l_{m} l_{m - l}}\Big]}{w_{l_{m} l_{m - 1}}} \\
  & = \pdv{\sigma\big(\net{l_{m}}{t - m}\big)}{\net{l_{m}}{t - m}} \cdot \pdv{\net{l_{m}}{t - m}}{w_{l_{m} l_{m - 1}}} \cdot \Big(1 - \sigma\big(\net{l_{m}}{t - m}\big)\Big) \cdot w_{l_{m} l_{m - l}} \\
  & \quad + \sigma\big(\net{l_{m}}{t - m}\big) \cdot \pdv{\Big(1 - \sigma\big(\net{l_{m}}{t - m}\big)\Big)}{\net{l_{m}}{t - m}} \cdot \pdv{\net{l_{m}}{t - m}}{w_{l_{m} l_{m - 1}}} \cdot w_{l_{m} l_{m - l}} \\
  & \quad + \sigma\big(\net{l_{m}}{t - m}\big) \cdot \Big(1 - \sigma\big(\net{l_{m}}{t - m}\big)\Big) \cdot \pdv{w_{l_{m} l_{m - 1}}}{w_{l_{m} l_{m - 1}}} \\
  & = \sigma\big(\net{l_{m}}{t - m}\big) \cdot \Big(1 - \sigma\big(\net{l_{m}}{t - m}\big)\Big)^2 \cdot y_{l_{m - 1}}(t - m - 1) \cdot w_{l_{m} l_{m - 1}} \\
  & \quad - \Big(\sigma\big(\net{l_{m}}{t - m}\big)\Big)^2 \cdot \Big(1 - \sigma\big(\net{l_{m}}{t - m}\big)\Big) \cdot y_{l_{m - 1}}(t - m - 1) \cdot w_{l_{m} l_{m - 1}} \\
  & \quad + \sigma\big(\net{l_{m}}{t - m}\big) \cdot \Big(1 - \sigma\big(\net{l_{m}}{t - m}\big)\Big) \\
  & = \Big[2 \Big(\sigma\big(\net{l_{m}}{t - m}\big)\Big)^3 - 3 \Big(\sigma\big(\net{l_{m}}{t - m}\big)\Big)^2 + \sigma\big(\net{l_{m}}{t - m}\big)\Big] \cdot \\
  & \quad \quad y_{l_{m - 1}}(t - m - 1) \cdot w_{l_{m} l_{m - 1}} \\
  & \quad + \sigma\big(\net{l_{m}}{t - m}\big) \cdot \Big(1 - \sigma\big(\net{l_{m}}{t - m}\big)\Big) \\
  & = \sigma\big(\net{l_{m}}{t - m}\big) \cdot \Big(2 \sigma\big(\net{l_{m}}{t - m}\big) - 1\Big) \cdot \Big(\sigma\big(\net{l_{m}}{t - m}\big) - 1\Big) \cdot \\
  & \quad \quad y_{l_{m - 1}}(t - m - 1) \cdot w_{l_{m} l_{m - 1}} \\
  & \quad + \sigma\big(\net{l_{m}}{t - m}\big) \cdot \Big(1 - \sigma\big(\net{l_{m}}{t - m}\big)\Big) \\
  & = 0
  \end{align*}
  $$

  移項後可以得到

  $$
  \begin{align*}
  & \sigma\big(\net{l_{m}}{t - m}\big) \cdot \Big(2 \sigma\big(\net{l_{m}}{t - m}\big) - 1\Big) \cdot \Big(1 - \sigma\big(\net{l_{m}}{t - m}\big)\Big) \cdot \\
  & \quad \quad y_{l_{m - 1}}(t - m - 1) \cdot w_{l_{m} l_{m - 1}} = \sigma\big(\net{l_{m}}{t - m}\big) \cdot \Big(1 - \sigma\big(\net{l_{m}}{t - m}\big)\Big) \\
  \implies & \Big(2 \sigma\big(\net{l_{m}}{t - m}\big) - 1\Big) \cdot y_{l_{m - 1}}(t - m - 1) \cdot w_{l_{m} l_{m - 1}} = 1 \\
  \implies & w_{l_{m} l_{m - 1}} = \frac{1}{y_{l_{m - 1}}(t - m - 1)} \cdot \frac{1}{2 \sigma\big(\net{l_{m}}{t - m}\big) - 1} \\
  \implies & w_{l_{m} l_{m - 1}} = \frac{1}{y_{l_{m - 1}}(t - m - 1)} \cdot \coth\bigg(\frac{\net{l_{m}}{t - m}}{2}\bigg)
  \end{align*}
  $$

  註：推論中使用了以下公式

  $$
  \begin{align*}
  \tanh(x) & = 2 \sigma(2x) - 1 \\
  \tanh(\frac{x}{2}) & = 2 \sigma(x) - 1 \\
  \coth(\frac{x}{2}) & = \frac{1}{\tanh(\frac{x}{2})} = \frac{1}{2 \sigma(x) - 1}
  \end{align*}
  $$

  但公式的前提不對，理由是 $w_{l_{m} l_{m - 1}}$ 根本不存在，應該改為 $w_{l_{m - 1} l_{m}}$（同 $\eqref{14}$）。

  接著我們可以計算 $t$ 時間點 $\dout$ 個**不同**節點 $\net{k_0^{\star}}{t}$ 對於**同一個** $t - n$ 時間點的 $\net{k_{n}}{t - n}$ 節點所貢獻的**梯度變化總和**：

  $$
  \sum_{k_{0}^{\star} = 1}^{\dout} \pdv{\vth{k_{n}}{t}{t - n}}{\vth{k_{0}^{\star}}{t}{t}} \tag{20}\label{20}
  $$

  由於**每個項次**都能遭遇**梯度消失**，因此**總和**也會遭遇**梯度消失**。

  ## 問題觀察

  ### 情境 1：模型輸出與內部節點 1-1 對應

  假設模型沒有任何輸入，啟發函數 $f_j$ 為未知且 $t - 1$ 時間點的輸出節點 $y_j(t - 1)$ 只與 $\net{j}{t}$ 相連，即

  $$
  \net{j}{t} = w_{j, j} \cdot y_j(t - 1) \tag{21}\label{21}
  $$

  則根據式子 $\eqref{11}$ 我們可以推得

  $$
  \vth{j}{t}{t - 1} = w_{j, j} \cdot \dfnet{j}{t - 1} \cdot \vth{j}{t}{t} \tag{22}\label{22}
  $$

  為了不讓梯度 $\vth{j}{t}{t}$ 在傳遞的過程消失，作者認為需要強制達成**梯度常數（Constant Error Flow）**

  $$
  w_{j, j} \cdot \dfnet{j}{t - 1} = 1.0 \tag{23}\label{23}
  $$

  透過 $\eqref{23}$ 的想法讓 $\eqref{12}$ 中梯度變化率的**連乘積項**為 $1.0$，因此

  - 不會像 $\eqref{15}$ 導致梯度**爆炸**
  - 不會像 $\eqref{16}$ 導致梯度**消失**

  如果 $\eqref{23}$ 能夠達成，則積分 $\eqref{23}$ 可以得到

  $$
  \begin{align*}
  & \int w_{j, j} \cdot \dfnet{j}{t - 1} \; d \big[\net{j}{t - 1}\big] = \int 1.0 \; d \big[\net{j}{t - 1}\big] \\
  \iff & w_{j, j} \cdot \fnet{j}{t - 1} = \net{j}{t - 1} \\
  \iff & y_j(t - 1) = \fnet{j}{t - 1} = \frac{\net{j}{t - 1}}{w_{j, j}}
  \end{align*} \tag{24}\label{24}
  $$

  觀察 $\eqref{24}$ 我們可以發現

  - 輸入 $\net{j}{t - 1}$ 與輸出 $\fnet{j}{t - 1}$ 之間的關係是乘上一個常數項 $w_{j, j}$
  - 代表函數 $f_j$ 其實是一個**線性函數**

  若採用 $\eqref{24}$ 的架構設計，我們可以發現**每個時間點**的**輸出**必須**完全相同**

  $$
  \begin{align*}
  y_j(t) & = \fnet{j}{t} = f_j\big(w_{j, j} \cdot y_j(t - 1)\big) \\
  & = f_j\big(w_{j, j} \cdot \frac{\net{j}{t - 1}}{w_{j, j}}\big) = \fnet{j}{t - 1} = y_j(t - 1) \tag{25}\label{25}
  \end{align*}
  $$

  這個現象稱為 **Constant Error Carousel**（簡稱 **CEC**），而作者設計的 LSTM 架構會完全基於 CEC 進行設計，但我覺得概念比較像 ResNet 的 residual connection。

  ### 情境 2：增加外部輸入

  將 $\eqref{21}$ 的假設改成每個模型內部節點可以額外接收**外部輸入**

  $$
  \net{j}{t} = w_{j, j} \cdot y_j(t - 1) + \sum_{i = 1}^{\din} w_{j, i} \cdot x_{i}(t - 1) \tag{26}\label{26}
  $$

  由於 $y_j(t - 1)$ 的設計功能是保留過去計算所擁有的資訊，在 $\eqref{26}$ 的假設中唯一能夠**更新**資訊的方法只有透過 $x_{i}(t - 1)$ 配合 $w_{j, i}$ 將新資訊合併進入 $\net{j}{t}$。

  但作者認為，在計算的過程中，部份時間點的**輸入**資訊 $x_{i}(\cdot)$ 可能是**雜訊**，因此可以（甚至必須）被**忽略**。
  但這代表與外部輸入相接的參數 $w_{j, i}$ 需要**同時**達成**兩種**任務：

  - **加入新資訊**：代表 $\abs{w_{j, i}} \neq 0$
  - **忽略新資訊**：代表 $\abs{w_{j, i}} \approx 0$

  因此**無法只靠一個** $w_{j, i}$ 決定**輸入**的影響，必須有**額外**能夠**理解當前內容 (context-sensitive)** 的功能模組幫忙決定是否**寫入** $x_{i}(\cdot)$。

  ### 情境 3：輸出回饋到多個節點

  將 $\eqref{21} \eqref{26}$ 的假設改回正常的模型架構

  $$
  \net{j}{t} = \sum_{i = 1}^{\dout} w_{j, i} \cdot y_i(t - 1) + \sum_{i = 1}^{\din} w_{j, \dout + i} \cdot x_{i}(t - 1) \tag{27}\label{27}
  $$

  由於 $y_j(t - 1)$ 的設計功能是保留過去計算所擁有的資訊，在 $\eqref{27}$ 的假設中唯一能夠讓**過去**資訊**影響未來**計算結果的方法只有透過 $y_i(t - 1)$ 配合 $w_{j, \din + i}$ 將新資訊合併進入 $\net{j}{t}$。

  但作者認為，在計算的過程中，部份時間點的**輸出**資訊 $y_i(*)$ 可能對預測沒有幫助，因此可以(甚至必須)被**忽略**。
  但這代表與輸出相接的參數 $w_{j, \din + i}$ 需要**同時**達成**兩種**任務：

  - **保留過去資訊**：代表 $\abs{w_{j, \din + i}} \neq 0$
  - **忽略過去資訊**：代表 $\abs{w_{j, \din + i}} \approx 0$

  因此**無法只靠一個** $w_{j, \din + i}$ 決定**輸出**的影響，必須有**額外**能夠**理解當前內容 (context-sensitive)** 的功能模組幫忙決定是否**讀取** $y_i(*)$。

  ## LSTM 架構

  <a name="paper-fig-1"></a>

  圖 1：記憶細胞內部架構。
  符號對應請見下個小節。
  圖片來源：[論文][論文]。

  ![圖 1](https://i.imgur.com/uhS4AgH.png)

  <a name="paper-fig-2"></a>

  圖 2：LSTM 全連接架構範例。
  線條真的多到讓人看不懂，看我整理過的公式比較好理解。
  圖片來源：[論文][論文]。

  ![圖 2](https://i.imgur.com/UQ5LAu8.png)

  為了解決**梯度爆炸 / 消失**問題，作者決定以 Constant Error Carousel 為出發點（見 $\eqref{25}$），提出 **3** 個主要的機制，並將這些機制的合體稱為**記憶細胞區域（memory cell blocks）**（見[圖 1](#paper-fig-1)）：

  - **乘法輸入閘門（Multiplicative Input Gate）**：用於決定是否**更新**記憶細胞的**內部狀態**
  - **乘法輸出閘門（Multiplicative Output Gate）**：用於決定是否**輸出**記憶細胞的**計算結果**
  - **自連接線性單元（Central Linear Unit with Fixed Self-connection）**：概念來自於 CEC（見 $\eqref{25}$），藉此保障**梯度不會消失**

.. footbibliography::

.. ====================================================================================================================
.. external links
.. ====================================================================================================================


