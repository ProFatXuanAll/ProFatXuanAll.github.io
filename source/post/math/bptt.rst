==================================
Back-Propagation Through Time 推導
==================================

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
    % Vectors' notation.
    \newcommand{\vx}{\mathbf{x}}
    \newcommand{\vy}{\mathbf{y}}
    \newcommand{\vyh}{\hat{\mathbf{y}}}
    \newcommand{\vz}{\mathbf{z}}

    % Matrixs' notation.
    \newcommand{\vW}{\mathbf{W}}

    % Symbols in mathcal.
    \newcommand{\cL}{\mathcal{L}}
    \newcommand{\cT}{\mathcal{T}}

    % Vectors with subscript.
    \newcommand{\vxj}{{\vx_j}}
    \newcommand{\vyi}{{\vy_i}}
    \newcommand{\vyj}{{\vy_j}}
    \newcommand{\vyk}{{\vy_k}}
    \newcommand{\vyl}{{\vy_\ell}}
    \newcommand{\vyhi}{{\vyh_i}}
    \newcommand{\vyhk}{{\vyh_k}}
    \newcommand{\vzi}{{\vz_i}}
    \newcommand{\vzj}{{\vz_j}}
    \newcommand{\vzk}{{\vz_k}}
    \newcommand{\vzl}{{\vz_\ell}}

    % Matrixs with subscripts.
    \newcommand{\vWiC}{{\vW_{i, :}}}
    \newcommand{\vWij}{{\vW_{i, j}}}
    \newcommand{\vWik}{{\vW_{i, k}}}
    \newcommand{\vWil}{{\vW_{i, \ell}}}
    \newcommand{\vWRj}{{\vW_{:, j}}}
    \newcommand{\vWkj}{{\vW_{k, j}}}
    \newcommand{\vWlj}{{\vW_{\ell, j}}}

    % Matrix with subscript and superscripts
    \newcommand{\vWkjn}{\vW_{k, j}^{\operatorname{new}}}
    \newcommand{\vWkjo}{\vW_{k, j}^{\operatorname{old}}}

    % Dimensions.
    \newcommand{\din}{{d_{\operatorname{in}}}}
    \newcommand{\dout}{{d_{\operatorname{out}}}}
  \]

基本符號定義
============

- 所有向量（vector）都以粗體表示，例如：:math:`\vx`
- 所有向量討論僅限於實數空間，例如 :math:`\R^3`
- The :math:`j`\-th coordinate of a vector 以下標表示，例如：:math:`\vxj`
- 所有向量皆為行向量（column vector），除非特別表示
- 所有矩陣（matrix）都以大寫粗體表示，例如：:math:`\vW`
- The :math:`i`\-th row of a matrix 以下標 :math:`i, :` 表示，例如：:math:`\vWiC`
- The :math:`i`\-th row and the :math:`j`\-th column of a matrix 以下標 :math:`i, j` 表示，例如：:math:`\vWij`
- The :math:`j`\-th column of a matrix 以下標 :math:`:, j` 表示，例如：:math:`\vWRj`
- 矩陣乘法以 :math:`\cdot` 表示，例如：:math:`\vW \cdot \vx`
- :math:`\delta_{a, b}` is :math:`1` if :math:`a = b`, else is :math:`0`
- :math:`\mathbb{1}(\operatorname{cond})` is :math:`1` if :math:`\operatorname{cond}` is true, else is :math:`0`

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

  - 定義 :math:`f_i: \R \to \R` 為 :math:`f` 的第 :math:`i` 個 real valued function，:math:`i \in \Set{1, \dots, \dout}`
  - :math:`f_i` 必須要可以\ **微分**
  - 每個 :math:`f_i` 所使用的 activation function 可以\ **不同**，但都只用 :math:`\vzi(t + 1)` 作為輸入

透過以上符號我們可以拆解矩陣乘法：

.. math::
  :nowrap:

  \[
    \begin{align*}
      & \algoProc{\operatorname{RNN}}(x, \cT) \\
      & \indent{1} \vy(0) \algoEq \zv \\
      & \indent{1} \algoFor{t \in \Set{0, \dots, \cT - 1}} \\
      & \indent{2} \algoFor{i \in \Set{1, \dots, \dout}} \\
      & \indent{3} \vzi(t + 1) \algoEq \sum_{j = 1}^\din \vW_{i, j} \cdot \vx_j(t) + \sum_{j = \din + 1}^{\din + \dout} \vW_{i, j} \cdot \vyj(t) \\
      & \indent{3} \vyi(t + 1) \algoEq f_i(\vzi(t + 1)) \\
      & \indent{2} \algoEndFor \\
      & \indent{1} \algoEndFor \\
      & \indent{1} \algoReturn \vy(1), \dots, \vy(\cT) \\
      & \algoEndProc
    \end{align*}
  \]

目標函數
=========

定義 :math:`\cL : \R^\dout \times \R^\dout \to \R` 代表\ **最小平方差**。
假設每個時間點的誤差計算法為最小平方差，則 :math:`t + 1` 時間點的誤差可以表達為

.. math::
  :nowrap:

  \[
    \cL(\vy(t + 1), \vyh(t + 1)) = \frac{1}{2} \sum_{i = 1}^\dout \qty[\vyi(t + 1) - \vyhi(t + 1)]^2.
    \tag{1}\label{1}
  \]

而目標函數（objective function）的定義如下

.. math::
  :nowrap:

  \[
    \sum_{t = 0}^{\cT - 1} \cL(\vy(t + 1), \vyh(t + 1)).
    \tag{2}\label{2}
  \]

接下來的討論將會專注在單一時間點的誤差上。

對目標函數微分
==============

為了將 forward pass 中使用的符號與微分計算對象區隔，我們需要定義以下符號：

- 令 :math:`t \in \Set{0, \dots, \cT - 1}`
- 令 :math:`i \in \Set{1, \dots, \dout}`
- 當 :math:`j` 為 :math:`\vx` 的下標時，令 :math:`j \in \Set{1, \dots, \din}`
- 當 :math:`j` 為 :math:`\vy` 或 :math:`\vz` 的下標時，令 :math:`j \in \Set{1, \dots, \dout}`
- 當 :math:`k` 為 :math:`\vW` 的 row index 時，令 :math:`k \in \Set{1, \dots, \dout}`
- 當 :math:`j` 為 :math:`\vW` 的 column index 時，令 :math:`j \in \Set{1, \dots, \din + \dout}`

根據目標函數 :math:`\eqref{1}` 的定義，我們可以計算 :math:`\vyi(t + 1)` 對 :math:`\cL(\vy(t + 1), \vyh(t + 1))` 的微分：

.. math::
  :nowrap:

  \[
    \dv{L(\vy(t + 1), \vyh(t + 1))}{\vyi(t + 1)} = \vyi(t + 1) - \vyhi(t + 1).
    \tag{3}\label{3}
  \]

.. dropdown:: 推導 :math:`\eqref{3}`

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{L(\vy(t + 1), \vyh(t + 1))}{\vyi(t + 1)}
        & = \dv{\frac{1}{2} \sum_{k = 1}^\dout \qty[\vyk(t + 1) - \vyhk(t + 1)]^2}{\vyi(t + 1)} \\
        & = \frac{1}{2} \sum_{k = 1}^\dout \dv{\qty[\vyk(t + 1) - \vyhk(t + 1)]^2}{\vyi(t + 1)} \\
        & = \frac{1}{2} \cdot \dv{\qty[\vyi(t + 1) - \vyhi(t + 1)]^2}{\vyi(t + 1)} \\
        & = \vyi(t + 1) - \vyhi(t + 1).
      \end{align*}
    \]

由於 :math:`\vyi(t + 1)` 是由 :math:`\vzi(t + 1)` 產生，我們求得 :math:`\vzi(t + 1)` 對 :math:`\vyi(t + 1)` 的微分：

.. math::
  :nowrap:

  \[
    \dv{\vyi(t + 1)}{\vzi(t + 1)} = f_i'\qty(\vzi(t + 1)).
    \tag{4}\label{4}
  \]

透過 :math:`\eqref{4}` 我們可以推得 :math:`\vzi(t + 1)` 對 :math:`\cL(\vy(t + 1), \vyh(t + 1))` 的微分：

.. math::
  :nowrap:

  \[
    \dv{\cL(\vy(t + 1), \vyh(t + 1))}{\vzi(t + 1)} = \qty[\vyi(t + 1) - \vyhi(t + 1)] \cdot f_i'\qty(\vzi(t + 1)).
    \tag{5}\label{5}
  \]


.. dropdown:: 推導 :math:`\eqref{5}`

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\cL(\vy(t + 1), \vyh(t + 1))}{\vzi(t + 1)}
        & = \dv{\cL(\vy(t + 1), \vyh(t + 1))}{\vyi(t + 1)} \cdot \dv{\vyi(t + 1)}{\vzi(t + 1)} \\
        & = \qty[\vyi(t + 1) - \vyhi(t + 1)] \cdot f_i'\qty(\vzi(t + 1)).
      \end{align*}
    \]

.. note::

  式子 :math:`\eqref{5}` 就是 LSTM 論文 :footcite:`hochreiter-etal-1997-long` 3.1.1 節的第一條公式。

接著討論與遞迴有關的微分。
由於 :math:`\vzi(t + 1)` 是由 :math:`\vyj(t)` 產生（注意時間差），因此我們可以求 :math:`\vyj(t)` 對 :math:`\vzi(t + 1)` 的微分：

.. math::
  :nowrap:

  \[
    \dv{\vzi(t + 1)}{\vyj(t)} = \vWij.
    \tag{6}\label{6}
  \]

.. dropdown:: 推導 :math:`\eqref{6}`

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\vzi(t + 1)}{\vyj(t)}
        & = \dv{\sum_{k = 1}^\dout \vWik \cdot \mqty[\vx(t) \\ \vy(t)]_k}{\vyj(t)} \\
        & = \sum_{k = 1}^\dout \dv{\vWik \cdot \mqty[\vx(t) \\ \vy(t)]_k}{\vyj(t)} \\
        & = \vWij.
      \end{align*}
    \]

根據 :math:`\eqref{5}\eqref{6}` 我們可以推得 :math:`\vyj(t)` 對 :math:`\cL(\vy(t + 1), \vyh(t + 1))` 的微分（注意時間差）：

.. math::
  :nowrap:

  \[
    \dv{\cL(\vy(t + 1), \vyh(t + 1))}{\vyj(t)} = \sum_{i = 1}^\dout \qty[\qty[\vyi(t + 1) - \vyhi(t + 1)] \cdot f_i'\qty(\vzi(t + 1)) \cdot \vWij].
    \tag{7}\label{7}
  \]

.. dropdown:: 推導 :math:`\eqref{7}`

  .. math::
    :nowrap:

    \[
      \begin{align*}
        & \dv{\cL(\vy(t + 1), \vyh(t + 1))}{\vyj(t)} \\
        & = \sum_{i = 1}^\dout \qty[\dv{\cL(\vy(t + 1), \vyh(t + 1))}{\vzi(t + 1)} \cdot \dv{\vzi(t + 1)}{\vyj(t)}] \\
        & = \sum_{i = 1}^\dout \qty[\qty[\vyi(t + 1) - \vyhi(t + 1)] \cdot f_i'\qty(\vzi(t + 1)) \cdot \vWij].
      \end{align*}
    \]

我們再利用 :math:`\eqref{4}\eqref{7}` 計算 :math:`\vzj(t)` 對 :math:`\cL(\vy(t + 1), \vyh(t + 1))` 的微分：

.. math::
  :nowrap:

  \[
    \dv{\cL(\vy(t + 1), \vyh(t + 1))}{\vzj(t)} = \qty(\sum_{i = 1}^\dout \qty[\qty[\vyi(t + 1) - \vyhi(t + 1)] \cdot f_i'\qty(\vzi(t + 1)) \cdot \vWij]) \cdot f_j'\qty(\vzj(t)).
    \tag{8}\label{8}
  \]

.. dropdown:: 推導 :math:`\eqref{8}`

  .. math::
    :nowrap:

    \[
      \begin{align*}
        & \dv{\cL(\vy(t + 1), \vyh(t + 1))}{\vzj(t)} \\
        & = \dv{\cL(\vy(t + 1), \vyh(t + 1))}{\vyj(t)} \cdot \dv{\vyj(t)}{\vzj(t)} \\
        & = \qty(\sum_{i = 1}^\dout \qty[\qty[\vyi(t + 1) - \vyhi(t + 1)] \cdot f_i'\qty(\vzi(t + 1)) \cdot \vWij]) \cdot f_j'\qty(\vzj(t)).
      \end{align*}
    \]

.. note::

  式子 :math:`\eqref{8}` 就是 LSTM 論文 :footcite:`hochreiter-etal-1997-long` 3.1.1 節的最後一條公式。

當 :math:`t = 0` 時，模型參數 :math:`\vWkj` 對於 :math:`\vzi(t + 1)` 微分可得：

.. math::
  :nowrap:

  \[
    \dv{\vzi(1)}{\vWkj} = \delta_{i, k} \cdot \mqty[\vx(0) \\ \vy(0)]_j.
    \tag{9}\label{9}
  \]

.. dropdown:: 推導 :math:`\eqref{9}`

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\vzi(1)}{\vWkj}
        & = \dv{\sum_{\ell = 1}^{\din + \dout} \vWil \cdot \mqty[\vx(0) \\ \vy(0)]_\ell}{\vWkj} \\
        & = \sum_{\ell = 1}^{\din + \dout} \dv{\vWil \cdot \mqty[\vx(0) \\ \vy(0)]_\ell}{\vWkj} \\
        & = \sum_{\ell = 1}^{\din + \dout} \delta_{i, k} \cdot \delta_{\ell, j} \cdot \mqty[\vx(0) \\ \vy(0)]_\ell \\
        & = \delta_{i, k} \cdot \mqty[\vx(0) \\ \vy(0)]_j.
      \end{align*}
    \]

當 :math:`t > 0` 時，模型參數 :math:`\vWkj` 對於 :math:`\vzi(t + 1)` 微分可得：

.. math::
  :nowrap:

  \[
    \dv{\vzi(t + 1)}{\vWkj} = \delta_{i, k} \cdot \mqty[\vx(t) \\ \vy(t)]_j + \sum_{\ell = 1}^{\din + \dout} \vWil \cdot \mathbb{1}\qty(\mqty[\vx(t) \\ \vy(t)]_\ell = \vy_\ell(t)) \cdot f_\ell'(\vzl(t)) \cdot \dv{\vzl(t)}{\vWkj}.
    \tag{10}\label{10}
  \]

.. dropdown:: 推導 :math:`\eqref{10}`

  .. math::
    :nowrap:

    \[
      \begin{align*}
        & \dv{\vzi(t + 1)}{\vWkj} \\
        & = \dv{\sum_{\ell = 1}^{\din + \dout} \vWil \cdot \mqty[\vx(t) \\ \vy(t)]_\ell}{\vWkj} \\
        & = \sum_{\ell = 1}^{\din + \dout} \dv{\vWil \cdot \mqty[\vx(t) \\ \vy(t)]_\ell}{\vWkj} \\
        & = \sum_{\ell = 1}^{\din + \dout} \qty(\dv{\vWil}{\vWkj} \cdot \mqty[\vx(t) \\ \vy(t)]_\ell + \vWil \cdot \dv{\mqty[\vx(t) \\ \vy(t)]_\ell}{\vWkj}) \\
        & = \sum_{\ell = 1}^{\din + \dout} \qty(\delta_{i, k} \cdot \delta_{\ell, j} \cdot \mqty[\vx(t) \\ \vy(t)]_\ell + \vWil \cdot \mathbb{1}\qty(\mqty[\vx(t) \\ \vy(t)]_\ell = \vy_\ell(t)) \cdot \dv{\vyl(t)}{\vzl(t)} \cdot \dv{\vzl(t)}{\vWkj}) \\
        & = \delta_{i, k} \cdot \mqty[\vx(t) \\ \vy(t)]_j + \sum_{\ell = 1}^{\din + \dout} \vWil \cdot \mathbb{1}\qty(\mqty[\vx(t) \\ \vy(t)]_\ell = \vy_\ell(t)) \cdot f_\ell'(\vzl(t)) \cdot \dv{\vzl(t)}{\vWkj}.
      \end{align*}
    \]

最後我們可以推得模型參數 :math:`\vWkj` 對於 :math:`\cL(\vy(t + 1), \vyh(t + 1))` 的微分：

.. math::
  :nowrap:

  \[
    \dv{\cL(\vy(t + 1), \vyh(t + 1))}{\vWkj} = \qty[\vyk(t + 1) - \vyhk(t + 1)] \cdot f_k'\qty(\vzk(t + 1)) \cdot \mqty[\vx(t) \\ \vy(t)]_j + \sum_{i = 1}^\dout \qty[\vyi(t + 1) - \vyhi(t + 1)] \cdot f_i'\qty(\vzi(t + 1)) \cdot \qty[\sum_{\ell = 1}^{\din + \dout} \vWil \cdot \mathbb{1}\qty(\mqty[\vx(t) \\ \vy(t)]_\ell = \vy_\ell(t)) \cdot f_\ell'(\vzl(t)) \cdot \dv{\vzl(t)}{\vWkj}].
    \tag{11}\label{11}
  \]

.. dropdown:: 推導 :math:`\eqref{11}`

  .. math::
    :nowrap:

    \[
      \begin{align*}
        & \dv{\cL(\vy(t + 1), \vyh(t + 1))}{\vWkj} \\
        & = \sum_{i = 1}^\dout \dv{\cL(\vy(t + 1), \vyh(t + 1))}{\vzi(t + 1)} \cdot \dv{\vzi(t + 1)}{\vWkj} \\
        & = \sum_{i = 1}^\dout \qty[\vyi(t + 1) - \vyhi(t + 1)] \cdot f_i'\qty(\vzi(t + 1)) \cdot \qty(\delta_{i, k} \cdot \mqty[\vx(t) \\ \vy(t)]_j + \sum_{\ell = 1}^{\din + \dout} \vWil \cdot \mathbb{1}\qty(\mqty[\vx(t) \\ \vy(t)]_\ell = \vy_\ell(t)) \cdot f_\ell'(\vzl(t)) \cdot \dv{\vzl(t)}{\vWkj}) \\
        & = \qty[\vyk(t + 1) - \vyhk(t + 1)] \cdot f_k'\qty(\vzk(t + 1)) \cdot \mqty[\vx(t) \\ \vy(t)]_j + \sum_{i = 1}^\dout \qty[\vyi(t + 1) - \vyhi(t + 1)] \cdot f_i'\qty(\vzi(t + 1)) \cdot \qty[\sum_{\ell = 1}^{\din + \dout} \vWil \cdot \mathbb{1}\qty(\mqty[\vx(t) \\ \vy(t)]_\ell = \vy_\ell(t)) \cdot f_\ell'(\vzl(t)) \cdot \dv{\vzl(t)}{\vWkj}].
      \end{align*}
    \]

.. note::

  式子 :math:`\eqref{11}` 的前半段是 LSTM 論文 :footcite:`hochreiter-etal-1997-long` 3.1.1 節最後一段文字中提到的參數更新演算法。

參數更新
========

根據式子 :math:`\eqref{11}` 我們可以求得 :math:`\vWkj` 對於目標函數 :math:`\eqref{2}` 的微分：

.. math::
  :nowrap:

  \[
    \dv{\sum_{t = 0}^{\cT - 1} \cL(\vy(t + 1), \vyh(t + 1))}{\vWkj} = \sum_{t = 0}^{\cT - 1} \dv{\cL(\vy(t + 1), \vyh(t + 1))}{\vWkj}.
    \tag{12}\label{12}
  \]

若 :math:`\alpha` 為 :term:`learning rate`，則使用 BPTT 更新 RNN 參數 :math:`\vW` 的方法如下：

.. math::
  :nowrap:

  \[
    \vWkjn = \vWkjo - \alpha \cdot \sum_{t = 0}^{\cT - 1} \dv{\cL(\vy(t + 1), \vyh(t + 1))}{\vWkjo}. \tag{13}\label{13}
  \]

.. footbibliography::
