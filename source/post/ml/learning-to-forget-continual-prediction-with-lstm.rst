==================================================
Learning to Forget: Continual Prediction with LSTM
==================================================

.. ====================================================================================================================
.. Set index for authors.
.. ====================================================================================================================

.. index::
  single: Felix A. Gers
  single: Jürgen Schmidhuber
  single: Fred Cummins

.. ====================================================================================================================
.. Set index for conference/journal.
.. ====================================================================================================================

.. index::
  single: Neural Computation

.. ====================================================================================================================
.. Set index for publishing time.
.. ====================================================================================================================

.. index::
  single: 2000

.. ====================================================================================================================
.. Setup SEO.
.. ====================================================================================================================

.. meta::
  :description:
    提出在 LSTM 上增加 forget gate
  :keywords:
    LSTM,
    RNN,
    Sequence Model,
    model architecture,
    neural network

.. ====================================================================================================================
.. Setup front matter.
.. ====================================================================================================================

.. tab-set::

  .. tab-item:: Tags

    :bdg-secondary:`LSTM`
    :bdg-secondary:`Model Architecture`
    :bdg-secondary:`RNN`
    :bdg-secondary:`Sequence Model`
    :bdg-primary:`Neural Computation`

  .. tab-item:: Authors

    Felix A. Gers, Jürgen Schmidhuber, Fred Cummins

  .. tab-item:: Date

    2000

  .. tab-item:: Journal

    Neural Computation

  .. tab-item:: Link

    論文連結 :footcite:`gers-etal-2000-learning`

    .. ================================================================================================================
    .. Define math macros. We put macros here so that user will not see them loading.
    .. ================================================================================================================

    .. math::
      :nowrap:

      \[
        % Operators.
        \newcommand{\opbk}{\operatorname{bk}}
        \newcommand{\opfg}{\operatorname{fg}}
        \newcommand{\opig}{\operatorname{ig}}
        \newcommand{\opin}{\operatorname{in}}
        \newcommand{\oplen}{\operatorname{len}}
        \newcommand{\opog}{\operatorname{og}}
        \newcommand{\opout}{\operatorname{out}}

        % Memory cell blocks.
        \newcommand{\bk}[1]{{\opbk^{#1}}}

        % Vectors' notations.
        \newcommand{\s}{\mathbf{s}}
        \newcommand{\sbk}[1]{\s^\bk{#1}}
        \newcommand{\x}{\mathbf{x}}
        \newcommand{\xout}{\x^\opout}
        \newcommand{\xt}{\tilde{\x}}
        \newcommand{\y}{\mathbf{y}}
        \newcommand{\yh}{\hat{\y}}
        \newcommand{\ybk}[1]{\y^\bk{#1}}
        \newcommand{\yfg}{\y^\opfg}
        \newcommand{\yig}{\y^\opig}
        \newcommand{\yog}{\y^\opog}
        \newcommand{\z}{\mathbf{z}}
        \newcommand{\zbk}[1]{\z^\bk{#1}}
        \newcommand{\zfg}{\z^\opfg}
        \newcommand{\zig}{\z^\opig}
        \newcommand{\zog}{\z^\opog}
        \newcommand{\zout}{\z^\opout}

        % Matrixs' notation.
        \newcommand{\W}{\mathbf{W}}
        \newcommand{\Wbk}[1]{\W^\bk{#1}}
        \newcommand{\Wfg}{\W^\opfg}
        \newcommand{\Wig}{\W^\opig}
        \newcommand{\Wog}{\W^\opog}
        \newcommand{\Wout}{\W^\opout}

        % Symbols in mathcal.
        \newcommand{\cL}{\mathcal{L}}
        \newcommand{\cT}{\mathcal{T}}

        % Dimensions.
        \newcommand{\din}{{d_\opin}}
        \newcommand{\dout}{{d_\opout}}
        \newcommand{\dbk}{{d_\opbk}}
        \newcommand{\nbk}{{n_\opbk}}

        % Gradient approximation by truncating gradient.
        \newcommand{\aptr}{\approx_{\operatorname{tr}}}
      \]

重點
====

- 此篇論文 :footcite:`gers-etal-2000-learning` 與原版 LSTM :footcite:`hochreiter-etal-1997-long` 都寫錯自己的數學公式，但我的筆記內容主要以正確版本為主，原版 LSTM 可以參考\ :doc:`我的筆記 </post/ml/long-short-term-memory>`
- 原版 LSTM 沒有 forget gate units，現今常用的 LSTM 都有 forget gate units，概念由此篇論文提出
- 包含多個子序列的\ **連續輸入**\會讓原版 LSTM 的 memory cell internal states 累加成極正或極負

  - 現實中的大多數資料並不存在好的分割序列演算法，導致輸入給模型的資料通常都包含多個子序列
  - 根據實驗 1 的分析發現 memory cell internal states 的累加導致預測結果完全錯誤

- 使用 forget gate units 讓模型學會適當的忘記已經處理過的子序列資訊
- 當 forget gate units 的 **bias term** 初始化為\ **正數**\時會記住 memory cell internal states，等同於使用原版的 LSTM
- 因此使用 forget gate units 的 LSTM 能夠達成原版 LSTM 的功能，並額外擁有自動重設 memory cells 的機制
- 這篇論文的理論背景較少，實驗為主的描述居多

原始 LSTM
=========

.. note::

  這篇論文不使用 conventional hidden units，因此我不列出相關的公式。

符號定義
--------

我使用的符號與論文不同，我的符號定義請參考\ :doc:`我的筆記 </post/ml/long-short-term-memory>`。

+--------------------+---------------------------------------------------------------------------------------------------+---------------------+
| Symbol             | Meaning                                                                                           | Value Range         |
+====================+===================================================================================================+=====================+
| :math:`\dbk`       | Number of memory cells in each memory cell block at time step :math:`t`.                          | :math:`\Z^+`        |
+--------------------+---------------------------------------------------------------------------------------------------+---------------------+
| :math:`\nbk`       | Number of memory cell blocks at time step :math:`t`.                                              | :math:`\Z^+`        |
+--------------------+---------------------------------------------------------------------------------------------------+---------------------+
| :math:`\x(t)`      | LSTM input at time step :math:`t`.                                                                | :math:`\R^\din`     |
+--------------------+---------------------------------------------------------------------------------------------------+---------------------+
| :math:`\yig(t)`    | Input gate units at time step :math:`t`.                                                          | :math:`[0, 1]^\nbk` |
+--------------------+---------------------------------------------------------------------------------------------------+---------------------+
| :math:`\yog(t)`    | Output gate units at time step :math:`t`.                                                         | :math:`[0, 1]^\nbk` |
+--------------------+---------------------------------------------------------------------------------------------------+---------------------+
| :math:`\ybk{k}(t)` | Output of the :math:`k`-th memory cell block at time step :math:`t`.                              | :math:`\R^\dbk`     |
+--------------------+---------------------------------------------------------------------------------------------------+---------------------+
| :math:`\sbk{k}(t)` | Internal states of the :math:`k`-th memory cell block at time step :math:`t`.                     | :math:`\R^\dbk`     |
+--------------------+---------------------------------------------------------------------------------------------------+---------------------+
| :math:`\y(t)`      | LSTM output at time step :math:`t`.                                                               | :math:`\R^\dout`    |
+--------------------+---------------------------------------------------------------------------------------------------+---------------------+
| :math:`\sigma`     | Sigmoid function.                                                                                 | :math:`[0, 1]`      |
+--------------------+---------------------------------------------------------------------------------------------------+---------------------+
| :math:`f^\opig`    | Activation function for input gate units, set to :math:`\sigma` in this paper.                    | :math:`[0, 1]`      |
+--------------------+---------------------------------------------------------------------------------------------------+---------------------+
| :math:`f^\opog`    | Activation function for output gate units, set to :math:`\sigma` in this paper.                   | :math:`[0, 1]`      |
+--------------------+---------------------------------------------------------------------------------------------------+---------------------+
| :math:`f^\opout`   | Activation function for output units, set to :math:`\sigma` in this paper.                        | :math:`[0, 1]`      |
+--------------------+---------------------------------------------------------------------------------------------------+---------------------+
| :math:`g`          | Activation function for memory cells, set to :math:`4 \sigma - 2` in this paper.                  | :math:`[-2, 2]`     |
+--------------------+---------------------------------------------------------------------------------------------------+---------------------+
| :math:`h`          | Activation function for memory cell block activations, set to :math:`2 \sigma - 1` in this paper. | :math:`[-1, 1]`     |
+--------------------+---------------------------------------------------------------------------------------------------+---------------------+

計算定義
--------

以下就是 LSTM（1997 版本）的計算流程。

.. math::
  :nowrap:

  \[
    \begin{align*}
      & \algoProc{\operatorname{LSTM1997}}(\x, \Wig, \Wog, \Wbk{1}, \dots, \Wbk{\nbk}, \Wout) \\
      & \indent{1} \algoCmt{Initialize activations with zeros.} \\
      & \indent{1} \cT \algoEq \oplen(\x) \\
      & \indent{1} \yig(0) \algoEq \zv \\
      & \indent{1} \yog(0) \algoEq \zv \\
      & \indent{1} \algoFor{k \in \Set{1, \dots, \nbk}} \\
      & \indent{2}   \sbk{k}(0) \algoEq \zv \\
      & \indent{2}   \ybk{k}(0) \algoEq \zv \\
      & \indent{1} \algoEndFor \\
      & \indent{1} \algoCmt{Do forward pass.} \\
      & \indent{1} \algoFor{t \in \Set{0, \dots, \cT - 1}} \\
      & \indent{2}   \algoCmt{Concatenate input units with activations.} \\
      & \indent{2}   \xt(t) \algoEq \begin{pmatrix}
                                      \x(t) \\
                                      \yig(t) \\
                                      \yog(t) \\
                                      \ybk{1}(t) \\
                                      \vdots \\
                                      \ybk{\nbk}(t)
                                    \end{pmatrix} \\
      & \indent{2}   \algoCmt{Compute input gate units' activations.} \\
      & \indent{2}   \zig(t + 1) \algoEq \Wig \cdot \xt(t) \\
      & \indent{2}   \yig(t + 1) \algoEq f^\opig\qty(\zig(t + 1)) \\
      & \indent{2}   \algoCmt{Compute output gate units' activations.} \\
      & \indent{2}   \zog(t + 1) \algoEq \Wog \cdot \xt(t) \\
      & \indent{2}   \yog(t + 1) \algoEq f^\opog\qty(\zog(t + 1)) \\
      & \indent{2}   \algoCmt{Compute the k-th memory cell block's activations.} \\
      & \indent{2}   \algoFor{k \in \Set{1, \dots, \nbk}} \\
      & \indent{3}     \zbk{k}(t + 1) \algoEq \Wbk{k} \cdot \xt(t) \\
      & \indent{3}     \sbk{k}(t + 1) \algoEq \sbk{k}(t) + \yig_k(t + 1) \cdot g\qty(\zbk{k}(t + 1)) \\
      & \indent{3}     \ybk{k}(t + 1) \algoEq \yog_k(t + 1) \cdot h\qty(\sbk{k}(t + 1)) \\
      & \indent{2}   \algoEndFor \\
      & \indent{2}   \algoCmt{Concatenate input units with new activations.} \\
      & \indent{2}   \xout(t + 1) \algoEq \begin{pmatrix}
                                            \x(t) \\
                                            \ybk{1}(t + 1) \\
                                            \vdots \\
                                            \ybk{\nbk}(t + 1) \\
                                          \end{pmatrix} \\
      & \indent{2}   \algoCmt{Compute outputs.} \\
      & \indent{2}   \zout(t + 1) \algoEq \Wout \cdot \xout(t + 1) \\
      & \indent{2}   \y(t + 1) \algoEq f^\opout\qty(\zout(t + 1)) \\
      & \indent{1} \algoEndFor \\
      & \indent{1} \algoReturn \y(1), \dots, \y(\cT) \\
      & \algoEndProc
    \end{align*}
  \]

.. error::

  此篇論文 :footcite:`gers-etal-2000-learning` 與原版 LSTM 的論文 :footcite:`hochreiter-etal-1997-long` 都不小心將 :math:`\y(t + 1)` 的輸入寫成 :math:`\ybk{k}(t)` 而不是 :math:`\ybk{k}(t + 1)`，我在上述公式中已經進行修正。
  對應的正確寫法在後續論文 :footcite:`gers-etal-2002-learning` 中才終於寫對。


參數結構
--------

+-----------------+----------------------------------------------------------------------------------------------------+---------------------+---------------------------------------+
| Parameter       | Meaning                                                                                            | Output Vector Shape | Input Vector Shape                    |
+=================+====================================================================================================+=====================+=======================================+
| :math:`\Wig`    | Weight matrix connect :math:`\xt(t)` to input gate units :math:`\yig(t + 1)`.                      | :math:`\nbk`        | :math:`\din + \nbk \times (2 + \dbk)` |
+-----------------+----------------------------------------------------------------------------------------------------+---------------------+---------------------------------------+
| :math:`\Wog`    | Weight matrix connect :math:`\xt(t)` to output gate units :math:`\yog(t + 1)`.                     | :math:`\nbk`        | :math:`\din + \nbk \times (2 + \dbk)` |
+-----------------+----------------------------------------------------------------------------------------------------+---------------------+---------------------------------------+
| :math:`\Wbk{k}` | Weight matrix connect :math:`\xt(t)` to the :math:`k`-th memory cell block :math:`\ybk{k}(t + 1)`. | :math:`\dbk`        | :math:`\din + \nbk \times (2 + \dbk)` |
+-----------------+----------------------------------------------------------------------------------------------------+---------------------+---------------------------------------+
| :math:`\Wout`   | Weight matrix connect :math:`\xout(t)` to output units :math:`\y(t + 1)`.                          | :math:`\dbk`        | :math:`\din + \nbk \times \dbk`       |
+-----------------+----------------------------------------------------------------------------------------------------+---------------------+---------------------------------------+

最佳化
------

原始 LSTM 論文 :footcite:`hochreiter-etal-1997-long` 提出與 truncated BPTT 相似的概念，透過 RTRL 進行參數更新，故意\ **丟棄部份微分值**\來近似全微分，避免梯度爆炸或梯度消失的問題，同時節省更新所需的空間與時間（local in time and space）。
丟棄微分後的近似結果我以 :math:`\aptr` 表示，推導細節請見\ :doc:`我的筆記 </post/ml/long-short-term-memory>`，以下我直接列出對應的公式。

:math:`\Wout` 相對於誤差的微分
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::
  :nowrap:

  \[
    \begin{align*}
      & \dv{\cL\qty(\y(t + 1), \yh(t + 1))}{\Wout_{p, q}} = \qty(\y_p(t + 1) - \yh_p(t + 1)) \cdot {f^\opout}'\qty(\zout_p(t + 1)) \cdot \xout_q(t + 1) \\
      & \qqtext{where} \begin{dcases}
                         p \in \Set{1, \dots, \dout} \\
                         q \in \Set{1, \dots, \din + \nbk \times \dbk} \\
                         t \in \Set{0, \dots, \cT - 1}
                       \end{dcases}.
    \end{align*}
    \tag{1}\label{1}
  \]

:math:`\Wog` 相對於誤差的微分近似值
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::
  :nowrap:

  \[
    \begin{align*}
      & \dv{\cL\qty(\y(t + 1) - \yh(t + 1))}{\Wog_{p, q}} \aptr \qty(\sum_{j = 1}^\dbk \qty[\sum_{i = 1}^\dout \qty(\y_i(t + 1) - \yh_i(t + 1)) \cdot {f^\opout}'\qty(\zout_i(t + 1)) \cdot \Wout_{i, \din + (p - 1) \times \dbk + j}] \cdot h\qty(\sbk{p}_j(t + 1))) \cdot {f^\opog}'\qty(\zog_p(t + 1)) \cdot \xt_q(t) \\
      & \qqtext{where} \begin{dcases}
                         p \in \Set{1, \dots, \nbk} \\
                         q \in \Set{1, \dots, \din + \nbk \times (2 + \dbk)} \\
                         t \in \Set{0, \dots, \cT - 1}
                       \end{dcases}.
    \end{align*}
    \tag{2}\label{2}
  \]

:math:`\Wig` 相對於誤差的微分近似值
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::
  :nowrap:

  \[
    \begin{align*}
      & \dv{\cL\qty(\y(t + 1) - \yh(t + 1))}{\Wig_{p, q}} \aptr \qty(\sum_{j = 1}^\dbk \qty[\sum_{i = 1}^\dout \qty(\y_i(t + 1) - \yh_i(t + 1)) \cdot {f^\opout}'\qty(\zout_i(t + 1)) \cdot \Wout_{i, \din + (p - 1) \times \dbk + j}] \cdot h'\qty(\sbk{p}_j(t + 1)) \cdot \sum_{t^\star = 0}^t \qty[{f^\opig}'\qty(\zig_p(t^\star + 1)) \cdot \xt_q(t^\star) \cdot g\qty(\zbk{p}_j(t^\star + 1))]) \cdot \yog_p(t + 1) \\
      & \qqtext{where} \begin{dcases}
                         p \in \Set{1, \dots, \nbk} \\
                         q \in \Set{1, \dots, \din + \nbk \times (2 + \dbk)} \\
                         t \in \Set{0, \dots, \cT - 1}
                       \end{dcases}.
    \end{align*}
    \tag{3}\label{3}
  \]

:math:`\Wbk{k}` 相對於誤差的微分近似值
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::
  :nowrap:

  \[
    \begin{align*}
      & \dv{\cL\qty(\y(t + 1) - \yh(t + 1))}{\Wbk{k}_{p, q}} \aptr \qty[\sum_{i = 1}^\dout \qty(\y_i(t + 1) - \yh_i(t + 1)) \cdot {f^\opout}'\qty(\zout_i(t + 1)) \cdot \Wout_{i, \din + (k - 1) \times \dbk + p}] \cdot \qty[\sum_{t^\star = 0}^t \yig_k(t^\star + 1) \cdot g'\qty(\zbk{k}_p(t^\star + 1)) \cdot \xt_q(t^\star)] \cdot \yog_k(t + 1) \cdot h'\qty(\sbk{k}_p(t + 1)) \\
      & \qqtext{where} \begin{dcases}
                         k \in \Set{1, \dots, \nbk} \\
                         p \in \Set{1, \dots, \dbk} \\
                         q \in \Set{1, \dots, \din + \nbk \times (2 + \dbk)} \\
                         t \in \Set{0, \dots, \cT - 1}
                       \end{dcases}.
    \end{align*}
    \tag{4}\label{4}
  \]

參數更新演算法
~~~~~~~~~~~~~~

參數更新使用的演算法為 :term:`gradient descent`，:math:`\alpha` 為 learning rate：

.. math::
  :nowrap:

  \[
    \begin{align*}
      \Wout_{p, q}   & \algoEq \Wout_{p, q} - \alpha \cdot \dv{\cL\qty(\y(t + 1), \yh(t + 1))}{\Wout_{p, q}} \qqtext{where} \begin{dcases}
                                                                                                                              p \in \Set{1, \dots, \dout} \\
                                                                                                                              q \in \Set{1, \dots, \din + \nbk \times \dbk} \\
                                                                                                                              t \in \Set{0, \dots, \cT - 1}
                                                                                                                            \end{dcases}. \\
      \Wog_{p, q}    & \algoEq \Wog_{p, q} - \alpha \cdot \dv{\cL\qty(\y(t + 1), \yh(t + 1))}{\Wog_{p, q}} \qqtext{where} \begin{dcases}
                                                                                                                            p \in \Set{1, \dots, \nbk} \\
                                                                                                                            q \in \Set{1, \dots, \din + \nbk \times (2 + \dbk)} \\
                                                                                                                            t \in \Set{0, \dots, \cT - 1}
                                                                                                                          \end{dcases}. \\
      \Wig_{p, q}    & \algoEq \Wig_{p, q} - \alpha \cdot \dv{\cL\qty(\y(t + 1), \yh(t + 1))}{\Wig_{p, q}} \qqtext{where} \begin{dcases}
                                                                                                                            p \in \Set{1, \dots, \nbk} \\
                                                                                                                            q \in \Set{1, \dots, \din + \nbk \times (2 + \dbk)} \\
                                                                                                                            t \in \Set{0, \dots, \cT - 1}
                                                                                                                          \end{dcases}. \\
      \Wbk{k}_{p, q} & \algoEq \Wbk{k}_{p, q} - \alpha \cdot \dv{\cL\qty(\y(t + 1), \yh(t + 1))}{\Wbk{k}_{p, q}} \qqtext{where} \begin{dcases}
                                                                                                                                  k \in \Set{1, \dots, \nbk} \\
                                                                                                                                  p \in \Set{1, \dots, \dbk} \\
                                                                                                                                  q \in \Set{1, \dots, \din + \nbk \times (2 + \dbk)} \\
                                                                                                                                  t \in \Set{0, \dots, \cT - 1}
                                                                                                                                \end{dcases}.
    \end{align*}
    \tag{5}\label{5}
  \]

由於使用基於 RTRL 的最佳化演算法，因此每個時間點 :math:`t + 1` 計算完誤差後就可以更新參數。

問題
====

一個 RNN 模型一次只能處理一個序列，並且假設每個序列有\ **明確的結尾**。
當一個輸入序列中包含多個斷點，通常會在前處理階段就將該序列切割成多個子序列，並分次處理。
但如果子序列\ **沒有**\明確的斷點標記，則模型就必須擁有\ **判斷序列斷點**\的能力，並且自動在\ **計算過程中重設計算狀態**。

原始 LSTM :footcite:`hochreiter-etal-1997-long` 架構假設輸入序列\ **不包含**\多個子序列，因此只會在處理序列\ **前**\重設模型的計算狀態，沒有在計算過程中重設計算狀態的需求。
但當輸入包含多個子序列，且沒有明確的方法辨識不同子序列的斷點時，LSTM 模型架構會讓計算出問題，主要原因來自於以下公式：

.. math::
  :nowrap:

  \[
    \sbk{k}(t + 1) \algoEq \sbk{k}(t) + \yig_k(t + 1) \cdot g\qty(\zbk{k}(t + 1))
    \tag{6}\label{6}
  \]
  \[
    \ybk{k}(t + 1) \algoEq \yog_k(t + 1) \cdot h\qty(\sbk{k}(t + 1))
    \tag{7}\label{7}
  \]

因為沒有明確的斷點，所以不會有\ **重設/歸零** memory cell internal states 的動作，因此 memory cell internal states 會透過式子 :math:`\eqref{6}` 不斷累加，朝向\ **極正**\或\ **極負**\值前進。
極值會導致式子 :math:`\eqref{7}` 內經由 :math:`h` 產生的 activation 為 :math:`2` （極正）或 :math:`-2` （極負），因此式子 :math:`\eqref{7}` 的輸出就會完全取決於 output gate units 的數值，同時也喪失了 memory cells 記憶的用途。

作者提出了幾個可行的方案，但都再自己一一否決：

- 使用 time-delay neural networks，但這代表必須對子序列斷點的長度進行假設，因此不可行
- 使用 weight decay 限制 memory cell internal states 數值增長的速度，但仍然會走向極值
- 改變最佳化演算法，沒有解釋原因作者直接說不行，我猜是實驗結論
- 將 memory cell internal states 乘上一個 decay constants，但這代表多了一個 hyperparameter 要調整，而且實驗結果也顯示效果不好

最後作者基於最後一個提案進行改良，提出了 **forget gate units** 的機制。

Forget Gate Units
=================

模型架構
--------

.. figure:: https://i.imgur.com/ILRsaEU.png
  :alt: 在原始 LSTM 架構上增加 forget gate units
  :name: paper-fig-1

  圖 1：在原始 LSTM 架構上增加 forget gate。

  表格來源：:footcite:`gers-etal-2000-learning`。

作者提出在模型中加入 **forget gate units**，概念是讓 memory cell internal states 能夠自動進行重設。
如同 input/output gate units，forget gate units 會初始化成 :math:`\zv`，並透過以下計算更新 forget gate units：

.. math::
  :nowrap:

  \[
    \begin{align*}
      & \xt(t) \algoEq \begin{pmatrix}
                         \x(t) \\
                         \yfg(t) \\
                         \yig(t) \\
                         \yog(t) \\
                         \ybk{1}(t) \\
                         \vdots \\
                         \ybk{\nbk}(t)
                       \end{pmatrix} \\
      & \zfg(t + 1) \algoEq \Wfg \cdot \xt(t) \\
      & \yfg(t + 1) \algoEq f^\opfg\qty(\zfg(t + 1))
    \end{align*}
    \tag{8}\label{8}
  \]

注意以下幾點連帶的改動：

- :math:`\xt(t)` 的輸入需要加上 :math:`\yfg(t)`
- 新增了參數 :math:`\Wfg`，該參數的 input vector shape 為 :math:`\din + \nbk \times (3 + \dbk)`，output vector shape 為 :math:`\nbk`
- 因為 :math:`\xt(t)` 做了更動，所以參數 :math:`\Wig, \Wog, \Wbk{k}` 的 input vector shape 都改成 :math:`\din + \nbk \times (3 + \dbk)`

.. note::

  式子 :math:`\eqref{8}` 就是論文中的 (3.1) 式。

由於 forget gate units 的設計出發點是作為 memory cell internal states 的 decay factor，因此作者將式子 :math:`\eqref{6}` 的計算方法改成如下：

.. math::
  :nowrap:

  \[
    \sbk{k}(t + 1) \algoEq \yfg_k(t + 1) \cdot \sbk{k}(t) + \yig_k(t + 1) \cdot g\qty(\zbk{k}(t + 1))
    \tag{9}\label{9}
  \]

- Forget gate units 是以\ **乘法**\參與計算，因此稱為 **multiplicative gate units**

  - Memory cells in the same memory cell block **share** the same forget gate unit
  - 因此 :math:`\yfg_k(t + 1) \cdot \sbk{k}` 中的乘法是\ **純量**\乘上\ **向量**

- 模型會在訓練的過程中學習\ **關閉**\與\ **開啟** forget gate units

  - :math:`\yfg_k(t + 1) \approx 0` 代表\ **關閉** :math:`t + 1` 時間點的第 :math:`k` 個 forget gate unit，並\ **重設** :math:`\sbk{k}` 的計算狀態
  - :math:`\yfg_k(t + 1) \approx 1` 代表\ **開啟** :math:`t + 1` 時間點的第 :math:`k` 個 forget gate unit，並\ **保留** :math:`\sbk{k}` 的計算狀態
  - 全部 :math:`\nbk` 個 forget gate units 不一定要同時關閉或開啟

.. note::

  式子 :math:`\eqref{9}` 就是論文中的 (3.2) 式。

計算定義
--------

加入 forget gate units 後我重新整理一次 LSTM 的計算定義，如下所示。

.. math::
  :nowrap:

  \[
    \begin{align*}
      & \algoProc{\operatorname{LSTM1997}}(\x, \Wfg, \Wig, \Wog, \Wbk{1}, \dots, \Wbk{\nbk}, \Wout) \\
      & \indent{1} \algoCmt{Initialize activations with zeros.} \\
      & \indent{1} \cT \algoEq \oplen(\x) \\
      & \indent{1} \yfg(0) \algoEq \zv \\
      & \indent{1} \yig(0) \algoEq \zv \\
      & \indent{1} \yog(0) \algoEq \zv \\
      & \indent{1} \algoFor{k \in \Set{1, \dots, \nbk}} \\
      & \indent{2}   \sbk{k}(0) \algoEq \zv \\
      & \indent{2}   \ybk{k}(0) \algoEq \zv \\
      & \indent{1} \algoEndFor \\
      & \indent{1} \algoCmt{Do forward pass.} \\
      & \indent{1} \algoFor{t \in \Set{0, \dots, \cT - 1}} \\
      & \indent{2}   \algoCmt{Concatenate input units with activations.} \\
      & \indent{2}   \xt(t) \algoEq \begin{pmatrix}
                                      \x(t) \\
                                      \yfg(t) \\
                                      \yig(t) \\
                                      \yog(t) \\
                                      \ybk{1}(t) \\
                                      \vdots \\
                                      \ybk{\nbk}(t)
                                    \end{pmatrix} \\
      & \indent{2}   \algoCmt{Compute forget gate units' activations.} \\
      & \indent{2}   \zfg(t + 1) \algoEq \Wfg \cdot \xt(t) \\
      & \indent{2}   \yfg(t + 1) \algoEq f^\opfg\qty(\zfg(t + 1)) \\
      & \indent{2}   \algoCmt{Compute input gate units' activations.} \\
      & \indent{2}   \zig(t + 1) \algoEq \Wig \cdot \xt(t) \\
      & \indent{2}   \yig(t + 1) \algoEq f^\opig\qty(\zig(t + 1)) \\
      & \indent{2}   \algoCmt{Compute output gate units' activations.} \\
      & \indent{2}   \zog(t + 1) \algoEq \Wog \cdot \xt(t) \\
      & \indent{2}   \yog(t + 1) \algoEq f^\opog\qty(\zog(t + 1)) \\
      & \indent{2}   \algoCmt{Compute the k-th memory cell block's activations.} \\
      & \indent{2}   \algoFor{k \in \Set{1, \dots, \nbk}} \\
      & \indent{3}     \zbk{k}(t + 1) \algoEq \Wbk{k} \cdot \xt(t) \\
      & \indent{3}     \sbk{k}(t + 1) \algoEq \yfg_k(t + 1) \cdot \sbk{k}(t) + \yig_k(t + 1) \cdot g\qty(\zbk{k}(t + 1)) \\
      & \indent{3}     \ybk{k}(t + 1) \algoEq \yog_k(t + 1) \cdot h\qty(\sbk{k}(t + 1)) \\
      & \indent{2}   \algoEndFor \\
      & \indent{2}   \algoCmt{Concatenate input units with new activations.} \\
      & \indent{2}   \xout(t + 1) \algoEq \begin{pmatrix}
                                            \x(t) \\
                                            \ybk{1}(t + 1) \\
                                            \vdots \\
                                            \ybk{\nbk}(t + 1) \\
                                          \end{pmatrix} \\
      & \indent{2}   \algoCmt{Compute outputs.} \\
      & \indent{2}   \zout(t + 1) \algoEq \Wout \cdot \xout(t + 1) \\
      & \indent{2}   \y(t + 1) \algoEq f^\opout\qty(\zout(t + 1)) \\
      & \indent{1} \algoEndFor \\
      & \indent{1} \algoReturn \y(1), \dots, \y(\cT) \\
      & \algoEndProc
    \end{align*}
  \]

.. error::

  根據論文中的 3.4 式，論文 2.5 式的 :math:`t - 1` 應該改成 :math:`t`。

Bias Terms
----------

原始 LSTM :footcite:`hochreiter-etal-1997-long` 提出對 input/output gate units 使用 **bias terms** 參數並初始化成\ **負數**，如此可以讓 input/output gate units 在需要的時候才被啟用，並同時避免一些 LSTM 計算上的問題（細節可以看\ :doc:`我的筆記 </post/ml/long-short-term-memory>`）。
而 forget gate units 也可以使用 bias terms，但初始化的數值應該為\ **正數**，理由是在模型計算前期應該要讓 forget gate units **開啟**，讓 memory cell internal states 的數值能夠進行改變。
注意 forget gate 只有在\ **關閉**\時才能進行狀態重設，這個名字取得不是很好。

.. dropdown:: 推導初始化 forget gate bias 為正數的邏輯

  .. math::
    :nowrap:

    \[
      \begin{align*}
                 & b_k^\opfg \gg 0 \qqtext{where} k \in \Set{1, \dots, \nbk} \\
        \implies & \zfg_k(1) \gg 0 \qqtext{where} k \in \Set{1, \dots, \nbk} \\
        \implies & \yfg_k(1) \approx 1 \qqtext{where} k \in \Set{1, \dots, \nbk} \\
        \implies & \yfg_k(1) \cdot \sbk{k}_i(0) \approx \sbk{k}_i(0) = 0 \qqtext{where} \begin{dcases}
                                                                                          i \in \Set{1, \dots, \dbk} \\
                                                                                          k \in \Set{1, \dots, \nbk}
                                                                                        \end{dcases} \\
        \implies & \sbk{k}_i(1) = \yfg_k(1) \cdot \sbk{k}_i(0) + \yfg_k(1) \cdot g\qty(\zbk{k}_i(1)) \approx \yfg_k(1) \cdot g\qty(\zbk{k}_i(1)) \qqtext{where} \begin{dcases}
                                                                                                                                                                  i \in \Set{1, \dots, \dbk} \\
                                                                                                                                                                  k \in \Set{1, \dots, \nbk}
                                                                                                                                                                \end{dcases}.
      \end{align*}
    \]

最佳化
------

此篇論文採用與原始 LSTM :footcite:`hochreiter-etal-1997-long` 相同的最佳化演算法，只是因為架構上多了 LSTM，因此需要對式子 :math:`\eqref{2} \eqref{3} \eqref{4}` 做一些修正，並新增 :math:`\Wfg` 相對於誤差的微分近似值。
以下我使用 :math:`\aptr` 代表微分近似的結果，符號與功能均遵循原版 LSTM 論文。
由於此篇論文不再使用 conventional hidden units，因此我將所有相關的公式都省略。

丟棄微分值
~~~~~~~~~~
同原始 LSTM 論文，此論文將所有與 **hidden units** 相連的節點 :math:`\xt(t)` 產生的微分值一律\ **丟棄**，公式如下：

.. math::
  :nowrap:

  \[
    \begin{align*}
      \dv{\zfg_k(t + 1)}{\xt_j(t)}    & \aptr 0 \qqtext{where} \begin{dcases}
                                                                 j \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                                 k \in \Set{1, \dots, \nbk} \\
                                                                 t \in \Set{0, \dots, \cT - 1}
                                                               \end{dcases}. \\
      \dv{\zig_k(t + 1)}{\xt_j(t)}    & \aptr 0 \qqtext{where} \begin{dcases}
                                                                 j \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                                 k \in \Set{1, \dots, \nbk} \\
                                                                 t \in \Set{0, \dots, \cT - 1}
                                                               \end{dcases}. \\
      \dv{\zog_k(t + 1)}{\xt_j(t)}    & \aptr 0 \qqtext{where} \begin{dcases}
                                                                 j \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                                 k \in \Set{1, \dots, \nbk} \\
                                                                 t \in \Set{0, \dots, \cT - 1}
                                                               \end{dcases}. \\
      \dv{\zbk{k}_i(t + 1)}{\xt_j(t)} & \aptr 0 \qqtext{where} \begin{dcases}
                                                                 i \in \Set{1, \dots, \dbk} \\
                                                                 j \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                                 k \in \Set{1, \dots, \nbk} \\
                                                                 t \in \Set{0, \dots, \cT - 1}
                                                               \end{dcases}. \\
      \dv{\sbk{k}_i(t)}{\xt_j(t)}     & \aptr 0 \qqtext{where} \begin{dcases}
                                                                 i \in \Set{1, \dots, \dbk} \\
                                                                 j \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                                 k \in \Set{1, \dots, \nbk} \\
                                                                 t \in \Set{0, \dots, \cT - 1}
                                                               \end{dcases}.
    \end{align*}
    \tag{10}\label{10}
  \]

.. note::

  上述公式與原版 LSTM 論文中所使用的公式多了 forget gate units，並增加了 :math:`\xt(t)` 的 shape（從 :math:`2 + \dbk` 變成 :math:`3 + \dbk`）。

根據 :math:`\eqref{10}` 我們可以進一步推得以下微分近似值：

.. math::
  :nowrap:

  \[
    \begin{align*}
      \dv{\yfg_k(t + 1)}{\xt_j(t)}    & \aptr 0 \qqtext{where} \begin{dcases}
                                                                 j \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                                 k \in \Set{1, \dots, \nbk} \\
                                                                 t \in \Set{0, \dots, \cT - 1}
                                                               \end{dcases}. \\
      \dv{\yig_k(t + 1)}{\xt_j(t)}    & \aptr 0 \qqtext{where} \begin{dcases}
                                                                 j \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                                 k \in \Set{1, \dots, \nbk} \\
                                                                 t \in \Set{0, \dots, \cT - 1}
                                                               \end{dcases}. \\
      \dv{\yog_k(t + 1)}{\xt_j(t)}    & \aptr 0 \qqtext{where} \begin{dcases}
                                                                 j \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                                 k \in \Set{1, \dots, \nbk} \\
                                                                 t \in \Set{0, \dots, \cT - 1}
                                                               \end{dcases}. \\
      \dv{\sbk{k}_i(t + 1)}{\xt_j(t)} & \aptr 0 \qqtext{where} \begin{dcases}
                                                                 i \in \Set{1, \dots, \dbk} \\
                                                                 j \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                                 k \in \Set{1, \dots, \nbk} \\
                                                                 t \in \Set{0, \dots, \cT - 1}
                                                               \end{dcases}. \\
      \dv{\ybk{k}_i(t + 1)}{\xt_j(t)} & \aptr 0 \qqtext{where} \begin{dcases}
                                                                 i \in \Set{1, \dots, \dbk} \\
                                                                 j \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                                 k \in \Set{1, \dots, \nbk} \\
                                                                 t \in \Set{0, \dots, \cT - 1}
                                                               \end{dcases}.
    \end{align*}
    \tag{11}\label{11}
  \]

.. dropdown:: 推導 :math:`\eqref{11}`

  首先根據式子 :math:`\eqref{10}` 的定義可以得到以下微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\yfg_k(t + 1)}{\xt_j(t)} & = \dv{\yfg_k(t + 1)}{\zfg_k(t + 1)} \cdot \cancelto{\aptr 0}{\dv{\zfg_k(t + 1)}{\xt_j(t)}} \\
                                     & \aptr 0 \qqtext{where} \begin{dcases}
                                                                j \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                                k \in \Set{1, \dots, \nbk} \\
                                                                t \in \Set{0, \dots, \cT - 1}
                                                              \end{dcases}. \\
        \dv{\yig_k(t + 1)}{\xt_j(t)} & = \dv{\yig_k(t + 1)}{\zig_k(t + 1)} \cdot \cancelto{\aptr 0}{\dv{\zig_k(t + 1)}{\xt_j(t)}} \\
                                     & \aptr 0 \qqtext{where} \begin{dcases}
                                                                j \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                                k \in \Set{1, \dots, \nbk} \\
                                                                t \in \Set{0, \dots, \cT - 1}
                                                              \end{dcases}. \\
        \dv{\yog_k(t + 1)}{\xt_j(t)} & = \dv{\yog_k(t + 1)}{\zog_k(t + 1)} \cdot \cancelto{\aptr 0}{\dv{\zog_k(t + 1)}{\xt_j(t)}} \\
                                     & \aptr 0 \qqtext{where} \begin{dcases}
                                                                j \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                                k \in \Set{1, \dots, \nbk} \\
                                                                t \in \Set{0, \dots, \cT - 1}
                                                              \end{dcases}.
      \end{align*}
    \]

  接著利用上述的結果結合 :math:`\eqref{10}` 推導出 :math:`\xt(t)` 對於 memory cell internal states 的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\sbk{k}_i(t + 1)}{\xt_j(t)} & = \cancelto{\aptr 0}{\dv{\yfg_k(t + 1)}{\xt_j(t)}} \cdot \dv{\sbk{k}_i(t)}{\xt_j(t)} + \dv{\yfg_k(t + 1)}{\xt_j(t)} \cdot \cancelto{\aptr 0}{\dv{\sbk{k}_i(t)}{\xt_j(t)}} + \cancelto{\aptr 0}{\dv{\yig_k(t + 1)}{\xt_j(t)}} \cdot g\qty(\zbk{k}_i(t + 1)) + \yig_k(t + 1) \cdot \dv{g\qty(\zbk{k}_i(t + 1))}{\zbk{k}_i(t + 1)} \cdot \cancelto{\aptr 0}{\dv{\zbk{k}_i(t + 1)}{\xt_j(t)}} \\
                                        & \aptr 0 \qqtext{where} \begin{dcases}
                                                                   i \in \Set{1, \dots, \dbk} \\
                                                                   j \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                                   k \in \Set{1, \dots, \nbk} \\
                                                                   t \in \Set{0, \dots, \cT - 1}
                                                                 \end{dcases}.
      \end{align*}
    \]

  最後總和上述推論得出 :math:`\xt(t)` 對於 memory cell block activations 的微分近似結果：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\ybk{k}_i(t + 1)}{\xt_j(t)} & = \cancelto{\aptr 0}{\dv{\yog_k(t + 1)}{\xt_j(t)}} \cdot h\qty(\sbk{k}_i(t + 1)) + \yog_k(t + 1) \cdot \dv{h\qty(\sbk{k}_i(t + 1))}{\sbk{k}_i(t + 1)} \cdot \cancelto{\aptr 0}{\dv{\sbk{k}_i(t + 1)}{\xt_j(t)}} \\
                                        & \aptr 0 \qqtext{where} \begin{dcases}
                                                                   i \in \Set{1, \dots, \dbk} \\
                                                                   j \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                                   k \in \Set{1, \dots, \nbk} \\
                                                                   t \in \Set{0, \dots, \cT - 1}
                                                                 \end{dcases}.
      \end{align*}
    \]

:math:`\Wout` 相對於誤差的微分
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

因為架構的修改並沒有影響輸出的\ **計算方式**，因此微分求法與式子 :math:`\eqref{1}` 相同。

:math:`\Wog` 相對於誤差的微分近似值
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

由於架構中新增了 forget gate units，因此影響了 :math:`\xt(t)` 的結構，導致式子 :math:`\eqref{2}` 與式子 :math:`\eqref{12}` 的公式相同，只是 :math:`q` 的範圍不同。

.. math::
  :nowrap:

  \[
    \begin{align*}
      & \dv{\cL\qty(\y(t + 1) - \yh(t + 1))}{\Wog_{p, q}} \aptr \qty(\sum_{j = 1}^\dbk \qty[\sum_{i = 1}^\dout \qty(\y_i(t + 1) - \yh_i(t + 1)) \cdot {f^\opout}'\qty(\zout_i(t + 1)) \cdot \Wout_{i, \din + (p - 1) \times \dbk + j}] \cdot h\qty(\sbk{p}_j(t + 1))) \cdot {f^\opog}'\qty(\zog_p(t + 1)) \cdot \xt_q(t) \\
      & \qqtext{where} \begin{dcases}
                         p \in \Set{1, \dots, \nbk} \\
                         q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                         t \in \Set{0, \dots, \cT - 1}
                       \end{dcases}.
    \end{align*}
    \tag{12}\label{12}
  \]

.. dropdown:: 推導式子 :math:`\eqref{12}`

  根據式子 :math:`\eqref{10}`，在丟棄部份微分後 :math:`\Wog` 將\ **無法**\透過 forget/input gate units 取得資訊：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\yfg_k(t + 1)}{\Wog_{p, q}} & = \dv{\yfg_k(t + 1)}{\zfg_k(t + 1)} \cdot \sum_{j = 1}^{\din + \nbk \times (3 + \dbk)} \qty[\cancelto{\aptr 0}{\dv{\zfg_k(t + 1)}{\xt_j(t)}} \cdot \dv{\xt_j(t)}{\Wog_{p, q}}] \\
                                        & \aptr 0 \qqtext{where} \begin{dcases}
                                                                   k \in \Set{1, \dots, \nbk} \\
                                                                   p \in \Set{1, \dots, \nbk} \\
                                                                   q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                                   t \in \Set{0, \dots, \cT - 1}
                                                                 \end{dcases}. \\
        \dv{\yig_k(t + 1)}{\Wog_{p, q}} & = \dv{\yig_k(t + 1)}{\zig_k(t + 1)} \cdot \sum_{j = 1}^{\din + \nbk \times (3 + \dbk)} \qty[\cancelto{\aptr 0}{\dv{\zig_k(t + 1)}{\xt_j(t)}} \cdot \dv{\xt_j(t)}{\Wog_{p, q}}] \\
                                        & \aptr 0 \qqtext{where} \begin{dcases}
                                                                   k \in \Set{1, \dots, \nbk} \\
                                                                   p \in \Set{1, \dots, \nbk} \\
                                                                   q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                                   t \in \Set{0, \dots, \cT - 1}
                                                                 \end{dcases}.
      \end{align*}
    \]

  結合式子 :math:`\eqref{10}` 與上式我們可以得出 :math:`\Wog` 相對於 memory cell internal states 的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\sbk{k}_i(t + 1)}{\Wog_{p, q}} & = \cancelto{\aptr 0}{\dv{\yfg_k(t + 1)}{\Wog_{p, q}}} \cdot \sbk{k}_i(t) + \yfg_k(t + 1) \cdot \dv{\sbk{k}_i(t)}{\Wog_{p, q}} + \cancelto{\aptr 0}{\dv{\yig_k(t + 1)}{\Wog_{p, q}}} \cdot g\qty(\zbk{k}_i(t + 1)) + \yig_k(t + 1) \cdot \dv{g\qty(\zbk{k}_i(t + 1))}{\zbk{k}_i(t + 1)} \cdot \sum_{j = 1}^{\din + \nbk \times (3 + \dbk)} \qty[\cancelto{\aptr 0}{\dv{\zbk{k}_i(t + 1)}{\xt_j(t)}} \cdot \dv{\xt_j(t)}{\Wog_{p, q}}] \\
                                           & \aptr \yfg_k(t + 1) \cdot \dv{\sbk{k}_i(t)}{\Wog_{p, q}} \\
                                           & \aptr \qty[\prod_{t^\star = t}^{t + 1} \yfg_k(t^\star)] \cdot \dv{\sbk{k}_i(t - 1)}{\Wog_{p, q}} \\
                                           & \vdots \\
                                           & \aptr \qty[\prod_{t^\star = 1}^{t + 1} \yfg_k(t^\star)] \cdot \cancelto{0}{\dv{\sbk{k}_i(0)}{\Wog_{p, q}}} \\
                                           & = 0 \qqtext{where} \begin{dcases}
                                                                  i \in \Set{1, \dots, \dbk} \\
                                                                  k \in \Set{1, \dots, \nbk} \\
                                                                  p \in \Set{1, \dots, \nbk} \\
                                                                  q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                                  t \in \Set{0, \dots, \cT - 1}
                                                                \end{dcases}.
      \end{align*}
    \]

  上式告訴我們，在丟棄部份微分後 :math:`\Wog` 將\ **無法**\透過 memory cell internal states 取得資訊。
  直覺上 :math:`\Wog` 唯一能夠取得資訊的管道只有 output gate units。
  所以接下來我們推導 :math:`\Wog` 相對於 output gate units 的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\yog_k(t + 1)}{\Wog_{p, q}} & = \dv{\yog_k(t + 1)}{\zog_k(t + 1)} \cdot \dv{\zog_k(t + 1)}{\Wog_{p, q}} \\
                                        & = {f^\opog}'\qty(\zog_k(t + 1)) \cdot \qty[\delta_{k, p} \cdot \xt_q(t) + \sum_{j = 1}^{\din + \nbk \times (3 + \dbk)} \qty[\Wog_{k, j} \cdot \dv{\xt_j(t)}{\Wog_{p, q}}]] \\
                                        & \qqtext{where} \begin{dcases}
                                                           k \in \Set{1, \dots, \nbk} \\
                                                           p \in \Set{1, \dots, \nbk} \\
                                                           q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                           t \in \Set{0, \dots, \cT - 1}
                                                         \end{dcases}.
      \end{align*}
    \]

  可以發現 :math:`\Wog` 對於 output gate units 的全微分會有 BPTT 的問題，因此原始 LSTM 論文中提出額外丟棄 output gate units 的部份微分，結果如下：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\yog_k(t + 1)}{\Wog_{p, q}} & = {f^\opog}'\qty(\zog_k(t + 1)) \cdot \qty[\delta_{k, p} \cdot \xt_q(t) + \cancelto{\aptr 0}{\sum_{j = 1}^{\din + \nbk \times (3 + \dbk)} \qty[\Wog_{k, j} \cdot \dv{\xt_j(t)}{\Wog_{p, q}}]}] \\
                                        & \aptr {f^\opog}'\qty(\zog_k(t + 1)) \cdot \delta_{k, p} \cdot \xt_q(t) \\
                                        & \qqtext{where} \begin{dcases}
                                                           k \in \Set{1, \dots, \nbk} \\
                                                           p \in \Set{1, \dots, \nbk} \\
                                                           q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                           t \in \Set{0, \dots, \cT - 1}
                                                         \end{dcases}.
      \end{align*}
    \]

  使用前述推導結果可以幫助我們推得 :math:`\Wog` 相對於 memory cell activation blocks 的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\ybk{k}_i(t + 1)}{\Wog_{p, q}} & = \dv{\yog_k(t + 1)}{\Wog_{p, q}} \cdot h\qty(\sbk{k}_i(t + 1)) + \yog_k(t + 1) \cdot \dv{h\qty(\sbk{k}_i(t + 1))}{\sbk{k}_i(t + 1)} \cdot \cancelto{\aptr 0}{\dv{\sbk{k}_i(t + 1)}{\Wog_{p, q}}} \\
                                           & \aptr {f^\opog}'\qty(\zog_k(t + 1)) \cdot \delta_{k, p} \cdot \xt_q(t) \cdot h\qty(\sbk{k}_i(t + 1)) \\
                                           & \qqtext{where} \begin{dcases}
                                                              i \in \Set{1, \dots, \dbk} \\
                                                              k \in \Set{1, \dots, \nbk} \\
                                                              p \in \Set{1, \dots, \nbk} \\
                                                              q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                              t \in \Set{0, \dots, \cT - 1}
                                                            \end{dcases}.
      \end{align*}
    \]

  最後我們推得 :math:`\Wog` 相對於誤差的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        & \dv{\cL\qty(\y(t + 1) - \yh(t + 1))}{\Wog_{p, q}} \\
        & = \sum_{i = 1}^\dout \dv{\frac{1}{2} \qty(\y_i(t + 1) - \yh_i(t + 1))^2}{\Wog_{p, q}} \\
        & = \sum_{i = 1}^\dout \qty[\dv{\frac{1}{2} \qty(\y_i(t + 1) - \yh_i(t + 1))^2}{\y_i(t + 1)} \cdot \dv{\y_i(t + 1)}{\zout_i(t + 1)} \cdot \sum_{j = 1}^{\din + \nbk \times \dbk} \qty[\dv{\zout_i(t + 1)}{\xout_j(t + 1)} \cdot \cancelto{\aptr 0}{\dv{\xout_j(t + 1)}{\Wog_{p, q}}}]] \\
        & \aptr \sum_{i = 1}^\dout \qty[\qty(\y_i(t + 1) - \yh_i(t + 1)) \cdot {f^\opout}'\qty(\zout_i(t + 1)) \cdot \sum_{k = 1}^\nbk \sum_{j = 1}^\dbk \qty[\Wout_{i, \din + (k - 1) \times \dbk + j} \cdot \dv{\ybk{k}_j(t + 1)}{\Wog_{p, q}}]] \\
        & \aptr \sum_{i = 1}^\dout \qty[\qty(\y_i(t + 1) - \yh_i(t + 1)) \cdot {f^\opout}'\qty(\zout_i(t + 1)) \cdot \sum_{k = 1}^\nbk \sum_{j = 1}^\dbk \qty[\Wout_{i, \din + (k - 1) \times \dbk + j} \cdot {f^\opog}'\qty(\zog_k(t + 1)) \cdot \delta_{k, p} \cdot \xt_q(t) \cdot h\qty(\sbk{k}_j(t + 1))]] \\
        & = \sum_{i = 1}^\dout \qty[\qty(\y_i(t + 1) - \yh_i(t + 1)) \cdot {f^\opout}'\qty(\zout_i(t + 1)) \cdot \sum_{j = 1}^\dbk \qty[\Wout_{i, \din + (p - 1) \times \dbk + j} \cdot {f^\opog}'\qty(\zog_p(t + 1)) \cdot \xt_q(t) \cdot h\qty(\sbk{p}_j(t + 1))]] \\
        & = \qty(\sum_{i = 1}^\dout \qty[\qty(\y_i(t + 1) - \yh_i(t + 1)) \cdot {f^\opout}'\qty(\zout_i(t + 1)) \cdot \sum_{j = 1}^\dbk \qty[\Wout_{i, \din + (p - 1) \times \dbk + j} \cdot h\qty(\sbk{p}_j(t + 1))]]) \cdot {f^\opog}'\qty(\zog_p(t + 1)) \cdot \xt_q(t) \\
        & = \qty(\sum_{j = 1}^\dbk \qty[\sum_{i = 1}^\dout \qty(\y_i(t + 1) - \yh_i(t + 1)) \cdot {f^\opout}'\qty(\zout_i(t + 1)) \cdot \Wout_{i, \din + (p - 1) \times \dbk + j}] \cdot h\qty(\sbk{p}_j(t + 1))) \cdot {f^\opog}'\qty(\zog_p(t + 1)) \cdot \xt_q(t) \\
        & \qqtext{where} \begin{dcases}
                           p \in \Set{1, \dots, \nbk} \\
                           q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                           t \in \Set{0, \dots, \cT - 1}
                         \end{dcases}.
      \end{align*}
    \]

:math:`\Wfg` 相對於誤差的微分近似值
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::
  :nowrap:

  \[
    \begin{align*}
      & \dv{\cL\qty(\y(t + 1) - \yh(t + 1))}{\Wfg_{p, q}} \aptr \qty(\sum_{i = 1}^\dout \qty[\qty(\y_i(t + 1) - \yh_i(t + 1)) \cdot {f^\opout}'\qty(\zout_i(t + 1)) \cdot \sum_{j = 1}^\dbk \qty[\Wout_{i, \din + (p - 1) \times \dbk + j} \cdot h'\qty(\sbk{p}_j(t + 1)) \cdot \dv{\sbk{p}_j(t + 1)}{\Wfg_{p, q}}]]) \cdot \yog_p(t + 1) \\
      & \qqtext{where} \begin{dcases}
                         p \in \Set{1, \dots, \nbk} \\
                         q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                         t \in \Set{0, \dots, \cT - 1}
                       \end{dcases}. \\
      & \dv{\sbk{k}_j(t + 1)}{\Wfg_{p, q}} \aptr \delta_{k, p} \cdot \qty[{f^\opfg}'\qty(\zfg_p(t + 1)) \cdot \xt_q(t) \cdot \sbk{p}_j(t) + \yfg_p(t + 1) \cdot \dv{\sbk{p}_j(t)}{\Wfg_{p, q}}] \\
      & \qqtext{where} \begin{dcases}
                         j \in \Set{1, \dots, \dbk} \\
                         k \in \Set{1, \dots, \nbk} \\
                         p \in \Set{1, \dots, \nbk} \\
                         q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                         t \in \Set{0, \dots, \cT - 1}
                       \end{dcases}.
    \end{align*}
    \tag{13}\label{13}
  \]

.. dropdown:: 推導式子 :math:`\eqref{13}`

  根據式子 :math:`\eqref{10}` 我們可以求得 :math:`\Wfg` 相對於 input/output gate units 的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\yig_k(t + 1)}{\Wfg_{p, q}} & = \dv{\yig_k(t + 1)}{\zig_k(t + 1)} \cdot \sum_{j = 1}^{\din + \nbk \times (3 + \dbk)} \qty[\cancelto{\aptr 0}{\dv{\zig_k(t + 1)}{\xt_j(t)}} \cdot \dv{\xt_j(t)}{\Wfg_{p, q}}] \\
                                        & \aptr 0 \qqtext{where} \begin{dcases}
                                                                   k \in \Set{1, \dots, \nbk} \\
                                                                   p \in \Set{1, \dots, \nbk} \\
                                                                   q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                                   t \in \Set{0, \dots, \cT - 1}
                                                                 \end{dcases}. \\
        \dv{\yog_k(t + 1)}{\Wfg_{p, q}} & = \dv{\yog_k(t + 1)}{\zog_k(t + 1)} \cdot \sum_{j = 1}^{\din + \nbk \times (3 + \dbk)} \qty[\cancelto{\aptr 0}{\dv{\zog_k(t + 1)}{\xt_j(t)}} \cdot \dv{\xt_j(t)}{\Wfg_{p, q}}] \\
                                        & \aptr 0 \qqtext{where} \begin{dcases}
                                                                   k \in \Set{1, \dots, \nbk} \\
                                                                   p \in \Set{1, \dots, \nbk} \\
                                                                   q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                                   t \in \Set{0, \dots, \cT - 1}
                                                                 \end{dcases}.
      \end{align*}
    \]

  在丟棄部份微分後 :math:`\Wfg` 將\ **無法**\透過 input/output gate units 取得資訊。
  直覺上我們認為 :math:`\Wfg` 應該可以透過 forget gate units 取得資訊。
  所以接下來我們推導 :math:`\Wfg` 相對於 forget gate units 的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\yfg_k(t + 1)}{\Wfg_{p, q}} & = \dv{\yfg_k(t + 1)}{\zfg_k(t + 1)} \cdot \dv{\zfg_k(t + 1)}{\Wfg_{p, q}} \\
                                        & = {f^\opfg}'\qty(\zfg_k(t + 1)) \cdot \qty[\delta_{k, p} \cdot \xt_q(t) + \sum_{j = 1}^{\din + \nbk \times (3 + \dbk)} \qty[\Wfg_{k, j} \cdot \dv{\xt_j(t)}{\Wfg_{p, q}}]] \\
                                        & \qqtext{where} \begin{dcases}
                                                           k \in \Set{1, \dots, \nbk} \\
                                                           p \in \Set{1, \dots, \nbk} \\
                                                           q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                           t \in \Set{0, \dots, \cT - 1}
                                                         \end{dcases}.
      \end{align*}
    \]

  可以發現 :math:`\Wfg` 對於 forget gate units 的全微分會有 BPTT 的問題，因此原始 LSTM 論文中提出額外丟棄 forget gate units 的部份微分，結果如下：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\yfg_k(t + 1)}{\Wfg_{p, q}} & = {f^\opfg}'\qty(\zfg_k(t + 1)) \cdot \qty[\delta_{k, p} \cdot \xt_q(t) + \cancelto{\aptr 0}{\sum_{j = 1}^{\din + \nbk \times (3 + \dbk)} \qty[\Wfg_{k, j} \cdot \dv{\xt_j(t)}{\Wfg_{p, q}}]}] \\
                                        & \aptr {f^\opfg}'\qty(\zfg_k(t + 1)) \cdot \delta_{k, p} \cdot \xt_q(t) \\
                                        & \qqtext{where} \begin{dcases}
                                                           k \in \Set{1, \dots, \nbk} \\
                                                           p \in \Set{1, \dots, \nbk} \\
                                                           q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                           t \in \Set{0, \dots, \cT - 1}
                                                         \end{dcases}.
      \end{align*}
    \]

  結合式子 :math:`\eqref{10}` 與前面的推導，我們可以得出 :math:`\Wfg` 相對於 memory cell internal states 的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\sbk{k}_i(t + 1)}{\Wfg_{p, q}} & = \dv{\yfg_k(t + 1)}{\Wfg_{p, q}} \cdot \sbk{k}_i(t) + \yfg_k(t + 1) \cdot \dv{\sbk{k}_i(t)}{\Wfg_{p, q}} + \cancelto{\aptr 0}{\dv{\yig_k(t + 1)}{\Wfg_{p, q}}} \cdot g\qty(\zbk{k}_i(t + 1)) + \yig_k(t + 1) \cdot \dv{g\qty(\zbk{k}_i(t + 1))}{\zbk{k}_i(t + 1)} \cdot \sum_{j = 1}^{\din + \nbk \times (3 + \dbk)} \qty[\cancelto{\aptr 0}{\dv{\zbk{k}_i(t + 1)}{\xt_j(t)}} \cdot \dv{\xt_j(t)}{\Wfg_{p, q}}] \\
                                           & \aptr {f^\opfg}'\qty(\zfg_k(t + 1)) \cdot \delta_{k, p} \cdot \xt_q(t) \cdot \sbk{k}_i(t) + \yfg_k(t + 1) \cdot \dv{\sbk{k}_i(t)}{\Wfg_{p, q}} \\
                                           & \qqtext{where} \begin{dcases}
                                                              i \in \Set{1, \dots, \dbk} \\
                                                              k \in \Set{1, \dots, \nbk} \\
                                                              p \in \Set{1, \dots, \nbk} \\
                                                              q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                              t \in \Set{0, \dots, \cT - 1}
                                                            \end{dcases}.
      \end{align*}
    \]

  觀察上式可以發現當 :math:`k \neq p` 時微分結果為 :math:`0`：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\sbk{k}_i(t + 1)}{\Wfg_{p, q}} & \aptr {f^\opfg}'\qty(\zfg_k(t + 1)) \cdot \cancelto{0}{\delta_{k, p}} \cdot \xt_q(t) \cdot \sbk{k}_i(t) + \yfg_k(t + 1) \cdot \dv{\sbk{k}_i(t)}{\Wfg_{p, q}} \\
                                           & = \yfg_k(t + 1) \cdot \dv{\sbk{k}_i(t)}{\Wfg_{p, q}} \\
                                           & \aptr \qty[\prod_{t^\star = t}^{t + 1} \yfg_k(t^\star)] \cdot \dv{\sbk{k}_i(t - 1)}{\Wfg_{p, q}} \\
                                           & \vdots \\
                                           & \aptr \qty[\prod_{t^\star = 1}^{t + 1} \yfg_k(t^\star)] \cdot \cancelto{0}{\dv{\sbk{k}_i(0)}{\Wfg_{p, q}}} \\
                                           & = 0 \\
                                           & \qqtext{where} \begin{dcases}
                                                              i \in \Set{1, \dots, \dbk} \\
                                                              k \in \Set{1, \dots, \nbk} \\
                                                              p \in \Set{1, \dots, \nbk} \\
                                                              q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                              t \in \Set{0, \dots, \cT - 1}
                                                            \end{dcases}.
      \end{align*}
    \]

  因此我們將 :math:`\Wfg` 相對於 memory cell internal states 的微分近似值改寫如下：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\sbk{k}_i(t + 1)}{\Wfg_{p, q}} & \aptr \delta_{k, p} \cdot \dv{\sbk{p}_i(t + 1)}{\Wfg_{p, q}} \\
                                           & \aptr \delta_{k, p} \cdot \qty[{f^\opfg}'\qty(\zfg_p(t + 1)) \cdot \xt_q(t) \cdot \sbk{k}_i(t) + \yfg_p(t + 1) \cdot \dv{\sbk{p}_i(t)}{\Wfg_{p, q}}] \\
                                           & \qqtext{where} \begin{dcases}
                                                              i \in \Set{1, \dots, \dbk} \\
                                                              k \in \Set{1, \dots, \nbk} \\
                                                              p \in \Set{1, \dots, \nbk} \\
                                                              q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                              t \in \Set{0, \dots, \cT - 1}
                                                            \end{dcases}.
      \end{align*}
    \]

  .. note::

    上式就是論文的 3.12 式。

  可以發現 :math:`\Wfg` 透過 memory cell internal states 得到的資訊其實都是來自於過去微分近似值的累加結果。
  實際上在執行參數更新演算法時只需要儲存過去累加而得的結果再結合當前計算結果，就可以得到最新的參數更新方向。
  使用前述推導結果我們可以得到 :math:`\Wfg` 相對於 memory cell activation blocks 的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\ybk{k}_i(t + 1)}{\Wfg_{p, q}} & = \cancelto{\aptr 0}{\dv{\yog_k(t + 1)}{\Wfg_{p, q}}} \cdot h\qty(\sbk{k}_i(t + 1)) + \yog_k(t + 1) \cdot \dv{h\qty(\sbk{k}_i(t + 1))}{\sbk{k}_i(t + 1)} \cdot \dv{\sbk{k}_i(t + 1)}{\Wfg_{p, q}} \\
                                           & \aptr \yog_k(t + 1) \cdot h'\qty(\sbk{k}_i(t + 1)) \cdot \delta_{k, p} \cdot \dv{\sbk{p}_i(t + 1)}{\Wfg_{p, q}} \\
                                           & \qqtext{where} \begin{dcases}
                                                              i \in \Set{1, \dots, \dbk} \\
                                                              k \in \Set{1, \dots, \nbk} \\
                                                              p \in \Set{1, \dots, \nbk} \\
                                                              q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                              t \in \Set{0, \dots, \cT - 1}
                                                            \end{dcases}.
      \end{align*}
    \]

  同前述結論，只需要儲存過去計算而得的結果，最後乘上一些當前的計算狀態，就可以得到最新的參數更新方向。
  最後我們推得 :math:`\Wfg` 相對於誤差的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        & \dv{\cL\qty(\y(t + 1) - \yh(t + 1))}{\Wfg_{p, q}} \\
        & = \sum_{i = 1}^\dout \dv{\frac{1}{2} \qty(\y_i(t + 1) - \yh_i(t + 1))^2}{\Wfg_{p, q}} \\
        & = \sum_{i = 1}^\dout \qty[\dv{\frac{1}{2} \qty(\y_i(t + 1) - \yh_i(t + 1))^2}{\y_i(t + 1)} \cdot \dv{\y_i(t + 1)}{\zout_i(t + 1)} \cdot \sum_{j = 1}^{\din + \nbk \times \dbk} \qty[\dv{\zout_i(t + 1)}{\xout_j(t + 1)} \cdot \cancelto{\aptr 0}{\dv{\xout_j(t + 1)}{\Wfg_{p, q}}}]] \\
        & \aptr \sum_{i = 1}^\dout \qty[\qty(\y_i(t + 1) - \yh_i(t + 1)) \cdot {f^\opout}'\qty(\zout_i(t + 1)) \cdot \sum_{k = 1}^\nbk \sum_{j = 1}^\dbk \qty[\Wout_{i, \din + (k - 1) \times \dbk + j} \cdot \dv{\ybk{k}_j(t + 1)}{\Wfg_{p, q}}]] \\
        & \aptr \sum_{i = 1}^\dout \qty[\qty(\y_i(t + 1) - \yh_i(t + 1)) \cdot {f^\opout}'\qty(\zout_i(t + 1)) \cdot \sum_{k = 1}^\nbk \sum_{j = 1}^\dbk \qty[\Wout_{i, \din + (k - 1) \times \dbk + j} \cdot \yog_k(t + 1) \cdot h'\qty(\sbk{k}_j(t + 1)) \cdot \delta_{k, p} \cdot \dv{\sbk{p}_j(t + 1)}{\Wfg_{p, q}}]] \\
        & = \sum_{i = 1}^\dout \qty[\qty(\y_i(t + 1) - \yh_i(t + 1)) \cdot {f^\opout}'\qty(\zout_i(t + 1)) \cdot \sum_{j = 1}^\dbk \qty[\Wout_{i, \din + (p - 1) \times \dbk + j} \cdot \yog_p(t + 1) \cdot h'\qty(\sbk{p}_j(t + 1)) \cdot \dv{\sbk{p}_j(t + 1)}{\Wfg_{p, q}}]] \\
        & = \qty(\sum_{i = 1}^\dout \qty[\qty(\y_i(t + 1) - \yh_i(t + 1)) \cdot {f^\opout}'\qty(\zout_i(t + 1)) \cdot \sum_{j = 1}^\dbk \qty[\Wout_{i, \din + (p - 1) \times \dbk + j} \cdot h'\qty(\sbk{p}_j(t + 1)) \cdot \dv{\sbk{p}_j(t + 1)}{\Wfg_{p, q}}]]) \cdot \yog_p(t + 1) \\
        & \qqtext{where} \begin{dcases}
                           p \in \Set{1, \dots, \nbk} \\
                           q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                           t \in \Set{0, \dots, \cT - 1}
                         \end{dcases}.
      \end{align*}
    \]

:math:`\Wig` 相對於誤差的微分近似值
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

由於架構中新增了 forget gate units，因此影響了 memory cell internal states 的結構，所以推導結果 :math:`\eqref{14}` 與原始 LSTM 論文推得的式子 :math:`\eqref{3}` 不同。

.. math::
  :nowrap:

  \[
    \begin{align*}
      & \dv{\cL\qty(\y(t + 1) - \yh(t + 1))}{\Wig_{p, q}} \aptr \qty(\sum_{i = 1}^\dout \qty[\qty(\y_i(t + 1) - \yh_i(t + 1)) \cdot {f^\opout}'\qty(\zout_i(t + 1)) \cdot \sum_{j = 1}^\dbk \qty[\Wout_{i, \din + (p - 1) \times \dbk + j} \cdot h'\qty(\sbk{p}_j(t + 1)) \cdot \dv{\sbk{p}_j(t + 1)}{\Wig_{p, q}}]]) \cdot \yog_p(t + 1) \\
      & \qqtext{where} \begin{dcases}
                         p \in \Set{1, \dots, \nbk} \\
                         q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                         t \in \Set{0, \dots, \cT - 1}
                       \end{dcases}. \\
      & \dv{\sbk{k}_j(t + 1)}{\Wig_{p, q}} \aptr \delta_{k, p} \cdot \qty[\yfg_p(t + 1) \cdot \dv{\sbk{p}_j(t)}{\Wig_{p, q}} + {f^\opig}'\qty(\zig_p(t + 1)) \cdot \xt_q(t) \cdot g\qty(\zbk{p}_j(t + 1))] \\
      & \qqtext{where} \begin{dcases}
                         j \in \Set{1, \dots, \dbk} \\
                         k \in \Set{1, \dots, \nbk} \\
                         p \in \Set{1, \dots, \nbk} \\
                         q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                         t \in \Set{0, \dots, \cT - 1}
                       \end{dcases}.
    \end{align*}
    \tag{14}\label{14}
  \]

.. dropdown:: 推導式子 :math:`\eqref{14}`

  根據式子 :math:`\eqref{10}` 我們可以求得 :math:`\Wig` 相對於 forget/output gate units 的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\yfg_k(t + 1)}{\Wig_{p, q}} & = \dv{\yfg_k(t + 1)}{\zfg_k(t + 1)} \cdot \sum_{j = 1}^{\din + \nbk \times (3 + \dbk)} \qty[\cancelto{\aptr 0}{\dv{\zfg_k(t + 1)}{\xt_j(t)}} \cdot \dv{\xt_j(t)}{\Wig_{p, q}}] \\
                                        & \aptr 0 \qqtext{where} \begin{dcases}
                                                                   k \in \Set{1, \dots, \nbk} \\
                                                                   p \in \Set{1, \dots, \nbk} \\
                                                                   q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                                   t \in \Set{0, \dots, \cT - 1}
                                                                 \end{dcases}. \\
        \dv{\yog_k(t + 1)}{\Wig_{p, q}} & = \dv{\yog_k(t + 1)}{\zog_k(t + 1)} \cdot \sum_{j = 1}^{\din + \nbk \times (3 + \dbk)} \qty[\cancelto{\aptr 0}{\dv{\zog_k(t + 1)}{\xt_j(t)}} \cdot \dv{\xt_j(t)}{\Wig_{p, q}}] \\
                                        & \aptr 0 \qqtext{where} \begin{dcases}
                                                                   k \in \Set{1, \dots, \nbk} \\
                                                                   p \in \Set{1, \dots, \nbk} \\
                                                                   q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                                   t \in \Set{0, \dots, \cT - 1}
                                                                 \end{dcases}.
      \end{align*}
    \]

  在丟棄部份微分後 :math:`\Wig` 將\ **無法**\透過 forget/output gate units 取得資訊。
  直覺上我們認為 :math:`\Wig` 應該可以透過 input gate units 取得資訊。
  所以接下來我們推導 :math:`\Wig` 相對於 input gate units 的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\yig_k(t + 1)}{\Wig_{p, q}} & = \dv{\yig_k(t + 1)}{\zig_k(t + 1)} \cdot \dv{\zig_k(t + 1)}{\Wig_{p, q}} \\
                                        & = {f^\opig}'\qty(\zig_k(t + 1)) \cdot \qty[\delta_{k, p} \cdot \xt_q(t) + \sum_{j = 1}^{\din + \nbk \times (3 + \dbk)} \qty[\Wig_{k, j} \cdot \dv{\xt_j(t)}{\Wig_{p, q}}]] \\
                                        & \qqtext{where} \begin{dcases}
                                                           k \in \Set{1, \dots, \nbk} \\
                                                           p \in \Set{1, \dots, \nbk} \\
                                                           q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                           t \in \Set{0, \dots, \cT - 1}
                                                         \end{dcases}.
      \end{align*}
    \]

  可以發現 :math:`\Wig` 對於 input gate units 的全微分會有 BPTT 的問題，因此原始 LSTM 論文中提出額外丟棄 input gate units 的部份微分，結果如下：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\yig_k(t + 1)}{\Wig_{p, q}} & = {f^\opig}'\qty(\zig_k(t + 1)) \cdot \qty[\delta_{k, p} \cdot \xt_q(t) + \cancelto{\aptr 0}{\sum_{j = 1}^{\din + \nbk \times (3 + \dbk)} \qty[\Wig_{k, j} \cdot \dv{\xt_j(t)}{\Wig_{p, q}}]}] \\
                                        & \aptr {f^\opig}'\qty(\zig_k(t + 1)) \cdot \delta_{k, p} \cdot \xt_q(t) \\
                                        & \qqtext{where} \begin{dcases}
                                                           k \in \Set{1, \dots, \nbk} \\
                                                           p \in \Set{1, \dots, \nbk} \\
                                                           q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                           t \in \Set{0, \dots, \cT - 1}
                                                         \end{dcases}.
      \end{align*}
    \]

  結合式子 :math:`\eqref{10}` 與前面的推導，我們可以得出 :math:`\Wig` 相對於 memory cell internal states 的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\sbk{k}_i(t + 1)}{\Wig_{p, q}} & = \cancelto{\aptr 0}{\dv{\yfg_k(t + 1)}{\Wig_{p, q}}} \cdot \sbk{k}_i(t) + \yfg_k(t + 1) \cdot \dv{\sbk{k}_i(t)}{\Wig_{p, q}} + \dv{\yig_k(t + 1)}{\Wig_{p, q}} \cdot g\qty(\zbk{k}_i(t + 1)) + \yig_k(t + 1) \cdot \dv{g\qty(\zbk{k}_i(t + 1))}{\zbk{k}_i(t + 1)} \cdot \sum_{j = 1}^{\din + \nbk \times (3 + \dbk)} \qty[\cancelto{\aptr 0}{\dv{\zbk{k}_i(t + 1)}{\xt_j(t)}} \cdot \dv{\xt_j(t)}{\Wig_{p, q}}] \\
                                           & \aptr \yfg_k(t + 1) \cdot \dv{\sbk{k}_i(t)}{\Wig_{p, q}} + {f^\opig}'\qty(\zig_k(t + 1)) \cdot \delta_{k, p} \cdot \xt_q(t) \cdot g\qty(\zbk{k}_i(t + 1)) \\
                                           & \qqtext{where} \begin{dcases}
                                                              i \in \Set{1, \dots, \dbk} \\
                                                              k \in \Set{1, \dots, \nbk} \\
                                                              p \in \Set{1, \dots, \nbk} \\
                                                              q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                              t \in \Set{0, \dots, \cT - 1}
                                                            \end{dcases}.
      \end{align*}
    \]

  觀察上式可以發現當 :math:`k \neq p` 時微分結果為 :math:`0`：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\sbk{k}_i(t + 1)}{\Wig_{p, q}} & \aptr \yfg_k(t + 1) \cdot \dv{\sbk{k}_i(t)}{\Wig_{p, q}} + {f^\opig}'\qty(\zig_k(t + 1)) \cdot \cancelto{0}{\delta_{k, p}} \cdot \xt_q(t) \cdot g\qty(\zbk{k}_i(t + 1)) \\
                                           & = \yfg_k(t + 1) \cdot \dv{\sbk{k}_i(t)}{\Wig_{p, q}} \\
                                           & \aptr \qty[\prod_{t^\star = t}^{t + 1} \yfg_k(t^\star)] \cdot \dv{\sbk{k}_i(t - 1)}{\Wig_{p, q}} \\
                                           & \vdots \\
                                           & \aptr \qty[\prod_{t^\star = 1}^{t + 1} \yfg_k(t^\star)] \cdot \cancelto{0}{\dv{\sbk{k}_i(0)}{\Wig_{p, q}}} \\
                                           & = 0 \\
                                           & \qqtext{where} \begin{dcases}
                                                              i \in \Set{1, \dots, \dbk} \\
                                                              k \in \Set{1, \dots, \nbk} \\
                                                              p \in \Set{1, \dots, \nbk} \\
                                                              q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                              t \in \Set{0, \dots, \cT - 1}
                                                            \end{dcases}.
      \end{align*}
    \]

  因此我們將 :math:`\Wig` 相對於 memory cell internal states 的微分近似值改寫如下：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\sbk{k}_i(t + 1)}{\Wig_{p, q}} & \aptr \delta_{k, p} \cdot \dv{\sbk{p}_i(t + 1)}{\Wig_{p, q}} \\
                                           & \aptr \delta_{k, p} \cdot \qty[\yfg_p(t + 1) \cdot \dv{\sbk{p}_i(t)}{\Wig_{p, q}} + {f^\opig}'\qty(\zig_p(t + 1)) \cdot \xt_q(t) \cdot g\qty(\zbk{p}_i(t + 1))] \\
                                           & \qqtext{where} \begin{dcases}
                                                              i \in \Set{1, \dots, \dbk} \\
                                                              k \in \Set{1, \dots, \nbk} \\
                                                              p \in \Set{1, \dots, \nbk} \\
                                                              q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                              t \in \Set{0, \dots, \cT - 1}
                                                            \end{dcases}.
      \end{align*}
    \]

  .. note::

    上式就是論文的 3.11 式。

  可以發現 :math:`\Wig` 透過 memory cell internal states 得到的資訊其實都是來自於過去微分近似值的累加結果。
  實際上在執行參數更新演算法時只需要儲存過去累加而得的結果再結合當前計算結果，就可以得到最新的參數更新方向。
  使用前述推導結果我們可以得到 :math:`\Wig` 相對於 memory cell activation blocks 的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\ybk{k}_i(t + 1)}{\Wig_{p, q}} & = \cancelto{\aptr 0}{\dv{\yog_k(t + 1)}{\Wig_{p, q}}} \cdot h\qty(\sbk{k}_i(t + 1)) + \yog_k(t + 1) \cdot \dv{h\qty(\sbk{k}_i(t + 1))}{\sbk{k}_i(t + 1)} \cdot \dv{\sbk{k}_i(t + 1)}{\Wig_{p, q}} \\
                                           & \aptr \yog_k(t + 1) \cdot h'\qty(\sbk{k}_i(t + 1)) \cdot \delta_{k, p} \cdot \dv{\sbk{p}_i(t + 1)}{\Wig_{p, q}} \\
                                           & \qqtext{where} \begin{dcases}
                                                              i \in \Set{1, \dots, \dbk} \\
                                                              k \in \Set{1, \dots, \nbk} \\
                                                              p \in \Set{1, \dots, \nbk} \\
                                                              q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                              t \in \Set{0, \dots, \cT - 1}
                                                            \end{dcases}.
      \end{align*}
    \]

  同前述結論，只需要儲存過去計算而得的結果，最後乘上一些當前的計算狀態，就可以得到最新的參數更新方向。
  最後我們推得 :math:`\Wig` 相對於誤差的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        & \dv{\cL\qty(\y(t + 1) - \yh(t + 1))}{\Wig_{p, q}} \\
        & = \sum_{i = 1}^\dout \dv{\frac{1}{2} \qty(\y_i(t + 1) - \yh_i(t + 1))^2}{\Wig_{p, q}} \\
        & = \sum_{i = 1}^\dout \qty[\dv{\frac{1}{2} \qty(\y_i(t + 1) - \yh_i(t + 1))^2}{\y_i(t + 1)} \cdot \dv{\y_i(t + 1)}{\zout_i(t + 1)} \cdot \sum_{j = 1}^{\din + \nbk \times \dbk} \qty[\dv{\zout_i(t + 1)}{\xout_j(t + 1)} \cdot \cancelto{\aptr 0}{\dv{\xout_j(t + 1)}{\Wig_{p, q}}}]] \\
        & \aptr \sum_{i = 1}^\dout \qty[\qty(\y_i(t + 1) - \yh_i(t + 1)) \cdot {f^\opout}'\qty(\zout_i(t + 1)) \cdot \sum_{k = 1}^\nbk \sum_{j = 1}^\dbk \qty[\Wout_{i, \din + (k - 1) \times \dbk + j} \cdot \dv{\ybk{k}_j(t + 1)}{\Wig_{p, q}}]] \\
        & \aptr \sum_{i = 1}^\dout \qty[\qty(\y_i(t + 1) - \yh_i(t + 1)) \cdot {f^\opout}'\qty(\zout_i(t + 1)) \cdot \sum_{k = 1}^\nbk \sum_{j = 1}^\dbk \qty[\Wout_{i, \din + (k - 1) \times \dbk + j} \cdot \yog_k(t + 1) \cdot h'\qty(\sbk{k}_j(t + 1)) \cdot \delta_{k, p} \cdot \dv{\sbk{p}_j(t + 1)}{\Wig_{p, q}}]] \\
        & = \sum_{i = 1}^\dout \qty[\qty(\y_i(t + 1) - \yh_i(t + 1)) \cdot {f^\opout}'\qty(\zout_i(t + 1)) \cdot \sum_{j = 1}^\dbk \qty[\Wout_{i, \din + (p - 1) \times \dbk + j} \cdot \yog_p(t + 1) \cdot h'\qty(\sbk{p}_j(t + 1)) \cdot \dv{\sbk{p}_j(t + 1)}{\Wig_{p, q}}]] \\
        & = \qty(\sum_{i = 1}^\dout \qty[\qty(\y_i(t + 1) - \yh_i(t + 1)) \cdot {f^\opout}'\qty(\zout_i(t + 1)) \cdot \sum_{j = 1}^\dbk \qty[\Wout_{i, \din + (p - 1) \times \dbk + j} \cdot h'\qty(\sbk{p}_j(t + 1)) \cdot \dv{\sbk{p}_j(t + 1)}{\Wig_{p, q}}]]) \cdot \yog_p(t + 1) \\
        & \qqtext{where} \begin{dcases}
                           p \in \Set{1, \dots, \nbk} \\
                           q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                           t \in \Set{0, \dots, \cT - 1}
                         \end{dcases}.
      \end{align*}
    \]

:math:`\Wbk{k}` 相對於誤差的微分近似值
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

由於架構中新增了 forget gate units，因此影響了 memory cell internal states 的結構，所以推導結果 :math:`\eqref{15}` 與原始 LSTM 論文推得的式子 :math:`\eqref{4}` 不同。

.. math::
  :nowrap:

  \[
    \begin{align*}
      & \dv{\cL\qty(\y(t + 1) - \yh(t + 1))}{\Wbk{k}_{p, q}} \aptr \qty[\sum_{i = 1}^\dout \qty(\y_i(t + 1) - \yh_i(t + 1)) \cdot {f^\opout}'\qty(\zout_i(t + 1)) \cdot \Wout_{i, \din + (k - 1) \times \dbk + p}] \cdot \yog_k(t + 1) \cdot h'\qty(\sbk{k}_p(t + 1)) \cdot \dv{\sbk{k}_p(t + 1)}{\Wbk{k}_{p, q}} \\
      & \qqtext{where} \begin{dcases}
                         k \in \Set{1, \dots, \nbk} \\
                         p \in \Set{1, \dots, \dbk} \\
                         q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                         t \in \Set{0, \dots, \cT - 1}
                       \end{dcases}. \\
      & \dv{\sbk{k^\star}_j(t + 1)}{\Wbk{k}_{p, q}} \aptr \delta_{k^\star, k} \cdot \delta_{j, p} \cdot \qty[\yfg_{k}(t + 1) \cdot \dv{\sbk{k}_p(t)}{\Wbk{k}_{p, q}} + \yig_{k}(t + 1) \cdot g'\qty(\zbk{k}_p(t + 1)) \cdot \xt_q(t)] \\
      & \qqtext{where} \begin{dcases}
                         j \in \Set{1, \dots, \dbk} \\
                         k \in \Set{1, \dots, \nbk} \\
                         p \in \Set{1, \dots, \nbk} \\
                         q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                         t \in \Set{0, \dots, \cT - 1}
                       \end{dcases}.
    \end{align*}
    \tag{15}\label{15}
  \]

.. dropdown:: 推導式子 :math:`\eqref{15}`

  根據式子 :math:`\eqref{10}` 我們可以求得 :math:`\Wbk{k}` 相對於 forget/input/output gate units 的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\yfg_{k^\star}(t + 1)}{\Wbk{k}_{p, q}} & = \dv{\yfg_{k^\star}(t + 1)}{\zfg_{k^\star}(t + 1)} \cdot \sum_{j = 1}^{\din + \nbk \times (3 + \dbk)} \qty[\cancelto{\aptr 0}{\dv{\zfg_{k^\star}(t + 1)}{\xt_j(t)}} \cdot \dv{\xt_j(t)}{\Wbk{k}_{p, q}}] \\
                                                   & \aptr 0 \qqtext{where} \begin{dcases}
                                                                              k \in \Set{1, \dots, \nbk} \\
                                                                              k^\star \in \Set{1, \dots, \nbk} \\
                                                                              p \in \Set{1, \dots, \dbk} \\
                                                                              q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                                              t \in \Set{0, \dots, \cT - 1}
                                                                            \end{dcases}. \\
        \dv{\yig_{k^\star}(t + 1)}{\Wbk{k}_{p, q}} & = \dv{\yig_{k^\star}(t + 1)}{\zig_{k^\star}(t + 1)} \cdot \sum_{j = 1}^{\din + \nbk \times (3 + \dbk)} \qty[\cancelto{\aptr 0}{\dv{\zig_{k^\star}(t + 1)}{\xt_j(t)}} \cdot \dv{\xt_j(t)}{\Wbk{k}_{p, q}}] \\
                                                   & \aptr 0 \qqtext{where} \begin{dcases}
                                                                              k \in \Set{1, \dots, \nbk} \\
                                                                              k^\star \in \Set{1, \dots, \nbk} \\
                                                                              p \in \Set{1, \dots, \dbk} \\
                                                                              q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                                              t \in \Set{0, \dots, \cT - 1}
                                                                            \end{dcases}. \\
        \dv{\yog_{k^\star}(t + 1)}{\Wbk{k}_{p, q}} & = \dv{\yog_{k^\star}(t + 1)}{\zog_{k^\star}(t + 1)} \cdot \sum_{j = 1}^{\din + \nbk \times (3 + \dbk)} \qty[\cancelto{\aptr 0}{\dv{\zog_{k^\star}(t + 1)}{\xt_j(t)}} \cdot \dv{\xt_j(t)}{\Wbk{k}_{p, q}}] \\
                                                   & \aptr 0 \qqtext{where} \begin{dcases}
                                                                              k \in \Set{1, \dots, \nbk} \\
                                                                              k^\star \in \Set{1, \dots, \nbk} \\
                                                                              p \in \Set{1, \dots, \dbk} \\
                                                                              q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                                              t \in \Set{0, \dots, \cT - 1}
                                                                            \end{dcases}.
      \end{align*}
    \]

  在丟棄部份微分後 :math:`\Wbk{k}` 將\ **無法**\透過 forget/input/output gate units 取得資訊。
  直覺上我們認為 :math:`\Wbk{k}` 應該可以透過 memory cell internal states 取得資訊。
  所以接下來我們推導 :math:`\Wbk{k}` 相對於 memory cell internal states 的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\sbk{k^\star}_i(t + 1)}{\Wbk{k}_{p, q}} & = \cancelto{\aptr 0}{\dv{\yfg_{k^\star}(t + 1)}{\Wbk{k}_{p, q}}} \cdot \sbk{k^\star}_i(t) + \yfg_{k^\star}(t + 1) \cdot \dv{\sbk{k^\star}_i(t)}{\Wbk{k}_{p, q}} + \cancelto{\aptr 0}{\dv{\yig_{k^\star}(t + 1)}{\Wbk{k}_{p, q}}} \cdot g\qty(\zbk{k^\star}_i(t + 1)) + \yig_{k^\star}(t + 1) \cdot \dv{g\qty(\zbk{k^\star}_i(t + 1))}{\zbk{k^\star}_i(t + 1)} \cdot \qty[\delta_{k^\star, k} \cdot \delta_{i, p} \cdot \xt_q(t) + \sum_{j = 1}^{\din + \nbk \times (3 + \dbk)} \qty[\Wbk{k^\star}_{i, j} \cdot \dv{\xt_j(t)}{\Wbk{k}_{p, q}}]] \\
                                                    & \aptr \yfg_{k^\star}(t + 1) \cdot \dv{\sbk{k^\star}_i(t)}{\Wbk{k}_{p, q}} + \yig_{k^\star}(t + 1) \cdot g'\qty(\zbk{k^\star}_i(t + 1)) \cdot \qty[\delta_{k^\star, k} \cdot \delta_{i, p} \cdot \xt_q(t) + \sum_{j = 1}^{\din + \nbk \times (3 + \dbk)} \qty[\Wbk{k^\star}_{i, j} \cdot \dv{\xt_j(t)}{\Wbk{k}_{p, q}}]] \\
                                                    & \qqtext{where} \begin{dcases}
                                                                       i \in \Set{1, \dots, \dbk} \\
                                                                       k \in \Set{1, \dots, \nbk} \\
                                                                       k^\star \in \Set{1, \dots, \nbk} \\
                                                                       p \in \Set{1, \dots, \dbk} \\
                                                                       q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                                       t \in \Set{0, \dots, \cT - 1}
                                                                     \end{dcases}.
      \end{align*}
    \]

  可以發現 :math:`\Wbk{k}` 對於 memory cell internal states 的全微分會有 BPTT 的問題，因此原始 LSTM 論文中提出額外丟棄 memory cell internal states 的部份微分，結果如下：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\sbk{k^\star}_i(t + 1)}{\Wbk{k}_{p, q}} & \aptr \yfg_{k^\star}(t + 1) \cdot \dv{\sbk{k^\star}_i(t)}{\Wbk{k}_{p, q}} + \yig_{k^\star}(t + 1) \cdot g'\qty(\zbk{k^\star}_i(t + 1)) \cdot \qty[\delta_{k^\star, k} \cdot \delta_{i, p} \cdot \xt_q(t) + \cancelto{\aptr 0}{\sum_{j = 1}^{\din + \nbk \times (3 + \dbk)} \qty[\Wbk{k^\star}_{i, j} \cdot \dv{\xt_j(t)}{\Wbk{k}_{p, q}}]}] \\
                                                    & \aptr \yfg_{k^\star}(t + 1) \cdot \dv{\sbk{k^\star}_i(t)}{\Wbk{k}_{p, q}} + \yig_{k^\star}(t + 1) \cdot g'\qty(\zbk{k^\star}_i(t + 1)) \cdot \delta_{k^\star, k} \cdot \delta_{i, p} \cdot \xt_q(t) \\
                                                    & \qqtext{where} \begin{dcases}
                                                                       i \in \Set{1, \dots, \dbk} \\
                                                                       k \in \Set{1, \dots, \nbk} \\
                                                                       k^\star \in \Set{1, \dots, \nbk} \\
                                                                       p \in \Set{1, \dots, \dbk} \\
                                                                       q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                                       t \in \Set{0, \dots, \cT - 1}
                                                                     \end{dcases}.
      \end{align*}
    \]

  觀察上式可以發現當 :math:`k \neq k^\star` 或 :math:`i \neq p` 時微分結果為 :math:`0`：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\sbk{k^\star}_i(t + 1)}{\Wbk{k}_{p, q}} & \aptr \yfg_{k^\star}(t + 1) \cdot \dv{\sbk{k^\star}_i(t)}{\Wbk{k}_{p, q}} + \yig_{k^\star}(t + 1) \cdot g'\qty(\zbk{k^\star}_i(t + 1)) \cdot \cancelto{0}{\delta_{k^\star, k} \cdot \delta_{i, p}} \cdot \xt_q(t) \\
                                                    & = \yfg_{k^\star}(t + 1) \cdot \dv{\sbk{k^\star}_i(t)}{\Wbk{k}_{p, q}} \\
                                                    & \aptr \qty[\prod_{t^\star = t}^{t + 1} \yfg_{k^\star}(t^\star)] \cdot \dv{\sbk{k^\star}_i(t - 1)}{\Wbk{k}_{p, q}} \\
                                                    & \vdots \\
                                                    & \aptr \qty[\prod_{t^\star = 1}^{t + 1} \yfg_{k^\star}(t^\star)] \cdot \cancelto{0}{\dv{\sbk{k^\star}_i(0)}{\Wbk{k}_{p, q}}} \\
                                                    & = 0 \\
                                                    & \qqtext{where} \begin{dcases}
                                                                       i \in \Set{1, \dots, \dbk} \\
                                                                       k \in \Set{1, \dots, \nbk} \\
                                                                       p \in \Set{1, \dots, \nbk} \\
                                                                       q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                                       t \in \Set{0, \dots, \cT - 1}
                                                                     \end{dcases}.
      \end{align*}
    \]

  因此我們將 :math:`\Wbk{k}` 相對於 memory cell internal states 的微分近似值改寫如下：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\sbk{k^\star}_i(t + 1)}{\Wbk{k}_{p, q}} & \aptr \delta_{k^\star, k} \cdot \delta_{i, p} \cdot \dv{\sbk{k}_p(t + 1)}{\Wbk{k}_{p, q}} \\
                                                    & \aptr \delta_{k^\star, k} \cdot \delta_{i, p} \cdot \qty[\yfg_{k}(t + 1) \cdot \dv{\sbk{k}_p(t)}{\Wbk{k}_{p, q}} + \yig_{k}(t + 1) \cdot g'\qty(\zbk{k}_p(t + 1)) \cdot \xt_q(t)] \\
                                                    & \qqtext{where} \begin{dcases}
                                                                       i \in \Set{1, \dots, \dbk} \\
                                                                       k \in \Set{1, \dots, \nbk} \\
                                                                       p \in \Set{1, \dots, \nbk} \\
                                                                       q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                                       t \in \Set{0, \dots, \cT - 1}
                                                                     \end{dcases}.
      \end{align*}
    \]

  .. note::

    上式就是論文的 3.10 式。

  可以發現 :math:`\Wbk{k}` 透過 memory cell internal states 得到的資訊其實都是來自於過去微分近似值的累加結果。
  實際上在執行參數更新演算法時只需要儲存過去累加而得的結果再結合當前計算結果，就可以得到最新的參數更新方向。
  使用前述推導結果我們可以得到 :math:`\Wbk{k}` 相對於 memory cell activation blocks 的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\ybk{k^\star}_i(t + 1)}{\Wbk{k}_{p, q}} & = \cancelto{\aptr 0}{\dv{\yog_{k^\star}(t + 1)}{\Wbk{k}_{p, q}}} \cdot h\qty(\sbk{k^\star}_i(t + 1)) + \yog_{k^\star}(t + 1) \cdot \dv{h\qty(\sbk{k^\star}_i(t + 1))}{\sbk{k^\star}_i(t + 1)} \cdot \dv{\sbk{k^\star}_i(t + 1)}{\Wbk{k}_{p, q}} \\
                                                    & \aptr \yog_{k^\star}(t + 1) \cdot h'\qty(\sbk{k^\star}_i(t + 1)) \cdot \delta_{k^\star, k} \cdot \delta_{i, p} \cdot \dv{\sbk{k}_p(t + 1)}{\Wbk{k}_{p, q}} \\
                                                    & \qqtext{where} \begin{dcases}
                                                                       i \in \Set{1, \dots, \dbk} \\
                                                                       k \in \Set{1, \dots, \nbk} \\
                                                                       k^\star \in \Set{1, \dots, \nbk} \\
                                                                       p \in \Set{1, \dots, \dbk} \\
                                                                       q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                                       t \in \Set{0, \dots, \cT - 1}
                                                                     \end{dcases}.
      \end{align*}
    \]

  同前述結論，只需要儲存過去計算而得的結果，最後乘上一些當前的計算狀態，就可以得到最新的參數更新方向。
  最後我們推得 :math:`\Wbk{k}` 相對於誤差的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        & \dv{\cL\qty(\y(t + 1) - \yh(t + 1))}{\Wbk{k}_{p, q}} \\
        & = \sum_{i = 1}^\dout \dv{\frac{1}{2} \qty(\y_i(t + 1) - \yh_i(t + 1))^2}{\Wbk{k}_{p, q}} \\
        & = \sum_{i = 1}^\dout \qty[\dv{\frac{1}{2} \qty(\y_i(t + 1) - \yh_i(t + 1))^2}{\y_i(t + 1)} \cdot \dv{\y_i(t + 1)}{\zout_i(t + 1)} \cdot \sum_{j = 1}^{\din + \nbk \times \dbk} \qty[\dv{\zout_i(t + 1)}{\xout_j(t + 1)} \cdot \cancelto{\aptr 0}{\dv{\xout_j(t + 1)}{\Wbk{k}_{p, q}}}]] \\
        & \aptr \sum_{i = 1}^\dout \qty[\qty(\y_i(t + 1) - \yh_i(t + 1)) \cdot {f^\opout}'\qty(\zout_i(t + 1)) \cdot \sum_{k^\star = 1}^\nbk \sum_{j = 1}^\dbk \qty[\Wout_{i, \din + (k^\star - 1) \times \dbk + j} \cdot \dv{\ybk{k^\star}_j(t + 1)}{\Wbk{k}_{p, q}}]] \\
        & \aptr \sum_{i = 1}^\dout \qty[\qty(\y_i(t + 1) - \yh_i(t + 1)) \cdot {f^\opout}'\qty(\zout_i(t + 1)) \cdot \sum_{k^\star = 1}^\nbk \sum_{j = 1}^\dbk \qty[\Wout_{i, \din + (k^\star - 1) \times \dbk + j} \cdot \yog_{k^\star}(t + 1) \cdot h'\qty(\sbk{k^\star}_j(t + 1)) \cdot \delta_{k^\star, k} \cdot \delta_{j, p} \cdot \dv{\sbk{k}_p(t + 1)}{\Wbk{k}_{p, q}}]] \\
        & = \qty[\sum_{i = 1}^\dout \qty(\y_i(t + 1) - \yh_i(t + 1)) \cdot {f^\opout}'\qty(\zout_i(t + 1)) \cdot \Wout_{i, \din + (k - 1) \times \dbk + p}] \cdot \yog_k(t + 1) \cdot h'\qty(\sbk{k}_p(t + 1)) \cdot \dv{\sbk{k}_p(t + 1)}{\Wbk{k}_{p, q}} \\
        & \qqtext{where} \begin{dcases}
                           k \in \Set{1, \dots, \nbk} \\
                           p \in \Set{1, \dots, \dbk} \\
                           q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                           t \in \Set{0, \dots, \cT - 1}
                         \end{dcases}.
      \end{align*}
    \]


時間複雜度
~~~~~~~~~~

參數更新依然使用 :term:`gradient descent`，由於使用基於 RTRL 的最佳化演算法，因此每個時間點 :math:`t + 1` 計算完誤差後就可以更新參數。
令 :math:`\alpha` 為 learning rate，我將式子 :math:`\eqref{5}` 的參數更新的演算法改成如下：

.. math::
  :nowrap:

  \[
    \begin{align*}
      \Wout_{p, q}   & \algoEq \Wout_{p, q} - \alpha \cdot \dv{\cL\qty(\y(t + 1), \yh(t + 1))}{\Wout_{p, q}} \qqtext{where} \begin{dcases}
                                                                                                                              p \in \Set{1, \dots, \dout} \\
                                                                                                                              q \in \Set{1, \dots, \din + \nbk \times \dbk} \\
                                                                                                                              t \in \Set{0, \dots, \cT - 1}
                                                                                                                            \end{dcases}. \\
      \Wog_{p, q}    & \algoEq \Wog_{p, q} - \alpha \cdot \dv{\cL\qty(\y(t + 1), \yh(t + 1))}{\Wog_{p, q}} \qqtext{where} \begin{dcases}
                                                                                                                            p \in \Set{1, \dots, \nbk} \\
                                                                                                                            q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                                                                                            t \in \Set{0, \dots, \cT - 1}
                                                                                                                          \end{dcases}. \\
      \Wfg_{p, q}    & \algoEq \Wfg_{p, q} - \alpha \cdot \dv{\cL\qty(\y(t + 1), \yh(t + 1))}{\Wfg_{p, q}} \qqtext{where} \begin{dcases}
                                                                                                                            p \in \Set{1, \dots, \nbk} \\
                                                                                                                            q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                                                                                            t \in \Set{0, \dots, \cT - 1}
                                                                                                                          \end{dcases}. \\
      \Wig_{p, q}    & \algoEq \Wig_{p, q} - \alpha \cdot \dv{\cL\qty(\y(t + 1), \yh(t + 1))}{\Wig_{p, q}} \qqtext{where} \begin{dcases}
                                                                                                                            p \in \Set{1, \dots, \nbk} \\
                                                                                                                            q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                                                                                            t \in \Set{0, \dots, \cT - 1}
                                                                                                                          \end{dcases}. \\
      \Wbk{k}_{p, q} & \algoEq \Wbk{k}_{p, q} - \alpha \cdot \dv{\cL\qty(\y(t + 1), \yh(t + 1))}{\Wbk{k}_{p, q}} \qqtext{where} \begin{dcases}
                                                                                                                                  k \in \Set{1, \dots, \nbk} \\
                                                                                                                                  p \in \Set{1, \dots, \dbk} \\
                                                                                                                                  q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                                                                                                  t \in \Set{0, \dots, \cT - 1}
                                                                                                                                \end{dcases}.
    \end{align*}
    \tag{16}\label{16}
  \]

因為 forget gate units 的計算過程與丟棄微分的概念與原版 LSTM 相同，因此時間複雜度同原始 LSTM 論文，細節可以看\ :doc:`我的筆記 </post/ml/long-short-term-memory>`。

.. math::
  :nowrap:

  \[
    \order{\dim(\Wout) + \dim(\Wog) + \dim(\Wig) \times \dbk + \nbk \times \dim(\Wbk{1})}
    \tag{17}\label{17}
  \]

空間複雜度
~~~~~~~~~~

根據 :math:`\eqref{13} \eqref{14} \eqref{15}`，當 forget gate units 關閉時，不只 memory cell internal states 會重設，與其相關的梯度也會重設，因此更新時需要額外紀錄以下的項次

.. math::
  :nowrap:

  \[
    \begin{align*}
      \dv{\sbk{p}_j(t + 1)}{\Wfg_{p, q}}    & \qqtext{where} \begin{dcases}
                                                               j \in \Set{1, \dots, \dbk} \\
                                                               p \in \Set{1, \dots, \nbk} \\
                                                               q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                               t \in \Set{0, \dots, \cT - 1}
                                                             \end{dcases}. \\
      \dv{\sbk{p}_j(t + 1)}{\Wig_{p, q}}    & \qqtext{where} \begin{dcases}
                                                               j \in \Set{1, \dots, \dbk} \\
                                                               p \in \Set{1, \dots, \nbk} \\
                                                               q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                               t \in \Set{0, \dots, \cT - 1}
                                                             \end{dcases}. \\
      \dv{\sbk{k}_p(t + 1)}{\Wbk{k}_{p, q}} & \qqtext{where} \begin{dcases}
                                                               k \in \Set{1, \dots, \nbk} \\
                                                               p \in \Set{1, \dots, \dbk} \\
                                                               q \in \Set{1, \dots, \din + \nbk \times (3 + \dbk)} \\
                                                               t \in \Set{0, \dots, \cT - 1}
                                                             \end{dcases}.
    \end{align*}
    \tag{18}\label{18}
  \]

同樣的概念在原始 LSTM 論文中也有出現，細節可以看\ :doc:`我的筆記 </post/ml/long-short-term-memory>`。
因此空間複雜度與原始 LSTM 論文相同：

.. math::
  :nowrap:

  \[
    \order{\dim(\Wout) + \dim(\Wog) + \dim(\Wig) \times \dbk + \nbk \times \dim(\Wbk{1})}.
    \tag{19}\label{19}
  \]

實驗 1：Continual Embedded Reber Grammar
========================================

.. figure:: https://i.imgur.com/rhHtVRN.png
  :alt: Continual Embedded Reber Grammar。
  :name: paper-fig-2

  圖 2：Continual Embedded Reber Grammar。

  圖片來源：:footcite:`gers-etal-2000-learning`。

任務定義
--------

- 根據原始 LSTM 論文 :footcite:`hochreiter-etal-1997-long` 中的實驗 1（Embedded Reber Grammar）進行修改，輸入為連續序列，連續序列的定義是由多個 Embedded Reber Grammar 產生的序列組合而成（細節可以看\ :doc:`我的筆記 </post/ml/long-short-term-memory>`）
- 每個分支的生成機率值為 :math:`0.5`

  - 生成序列的 minimal time lag 為 :math:`7`，由於可生成任意長度，因此 time lag 沒有上界
  - 作者有額外描述生成序列的屬性，例如生成序列長度的期望值為 :math:`11.5`，推導來源應該是作者的其他論文

- 當所有輸出單元的 MSE 低於 :math:`0.49` 時就當成預測正確
- 一個 input stream 由 :math:`10^5` 個輸入組成，當模型產生一個輸出的預測錯誤時就停止當前的 input stream
- 每次 training 停止就進行一次 test

  - 一次 training 會給予 :math:`1` 個 training stream
  - 一次 test 會給予 :math:`10` 個 test stream
  - 當模型連續預測 :math:`10^6` 個結果稱為 perfect solution
  - 當模型在 :math:`10` 個 test stream 上平均連續成功預測 :math:`\gt 1000` 個結果稱為 good solution
  - 當模型在 :math:`10` 個 test stream 上平均連續成功預測 :math:`\le 1000` 個結果稱為 bad solution

- 每輸入一個訊號就進行更新（RTRL）
- 訓練最多執行 :math:`30000` 次，實驗結果由 :math:`100` 個訓練模型實驗進行平均

LSTM 架構
---------

.. figure:: https://i.imgur.com/uUJjmSz.png
  :alt: LSTM 架構。
  :name: paper-fig-3

  圖 3：LSTM 架構。

  圖片來源：:footcite:`gers-etal-2000-learning`。

+---------------------------------------+----------------------------------------------------+--------------------------------------------------------------------------------+
| Hyperparameters                       | Value or Range                                     | Notes                                                                          |
+=======================================+====================================================+================================================================================+
| :math:`\din`                          | :math:`7`                                          | ``BEPSTVX``                                                                    |
+---------------------------------------+----------------------------------------------------+--------------------------------------------------------------------------------+
| :math:`\dbk`                          | :math:`2`                                          |                                                                                |
+---------------------------------------+----------------------------------------------------+--------------------------------------------------------------------------------+
| :math:`\nbk`                          | :math:`4`                                          |                                                                                |
+---------------------------------------+----------------------------------------------------+--------------------------------------------------------------------------------+
| :math:`\dout`                         | :math:`7`                                          | ``BEPSTVX``                                                                    |
+---------------------------------------+----------------------------------------------------+--------------------------------------------------------------------------------+
| :math:`\dim(\Wbk{k})`                 | :math:`\dbk \times (\din + \nbk \times \dbk)`      | The seven input units are fully connected to a hidden layer consisting of four |
+---------------------------------------+----------------------------------------------------+ memory blocks with 2 cells each (8 cells and 12 gates in total). The cell      |
| :math:`\dim(\Wfg)`                    | :math:`\nbk \times (\din + \nbk \times \dbk + 1)`  | outputs are fully connected to the cell inputs, all gates, and the seven       |
+---------------------------------------+----------------------------------------------------+ output units. The output units have additional "shortcut" connection from the  |
| :math:`\dim(\Wig)`                    | :math:`\nbk \times (\din + \nbk \times \dbk + 1)`  | input units (see Figure 3). All gates and output units are biased.             |
+---------------------------------------+----------------------------------------------------+                                                                                |
| :math:`\dim(\Wog)`                    | :math:`\nbk \times (\din + \nbk \times \dbk + 1)`  |                                                                                |
+---------------------------------------+----------------------------------------------------+                                                                                |
| :math:`\dim(\Wout)`                   | :math:`\dout \times (\din + \nbk \times \dbk + 1)` |                                                                                |
+---------------------------------------+----------------------------------------------------+--------------------------------------------------------------------------------+
| Total number of parameters            | :math:`424`                                        |                                                                                |
+---------------------------------------+----------------------------------------------------+--------------------------------------------------------------------------------+
| Weight initalization range            | :math:`[-0.2, 0.2]`                                | Bias weights to input and output gates are initialized blockwise: :math:`-0.5` |
+---------------------------------------+----------------------------------------------------+ for the first block, :math:`-1.0` for the second, :math:`-1.5` for the third,  |
| Forget gate bias initialization range | :math:`\Set{0.5, 1.0, 1.5, 2.0}`                   | and so forth. ... Forget gates are initialized with symmetric positive values: |
+---------------------------------------+----------------------------------------------------+ :math:`+0.5` for the first block, :math:`+1` for the second block, and so on.  |
| Input gate bias initialization range  | :math:`\Set{-0.5, -1.0, -1.5, -2.0}`               | Precise bias initialization is not critical, though; other values work just as |
+---------------------------------------+----------------------------------------------------+ well. All other weights including the output bias are initialized randomly in  |
| Output gate bias initialization range | :math:`\Set{-0.5, -1.0, -1.5, -2.0}`               | the range :math:`[-0.2, 0.2]`.                                                 |
+---------------------------------------+----------------------------------------------------+--------------------------------------------------------------------------------+
| Learning rate                         | :math:`0.5`                                        | At the beginning of each training stream, the learning rate :math:`\alpha` is  |
|                                       |                                                    | initialized with :math:`0.5`. It either remains fixed or decays by a factor of |
|                                       |                                                    | :math:`0.99` per time step (LSTM with :math:`\alpha`-decay).                   |
+---------------------------------------+----------------------------------------------------+--------------------------------------------------------------------------------+

實驗結果
--------

.. figure:: https://i.imgur.com/uu9Nccj.png
  :alt: Continual Embedded Reber Grammar 實驗結果。
  :name: paper-fig-4

  圖 4：Continual Embedded Reber Grammar 實驗結果。

  圖片來源：:footcite:`gers-etal-2000-learning`。

- 原始 LSTM :footcite:`hochreiter-etal-1997-long` 在有手動進行計算狀態的重置時表現非常好，但當沒有手動重置時完全無法執行任務
- 對 memory cell internal states 乘上 decay factor 也無濟於事（見 LSTM with state decay），作者嘗試了各種不同數值的 decay factor 並報告最好結果，但仍然表現很差
- 使用 forget gate units 的 LSTM 不需要手動重置計算狀態也能達成 perfect solution
- 嘗試使用不同的 learning rate decay 發現可以讓模型效果變好
- 額外實驗：

  - 將 Embedded Reber Grammar 開頭的 ``B`` 與結尾的 ``E`` 去除
  - 因為沒有了開頭與結尾的輸入提示，模型變得更難判斷一個序列的斷點
  - 實驗證實使用 forget gate units 的 LSTM 仍然可以達成 perfect solution，只是達成的比例下降

分析
----

.. figure:: https://i.imgur.com/qwU4pnG.png
  :alt: 原版 LSTM memory cell internal states 的累加值。
  :name: paper-fig-5

  圖 5：原版 LSTM memory cell internal states 的累加值。

  圖片來源：:footcite:`gers-etal-2000-learning`。

.. figure:: https://i.imgur.com/jtLnfu2.png
  :alt: Forget gate units 重設 memory cell internal states。
  :name: paper-fig-6

  圖 6：Forget gate units 重設 memory cell internal states。

  作者在第三個 memory cell block 觀察到 memory cell internal states 的重設現象，而 forget gate units 也在 memory cell internal states 重設時數值為 :math:`0`，證實 forget gate units 的效用。
  圖片來源：:footcite:`gers-etal-2000-learning`。

.. figure:: https://i.imgur.com/K1mp9rg.png
  :alt: LSTM 加上 forget gate units 後第一個 memory cell internal states。
  :name: paper-fig-7

  圖 7：LSTM 加上 forget gate units 後第一個 memory cell internal states。

  圖片來源：:footcite:`gers-etal-2000-learning`。

- 觀察原版 LSTM :footcite:`hochreiter-etal-1997-long` 的 memory cell internal states

  - 實驗結果請見 :ref:`paper-fig-5`
  - 發現在不進行手動重設的狀態下，memory cell internal states 的數值只會不斷的累加（朝向極正或極負前進）
  - 發現增加的程度成線性增長，與 memory cell internal states 的計算機制相符

- 觀察加上 forget gate units 後 LSTM 的 memory cell internal states

  - 實驗結果請見 :ref:`paper-fig-6`
  - 發現模型學會自動重設 memory cell internal states，且 forget gate units 也在對應重設的時間點為 :math:`0`，以此證實 forget gate units 真的達成設計的目的
  - 發現 forget gate units 長時間維持在 :math:`1` 的狀態，只在真的需要重設 memory cell internal states 時轉為 :math:`0`

- 觀察只發現第三個 memory cell block 直接符合作者對 forget gate units 的預期

  - 比較 :ref:`paper-fig-6` 與 :ref:`paper-fig-7`
  - 作者宣稱此實驗中 LSTM 的第三個 memory cell block 負責決定輸入的斷點，展現了重設長期記憶的能力
  - 作者宣稱此實驗中 LSTM 的第一個 memory cell block 負責決定預測 Reber grammar 的字元，展現了重設短期記憶的能力
  - 雖然沒有畫圖，但作者說其他 memory cell blocks 也有展現重設 memory cell internal states 的現象

實驗 2：Noisy Temporal Order Problem
====================================

任務定義
--------

- 作者想要驗證是否有任務是只能使用原版 LSTM 可以解決但增加 forget gate units 後不能解決
- 此任務就是原始 LSTM 論文 :footcite:`hochreiter-etal-1997-long` 中的實驗 6b（細節可以看\ :doc:`我的筆記 </post/ml/long-short-term-memory>`）
- 此任務的 minimal time lag 為 :math:`80`
- 由於此任務需要讓記憶維持一段不短的時間，因此遺忘資訊對於這個任務可能有害

LSTM 架構
---------

與實驗 1 大致相同，只做以下修改

- :math:`\din = \dout = 8`
- 將 forget gate units 的 bias term 初始化成較大的正數（論文使用 :math:`5`），讓 forget gate units 很難被關閉，藉此達到跟原本 LSTM 幾乎相同的計算能力

實驗結果
--------

- 使用 forget gate units 的 LSTM 仍然能夠解決 Noisy Temporal Order Problem

  - 當 bias term 初始化成較大的正數（例如 :math:`5`）時，訓練成功所需時間與原版 LSTM 相同
  - 當 bias term 初始化成較小的正數（例如 :math:`1`）時，訓練成功所需時間約為原版 LSTM 的 :math:`3` 倍，作者認為速度變慢是因為模型必須學會不能忘記資訊

- 根據實驗結果，原版 LSTM 可以解決的任務，在加上 forget gate units 後仍然可以解決

實驗 3：Continual Noisy Temporal Order Problem
==============================================

任務定義
--------

- 此任務是根據原始 LSTM 論文 :footcite:`hochreiter-etal-1997-long` 中的實驗 6b 進行修改

  - 輸入改為如同實驗 1 的 input stream
  - 每個 input stream 由 :math:`100` 筆 Noisy Temporal Order 序列所組成

- 每次 training 停止就進行一次 test

  - 一次 training 會給予 :math:`1` 個 training stream
  - 一次 test 會給予 :math:`10` 個 test stream
  - 當模型連續預測 :math:`100` 個 input stream 稱為 perfect solution
  - 當模型在 :math:`10` 個 test stream 上平均連續成功預測 :math:`\le 100` 個結果稱為 partial solution

- 訓練最多執行 :math:`10^5` 次，實驗結果由 :math:`100` 個訓練模型實驗進行平均
- 論文沒有講怎麼計算誤差與更新，我猜變成每個非預測時間點必須輸出 :math:`0`，預測時間點時輸出預測結果

LSTM 架構
---------

與實驗 2 相同。

實驗結果
--------

.. figure:: https://i.imgur.com/VV5wQVG.png
  :alt: Continual Noisy Temporal Order Problem 實驗結果。
  :name: paper-fig-8

  圖 8：Continual Noisy Temporal Order Problem 實驗結果。

  圖片來源：:footcite:`gers-etal-2000-learning`。

- 實驗再次驗證原版 LSTM 無法解決連續輸入，但使用 forget gate units 後就能夠解決問題
- 將 learning rate 使用 decay factor :math:`0.9` 逐漸下降可以讓模型表現變更好，但作者認為這不重要

.. footbibliography::

.. ====================================================================================================================
.. external links
.. ====================================================================================================================

.. _Pytorch-LSTM: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html?highlight=lstm#torch.nn.LSTM
