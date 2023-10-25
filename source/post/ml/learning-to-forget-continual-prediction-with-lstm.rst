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
        \newcommand{\opblk}{\operatorname{block}}
        \newcommand{\opfg}{\operatorname{fg}}
        \newcommand{\opig}{\operatorname{ig}}
        \newcommand{\opin}{\operatorname{in}}
        \newcommand{\oplen}{\operatorname{len}}
        \newcommand{\opnet}{\operatorname{net}}
        \newcommand{\opog}{\operatorname{og}}
        \newcommand{\opout}{\operatorname{out}}
        \newcommand{\opseq}{\operatorname{seq}}

        % Memory cell blocks.
        \newcommand{\blk}[1]{{\opblk^{#1}}}

        % Vectors' notations.
        \newcommand{\vs}{\mathbf{s}}
        \newcommand{\vsopblk}[1]{\vs^\blk{#1}}
        \newcommand{\vx}{\mathbf{x}}
        \newcommand{\vxopout}{\vx^\opout}
        \newcommand{\vxt}{\tilde{\vx}}
        \newcommand{\vy}{\mathbf{y}}
        \newcommand{\vyh}{\hat{\vy}}
        \newcommand{\vyopblk}[1]{\vy^\blk{#1}}
        \newcommand{\vyopfg}{\vy^\opfg}
        \newcommand{\vyopig}{\vy^\opig}
        \newcommand{\vyopog}{\vy^\opog}
        \newcommand{\vz}{\mathbf{z}}
        \newcommand{\vzopblk}[1]{\vz^\blk{#1}}
        \newcommand{\vzopfg}{\vz^\opfg}
        \newcommand{\vzopig}{\vz^\opig}
        \newcommand{\vzopog}{\vz^\opog}
        \newcommand{\vzopout}{\vz^\opout}

        % Matrixs' notation.
        \newcommand{\vW}{\mathbf{W}}
        \newcommand{\vWopblk}[1]{\vW^\blk{#1}}
        \newcommand{\vWopfg}{\vW^\opfg}
        \newcommand{\vWopig}{\vW^\opig}
        \newcommand{\vWopog}{\vW^\opog}
        \newcommand{\vWopout}{\vW^\opout}

        % Symbols in mathcal.
        \newcommand{\cL}{\mathcal{L}}
        \newcommand{\cT}{\mathcal{T}}

        % Vectors with subscript.
        \newcommand{\vxj}{{\vx_j}}
        \newcommand{\vyi}{{\vy_i}}
        \newcommand{\vyj}{{\vy_j}}
        \newcommand{\vzi}{{\vz_i}}

        % Matrixs with subscripts.
        \newcommand{\vWii}{{\vW_{i, i}}}
        \newcommand{\vWij}{{\vW_{i, j}}}

        % Dimensions.
        \newcommand{\din}{{d_\opin}}
        \newcommand{\dout}{{d_\opout}}
        \newcommand{\dblk}{{d_\opblk}}
        \newcommand{\nblk}{{n_\opblk}}

        % Derivative of loss(#2) with respect to net input #1 at time #3.
        \newcommand{\vth}[2]{{\vartheta_{#1}^{#2}}}

        % Gradient approximation by truncating gradient.
        \newcommand{\aptr}{\approx_{\operatorname{tr}}}
      \]

..
  <!-- Operator in. -->
  $\providecommand{\opnet}{}$
  $\renewcommand{\opnet}{\operatorname{net}}$
  <!-- Operator in. -->
  $\providecommand{\opin}{}$
  $\renewcommand{\opin}{\operatorname{in}}$
  <!-- Operator out. -->
  $\providecommand{\opout}{}$
  $\renewcommand{\opout}{\operatorname{out}}$
  <!-- Operator cell block. -->
  $\providecommand{\opblk}{}$
  $\renewcommand{\opblk}{\operatorname{block}}$
  <!-- Operator cell multiplicative forget gate. -->
  $\providecommand{\opfg}{}$
  $\renewcommand{\opfg}{\operatorname{fg}}$
  <!-- Operator cell multiplicative input gate. -->
  $\providecommand{\opig}{}$
  $\renewcommand{\opig}{\operatorname{ig}}$
  <!-- Operator cell multiplicative output gate. -->
  $\providecommand{\opog}{}$
  $\renewcommand{\opog}{\operatorname{og}}$
  <!-- Operator sequence. -->
  $\providecommand{\opseq}{}$
  $\renewcommand{\opseq}{\operatorname{seq}}$
  <!-- Operator loss. -->
  $\providecommand{\oploss}{}$
  $\renewcommand{\oploss}{\operatorname{loss}}$

  <!-- Net input. -->
  $\providecommand{\net}{}$
  $\renewcommand{\net}[2]{\opnet_{#1}(#2)}$
  <!-- Net input with activatiton f. -->
  $\providecommand{\fnet}{}$
  $\renewcommand{\fnet}[2]{f_{#1}\big(\net{#1}{#2}\big)}$
  <!-- Derivative of f with respect to net input. -->
  $\providecommand{\dfnet}{}$
  $\renewcommand{\dfnet}[2]{f_{#1}'\big(\net{#1}{#2}\big)}$

  <!-- Input dimension. -->
  $\providecommand{\din}{}$
  $\renewcommand{\din}{d_{\opin}}$
  <!-- Output dimension. -->
  $\providecommand{\dout}{}$
  $\renewcommand{\dout}{d_{\opout}}$
  <!-- Cell block dimension. -->
  $\providecommand{\dblk}{}$
  $\renewcommand{\dblk}{d_{\opblk}}$

  <!-- Number of cell blocks. -->
  $\providecommand{\nblk}{}$
  $\renewcommand{\nblk}{n_{\opblk}}$

  <!-- Cell block k. -->
  $\providecommand{\blk}{}$
  $\renewcommand{\blk}[1]{\opblk^{#1}}$

  <!-- Weight of multiplicative forget gate. -->
  $\providecommand{\wfg}{}$
  $\renewcommand{\wfg}{w^{\opfg}}$
  <!-- Weight of multiplicative input gate. -->
  $\providecommand{\vWopig}{}$
  $\renewcommand{\vWopig}{w^{\opig}}$
  <!-- Weight of multiplicative output gate. -->
  $\providecommand{\vWopog}{}$
  $\renewcommand{\vWopog}{w^{\opog}}$
  <!-- Weight of cell units. -->
  $\providecommand{\wblk}{}$
  $\renewcommand{\wblk}[1]{w^{\blk{#1}}}$
  <!-- Weight of output units. -->
  $\providecommand{\wout}{}$
  $\renewcommand{\wout}{w^{\opout}}$

  <!-- Net input of multiplicative forget gate. -->
  $\providecommand{\netfg}{}$
  $\renewcommand{\netfg}[2]{\opnet_{#1}^{\opfg}(#2)}$
  <!-- Net input of multiplicative forget gate with activatiton f. -->
  $\providecommand{\fnetfg}{}$
  $\renewcommand{\fnetfg}[2]{f_{#1}^{\opfg}\big(\netfg{#1}{#2}\big)}$
  <!-- Derivative of f with respect to net input of forget gate. -->
  $\providecommand{\dfnetfg}{}$
  $\renewcommand{\dfnetfg}[2]{f_{#1}^{\opfg}{'}\big(\netfg{#1}{#2}\big)}$
  <!-- Net input of multiplicative input gate. -->
  $\providecommand{\netig}{}$
  $\renewcommand{\netig}[2]{\opnet_{#1}^{\opig}(#2)}$
  <!-- Net input of multiplicative input gate with activatiton f. -->
  $\providecommand{\fnetig}{}$
  $\renewcommand{\fnetig}[2]{f_{#1}^{\opig}\big(\netig{#1}{#2}\big)}$
  <!-- Derivative of f with respect to net input of input gate. -->
  $\providecommand{\dfnetig}{}$
  $\renewcommand{\dfnetig}[2]{f_{#1}^{\opig}{'}\big(\netig{#1}{#2}\big)}$
  <!-- Net input of multiplicative output gate. -->
  $\providecommand{\netog}{}$
  $\renewcommand{\netog}[2]{\opnet_{#1}^{\opog}(#2)}$
  <!-- Net input of multiplicative output gate with activatiton f. -->
  $\providecommand{\fnetog}{}$
  $\renewcommand{\fnetog}[2]{f_{#1}^{\opog}\big(\netog{#1}{#2}\big)}$
  <!-- Derivative of f with respect to net input of output gate. -->
  $\providecommand{\dfnetog}{}$
  $\renewcommand{\dfnetog}[2]{f_{#1}^{\opog}{'}\big(\netog{#1}{#2}\big)}$
  <!-- Net input of output units. -->
  $\providecommand{\netout}{}$
  $\renewcommand{\netout}[2]{\opnet_{#1}^{\opout}(#2)}$
  <!-- Net input of output units with activatiton f. -->
  $\providecommand{\fnetout}{}$
  $\renewcommand{\fnetout}[2]{f_{#1}^{\opout}\big(\netout{#1}{#2}\big)}$
  <!-- Derivative of f with respect to net input of output units. -->
  $\providecommand{\dfnetout}{}$
  $\renewcommand{\dfnetout}[2]{f_{#1}^{\opout}{'}\big(\netout{#1}{#2}\big)}$

  <!-- Net input of cell unit. -->
  $\providecommand{\netblk}{}$
  $\renewcommand{\netblk}[3]{\opnet_{#1}^{\blk{#2}}(#3)}$
  <!-- Net input of cell unit with activatiton g. -->
  $\providecommand{\gnetblk}{}$
  $\renewcommand{\gnetblk}[3]{g_{#1}\big(\netblk{#1}{#2}{#3}\big)}$
  <!-- Derivative of g with respect to net input of cell unit. -->
  $\providecommand{\dgnetblk}{}$
  $\renewcommand{\dgnetblk}[3]{g_{#1}'\big(\netblk{#1}{#2}{#3}\big)}$
  <!-- Cell unit with activatiton h. -->
  $\providecommand{\hblk}{}$
  $\renewcommand{\hblk}[3]{h_{#1}\big(s_{#1}^{\blk{#2}}(#3)\big)}$
  <!-- Derivative of h with respect to cell unit. -->
  $\providecommand{\dhblk}{}$
  $\renewcommand{\dhblk}[3]{h_{#1}'\big(s_{#1}^{\blk{#2}}(#3)\big)}$

  <!-- Gradient approximation by truncating gradient. -->
  $\providecommand{\aptr}{}$
  $\renewcommand{\aptr}{\approx_{\operatorname{tr}}}$

重點
====

- 此篇論文 :footcite:`gers-etal-2000-learning` 與原版 LSTM :footcite:`hochreiter-etal-1997-long` 都寫錯自己的數學公式，但我的筆記內容主要以正確版本為主，原版 LSTM 可以參考\ :doc:`我的筆記 </post/ml/long-short-term-memory>`
- 原版 LSTM 沒有 forget gate ，現今常用的 LSTM 都有 forget gate ，概念由此篇論文提出
- 包含多個子序列的\ **連續輸入**\會讓 LSTM 的 memory cell internal states 沒有上下界

  - 現實中的大多數資料並不存在好的分割序列演算法，導致輸入給模型的資料通常都包含多個子序列
  - 根據實驗 1 的分析發現 memory cell internal states 的累積導致預測結果完全錯誤

- 使用 forget gate 讓模型學會適當的忘記已經處理過的子序列資訊

  - 當 forget gate 的 **bias term** 初始化為 **正數** 時會保持 memory cell internal states，等同於使用原版的 LSTM
  - 因此使用 forget gate 的 LSTM 能夠達成原版 LSTM 的功能，並額外擁有自動重設 memory cells 的機制

- 這篇模型的理論背景較少，實驗為主的描述居多

原始 LSTM
=========

.. note::

  這篇論文不使用 conventional hidden units，因此我不列出相關的公式。

符號定義
--------

我使用的符號與論文不同，我的符號定義請參考\ :doc:`我的筆記 </post/ml/long-short-term-memory>`。

+------------------------+---------------------------------------------------------------------------------------------------+----------------------+
| Symbol                 | Meaning                                                                                           | Value Range          |
+========================+===================================================================================================+======================+
| :math:`\dblk`          | Number of memory cells in each memory cell block at time step :math:`t`.                          | :math:`\Z^+`         |
+------------------------+---------------------------------------------------------------------------------------------------+----------------------+
| :math:`\nblk`          | Number of memory cell blocks at time step :math:`t`.                                              | :math:`\Z^+`         |
+------------------------+---------------------------------------------------------------------------------------------------+----------------------+
| :math:`\vx(t)`         | LSTM input at time step :math:`t`.                                                                | :math:`\R^\din`      |
+------------------------+---------------------------------------------------------------------------------------------------+----------------------+
| :math:`\vyopig(t)`     | Input gate units at time step :math:`t`.                                                          | :math:`[0, 1]^\nblk` |
+------------------------+---------------------------------------------------------------------------------------------------+----------------------+
| :math:`\vyopog(t)`     | Output gate units at time step :math:`t`.                                                         | :math:`[0, 1]^\nblk` |
+------------------------+---------------------------------------------------------------------------------------------------+----------------------+
| :math:`\vyopblk{k}(t)` | Output of the :math:`k`-th memory cell block at time step :math:`t`.                              | :math:`\R^\dblk`     |
+------------------------+---------------------------------------------------------------------------------------------------+----------------------+
| :math:`\vsopblk{k}(t)` | Internal states of the :math:`k`-th memory cell block at time step :math:`t`.                     | :math:`\R^\dblk`     |
+------------------------+---------------------------------------------------------------------------------------------------+----------------------+
| :math:`\vy(t)`         | LSTM output at time step :math:`t`.                                                               | :math:`\R^\dout`     |
+------------------------+---------------------------------------------------------------------------------------------------+----------------------+
| :math:`\sigma`         | Sigmoid function.                                                                                 | :math:`[0, 1]`       |
+------------------------+---------------------------------------------------------------------------------------------------+----------------------+
| :math:`f^\opig`        | Activation function for input gate units, set to :math:`\sigma` in this paper.                    | :math:`[0, 1]`       |
+------------------------+---------------------------------------------------------------------------------------------------+----------------------+
| :math:`f^\opog`        | Activation function for output gate units, set to :math:`\sigma` in this paper.                   | :math:`[0, 1]`       |
+------------------------+---------------------------------------------------------------------------------------------------+----------------------+
| :math:`f^\opout`       | Activation function for output units, set to :math:`\sigma` in this paper.                        | :math:`[0, 1]`       |
+------------------------+---------------------------------------------------------------------------------------------------+----------------------+
| :math:`g`              | Activation function for memory cells, set to :math:`4 \sigma - 2` in this paper.                  | :math:`[-2, 2]`      |
+------------------------+---------------------------------------------------------------------------------------------------+----------------------+
| :math:`h`              | Activation function for memory cell block activations, set to :math:`2 \sigma - 1` in this paper. | :math:`[-1, 1]`      |
+------------------------+---------------------------------------------------------------------------------------------------+----------------------+

計算定義
--------

以下就是 LSTM（1997 版本）的計算流程。

.. math::
  :nowrap:

  \[
    \begin{align*}
      & \algoProc{\operatorname{LSTM1997}}(\vx, \vWopig, \vWopog, \vWopblk{1}, \dots, \vWopblk{\nblk}, \vWopout) \\
      & \indent{1} \algoCmt{Initialize activations with zeros.} \\
      & \indent{1} \cT \algoEq \oplen(\vx) \\
      & \indent{1} \vyopig(0) \algoEq \zv \\
      & \indent{1} \vyopog(0) \algoEq \zv \\
      & \indent{1} \algoFor{k \in \Set{1, \dots, \nblk}} \\
      & \indent{2}   \vsopblk{k}(0) \algoEq \zv \\
      & \indent{2}   \vyopblk{k}(0) \algoEq \zv \\
      & \indent{1} \algoEndFor \\
      & \indent{1} \algoCmt{Do forward pass.} \\
      & \indent{1} \algoFor{t \in \Set{0, \dots, \cT - 1}} \\
      & \indent{2}   \algoCmt{Concatenate input units with activations.} \\
      & \indent{2}   \vxt(t) \algoEq \begin{pmatrix}
                                       \vx(t) \\
                                       \vyopig(t) \\
                                       \vyopog(t) \\
                                       \vyopblk{1}(t) \\
                                       \vdots \\
                                       \vyopblk{\nblk}(t)
                                     \end{pmatrix} \\
      & \indent{2}   \algoCmt{Compute input gate units' activations.} \\
      & \indent{2}   \vzopig(t + 1) \algoEq \vWopig \cdot \vxt(t) \\
      & \indent{2}   \vyopig(t + 1) \algoEq f^\opig\qty(\vzopig(t + 1)) \\
      & \indent{2}   \algoCmt{Compute output gate units' activations.} \\
      & \indent{2}   \vzopog(t + 1) \algoEq \vWopog \cdot \vxt(t) \\
      & \indent{2}   \vyopog(t + 1) \algoEq f^\opog\qty(\vzopog(t + 1)) \\
      & \indent{2}   \algoCmt{Compute the k-th memory cell block's activations.} \\
      & \indent{2}   \algoFor{k \in \Set{1, \dots, \nblk}} \\
      & \indent{3}     \vzopblk{k}(t + 1) \algoEq \vWopblk{k} \cdot \vxt(t) \\
      & \indent{3}     \vsopblk{k}(t + 1) \algoEq \vsopblk{k}(t) + \vyopig_k(t + 1) \cdot g\qty(\vzopblk{k}(t + 1)) \\
      & \indent{3}     \vyopblk{k}(t + 1) \algoEq \vyopog_k(t + 1) \cdot h\qty(\vsopblk{k}(t + 1)) \\
      & \indent{2}   \algoEndFor \\
      & \indent{2}   \algoCmt{Concatenate input units with new activations.} \\
      & \indent{2}   \vxopout(t + 1) \algoEq \begin{pmatrix}
                                               \vx(t) \\
                                               \vyopblk{1}(t + 1) \\
                                               \vdots \\
                                               \vyopblk{\nblk}(t + 1) \\
                                             \end{pmatrix} \\
      & \indent{2}   \algoCmt{Compute outputs.} \\
      & \indent{2}   \vzopout(t + 1) \algoEq \vWopout \cdot \vxopout(t + 1) \\
      & \indent{2}   \vy(t + 1) \algoEq f^\opout\qty(\vzopout(t + 1)) \\
      & \indent{1} \algoEndFor \\
      & \indent{1} \algoReturn \vy(1), \dots, \vy(\cT) \\
      & \algoEndProc
    \end{align*}
  \]

.. error::

  此篇論文 :footcite:`gers-etal-2000-learning` 與原版 LSTM 的論文 :footcite:`hochreiter-etal-1997-long` 都不小心將 :math:`\vy(t + 1)` 的輸入寫成 :math:`\vyopblk{k}(t)` 而不是 :math:`\vyopblk{k}(t + 1)`，我在上述公式中已經進行修正。
  對應的正確寫法在後續論文 :footcite:`gers-etal-2002-learning` 中才終於寫對。


參數結構
--------

+---------------------+---------------------------------------------------------------------------------------------------------+---------------------+-----------------------------------------+
| Parameter           | Meaning                                                                                                 | Output Vector Shape | Input Vector Shape                      |
+=====================+=========================================================================================================+=====================+=========================================+
| :math:`\vWopig`     | Weight matrix connect :math:`\vxt(t)` to input gate units :math:`\vyopig(t + 1)`.                       | :math:`\nblk`       | :math:`\din + \nblk \times (2 + \dblk)` |
+---------------------+---------------------------------------------------------------------------------------------------------+---------------------+-----------------------------------------+
| :math:`\vWopog`     | Weight matrix connect :math:`\vxt(t)` to output gate units :math:`\vyopog(t + 1)`.                      | :math:`\nblk`       | :math:`\din + \nblk \times (2 + \dblk)` |
+---------------------+---------------------------------------------------------------------------------------------------------+---------------------+-----------------------------------------+
| :math:`\vWopblk{k}` | Weight matrix connect :math:`\vxt(t)` to the :math:`k`-th memory cell block :math:`\vyopblk{k}(t + 1)`. | :math:`\dblk`       | :math:`\din + \nblk \times (2 + \dblk)` |
+---------------------+---------------------------------------------------------------------------------------------------------+---------------------+-----------------------------------------+
| :math:`\vWopout`    | Weight matrix connect :math:`\vxopout(t)` to output units :math:`\vy(t + 1)`.                           | :math:`\dblk`       | :math:`\din + \nblk \times \dblk`       |
+---------------------+---------------------------------------------------------------------------------------------------------+---------------------+-----------------------------------------+

最佳化
------

原始 LSTM 論文 :footcite:`hochreiter-etal-1997-long` 提出與 truncated BPTT 相似的概念，透過 RTRL 進行參數更新，故意\ **丟棄部份微分值**\來近似全微分，避免梯度爆炸或梯度消失的問題，同時節省更新所需的空間與時間（local in time and space）。
丟棄微分後的近似結果我以 :math:`\aptr` 表示，推導細節請見\ :doc:`我的筆記 </post/ml/long-short-term-memory>`，以下我直接列出對應的公式。

:math:`\vWopout` 相對於誤差的微分
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::
  :nowrap:

  \[
    \begin{align*}
      & \dv{\cL\qty(\vy(t + 1), \vyh(t + 1))}{\vWopout_{p, q}} = \qty(\vy_p(t + 1) - \vyh_p(t + 1)) \cdot {f^\opout}'\qty(\vzopout_p(t + 1)) \cdot \vxopout_q(t + 1) \\
      & \qqtext{where} \begin{dcases}
                         p \in \Set{1, \dots, \dout} \\
                         q \in \Set{1, \dots, \din + \nblk \times \dblk} \\
                         t \in \Set{0, \dots, \cT - 1}
                       \end{dcases}.
    \end{align*}
    \tag{1}\label{1}
  \]

:math:`\vWopog` 相對於誤差的微分近似值
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::
  :nowrap:

  \[
    \begin{align*}
      & \dv{\cL\qty(\vy(t + 1) - \vyh(t + 1))}{\vWopog_{p, q}} \aptr \qty(\sum_{j = 1}^\dblk \qty[\sum_{i = 1}^\dout \qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, \din + (p - 1) \times \dblk + j}] \cdot h\qty(\vsopblk{p}_j(t + 1))) \cdot {f^\opog}'\qty(\vzopog_p(t + 1)) \cdot \vxt_q(t) \\
      & \qqtext{where} \begin{dcases}
                         p \in \Set{1, \dots, \nblk} \\
                         q \in \Set{1, \dots, \din + \nblk \times (2 + \dblk)} \\
                         t \in \Set{0, \dots, \cT - 1}
                       \end{dcases}.
    \end{align*}
    \tag{2}\label{2}
  \]

:math:`\vWopig` 相對於誤差的微分近似值
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::
  :nowrap:

  \[
    \begin{align*}
      & \dv{\cL\qty(\vy(t + 1) - \vyh(t + 1))}{\vWopig_{p, q}} \aptr \qty(\sum_{j = 1}^\dblk \qty[\sum_{i = 1}^\dout \qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, \din + (p - 1) \times \dblk + j}] \cdot h'\qty(\vsopblk{p}_j(t + 1)) \cdot \sum_{t^\star = 0}^t \qty[{f^\opig}'\qty(\vzopig_p(t^\star + 1)) \cdot \vxt_q(t^\star) \cdot g\qty(\vzopblk{p}_j(t^\star + 1))]) \cdot \vyopog_p(t + 1) \\
      & \qqtext{where} \begin{dcases}
                         p \in \Set{1, \dots, \nblk} \\
                         q \in \Set{1, \dots, \din + \nblk \times (2 + \dblk)} \\
                         t \in \Set{0, \dots, \cT - 1}
                       \end{dcases}.
    \end{align*}
    \tag{3}\label{3}
  \]

:math:`\vWopblk{k}` 相對於誤差的微分近似值
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::
  :nowrap:

  \[
    \begin{align*}
      & \dv{\cL\qty(\vy(t + 1) - \vyh(t + 1))}{\vWopblk{k}_{p, q}} \aptr \qty[\sum_{i = 1}^\dout \qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, \din + (k - 1) \times \dblk + p}] \cdot \qty[\sum_{t^\star = 0}^t \vyopig_k(t^\star + 1) \cdot g'\qty(\vzopblk{k}_p(t^\star + 1)) \cdot \vxt_q(t^\star)] \cdot \vyopog_k(t + 1) \cdot h'\qty(\vsopblk{k}_p(t + 1)) \\
      & \qqtext{where} \begin{dcases}
                         k \in \Set{1, \dots, \nblk} \\
                         p \in \Set{1, \dots, \dblk} \\
                         q \in \Set{1, \dots, \din + \nblk \times (2 + \dblk)} \\
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
      \vWopout_{p, q}    & \algoEq \vWopout_{p, q} - \alpha \cdot \dv{\cL\qty(\vy(t + 1), \vyh(t + 1))}{\vWopout_{p, q}} \qqtext{where} \begin{dcases}
                                                                                                                                          p \in \Set{1, \dots, \dout} \\
                                                                                                                                          q \in \Set{1, \dots, \din + \nblk \times \dblk}
                                                                                                                                        \end{dcases}. \\
      \vWopog_{p, q}     & \algoEq \vWopog_{p, q} - \alpha \cdot \dv{\cL\qty(\vy(t + 1), \vyh(t + 1))}{\vWopog_{p, q}} \qqtext{where} \begin{dcases}
                                                                                                                                        p \in \Set{1, \dots, \nblk} \\
                                                                                                                                        q \in \Set{1, \dots, \din + \nblk \times (2 + \dblk)}
                                                                                                                                      \end{dcases}. \\
      \vWopig_{p, q}     & \algoEq \vWopig_{p, q} - \alpha \cdot \dv{\cL\qty(\vy(t + 1), \vyh(t + 1))}{\vWopig_{p, q}} \qqtext{where} \begin{dcases}
                                                                                                                                        p \in \Set{1, \dots, \nblk} \\
                                                                                                                                        q \in \Set{1, \dots, \din + \nblk \times (2 + \dblk)}
                                                                                                                                      \end{dcases}. \\
      \vWopblk{k}_{p, q} & \algoEq \vWopblk{k}_{p, q} - \alpha \cdot \dv{\cL\qty(\vy(t + 1), \vyh(t + 1))}{\vWopblk{k}_{p, q}} \qqtext{where} \begin{dcases}
                                                                                                                                                k \in \Set{1, \dots, \nblk} \\
                                                                                                                                                p \in \Set{1, \dots, \dblk} \\
                                                                                                                                                q \in \Set{1, \dots, \din + \nblk \times (2 + \dblk)}
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
    \vsopblk{k}(t + 1) \algoEq \vsopblk{k}(t) + \vyopig_k(t + 1) \cdot g\qty(\vzopblk{k}(t + 1))
    \tag{6}\label{6}
  \]
  \[
    \vyopblk{k}(t + 1) \algoEq \vyopog_k(t + 1) \cdot h\qty(\vsopblk{k}(t + 1))
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

  圖 1：在原始 LSTM 架構上增加 forget gate 。

  表格來源：:footcite:`gers-etal-2000-learning`。

作者提出在模型中加入 **forget gate units**，概念是讓 memory cell internal states 能夠自動進行重設。
如同 input/output gate units，forget gate units 會初始化成 :math:`\zv`，並透過以下計算更新 forget gate units：

.. math::
  :nowrap:

  \[
    \begin{align*}
      & \vxt(t) \algoEq \begin{pmatrix}
                          \vx(t) \\
                          \vyopfg(t) \\
                          \vyopig(t) \\
                          \vyopog(t) \\
                          \vyopblk{1}(t) \\
                          \vdots \\
                          \vyopblk{\nblk}(t)
                        \end{pmatrix} \\
      & \vzopfg(t + 1) \algoEq \vWopfg \cdot \vxt(t) \\
      & \vyopfg(t + 1) \algoEq f^\opfg\qty(\vzopfg(t + 1))
    \end{align*}
    \tag{8}\label{8}
  \]

注意以下幾點連帶的改動：

- :math:`\vxt(t)` 的輸入需要加上 :math:`\vyopfg(t)`
- 新增了參數 :math:`\vWopfg`，該參數的 input vector shape 為 :math:`\din + \nblk \times (3 + \dblk)`，output vector shape 為 :math:`\nblk`
- 因為 :math:`\vxt(t)` 做了更動，所以參數 :math:`\vWopig, \vWopog, \vWopblk{k}` 的 input vector shape 都改成 :math:`\din + \nblk \times (3 + \dblk)`

.. note::

  式子 :math:`\eqref{8}` 就是論文中的 (3.1) 式。

由於 forget gate units 的設計出發點是作為 memory cell internal states 的 decay factor，因此作者將式子 :math:`\eqref{6}` 的計算方法改成如下：

.. math::
  :nowrap:

  \[
    \vsopblk{k}(t + 1) \algoEq \vyopfg_k(t + 1) \cdot \vsopblk{k}(t) + \vyopig_k(t + 1) \cdot g\qty(\vzopblk{k}(t + 1))
    \tag{9}\label{9}
  \]

- Forget gate units 是以\ **乘法**\參與計算，因此稱為 **multiplicative gate units**

  - Memory cells in the same memory cell block **share** the same forget gate unit
  - 因此 :math:`\vyopfg_k(t + 1) \cdot \vsopblk{k}` 中的乘法是\ **純量**\乘上\ **向量**

- 模型會在訓練的過程中學習\ **關閉**\與\ **開啟** forget gate units

  - :math:`\vyopfg_k(t + 1) \approx 0` 代表\ **關閉** :math:`t + 1` 時間點的第 :math:`k` 個 forget gate unit，並\ **重設** :math:`\vsopblk{k}` 的計算狀態
  - :math:`\vyopfg_k(t + 1) \approx 1` 代表\ **開啟** :math:`t + 1` 時間點的第 :math:`k` 個 forget gate unit，並\ **保留** :math:`\vsopblk{k}` 的計算狀態
  - 全部 :math:`\nblk` 個 forget gate units 不一定要同時關閉或開啟

.. note::

  式子 :math:`\eqref{9}` 就是論文中的 (3.2) 式。

計算定義
--------

加入 forget gate units 後我重新整理一次 LSTM 的計算定義，如下所示。

.. math::
  :nowrap:

  \[
    \begin{align*}
      & \algoProc{\operatorname{LSTM1997}}(\vx, \vWopfg, \vWopig, \vWopog, \vWopblk{1}, \dots, \vWopblk{\nblk}, \vWopout) \\
      & \indent{1} \algoCmt{Initialize activations with zeros.} \\
      & \indent{1} \cT \algoEq \oplen(\vx) \\
      & \indent{1} \vyopfg(0) \algoEq \zv \\
      & \indent{1} \vyopig(0) \algoEq \zv \\
      & \indent{1} \vyopog(0) \algoEq \zv \\
      & \indent{1} \algoFor{k \in \Set{1, \dots, \nblk}} \\
      & \indent{2}   \vsopblk{k}(0) \algoEq \zv \\
      & \indent{2}   \vyopblk{k}(0) \algoEq \zv \\
      & \indent{1} \algoEndFor \\
      & \indent{1} \algoCmt{Do forward pass.} \\
      & \indent{1} \algoFor{t \in \Set{0, \dots, \cT - 1}} \\
      & \indent{2}   \algoCmt{Concatenate input units with activations.} \\
      & \indent{2}   \vxt(t) \algoEq \begin{pmatrix}
                                       \vx(t) \\
                                       \vyopfg(t) \\
                                       \vyopig(t) \\
                                       \vyopog(t) \\
                                       \vyopblk{1}(t) \\
                                       \vdots \\
                                       \vyopblk{\nblk}(t)
                                     \end{pmatrix} \\
      & \indent{2}   \algoCmt{Compute forget gate units' activations.} \\
      & \indent{2}   \vzopfg(t + 1) \algoEq \vWopfg \cdot \vxt(t) \\
      & \indent{2}   \vyopfg(t + 1) \algoEq f^\opfg\qty(\vzopfg(t + 1)) \\
      & \indent{2}   \algoCmt{Compute input gate units' activations.} \\
      & \indent{2}   \vzopig(t + 1) \algoEq \vWopig \cdot \vxt(t) \\
      & \indent{2}   \vyopig(t + 1) \algoEq f^\opig\qty(\vzopig(t + 1)) \\
      & \indent{2}   \algoCmt{Compute output gate units' activations.} \\
      & \indent{2}   \vzopog(t + 1) \algoEq \vWopog \cdot \vxt(t) \\
      & \indent{2}   \vyopog(t + 1) \algoEq f^\opog\qty(\vzopog(t + 1)) \\
      & \indent{2}   \algoCmt{Compute the k-th memory cell block's activations.} \\
      & \indent{2}   \algoFor{k \in \Set{1, \dots, \nblk}} \\
      & \indent{3}     \vzopblk{k}(t + 1) \algoEq \vWopblk{k} \cdot \vxt(t) \\
      & \indent{3}     \vsopblk{k}(t + 1) \algoEq \vyopfg_k(t + 1) \cdot \vsopblk{k}(t) + \vyopig_k(t + 1) \cdot g\qty(\vzopblk{k}(t + 1)) \\
      & \indent{3}     \vyopblk{k}(t + 1) \algoEq \vyopog_k(t + 1) \cdot h\qty(\vsopblk{k}(t + 1)) \\
      & \indent{2}   \algoEndFor \\
      & \indent{2}   \algoCmt{Concatenate input units with new activations.} \\
      & \indent{2}   \vxopout(t + 1) \algoEq \begin{pmatrix}
                                               \vx(t) \\
                                               \vyopblk{1}(t + 1) \\
                                               \vdots \\
                                               \vyopblk{\nblk}(t + 1) \\
                                             \end{pmatrix} \\
      & \indent{2}   \algoCmt{Compute outputs.} \\
      & \indent{2}   \vzopout(t + 1) \algoEq \vWopout \cdot \vxopout(t + 1) \\
      & \indent{2}   \vy(t + 1) \algoEq f^\opout\qty(\vzopout(t + 1)) \\
      & \indent{1} \algoEndFor \\
      & \indent{1} \algoReturn \vy(1), \dots, \vy(\cT) \\
      & \algoEndProc
    \end{align*}
  \]

..
  ### bias term

  如同[原始 LSTM][LSTM1997]，**輸入閘門**與**輸出閘門**可以使用**bias term**（bias term），將bias term初始化成**負數**可以讓輸入閘門與輸出閘門在需要的時候才被啟用（細節可以看[我的筆記][note-LSTM1997]）。

  而 forget gate 也可以使用bias term，但初始化的數值應該為**正數**，理由是在模型計算前期應該要讓 forget gate 開啟（$\vyopfg \approx 1$），讓 memory cell internal states 的數值能夠進行改變。

  注意 forget gate 只有在**關閉**（$\vyopfg \approx 0$）時才能進行遺忘，這個名字取得不是很好。

  ### 最佳化

  基於[原始 LSTM][LSTM1997] 的最佳化演算法，將流出 forget gate 的梯度也一起**丟棄**

  $$
  \begin{align*}
  \pd{\netfg{k}{t + 1}}{y_{k^{\star}}^{\opfg}(t)} & \aptr 0 && k = 1, \dots, \nblk \\
  \pd{\netfg{k}{t + 1}}{y_{k^{\star}}^{\opig}(t)} & \aptr 0 && k^{\star} = 1, \dots, \nblk \\
  \pd{\netfg{k}{t + 1}}{y_{k^{\star}}^{\opog}(t)} & \aptr 0 \\
  \pd{\netfg{k}{t + 1}}{y_i^{\blk{k^{\star}}}(t)} & \aptr 0 && i = 1, \dots, \dblk
  \end{align*} \tag{13}\label{13}
  $$

  因此 forget gate 的參數剩餘梯度為

  $$
  \begin{align*}
  & \pd{\oploss(t + 1)}{\wfg_{k, q}} \\
  & \aptr \sum_{i = 1}^{\dout} \Bigg[\pd{\oploss(t + 1)}{y_i(t + 1)} \cdot \pd{y_i(t + 1)}{\netout{i}{t + 1}} \cdot \\
  & \quad \pa{\sum_{j = 1}^{\dblk} \pd{\netout{i}{t + 1}}{y_j^{\blk{k}}(t + 1)} \cdot \pd{y_j^{\blk{k}}(t + 1)}{s_j^{\blk{k}}(t + 1)} \cdot \pd{s_j^{\blk{k}}(t + 1)}{\wfg_{k, q}}}\Bigg] \\
  & \aptr \sum_{i = 1}^{\dout} \Bigg[\pd{\oploss(t + 1)}{y_i(t + 1)} \cdot \pd{y_i(t + 1)}{\netout{i}{t + 1}} \cdot \Bigg(\sum_{j = 1}^{\dblk} \pd{\netout{i}{t + 1}}{y_j^{\blk{k}}(t + 1)} \cdot \pd{y_j^{\blk{k}}(t + 1)}{s_j^{\blk{k}}(t + 1)} \cdot \\
  & \quad \quad \br{y_k^{\opfg}(t + 1) \cdot \pd{s_j^{\blk{k}}(t)}{\wfg_{k, q}} + s_j^{\blk{k}}(t) \cdot \pd{y_k^{\opfg}(t + 1)}{\netfg{k}{t + 1}} \cdot \pd{\netfg{k}{t + 1}}{\wfg_{k, q}}}\Bigg)\Bigg] \\
  & \aptr \sum_{i = 1}^{\dout} \Bigg[\big(y_i(t + 1) - \hat{y}_i(t + 1)\big) \cdot \dfnetout{i}{t + 1} \cdot \\
  & \quad \Bigg(\sum_{j = 1}^{\dblk} \wout_{i, \din + (k - 1) \cdot \dblk + j} \cdot y_k^{\opog}(t + 1) \cdot \dhblk{j}{k}{t + 1} \cdot \\
  & \quad \quad \br{y_k^{\opfg}(t + 1) \cdot \pd{s_j^{\blk{k}}(t)}{\wfg_{k, q}}  + s_j^{\blk{k}}(t) \cdot \dfnetog{k}{t + 1} \cdot \begin{pmatrix}
  x(t) \\
  \vyopfg(t) \\
  y^{\opig}(t) \\
  y^{\opog}(t) \\
  y^{\blk{1}}(t) \\
  \vdots \\
  y^{\blk{\nblk}}(t)
  \end{pmatrix}_q}\Bigg)\Bigg]
  \end{align*} \tag{14}\label{14}
  $$

  $\eqref{14}$ 式就是論文的 3.12 式，其中 $1 \leq k \leq \nblk$ 且 $1 \leq q \leq \din + \nblk \times (3 + \dblk)$。

  由於 $\eqref{12}$ 的修改，$\eqref{9} \eqref{10}$ 最佳化的過程也需要跟著修改。

  輸入閘門的參數剩餘梯度改為

  $$
  \begin{align*}
  & \pd{\oploss(t + 1)}{\vWopig_{k, q}} \\
  & \aptr \sum_{i = 1}^{\dout} \Bigg[\big(y_i(t + 1) - \hat{y}_i(t + 1)\big) \cdot \dfnetout{i}{t + 1} \cdot \\
  & \quad \Bigg(\sum_{j = 1}^{\dblk} \wout_{i, \din + (k - 1) \cdot \dblk + j} \cdot y_k^{\opog}(t + 1) \cdot \dhblk{j}{k}{t + 1} \cdot \\
  & \quad \quad \br{y_k^{\opfg}(t + 1) \cdot \pd{s_j^{\blk{k}}(t)}{\vWopig_{k, q}} + \gnetblk{j}{k}{t + 1} \cdot \dfnetig{k}{t + 1} \cdot \begin{pmatrix}
  x(t) \\
  \vyopfg(t) \\
  y^{\opig}(t) \\
  y^{\opog}(t) \\
  y^{\blk{1}}(t) \\
  \vdots \\
  y^{\blk{\nblk}}(t)
  \end{pmatrix}_q}\Bigg)\Bigg]
  \end{align*} \tag{15}\label{15}
  $$

  $\eqref{14}$ 式就是論文的 3.11 式，其中 $1 \leq k \leq \nblk$ 且 $1 \leq q \leq \din + \nblk \times (3 + \dblk)$。

   memory cells 淨輸入參數的剩餘梯度改為

  $$
  \begin{align*}
  & \pd{\oploss(t + 1)}{\vWopblk{k}_{p, q}} \\
  & \aptr \sum_{i = 1}^{\dout} \Bigg[\big(y_i(t + 1) - \hat{y}_i(t + 1)\big) \cdot \dfnetout{i}{t + 1} \cdot \wout_{i, \din + (k - 1) \cdot \dblk + j} \cdot \\
  & \quad y_k^{\opog}(t + 1) \cdot \dhblk{j}{k}{t + 1} \cdot \\
  & \quad \br{y_k^{\opfg}(t + 1) \cdot \pd{s_p^{\blk{k}}(t)}{\vWopblk{k}_{p, q}} + y_k^{\opig}(t + 1) \cdot \dgnetblk{p}{k}{t + 1} \cdot \begin{pmatrix}
  x(t) \\
  \vyopfg(t) \\
  y^{\opig}(t) \\
  y^{\opog}(t) \\
  y^{\blk{1}}(t) \\
  \vdots \\
  y^{\blk{\nblk}}(t)
  \end{pmatrix}_q}\Bigg]
  \end{align*} \tag{16}\label{16}
  $$

  $\eqref{14}$ 式就是論文的 3.10 式，其中 $1 \leq k \leq \nblk$， $1 \leq p \leq \dblk$ 且 $1 \leq q \leq \din + \nblk \times (3 + \dblk)$。

  **注意錯誤**：根據論文中的 3.4 式，論文 2.5 式的 $t - 1$ 應該改成 $t$。

  根據 $\eqref{14}\eqref{15}\eqref{16}$，當 forget gate $y_k^{\opfg}(t + 1) \approx 0$ （關閉）時，不只 memory cells  $s^{\blk{k}}(t + 1)$ 會重設，與其相關的梯度也會重設，因此更新時需要額外紀錄以下的項次

  $$
  \pd{s_i^{\blk{k}}(t + 1)}{\wfg_{k, q}}, \pd{s_i^{\blk{k}}(t + 1)}{\vWopig_{k, q}}, \pd{s_i^{\blk{k}}(t + 1)}{\vWopblk{k}_{p, q}}
  $$

  同樣的概念在[原始 LSTM][LSTM1997] 中也有出現，細節可以看[我的筆記][note-LSTM1997]。

  ## 實驗 1：Continual Embedded Reber Grammar

  <a name="paper-fig-2"></a>

  圖 2：Continual Embedded Reber Grammar。
  圖片來源：[論文][論文]。

  ![圖 2](https://i.imgur.com/rhHtVRN.png)

  ### 任務定義

  - 根據[原始 LSTM 論文][LSTM1997]中的實驗 1（Embedded Reber Grammar）進行修改，輸入為連續序列，連續序列的定義是由多個 Embedded Reber Grammar 產生的序列組合而成（細節可以看[我的筆記][note-LSTM1997]）
  - 每個分支的生成機率值為 $0.5$
  - 當所有輸出單元的平方誤差低於 $0.49$ 時就當成預測正確
  - 在一次的訓練過程中，給予模型的輸入只會在以下兩種狀況之一發生時停止
    - 當模型產生一次的預測錯誤
    - 模型連續接收 $10^6$ 個輸入
  - 每次訓練停止就進行一次測試
    - 一次測試會執行 $10$ 次的連續輸入
    - 評估結果是 $10$ 次連續輸入的平均值
  - 每輸入一個訊號就進行更新（RTRL）
  - 訓練最多執行 $30000$ 次，實驗結果由 $100$ 個訓練模型實驗進行平均

  ### LSTM 架構

  <a name="paper-fig-3"></a>

  圖 3：LSTM 架構。
  圖片來源：[論文][論文]。

  ![圖 3](https://i.imgur.com/uUJjmSz.png)

  |參數|數值（或範圍）|備註|
  |-|-|-|
  |$\din$|$7$||
  |$\nblk$|$4$||
  |$\dblk$|$2$||
  |$\dout$|$7$||
  |$\dim(\vWopblk{k})$|$\dblk \times [\din + \nblk \cdot \dblk]$|訊號來源為外部輸入與 memory cells |
  |$\dim(\wfg)$|$\nblk \times [\din + \nblk \cdot \dblk + 1]$|訊號來源為外部輸入與 memory cells ，有額外使用bias term|
  |$\dim(\vWopig)$|$\nblk \times [\din + \nblk \cdot \dblk + 1]$|訊號來源為外部輸入與 memory cells ，有額外使用bias term|
  |$\dim(\vWopog)$|$\nblk \times [\din + \nblk \cdot \dblk + 1]$|訊號來源為外部輸入與 memory cells ，有額外使用bias term|
  |$\dim(\wout)$|$\dout \times [\din + \nblk \cdot \dblk + 1]$|訊號來源為外部輸入與 memory cells ，有額外使用bias term|
  |總參數量|$424$||
  |參數初始化|$[-0.2, 0.2]$|平均分佈|
  |輸入閘門bias term初始化|$\set{-0.5, -1.0, -1.5, -2.0}$|依序初始化成不同數值|
  |輸出閘門bias term初始化|$\set{-0.5, -1.0, -1.5, -2.0}$|依序初始化成不同數值|
  | forget gate bias term初始化|$\set{0.5, 1.0, 1.5, 2.0}$|依序初始化成不同數值|
  |Learning rate $\alpha$|$0.5$|訓練過程可以固定 $\alpha$，或是以 $0.99$ 的 decay factor 在每次更新後進行衰減|

  ### 實驗結果

  <a name="paper-fig-4"></a>

  圖 4：Continual Embedded Reber Grammar 實驗結果。
  圖片來源：[論文][論文]。

  ![圖 4](https://i.imgur.com/uu9Nccj.png)

  - [原始 LSTM][LSTM1997] 在有手動進行計算狀態的重置時表現非常好，但當沒有手動重置時完全無法執行任務
    - 就算讓 memory cell internal states 進行 decay 也無濟於事
  - 使用 forget gate 的 LSTM 不需要手動重置計算狀態也能達成完美預測
    - 完美預測指的是連續 $10^6$ 輸入都預測正確
  - 有嘗試使用 $\alpha / t$ 或 $\alpha / \sqrt{T}$ 作為 learning rate，實驗發現不論是哪種最佳化的方法使用 forget gate 的 LSTM 都表現的不錯
    - 在其他模型架構上（包含原版 LSTM）就算使用這些最佳化演算法也無法解決任務
  - 額外實驗在將 Embedded Reber Grammar 開頭的 `B` 與結尾的 `E` 去除的狀態下，使用 forget gate 的 LSTM 仍然表現不錯

  ### 分析

  <a name="paper-fig-5"></a>

  圖 5：[原版 LSTM][LSTM1997]  memory cell internal states 的累加值。
  圖片來源：[論文][論文]。

  ![圖 5](https://i.imgur.com/qwU4pnG.png)

  <a name="paper-fig-6"></a>

  圖 6：LSTM 加上 forget gate 後第三個 memory cell internal states 。
  圖片來源：[論文][論文]。

  ![圖 6](https://i.imgur.com/jtLnfu2.png)

  <a name="paper-fig-7"></a>

  圖 7：LSTM 加上 forget gate 後第一個 memory cell internal states 。
  圖片來源：[論文][論文]。

  ![圖 7](https://i.imgur.com/K1mp9rg.png)

  - 觀察[原版 LSTM][LSTM1997] 的 memory cell internal states ，可以發現在不進行手動重設的狀態下， memory cell internal states 的數值只會不斷的累加（朝向極正或極負前進）
  - 觀察架上 forget gate 後 LSTM 的 memory cell internal states ，可以發現模型學會自動重設
    - 在第三個 memory cells 中展現了長期記憶重設的能力
    - 在第一個 memory cells 中展現了短期記憶重設的能力

  ## 實驗 2：Noisy Temporal Order Problem

  ### 任務定義

  - 就是[原始 LSTM 論文][LSTM1997]中的實驗 6b，細節可以看[我的筆記][note-LSTM1997]
  - 由於此任務需要讓記憶維持一段不短的時間，因此遺忘資訊對於這個任務可能有害，透過這個任務想要驗證是否有任務是只能使用原版 LSTM 可以解決但增加 forget gate 後不能解決

  ### LSTM 架構

  與實驗 1 大致相同，只做以下修改

  - $\din = \dout = 8$
  - 將 forget gate 的bias term初始化成較大的正數（論文使用 $5$），讓 forget gate 很難被關閉，藉此達到跟原本 LSTM 幾乎相同的計算能力

  ### 實驗結果

  - 使用 forget gate 的 LSTM 仍然能夠解決 Noisy Temporal Order Problem
    - 當bias term初始化成較大的正數（例如 $5$）時，收斂速度與原版 LSTM 一樣快
    - 當bias term初始化成較小的正數（例如 $1$）時，收斂速度約為原版 LSTM 的 $3$ 倍
  - 因此根據實驗沒有什麼任務是原版 LSTM 可以解決但加上 forget gate 後不能解決的

  ## 實驗 3：Continual Noisy Temporal Order Problem

  ### 任務定義

  - 根據[原始 LSTM 論文][LSTM1997]中的實驗 6b 進行修改，輸入為連續序列，連續序列的定義是由 $100$ 筆 Noisy Temporal Order 序列所組成
  - 在一次的訓練過程中，給予模型的輸入只會在以下兩種狀況之一發生時停止
    - 當模型產生一次的預測錯誤
    - 模型連續接收 $100$ 個 Noisy Temporal Order 序列
  - 每次訓練停止就進行一次測試
    - 一次測試會執行 $10$ 次的連續輸入
    - 評估結果是 $10$ 次連續輸入中預測正確的序列個數平均值
  - 論文沒有講怎麼計算誤差與更新，我猜變成每個非預測時間點必須輸出 $0$，預測時間點時輸出預測結果
  - 訓練最多執行 $10^5$ 次，實驗結果由 $100$ 個訓練模型實驗進行平均

  ### LSTM 架構

  與實驗 2 相同。

  ### 實驗結果

  <a name="paper-fig-8"></a>

  圖 8：Continual Noisy Temporal Order Problem 實驗結果。
  圖片來源：[論文][論文]。

  ![圖 8](https://i.imgur.com/VV5wQVG.png)

  - [圖 8](#paper-fig-8) 中的註解 a 應該寫錯了，應該改為 correct classification of 100 successive NTO sequences
  - 實驗再次驗證原版 LSTM 無法解決連續輸入，但使用輸入閘門後就能夠解決問題
  - 將 learning rate 使用 decay factor $0.9$ 逐漸下降可以讓模型表現變更好，但作者認為這不重要

.. footbibliography::

.. ====================================================================================================================
.. external links
.. ====================================================================================================================

.. _Pytorch-LSTM: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html?highlight=lstm#torch.nn.LSTM
