========================================================================================================================
Learning Precise Timing with LSTM Recurrent Networks
========================================================================================================================

.. =====================================================================================================================
.. Set index for authors.
.. =====================================================================================================================

.. index::
  single: Felix A. Gers
  single: Jürgen Schmidhuber
  single: Fred Cummins

.. =====================================================================================================================
.. Set index for conference/journal.
.. =====================================================================================================================

.. index::
  single: Neural Computation

.. =====================================================================================================================
.. Set index for publishing time.
.. =====================================================================================================================

.. index::
  single: 2000

.. =====================================================================================================================
.. Setup SEO.
.. =====================================================================================================================

.. meta::
  :description:
    提出在 LSTM 上增加 forget gate
  :keywords:
    LSTM,
    RNN,
    Sequence Model,
    model architecture,
    neural network

.. =====================================================================================================================
.. Setup front matter.
.. =====================================================================================================================

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

    .. =================================================================================================================
    .. Define math macros. We put macros here so that user will not see them loading.
    .. =================================================================================================================

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
========================================================================================================================

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
========================================================================================================================
