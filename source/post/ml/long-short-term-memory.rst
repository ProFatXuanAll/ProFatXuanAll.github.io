======================
Long Short-Term Memory
======================

.. ====================================================================================================================
.. Set index for authors.
.. ====================================================================================================================

.. index::
  single: Sepp Hochreiter
  single: Jürgen Schmidhuber

.. ====================================================================================================================
.. Set index for conference/journal.
.. ====================================================================================================================

.. index::
  single: Neural Computation

.. ====================================================================================================================
.. Set index for publishing time.
.. ====================================================================================================================

.. index::
  single: 1997

.. ====================================================================================================================
.. Setup SEO.
.. ====================================================================================================================

.. meta::
  :description:
    提出 RNN 模型進行最佳化時遇到的問題，並提出新的模型架構「LSTM」與最佳化演算法「truncated RTRL」嘗試解決
  :keywords:
    BPTT,
    Sequence Model,
    Gradient Descent,
    Gradient Explosion,
    Gradient Vanishing,
    LSTM,
    Model Architecture,
    Optimization,
    RNN,
    RTRL

.. ====================================================================================================================
.. Setup front matter.
.. ====================================================================================================================

.. article-info::
  :author: Sepp Hochreiter, Jürgen Schmidhuber
  :date: Neural Computation, 1997
  :class-container: sd-p-2 sd-outline-muted sd-rounded-1

.. ====================================================================================================================
.. Create visible tags from SEO keywords.
.. ====================================================================================================================

:bdg-secondary:`BPTT`
:bdg-secondary:`Sequence Model`
:bdg-secondary:`Gradient Descent`
:bdg-secondary:`Gradient Explosion`
:bdg-secondary:`Gradient Vanishing`
:bdg-secondary:`LSTM`
:bdg-secondary:`Optimization`
:bdg-secondary:`Model Architecture`
:bdg-secondary:`RNN`
:bdg-secondary:`RTRL`
:bdg-primary:`Neural Computation`

.. ====================================================================================================================
.. Define math macros.
.. ====================================================================================================================

.. math::
  :nowrap:

  \[
    % Operators.
    \newcommand{\opblk}{\operatorname{block}}
    \newcommand{\opig}{\operatorname{ig}}
    \newcommand{\opin}{\operatorname{in}}
    \newcommand{\ophid}{\operatorname{hid}}
    \newcommand{\oplen}{\operatorname{len}}
    \newcommand{\opnet}{\operatorname{net}}
    \newcommand{\opog}{\operatorname{og}}
    \newcommand{\opout}{\operatorname{out}}
    \newcommand{\opseq}{\operatorname{seq}}

    % Memory cell blocks.
    \newcommand{\blk}[1]{{\opblk^{#1}}}

    % Vectors' notations.
    \newcommand{\vh}{\mathbf{h}}
    \newcommand{\vs}{\mathbf{s}}
    \newcommand{\vsopblk}[1]{\vs^\blk{#1}}
    \newcommand{\vw}{\mathbf{w}}
    \newcommand{\vx}{\mathbf{x}}
    \newcommand{\vxopout}{\vx^\opout}
    \newcommand{\vxt}{\tilde{\vx}}
    \newcommand{\vy}{\mathbf{y}}
    \newcommand{\vyh}{\hat{\vy}}
    \newcommand{\vyopblk}[1]{\vy^\blk{#1}}
    \newcommand{\vyopig}{\vy^\opig}
    \newcommand{\vyophid}{\vy^\ophid}
    \newcommand{\vyopog}{\vy^\opog}
    \newcommand{\vz}{\mathbf{z}}
    \newcommand{\vzopblk}[1]{\vz^\blk{#1}}
    \newcommand{\vzopig}{\vz^\opig}
    \newcommand{\vzophid}{\vz^\ophid}
    \newcommand{\vzopog}{\vz^\opog}
    \newcommand{\vzopout}{\vz^\opout}

    % Matrixs' notation.
    \newcommand{\vW}{\mathbf{W}}
    \newcommand{\vWopblk}[1]{\vW^\blk{#1}}
    \newcommand{\vWopig}{\vW^\opig}
    \newcommand{\vWophid}{\vW^\ophid}
    \newcommand{\vWopog}{\vW^\opog}
    \newcommand{\vWopout}{\vW^\opout}

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
    \newcommand{\vWii}{{\vW_{i, i}}}
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
    \newcommand{\din}{{d_\opin}}
    \newcommand{\dhid}{{d_\ophid}}
    \newcommand{\dout}{{d_\opout}}
    \newcommand{\dblk}{{d_\opblk}}
    \newcommand{\nblk}{{n_\opblk}}

    % Derivative of loss(#2) with respect to net input #1 at time #3.
    \newcommand{\vth}[2]{{\vartheta_{#1}^{#2}}}

    % Gradient approximation by truncating gradient.
    \newcommand{\aptr}{\approx_{\operatorname{tr}}}
  \]

重點
====

- 提出 :term:`RNN` 模型進行最佳化時遇到的問題，並提出新的模型架構「:term:`LSTM`」與最佳化演算法「truncated RTRL」嘗試解決

  - **梯度爆炸**\（:term:`gradient explosion`）\造成神經網路的\ **參數數值劇烈振盪**\（**oscillating weights**）
  - **梯度消失**\（:term:`gradient vanishing`）\造成\ **訓練時間慢長**
  - 關鍵輸入資訊\ **時間差較長**\（**long time lags**）導致模型無法處理資訊

- LSTM 架構設計

  - \ **記憶細胞區域**\（**memory cell blocks**）

    - 目標為解決關鍵輸入資訊時間差較長的問題
    - 必須配合閘門單元一起運作
    - 學習\ **協助**\閘門單元完成\ **寫入**/\ **讀取**\記憶細胞區域

  - 基於\ **乘法**\計算機制的\ **閘門單元**\（**multiplicative gate**）

    - 目標為解決關鍵輸入資訊時間差較長的問題
    - 提出兩種閘門單元：\ **輸入**\閘門單元（**input gate**）與\ **輸出**\閘門單元（**output gate**）
    - 輸\ **入**\閘門單元學習\ **寫入**\（\ **開啟**）/**保留**\（\ **關閉**）記憶細胞區域中的資訊
    - 輸\ **出**\閘門單元學習\ **讀取**\（\ **開啟**）/**忽略**\（\ **關閉**）記憶細胞區域中的資訊
    - 必須配合記憶細胞區域一起運作

  - **閘門單元參數**\中的\ **偏差項**\（**bias term**）必須\ **初始化**\成\ **負數**

    - 輸\ **入**\閘門偏差項初始化成負數能夠解決\ **內部狀態偏差行為**\（**internal state drift**）
    - 輸\ **出**\閘門偏差項初始化成負數能夠避免模型\ **濫用記憶細胞初始值**\與\ **訓練初期梯度過大**
    - 如果沒有輸出閘門，則\ **收斂速度會變慢**

- truncated-RTRL 最佳化演算法設計

  - 目標為\ **有效率**\的避免梯度\ **爆炸**\或\ **消失**
  - 以\ **捨棄計算部份梯度**\做為近似全微分的手段，因此只能使用 RTRL 而不能使用 BPTT
  - Backward pass 演算法\ **時間複雜度**\為 :math:`\order{w}`，:math:`w` 代表模型參數
  - Backward pass 演算法\ **空間複雜度**\也為 :math:`\order{w}`，因此\ **沒有輸入長度的限制**

- 根據實驗，LSTM 能夠達成以下任務

  - 能夠處理關鍵資訊時間差\ **短**\（**short time lag**）的任務
  - 能夠處理關鍵資訊時間差\ **長**\（**long time lag**）的任務
  - 能夠處理關鍵資訊時間差長達 1000 個單位的任務
  - 輸入訊號含有雜訊時也能處理

- LSTM 的缺點

  - 仍然無法解決 delayed XOR 問題

    - 改成以 BPTT 進行最佳化可能可以解決，但計算複雜度變高
    - CEC 在使用 BPTT 後有可能無效，但根據實驗使用 BPTT 時誤差傳遞的過程中很快就消失

  - 在部份任務上無法比 random weight guessing 最佳化速度還要快

    - 例如 500-bit parity
    - 使用 CEC 才導致此後果
    - 但計算效率高，最佳化過程也比較穩定

  - 無法精確的判斷重要訊號的輸入時間

    - 作者宣稱所有使用梯度下降作為最佳演算法的模型都有相同問題
    - 如果精確判斷是很重要的功能，則作者認為需要幫模型引入計數器的功能

- 當單一字元的\ **出現次數期望值增加**\時，**學習速度會下降**

  - 作者認為是常見字詞的出現導致參數開始振盪

- 此篇論文 :footcite:`hochreiter-etal-1997-long` 與 2000 年 :footcite:`gers-etal-2000-learning` 的後續延伸論文（以下稱為 LSTM-2000）都寫錯自己的數學公式，我的筆記內容將會嘗試進行勘誤
- 此篇論文與 `PyTorch <Pytorch-LSTM_>`_ 實作的 LSTM 完全不同

  - 本篇論文的架構定義更為\ **廣義**
  - 本篇論文只有輸入閘門跟輸出閘門，並沒有使用\ **失憶閘門**\（**Forget Gate**）\ :footcite:`gers-etal-2000-learning`

- Alex Graves 的 LSTM 教學：https://link.springer.com/chapter/10.1007/978-3-642-24797-2_4

此篇論文討論的 RNN
===================

類型定義
--------

:term:`RNN` 分成兩種：

- 隨著時間改變輸入（time-varying inputs）
- 不隨時間改變輸入（stationary inputs）

此論文討論的主要對象為隨著時間改變輸入的 RNN，計算定義請見 :doc:`BPTT </post/math/bptt>` 介紹，此篇筆記採用相同符號。

過往 RNN 模型的問題
-------------------

- 常用於 RNN 模型的最佳化演算法 :term:`BPTT` 與 :term:`RTRL` 都會遇到\ **梯度爆炸**\（:term:`gradient explosion`）或\ **梯度消失**\（:term:`gradient vanishing`）的問題

  - 梯度爆炸造成神經網路的\ **參數數值劇烈振盪**\（**oscillating weights**）
  - 梯度消失造成\ **訓練時間慢長**

- 關鍵輸入資訊\ **時間差較短**\（**short time lags**）的任務可以使用 time-delay neural network :footcite:`lang-etal-1990-a` 解決，但關鍵輸入資訊\ **時間差較長**\（**long time lags**）的任務並沒有好的解決方案

  - 已知的模型解決方案會隨著時間差越長導致模型所需參數越多
  - 已知的最佳化解決方案時間複雜度過高
  - 部份已知的測試任務可能過於簡單，甚至可依靠隨機參數猜測（random weight guessing）解決

梯度爆炸 / 消失
---------------

接下來我們將推導 RNN 模型產生\ **梯度爆炸**\與\ **梯度消失**\的原因。
為了方便討論，我們定義新的符號：

.. math::
  :nowrap:

  \[
    \vth{i_1, t_1}{i_2, t_2} = \dv{\frac{1}{2} \qty(\vy_{i_2}(t_2) - \vyh_{i_2}(t_2))^2}{\vz_{i_1}(t_1)}.
    \tag{1}\label{1}
  \]

意思是節點 :math:`\vz_{i_1}(t_1)` 透過輸出 :math:`\vy_{i_2}(t_2)` 貢獻的誤差計算所得之\ **微分**。

- 根據時間的限制我們有不等式 :math:`0 \leq t_1 \leq t_2 \leq \cT`
- 下標 :math:`i_1, i_2` 的數值範圍為 :math:`i_1, i_2 \in \Set{1, \dots, \dout}`，見 :doc:`RNN 計算定義 </post/math/bptt>`
- 式子 :math:`\eqref{1}` 採用 mean square error 作為誤差計算法，但其實可以採用任意的誤差計算法，不影響結論

對於任意 :math:`i_0 \in \Set{1, \dots, \dout}`，我們可以得出以下等式：

.. math::
  :nowrap:

  \[
    \begin{align*}
      \vth{i_1, t - 1}{i_0, t} & = \vth{i_0, t}{i_0, t} \cdot \qty[\prod_{q = 1}^1 \vW_{i_{q - 1}, i_q} \cdot f_{i_q}'\qty(\vz_{i_q}(t - q))]. \\
      \vth{i_2, t - 2}{i_0, t} & = \sum_{i_1 = 1}^\dout \vth{i_0, t}{i_0, t} \cdot \qty[\prod_{q = 1}^2 \vW_{i_{q - 1}, i_q} \cdot f_{i_q}'\qty(\vz_{i_q}(t - q))]. \\
      \vth{i_3, t - 3}{i_0, t} & = \sum_{i_2 = 1}^\dout \sum_{i_1 = 1}^\dout \vth{i_0, t}{i_0, t} \cdot \qty[\prod_{q = 1}^3 \vW_{i_{q - 1}, i_q} \cdot f_{i_q}'\qty(\vz_{i_q}(t - q))].
    \end{align*}
    \tag{2}\label{2}
  \]

.. dropdown:: 推導 :math:`\eqref{2}`

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \vth{i_1, t - 1}{i_0, t} & = \dv{\frac{1}{2} \qty(\vy_{i_0}(t) - \vyh_{i_0}(t))^2}{\vz_{i_1}(t - 1)} \\
                                 & = \dv{\frac{1}{2} \qty(\vy_{i_0}(t) - \vyh_{i_0}(t))^2}{\vz_{i_0}(t)} \cdot \dv{\vz_{i_0}(t)}{\vy_{i_1}(t - 1)} \cdot \dv{\vy_{i_1}(t - 1)}{\vz_{i_1}(t - 1)} \\
                                 & = \vth{i_0, t}{i_0, t} \cdot \vW_{i_0, i_1} \cdot f_{i_1}'\qty(\vz_{i_1}(t - 1)) \\
                                 & = \vth{i_0, t}{i_0, t} \cdot \qty[\prod_{q = 1}^1 \vW_{i_{q - 1}, i_q} \cdot f_{i_q}'\qty(\vz_{i_q}(t - q))]. \\
        \vth{i_2, t - 2}{i_0, t} & = \dv{\frac{1}{2} \qty(\vy_{i_0}(t) - \vyh_{i_0}(t))^2}{\vz_{i_2}(t - 2)} \\
                                 & = \sum_{i_1 = 1}^\dout \dv{\frac{1}{2} \qty(\vy_{i_0}(t) - \vyh_{i_0}(t))^2}{\vz_{i_1}(t - 1)} \cdot \dv{\vz_{i_1}(t - 1)}{\vy_{i_2}(t - 2)} \cdot \dv{\vy_{i_2}(t - 2)}{\vz_{i_2}(t - 2)} \\
                                 & = \sum_{i_1 = 1}^\dout \vth{i_1, t - 1}{i_0, t} \cdot \vW_{i_1, i_2} \cdot f_{i_2}'\qty(\vz_{i_2}(t - 2)) \\
                                 & = \sum_{i_1 = 1}^\dout \qty(\vth{i_0, t}{i_0, t} \cdot \qty[\prod_{q = 1}^1 \vW_{i_{q - 1}, i_q} \cdot f_{i_q}'\qty(\vz_{i_q}(t - q))]) \cdot \vW_{i_1, i_2} \cdot f_{i_2}'\qty(\vz_{i_2}(t - 2)) \\
                                 & = \sum_{i_1 = 1}^\dout \vth{i_0, t}{i_0, t} \cdot \qty[\prod_{q = 1}^2 \vW_{i_{q - 1}, i_q} \cdot f_{i_q}'\qty(\vz_{i_q}(t - q))]. \\
        \vth{i_3, t - 3}{i_0, t} & = \sum_{i_2 = 1}^\dout \dv{\frac{1}{2} \qty(\vy_{i_0}(t) - \vyh_{i_0}(t))^2}{\vz_{i_2}(t - 2)} \cdot \dv{\vz_{i_2}(t - 2)}{\vy_{i_3}(t - 3)} \cdot \dv{\vy_{i_3}(t - 3)}{\vz_{i_3}(t - 3)} \\
                                 & = \sum_{i_2 = 1}^\dout \vth{i_2, t - 2}{i_0, t} \cdot \vW_{i_2, i_3} \cdot f_{i_3}'\qty(\vz_{i_3}(t - 3)) \\
                                 & = \sum_{i_2 = 1}^\dout \qty(\sum_{i_1 = 1}^\dout \vth{i_0, t}{i_0, t} \cdot \qty[\prod_{q = 1}^2 \vW_{i_{q - 1}, i_q} \cdot f_{i_q}'\qty(\vz_{i_q}(t - q))]) \cdot \vW_{i_2, i_3} \cdot f_{i_3}'\qty(\vz_{i_3}(t - 3)) \\
                                 & = \sum_{i_2 = 1}^\dout \sum_{i_1 = 1}^\dout \vth{i_0, t}{i_0, t} \cdot \qty[\prod_{q = 1}^3 \vW_{i_{q - 1}, i_q} \cdot f_{i_q}'\qty(\vz_{i_q}(t - q))].
      \end{align*}
    \]

觀察式子 :math:`\eqref{2}`，我們可以歸納得出當 :math:`n \geq 1` 時，:math:`\vth{i_n, t - n}{i_0, t}` 的公式：

.. math::
  :nowrap:

  \[
    \vth{i_n, t - n}{i_0, t} = \sum_{i_{n - 1} = 1}^\dout \cdots \sum_{i_1 = 1}^\dout \vth{i_0, t}{i_0, t} \cdot \qty[\prod_{q = 1}^n \vW_{i_{q - 1}, i_q} \cdot f_{i_q}'\qty(\vz_{i_q}(t - q))].
    \tag{3}\label{3}
  \]

由 :math:`\eqref{3}` 我們可以看出對於任意 :math:`n \geq 1`，:math:`\vth{i_n, t - n}{i_0, t}` 都與 :math:`\vth{i_0, t}{i_0, t}` 相關。
因此當 :math:`\vth{i_0, t}{i_0, t}` 變動時，:math:`\vth{i_n, t - n}{i_0, t}` 也會\ **跟著變動**，這就是 :term:`back-propagation` 演算法的本質。

.. note::

  式子 :math:`\eqref{3}` 就是論文中的（3.1）與（3.2）式的來源。

接下來此論文將會以 :math:`\eqref{3}` 為出發點進行分析。
首先我們固定 :math:`i_0 \in \Set{1, \dots, \dout}`，並計算 :math:`\vth{i_0, t}{i_0, t}` 對於 :math:`\vth{i_n, t - n}{i_0, t}` 的微分，分析\ **微分結果**\在 back-propagation 過程中的\ **數值變化**：

.. math::
  :nowrap:

  \[
    \dv{\vth{i_n, t - n}{i_0, t}}{\vth{i_0, t}{i_0, t}} = \begin{dcases}
      \vW_{i_0, i_1} \cdot f_{i_1}'\qty(\vz_{i_1}(t - 1))                                                                                                             & \text{if } n = 1. \\
      \sum_{i_{n - 1} = 1}^\dout \cdots \sum_{i_1 = 1}^\dout \qty[\prod_{q = 1}^n \vW_{i_{q - 1}, i_q} \cdot f_{i_q}'\qty(\vz_{i_q}(t - q))] & \text{if } n > 1.
    \end{dcases}
    \tag{4}\label{4}
  \]

觀察可以發現，當 :math:`n > 1` 時，式子 :math:`\eqref{4}` 內共有 :math:`\dout^{n - 1}` 個連乘積項進行\ **加總**。
直覺上式子 :math:`\eqref{4}` 告訴我們，在這麼多個項次加總的狀況下，RNN 在 back-propagation 的過程中遞迴次數越多（:math:`n` 越大），微分數值\ **變化**\越大。
但其實該直覺不太正確，理由是每個連乘積項可能正負號不同，經過加法後可以互相抵銷。
因此後續的討論將會進行一些假設，進而推導出與直覺相符的結論。

.. note::

  :math:`\eqref{4}` 中的 :math:`n = 1` 就是論文中的（3.1）式，:math:`n > 1` 就是論文中的（3.2）式。

.. error::

  論文中（3.2）式最後乘法項次 :math:`w_{l_m l_{m - 1}}` 正確應為 :math:`w_{l_{m - 1} l_m}`，因此（3.2）應改成

  .. math::
    :nowrap:

    \[
      \dv{\vartheta_v(t - q)}{\vartheta_u(t)} = \sum_{l_1 = 1}^n \cdots \sum_{l_{q - 1} = 1}^n \prod_{m = 1}^q f'_{l_m}\qty(\opnet_{l_m}(t - m)) w_{l_{m - 1} l_m}.
    \]

假設式子 :math:`\eqref{4}` 中的 :math:`\dout^{n - 1}` 個加總項次中，**存在至少一個**\連乘積項 :math:`\prod_{q = 1}^n \vW_{i_{q - 1}, i_q} \cdot f_{i_q}'\qty(\vz_{i_q}(t - q))` 滿足以下條件：

.. math::
  :nowrap:

  \[
    \forall q \in \Set{1, \dots, n}, \abs{\vW_{i_{q - 1}, i_q} \cdot f_{i_q}'\qty(\vz_{i_q}(t - q))} > 1.0.
    \tag{5}\label{5}
  \]

則該連乘積項的\ **絕對值**\將隨著 :math:`n` 增加成\ **指數增長**，甚至數值可以大到 dominate 其他 :math:`\dout^{n - 1} - 1` 個連乘積項次。
這代表 back-propagation 過程中 RNN 遞迴的次數越多（i.e., :math:`n` 越大），微分數值\ **變化**\越大。
微分數值\ **變化大**\代表用來更新參數的微分值也\ **變大**\（以向量的角度來說，梯度的 norm 也變大），容易導致\ **梯度爆炸**，參數在使用 gradient descent 更新的過程中數值\ **劇烈振盪**，無法進行順利更新。
論文認為上述假設是可能發生的，例如當 :math:`f_{i_q}` 為線性函數時。

假設式子 :math:`\eqref{4}` 中的 :math:`\dout^{n - 1}` 個加總項次中，**所有**\連乘積項皆滿足以下條件：

.. math::
  :nowrap:

  \[
    \forall q \in \Set{1, \dots, n}, \abs{\vW_{i_{q - 1}, i_q} \cdot f_{i_q}'\qty(\vz_{i_q}(t - q))} < 1.0.
    \tag{6}\label{6}
  \]

則該連乘積項的\ **絕對值**\將隨著 :math:`n` 增加成\ **指數縮小**，甚至數值可以小到幾乎變成 :math:`0`。
這代表 back-propagation 過程中 RNN 遞迴的次數越多（i.e., :math:`n` 越大），微分數值\ **變化**\越小。
微分數值\ **變化小**\代表用來更新參數的微分值\ **接近常數**\（準確的說，微分值 :math:`\vth{i_n, t - n}{i_0, t}` 會接近 :math:`\vth{i_0, t}{i_0, t}`），而從更新的角度來看該常數值只能逼近 :math:`0`，因為學習的過程會讓誤差遞減成 :math:`0`，即 :math:`\vth{i_0, t}{i_0, t} \approx 0`。
此假設可以得出\ **梯度消失**\的結論，參數在使用 gradient descent 更新的過程中數值變化\ **非常緩慢**，無法進行順利更新。
論文認為上述假設是可能發生的，例如當 :math:`f_{i_q}` 為 sigmoid 函數 :math:`\sigma` 時。

我們知道 sigmoid 函數的微分 :math:`\sigma'` 最大值為 :math:`0.25`\（見 :doc:`sigmoid 函數特性 </post/math/sigmoid>`）。
因此當某些 :math:`q` 滿足 :math:`f_{i_{q}} = \sigma` 且 :math:`\abs{\vW_{i_{q - 1}, i_{q}}} < 4.0` 時，我們可以發現

.. math::
  :nowrap:

  \[
    \abs{\vW_{i_{q - 1}, i_{q}} \cdot \sigma'\qty(\vz_{i_{q}}(t - q))} < 4.0 \cdot 0.25 = 1.0.
    \tag{7}\label{7}
  \]

所以我們可以將 :math:`\eqref{6}` 的結論套用至 :math:`\eqref{7}` 的結果：當\ **所有** :math:`q` 都滿足 :math:`f_{i_q} = \sigma` 且 :math:`\abs{\vW_{i_{q - 1}, i_q}} < 4.0` 時會造成\ **梯度消失**。
而由於 sigmoid 常作為 activation function of RNN，並且訓練初期通常會將參數初始化至數值小於 :math:`1` 的狀態，因此梯度消失常見於 RNN 訓練過程。

根據上述討論，直覺上應該將參數初始值加大，但以下推論將會告訴我們加大參數初始值仍然會遇到梯度消失的問題。
假設某些 :math:`q` 滿足 :math:`\abs{\vW_{i_{q - 1}, i_{q}}} \to \infty`。
我們可以透過 sigmoid 函數特性推得：

.. math::
  :nowrap:

  \[
    \abs{\vW_{i_{q - 2}, i_{q - 1}} \cdot \sigma'\qty(\vz_{i_{q - 1}}(t - q + 1))} \to 0.
    \tag{8}\label{8}
  \]

.. dropdown:: 推導 :math:`\eqref{8}`

  .. math::
    :nowrap:

    \[
      \begin{align*}
                 & \abs{\vW_{i_{q - 1}, i_{q}} \cdot \mqty[\vx(t - q) \\ \vy(t - q)]_{i_{q}}} \to \infty \\
        \implies & \abs{\vz_{i_{q - 1}}(t - q + 1)} \to \infty \\
        \implies & \begin{dcases}
                     \sigma\qty(\vz_{i_{q - 1}}(t - q + 1)) \to 1 & \text{if } \vz_{i_{q - 1}}(t - q + 1) \to \infty \\
                     \sigma\qty(\vz_{i_{q - 1}}(t - q + 1)) \to 0 & \text{if } \vz_{i_{q - 1}}(t - q + 1) \to -\infty
                   \end{dcases} \\
        \implies & \sigma\qty(\vz_{i_{q - 1}}(t - q + 1)) \cdot \qty[1 - \sigma\qty(\vz_{i_{{q} - 1}}(t - q + 1))] \to 0 \\
        \implies & \sigma'\qty(\vz_{i_{q - 1}}(t - q + 1)) \to 0 \\
        \implies & \vW_{i_{q - 2}, i_{q - 1}} \cdot \sigma'\qty(\vz_{i_{q - 1}}(t - q + 1)) \to 0 \\
        \implies & \abs{\vW_{i_{q - 2}, i_{q - 1}} \cdot \sigma'\qty(\vz_{i_{q - 1}}(t - q + 1))} \to 0.
      \end{align*}
    \]

  最後一個推論的原理是 :math:`\sigma'\qty(\vz_{i_{q - 1}}(t - q + 1))` 因為指數函數，**收斂速度**\比線性函數 :math:`\vW_{i_{q - 2}, i_{q - 1}}` \ **快**。

因此我們可以再一次將 :math:`\eqref{6}` 的結論套用至 :math:`\eqref{8}` 的結果：
當部份參數初始值過大時，我們會遇到梯度消失的問題。

.. error::

  論文中關於 3.1.3 節最後一個段落的推論出發點

  .. math::
    :nowrap:

    \[
      \abs{f_{l_m}'\qty(\opnet_{l_m}) w_{l_m l_{m - 1}}}
    \]

  有幾點錯誤：

  - 作者少寫了時間參數，所以 :math:`\opnet_{l_m}` 應改為 :math:`\opnet_{l_m}(t - m)`
  - 作者不小心把時間先後順序寫反了，所以 :math:`w_{l_m l_{m - 1}}` 應改為 :math:`w_{l_{m - 1} l_m}`
  - 後續分析其實是基於 :math:`w_{l_m l_{m + 1}}`，所以 :math:`w_{l_{m - 1} l_m}` 應改為 :math:`w_{l_m l_{m + 1}}`

  全部更正後的寫法應為

  .. math::
    :nowrap:

    \[
      \abs{f_{l_m}'\qty(\opnet_{l_m}(t - m)) w_{l_m l_{m + 1}}}.
    \]

.. note::

  論文中進行了以下\ **函數最大值**\的推論：

  .. math::
    :nowrap:

    \[
      f_{l_m}'\qty(\opnet_{l_m}(t - m)) \cdot w_{l_m l_{m + 1}}.
    \]

  當 :math:`y^{l_{m + 1}}(t - m - 1)` 為非負常數時，前述函數最大值發生於

  .. math::
    :nowrap:

    \[
      w_{l_m l_{m + 1}} = \frac{1}{y^{l_{m + 1}}(t - m - 1)} \cdot \coth(\frac{1}{2} \opnet_{l_m}(t - m)).
    \]

  注意我已將前述錯誤修正，否則後續討論無意義。

  .. dropdown:: 推導最大值

    最大值發生於微分值為 :math:`0` 的點，即我們想求出滿足以下式子的 :math:`w_{l_m l_{m + 1}}`

    .. math::
      :nowrap:

      \[
        \dv{f_{l_m}'\qty(\opnet_{l_m}(t - m)) \cdot w_{l_m l_{m + 1}}}{w_{l_m l_{m + 1}}} = 0
      \]

    拆解微分式可得

    .. math::
      :nowrap:

      \[
        \begin{align*}
          & \dv{f_{l_m}'\qty(\opnet_{l_m}(t - m)) \cdot w_{l_m l_{m + 1}}}{w_{l_m l_{m + 1}}} \\
          & = \dv{f_{l_m}'\qty(\opnet_{l_m}(t - m))}{w_{l_m l_{m + 1}}} \cdot w_{l_m l_{m + 1}} + f_{l_m}'\qty(\opnet_{l_m}(t - m)) \cdot \dv{w_{l_m l_{m + 1}}}{w_{l_m l_{m + 1}}} \\
          & = \dv{f_{l_m}'\qty(\opnet_{l_m}(t - m))}{\opnet_{l_m}(t - m)} \cdot \dv{\opnet_{l_m}(t - m)}{w_{l_m l_{m + 1}}} \cdot w_{l_m l_{m + 1}} + f_{l_m}'\qty(\opnet_{l_m}(t - m)) \\
          & = f_{l_m}''\qty(\opnet_{l_m}(t - m)) \cdot y^{l_{m + 1}}(t - m - 1) \cdot w_{l_m l_{m + 1}} + f_{l_m}'\qty(\opnet_{l_m}(t - m)) \\
          & = \sigma''\qty(\opnet_{l_m}(t - m)) \cdot y^{l_{m + 1}}(t - m - 1) \cdot w_{l_m l_{m + 1}} + \sigma'\qty(\opnet_{l_m}(t - m)) \\
          & = \sigma\qty(\opnet_{l_m}(t - m)) \cdot \qty[1 - \sigma\qty(\opnet_{l_m}(t - m))] \cdot \qty[1 - 2\sigma\qty(\opnet_{l_m}(t - m))] \cdot y^{l_{m + 1}}(t - m - 1) \cdot w_{l_m l_{m + 1}} \\
          & \quad + \sigma\qty(\opnet_{l_m}(t - m)) \cdot \qty[1 - \sigma\qty(\opnet_{l_m}(t - m))].
        \end{align*}
      \]

    令上式等於 :math:`0` 後我們可以進行移項得到以下內容：

    .. math::
      :nowrap:

      \[
        \begin{align*}
                   & \sigma\qty(\opnet_{l_m}(t - m)) \cdot \qty[1 - \sigma\qty(\opnet_{l_m}(t - m))] \cdot \qty[1 - 2\sigma\qty(\opnet_{l_m}(t - m))] \cdot y^{l_{m + 1}}(t - m - 1) \cdot w_{l_m l_{m + 1}} \\
                   & \quad = -\sigma\qty(\opnet_{l_m}(t - m)) \cdot \qty[1 - \sigma\qty(\opnet_{l_m}(t - m))] \\
          \implies & \qty[1 - 2\sigma\qty(\opnet_{l_m}(t - m))] \cdot y^{l_{m + 1}}(t - m - 1) \cdot w_{l_m l_{m + 1}} = -1 \\
          \implies & w_{l_m l_{m + 1}} = \frac{1}{y^{l_{m + 1}}(t - m - 1)} \cdot \frac{1}{2\sigma\qty(\opnet_{l_m}(t - m)) - 1} \\
                   & = \frac{1}{y_{l_{m + 1}}(t - m - 1)} \cdot \coth(\frac{\opnet_{l_m}(t - m)}{2}).
        \end{align*}
      \]

    最後一段推論使用了以下公式

    .. math::
      :nowrap:

      \[
        \begin{align*}
          \tanh(x)           & = 2 \sigma(2x) - 1. \\
          \tanh(\frac{x}{2}) & = 2 \sigma(x) - 1. \\
          \coth(\frac{x}{2}) & = \frac{1}{\tanh(\frac{x}{2})} = \frac{1}{2 \sigma(x) - 1}.
        \end{align*}
      \]

由前述討論可以得出以下結論：

- 將參數初始化成過小的數值會導致梯度消失
- 將參數初始化成較大的數值會導致梯度爆炸
- 誤差傳遞遞迴次數越多（:math:`n` 越大），越容易導致梯度爆炸 / 消失

  - 代表 BPTT 對於時間差較短的資訊比較敏感
  - 在此狀態下增加 learning rate 也沒有用

- 將前述梯度消失的分析套用至總誤差仍然成立，推導如下：

  .. dropdown:: 推導

    :math:`\vz_{i_n}(t - n)` 對 :math:`t` 時間點的總誤差 :math:`\cL\qty(\vy(t), \vyh(t))` 微分可得：

    .. math::
      :nowrap:

      \[
        \begin{align*}
          \dv{\cL\qty(\vy(t), \vyh(t))}{\vz_{i_n}(t - n)} & = \dv{\sum_{i_0 = 1}^\dout \frac{1}{2} \qty(\vy_{i_0}(t) - \vyh_{i_0}(t))^2}{\vz_{i_n}(t - n)} \\
                                                          & = \sum_{i_0 = 1}^\dout \dv{\frac{1}{2} \qty(\vy_{i_0}(t) - \vyh_{i_0}(t))^2}{\vz_{i_n}(t - n)} \\
                                                          & = \sum_{i_0 = 1}^\dout \vth{i_n, t - n}{i_0, t}.
        \end{align*}
      \]

    觀察以下式子：

    .. math::
      :nowrap:

      \[
        \dv{\sum_{i_0 = 1}^\dout \vth{i_n, t - n}{i_0, t}}{\vth{i_0, t}{i_0, t}} = \sum_{i_0 = 1}^\dout \dv{\vth{i_n, t - n}{i_0, t}}{\vth{i_0, t}{i_0, t}}
      \]

    由於\ **每個項次** :math:`\dv{\vth{i_n, t - n}{i_0, t}}{\vth{i_0, t}{i_0, t}}` 都會遭遇梯度消失，因此\ **總和**\也會遭遇\ **梯度消失**。

解決梯度爆炸 / 消失的關鍵
=========================

觀察 1：自連接參數
------------------

首先我們針對式子 :math:`\eqref{3}` 中 RNN 模型透過自連接參數所得的微分值（即 :math:`i_{q - 1} = i_q`）進行探討，下標改以 :math:`i` 表示。
要如何避免透過自連接參數獲得的微分導致梯度爆炸 / 消失？
根據前述討論，我們的模型不能擁有以下條件：

.. math::
  :nowrap:

  \[
    \forall q \in \Set{1, \dots, n}, \begin{dcases}
      \abs{\vWii \cdot f_i'\qty(\vzi(t - q))} > 1.0 \\
      \abs{\vWii \cdot f_i'\qty(\vzi(t - q))} < 1.0
    \end{dcases}.
  \]

這代表我們的模型必須滿足以下條件：

.. math::
  :nowrap:

  \[
    \forall q \in \Set{1, \dots, n}, \abs{\vWii \cdot f_i'\qty(\vzi(t - q))} = 1.0.
    \tag{9}\label{9}
  \]

對式子 :math:`\eqref{9}` 左右兩側積分並移項，我們可以得到：

.. math::
  :nowrap:

  \[
    \forall q \in \Set{1, \dots, n}, f_i\qty(\vzi(t - q)) = \pm \frac{\vzi(t - q)}{\vWii}.
    \tag{10}\label{10}
  \]

式子 :math:`\eqref{10}` 告訴我們 :math:`f_i` 是一個線性函數。

.. dropdown:: 推導 :math:`\eqref{10}`

  .. math::
    :nowrap:

    \[
      \begin{align*}
                 & \abs{\vWii \cdot f_i'\qty(\vzi(t - q))} = \abs{\vWii \cdot \dv{f_i\qty(\vzi(t - q))}{\vzi(t - q)}} = 1.0 \\
        \implies & \vWii \cdot \dv{f_i\qty(\vzi(t - q))}{\vzi(t - q)} = \pm 1.0 \\
        \implies & \int \vWii \cdot \dv{f_i\qty(\vzi(t - q))}{\vzi(t - q)} \; d \vzi(t - q) = \pm \int 1.0 \; d \vzi(t - q) \\
        \implies & \vWii \cdot f_i\qty(\vzi(t - q)) = \pm \vzi(t - q) \\
        \implies & f_i\qty(\vzi(t - q)) = \pm \frac{\vzi(t - q)}{\vWii}.
      \end{align*}
    \]

如果我們進一步簡化模型，假設所有節點只會跟自己連接（即 :math:`\vzi(t + 1) = \vWii \cdot \vyi(t)`），則根據式子 :math:`\eqref{10}` 我們可以得出以下結論：

.. math::
  :nowrap:

  \[
    \vyi(t + 1) = f_i\qty(\vzi(t + 1)) = f_i\qty(\vWii \cdot \vyi(t)) = \pm \vyi(t).
    \tag{11}\label{11}
  \]

在不考慮負號的情況下，我們可以將 :math:`f_i` 設成 identity function 且設定 :math:`\vWii = 1.0` 從而滿足上述等式。
此論文認為，雖然模型並非只存在自連接節點，但若要讓自連接節點成功運作，可以透過 :math:`\eqref{11}` 推導得出 activation function 必須為 identity function，且 :math:`\vWii` 必須為 :math:`1.0` 的結論。
此論文將該結論稱為 **constant error carousel**\（**CEC**），並將 CEC 納入 LSTM 的核心設計。

觀察 2：輸入訊號衝突
--------------------

在計算的過程中，部份時間點的輸入資訊 :math:`\vxj(t)` 可能是\ **雜訊**，因此可以（甚至必須）被\ **忽略**。
但這代表與輸入相接的參數 :math:`\vWij` 需要\ **同時**\達成\ **兩種**\任務：

- **加入當前輸入**：代表 :math:`\abs{\vWij} \neq 0`
- **忽略當前輸入**：代表 :math:`\abs{\vWij} \approx 0`

因此\ **無法只靠一個** :math:`\vWij` 決定\ **當前輸入**\的影響，必須有\ **額外**\能夠\ **理解當前內容**\（**context-sensitive**）的功能模組幫忙決定是否\ **寫入** :math:`\vxj(t)`。
這便是此論文提出 **input gate units** 的原因。

觀察 3：輸出回饋到多個節點
--------------------------

在計算的過程中，部份時間點的輸出資訊 :math:`\vyi(t)` 可能對預測沒有幫助，因此可以（甚至必須）被\ **忽略**。
但這代表與輸出相接的參數 :math:`\vWij` 需要\ **同時**\達成\ **兩種**\任務：

- **保留過去輸出**：代表 :math:`\abs{\vWij} \neq 0`
- **忽略過去輸出**：代表 :math:`\abs{\vWij} \approx 0`

因此\ **無法只靠一個** :math:`\vWij` 決定\ **過去輸出**\的影響，必須有\ **額外**\能夠\ **理解當前內容**\（**context-sensitive**）的功能模組幫忙決定是否\ **讀取** :math:`\vyj(t)`。
這便是此論文提出 **output gate units** 的原因。

LSTM 架構
=========

.. figure:: https://i.imgur.com/uhS4AgH.png
  :alt: memory cell 內部架構
  :name: paper-fig-1

  圖 1：memory cell 內部架構。

  符號對應請見下個小節。
  圖片來源：:footcite:`hochreiter-etal-1997-long`。

.. figure:: https://i.imgur.com/UQ5LAu8.png
  :alt: LSTM 連接架構範例
  :name: paper-fig-2

  圖 2：LSTM 連接架構範例。

  線條真的多到讓人看不懂，看我整理過的公式比較好理解。
  圖片來源：:footcite:`hochreiter-etal-1997-long`。

為了解決梯度爆炸 / 消失問題，作者基於前述討論的結果，提出三個主要的機制，並將這些機制的合體稱為 **memory cells**：

- **Input gate units**：用於決定是否\ **更新** memory cell internal states
- **Output gate units**：用於決定是否\ **輸出** memory cell block activations
- **Central linear unit with fixed self-connection**：概念來自於 CEC（見 :math:`\eqref{11}`），藉此保障\ **梯度不會消失**

符號定義
--------

+------------------------+-------------------------------------------------------------------------------+----------------------+
| Symbol                 | Meaning                                                                       | Value Range          |
+========================+===============================================================================+======================+
| :math:`\dhid`          | Number of conventional hidden units at time step :math:`t`.                   | :math:`\N`           |
+------------------------+-------------------------------------------------------------------------------+----------------------+
| :math:`\dblk`          | Number of memory cells in each memory cell block at time step :math:`t`.      | :math:`\Z^+`         |
+------------------------+-------------------------------------------------------------------------------+----------------------+
| :math:`\nblk`          | Number of memory cell blocks at time step :math:`t`.                          | :math:`\Z^+`         |
+------------------------+-------------------------------------------------------------------------------+----------------------+
| :math:`\vx(t)`         | LSTM input at time step :math:`t`.                                            | :math:`\R^\din`      |
+------------------------+-------------------------------------------------------------------------------+----------------------+
| :math:`\vyophid(t)`    | Conventional hidden units at time step :math:`t`.                             | :math:`\R^\dhid`     |
+------------------------+-------------------------------------------------------------------------------+----------------------+
| :math:`\vyopig(t)`     | Input gate units at time step :math:`t`.                                      | :math:`[0, 1]^\nblk` |
+------------------------+-------------------------------------------------------------------------------+----------------------+
| :math:`\vyopog(t)`     | Output gate units at time step :math:`t`.                                     | :math:`[0, 1]^\nblk` |
+------------------------+-------------------------------------------------------------------------------+----------------------+
| :math:`\vyopblk{k}(t)` | Output of the :math:`k`-th memory cell block at time step :math:`t`.          | :math:`\R^\dblk`     |
+------------------------+-------------------------------------------------------------------------------+----------------------+
| :math:`\vsopblk{k}(t)` | Internal states of the :math:`k`-th memory cell block at time step :math:`t`. | :math:`\R^\dblk`     |
+------------------------+-------------------------------------------------------------------------------+----------------------+
| :math:`\vy(t)`         | LSTM output at time step :math:`t`.                                           | :math:`\R^\dout`     |
+------------------------+-------------------------------------------------------------------------------+----------------------+

計算定義
--------

以下就是 LSTM（1997 版本）的計算流程（見論文 4.1 節）。

.. math::
  :nowrap:

  \[
    \begin{align*}
      & \algoProc{\operatorname{LSTM1997}}(\vx, \vWophid, \vWopig, \vWopog, \vWopblk{1}, \dots, \vWopblk{\nblk}, \vWopout) \\
      & \indent{1} \algoCmt{Initialize activations with zeros.} \\
      & \indent{1} \cT \algoEq \oplen(\vx) \\
      & \indent{1} \vyophid(0) \algoEq \zv \\
      & \indent{1} \vyopig(0) \algoEq \zv \\
      & \indent{1} \vyopog(0) \algoEq \zv \\
      & \indent{1} \algoFor{k \in \Set{1, \dots, \nblk}} \\
      & \indent{2}   \vyopblk{k}(0) \algoEq \zv \\
      & \indent{1} \algoEndFor \\
      & \indent{1} \algoCmt{Do forward pass.} \\
      & \indent{1} \algoFor{t \in \Set{0, \dots, \cT - 1}} \\
      & \indent{2}   \algoCmt{Concatenate input units with activations.} \\
      & \indent{2}   \vxt(t) \algoEq \begin{pmatrix}
                                       \vx(t) \\
                                       \vyophid(t) \\
                                       \vyopig(t) \\
                                       \vyopog(t) \\
                                       \vyopblk{1}(t) \\
                                       \vdots \\
                                       \vyopblk{\nblk}(t)
                                     \end{pmatrix} \\
      & \indent{2}   \algoCmt{Compute conventional hidden units' activations.} \\
      & \indent{2}   \vzophid(t + 1) \algoEq \vWophid \cdot \vxt(t) \\
      & \indent{2}   \vyophid(t + 1) \algoEq f^\ophid\qty(\vzophid(t + 1)) \\
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
      & \indent{2}   \algoCmt{Compute outputs.} \\
      & \indent{2}   \vxopout(t + 1) \algoEq \begin{pmatrix}
                                               \vx(t) \\
                                               \vyophid(t + 1) \\
                                               \vyopblk{1}(t + 1) \\
                                               \vdots \\
                                               \vyopblk{\nblk}(t + 1) \\
                                             \end{pmatrix} \\
      & \indent{2}   \vzopout(t + 1) \algoEq \vWopout \cdot \vxopout(t + 1) \\
      & \indent{2}   \vy(t + 1) \algoEq f^\opout\qty(\vzopout(t + 1)) \\
      & \indent{1} \algoEndFor \\
      & \indent{1} \algoReturn \vy(1), \dots, \vy(\cT) \\
      & \algoEndProc
    \end{align*}
  \]

Memory Cell Blocks and Memory Cells
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Memory cells 的主要功能為記憶過去的輸入資訊。

- 在 :math:`t` 時間點時，一個 LSTM 模型有 :math:`\nblk` 個 memory cell blocks :math:`\vyopblk{1}(t), \dots, \vyopblk{\nblk}(t)`
- 在 :math:`t` 時間點時，第 :math:`k` 個 memory cell block :math:`\vyopblk{k}(t)` 內有 :math:`\dblk` 個 memory cells :math:`\vyopblk{k}_1(t), \dots, \vyopblk{k}_\dblk(t)`
- 例如：:ref:`paper-fig-2`

  - 共有 :math:`2` 個不同的 memory cell blocks
  - 每個 memory cell block 中包含 :math:`2` 個 memory cells

Input Gate Units
~~~~~~~~~~~~~~~~

Input gate units 決定與控制計算資訊是否需要流入 memory cells，LSTM 以此設計避免因輸入訊號衝突造成的參數更新矛盾。

- Input gate units :math:`\vyopig(t + 1)` 是以\ **乘法**\參與計算，因此稱為 **multiplicative gate units**

  - Memory cells in the same memory cell block **share** the same input gate unit（見論文 4.4 節）
  - 因此 :math:`\vyopig_k(t + 1) \cdot g\qty(\vzopblk{k}(t + 1))` 中的乘法是\ **純量乘上向量**

- 當模型認為 :math:`t` 時間點的計算資訊 :math:`\vxt(t)` **不重要**\時，模型應該要\ **關閉 input gate units**

  - 更準確的說，當模型認為 :math:`g\qty(\vzopblk{k}(t))` 對第 :math:`k` 個 memory cell block 來說不重要時，模型應該要關閉第 :math:`k` 個 input gate unit，即 :math:`\vyopig_k(t + 1) \approx 0`
  - 關閉 input gate units 代表丟棄當前輸入訊號，只以\ **過去資訊**\進行決策
  - 在此狀態下 memory cell internal states :math:`\vsopblk{k}(t + 1)` 與 :math:`\vsopblk{k}(t)` 時間點\ **完全相同**，達成 CEC（見 :math:`\eqref{11}`），藉此保障\ **梯度不會消失**
  - 不論 :math:`g\qty(\vzopblk{k}(t + 1))` 的大小，只要 :math:`\vyopig_k(t + 1) \approx 0`，則計算資訊 :math:`\vxt(t)` **完全無法影響**\接下來的所有計算

- 當模型認為 :math:`t` 時間點的計算資訊 :math:`\vxt(t)` **重要**\時，模型應該要\ **開啟 input gate units**

  - 更準確的說，當模型認為 :math:`g\qty(\vzopblk{k}(t))` 對第 :math:`k` 個 memory cell block 來說重要時，模型應該要開啟第 :math:`k` 個 input gate unit，即 :math:`\vyopig_k(t + 1) \approx 1`

- 我的 :math:`\vyopig_j(t + 1)` 是對應到論文中的 :math:`y^{\opin_j}(t + 1)`，見論文 4.1 節
- 例如：:ref:`paper-fig-2`

  - Memory cells ``cell 1`` and ``cell 2`` in memory cell block ``block 1`` 共享 input gate unit ``in 1``
  - Memory cells ``cell 1`` and ``cell 2`` in memory cell block ``block 2`` 共享 input gate unit ``in 2``

Memory Cell Internal States
~~~~~~~~~~~~~~~~~~~~~~~~~~~

將 CEC 融入 LSTM 的主要機制。

- 有時簡稱 memory cell internal states 為 internal states
- 更新 internal states 的唯一管道是計算資訊 :math:`\vxt(t)`
- 更新 internal states 的決策取決於 input gate units :math:`\vyopig(t + 1)`
- 由於第 :math:`k` 個 memory cell blocks 中的 internal states :math:`\vsopblk{k}(t + 1)` 主要只與第 :math:`k` 個 internal states :math:`\vsopblk{k}(t)` 連接，因此稱為 **fixed self-connection**
- 由於第 :math:`k` 個 memory cell blocks 中的 internal states :math:`\vsopblk{k}(t + 1)` 是透過加法與 :math:`\vsopblk{k}(t)` 結合，因此稱為 central **linear** unit
- 我的 :math:`\vsopblk{k}_j(t + 1)` 是對應到論文中的 :math:`s_{c_j}(t + 1)`，見論文 4.1 節

Output Gate Units
~~~~~~~~~~~~~~~~~

Output gate units 決定與控制 memory cell block activations 是否需要用於當前輸出與未來資訊的計算，LSTM 以此設計避免因輸出訊號衝突造成的參數更新矛盾。

- Output gate units :math:`\vyopog(t + 1)` 是以\ **乘法**\參與計算，因此稱為 **multiplicative gate units**

  - Memory cells in the same memory cell block **share** the same output gate unit（見論文 4.4 節）
  - 因此 :math:`\vyopog_k(t + 1) \cdot h\qty(\vsopblk{k}(t + 1))` 中的乘法是\ **純量乘上向量**

- 當模型認為 :math:`t + 1` 時間點的 internal state activations 會導致\ **當前計算錯誤**\時，模型應該\ **關閉 output gate units**

  - 更準確的說，當模型認為 :math:`h\qty(\vsopblk{k}(t + 1))` 對後續計算來說不重要時，模型應該要關閉第 :math:`k` 個 output gate unit，即 :math:`\vyopog_k(t + 1) \approx 0`
  - 在 **input gate units 開啟**\的狀況下，**關閉 output gate units** 代表不讓\ **現在**\時間點的資訊影響當前計算
  - 在 **input gate units 關閉**\的狀況下，**關閉 output gate units** 代表不讓\ **過去**\時間點的資訊影響當前計算
  - 不論 :math:`h\qty(\vsopblk{k}(t + 1))` 的大小，只要 :math:`\vyopog_k(t + 1) \approx 0`，則 internal states :math:`\vsopblk{k}(t + 1)` **完全無法影響**\接下來的所有計算

- 當模型認為 :math:`t + 1` 時間點的 internal state activations 包含\ **重要資訊**\時，模型應該\ **開啟 output gate units**

  - 更準確的說，當模型認為 :math:`h\qty(\vsopblk{k}(t + 1))` 對後續計算來說很重要時，模型應該要開啟第 :math:`k` 個 output gate unit，即 :math:`\vyopog_k(t + 1) \approx 1`
  - 在 **input gate units 開啟**\的狀況下，**開啟 output gate units** 代表讓\ **現在**\時間點的資訊影響當前計算
  - 在 **input gate units 關閉**\的狀況下，**開啟 output gate units** 代表不讓\ **過去**\時間點的資訊影響當前計算

- 我的 :math:`\vyopog_j(t + 1)` 是對應到論文中的 :math:`y^{\opout_j}(t + 1)`，見論文 4.1 節
- `PyTorch 實作的 LSTM <Pytorch-LSTM_>`_ 中 :math:`h(t)` 表達的意思是 memory cell block activation :math:`\vyopblk{k}(t)`
- 例如：:ref:`paper-fig-2`

  - Memory cells ``cell 1`` and ``cell 2`` in memory cell block ``block 1`` 共享 output gate unit ``out 1``
  - Memory cells ``cell 1`` and ``cell 2`` in memory cell block ``block 2`` 共享 output gate unit ``out 2``

Activation Functions
~~~~~~~~~~~~~~~~~~~~

- :math:`f^\ophid, f^\opig, f^\opog, f^\opout, g, h` 都是 differentiable element-wise activation function，大部份都是 sigmoid 或是 sigmoid 的變形
- :math:`f^\opig, f^\opog` 的數值範圍（range）必須限制在 :math:`[0, 1]`，才能達成 multiplicative gate 的功能
- :math:`f^\opout` 的數值範圍只跟任務有關
- 論文並沒有給 :math:`f^\ophid, g, h` 任何數值範圍的限制

Hidden Units
~~~~~~~~~~~~

- 作者將此論文新定義的 input/output gate units 與 memory cells 稱為 hidden units（見論文 4.3 節）
- 作者將 :math:`\vyophid(t)` 稱為 conventional hidden units，因此當我說到 hidden units 時泛指 gate units、memory cells 與 conventional hidden units
- 可以將 conventional hidden units 與 LSTM 視為平行的機制
- Hidden layer 由 hidden units 組成
- 此論文的後續研究都基於此論文 hidden layer 的設計進行改良，例如 LSTM-2000 :footcite:`gers-etal-2000-learning` 與 LSTM-2002 :footcite:`gers-etal-2002-learning`
- Hidden units 的設計等同於\ **保留** 造成梯度爆炸 / 消失的架構，是個不好的設計，因此論文後續在\ **最佳化**\的過程中動了手腳
- 所有 hidden units 全部\ **初始化**\成\ **零向量**，也就是 :math:`t = 0` 時模型\ **所有節點**\（除了輸入 :math:`\vx(0)`）都是 :math:`0`

節點連接機制
~~~~~~~~~~~~

- Input layer 會與 hidden layer 直接連接
- Input layer 也會與 output layer 直接連接
- Hidden layer 會與 output layer 連接
- 但 gate units 不會與 output layer 連接

.. pull-quote::

  ... **All units** (except for gate units) in all layers have **directed** connections (serve as input) to **all units** in the **layer above** ...

.. error::

  根據論文 A.7 式下方的描述

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \opnet_k(t) &= \sum_{u : u \text{ not a gate}} w_{ku} y^u(t - 1). \\
        y^k(t)      & = f_k\qty(\opnet_k(t)).
      \end{align*}
    \]

  代表 :math:`t + 1` 時間點的\ **輸出**\只與 :math:`t` 時間點的計算結果有關係，並\ **沒有**\包含 :math:`t + 1` 時間點的 memory cell block activations。
  所以計算 :math:`t + 1` 時間點的 memory cell block activations 都只是在幫助 :math:`t + 2` 時間點的計算狀態\ **鋪陳**。
  我不確定這是否為作者的筆誤，畢竟論文 appendix 中所有分析的數學式都寫的蠻正確的。
  但我仍然認為這裡是筆誤，理由如下：

  - 同個實驗室後續的研究（例如 :footcite:`gers-etal-2002-learning`）寫的式子不同
  - Memory cell block activations 至少要傳播 :math:`2` 個時間點才能影響輸出，代表第 :math:`1` 個時間點的輸出完全無法利用到 memory cell 的資訊
  - 後續的實驗架構設計中沒有將 input layer 連接到 output layer，代表第 :math:`1` 個時間點的輸出完全依賴模型的初始狀態（常數），非常不合理

  因此我決定改用我認為是正確的版本撰寫後續的筆記，即 :math:`t + 1` 時間點的\ **輸出**\與 :math:`t` 時間點的 memory cell block activations **有關**。

.. note::

  注意在計算 input/output gate units 時並\ **沒有**\使用 **bias term**，但後續的分析會提到可以使用 bias term 進行\ **計算缺陷**\的修正。

參數結構
--------

+---------------------+--------------------------------------------------------------+---------------------+-------------------------------------------------+
| Parameter           | Meaning                                                      | Output Vector Shape | Input Vector Shape                              |
+=====================+==============================================================+=====================+=================================================+
| :math:`\vWophid`    | Weight matrix connect to conventional hidden units.          | :math:`\dhid`       | :math:`\din + \dhid + \nblk \times (2 + \dblk)` |
+---------------------+--------------------------------------------------------------+---------------------+-------------------------------------------------+
| :math:`\vWopig`     | Weight matrix connect to input gate units.                   | :math:`\nblk`       | :math:`\din + \dhid + \nblk \times (2 + \dblk)` |
+---------------------+--------------------------------------------------------------+---------------------+-------------------------------------------------+
| :math:`\vWopog`     | Weight matrix connect to output gate units.                  | :math:`\nblk`       | :math:`\din + \dhid + \nblk \times (2 + \dblk)` |
+---------------------+--------------------------------------------------------------+---------------------+-------------------------------------------------+
| :math:`\vWopblk{k}` | Weight matrix connect to the :math:`k`-th memory cell block. | :math:`\dblk`       | :math:`\din + \dhid + \nblk \times (2 + \dblk)` |
+---------------------+--------------------------------------------------------------+---------------------+-------------------------------------------------+
| :math:`\vWopout`    | Weight matrix connect to output layer.                       | :math:`\dblk`       | :math:`\din + \dhid + \nblk \times \dblk`       |
+---------------------+--------------------------------------------------------------+---------------------+-------------------------------------------------+

LSTM 最佳化
===========

過去的論文中提出以\ **修改最佳化過程**\避免 RNN 訓練遇到\ **梯度爆炸 / 消失**\的問題（例如 Truncated BPTT）。
作者在論文 4.5 節提出\ **最佳化** LSTM 的方法為 **RTRL 的變種**，主要精神如下：

- 最佳化的核心思想是確保能夠達成 **CEC** （見 :math:`\eqref{11}`）

  - Internal states 透過遞迴過程傳遞的微分不會被放大或縮小
  - 因此不會造成梯度爆炸 / 消失

- 使用的手段是要求 **back propagation** 的過程在經過 **hidden units** 後便\ **停止**\計算微分

  - 微分會傳遞至產生 conventional hidden units 的參數 :math:`\vWophid`
  - 微分不會傳遞至產生 conventional hidden units 的節點 :math:`\vxt(t)`
  - 微分會傳遞至產生 output gate units 的參數 :math:`\vWopog`
  - 微分不會傳遞至產生 output gate units 的節點 :math:`\vxt(t)`
  - Internal states 收到的微分會經由 output gate units 縮放
  - 微分會傳遞至產生 input gate units 的參數 :math:`\vWopig`
  - 微分不會傳遞至產生 input gate units 的節點 :math:`\vxt(t)`
  - 微分會傳遞至產生 internal states 的參數 :math:`\vWopblk{1}, \dots, \vWopblk{\nblk}`
  - 微分不會傳遞至產生 internal states 的節點 :math:`\vxt(t)`
  - :math:`\vWopblk{1}, \dots, \vWopblk{\nblk}` 收到的微分會經由 input gate units 縮放

- 停止 back propagation 導致在完成 :math:`t + 1` 時間點的 forward pass 後可以\ **即時計算**\參數對 :math:`t + 1` 時間點誤差計算所得微分（real time 的精神便是來自於此）

首先我們定義新的符號 :math:`\aptr`，代表計算\ **微分**\的過程會有\ **部份微分**\故意被\ **丟棄**\（設定為 :math:`0`），並以丟棄結果\ **近似**\真正的\ **全微分**。
此論文將所有與 **hidden units** 相連的節點 :math:`\vxt(t)` 產生的微分值一律\ **丟棄**。

.. math::
  :nowrap:

  \[
    \begin{align*}
      \dv{\vzophid_i(t + 1)}{\vxt_j(t)}    & \aptr 0 \qqtext{where} \begin{dcases}
                                                                      i \in \Set{1, \dots, \dhid} \\
                                                                      j \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                      t \in \Set{0, \dots, \cT - 1}
                                                                    \end{dcases}. \\
      \dv{\vzopig_k(t + 1)}{\vxt_j(t)}     & \aptr 0 \qqtext{where} \begin{dcases}
                                                                      j \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                      k \in \Set{1, \dots, \nblk} \\
                                                                      t \in \Set{0, \dots, \cT - 1}
                                                                    \end{dcases}. \\
      \dv{\vzopog_k(t + 1)}{\vxt_j(t)}     & \aptr 0 \qqtext{where} \begin{dcases}
                                                                      j \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                      k \in \Set{1, \dots, \nblk} \\
                                                                      t \in \Set{0, \dots, \cT - 1}
                                                                    \end{dcases}. \\
      \dv{\vzopblk{k}_i(t + 1)}{\vxt_j(t)} & \aptr 0 \qqtext{where} \begin{dcases}
                                                                      i \in \Set{1, \dots, \dblk} \\
                                                                      j \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                      k \in \Set{1, \dots, \nblk} \\
                                                                      t \in \Set{0, \dots, \cT - 1}
                                                                    \end{dcases}. \\
      \dv{\vsopblk{k}_i(t)}{\vxt_j(t)}     & \aptr 0 \qqtext{where} \begin{dcases}
                                                                      i \in \Set{1, \dots, \dblk} \\
                                                                      j \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                      k \in \Set{1, \dots, \nblk} \\
                                                                      t \in \Set{0, \dots, \cT - 1}
                                                                    \end{dcases}.
    \end{align*}
    \tag{12}\label{12}
  \]

.. note::

  注意論文在 A.1.2 節的開頭只提到 **input gate units**、**output gate units**、**memory cells** 要\ **丟棄微分值**，但論文在 A.9 式描述可以將 **conventional hidden units** 的微分一起\ **丟棄**，害我白白推敲公式好幾天。

  .. pull-quote::

    ... Here it would be possible to use the full gradient without affecting constant error flow through internal states of memory cells. ...

.. error::

  論文中沒有描述到 :math:`\dv{\vsopblk{k}_i(t)}{\vxt_j(t)} \aptr 0`，但在 A.1.2 節卻使用了該項近似，才有辦法透過式子 :math:`\eqref{12}` 推出式子 :math:`\eqref{13}`。

根據 :math:`\eqref{12}` 我們可以進一步推得

.. math::
  :nowrap:

  \[
    \begin{align*}
      \dv{\vyophid_i(t + 1)}{\vxt_j(t)}    & \aptr 0 \qqtext{where} \begin{dcases}
                                                                      i \in \Set{1, \dots, \dhid} \\
                                                                      j \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                      t \in \Set{0, \dots, \cT - 1}
                                                                    \end{dcases}. \\
      \dv{\vyopig_k(t + 1)}{\vxt_j(t)}     & \aptr 0 \qqtext{where} \begin{dcases}
                                                                      j \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                      k \in \Set{1, \dots, \nblk} \\
                                                                      t \in \Set{0, \dots, \cT - 1}
                                                                    \end{dcases}. \\
      \dv{\vyopog_k(t + 1)}{\vxt_j(t)}     & \aptr 0 \qqtext{where} \begin{dcases}
                                                                      j \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                      k \in \Set{1, \dots, \nblk} \\
                                                                      t \in \Set{0, \dots, \cT - 1}
                                                                    \end{dcases}. \\
      \dv{\vsopblk{k}_i(t + 1)}{\vxt_j(t)} & \aptr 0 \qqtext{where} \begin{dcases}
                                                                      i \in \Set{1, \dots, \dblk} \\
                                                                      j \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                      k \in \Set{1, \dots, \nblk} \\
                                                                      t \in \Set{0, \dots, \cT - 1}
                                                                    \end{dcases}. \\
      \dv{\vyopblk{k}_i(t + 1)}{\vxt_j(t)} & \aptr 0 \qqtext{where} \begin{dcases}
                                                                      i \in \Set{1, \dots, \dblk} \\
                                                                      j \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                      k \in \Set{1, \dots, \nblk} \\
                                                                      t \in \Set{0, \dots, \cT - 1}
                                                                    \end{dcases}.
    \end{align*}
    \tag{13}\label{13}
  \]

.. dropdown:: 推導 :math:`\eqref{13}`

  首先根據式子 :math:`\eqref{12}` 的定義可以得到以下公式：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\vyophid_i(t + 1)}{\vxt_j(t)} & = \dv{\vyophid_i(t + 1)}{\vzophid_i(t + 1)} \cdot \cancelto{0}{\dv{\vzophid_i(t + 1)}{\vxt_j(t)}} \\
                                          & \aptr 0 \qqtext{where} \begin{dcases}
                                                                     i \in \Set{1, \dots, \dhid} \\
                                                                     j \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                     t \in \Set{0, \dots, \cT - 1}
                                                                   \end{dcases}. \\
        \dv{\vyopig_k(t + 1)}{\vxt_j(t)}  & = \dv{\vyopig_k(t + 1)}{\vzopig_k(t + 1)} \cdot \cancelto{0}{\dv{\vzopig_k(t + 1)}{\vxt_j(t)}} \\
                                          & \aptr 0 \qqtext{where} \begin{dcases}
                                                                     j \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                     k \in \Set{1, \dots, \nblk} \\
                                                                     t \in \Set{0, \dots, \cT - 1}
                                                                   \end{dcases}. \\
        \dv{\vyopog_k(t + 1)}{\vxt_j(t)}  & = \dv{\vyopog_k(t + 1)}{\vzopog_k(t + 1)} \cdot \cancelto{0}{\dv{\vzopog_k(t + 1)}{\vxt_j(t)}} \\
                                          & \aptr 0 \qqtext{where} \begin{dcases}
                                                                     j \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                     k \in \Set{1, \dots, \nblk} \\
                                                                     t \in \Set{0, \dots, \cT - 1}
                                                                   \end{dcases}.
      \end{align*}
    \]

  接著利用上述的結果結合 :math:`\eqref{12}` 推導出與 memory cells 相關的微分近似結果：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\vsopblk{k}_i(t + 1)}{\vxt_j(t)} & = \cancelto{0}{\dv{\vsopblk{k}_i(t)}{\vxt_j(t)}} + \cancelto{0}{\dv{\vyopig_k(t + 1)}{\vxt_j(t)}} \cdot g\qty(\vzopblk{k}_i(t + 1)) + \vyopig_k(t + 1) \cdot \dv{g\qty(\vzopblk{k}_i(t + 1))}{\vzopblk{k}_i(t + 1)} \cdot \cancelto{0}{\dv{\vzopblk{k}_i(t + 1)}{\vxt_j(t)}} \\
                                             & \aptr 0 \qqtext{where} \begin{dcases}
                                                                        i \in \Set{1, \dots, \dblk} \\
                                                                        j \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                        k \in \Set{1, \dots, \nblk} \\
                                                                        t \in \Set{0, \dots, \cT - 1}
                                                                      \end{dcases}. \\
        \dv{\vyopblk{k}_i(t + 1)}{\vxt_j(t)} & = \cancelto{0}{\dv{\vyopog_k(t + 1)}{\vxt_j(t)}} \cdot h\qty(\vsopblk{k}_i(t + 1)) + \vyopog_k(t + 1) \cdot \dv{h\qty(\vsopblk{k}_i(t + 1))}{\vsopblk{k}_i(t + 1)} \cdot \cancelto{0}{\dv{\vsopblk{k}_i(t + 1)}{\vxt_j(t)}} \\
                                             & \aptr 0 \qqtext{where} \begin{dcases}
                                                                        i \in \Set{1, \dots, \dblk} \\
                                                                        j \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                        k \in \Set{1, \dots, \nblk} \\
                                                                        t \in \Set{0, \dots, \cT - 1}
                                                                      \end{dcases}.
      \end{align*}
    \]

由於 :math:`\vyopig(t + 1), \vyopog(t + 1), \vzopblk{k}(t + 1)` 並不是\ **直接**\透過 :math:`\vWophid` 產生，因此 :math:`\vWophid` 只能透過參與 :math:`t` 時間點\ **以前**\的計算\ **間接**\對 :math:`t + 1` 時間點的計算造成影響。
這也代表在 :math:`\eqref{13}` 作用的情況下 :math:`\vWophid` **無法**\從 :math:`\vyopig(t + 1), \vyopog(t + 1), \vyopblk{k}(t + 1)` 收到任何的\ **微分**：

.. math::
  :nowrap:

  \[
    \begin{align*}
      \dv{\vyopig_k(t + 1)}{\vWophid_{p, q}}     & \aptr 0 \qqtext{where} \begin{dcases}
                                                                            k \in \Set{1, \dots, \nblk} \\
                                                                            p \in \Set{1, \dots, \dhid} \\
                                                                            q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                            t \in \Set{0, \dots, \cT - 1}
                                                                          \end{dcases}. \\
      \dv{\vyopog_k(t + 1)}{\vWophid_{p, q}}     & \aptr 0 \qqtext{where} \begin{dcases}
                                                                            k \in \Set{1, \dots, \nblk} \\
                                                                            p \in \Set{1, \dots, \dhid} \\
                                                                            q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                            t \in \Set{0, \dots, \cT - 1}
                                                                          \end{dcases}. \\
      \dv{\vyopblk{k}_i(t + 1)}{\vWophid_{p, q}} & \aptr 0 \qqtext{where} \begin{dcases}
                                                                            i \in \Set{1, \dots, \dblk} \\
                                                                            k \in \Set{1, \dots, \nblk} \\
                                                                            p \in \Set{1, \dots, \dhid} \\
                                                                            q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                            t \in \Set{0, \dots, \cT - 1}
                                                                          \end{dcases}.
    \end{align*}
    \tag{14}\label{14}
  \]

.. dropdown:: 推導式子 :math:`\eqref{14}`

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\vyopig_k(t + 1)}{\vWophid_{p, q}}     & = \sum_{j = \din + 1}^{\din + \dhid + \nblk \times (2 + \dblk)} \qty[\cancelto{0}{\dv{\vyopig_k(t + 1)}{\vxt_j(t)}} \cdot \dv{\vxt_j(t)}{\vyophid_p(t)} \cdot \dv{\vyophid_p(t)}{\vWophid_{p, q}}] \\
                                                   & \aptr 0 \qqtext{where} \begin{dcases}
                                                                              k \in \Set{1, \dots, \nblk} \\
                                                                              p \in \Set{1, \dots, \dhid} \\
                                                                              q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                              t \in \Set{0, \dots, \cT - 1}
                                                                            \end{dcases}. \\
        \dv{\vyopog_k(t + 1)}{\vWophid_{p, q}}     & = \sum_{j = \din + 1}^{\din + \dhid + \nblk \times (2 + \dblk)} \qty[\cancelto{0}{\dv{\vyopog_k(t + 1)}{\vxt_j(t)}} \cdot \dv{\vxt_j(t)}{\vyophid_p(t)} \cdot \dv{\vyophid_p(t)}{\vWophid_{p, q}}] \\
                                                   & \aptr 0 \qqtext{where} \begin{dcases}
                                                                              k \in \Set{1, \dots, \nblk} \\
                                                                              p \in \Set{1, \dots, \dhid} \\
                                                                              q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                              t \in \Set{0, \dots, \cT - 1}
                                                                            \end{dcases}. \\
        \dv{\vyopblk{k}_i(t + 1)}{\vWophid_{p, q}} & = \sum_{j = \din + 1}^{\din + \dhid + \nblk \times (2 + \dblk)} \qty[\cancelto{0}{\dv{\vyopblk{k}_i(t + 1)}{\vxt_j(t)}} \cdot \dv{\vxt_j(t)}{\vyophid_p(t)} \cdot \dv{\vyophid_p(t)}{\vWophid_{p, q}}] \\
                                                   & \aptr 0 \qqtext{where} \begin{dcases}
                                                                              i \in \Set{1, \dots, \dblk} \\
                                                                              k \in \Set{1, \dots, \nblk} \\
                                                                              p \in \Set{1, \dots, \dhid} \\
                                                                              q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                              t \in \Set{0, \dots, \cT - 1}
                                                                            \end{dcases}.
      \end{align*}
    \]

相對於總輸出所得剩餘梯度
------------------------

我們將論文的 A.8 式拆解成 :math:`\eqref{15} \eqref{16} \eqref{17} \eqref{18}`。

總輸出參數
~~~~~~~~~~

令 :math:`\delta_{a, b}` 為 **Kronecker delta**，i.e.，

.. math::
  :nowrap:

  \[
    \delta_{a, b} = \begin{dcases}
                      1 & \qqtext*{if} a = b \\
                      0 & \qqtext*{otherwise}
                    \end{dcases}.
  \]

由於\ **總輸出** :math:`\vy(t + 1)` 不會像是傳統 RNN 的方式\ **回饋**\到模型的計算狀態中，因此\ **總輸出參數** :math:`\vWopout` 對\ **總輸出** :math:`\vy(t + 1)` 計算所得的\ **微分**\為

.. math::
  :nowrap:

  \[
    \begin{align*}
    \dv{\vy_i(t + 1)}{\vWopout_{p, q}} & = {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \delta_{i, p} \cdot \vxopout(t + 1)_q \\
                                       & \qqtext{where} \begin{dcases}
                                                          i \in \Set{1, \dots, \dout} \\
                                                          p \in \Set{1, \dots, \dout} \\
                                                          q \in \Set{1, \dots, \din + \dhid + \nblk \times \dblk} \\
                                                          t \in \Set{0, \dots, \cT - 1}
                                                        \end{dcases}.
    \end{align*}
    \tag{15}\label{15}
  \]

.. dropdown:: 推導式子 :math:`\eqref{15}`

  .. math::
    :nowrap:

    \[
      \begin{align*}
      \dv{\vy_i(t + 1)}{\vWopout_{p, q}} & = \dv{\vy_i(t + 1)}{\vzopout_i(t + 1)} \cdot \dv{\vzopout_i(t + 1)}{\vWopout_{p, q}} \\
                                         & = {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \delta_{i, p} \cdot \vxopout(t + 1)_q \\
                                         & \qqtext{where} \begin{dcases}
                                                            i \in \Set{1, \dots, \dout} \\
                                                            p \in \Set{1, \dots, \dout} \\
                                                            q \in \Set{1, \dots, \din + \dhid + \nblk \times \dblk} \\
                                                            t \in \Set{0, \dots, \cT - 1}
                                                          \end{dcases}.
      \end{align*}
    \]

- :math:`\eqref{15}` 就是論文中 A.8 式的第一個 case
- 由於 :math:`p` 可以是\ **任意**\的輸出節點，因此在 :math:`i \neq p` 時 :math:`\vWopout_{p, q}` 對於 :math:`\vy_i(t + 1)` 的微分為 :math:`0`

隱藏單元參數
~~~~~~~~~~~~

在 :math:`\eqref{12} \eqref{13} \eqref{14}` 的作用下，我們可以求得\ **隱藏單元參數** :math:`\vWophid` 在\ **丟棄**\部份微分後對於\ **總輸出** :math:`\vy(t + 1)` 計算所得的\ **剩餘微分**

.. math::
  :nowrap:

  \[
    \begin{align*}
      \dv{\vy_i(t + 1)}{\vWophid_{p, q}} & \aptr {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, p} \cdot \dv{\vyophid_p(t + 1)}{\vWophid_{p, q}} \\
                                         & \qqtext{where} \begin{dcases}
                                                            i \in \Set{1, \dots, \dout} \\
                                                            p \in \Set{1, \dots, \dhid} \\
                                                            q \in \Set{1, \dots, \din + \dhid + \nblk \times \dblk} \\
                                                            t \in \Set{0, \dots, \cT - 1}
                                                          \end{dcases}.
    \end{align*}
    \tag{16}\label{16}
  \]

.. dropdown:: 推導式子 :math:`\eqref{16}`

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\vy_i(t + 1)}{\vWophid_{p, q}} & = \dv{\vy_i(t + 1)}{\vzopout_i(t + 1)} \cdot \dv{\vzopout_i(t + 1)}{\vWophid_{p, q}} \\
                                           & = {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \sum_{j = 1}^{\din + \dhid + \nblk \times \dblk} \qty[\dv{\vzopout_i(t + 1)}{\vxopout_j(t + 1)} \cdot \cancelto{\aptr 0}{\dv{\vxopout_j(t + 1)}{\vWophid_{p, q}}}] \\
                                           & \aptr {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, p} \cdot \dv{\vyophid_p(t + 1)}{\vWophid_{p, q}} \\
                                           & \qqtext{where} \begin{dcases}
                                                              i \in \Set{1, \dots, \dout} \\
                                                              p \in \Set{1, \dots, \dhid} \\
                                                              q \in \Set{1, \dots, \din + \dhid + \nblk \times \dblk} \\
                                                              t \in \Set{0, \dots, \cT - 1}
                                                            \end{dcases}.
      \end{align*}
    \]

:math:`\eqref{16}` 就是論文中 A.8 式的最後一個 case。

閘門單元參數
~~~~~~~~~~~~

同 :math:`\eqref{16}`，我們可以計算\ **閘門單元參數** :math:`\vWopig, \vWopog` 對\ **總輸出** :math:`\vy(t + 1)` 計算所得的\ **剩餘微分**

.. math::
  :nowrap:

  \[
    \begin{align*}
      \dv{\vy_i(t + 1)}{\vWopog_{k,q}} & \aptr {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \sum_{j = 1}^{\dblk} \qty[\vWopout_{i, \din + \dhid + (k - 1) \times \dblk + j} \cdot \dv{\vyopblk{k}_j(t + 1)}{\vWopog_{k,q}}] \\
                                       & \qqtext{where} \begin{dcases}
                                                          i \in \Set{1, \dots, \dout} \\
                                                          k \in \Set{1, \dots, \nblk} \\
                                                          q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                          t \in \Set{0, \dots, \cT - 1}
                                                        \end{dcases}. \\
      \dv{\vy_i(t + 1)}{\vWopig_{k,q}} & \aptr {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \sum_{j = 1}^{\dblk} \qty[\vWopout_{i, \din + \dhid + (k - 1) \times \dblk + j} \cdot \dv{\vyopblk{k}_j(t + 1)}{\vWopig_{k, q}}] \\
                                       & \qqtext{where} \begin{dcases}
                                                          i \in \Set{1, \dots, \dout} \\
                                                          k \in \Set{1, \dots, \nblk} \\
                                                          q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                          t \in \Set{0, \dots, \cT - 1}
                                                        \end{dcases}.
    \end{align*}
    \tag{17}\label{17}
  \]

.. dropdown:: 推導式子 :math:`\eqref{17}`

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\vy_i(t + 1)}{\vWopog_{k,q}} & = \dv{\vy_i(t + 1)}{\vzopout_i(t + 1)} \cdot \dv{\vzopout_i(t + 1)}{\vWopog_{k,q}} \\
                                         & = {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \sum_{j = 1}^{\din + \dhid + \nblk \times \dblk} \qty[\dv{\vzopout_i(t + 1)}{\vxopout_j(t + 1)} \cdot \cancelto{\aptr 0}{\dv{\vxopout_j(t + 1)}{\vWopog_{k,q}}}] \\
                                         & \aptr {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \sum_{j = 1}^{\dblk} \qty[\vWopout_{i, \din + \dhid + (k - 1) \times \dblk + j} \cdot \dv{\vyopblk{k}_j(t + 1)}{\vWopog_{k,q}}] \\
                                         & \qqtext{where} \begin{dcases}
                                                            i \in \Set{1, \dots, \dout} \\
                                                            k \in \Set{1, \dots, \nblk} \\
                                                            q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                            t \in \Set{0, \dots, \cT - 1}
                                                          \end{dcases}. \\
        \dv{\vy_i(t + 1)}{\vWopig_{k,q}} & \aptr {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \sum_{j = 1}^{\dblk} \qty[\vWopout_{i, \din + \dhid + (k - 1) \times \dblk + j} \cdot \dv{\vyopblk{k}_j(t + 1)}{\vWopig_{k, q}}] \\
                                         & \qqtext{where} \begin{dcases}
                                                            i \in \Set{1, \dots, \dout} \\
                                                            k \in \Set{1, \dots, \nblk} \\
                                                            q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                            t \in \Set{0, \dots, \cT - 1}
                                                          \end{dcases}.
      \end{align*}
    \]

:math:`\eqref{17}` 就是論文中 A.8 式的第三個 case。

記憶細胞淨輸入參數
~~~~~~~~~~~~~~~~~~

**記憶細胞淨輸入參數** :math:`\vWopblk{k}` 對\ **總輸出** :math:`y(t + 1)` 計算所得的\ **剩餘梯度**\與 :math:`\eqref{17}` 幾乎\ **相同**

.. math::
  :nowrap:

  \[
    \begin{align*}
      \dv{\vy_i(t + 1)}{\vWopblk{k}_{p, q}} & \aptr {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, \din + \dhid + (k - 1) \times \dblk + p} \cdot \dv{\vyopblk{k}_p(t + 1)}{\vWopblk{k}_{p, q}} \\
                                            & \qqtext{where} \begin{dcases}
                                                                i \in \Set{1, \dots, \dout} \\
                                                                k \in \Set{1, \dots, \nblk} \\
                                                                p \in \Set{1, \dots, \dblk} \\
                                                                q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                t \in \Set{0, \dots, \cT - 1}
                                                              \end{dcases}.
    \end{align*}
    \tag{18}\label{18}
  \]

.. dropdown:: 推導式子 :math:`\eqref{17}`

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\vy_i(t + 1)}{\vWopblk{k}_{p, q}} & = \dv{\vy_i(t + 1)}{\vzopout_i(t + 1)} \cdot \dv{\vzopout_i(t + 1)}{\vWopblk{k}_{p, q}} \\
                                              & = {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \sum_{j = 1}^{\din + \dhid + \nblk \times \dblk} \qty[\dv{\vzopout_i(t + 1)}{\vxopout_j(t + 1)} \cdot \cancelto{\aptr 0}{\dv{\vxopout_j(t + 1)}{\vWopblk{k}_{p, q}}}] \\
                                              & \aptr {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, \din + \dhid + (k - 1) \times \dblk + p} \cdot \dv{\vyopblk{k}_p(t + 1)}{\vWopblk{k}_{p, q}} \\
                                              & \qqtext{where} \begin{dcases}
                                                                 i \in \Set{1, \dots, \dout} \\
                                                                 k \in \Set{1, \dots, \nblk} \\
                                                                 p \in \Set{1, \dots, \dblk} \\
                                                                 q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                 t \in \Set{0, \dots, \cT - 1}
                                                               \end{dcases}.
      \end{align*}
    \]

:math:`\eqref{18}` 就是論文中 A.8 式的第二個 case。

..
  ### 相對於隱藏單元所得剩餘梯度

  我們將論文的 A.9 式拆解成 $\eqref{45} \eqref{46} \eqref{47}$。

  #### 隱藏單元參數

  根據 $\eqref{12} \eqref{13}$ 我們可以得到**隱藏單元參數** $\vWophid$ 對於**隱藏單元** $\vyophid(t + 1)$ 計算所得**剩餘梯度**

  $$
  \begin{align*}
  i, p & \in \Set{1, \dots, \dhid} \\
  q & \in \Set{1, \dots, \din + \dhid + \nblk \cdot (2 + \dblk)} \\
  \dv{y_i^{\ophid}(t + 1)}{\vWophid_{p, q}} & = \dv{y_i^{\ophid}(t + 1)}{\nethid{i}{t + 1}} \cdot \cancelto{\aptr 0}{\dv{\nethid{i}{t + 1}}{\vWophid_{p, q}}} \\
  & \aptr \dfnethid{i}{t + 1} \cdot \delta_{i, p} \cdot \begin{pmatrix}
  \vx(t) \\
  \vyophid(t) \\
  y^{\opig}(t) \\
  \vyopog(t) \\
  \vyopblk{1}(t) \\
  \vdots \\
  \vyopblk{\nblk}(t)
  \end{pmatrix}_q
  \end{align*} \tag{45}\label{45}
  $$

  #### 閘門單元參數

  由於**隱藏單元** $\vyophid(t + 1)$ 並不是**直接**透過**閘門參數** $\vWopig, \vWopog$ 產生，因此根據 $\eqref{12}$ 我們可以推得 $\vWopig, \vWopog$ 對於 $\vyophid(t + 1)$ **剩餘梯度**為 $0$

  $$
  \begin{align*}
  D & = \din + \dhid + \nblk \cdot (2 + \dblk) \\
  \tilde{x}(t) & = \begin{pmatrix}
  \vx(t) \\
  \vyophid(t) \\
  y^{\opig}(t) \\
  \vyopog(t) \\
  \vyopblk{1}(t) \\
  \vdots \\
  \vyopblk{\nblk}(t)
  \end{pmatrix} \in \R^D \\
  i & \in \Set{1, \dots, \dhid} \\
  p & \in \Set{1, \dots, \nblk} \\
  q & \in \Set{1, \dots, D} \\
  \dv{y_i^{\ophid}(t + 1)}{\vWopog_{p, q}} & = \dv{y_i^{\ophid}(t + 1)}{\nethid{i}{t + 1}} \cdot \sum_{j = 1}^D \br{\cancelto{0}{\dv{\nethid{i}{t + 1}}{\tilde{x}_j(t)}} \cdot \dv{\tilde{x}_j(t)}{\vWopog_{p, q}}} \aptr 0 \\
  \dv{y_i^{\ophid}(t + 1)}{\vWopig_{p, q}} & \aptr 0
  \end{align*} \tag{46}\label{46}
  $$

  #### 記憶細胞淨輸入參數

  同 $\eqref{46}$，由於**隱藏單元** $\vyophid(t + 1)$ 並不是**直接**透過**記憶細胞淨輸入參數** $\vWopblk{k}$ 產生，因此根據 $\eqref{12}$ 我們可以推得 $\vWopblk{k}$ 對於 $\vyophid(t + 1)$ **剩餘梯度**為 $0$

  $$
  \begin{align*}
  D & = \din + \dhid + \nblk \cdot (2 + \dblk) \\
  \tilde{x}(t) & = \begin{pmatrix}
  \vx(t) \\
  \vyophid(t) \\
  y^{\opig}(t) \\
  \vyopog(t) \\
  \vyopblk{1}(t) \\
  \vdots \\
  \vyopblk{\nblk}(t)
  \end{pmatrix} \in \R^D \\
  i & \in \Set{1, \dots, \dhid} \\
  k & \in \Set{1, \dots, \nblk} \\
  p & \in \Set{1, \dots, \dblk} \\
  q & \in \Set{1, \dots, D} \\
  \dv{y_i^{\ophid}(t + 1)}{\vWopblk{k}_{p, q}} & = \dv{y_i^{\ophid}(t + 1)}{\nethid{i}{t + 1}} \cdot \sum_{j = 1}^D \br{\cancelto{0}{\dv{\nethid{i}{t + 1}}{\tilde{x}_j(t)}} \cdot \dv{\tilde{x}_j(t)}{\vWopblk{k}_{p, q}}} \aptr 0
  \end{align*} \tag{47}\label{47}
  $$

  ### 相對於記憶細胞輸出所得剩餘梯度

  我們將論文的 A.13 式拆解成 $\eqref{48} \eqref{49} \eqref{50}$。

  #### 閘門單元參數

  根據 $\eqref{12}$ 我們可以推得**閘門單元參數** $\vWopig, \vWopog$ 對於**記憶細胞輸出** $\vyopblk{k}(t + 1)$ 計算所得**剩餘梯度**

  $$
  \begin{align*}
  i & \in \Set{1, \dots, \dblk} \\
  k, p & \in \Set{1, \dots, \nblk} \\
  q & \in \Set{1, \dots, \din + \dhid + \nblk \cdot (2 + \dblk)} \\
  \dv{\vyopblk{k}_i(t + 1)}{\vWopog_{p, q}} & = \dv{\vyopblk{k}_i(t + 1)}{\vyopog_k(t + 1)} \cdot \dv{\vyopog_k(t + 1)}{\vWopog_{p, q}} + \dv{\vyopblk{k}_i(t + 1)}{s_i^{\blk{k}}(t + 1)} \cdot \cancelto{0}{\dv{s_i^{\blk{k}}(t + 1)}{\vWopog_{p, q}}} \\
  & \aptr h_i\pa{s_i^{\blk{k}}(t + 1)} \cdot \delta_{k, p} \cdot \dv{\vyopog_k(t + 1)}{\vWopog_{k, q}} \tag{48}\label{48} \\
  \dv{\vyopblk{k}_i(t + 1)}{\vWopig_{p, q}} & = \dv{\vyopblk{k}_i(t + 1)}{\vyopog_k(t + 1)} \cdot \cancelto{0}{\dv{\vyopog_k(t + 1)}{\vWopig_{p, q}}} + \dv{\vyopblk{k}_i(t + 1)}{s_i^{\blk{k}}(t + 1)} \cdot \dv{s_i^{\blk{k}}(t + 1)}{\vWopig_{p, q}} \\
  & \aptr \vyopog_k(t + 1) \cdot h_i'\pa{s_i^{\blk{k}}(t + 1)} \cdot \delta_{k, p} \cdot \dv{s_i^{\blk{k}}(t + 1)}{\vWopig_{k, q}} \tag{49}\label{49}
  \end{align*}
  $$

  #### 記憶細胞淨輸入參數

  同 $\eqref{49}$，使用 $\eqref{12}$ 推得**記憶細胞淨輸入參數** $\wblk{k^\star}$ 對於**記憶細胞輸出** $\vyopblk{k}(t + 1)$ 計算所得**剩餘梯度**（注意 $k^\star$ 可以**不等於** $k$）

  $$
  \begin{align*}
  i, p & \in \Set{1, \dots, \dblk} \\
  k, k^\star & \in \Set{1, \dots, \nblk} \\
  q & \in \Set{1, \dots, \din + \dhid + \nblk \cdot (2 + \dblk)} \\
  \dv{\vyopblk{k}_i(t + 1)}{\wblk{k^\star}_{p, q}} & = \dv{\vyopblk{k}_i(t + 1)}{\vyopog_k(t + 1)} \cdot \cancelto{0}{\dv{\vyopog_k(t + 1)}{\wblk{k^\star}_{p, q}}} + \dv{\vyopblk{k}_i(t + 1)}{s_i^{\blk{k}}(t + 1)} \cdot \dv{s_i^{\blk{k}}(t + 1)}{\wblk{k^\star}_{p, q}} \\
  & \aptr \vyopog_k(t + 1) \cdot h_i'\pa{s_i^{\blk{k}}(t + 1)} \cdot \delta_{k, k^\star} \cdot \delta_{i, p} \cdot \dv{s_i^{\blk{k}}(t + 1)}{\vWopblk{k}_{i, q}}
  \end{align*} \tag{50}\label{50}
  $$

  **注意錯誤**：論文 A.13 式最後使用**加法** $\delta_{\opin_j l} + \delta_{c_j^v l}$，可能會導致梯度**乘上常數** $2$，因此應該修正成**乘法** $\delta_{\opin_j l} \cdot \delta_{c_j^v l}$

  ### 相對於閘門單元所得剩餘梯度

  我們將論文的 A.10, A.11 式拆解成 $\eqref{51} \eqref{52}$。

  #### 閘門單元參數

  根據 $\eqref{12} \eqref{13}$ 我們可以得到**閘門單元參數** $\vWopig, \vWopog$ 對於**閘門單元** $\vyopig(t + 1), \vyopog(t + 1)$ 計算所得**剩餘梯度**

  $$
  \begin{align*}
  D & = \din + \dhid + \nblk \cdot (2 + \dblk) \\
  \tilde{x}(t) & = \begin{pmatrix}
  \vx(t) \\
  \vyophid(t) \\
  y^{\opig}(t) \\
  \vyopog(t) \\
  \vyopblk{1}(t) \\
  \vdots \\
  \vyopblk{\nblk}
  \end{pmatrix} \in \R^D \\
  k, p & \in \Set{1, \dots, \nblk} \\
  q & \in \Set{1, \dots, D} \\
  \dv{\vyopig_k(t + 1)}{[\vWopig ; \vWopog]_{p, q}} & = \dv{\vyopig_k(t + 1)}{\netig{k}{t + 1}} \cdot \cancelto{\aptr 0}{\dv{\netig{k}{t + 1}}{[\vWopig ; \vWopog]_{p, q}}} \\
  & \aptr \dfnetig{k}{t + 1} \cdot \delta_{k, p} \cdot \tilde{x}_q(t) \\
  \dv{\vyopog_k(t + 1)}{[\vWopig ; \vWopog]_{p, q}} & \aptr \delta_{k, p} \cdot \dfnetog{k}{t + 1} \cdot \tilde{x}_q(t)
  \end{align*} \tag{51}\label{51}
  $$

  #### 記憶細胞淨輸入參數

  由於**閘門單元** $\vyopig(t + 1), \vyopog(t + 1)$ 並不是**直接**透過**記憶細胞淨輸入參數** $\vWopblk{k}$ 產生，因此根據 $\eqref{12}$ 我們可以推得 $\vWopblk{k}$ 對於 $\vyopig(t + 1), \vyopog(t + 1)$ **剩餘梯度**為 $0$

  $$
  \begin{align*}
  D & = \din + \dhid + \nblk \cdot (2 + \dblk) \\
  \tilde{x}(t) & = \begin{pmatrix}
  \vx(t) \\
  \vyophid(t) \\
  y^{\opig}(t) \\
  \vyopog(t) \\
  \vyopblk{1}(t) \\
  \vdots \\
  \vyopblk{\nblk}
  \end{pmatrix} \in \R^D \\
  k & \in \Set{1, \dots, \nblk} \\
  p & \in \Set{1, \dots, \dblk} \\
  q & \in \Set{1, \dots, D} \\
  \dv{\vyopig_k(t + 1)}{\vWopblk{k}_{p, q}} & = \dv{\vyopig_k(t + 1)}{\netig{k}{t + 1}} \cdot \sum_{j = 1}^D \br{\cancelto{0}{\dv{\netig{k}{t + 1}}{\tilde{x}_j(t)}} \cdot \dv{\tilde{x}_j(t)}{\vWopblk{k}_{p, q}}} \aptr 0 \\
  \dv{\vyopog_k(t + 1)}{\vWopblk{k}_{p, q}} & \aptr 0
  \end{align*} \tag{52}\label{52}
  $$

  ### 相對於記憶細胞內部狀態所得剩餘梯度

  我們將論文的 A.12 式拆解成 $\eqref{53} \eqref{54} \eqref{55}$。

  #### 閘門單元參數

  將 $\eqref{12}$ 結合 $\eqref{51}$ 我們可以推得**閘門單元參數** $\vWopig, \vWopog$ 對於**記憶細胞內部狀態** $\vsopblk{k}(t + 1)$ 計算所得**剩餘梯度**

  $$
  \begin{align*}
  D & = \din + \dhid + \nblk \cdot (2 + \dblk) \\
  \tilde{x}(t) & = \begin{pmatrix}
  \vx(t) \\
  \vyophid(t) \\
  y^{\opig}(t) \\
  \vyopog(t) \\
  \vyopblk{1}(t) \\
  \vdots \\
  \vyopblk{\nblk}(t)
  \end{pmatrix} \in \R^D \\
  i & \in \Set{1, \dots, \dblk} \\
  k, p & \in \Set{1, \dots, \nblk} \\
  q & \in \Set{1, \dots, D} \\
  \dv{s_i^{\blk{k}}(t + 1)}{\vWopog_{p, q}} & = \dv{s_i^{\blk{k}}(t + 1)}{s_i^{\blk{k}}(t)} \cdot \cancelto{0}{\dv{s_i^{\blk{k}}(t)}{\vWopog_{p, q}}} + \dv{s_i^{\blk{k}}(t + 1)}{\vyopig_k(t + 1)} \cdot \cancelto{0}{\dv{\vyopig_k(t + 1)}{\vWopog_{p, q}}} \\
  & \quad + \dv{s_i^{\blk{k}}(t + 1)}{\vzopblk{k}_i(t + 1)} \cdot \cancelto{0}{\dv{\vzopblk{k}_i(t + 1)}{\vWopog_{p, q}}} \\
  & \aptr 0 \tag{53}\label{53} \\
  \dv{s_i^{\blk{k}}(t + 1)}{\vWopig_{p, q}} & = \dv{s_i^{\blk{k}}(t + 1)}{s_i^{\blk{k}}(t)} \cdot \dv{s_i^{\blk{k}}(t)}{\vWopig_{p, q}} + \dv{s_i^{\blk{k}}(t + 1)}{\vyopig_k(t + 1)} \cdot \dv{\vyopig_k(t + 1)}{\vWopig_{p, q}} \\
  & \quad + \dv{s_i^{\blk{k}}(t + 1)}{\vzopblk{k}_i(t + 1)} \cdot \cancelto{0}{\dv{\vzopblk{k}_i(t + 1)}{\vWopig_{p, q}}} \\
  & \aptr 1 \cdot \delta_{k, p} \cdot \dv{s_i^{\blk{k}}(t)}{\vWopig_{k, q}} + g_i\pa{\vzopblk{k}_i(t + 1)} \cdot \delta_{k, p} \cdot \cancelto{\aptr 0}{\dv{\vyopig_k(t + 1)}{\vWopig_{k, q}}} \\
  & \aptr \delta_{k, p} \cdot \br{\dv{s_i^{\blk{k}}(t)}{\vWopig_{k, q}} + g_i\pa{\vzopblk{k}_i(t + 1)} \cdot \dfnetig{k}{t + 1} \cdot \tilde{x}_q(t)} \tag{54}\label{54}
  \end{align*}
  $$

  #### 記憶細胞淨輸入參數

  使用 $\eqref{12}$ 推得**記憶細胞淨輸入參數** $\wblk{k^\star}$ 對於**記憶細胞內部狀態** $\vsopblk{k}(t + 1)$ 計算所得**剩餘梯度**（注意 $k^\star$ 可以**不等於** $k$）

  $$
  \begin{align*}
  D & = \din + \dhid + \nblk \cdot (2 + \dblk) \\
  \tilde{x}(t) & = \begin{pmatrix}
  \vx(t) \\
  \vyophid(t) \\
  y^{\opig}(t) \\
  \vyopog(t) \\
  \vyopblk{1}(t) \\
  \vdots \\
  \vyopblk{\nblk}(t)
  \end{pmatrix} \in \R^D \\
  i, p & \in \Set{1, \dots, \dblk} \\
  k, k^\star & \in \Set{1, \dots, \nblk} \\
  q & \in \Set{1, \dots, D} \\
  \dv{s_i^{\blk{k}}(t + 1)}{\wblk{k^\star}_{p, q}} & = \dv{s_i^{\blk{k}}(t + 1)}{s_i^{\blk{k}}(t)} \cdot \dv{s_i^{\blk{k}}(t)}{\wblk{k^\star}_{p, q}} + \dv{s_i^{\blk{k}}(t + 1)}{\vyopig_k(t + 1)} \cdot \cancelto{0}{\dv{\vyopig_k(t + 1)}{\wblk{k^\star}_{p, q}}} \\
  & \quad + \dv{s_i^{\blk{k}}(t + 1)}{\vzopblk{k}_i(t + 1)} \cdot \dv{\vzopblk{k}_i(t + 1)}{\wblk{k^\star}_{p, q}} \\
  & \aptr \delta_{k, k^\star} \cdot \delta_{i, p} \cdot 1 \cdot \dv{s_i^{\blk{k}}(t)}{\vWopblk{k}_{i, q}} \\
  & \quad + \delta_{k, k^\star} \cdot \delta_{i, p} \cdot \vyopig_k(t + 1) \cdot g_i'\pa{\vzopblk{k}_i(t + 1)} \cdot \tilde{x}_q(t) \\
  & = \delta_{k, k^\star} \cdot \delta_{i, p} \cdot \br{\dv{s_i^{\blk{k}}(t)}{\vWopblk{k}_{i, q}} + \vyopig_k(t + 1) \cdot g_i'\pa{\vzopblk{k}_i(t + 1)} \cdot \tilde{x}_q(t)}
  \end{align*} \tag{55}\label{55}
  $$

  **注意錯誤**：論文 A.12 式最後使用**加法** $\delta_{\opin_j l} + \delta_{c_j^v l}$，可能會導致梯度**乘上常數** $2$，因此應該修正成**乘法** $\delta_{\opin_j l} \cdot \delta_{c_j^v l}$

  ## 更新模型參數

  ### 總輸出參數

  從 $\eqref{4}$ 我們可以觀察出以下結論

  $$
  \begin{align*}
  \tilde{x}(t + 1) & = \begin{pmatrix}
  \vx(t) \\
  \vyophid(t + 1) \\
  \vyopblk{1}(t + 1) \\
  \vdots \\
  \vyopblk{\nblk}(t + 1)
  \end{pmatrix} \\
  i & \in \Set{1, \dots, \dout} \\
  j & \in \Set{1, \dots, \din + \dhid + \nblk \times \dblk} \\
  \dv{\cL(t + 1)}{\vWopout_{i, j}} & = \dv{\cL(t + 1)}{\loss{i}{t + 1}} \cdot \dv{\loss{i}{t + 1}}{\vy_i(t + 1)} \cdot \dv{\vy_i(t + 1)}{\vWopout_{i, j}} \\
  & = \big(y_i(t + 1) - \vyh_i(t + 1)\big) \cdot \dv{\vy_i(t + 1)}{\vWopout_{i, j}} \\
  & = \big(y_i(t + 1) - \vyh_i(t + 1)\big) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \tilde{x}_j(t + 1)
  \end{align*} \tag{56}\label{56}
  $$

  ### 隱藏單元參數

  從 $\eqref{4} \eqref{14} \eqref{16} \eqref{45}$ 我們可以觀察出以下結論

  $$
  \begin{align*}
  & p \in \Set{1, \dots, \dhid} \\
  & q \in \Set{1, \dots, \din + \dhid + \nblk \cdot (2 + \dblk)} \\
  & \dv{\cL(t + 1)}{\vWophid_{p, q}} = \sum_{i = 1}^\dout \br{\dv{\cL(t + 1)}{\loss{i}{t + 1}} \cdot \dv{\loss{i}{t + 1}}{\vy_i(t + 1)} \cdot \dv{\vy_i(t + 1)}{\vWophid_{p, q}}} \\
  & \aptr \sum_{i = 1}^\dout \br{\pa{\vy_i(t + 1) - \vyh_i(t + 1)} \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, p} \cdot \dv{\vyophid_p(t + 1)}{\vWophid_{p, q}}} \\
  & = \sum_{i = 1}^\dout \br{\pa{\vy_i(t + 1) - \vyh_i(t + 1)} \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, p}} \cdot \dv{\vyophid_p(t + 1)}{\vWophid_{p, q}} \\
  & \aptr \sum_{i = 1}^\dout \br{\pa{\vy_i(t + 1) - \vyh_i(t + 1)} \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, p}} \cdot \\
  & \quad \quad \dfnethid{p}{t + 1} \cdot \begin{pmatrix}
  \vx(t) \\
  \vyophid(t) \\
  \vyopblk{1}(t) \\
  \vdots \\
  \vyopblk{\nblk}(t)
  \end{pmatrix}_j
  \end{align*} \tag{57}\label{57}
  $$

  ### 輸出閘門單元參數

  從 $\eqref{4} \eqref{17} \eqref{48} \eqref{51} \eqref{53}$ 我們可以觀察出以下結論

  $$
  \begin{align*}
  k & \in \Set{1, \dots, \nblk} \\
  q & \in \Set{1, \dots, \din + \dhid + \nblk \cdot (2 + \dblk)} \\
  \dv{\cL(t + 1)}{\vWopog_{k, q}} & = \sum_{i = 1}^\dout \br{\dv{\cL(t + 1)}{\loss{i}{t + 1}} \cdot \dv{\loss{i}{t + 1}}{\vy_i(t + 1)} \cdot \dv{\vy_i(t + 1)}{\vWopog_{k, q}}} \\
  & \aptr \sum_{i = 1}^\dout \Bigg[\big(y_i(t + 1) - \vyh_i(t + 1)\big) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \\
  & \quad \quad \sum_{j = 1}^{\dblk} \pa{\vWopout_{i, \din + \dhid + (k - 1) \times \dblk + j} \cdot \dv{\vyopblk{k}_j(t + 1)}{\vWopog_{k, q}}}\Bigg] \\
  & \aptr \sum_{i = 1}^\dout \Bigg[\big(y_i(t + 1) - \vyh_i(t + 1)\big) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \\
  & \quad \quad \sum_{j = 1}^{\dblk} \pa{\vWopout_{i, \din + \dhid + (k - 1) \times \dblk + j} \cdot h_j\pa{s_j^{\blk{k}}(t + 1)} \cdot \dv{\vyopog_k(t + 1)}{\vWopog_{k, q}}}\Bigg] \\
  & = \Bigg[\sum_{i = 1}^\dout \big(y_i(t + 1) - \vyh_i(t + 1)\big) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \\
  & \quad \quad \pa{\sum_{j = 1}^{\dblk} \vWopout_{i, \din + \dhid + (k - 1) \times \dblk + j} \cdot h_j\pa{s_j^{\blk{k}}(t + 1)}}\Bigg] \cdot \dv{\vyopog_k(t + 1)}{\vWopog_{k, q}} \\
  & \aptr \Bigg[\sum_{i = 1}^\dout \big(y_i(t + 1) - \vyh_i(t + 1)\big) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \\
  & \quad \quad \pa{\sum_{j = 1}^{\dblk} \vWopout_{i, \din + \dhid + (k - 1) \times \dblk + j} \cdot h_j\pa{s_j^{\blk{k}}(t + 1)}}\Bigg] \cdot \\
  & \quad \quad \dfnetog{k}{t + 1} \cdot \begin{pmatrix}
  \vx(t) \\
  \vyophid(t) \\
  y^{\opig}(t) \\
  \vyopog(t) \\
  \vyopblk{1}(t) \\
  \vdots \\
  \vyopblk{\nblk}(t)
  \end{pmatrix}_q
  \end{align*} \tag{58}\label{58}
  $$

  ### 輸入閘門單元參數

  從 $\eqref{4} \eqref{17} \eqref{49} \eqref{51} \eqref{54}$ 我們可以觀察出以下結論

  $$
  \begin{align*}
  & \tilde{x}(t) = \begin{pmatrix}
  \vx(t) \\
  \vyophid(t) \\
  y^{\opig}(t) \\
  \vyopog(t) \\
  \vyopblk{1}(t) \\
  \vdots \\
  \vyopblk{\nblk}(t)
  \end{pmatrix} \\
  & k \in \Set{1, \dots, \nblk} \\
  & q \in \Set{1, \dots, \din + \dhid + \nblk \cdot (2 + \dblk)} \\
  & \dv{\cL(t + 1)}{\vWopig_{k, q}} = \sum_{i = 1}^\dout \br{\dv{\cL(t + 1)}{\loss{i}{t + 1}} \cdot \dv{\loss{i}{t + 1}}{\vy_i(t + 1)} \cdot \dv{\vy_i(t + 1)}{\vWopig_{k, q}}} \\
  & \aptr \sum_{i = 1}^\dout \Bigg[\big(y_i(t + 1) - \vyh_i(t + 1)\big) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \\
  & \quad \quad \sum_{j = 1}^{\dblk} \pa{\vWopout_{i, \din + \dhid + (k - 1) \times \dblk + j} \cdot \dv{\vyopblk{k}_j(t + 1)}{\vWopig_{k, q}}}\Bigg] \\
  & \aptr \sum_{i = 1}^\dout \Bigg[\big(y_i(t + 1) - \vyh_i(t + 1)\big) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \\
  & \quad \quad \sum_{j = 1}^{\dblk} \pa{\vWopout_{i, \din + \dhid + (k - 1) \times \dblk + j} \cdot \vyopog_k(t + 1) \cdot h_j'\pa{s_j^{\blk{k}}(t + 1)} \cdot \dv{s_j^{\blk{k}}(t + 1)}{\vWopig_{k, q}}}\Bigg] \\
  & = \Bigg(\sum_{i = 1}^\dout \Bigg[\big(y_i(t + 1) - \vyh_i(t + 1)\big) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \\
  & \quad \quad \sum_{j = 1}^{\dblk} \pa{\vWopout_{i, \din + \dhid + (k - 1) \times \dblk + j} \cdot h_j'\pa{s_j^{\blk{k}}(t + 1)} \cdot \dv{s_j^{\blk{k}}(t + 1)}{\vWopig_{k, q}}}\Bigg]\Bigg) \cdot \vyopog_k(t + 1) \\
  & \aptr \Bigg(\sum_{i = 1}^\dout \Bigg[\big(y_i(t + 1) - \vyh_i(t + 1)\big) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \\
  & \quad \quad \sum_{j = 1}^{\dblk} \bigg(\vWopout_{i, \din + \dhid + (k - 1) \times \dblk + j} \cdot h_j'\pa{s_j^{\blk{k}}(t + 1)} \cdot \bigg[\dv{s_j^{\blk{k}}(t)}{\vWopig_{k, q}} + \\
  & \quad \quad g_j\pa{\netcell{j}{k}{t + 1}} \cdot \dfnetig{k}{t + 1} \cdot \tilde{x}_q(t)\bigg]\bigg)\Bigg]\Bigg) \cdot \vyopog_k(t + 1)
  \end{align*} \tag{59}\label{59}
  $$

  ### 記憶細胞淨輸入參數

  從 $\eqref{4} \eqref{18} \eqref{47} \eqref{50} \eqref{52} \eqref{55}$ 我們可以觀察出以下結論

  $$
  \begin{align*}
  & \tilde{x}(t) = \begin{pmatrix}
  \vx(t) \\
  \vyophid(t) \\
  y^{\opig}(t) \\
  \vyopog(t) \\
  \vyopblk{1}(t) \\
  \vdots \\
  \vyopblk{\nblk}(t)
  \end{pmatrix} \\
  & k \in \Set{1, \dots, \nblk} \\
  & p \in \Set{1, \dots, \dblk} \\
  & q \in \Set{1, \dots, \din + \dhid + \nblk \cdot (2 + \dblk)} \\
  & \dv{\cL(t + 1)}{\vWopblk{k}_{p, q}} = \sum_{i = 1}^\dout \br{\dv{\cL(t + 1)}{\loss{i}{t + 1}} \cdot \dv{\loss{i}{t + 1}}{\vy_i(t + 1)} \cdot \dv{\vy_i(t + 1)}{\vWopblk{k}_{p, q}}} \\
  & \aptr \sum_{i = 1}^\dout \bigg[\big(y_i(t + 1) - \vyh_i(t + 1)\big) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \\
  & \quad \quad \vWopout_{i, \din + \dhid + (k - 1) \times \dblk + p} \cdot \dv{\vyopblk{k}_p(t + 1)}{\vWopblk{k}_{p, q}}\bigg] \\
  & = \br{\sum_{i = 1}^\dout \big(y_i(t + 1) - \vyh_i(t + 1)\big) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, \din + \dhid + (k - 1) \times \dblk + p}} \cdot \\
  & \quad \quad \dv{\vyopblk{k}_p(t + 1)}{\vWopblk{k}_{p, q}} \\
  & \aptr \br{\sum_{i = 1}^\dout \big(y_i(t + 1) - \vyh_i(t + 1)\big) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, \din + \dhid + (k - 1) \times \dblk + p}} \cdot \\
  & \quad \quad \vyopog_k(t + 1) \cdot h_p'\pa{s_p^{\blk{k}}(t + 1)} \cdot \dv{s_p^{\blk{k}}(t + 1)}{\vWopblk{k}_{p, q}}\Bigg] \\
  & \aptr \br{\sum_{i = 1}^\dout \big(y_i(t + 1) - \vyh_i(t + 1)\big) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, \din + \dhid + (k - 1) \times \dblk + p}} \cdot \\
  & \quad \quad \vyopog_k(t + 1) \cdot h_p'\pa{s_p^{\blk{k}}(t + 1)} \cdot \Bigg[\dv{s_p^{\blk{k}}(t)}{\vWopblk{k}_{p, q}} + \\
  & \quad \quad \vyopig_k(t + 1) \cdot g_p'\pa{\netcell{p}{k}{t + 1}} \cdot \tilde{x}_q(t)\Bigg]
  \end{align*} \tag{60}\label{60}
  $$

  ## 架構分析

  ### 時間複雜度

  假設 $t + 1$ 時間點的 **forward pass** 已經執行完成，則**更新** $t + 1$ 時間點**所有參數**的**時間複雜度**為

  $$
  O(\dim(\vWophid) + \dim(\vWopog) + \dim(\vWopig) + \nblk \cdot \dim(\wblk{1}) + \dim(\vWopout)) \tag{61}\label{61}
  $$

  - $\eqref{61}$ 就是論文中的 A.27 式
  - 在 $t + 1$ 時間點**參數更新**需要考慮 $t$ 時間點的**計算狀態**，請見 $\eqref{57} \eqref{58} \eqref{59} \eqref{60}$
  - 沒有如同 $\eqref{14}$ 的**連乘積**項，因此不會有**梯度消失**問題
  - 整個計算過程需要額外紀錄的**梯度**項次**只有** $\eqref{59} \eqref{60}$ 中的 $\dv{s_j^{\blk{k}}(t)}{\vWopig_{k, q}}, \dv{s_p^{\blk{k}}(t)}{\vWopblk{k}_{p, q}}$
    - 紀錄讓 LSTM 可以隨著 **forward pass** 的過程**即時更新**
    - **不需要**等到 $T$ 時間點的計算結束，因此不是採用 **BPTT** 的演算法
    - **即時更新**（意思是 $t + 1$ 時間點的 forward pass 完成後便可計算 $t + 1$ 時間點的誤差梯度）是 **RTRL** 的主要精神

  總共會執行 $t + 1$ 個 **forward pass**，因此**更新所有參數**所需的**總時間複雜度**為

  $$
  O\big(T \cdot \big[\dim(\vWophid) + \dim(\vWopog) + \dim(\vWopig) + \nblk \cdot \dim(\wblk{1}) + \dim(\vWopout)\big]\big) \tag{62}\label{62}
  $$

  ### 空間複雜度

  我們也可以推得在 $t + 1$ 時間點**更新所有參數**所需的**空間複雜度**

  $$
  O(\dim(\vWophid) + \dim(\vWopog) + \dim(\vWopig) + \nblk \cdot \dim(\wblk{1}) + \dim(\vWopout)) \tag{63}\label{63}
  $$

  總共會執行 $T$ 個 **forward pass**，但**更新**所需的**總空間複雜度**仍然同 $\eqref{63}$

  - 依照**時間順序**計算梯度，計算完 $t + 1$ 時間點的梯度時 $t$ 的資訊便可丟棄
  - 這就是 **RTRL** 的最大優點

  ### 達成梯度常數

  根據 $\eqref{12} \eqref{13}$ 我們可以推得

  $$
  \begin{align*}
  i & \in \Set{1, \dots, \dblk} \\
  k & \in \Set{1, \dots, \nblk} \\
  \dv{s_i^{\blk{k}}(t + 1)}{s_i^{\blk{k}}(t)} & = \dv{s_i^{\blk{k}}(t)}{s_i^{\blk{k}}(t)} + \cancelto{0}{\dv{\vyopig_k(t + 1)}{s_i^{\blk{k}}(t)}} \cdot g_i\pa{\vzopblk{k}_i(t + 1)} + \\
  & \quad \vyopig_k(t + 1) \cdot \cancelto{0}{\dv{g_i\pa{\vzopblk{k}_i(t + 1)}}{s_i^{\blk{k}}(t)}} \\
  & \aptr 1
  \end{align*} \tag{64}\label{64}
  $$

  由於**丟棄部份梯度**的作用，$\vsopblk{k}$ 的**梯度**是模型中**唯一**進行**遞迴**（跨過多個時間點）的計算節點。
  透過丟棄部份梯度我們從 $\eqref{64}$ 可以看出 LSTM 達成 $\eqref{23}$ 所設想的情況。

  ### 內部狀態偏差行為

  觀察 $\eqref{54} \eqref{59}$，當 $h$ 是 sigmoid 函數時，我們可以發現

  - 如果 $\vsopblk{k}(t + 1)$ 是一個**非常大**的**正數**，則 $h_j'\pa{s_j^{\blk{k}}(t + 1)}$ 會變得**非常小**
  - 如果 $\vsopblk{k}(t + 1)$ 是一個**非常小**的**負數**，則 $h_j'\pa{s_j^{\blk{k}}(t + 1)}$ 也會變得**非常小**
  - 在 $\vsopblk{k}(t + 1)$ 極正或極負的情況下，**輸入閘門參數** $\vWopig$ 的**梯度**會**消失**
  - 此現象稱為**內部狀態偏差行為**（**Internal State Drift**）
  - 同樣的現象也會發生在**記憶細胞淨輸入參數** $\wblk{1}, \dots \wblk{\nblk}$ 身上，請見 $\eqref{60}$
  - 此分析就是論文的 A.39 式改寫而來

  ### 解決 Internal State Drift

  作者提出可以在 $\vz^{\opig}$ 加上偏差項，並在**訓練初期**將偏差項弄成很小的**負數**，邏輯如下

  $$
  \begin{align*}
  & b^{\opig} \ll 0 \\
  \implies & \vz^{\opig}(1) \ll 0 \\
  \implies & y^{\opig}(1) \approx 0 \\
  \implies & s^{\vWopblk{k}}(1) = s^{\vWopblk{k}}(0) + y^{\opig}(1) \odot g\big(\vz^{\vWopblk{k}}(1)\big) \\
  & = y^{\opig}(1) \odot g\big(\vz^{\vWopblk{k}}(1)\big) \approx 0 \\
  \implies & \begin{dcases}
  s^{\vWopblk{k}}(t + 1) \not\ll 0 \\
  s^{\vWopblk{k}}(t + 1) \not\gg 0
  \end{dcases} \quad \forall t = 0, \dots, \cT - 1
  \end{align*} \tag{65}\label{65}
  $$

  根據 $\eqref{65}$ 我們就不會得到 $\vsopblk{k}(t)$ 極正或極負的情況，也就不會出現 Internal State Drift。

  雖然這種作法是種**模型偏差**（**Model Bias**）而且會導致 $y^{\opig}(\star)$ 與 $\dfnetig{k}\star$ **變小**，但作者認為這些影響比起 Internal State Drift 一點都不重要。

  ### 輸出閘門初始化

  論文 4.7 節表示，在訓練的初期模型有可能濫用**記憶細胞的初始值**作為計算的常數項（細節請見 $\eqref{15}$），導致模型在訓練的過程中學會完全**不紀錄資訊**。

  因此可以將**輸出閘門**加上偏差項，並初始化成**較小的負數**（理由類似於 $\eqref{65}$），讓記憶細胞在**計算初期**輸出值為 $0$，迫使模型只在**需要**時指派記憶細胞進行**記憶**。

  如果有多個記憶細胞，則可以給予**不同的負數**，讓模型能夠按照需要**依大小順序**取得記憶細胞（**愈大的負數**愈容易被取得）。

  ### 輸出閘門的優點

  在訓練的初期**誤差**通常比較**大**，導致**梯度**跟著變**大**，使得模型在訓練初期的參數劇烈振盪。

  由於**輸出閘門**所使用的**啟發函數** $f^\opog$ 是 sigmoid，數值範圍是 $(0, 1)$，我們可以發現 $\eqref{59} \eqref{60}$ 的梯度乘積包含 $\vyopog$，可以避免**過大誤差**造成的**梯度變大**。

  但這些說法並沒有辦法真的保證一定會實現，算是這篇論文說服力比較薄弱的點。

  ## 實驗

  ### 實驗設計

  - 要測試較長的時間差
    - 資料集不可以出現短時間差
  - 任務要夠難
    - 不可以只靠 random weight guessing 解決
    - 需要比較多的參數或是高計算精度 (sparse in weight space)

  ### 控制變因

  - 使用 Online Learning 進行最佳化
    - 意思就是 batch size 為 1
    - 不要被 Online 這個字誤導
  - 使用 sigmoid 作為啟發函數
    - 包含 $f^\opout, f^\ophid, f^\opig, f^\opog$
  - 資料隨機性
    - 資料生成為隨機
    - 訓練順序為隨機
  - 在每個時間點 $t$ 的計算順序為
    1. 將外部輸入 $\vx(t)$ 丟入模型
    2. 計算輸入閘門、輸出閘門、記憶細胞、隱藏單元
    3. 計算總輸出
  - 訓練初期只使用一個記憶細胞，即 $\nblk = 1$
    - 如果訓練中發現最佳化做的不好，開始增加記憶細胞，即 $\nblk = \nblk + 1$
    - 一旦記憶細胞增加，輸入閘門與輸出閘門也需要跟著增加
    - 這個概念稱為 Sequential Network Construction
  - $h^{\blk{k}}$ 與 $g^{\blk{k}}$ 函數如果沒有特別提及，就是使用 $\eqref{66} \eqref{67}$ 的定義

  $h^{\blk{k}} : \R \to [-1, 1]$ 函數的定義為

  $$
  h^{\blk{k}}(x) = \frac{2}{1 + \exp(-x)} - 1 = 2 \sigma(x) - 1 \tag{66}\label{66}
  $$

  $g^{\blk{k}} : \R \to [-2, 2]$ 函數的定義為

  $$
  g^{\blk{k}}(x) = \frac{4}{1 + \exp(-x)} - 2 = 4 \sigma(x) - 2 \tag{67}\label{67}
  $$

  ### 實驗 1：Embedded Reber Grammar

  <a name="paper-fig-3"></a>

  圖 3：Reber Grammar。
  一個簡單的有限狀態機，能夠生成的字母包含 BEPSTVX。
  圖片來源：[論文][論文]。

  ![圖 3](https://i.imgur.com/frOl0Tf.png)

  <a name="paper-fig-4"></a>

  圖 4：Embedded Reber Grammar。
  一個簡單的有限狀態機，包含兩個完全相同的 Reber Grammar，開頭跟結尾只能是 BT...TE 與 BP...PE。
  圖片來源：[論文][論文]。

  ![圖 4](https://i.imgur.com/SVfVbJN.png)

  #### 任務定義

  - Embedded Reber Grammar 是實驗 RNN 短時間差（Short Time Lag）的基準測試資料集
    - [圖 3](#paper-fig-3) 只是 Reber Grammar，真正的生成資料是使用[圖 4](#paper-fig-4) 的 Embedded Reber Grammar
    - Embedded Reber Grammar 時間差最短只有 $9$ 個單位
    - 傳統 RNN 在此資料集上仍然表現不錯
    - 資料生成為隨機，任何一個分支都有 $0.5$ 的機率被生成
  - 根據[圖 3](#paper-fig-3) 的架構，生成的第一個字為 B，接著是 T 或 P
    - 因此前兩個字生成 BT 或 BP 的機率各為 $0.5$
    - 能夠生成的字母包含 BEPSTVX
    - 生成直到產生 E 結束，結尾一定是 SE 或 VE
    - 由於有限狀態機中有 Loop，因此 Reber Grammar 有可能產生**任意長度**的文字
  - 根據[圖 4](#paper-fig-4) 的架構，生成的開頭為 BT 或 BP
    - 前兩個字生成 BT 或 BP 的機率各為 $0.5$
    - 如果生成 BT，則結尾一定要是 TE
    - 如果生成 BP，則結尾一定要是 PE
    - 因此 RNN 模型必須學會記住**開頭**的 T / P 與**結尾搭配**，判斷一個文字序列是否由 Embedded Reber Grammar 生成
  - 模型會在每個時間點 $t$ 收到一個字元，並輸出下一個時間點 $t + 1$ 會收到的字元
    - 輸入與輸出都是 one-hot vector，維度為 $7$，每個維度各自代表 BEPSTVX 中的一個字元，取數值最大的維度作為預測結果
    - 模型必須根據 $0, 1, \dots t - 1, t$ 時間點收到的字元預測 $t + 1$ 時間點輸出的字元
    - 概念就是 Language Model
  - 資料數
    - 訓練集：256 筆
    - 測試集：256 筆
    - 總共產生 3 組不同的訓練測試集
    - 每組資料集都跑 $10$ 次實驗，每次實驗模型都隨機初始化
    - 總共執行 $30$ 次實驗取平均
  - 評估方法
    - Accuracy

  #### LSTM 架構

  |參數|數值（或範圍）|備註|
  |-|-|-|
  |$\din$|$7$||
  |$\dhid$|$0$|沒有隱藏單元|
  |$(\nblk, \dblk)$|$\Set{(3, 2), (4, 1)}$|至少有 $3$ 個記憶細胞|
  |$\dout$|$7$||
  |$\dim(\vWophid)$|$0$|沒有隱藏單元|
  |$\dim(\vWopblk{k})$|$\dblk \times [\din + \nblk \cdot (2 + \dblk)]$|全連接隱藏層|
  |$\dim(\vWopig)$|$\nblk \times [\din + \nblk \cdot (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\vWopog)$|$\nblk \times [\din + \nblk \cdot (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\vWopout)$|$\dout \times [\nblk \times \dblk]$|外部輸入沒有直接連接到總輸出|
  |參數初始化範圍|$[-0.2, 0.2]$||
  |輸出閘門偏差項初始化範圍|$\Set{-1, -2, -3, -4}$|由大到小依序初始化不同記憶細胞對應輸出閘門偏差項|
  |Learning rate|$\Set{0.1, 0.2, 0.5}$||
  |總參數量|$\Set{264, 276}$||

  #### 實驗結果

  <a name="paper-table-1"></a>

  表格 1：Embedded Reber Grammar 實驗結果。
  表格來源：[論文][論文]。

  ![表 1](https://i.imgur.com/51yPwmH.png)

  - LSTM + 丟棄梯度 + RTRL 在不同的實驗架構中都能解決任務
    - RNN + RTRL 無法完成
    - Elman Net + ELM 無法完成
  - LSTM 收斂速度比其他模型都還要快
  - LSTM 使用的參數數量並沒有比其他的模型多太多
  - 驗證**輸出閘門**的有效性
    - 當 LSTM 模型記住第二個輸入是 T / P 之後，輸出閘門就會讓後續運算的啟發值接近 $0$，不讓記憶細胞內部狀態影響模型學習簡單的 Reber Grammar
    - 如果沒有輸出閘門，則**收斂速度會變慢**

  ### 實驗 2a：無雜訊長時間差任務

  #### 任務定義

  定義 $p + 1$ 種不同的字元，標記為 $V = \Set{\alpha, \beta, c_1, c_2, \dots, c_{p - 1}}$。

  定義 $2$ 種長度為 $p + 1$ 不同的序列 $\opseq_1, \opseq_2$，分別為

  $$
  \begin{align*}
  \opseq_1 & = \alpha, c_1, c_2, \dots, c_{p - 2}, c_{p - 1}, \alpha \\
  \opseq_2 & = \beta, c_1, c_2, \dots, c_{p - 2}, c_{p - 1}, \beta
  \end{align*}
  $$

  令 $\opseq_\star \in \Set{\opseq_1, \opseq_2}$，令 $\opseq_\star$ 第 $t$ 個時間點的字元為 $\opseq_\star(t) \in V$。

  當給予模型 $\opseq_\star(t)$ 時，模型要能夠根據 $\opseq_\star(0), \opseq_\star(1), \dots \opseq_\star(t - 1), \opseq_\star(t)$ 預測 $\opseq_\star(t + 1)$。

  - 模型需要記住 $c_1, \dots, c_{p - 1}$ 的順序
  - 模型也需要記住開頭的 $\opseq_\star(0)$ 是 $\alpha$ 還是 $\beta$，並利用 $\opseq_\star(0)$ 的資訊預測 $\opseq_\star(p + 1)$
  - 根據 $p$ 的大小這個任務可以是**短**時間差或**長**時間差
  - 訓練資料
    - 每次以各 $0.5$ 的機率抽出 $\opseq_1, \opseq_2$ 作為輸入
    - 總共執行 $5000000$ 次抽樣與更新
  - 測試資料
    - 每次以各 $0.5$ 的機率抽出 $\opseq_1, \opseq_2$ 作為輸入
    - 每次錯誤率在 $0.25$ 以下就是成功，反之失敗
    - 總共執行 $10000$ 次成功與失敗的判斷

  #### LSTM 架構

  |參數|數值（或範圍）|備註|
  |-|-|-|
  |$\din$|$p + 1$||
  |$\dhid$|$0$|沒有隱藏單元|
  |$\dblk$|$\dout$|總輸出就是記憶細胞的輸出|
  |$\nblk$|$1$|當誤差停止下降時，增加記憶細胞|
  |$\dout$|$p + 1$||
  |$g$|$g(x) = \sigma(x)$|Sigmoid 函數|
  |$h$|$h(x) = x$||
  |$\dim(\vWophid)$|$0$|沒有隱藏單元|
  |$\dim(\vWopblk{k})$|$\dblk \times [\din + (1 + \nblk) \times \dblk]$|全連接隱藏層|
  |$\dim(\vWopig)$|$\nblk \times [\din + (1 + \nblk) \times \dblk]$|全連接隱藏層|
  |$\dim(\vWopog)$|$0$|沒有輸出閘門|
  |$\dim(\vWopout)$|$0$|總輸出就是記憶細胞的輸出|
  |參數初始化範圍|$[-0.2, 0.2]$||
  |Learning rate|$1$||
  |最大更新次數|$5000000$||

  #### 實驗結果

  <a name="paper-table-2"></a>

  表格 2：無雜訊長時間差任務實驗結果。
  表格來源：[論文][論文]。

  ![表 2](https://i.imgur.com/638FPkg.png)

  - 在 $p = 4$ 時使用 RNN + RTRL 時部份實驗能夠預測序列
    - 序列很短時 RNN 還是有能力完成任務
  - 在 $p \geq 10$ 時使用 RNN + RTRL 時直接失敗
  - 在 $p = 100$ 時只剩 LSTM 能夠完全完成任務
  - LSTM 收斂速度最快

  ### 實驗 2b：有雜訊長時間差任務

  實驗設計和 LSTM 的架構與實驗 2a 完全相同，只是序列 $\opseq_1, \opseq_2$ 中除了頭尾之外的字元可以替換成 $V$ 中任意的文字，總長度維持 $p + 1$。

  - 此設計目的是為了確保實驗 2a 中的順序性無法被順利壓縮
  - 先創造訓練資料，測試使用與訓練完全相同的資料
  - 仍然只有 LSTM 能夠完全完成任務
  - LSTM 的誤差仍然很快就收斂
    - 當 $p = 100$ 時只需要 $5680$ 次更新就能完成任務
    - 代表 LSTM 能夠在有雜訊的情況下正常運作

  ### 實驗 2c：有雜訊超長時間差任務

  #### 任務定義

  實驗設計和 LSTM 的架構與實驗 2a 概念相同，只是 $V$ 增加了兩個字元 $b, e$，而序列長度可以不同。

  生成一個序列的概念如下：

  1. 固定一個正整數 $q$，代表序列基本長度
  2. 從 $c_1, \dots, c_{p - 1}$ 中隨機抽樣生成長度為 $q$ 的序列 $\opseq$
  3. 在序列的開頭補上 $b \alpha$ 或 $b \beta$（機率各為 $0.5$），讓序列長度變成 $q + 2$
  4. 接著以 $0.9$ 的機率從 $c_1, \dots, c_{p - 1}$ 中挑一個字補在序列 $\opseq$ 的尾巴，或是以 $0.1$ 的機率補上 $e$
  5. 如果生成 $e$ 就再補上 $\alpha$ 或 $\beta$（與開頭第二個字元相同）並結束
  6. 如果不是生成 $e$ 則重複步驟 4

  假設步驟 $4$ 執行了 $k + 1$ 次，則序列長度為 $2 + q + (k + 1) + 1 = q + k + 4$。
  序列的最短長度為 $q + 4$，長度的期望值為

  $$
  \begin{align*}
  & 4 + \sum_{k = 0}^\infty \frac{1}{10} \pa{\frac{9}{10}}^k (q + k) \\
  & = 4 + \frac{q}{10} \br{\sum_{k = 0}^\infty \pa{\frac{9}{10}}^k} + \frac{1}{10} \br{\sum_{k = 0}^\infty \pa{\frac{9}{10}}^k \cdot k} \\
  & = 4 + \frac{q}{10} \cdot 10 + \frac{1}{10} \cdot 100 \\
  & = q + 14
  \end{align*}
  $$

  其中

  $$
  \begin{align*}
  & \br{\sum_{k = 0}^n k x^k} - x \br{\sum_{k = 0}^n k x^k} \\
  & = (0x^0 + 1x^1 + 2x^2 + 3x^3 + \dots + nx^n) - \\
  & \quad (0x^1 + 1x^2 + 2x^3 + 3x^4 + \dots + nx^{n + 1}) \\
  & = 0x^0 + 1x^1 + 1x^2 + 1x^3 + \dots + 1x^n - nx^{n + 1} \\
  & = \br{\sum_{k = 0}^n x^k} - nx^{n + 1} \\
  & = \frac{1 - x^{n + 1}}{1 - x} - nx^{n + 1} \\
  & = \frac{1 - x^{n + 1} - nx^{n + 1} + nx^{n + 2}}{1 - x}
  \end{align*}
  $$

  因此

  $$
  \begin{align*}
  & \br{\sum_{k = 0}^n k x^k} - x \br{\sum_{k = 0}^n k x^k} = \frac{1 - x^{n + 1} - nx^{n + 1} + nx^{n + 2}}{1 - x} \\
  \implies & \sum_{k = 0}^n k x^k = \frac{1 - x^{n + 1} - nx^{n + 1} + nx^{n + 2}}{(1 - x)^2} \\
  \implies & \sum_{k = 0}^\infty k x^k = \frac{1}{(1 - x)^2} \text{ when } 0 \leq x \lt 1
  \end{align*}
  $$

  利用二項式分佈的期望值公式我們可以推得 $c_i \in V$ 出現次數的期望值

  $$
  \begin{align*}
  & \sum_{k = 0}^\infty \frac{1}{10} \cdot \pa{\frac{9}{10}}^k \cdot \br{\sum_{i = 0}^{q + k} \binom{q + k}{i} \cdot \pa{\frac{1}{p - 1}}^i \cdot \pa{1 - \frac{1}{p - 1}}^{q + k - i}} \\
  & = \sum_{k = 0}^\infty \frac{1}{10} \cdot \pa{\frac{9}{10}}^k \cdot \frac{q + k}{p - 1} \\
  & = \frac{q}{10(p - 1)} \br{\sum_{k = 0}^\infty \pa{\frac{9}{10}}^k} + \frac{1}{10(p - 1)} \br{\sum_{k = 0}^\infty \pa{\frac{9}{10}}^k \cdot k} \\
  & = \frac{q}{p - 1} + \frac{10}{p - 1} \\
  & \approx \frac{q}{p - 1} \text{ when } q \gg 0
  \end{align*}
  $$

  訓練誤差只考慮最後一個時間點 $\opseq(2 + q + k + 2)$ 的預測結果，必須要跟第 $\opseq(1)$ 個時間點的輸入相同（概念同實驗 2a）。

  測試時會連續執行 $10000$ 次的實驗，預測誤差必須要永遠小於 $0.2$。
  會以 $20$ 次的測試結果取平均。

  #### LSTM 架構

  |參數|數值（或範圍）|備註|
  |-|-|-|
  |$\din$|$p + 4$||
  |$\dhid$|$0$|沒有隱藏單元|
  |$\dblk$|$1$||
  |$\nblk$|$2$|作者認為其實只要一個記憶細胞就夠了|
  |$\dout$|$2$|只考慮最後一個時間點的預測誤差，並且預測的可能結果只有 $2$ 種（$\alpha$ 或 $\beta$）|
  |$\dim(\vWophid)$|$0$|沒有隱藏單元|
  |$\dim(\vWopblk{k})$|$\dblk \times [\din + \nblk \cdot (2 + \dblk)]$|全連接隱藏層|
  |$\dim(\vWopig)$|$\nblk \times [\din + \nblk \cdot (2 + \dblk)]$|全連接隱藏層|
  |$\dim(\vWopog)$|$\nblk \times [\din + \nblk \cdot (2 + \dblk)]$|全連接隱藏層|
  |$\dim(\vWopout)$|$\dout \times [\nblk \times \dblk]$|外部輸入沒有直接連接到總輸出|
  |參數初始化範圍|$[-0.2, 0.2]$||
  |Learning rate|$0.01$||

  #### 實驗結果

  <a name="paper-table-3"></a>

  表格 3：有雜訊超長時間差任務實驗結果。
  表格來源：[論文][論文]。

  ![表 3](https://i.imgur.com/j8e0W2U.png)

  - 其他方法沒有辦法完成任務，因此不列入表格比較
  - 輸入序列長度可到達 $1000$
  - 當輸入字元種類與輸入長度一起增加時，訓練時間只會緩慢增加
  - 當單一字元的**出現次數期望值增加**時，**學習速度會下降**
    - 作者認為是常見字詞的出現導致參數開始振盪

  ### 實驗 3a：Two-Sequence Problem

  #### 任務定義

  給予一個**實數**序列 $\opseq$，該序列可能隸屬於兩種類別 $C_1, C_2$，隸屬機率分別是 $0.5$。

  如果 $\opseq \in C_1$，則該序列的前 $N$ 個數字都是 $1.0$，序列的最後一個數字為 $1.0$。
  如果 $\opseq \in C_2$，則該序列的前 $N$ 個數字都是 $-1.0$，序列的最後一個數字為 $0.0$。

  給定一個常數 $T$，並從 $[T, T + \frac{\cT}{10}]$ 的區間中隨機挑選一個整數作為序列 $\opseq$ 的長度 $L$。

  當 $L \geq N$ 時，任何在 $\opseq(N + 1), \dots \opseq(L - 1)$ 中的數字都是由常態分佈隨機產生，常態分佈的平均為 $0$ 變異數為 $0.2$。

  - 此任務由 Bengio 提出
  - 作者發現只要用隨機權重猜測（Random Weight Guessing）就能解決，因此在實驗 3c 提出任務的改進版本
  - 訓練分成兩個階段
    - ST1：事先隨機抽取的 $256$ 筆測試資料完全分類正確
    - ST2：達成 ST1 後在 $2560$ 筆測試資料上平均錯誤低於 $0.01$
  - 實驗結果是執行 $10$ 次實驗的平均值

  #### LSTM 架構

  |參數|數值（或範圍）|備註|
  |-|-|-|
  |$\din$|$1$||
  |$\dhid$|$0$|沒有隱藏單元|
  |$\dblk$|$1$||
  |$\nblk$|$3$||
  |$\dout$|$1$||
  |$\dim(\vWophid)$|$0$|沒有隱藏單元|
  |$\dim(\vWopblk{k})$|$\dblk \times [\din + \nblk \cdot (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\vWopig)$|$\nblk \times [\din + \nblk \cdot (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\vWopog)$|$\nblk \times [\din + \nblk \cdot (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\vWopout)$|$\dout \times [\nblk \times \dblk]$|外部輸入沒有直接連接到總輸出|
  |參數初始化範圍|$[-0.1, 0.1]$||
  |輸入閘門偏差項初始化範圍|$\Set{-1, -3, -5}$|由大到小依序初始化不同記憶細胞對應輸入閘門偏差項|
  |輸出閘門偏差項初始化範圍|$\Set{-2, -4, -6}$|由大到小依序初始化不同記憶細胞對應輸出閘門偏差項|
  |Learning rate|$1$||

  #### 實驗結果

  <a name="paper-table-4"></a>

  表格 4：Two-Sequence Problem 實驗結果。
  表格來源：[論文][論文]。

  ![表 4](https://i.imgur.com/e1OKDP5.png)

  - 偏差項初始化的數值其實不需要這麼準確
  - LSTM 能夠快速解決任務
  - LSTM 在輸入有雜訊（高斯分佈）時仍然能夠正常表現

  ### 實驗 3b：Two-Sequence Problem + 雜訊

  <a name="paper-table-5"></a>

  表格 5：Two-Sequence Problem + 雜訊實驗結果。
  表格來源：[論文][論文]。

  ![表 5](https://i.imgur.com/DEkS8ST.png)

  實驗設計與 LSTM 完全與實驗 3a 相同，但對於序列 $\opseq$ 前 $N$ 個實數加上雜訊（與實驗 2a 相同的高斯分佈）。

  - 兩階段訓練稍微做點修改
    - ST1：事先隨機抽取的 $256$ 筆測試資料少於 $6$ 筆資料分類錯誤
    - ST2：達成 ST1 後在 $2560$ 筆測試資料上平均錯誤低於 $0.04$
  - 結論
    - 增加雜訊導致誤差收斂時間變長
    - 相較於實驗 3a，雖然分類錯誤率上升，但 LSTM 仍然能夠保持較低的分類錯誤率

  ### 實驗 3c：強化版 Two-Sequence Problem

  <a name="paper-table-6"></a>

  表格 6：強化版 Two-Sequence Problem 實驗結果。
  表格來源：[論文][論文]。

  ![表 6](https://i.imgur.com/1eXhAr4.png)

  實驗設計與 LSTM 完全與實驗 3b 相同，但進行以下修改

  - $C_1$ 類別必須輸出 $0.2$，$C_2$ 類別必須輸出 $0.8$
  - 高斯分佈變異數改為 $0.1$
  - 預測結果與答案絕對誤差大於 $0.1$ 就算分類錯誤
  - 任務目標是所有的預測絕對誤差平均值小於 $0.015$
  - 兩階段訓練改為一階段
    - 事先隨機抽取的 $256$ 筆測試資料完全分類正確
    - $2560$ 筆測試資料上絕對誤差平均值小於 $0.015$
  - Learning rate 改成 $0.1$
  - 結論
    - 任務變困難導致收斂時間變更長
    - 相較於實驗 3a，雖然分類錯誤率上升，但 LSTM 仍然能夠保持較低的分類錯誤率

  ### 實驗 4：加法任務

  #### 任務定義

  定義一個序列 $\opseq$，序列的每個元素都是由兩個實數組合而成，具體的數值範圍如下

  $$
  \opseq(t) \in [-1, 1] \times \Set{-1, 0, 1} \quad \forall t = 0, \dots, T
  $$

  每個時間點的元素的第一個數值都是隨機從 $[-1, 1]$ 中取出，第二個數值只能是 $-1, 0, 1$ 三個數值的其中一個。

  令 $T$ 為序列的最小長度，則序列 $\opseq$ 的長度 $L$ 將會落在 $[T, T + T / 10]$ 之間。

  決定每個時間點的元素的第二個數值的方法如下：

  1. 首先將所有元素的第二個數值初始化成 $0$
  2. 將 $t = 0$ 與 $t = L$ 的第二個數值初始化成 $-1$
  3. 從 $t = 0, \dots, 9$ 隨機挑選一個時間點，並將該時間點的第二個數值加上 $1$
  4. 如果前一個步驟剛好挑到 $t = 0$，則 $t = 0$ 的第二個數值將會是 $0$，否則為 $-1$
  5. 從 $t = 0, \dots, T / 2 - 1$ 隨機挑選一個時間點，並只挑選第二個數值仍為 $0$ 的時間點，挑選後將該時間點的第二個數值設為 $1$

  透過上述步驟 $\opseq$ 最少會包含一個元素其第二個數值為 $1$，最多會包含二個元素其第二個數值為 $1$。

  模型在 $L + 1$ 時間點必須輸出所有元素中第二個數值為 $1$ 的元素，其第一個數值的總和，並轉換到 $[0, 1]$ 區間的數值，即

  $$
  \vyh(L + 1) = 0.5 + \frac{1}{4} \sum_{t = 0}^{L} \br{\mathbb{1}(\opseq_1(t) = 1) \cdot \opseq_2(t)}
  $$

  只考慮 $L + 1$ 時間點的誤差，誤差必須要低於 $0.04$ 才算預測正確。

  - 模型必須要學會長時間關閉輸入閘門
  - 在實驗中故意對所有參數加上偏差項，實驗**內部狀態偏差行為**造成的影響
  - 當連續 $2000$ 次的誤差第於 $0.04$，且平均絕對誤差低於 $0.01$ 時停止訓練
  - 測試資料集包含 $2560$ 筆資料

  #### LSTM 架構

  |參數|數值（或範圍）|備註|
  |-|-|-|
  |$\din$|$2$||
  |$\dhid$|$0$|沒有隱藏單元|
  |$\dblk$|$2$||
  |$\nblk$|$2$||
  |$\dout$|$1$||
  |$\dim(\vWophid)$|$0$|沒有隱藏單元|
  |$\dim(\vWopblk{k})$|$\dblk \times [\din + \nblk \cdot (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\vWopig)$|$\nblk \times [\din + \nblk \cdot (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\vWopog)$|$\nblk \times [\din + \nblk \cdot (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\vWopout)$|$\dout \times [\nblk \times \dblk + 1]$|外部輸入沒有直接連接到總輸出，有額外使用偏差項|
  |參數初始化範圍|$[-0.1, 0.1]$||
  |輸入閘門偏差項初始化範圍|$\Set{-3, -6}$|由大到小依序初始化不同記憶細胞對應輸入閘門偏差項|
  |Learning rate|$0.5$||

  #### 實驗結果

  <a name="paper-table-7"></a>

  表格 7：加法任務實驗結果。
  表格來源：[論文][論文]。

  ![表 7](https://i.imgur.com/pGuMKyt.png)

  - LSTM 能夠達成任務目標
    - 不超過 $3$ 筆以上預測錯誤的資料
  - LSTM 能夠摹擬加法器，具有作為 distributed representation 的能力
  - 能夠儲存時間差至少有 $T / 2$ 以上的資訊，因此不會被**內部狀態偏差行為**影響

  ### 實驗 5：乘法任務

  #### 任務定義

  從 LSTM 的架構上來看實驗 4 的加法任務可以透過 $\eqref{14}$ 輕鬆完成，因此實驗 5 的目標是確認模型是否能夠從加法上延伸出乘法的概念，確保實驗 4 並不只是單純因模型架構而解決。

  概念與實驗 4 的任務幾乎相同，只做以下修改：

  - 每個時間點的元素第一個數值改為 $[0, 1]$ 之間的隨機值
  - $L + 1$ 時間點的輸出目標改成

  $$
  \vyh(L + 1) = 0.5 + \frac{1}{4} \prod_{t = 0}^{L} \br{\mathbb{1}(\opseq_1(t) = 1) \cdot \opseq_2(t)}
  $$

  - 當連續 $2000$ 筆訓練資料中，不超過 $n_{\opseq}$ 筆資料的絕對誤差小於 $0.04$ 就停止訓練
  - $n_{\opseq} \in \Set{13, 140}$
    - 選擇 $140$ 的理由是模型已經有能力記住資訊，但計算結果不夠精確
    - 選擇 $13$ 的理由是模型能夠精確達成任務

  #### LSTM 架構

  與實驗 4 完全相同，只做以下修改：

  - 輸入閘門偏差項改成隨機初始化
  - Learning rate 改為 $0.1$

  #### 實驗結果

  <a name="paper-table-8"></a>

  表格 8：乘法任務實驗結果。
  表格來源：[論文][論文]。

  ![表 8](https://i.imgur.com/bi9jJ3W.png)

  - LSTM 能夠達成任務目標
    - 在 $n_{\opseq} = 140$ 時不超過 $170$ 筆以上預測錯誤的資料
    - 在 $n_{\opseq} = 13$ 時不超過 $15$ 筆以上預測錯誤的資料
  - 如果額外使用隱藏單元，則收斂速度會更快
  - LSTM 能夠摹擬乘法器，具有作為 distributed representation 的能力
  - 能夠儲存時間差至少有 $T / 2$ 以上的資訊，因此不會被**內部狀態偏差行為**影響

  ### 實驗 6a：Temporal Order with 4 Classes

  #### 任務定義

  給予一個序列 $\opseq$，其長度 $L$ 會落在 $[100, 110]$ 之間，序列中的所有元素都來自於集合 $V = \Set{a, b, c, d, B, E, X, Y}$。

  序列 $\opseq$ 的開頭必定為 $B$，最後為 $E$，剩餘所有的元素都是 $a, b, c, d$，除了兩個時間點 $t_1, t_2$。

  $t_1, t_2$ 時間點只能出現 $X$ 或 $Y$，$t_1$ 時間點會落在 $[10, 20]$，$t_2$ 時間點會落在 $[50, 60]$。

  因此根據 $X, Y$ 出現的**次數**與**順序**共有 $4$ 種不同的類別

  $$
  \begin{align*}
  C_1 & = XX \\
  C_2 & = XY \\
  C_3 & = YX \\
  C_4 & = YY
  \end{align*}
  $$

  模型必須要在 $L + 1$ 時間點進行類別預測，誤差只會出現在 $L + 1$ 時間點。

  - $t_1, t_2$ 的最少時間差為 $30$
  - 模型必須要記住資訊與**出現順序**
  - 當模型成功預測連續 $2000$ 筆資料，並且預測平均誤差低於 $0.1$ 時便停止訓練
  - 測試資料共有 $2560$ 筆

  #### LSTM 架構

  |參數|數值（或範圍）|備註|
  |-|-|-|
  |$\din$|$8$||
  |$\dhid$|$0$|沒有隱藏單元|
  |$\dblk$|$2$||
  |$\nblk$|$2$||
  |$\dout$|$4$||
  |$\dim(\vWophid)$|$0$|沒有隱藏單元|
  |$\dim(\vWopblk{k})$|$\dblk \times [\din + \nblk \cdot (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\vWopig)$|$\nblk \times [\din + \nblk \cdot (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\vWopog)$|$\nblk \times [\din + \nblk \cdot (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\vWopout)$|$\dout \times [\nblk \times \dblk + 1]$|外部輸入沒有直接連接到總輸出，有額外使用偏差項|
  |參數初始化範圍|$[-0.1, 0.1]$||
  |輸入閘門偏差項初始化範圍|$\Set{-2, -4}$|由大到小依序初始化不同記憶細胞對應輸入閘門偏差項|
  |Learning rate|$0.5$||

  #### 實驗結果

  <a name="paper-table-9"></a>

  表格 9：Temporal Order with 4 Classes 任務實驗結果。
  表格來源：[論文][論文]。

  ![表 9](https://i.imgur.com/ucyQoeQ.png)

  - LSTM 的平均誤差低於 $0.1$

    - 沒有超過 $3$ 筆以上的預測錯誤
  - LSTM 可能使用以下的方法進行解答
    - 擁有 $2$ 個記憶細胞時，依照順序記住出現的資訊
    - 只有 $1$ 個記憶細胞時，LSTM 可以改成記憶狀態的轉移

  ### 實驗 6b：Temporal Order with 8 Classes

  #### 任務定義

  與實驗 6a 完全相同，只是多了一個 $t_3$ 時間點可以出現 $X, Y$。

  - $t_2$ 時間點改成落在 $[33, 43]$
  - $t_3$ 時間點落在 $[66, 76]$
  - 類別變成 $8$ 種

  #### LSTM 架構

  |參數|數值（或範圍）|備註|
  |-|-|-|
  |$\din$|$8$||
  |$\dhid$|$0$|沒有隱藏單元|
  |$\dblk$|$2$||
  |$\nblk$|$3$||
  |$\dout$|$8$||
  |$\dim(\vWophid)$|$0$|沒有隱藏單元|
  |$\dim(\vWopblk{k})$|$\dblk \times [\din + \nblk \cdot (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\vWopig)$|$\nblk \times [\din + \nblk \cdot (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\vWopog)$|$\nblk \times [\din + \nblk \cdot (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\vWopout)$|$\dout \times [\nblk \times \dblk + 1]$|外部輸入沒有直接連接到總輸出，有額外使用偏差項|
  |參數初始化範圍|$[-0.1, 0.1]$||
  |輸入閘門偏差項初始化範圍|$\Set{-2, -4, -6}$|由大到小依序初始化不同記憶細胞對應輸入閘門偏差項|
  |Learning rate|$0.1$||

  #### 實驗結果

  見[表格 9](#paper-table-9)。

  [Pytorch-LSTM]: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html?highlight=lstm#torch.nn.LSTM
  [論文]: https://ieeexplore.ieee.org/abstract/document/6795963
  [LSTM2000]: https://direct.mit.edu/neco/article-abstract/12/10/2451/6415/Learning-to-Forget-Continual-Prediction-with-LSTM
  [LSTM2002]: https://www.jmlr.org/papers/v3/gers02a.html


.. footbibliography::

.. ====================================================================================================================
.. external links
.. ====================================================================================================================

.. _Pytorch-LSTM: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html?highlight=lstm#torch.nn.LSTM

