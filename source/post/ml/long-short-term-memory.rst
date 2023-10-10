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

- 探討 :term:`RNN` 模型進行最佳化時遇到的問題，提出的解決方案為新的模型架構「:term:`LSTM`」與最佳化演算法「truncated RTRL」

  - **梯度爆炸**\（:term:`gradient explosion`）\造成神經網路的\ **參數數值劇烈振盪**\（**oscillating weights**）
  - **梯度消失**\（:term:`gradient vanishing`）\造成\ **訓練時間慢長**
  - 關鍵輸入資訊\ **時間差較長**\（**long time lags**）導致模型無法處理資訊

- LSTM 架構設計

  - **Memory cells and memory cell blocks**

    - 目標為解決關鍵輸入資訊時間差較長的問題
    - 取代 RNN 的遞迴節點
    - 學習\ **協助** gate units 完成\ **寫入**/\ **讀取** memory cells

  - **Gate units**

    - 解決參數必須同時學習不同目標而導致的更新數值衝突
    - 基於\ **乘法**\計算機制，提出兩種 gate units：

      - **Input gate units**：學習\ **寫入**\（\ **開啟**）/**保留**\（\ **關閉**）memory cells
      - **Output gate units**：學習\ **讀取**\（\ **開啟**）/**忽略**\（\ **關閉**）memory cells

    - Gate units 中的 **bias term** 必須\ **初始化**\成\ **負數**

      - Input gate bias 初始化成負數能夠解決 **internal state drift**
      - Output gate bias 初始化成負數能夠避免模型\ **濫用 memory cells 初始值**
      - 如果沒有 output gate units，則\ **收斂速度會變慢**

- truncated-RTRL 最佳化演算法設計

  - 目標為避免梯度\ **爆炸**\或\ **消失**
  - 以\ **捨棄計算部份微分**\做為近似全微分的手段，因此只能使用 RTRL 而不能使用 BPTT
  - Backward pass 演算法的\ **時間複雜度**\說明該最佳化演算法計算上\ **非常有效率**
  - Backward pass 演算法的\ **空間複雜度**\說明該最佳化演算法理論上\ **沒有輸入長度的限制**

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
  - 本篇論文只有 input/output gate units，並沒有使用 forget gate units :footcite:`gers-etal-2000-learning`

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
此論文認為，雖然模型並非只存在自連接節點，但若要讓自連接節點成功運作，可以透過 :math:`\eqref{11}` 推導得出以下結論：

- 自連接節點使用的 activation function 必須為 identity function
- 自連接節點使用的參數 :math:`\vWii` 必須為 :math:`1.0`

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
      & \indent{2}   \vsopblk{k}(0) \algoEq \zv \\
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
      & \indent{2}   \algoCmt{Concatenate input units with new activations.} \\
      & \indent{2}   \vxopout(t + 1) \algoEq \begin{pmatrix}
                                               \vx(t) \\
                                               \vyophid(t + 1) \\
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

.. note::

  以上計算定義請見論文中式子 A.1, A.2, A.3, A.4, A.5, A.6, A.7。

Memory Cell Blocks and Memory Cells
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Memory cells 的主要功能為記憶過去的輸入資訊。

- 在 :math:`t` 時間點時，一個 LSTM 模型有 :math:`\nblk` 個 memory cell blocks
- 在 :math:`t` 時間點時，第 :math:`k` 個 memory cell block 內有 :math:`\dblk` 個 memory cells
- 例如：:ref:`paper-fig-2`

  - 共有 :math:`2` 個不同的 memory cell blocks
  - 每個 memory cell block 中包含 :math:`2` 個 memory cells

Input Gate Units
~~~~~~~~~~~~~~~~

Input gate units 決定與控制計算資訊是否需要流入 memory cells，LSTM 以此設計避免因輸入訊號衝突造成的參數更新矛盾。

- Input gate units 是以\ **乘法**\參與計算，因此稱為 **multiplicative gate units**

  - Memory cells in the same memory cell block **share** the same input gate unit（見論文 4.4 節）
  - 因此 :math:`\vyopig_k(t + 1) \cdot g\qty(\vzopblk{k}(t + 1))` 中的乘法是\ **純量**\乘上\ **向量**

- 模型會在訓練的過程中學習\ **關閉**\與\ **開啟** input gate units

  - :math:`\vyopig_k(t + 1) \approx 0` 代表\ **關閉** :math:`t + 1` 時間點的第 :math:`k` 個 input gate unit
  - :math:`\vyopig_k(t + 1) \approx 1` 代表\ **開啟** :math:`t + 1` 時間點的第 :math:`k` 個 input gate unit
  - 全部 :math:`\nblk` 個 input gate units 不一定要同時關閉或開啟

- 當模型認為 :math:`g\qty(\vzopblk{k}(t + 1))` **不重要**\時，模型應該要\ **關閉**\第 :math:`k` 個 input gate unit

  - 不論 :math:`g\qty(\vzopblk{k}(t + 1))` 的大小，只要關閉 :math:`\vyopig_k(t + 1)`，就代表丟棄輸入訊號 :math:`\vxt(t)`，只以\ **過去資訊** :math:`\vsopblk{k}(t)` 進行決策，且計算資訊 :math:`\vxt(t)` **完全無法影響**\接下來的所有計算
  - 關閉 :math:`\vyopig_k(t + 1)` 時會得到 :math:`\vsopblk{k}(t + 1) = \vsopblk{k}(t)`，達成 CEC（見 :math:`\eqref{11}`），藉此保障\ **梯度不會消失**

- 當模型認為 :math:`g\qty(\vzopblk{k}(t + 1))` **重要**\時，模型應該要\ **開啟**\第 :math:`k` 個 input gate unit
- 例如：:ref:`paper-fig-2`

  - Memory cells ``cell 1`` and ``cell 2`` in memory cell block ``block 1`` 共享 input gate unit ``in 1``
  - Memory cells ``cell 1`` and ``cell 2`` in memory cell block ``block 2`` 共享 input gate unit ``in 2``

.. note::

  我的 :math:`\vyopig_j(t + 1)` 是對應到論文中的 :math:`y^{\opin_j}(t + 1)`，見論文 4.1 節。

Memory Cell Internal States
~~~~~~~~~~~~~~~~~~~~~~~~~~~

將 CEC 融入 LSTM 的主要機制。

- 有時簡稱 memory cell internal states 為 internal states
- 更新 :math:`t + 1` 時間點 internal states 的唯一管道是 :math:`t` 時間點的計算資訊 :math:`\vxt(t)`
- 更新 :math:`t + 1` 時間點 internal states 的決策取決於 :math:`t + 1` 時間點的 input gate units :math:`\vyopig(t + 1)`
- 由於第 :math:`k` 個 memory cell blocks 中的 internal states :math:`\vsopblk{k}(t + 1)` 主要只與第 :math:`k` 個 internal states :math:`\vsopblk{k}(t)` 連接，因此稱為 **fixed self-connection**
- 由於第 :math:`k` 個 memory cell blocks 中的 internal states :math:`\vsopblk{k}(t + 1)` 是透過加法與 :math:`\vsopblk{k}(t)` 結合，因此稱為 central **linear** unit

.. note::

  我的 :math:`\vsopblk{k}_j(t + 1)` 是對應到論文中的 :math:`s_{c_j}(t + 1)`，見論文 4.1 節。

Output Gate Units
~~~~~~~~~~~~~~~~~

Output gate units 決定與控制 memory cell block activations 是否需要用於當前輸出與未來資訊的計算，LSTM 以此設計避免因輸出訊號衝突造成的參數更新矛盾。

- Output gate units 是以\ **乘法**\參與計算，因此稱為 **multiplicative gate units**

  - Memory cells in the same memory cell block **share** the same output gate unit（見論文 4.4 節）
  - 因此 :math:`\vyopog_k(t + 1) \cdot h\qty(\vsopblk{k}(t + 1))` 中的乘法是\ **純量**\乘上\ **向量**

- 模型會在訓練的過程中學習\ **關閉**\與\ **開啟** output gate units

  - :math:`\vyopog_k(t + 1) \approx 0` 代表\ **關閉** :math:`t + 1` 時間點的第 :math:`k` 個 output gate unit
  - :math:`\vyopog_k(t + 1) \approx 1` 代表\ **開啟** :math:`t + 1` 時間點的第 :math:`k` 個 output gate unit
  - 全部 :math:`\nblk` 個 output gate units 不一定要同時關閉或開啟

- 當模型認為 :math:`h\qty(\vsopblk{k}(t + 1))` **不重要**\時，模型應該要\ **關閉**\第 :math:`k` 個 output gate unit

  - 在 :math:`\vyopig_k(t + 1)` **開啟**\的狀況下，**關閉** :math:`\vyopog_k(t + 1)` 代表不讓 :math:`h\qty(\vsopblk{k}(t + 1))` 影響當前計算
  - 在 :math:`\vyopig_k(t + 1)` **關閉**\的狀況下，**關閉** :math:`\vyopog_k(t + 1)` 代表不讓 :math:`h\qty(\vsopblk{k}(t))` 影響當前計算
  - 不論 :math:`h\qty(\vsopblk{k}(t + 1))` 的大小，只要關閉 :math:`\vyopog_k(t + 1)`，則 :math:`\vsopblk{k}(t + 1)` **無法影響**\當前計算，但仍可能影響未來計算（例如關閉 :math:`\vyopig_k(t + 2)` 且開啟 :math:`\vyopog_k(t + 2)` 時）

- 當模型認為 :math:`h\qty(\vsopblk{k}(t + 1))` **重要**\時，模型應該要\ **開啟**\第 :math:`k` 個 output gate unit

  - 在 :math:`\vyopig_k(t + 1)` **開啟**\的狀況下，**開啟** :math:`\vyopog_k(t + 1)` 代表讓 :math:`\vxt(t)` 影響當前計算
  - 在 :math:`\vyopig_k(t + 1)` **關閉**\的狀況下，**開啟** :math:`\vyopog_k(t + 1)` 代表不讓 :math:`\vxt(t)` 影響當前計算

- `PyTorch 實作的 LSTM <Pytorch-LSTM_>`_ 中 :math:`h(t)` 表達的意思是 memory cell block activation :math:`\vyopblk{k}(t)`
- 例如：:ref:`paper-fig-2`

  - Memory cells ``cell 1`` and ``cell 2`` in memory cell block ``block 1`` 共享 output gate unit ``out 1``
  - Memory cells ``cell 1`` and ``cell 2`` in memory cell block ``block 2`` 共享 output gate unit ``out 2``

.. note::

  我的 :math:`\vyopog_j(t + 1)` 是對應到論文中的 :math:`y^{\opout_j}(t + 1)`，見論文 4.1 節。

Activation Functions
~~~~~~~~~~~~~~~~~~~~

- :math:`f^\ophid, f^\opig, f^\opog, f^\opout, g, h` 都是 differentiable element-wise activation function，大部份都是 sigmoid 或是 sigmoid 的變形
- :math:`f^\opig, f^\opog` 的數值範圍（range）必須限制在 :math:`[0, 1]`，才能達成 multiplicative gate 的功能
- :math:`f^\opout` 的數值範圍只跟任務有關
- 論文並沒有給 :math:`f^\ophid, g, h` 任何數值範圍的限制

Hidden Units
~~~~~~~~~~~~

- 作者將此論文新定義的 input/output gate units 與 memory cells 稱為 hidden units（見論文 4.3 節）
- 作者將 :math:`\vyophid(t)` 稱為 **conventional hidden units**，因此當我說到 hidden units 時泛指 gate units、memory cells 與 conventional hidden units
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
  - Memory cell block activations 至少要傳播 :math:`2` 個時間點才能影響輸出，代表 :math:`t = 1` 的輸出完全無法利用到 memory cells 的資訊
  - 後續的實驗架構設計中沒有將 input layer 連接到 output layer，代表 :math:`t = 1` 的輸出完全依賴模型的初始狀態（常數），非常不合理

  因此我決定改用我認為是正確的版本撰寫後續的筆記，即 :math:`t + 1` 時間點的\ **輸出**\與 :math:`t` 時間點的 memory cell block activations **有關**。

.. note::

  注意在計算 input/output gate units 時並\ **沒有**\使用 **bias term**，但可以將 bias term 想成 :math:`\vx(t)` 中的某個 coordinate 的數值永遠為 :math:`1`。
  後續的分析會提到可以使用 bias term 進行\ **計算缺陷**\的修正。

參數結構
--------

+---------------------+---------------------------------------------------------------------------------------------------------+---------------------+-------------------------------------------------+
| Parameter           | Meaning                                                                                                 | Output Vector Shape | Input Vector Shape                              |
+=====================+=========================================================================================================+=====================+=================================================+
| :math:`\vWophid`    | Weight matrix connect :math:`\vxt(t)` to conventional hidden units :math:`\vyophid(t + 1)`.             | :math:`\dhid`       | :math:`\din + \dhid + \nblk \times (2 + \dblk)` |
+---------------------+---------------------------------------------------------------------------------------------------------+---------------------+-------------------------------------------------+
| :math:`\vWopig`     | Weight matrix connect :math:`\vxt(t)` to input gate units :math:`\vyopig(t + 1)`.                       | :math:`\nblk`       | :math:`\din + \dhid + \nblk \times (2 + \dblk)` |
+---------------------+---------------------------------------------------------------------------------------------------------+---------------------+-------------------------------------------------+
| :math:`\vWopog`     | Weight matrix connect :math:`\vxt(t)` to output gate units :math:`\vyopog(t + 1)`.                      | :math:`\nblk`       | :math:`\din + \dhid + \nblk \times (2 + \dblk)` |
+---------------------+---------------------------------------------------------------------------------------------------------+---------------------+-------------------------------------------------+
| :math:`\vWopblk{k}` | Weight matrix connect :math:`\vxt(t)` to the :math:`k`-th memory cell block :math:`\vyopblk{k}(t + 1)`. | :math:`\dblk`       | :math:`\din + \dhid + \nblk \times (2 + \dblk)` |
+---------------------+---------------------------------------------------------------------------------------------------------+---------------------+-------------------------------------------------+
| :math:`\vWopout`    | Weight matrix connect :math:`\vxopout(t)` to output units :math:`\vy(t + 1)`.                           | :math:`\dblk`       | :math:`\din + \dhid + \nblk \times \dblk`       |
+---------------------+---------------------------------------------------------------------------------------------------------+---------------------+-------------------------------------------------+

LSTM 最佳化
===========

過去的論文中提出以\ **修改最佳化過程**\避免 RNN 訓練遇到\ **梯度爆炸 / 消失**\的問題（例如 Truncated BPTT）。
作者在論文 4.5 節提出\ **最佳化** LSTM 的方法為 **RTRL 的變種**，主要精神如下：

- 透過設計模型計算架構確保達成 **CEC** （見 :math:`\eqref{11}`）
- 最佳化過程必須避免進行\ **遞迴 back propagation**，否則會遇到梯度爆炸 / 消失
- 結合 RTRL 的概念，每次透過 :math:`t` 時間點的輸入得到 :math:`t + 1` 時間點的誤差時，**馬上**\進行 back propagation 並\ **更新參數**

接下來我們將描述 LSTM 所使用的最佳化演算法。
我們定義新的符號 :math:`\aptr`，代表進行 back propagation 的過程會有\ **部份微分**\故意被\ **丟棄**\（設定為 :math:`0`），並以丟棄結果\ **近似**\參數對誤差求得的\ **全微分**。
此論文將所有與 **hidden units** 相連的節點 :math:`\vxt(t)` 產生的微分值一律\ **丟棄**，公式如下：

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

根據 :math:`\eqref{12}` 我們可以進一步推得以下微分近似值：

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

  首先根據式子 :math:`\eqref{12}` 的定義可以得到以下微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\vyophid_i(t + 1)}{\vxt_j(t)} & = \dv{\vyophid_i(t + 1)}{\vzophid_i(t + 1)} \cdot \cancelto{\aptr 0}{\dv{\vzophid_i(t + 1)}{\vxt_j(t)}} \\
                                          & \aptr 0 \qqtext{where} \begin{dcases}
                                                                     i \in \Set{1, \dots, \dhid} \\
                                                                     j \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                     t \in \Set{0, \dots, \cT - 1}
                                                                   \end{dcases}. \\
        \dv{\vyopig_k(t + 1)}{\vxt_j(t)}  & = \dv{\vyopig_k(t + 1)}{\vzopig_k(t + 1)} \cdot \cancelto{\aptr 0}{\dv{\vzopig_k(t + 1)}{\vxt_j(t)}} \\
                                          & \aptr 0 \qqtext{where} \begin{dcases}
                                                                     j \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                     k \in \Set{1, \dots, \nblk} \\
                                                                     t \in \Set{0, \dots, \cT - 1}
                                                                   \end{dcases}. \\
        \dv{\vyopog_k(t + 1)}{\vxt_j(t)}  & = \dv{\vyopog_k(t + 1)}{\vzopog_k(t + 1)} \cdot \cancelto{\aptr 0}{\dv{\vzopog_k(t + 1)}{\vxt_j(t)}} \\
                                          & \aptr 0 \qqtext{where} \begin{dcases}
                                                                     j \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                     k \in \Set{1, \dots, \nblk} \\
                                                                     t \in \Set{0, \dots, \cT - 1}
                                                                   \end{dcases}.
      \end{align*}
    \]

  接著利用上述的結果結合 :math:`\eqref{12}` 推導出 :math:`\vxt(t)` 對於 memory cell internal states 的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\vsopblk{k}_i(t + 1)}{\vxt_j(t)} & = \cancelto{\aptr 0}{\dv{\vsopblk{k}_i(t)}{\vxt_j(t)}} + \cancelto{\aptr 0}{\dv{\vyopig_k(t + 1)}{\vxt_j(t)}} \cdot g\qty(\vzopblk{k}_i(t + 1)) + \vyopig_k(t + 1) \cdot \dv{g\qty(\vzopblk{k}_i(t + 1))}{\vzopblk{k}_i(t + 1)} \cdot \cancelto{\aptr 0}{\dv{\vzopblk{k}_i(t + 1)}{\vxt_j(t)}} \\
                                             & \aptr 0 \qqtext{where} \begin{dcases}
                                                                        i \in \Set{1, \dots, \dblk} \\
                                                                        j \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                        k \in \Set{1, \dots, \nblk} \\
                                                                        t \in \Set{0, \dots, \cT - 1}
                                                                      \end{dcases}.
      \end{align*}
    \]

  最後總和上述推論得出 :math:`\vxt(t)` 對於 memory cell block activations 的微分近似結果：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\vyopblk{k}_i(t + 1)}{\vxt_j(t)} & = \cancelto{\aptr 0}{\dv{\vyopog_k(t + 1)}{\vxt_j(t)}} \cdot h\qty(\vsopblk{k}_i(t + 1)) + \vyopog_k(t + 1) \cdot \dv{h\qty(\vsopblk{k}_i(t + 1))}{\vsopblk{k}_i(t + 1)} \cdot \cancelto{\aptr 0}{\dv{\vsopblk{k}_i(t + 1)}{\vxt_j(t)}} \\
                                             & \aptr 0 \qqtext{where} \begin{dcases}
                                                                        i \in \Set{1, \dots, \dblk} \\
                                                                        j \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                        k \in \Set{1, \dots, \nblk} \\
                                                                        t \in \Set{0, \dots, \cT - 1}
                                                                      \end{dcases}.
      \end{align*}
    \]

.. note::

  式子 :math:`\eqref{13}` 就是論文 A.1.2 節開頭的前幾項公式。

我們可以將 :math:`\eqref{13}` 直觀的理解為：任何在 :math:`t + 1` 時間點的誤差資訊\ **無法**\傳遞回 :math:`t` 時間點的節點，因此 :math:`t + 1` 時間點誤差產生的微分只會用於更新參數\ **一次**，**不會**\透過\ **遞迴式**\做 back propagation。
後續我們將會根據 :math:`\eqref{12} \eqref{13}` 推導出每個參數對誤差的微分近似值。

:math:`\vWopout` 相對於誤差的微分
---------------------------------

由於輸出 :math:`\vy(t + 1)` **不會**\如傳統 RNN 的方式\ **回饋**\到模型的計算狀態中，因此計算輸出參數 :math:`\vWopout` 對誤差所得的微分不需近似，結果如下：

.. math::
  :nowrap:

  \[
    \begin{align*}
      & \dv{\cL\qty(\vy(t + 1), \vyh(t + 1))}{\vWopout_{p, q}} = \qty(\vy_p(t + 1) - \vyh_p(t + 1)) \cdot {f^\opout}'\qty(\vzopout_p(t + 1)) \cdot \vxopout_q(t + 1) \\
      & \qqtext{where} \begin{dcases}
                         p \in \Set{1, \dots, \dout} \\
                         q \in \Set{1, \dots, \din + \dhid + \nblk \times \dblk} \\
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
        & \dv{\cL\qty(\vy(t + 1), \vyh(t + 1))}{\vWopout_{p, q}} \\
        & = \sum_{i = 1}^\dout \dv{\frac{1}{2} \qty(\vy_i(t + 1) - \vyh_i(t + 1))^2}{\vWopout_{p, q}} \\
        & = \sum_{i = 1}^\dout \dv{\frac{1}{2} \qty(\vy_i(t + 1) - \vyh_i(t + 1))^2}{\vy_i(t + 1)} \cdot \dv{\vy_i(t + 1)}{\vzopout_i(t + 1)} \cdot \dv{\vzopout_i(t + 1)}{\vWopout_{p, q}} \\
        & = \sum_{i = 1}^\dout \qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \delta_{i, p} \cdot \vxopout_q(t + 1) \\
        & = \qty(\vy_p(t + 1) - \vyh_p(t + 1)) \cdot {f^\opout}'\qty(\vzopout_p(t + 1)) \cdot \vxopout_q(t + 1) \\
        & \qqtext{where} \begin{dcases}
                           p \in \Set{1, \dots, \dout} \\
                           q \in \Set{1, \dots, \din + \dhid + \nblk \times \dblk} \\
                           t \in \Set{0, \dots, \cT - 1}
                         \end{dcases}.
      \end{align*}
    \]

.. note::

  :math:`\eqref{14}` 就是論文中 A.8 式中 :math:`l = k` 的 case。

:math:`\vWophid` 相對於誤差的微分近似值
---------------------------------------

.. math::
  :nowrap:

  \[
    \begin{align*}
      & \dv{\cL\qty(\vy(t + 1) - \vyh(t + 1))}{\vWophid_{p, q}} \aptr \qty(\sum_{i = 1}^\dout \qty[\qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, \din + p}]) \cdot {f^\ophid}'\qty(\vzophid_p(t + 1)) \cdot \vxt_q(t) \\
      & \qqtext{where} \begin{dcases}
                         p \in \Set{1, \dots, \dhid} \\
                         q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                         t \in \Set{0, \dots, \cT - 1}
                       \end{dcases}.
    \end{align*}
    \tag{15}\label{15}
  \]

.. dropdown:: 推導式子 :math:`\eqref{15}`

  根據式子 :math:`\eqref{13}` 我們可以得到 :math:`\vWophid` 對於 input/output gate units 的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\vyopig_k(t + 1)}{\vWophid_{p, q}} & = \sum_{j = 1}^{\din + \dhid + \nblk \times (2 + \dblk)} \qty[\cancelto{\aptr 0}{\dv{\vyopig_k(t + 1)}{\vxt_j(t)}} \cdot \dv{\vxt_j(t)}{\vWophid_{p, q}}] \\
                                               & \aptr 0 \qqtext{where} \begin{dcases}
                                                                          k \in \Set{1, \dots, \nblk} \\
                                                                          p \in \Set{1, \dots, \dhid} \\
                                                                          q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                          t \in \Set{0, \dots, \cT - 1}
                                                                        \end{dcases}. \\
        \dv{\vyopog_k(t + 1)}{\vWophid_{p, q}} & = \sum_{j = 1}^{\din + \dhid + \nblk \times (2 + \dblk)} \qty[\cancelto{\aptr 0}{\dv{\vyopog_k(t + 1)}{\vxt_j(t)}} \cdot \dv{\vxt_j(t)}{\vWophid_{p, q}}] \\
                                               & \aptr 0 \qqtext{where} \begin{dcases}
                                                                          k \in \Set{1, \dots, \nblk} \\
                                                                          p \in \Set{1, \dots, \dhid} \\
                                                                          q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                          t \in \Set{0, \dots, \cT - 1}
                                                                        \end{dcases}.
      \end{align*}
    \]

  這代表在丟棄部份微分後 :math:`\vWophid` 將\ **無法**\透過 input/output gate units 取得資訊。
  接著我們推導 :math:`\vWophid` 對於 memory cell internal states 的微分近似值。
  結合式子 :math:`\eqref{12}` 與前面的推導，我們可以遞迴推論得出以下結果：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\vsopblk{k}_i(t + 1)}{\vWophid_{p, q}} & = \dv{\vsopblk{k}_i(t)}{\vWophid_{p, q}} + \cancelto{\aptr 0}{\dv{\vyopig_k(t + 1)}{\vWophid_{p, q}}} \cdot g\qty(\vzopblk{k}_i(t + 1)) + \vyopig_k(t + 1) \cdot \dv{g\qty(\vzopblk{k}_i(t + 1))}{\vzopblk{k}_i(t + 1)} \cdot \sum_{j = 1}^{\din + \dhid + \nblk \times (2 + \dblk)} \qty[\cancelto{\aptr 0}{\dv{\vzopblk{k}_i(t + 1)}{\vxt_j(t)}} \cdot \dv{\vxt_j(t)}{\vWophid_{p, q}}] \\
                                                   & \aptr \dv{\vsopblk{k}_i(t)}{\vWophid_{p, q}} \\
                                                   & \aptr \dv{\vsopblk{k}_i(t - 1)}{\vWophid_{p, q}} \\
                                                   & \vdots \\
                                                   & \aptr \cancelto{0}{\dv{\vsopblk{k}_i(0)}{\vWophid_{p, q}}} \\
                                                   & = 0 \qqtext{where} \begin{dcases}
                                                                              i \in \Set{1, \dots, \dblk} \\
                                                                              k \in \Set{1, \dots, \nblk} \\
                                                                              p \in \Set{1, \dots, \dhid} \\
                                                                              q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                              t \in \Set{0, \dots, \cT - 1}
                                                                            \end{dcases}.
      \end{align*}
    \]

  上式告訴我們在丟棄部份微分後 :math:`\vWophid` **無法**\透過 memory cell internal states 取得資訊。
  綜合前述結論，直覺告訴我們 :math:`\vWophid` 對於 memory cell block activations 的微分近似值應該為 :math:`0`。
  以下推導證實該直覺為真：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\vyopblk{k}_i(t + 1)}{\vWophid_{p, q}} & = \cancelto{\aptr 0}{\dv{\vyopog_k(t + 1)}{\vWophid_{p, q}}} \cdot h\qty(\vsopblk{k}_i(t + 1)) + \vyopog_k(t + 1) \cdot \dv{h\qty(\vsopblk{k}_i(t + 1))}{\vsopblk{k}_i(t + 1)} \cdot \cancelto{\aptr 0}{\dv{\vsopblk{k}_i(t + 1)}{\vWophid_{p, q}}} \\
                                                   & \aptr 0 \qqtext{where} \begin{dcases}
                                                                              i \in \Set{1, \dots, \dblk} \\
                                                                              k \in \Set{1, \dots, \nblk} \\
                                                                              p \in \Set{1, \dots, \dhid} \\
                                                                              q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                              t \in \Set{0, \dots, \cT - 1}
                                                                            \end{dcases}.
      \end{align*}
    \]

  觀察前面的推導結果，可以發現參數 :math:`\vWophid` 僅剩下一個管道可以取得由誤差造成的微分，即是透過 conventional hidden units。
  所以接下來我們推導 :math:`\vWophid` 對於 conventional hidden units 的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\vyophid_i(t + 1)}{\vWophid_{p, q}} & = \dv{\vyophid_i(t + 1)}{\vzophid_i(t + 1)} \cdot \dv{\vzophid_i(t + 1)}{\vWophid_{p, q}} \\
                                                & = {f^\ophid}'\qty(\vzophid_i(t + 1)) \cdot \qty[\delta_{i, p} \cdot \vxt_q(t) + \sum_{j = 1}^{\din + \dhid + \nblk \times (2 + \dblk)} \qty[\vWophid_{i, j} \cdot \dv{\vxt_j(t)}{\vWophid_{p, q}}]] \\
                                                & \qqtext{where} \begin{dcases}
                                                                   i \in \Set{1, \dots, \dhid} \\
                                                                   p \in \Set{1, \dots, \dhid} \\
                                                                   q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                   t \in \Set{0, \dots, \cT - 1}
                                                                 \end{dcases}.
      \end{align*}
    \]

  可以發現 :math:`\vWophid` 對於 conventional hidden units 的全微分會有 BPTT 的問題，因此作者在論文中提出額外丟棄 conventional hidden units 的部份微分，結果如下：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\vyophid_i(t + 1)}{\vWophid_{p, q}} & = {f^\ophid}'\qty(\vzophid_i(t + 1)) \cdot \qty[\delta_{i, p} \cdot \vxt_q(t) + \cancelto{\aptr 0}{\sum_{j = 1}^{\din + \dhid + \nblk \times (2 + \dblk)} \qty[\vWophid_{i, j} \cdot \dv{\vxt_j(t)}{\vWophid_{p, q}}]}] \\
                                                & \aptr {f^\ophid}'\qty(\vzophid_i(t + 1)) \cdot \delta_{i, p} \cdot \vxt_q(t) \\
                                                & \qqtext{where} \begin{dcases}
                                                                   i \in \Set{1, \dots, \dhid} \\
                                                                   p \in \Set{1, \dots, \dhid} \\
                                                                   q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                   t \in \Set{0, \dots, \cT - 1}
                                                                 \end{dcases}.
      \end{align*}
    \]

  .. note::

    上式就是論文中的 A.9 式。

  最後我們可以推得 :math:`\vWophid` 相對於誤差的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        & \dv{\cL\qty(\vy(t + 1) - \vyh(t + 1))}{\vWophid_{p, q}} \\
        & = \sum_{i = 1}^\dout \dv{\frac{1}{2} \qty(\vy_i(t + 1) - \vyh_i(t + 1))^2}{\vWophid_{p, q}} \\
        & = \sum_{i = 1}^\dout \qty[\dv{\frac{1}{2} \qty(\vy_i(t + 1) - \vyh_i(t + 1))^2}{\vy_i(t + 1)} \cdot \dv{\vy_i(t + 1)}{\vzopout_i(t + 1)} \cdot \sum_{j = 1}^{\din + \dhid + \nblk \times \dblk} \qty[\dv{\vzopout_i(t + 1)}{\vxopout_j(t + 1)} \cdot \cancelto{\aptr 0}{\dv{\vxopout_j(t + 1)}{\vWophid_{p, q}}}]] \\
        & \aptr \sum_{i = 1}^\dout \qty[\qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \sum_{j = 1}^\dhid \qty[\vWopout_{i, \din + j} \cdot \dv{\vyophid_j(t + 1)}{\vWophid_{p, q}}]] \\
        & \aptr \sum_{i = 1}^\dout \qty[\qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \sum_{j = 1}^\dhid \qty[\vWopout_{i, \din + j} \cdot {f^\ophid}'\qty(\vzophid_j(t + 1)) \cdot \delta_{j, p} \cdot \vxt_q(t)]] \\
        & = \sum_{i = 1}^\dout \qty[\qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, \din + p} \cdot {f^\ophid}'\qty(\vzophid_p(t + 1)) \cdot \vxt_q(t)] \\
        & = \qty(\sum_{i = 1}^\dout \qty[\qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, \din + p}]) \cdot {f^\ophid}'\qty(\vzophid_p(t + 1)) \cdot \vxt_q(t) \\
        & \qqtext{where} \begin{dcases}
                           p \in \Set{1, \dots, \dhid} \\
                           q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                           t \in \Set{0, \dots, \cT - 1}
                         \end{dcases}.
      \end{align*}
    \]

.. note::

  :math:`\eqref{15}` 就是論文中 A.8 式 :math:`l` otherwise 的 case。

:math:`\vWopog` 相對於誤差的微分近似值
---------------------------------------

.. math::
  :nowrap:

  \[
    \begin{align*}
      & \dv{\cL\qty(\vy(t + 1) - \vyh(t + 1))}{\vWopog_{p, q}} \aptr \qty(\sum_{j = 1}^\dblk \qty[\sum_{i = 1}^\dout \qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, \din + \dhid + (p - 1) \times \dblk + j}] \cdot h\qty(\vsopblk{p}_j(t + 1))) \cdot {f^\opog}'\qty(\vzopog_p(t + 1)) \cdot \vxt_q(t) \\
      & \qqtext{where} \begin{dcases}
                         p \in \Set{1, \dots, \nblk} \\
                         q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                         t \in \Set{0, \dots, \cT - 1}
                       \end{dcases}.
    \end{align*}
    \tag{16}\label{16}
  \]

.. dropdown:: 推導式子 :math:`\eqref{16}`

  根據式子 :math:`\eqref{12}` 我們可以求得 :math:`\vWopog` 相對於 conventional hidden units 的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\vyophid_i(t + 1)}{\vWopog_{p, q}} & = \dv{\vyophid_i(t + 1)}{\vzophid_i(t + 1)} \cdot \sum_{j = 1}^{\din + \dhid + \nblk \times (2 + \dblk)} \qty[\cancelto{\aptr 0}{\dv{\vzophid_i(t + 1)}{\vxt_j(t)}} \cdot \dv{\vxt_j(t)}{\vWopog_{p, q}}] \\
                                               & \aptr 0 \qqtext{where} \begin{dcases}
                                                                          i \in \Set{1, \dots, \dhid} \\
                                                                          p \in \Set{1, \dots, \nblk} \\
                                                                          q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                          t \in \Set{0, \dots, \cT - 1}
                                                                        \end{dcases}.
      \end{align*}
    \]

  這代表在丟棄部份微分後 :math:`\vWopog` 將\ **無法**\透過 conventional hidden units 取得資訊。
  同理，我們也可以求得 :math:`\vWopog` 相對於 input gate units 的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\vyopig_k(t + 1)}{\vWopog_{p, q}} & = \dv{\vyopig_k(t + 1)}{\vzopig_k(t + 1)} \cdot \sum_{j = 1}^{\din + \dhid + \nblk \times (2 + \dblk)} \qty[\cancelto{\aptr 0}{\dv{\vzopig_k(t + 1)}{\vxt_j(t)}} \cdot \dv{\vxt_j(t)}{\vWopog_{p, q}}] \\
                                              & \aptr 0 \qqtext{where} \begin{dcases}
                                                                         k \in \Set{1, \dots, \nblk} \\
                                                                         p \in \Set{1, \dots, \nblk} \\
                                                                         q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                         t \in \Set{0, \dots, \cT - 1}
                                                                       \end{dcases}.
      \end{align*}
    \]

  我們可以得到相同的結論：在丟棄部份微分後 :math:`\vWopog` 將\ **無法**\透過 input gate units 取得資訊。
  結合式子 :math:`\eqref{12}` 與前面的推導，我們可以得出 :math:`\vWopog` 相對於 memory cell internal states 的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\vsopblk{k}_i(t + 1)}{\vWopog_{p, q}} & = \dv{\vsopblk{k}_i(t)}{\vWopog_{p, q}} + \cancelto{\aptr 0}{\dv{\vyopig_k(t + 1)}{\vWopog_{p, q}}} \cdot g\qty(\vzopblk{k}_i(t + 1)) + \vyopig_k(t + 1) \cdot \dv{g\qty(\vzopblk{k}_i(t + 1))}{\vzopblk{k}_i(t + 1)} \cdot \sum_{j = 1}^{\din + \dhid + \nblk \times (2 + \dblk)} \qty[\cancelto{\aptr 0}{\dv{\vzopblk{k}_i(t + 1)}{\vxt_j(t)}} \cdot \dv{\vxt_j(t)}{\vWopog_{p, q}}] \\
                                                  & \aptr \dv{\vsopblk{k}_i(t)}{\vWopog_{p, q}} \\
                                                  & \aptr \dv{\vsopblk{k}_i(t - 1)}{\vWopog_{p, q}} \\
                                                  & \vdots \\
                                                  & \aptr \cancelto{0}{\dv{\vsopblk{k}_i(0)}{\vWopog_{p, q}}} \\
                                                  & = 0 \qqtext{where} \begin{dcases}
                                                                         i \in \Set{1, \dots, \dblk} \\
                                                                         k \in \Set{1, \dots, \nblk} \\
                                                                         p \in \Set{1, \dots, \nblk} \\
                                                                         q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                         t \in \Set{0, \dots, \cT - 1}
                                                                       \end{dcases}.
      \end{align*}
    \]

  上式告訴我們，在丟棄部份微分後 :math:`\vWopog` 將\ **無法**\透過 memory cell internal states 取得資訊。
  直覺上 :math:`\vWopog` 唯一能夠取得資訊的管道只有 output gate units。
  所以接下來我們推導 :math:`\vWopog` 相對於 output gate units 的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\vyopog_k(t + 1)}{\vWopog_{p, q}} & = \dv{\vyopog_k(t + 1)}{\vzopog_k(t + 1)} \cdot \dv{\vzopog_k(t + 1)}{\vWopig_{p, q}} \\
                                              & = {f^\opog}'\qty(\vzopog_k(t + 1)) \cdot \qty[\delta_{k, p} \cdot \vxt_q(t) + \sum_{j = 1}^{\din + \dhid + \nblk \times (2 + \dblk)} \qty[\vWopog_{k, j} \cdot \dv{\vxt_j(t)}{\vWopog_{p, q}}]] \\
                                              & \qqtext{where} \begin{dcases}
                                                                 k \in \Set{1, \dots, \nblk} \\
                                                                 p \in \Set{1, \dots, \nblk} \\
                                                                 q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                 t \in \Set{0, \dots, \cT - 1}
                                                               \end{dcases}.
      \end{align*}
    \]

  可以發現 :math:`\vWopog` 對於 output gate units 的全微分會有 BPTT 的問題，因此作者在論文中提出額外丟棄 output gate units 的部份微分，結果如下：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\vyopog_k(t + 1)}{\vWopog_{p, q}} & = {f^\opog}'\qty(\vzopog_k(t + 1)) \cdot \qty[\delta_{k, p} \cdot \vxt_q(t) + \cancelto{\aptr 0}{\sum_{j = 1}^{\din + \dhid + \nblk \times (2 + \dblk)} \qty[\vWopog_{k, j} \cdot \dv{\vxt_j(t)}{\vWopog_{p, q}}]}] \\
                                              & \aptr {f^\opog}'\qty(\vzopog_k(t + 1)) \cdot \delta_{k, p} \cdot \vxt_q(t) \\
                                              & \qqtext{where} \begin{dcases}
                                                                 k \in \Set{1, \dots, \nblk} \\
                                                                 p \in \Set{1, \dots, \nblk} \\
                                                                 q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                 t \in \Set{0, \dots, \cT - 1}
                                                               \end{dcases}.
      \end{align*}
    \]

  .. note::

    上式就是論文中的 A.11 式。

  使用前述推導結果可以幫助我們推得 :math:`\vWopog` 相對於 memory cell activation blocks 的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\vyopblk{k}_i(t + 1)}{\vWopog_{p, q}} & = \dv{\vyopog_k(t + 1)}{\vWopog_{p, q}} \cdot h\qty(\vsopblk{k}_i(t + 1)) + \vyopog_k(t + 1) \cdot \dv{h\qty(\vsopblk{k}_i(t + 1))}{\vsopblk{k}_i(t + 1)} \cdot \cancelto{\aptr 0}{\dv{\vsopblk{k}_i(t + 1)}{\vWopog_{p, q}}} \\
                                                  & \aptr {f^\opog}'\qty(\vzopog_k(t + 1)) \cdot \delta_{k, p} \cdot \vxt_q(t) \cdot h\qty(\vsopblk{k}_i(t + 1)) \\
                                                  & \qqtext{where} \begin{dcases}
                                                                     i \in \Set{1, \dots, \dblk} \\
                                                                     k \in \Set{1, \dots, \nblk} \\
                                                                     p \in \Set{1, \dots, \nblk} \\
                                                                     q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                     t \in \Set{0, \dots, \cT - 1}
                                                                   \end{dcases}.
      \end{align*}
    \]

  .. note::

    上式就是論文中的 A.13 式 :math:`\delta_{\opout_j l} = 1` 且 :math:`\delta_{\opin_j l} = \delta_{c_j^v l} = 0` 的部份 。

  最後我們推得 :math:`\vWopog` 相對於誤差的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        & \dv{\cL\qty(\vy(t + 1) - \vyh(t + 1))}{\vWopog_{p, q}} \\
        & = \sum_{i = 1}^\dout \dv{\frac{1}{2} \qty(\vy_i(t + 1) - \vyh_i(t + 1))^2}{\vWopog_{p, q}} \\
        & = \sum_{i = 1}^\dout \qty[\dv{\frac{1}{2} \qty(\vy_i(t + 1) - \vyh_i(t + 1))^2}{\vy_i(t + 1)} \cdot \dv{\vy_i(t + 1)}{\vzopout_i(t + 1)} \cdot \sum_{j = 1}^{\din + \dhid + \nblk \times \dblk} \qty[\dv{\vzopout_i(t + 1)}{\vxopout_j(t + 1)} \cdot \cancelto{\aptr 0}{\dv{\vxopout_j(t + 1)}{\vWopog_{p, q}}}]] \\
        & \aptr \sum_{i = 1}^\dout \qty[\qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \sum_{k = 1}^\nblk \sum_{j = 1}^\dblk \qty[\vWopout_{i, \din + \dhid + (k - 1) \times \dblk + j} \cdot \dv{\vyopblk{k}_j(t + 1)}{\vWopog_{p, q}}]] \\
        & \aptr \sum_{i = 1}^\dout \qty[\qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \sum_{k = 1}^\nblk \sum_{j = 1}^\dblk \qty[\vWopout_{i, \din + \dhid + (k - 1) \times \dblk + j} \cdot {f^\opog}'\qty(\vzopog_k(t + 1)) \cdot \delta_{k, p} \cdot \vxt_q(t) \cdot h\qty(\vsopblk{k}_j(t + 1))]] \\
        & = \sum_{i = 1}^\dout \qty[\qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \sum_{j = 1}^\dblk \qty[\vWopout_{i, \din + \dhid + (p - 1) \times \dblk + j} \cdot {f^\opog}'\qty(\vzopog_p(t + 1)) \cdot \vxt_q(t) \cdot h\qty(\vsopblk{p}_j(t + 1))]] \\
        & = \qty(\sum_{i = 1}^\dout \qty[\qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \sum_{j = 1}^\dblk \qty[\vWopout_{i, \din + \dhid + (p - 1) \times \dblk + j} \cdot h\qty(\vsopblk{p}_j(t + 1))]]) \cdot {f^\opog}'\qty(\vzopog_p(t + 1)) \cdot \vxt_q(t) \\
        & = \qty(\sum_{j = 1}^\dblk \qty[\sum_{i = 1}^\dout \qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, \din + \dhid + (p - 1) \times \dblk + j}] \cdot h\qty(\vsopblk{p}_j(t + 1))) \cdot {f^\opog}'\qty(\vzopog_p(t + 1)) \cdot \vxt_q(t) \\
        & \qqtext{where} \begin{dcases}
                           p \in \Set{1, \dots, \nblk} \\
                           q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                           t \in \Set{0, \dots, \cT - 1}
                         \end{dcases}.
      \end{align*}
    \]

.. note::

  :math:`\eqref{16}` 就是論文中 A.8 式 :math:`l = \opout_j` 的 case。

:math:`\vWopig` 相對於誤差的微分近似值
---------------------------------------

.. math::
  :nowrap:

  \[
    \begin{align*}
      & \dv{\cL\qty(\vy(t + 1) - \vyh(t + 1))}{\vWopig_{p, q}} \aptr \qty(\sum_{j = 1}^\dblk \qty[\sum_{i = 1}^\dout \qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, \din + \dhid + (p - 1) \times \dblk + j}] \cdot h'\qty(\vsopblk{p}_j(t + 1)) \cdot \sum_{t^\star = 0}^t \qty[{f^\opig}'\qty(\vzopig_p(t^\star + 1)) \cdot \vxt_q(t^\star) \cdot g\qty(\vzopblk{p}_j(t^\star + 1))]) \cdot \vyopog_p(t + 1) \\
      & \qqtext{where} \begin{dcases}
                         p \in \Set{1, \dots, \nblk} \\
                         q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                         t \in \Set{0, \dots, \cT - 1}
                       \end{dcases}.
    \end{align*}
    \tag{17}\label{17}
  \]

.. dropdown:: 推導式子 :math:`\eqref{17}`

  根據式子 :math:`\eqref{12}` 我們可以求得 :math:`\vWopig` 相對於 conventional hidden units 的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\vyophid_i(t + 1)}{\vWopig_{p, q}} & = \dv{\vyophid_i(t + 1)}{\vzophid_i(t + 1)} \cdot \sum_{j = 1}^{\din + \dhid + \nblk \times (2 + \dblk)} \qty[\cancelto{\aptr 0}{\dv{\vzophid_i(t + 1)}{\vxt_j(t)}} \cdot \dv{\vxt_j(t)}{\vWopig_{p, q}}] \\
                                               & \aptr 0 \qqtext{where} \begin{dcases}
                                                                          i \in \Set{1, \dots, \dhid} \\
                                                                          p \in \Set{1, \dots, \nblk} \\
                                                                          q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                          t \in \Set{0, \dots, \cT - 1}
                                                                        \end{dcases}.
      \end{align*}
    \]

  這代表在丟棄部份微分後 :math:`\vWopig` 將\ **無法**\透過 conventional hidden units 取得資訊。
  同理，我們也可以求得 :math:`\vWopig` 相對於 output gate units 的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\vyopog_k(t + 1)}{\vWopig_{p, q}} & = \dv{\vyopog_k(t + 1)}{\vzopog_k(t + 1)} \cdot \sum_{j = 1}^{\din + \dhid + \nblk \times (2 + \dblk)} \qty[\cancelto{\aptr 0}{\dv{\vzopog_k(t + 1)}{\vxt_j(t)}} \cdot \dv{\vxt_j(t)}{\vWopig_{p, q}}] \\
                                              & \aptr 0 \qqtext{where} \begin{dcases}
                                                                         k \in \Set{1, \dots, \nblk} \\
                                                                         p \in \Set{1, \dots, \nblk} \\
                                                                         q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                         t \in \Set{0, \dots, \cT - 1}
                                                                       \end{dcases}.
      \end{align*}
    \]

  我們可以得到相同的結論：在丟棄部份微分後 :math:`\vWopig` 將\ **無法**\透過 output gate units 取得資訊。
  直覺上我們認為 :math:`\vWopig` 應該可以透過 input gate units 取得資訊。
  所以接下來我們推導 :math:`\vWopig` 相對於 input gate units 的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\vyopig_k(t + 1)}{\vWopig_{p, q}} & = \dv{\vyopig_k(t + 1)}{\vzopig_k(t + 1)} \cdot \dv{\vzopig_k(t + 1)}{\vWopig_{p, q}} \\
                                              & = {f^\opig}'\qty(\vzopig_k(t + 1)) \cdot \qty[\delta_{k, p} \cdot \vxt_q(t) + \sum_{j = 1}^{\din + \dhid + \nblk \times (2 + \dblk)} \qty[\vWopig_{k, j} \cdot \dv{\vxt_j(t)}{\vWopig_{p, q}}]] \\
                                              & \qqtext{where} \begin{dcases}
                                                                 k \in \Set{1, \dots, \nblk} \\
                                                                 p \in \Set{1, \dots, \nblk} \\
                                                                 q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                 t \in \Set{0, \dots, \cT - 1}
                                                               \end{dcases}.
      \end{align*}
    \]

  可以發現 :math:`\vWopig` 對於 input gate units 的全微分會有 BPTT 的問題，因此作者在論文中提出額外丟棄 input gate units 的部份微分，結果如下：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\vyopig_k(t + 1)}{\vWopig_{p, q}} & = {f^\opig}'\qty(\vzopig_k(t + 1)) \cdot \qty[\delta_{k, p} \cdot \vxt_q(t) + \cancelto{\aptr 0}{\sum_{j = 1}^{\din + \dhid + \nblk \times (2 + \dblk)} \qty[\vWopig_{k, j} \cdot \dv{\vxt_j(t)}{\vWopig_{p, q}}]}] \\
                                              & \aptr {f^\opig}'\qty(\vzopig_k(t + 1)) \cdot \delta_{k, p} \cdot \vxt_q(t) \\
                                              & \qqtext{where} \begin{dcases}
                                                                 k \in \Set{1, \dots, \nblk} \\
                                                                 p \in \Set{1, \dots, \nblk} \\
                                                                 q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                 t \in \Set{0, \dots, \cT - 1}
                                                               \end{dcases}.
      \end{align*}
    \]

  .. note::

    上式就是論文中的 A.10 式。

  結合式子 :math:`\eqref{12}` 與前面的推導，我們可以得出 :math:`\vWopig` 相對於 memory cell internal states 的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\vsopblk{k}_i(t + 1)}{\vWopig_{p, q}} & = \dv{\vsopblk{k}_i(t)}{\vWopig_{p, q}} + \dv{\vyopig_k(t + 1)}{\vWopig_{p, q}} \cdot g\qty(\vzopblk{k}_i(t + 1)) + \vyopig_k(t + 1) \cdot \dv{g\qty(\vzopblk{k}_i(t + 1))}{\vzopblk{k}_i(t + 1)} \cdot \sum_{j = 1}^{\din + \dhid + \nblk \times (2 + \dblk)} \qty[\cancelto{\aptr 0}{\dv{\vzopblk{k}_i(t + 1)}{\vxt_j(t)}} \cdot \dv{\vxt_j(t)}{\vWopig_{p, q}}] \\
                                                  & \aptr \dv{\vsopblk{k}_i(t)}{\vWopig_{p, q}} + {f^\opig}'\qty(\vzopig_k(t + 1)) \cdot \delta_{k, p} \cdot \vxt_q(t) \cdot g\qty(\vzopblk{k}_i(t + 1)) \\
                                                  & \aptr \dv{\vsopblk{k}_i(t - 1)}{\vWopig_{p, q}} + \sum_{t^\star = t - 1}^t \qty[{f^\opig}'\qty(\vzopig_k(t^\star + 1)) \cdot \delta_{k, p} \cdot \vxt_q(t^\star) \cdot g\qty(\vzopblk{k}_i(t^\star + 1))] \\
                                                  & \vdots \\
                                                  & \aptr \cancelto{0}{\dv{\vsopblk{k}_i(0)}{\vWopig_{p, q}}} + \sum_{t^\star = 0}^t \qty[{f^\opig}'\qty(\vzopig_k(t^\star + 1)) \cdot \delta_{k, p} \cdot \vxt_q(t^\star) \cdot g\qty(\vzopblk{k}_i(t^\star + 1))] \\
                                                  & = \sum_{t^\star = 0}^t \qty[{f^\opig}'\qty(\vzopig_k(t^\star + 1)) \cdot \delta_{k, p} \cdot \vxt_q(t^\star) \cdot g\qty(\vzopblk{k}_i(t^\star + 1))] \\
                                                  & \qqtext{where} \begin{dcases}
                                                                     i \in \Set{1, \dots, \dblk} \\
                                                                     k \in \Set{1, \dots, \nblk} \\
                                                                     p \in \Set{1, \dots, \nblk} \\
                                                                     q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                     t \in \Set{0, \dots, \cT - 1}
                                                                   \end{dcases}.
      \end{align*}
    \]

  .. note::

    上式就是論文中的 A.12 式 :math:`\delta_{\opin_j l} = 1` 且 :math:`\delta_{c_j^v l} = 0` 的部份 。

  可以發現 :math:`\vWopig` 透過 memory cell internal states 得到的資訊其實都是來自於過去微分近似值的累加結果。
  實際上在執行參數更新演算法時只需要儲存過去累加而得的結果在加上當前計算結果，就可以得到最新的參數更新方向。
  使用前述推導結果我們可以得到 :math:`\vWopig` 相對於 memory cell activation blocks 的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\vyopblk{k}_i(t + 1)}{\vWopig_{p, q}} & = \cancelto{\aptr 0}{\dv{\vyopog_k(t + 1)}{\vWopig_{p, q}}} \cdot h\qty(\vsopblk{k}_i(t + 1)) + \vyopog_k(t + 1) \cdot \dv{h\qty(\vsopblk{k}_i(t + 1))}{\vsopblk{k}_i(t + 1)} \cdot \dv{\vsopblk{k}_i(t + 1)}{\vWopig_{p, q}} \\
                                                  & \aptr \vyopog_k(t + 1) \cdot h'\qty(\vsopblk{k}_i(t + 1)) \cdot \sum_{t^\star = 0}^t \qty[{f^\opig}'\qty(\vzopig_k(t^\star + 1)) \cdot \delta_{k, p} \cdot \vxt_q(t^\star) \cdot g\qty(\vzopblk{k}_i(t^\star + 1))] \\
                                                  & \qqtext{where} \begin{dcases}
                                                                     i \in \Set{1, \dots, \dblk} \\
                                                                     k \in \Set{1, \dots, \nblk} \\
                                                                     p \in \Set{1, \dots, \nblk} \\
                                                                     q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                     t \in \Set{0, \dots, \cT - 1}
                                                                   \end{dcases}.
      \end{align*}
    \]

  .. note::

    上式就是論文中的 A.13 式 :math:`\delta_{\opin_j l} = 1` 且 :math:`\delta_{\opout_j l} = \delta_{c_j^v l} = 0` 的部份 。

  同前述結論，只需要儲存過去累加而得的結果再加上當前計算結果，最後乘上一些當前的計算狀態，就可以得到最新的參數更新方向。
  最後我們推得 :math:`\vWopig` 相對於誤差的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        & \dv{\cL\qty(\vy(t + 1) - \vyh(t + 1))}{\vWopig_{p, q}} \\
        & = \sum_{i = 1}^\dout \dv{\frac{1}{2} \qty(\vy_i(t + 1) - \vyh_i(t + 1))^2}{\vWopig_{p, q}} \\
        & = \sum_{i = 1}^\dout \qty[\dv{\frac{1}{2} \qty(\vy_i(t + 1) - \vyh_i(t + 1))^2}{\vy_i(t + 1)} \cdot \dv{\vy_i(t + 1)}{\vzopout_i(t + 1)} \cdot \sum_{j = 1}^{\din + \dhid + \nblk \times \dblk} \qty[\dv{\vzopout_i(t + 1)}{\vxopout_j(t + 1)} \cdot \cancelto{\aptr 0}{\dv{\vxopout_j(t + 1)}{\vWopig_{p, q}}}]] \\
        & \aptr \sum_{i = 1}^\dout \qty[\qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \sum_{k = 1}^\nblk \sum_{j = 1}^\dblk \qty[\vWopout_{i, \din + \dhid + (k - 1) \times \dblk + j} \cdot \dv{\vyopblk{k}_j(t + 1)}{\vWopig_{p, q}}]] \\
        & \aptr \sum_{i = 1}^\dout \qty[\qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \sum_{k = 1}^\nblk \sum_{j = 1}^\dblk \qty[\vWopout_{i, \din + \dhid + (k - 1) \times \dblk + j} \cdot \vyopog_k(t + 1) \cdot h'\qty(\vsopblk{k}_j(t + 1)) \cdot \sum_{t^\star = 0}^t \qty[{f^\opig}'\qty(\vzopig_k(t^\star + 1)) \cdot \delta_{k, p} \cdot \vxt_q(t^\star) \cdot g\qty(\vzopblk{k}_j(t^\star + 1))]]] \\
        & = \sum_{i = 1}^\dout \qty[\qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \sum_{j = 1}^\dblk \qty[\vWopout_{i, \din + \dhid + (p - 1) \times \dblk + j} \cdot \vyopog_p(t + 1) \cdot h'\qty(\vsopblk{p}_j(t + 1)) \cdot \sum_{t^\star = 0}^t \qty[{f^\opig}'\qty(\vzopig_p(t^\star + 1)) \cdot \vxt_q(t^\star) \cdot g\qty(\vzopblk{p}_j(t^\star + 1))]]] \\
        & = \qty(\sum_{i = 1}^\dout \qty[\qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \sum_{j = 1}^\dblk \qty[\vWopout_{i, \din + \dhid + (p - 1) \times \dblk + j} \cdot h'\qty(\vsopblk{p}_j(t + 1)) \cdot \sum_{t^\star = 0}^t \qty[{f^\opig}'\qty(\vzopig_p(t^\star + 1)) \cdot \vxt_q(t^\star) \cdot g\qty(\vzopblk{p}_j(t^\star + 1))]]]) \cdot \vyopog_p(t + 1) \\
        & = \qty(\sum_{j = 1}^\dblk \qty[\sum_{i = 1}^\dout \qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, \din + \dhid + (p - 1) \times \dblk + j}] \cdot h'\qty(\vsopblk{p}_j(t + 1)) \cdot \sum_{t^\star = 0}^t \qty[{f^\opig}'\qty(\vzopig_p(t^\star + 1)) \cdot \vxt_q(t^\star) \cdot g\qty(\vzopblk{p}_j(t^\star + 1))]) \cdot \vyopog_p(t + 1) \\
        & \qqtext{where} \begin{dcases}
                           p \in \Set{1, \dots, \nblk} \\
                           q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                           t \in \Set{0, \dots, \cT - 1}
                         \end{dcases}.
      \end{align*}
    \]

.. note::

  :math:`\eqref{17}` 就是論文中 A.8 式 :math:`l = \opin_j` 的 case。

:math:`\vWopblk{k}` 相對於誤差的微分近似值
-------------------------------------------

.. math::
  :nowrap:

  \[
    \begin{align*}
      & \dv{\cL\qty(\vy(t + 1) - \vyh(t + 1))}{\vWopblk{k}_{p, q}} \aptr \qty[\sum_{i = 1}^\dout \qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, \din + \dhid + (k - 1) \times \dblk + p}] \cdot \qty[\sum_{t^\star = 0}^t \vyopig_k(t^\star + 1) \cdot g'\qty(\vzopblk{k}_p(t^\star + 1)) \cdot \vxt_q(t^\star)] \cdot \vyopog_k(t + 1) \cdot h'\qty(\vsopblk{k}_p(t + 1)) \\
      & \qqtext{where} \begin{dcases}
                         k \in \Set{1, \dots, \nblk} \\
                         p \in \Set{1, \dots, \dblk} \\
                         q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                         t \in \Set{0, \dots, \cT - 1}
                       \end{dcases}.
    \end{align*}
    \tag{18}\label{18}
  \]

.. dropdown:: 推導式子 :math:`\eqref{18}`

  根據式子 :math:`\eqref{12}` 我們可以求得 :math:`\vWopblk{k}` 相對於 conventional hidden units 的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\vyophid_i(t + 1)}{\vWopblk{k}_{p, q}} & = \dv{\vyophid_i(t + 1)}{\vzophid_i(t + 1)} \cdot \sum_{j = 1}^{\din + \dhid + \nblk \times (2 + \dblk)} \qty[\cancelto{\aptr 0}{\dv{\vzophid_i(t + 1)}{\vxt_j(t)}} \cdot \dv{\vxt_j(t)}{\vWopblk{k}_{p, q}}] \\
                                               & \aptr 0 \qqtext{where} \begin{dcases}
                                                                          i \in \Set{1, \dots, \dhid} \\
                                                                          k \in \Set{1, \dots, \nblk} \\
                                                                          p \in \Set{1, \dots, \dblk} \\
                                                                          q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                          t \in \Set{0, \dots, \cT - 1}
                                                                        \end{dcases}.
      \end{align*}
    \]

  這代表在丟棄部份微分後 :math:`\vWopblk{k}` 將\ **無法**\透過 conventional hidden units 取得資訊。
  同理，我們也可以求得 :math:`\vWopblk{k}` 相對於 input/output gate units 的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\vyopig_{k^\star}(t + 1)}{\vWopblk{k}_{p, q}} & = \dv{\vyopig_{k^\star}(t + 1)}{\vzopig_{k^\star}(t + 1)} \cdot \sum_{j = 1}^{\din + \dhid + \nblk \times (2 + \dblk)} \qty[\cancelto{\aptr 0}{\dv{\vzopig_{k^\star}(t + 1)}{\vxt_j(t)}} \cdot \dv{\vxt_j(t)}{\vWopblk{k}_{p, q}}] \\
                                                          & \aptr 0 \qqtext{where} \begin{dcases}
                                                                                     k \in \Set{1, \dots, \nblk} \\
                                                                                     k^\star \in \Set{1, \dots, \nblk} \\
                                                                                     p \in \Set{1, \dots, \dblk} \\
                                                                                     q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                                     t \in \Set{0, \dots, \cT - 1}
                                                                                   \end{dcases}. \\
        \dv{\vyopog_{k^\star}(t + 1)}{\vWopblk{k}_{p, q}} & = \dv{\vyopog_{k^\star}(t + 1)}{\vzopog_{k^\star}(t + 1)} \cdot \sum_{j = 1}^{\din + \dhid + \nblk \times (2 + \dblk)} \qty[\cancelto{\aptr 0}{\dv{\vzopog_{k^\star}(t + 1)}{\vxt_j(t)}} \cdot \dv{\vxt_j(t)}{\vWopblk{k}_{p, q}}] \\
                                                          & \aptr 0 \qqtext{where} \begin{dcases}
                                                                                     k \in \Set{1, \dots, \nblk} \\
                                                                                     k^\star \in \Set{1, \dots, \nblk} \\
                                                                                     p \in \Set{1, \dots, \dblk} \\
                                                                                     q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                                     t \in \Set{0, \dots, \cT - 1}
                                                                                   \end{dcases}.
      \end{align*}
    \]

  我們可以得到相同的結論：在丟棄部份微分後 :math:`\vWopblk{k}` 將\ **無法**\透過 input/output gate units 取得資訊。
  直覺上我們認為 :math:`\vWopblk{k}` 應該可以透過 memory cell internal states 取得資訊。
  所以接下來我們推導 :math:`\vWopblk{k}` 相對於 memory cell internal states 的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\vsopblk{k^\star}_i(t + 1)}{\vWopblk{k}_{p, q}} & = \dv{\vsopblk{k^\star}_i(t)}{\vWopblk{k}_{p, q}} + \cancelto{\aptr 0}{\dv{\vyopig_{k^\star}(t + 1)}{\vWopblk{k}_{p, q}}} \cdot g\qty(\vzopblk{k^\star}_i(t + 1)) + \vyopig_{k^\star}(t + 1) \cdot \dv{g\qty(\vzopblk{k^\star}_i(t + 1))}{\vzopblk{k^\star}_i(t + 1)} \cdot \qty[\delta_{k^\star, k} \cdot \delta_{i, p} \cdot \vxt_q(t) + \sum_{j = 1}^{\din + \dhid + \nblk \times (2 + \dblk)} \qty[\vWopblk{k^\star}_{i, j} \cdot \dv{\vxt_j(t)}{\vWopblk{k}_{p, q}}]] \\
                                                            & \aptr \dv{\vsopblk{k^\star}_i(t)}{\vWopblk{k}_{p, q}} + \vyopig_{k^\star}(t + 1) \cdot g'\qty(\vzopblk{k^\star}_i(t + 1)) \cdot \qty[\delta_{k^\star, k} \cdot \delta_{i, p} \cdot \vxt_q(t) + \sum_{j = 1}^{\din + \dhid + \nblk \times (2 + \dblk)} \qty[\vWopblk{k^\star}_{i, j} \cdot \dv{\vxt_j(t)}{\vWopblk{k}_{p, q}}]] \\
                                                            & \aptr \dv{\vsopblk{k^\star}_i(t - 1)}{\vWopblk{k}_{p, q}} + \sum_{t^\star = t - 1}^t \qty[\vyopig_{k^\star}(t^\star + 1) \cdot g'\qty(\vzopblk{k^\star}_i(t^\star + 1)) \cdot \qty[\delta_{k^\star, k} \cdot \delta_{i, p} \cdot \vxt_q(t^\star) + \sum_{j = 1}^{\din + \dhid + \nblk \times (2 + \dblk)} \qty[\vWopblk{k^\star}_{i, j} \cdot \dv{\vxt_j(t^\star)}{\vWopblk{k}_{p, q}}]]] \\
                                                            & \vdots \\
                                                            & \aptr \cancelto{0}{\dv{\vsopblk{k^\star}_i(0)}{\vWopblk{k}_{p, q}}} + \sum_{t^\star = 0}^t \qty[\vyopig_{k^\star}(t^\star + 1) \cdot g'\qty(\vzopblk{k^\star}_i(t^\star + 1)) \cdot \qty[\delta_{k^\star, k} \cdot \delta_{i, p} \cdot \vxt_q(t^\star) + \sum_{j = 1}^{\din + \dhid + \nblk \times (2 + \dblk)} \qty[\vWopblk{k^\star}_{i, j} \cdot \dv{\vxt_j(t^\star)}{\vWopblk{k}_{p, q}}]]] \\
                                                            & = \sum_{t^\star = 0}^t \qty[\vyopig_{k^\star}(t^\star + 1) \cdot g'\qty(\vzopblk{k^\star}_i(t^\star + 1)) \cdot \qty[\delta_{k^\star, k} \cdot \delta_{i, p} \cdot \vxt_q(t^\star) + \sum_{j = 1}^{\din + \dhid + \nblk \times (2 + \dblk)} \qty[\vWopblk{k^\star}_{i, j} \cdot \dv{\vxt_j(t^\star)}{\vWopblk{k}_{p, q}}]]] \\
                                                            & \qqtext{where} \begin{dcases}
                                                                               i \in \Set{1, \dots, \dblk} \\
                                                                               k \in \Set{1, \dots, \nblk} \\
                                                                               k^\star \in \Set{1, \dots, \nblk} \\
                                                                               p \in \Set{1, \dots, \dblk} \\
                                                                               q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                               t \in \Set{0, \dots, \cT - 1}
                                                                             \end{dcases}.
      \end{align*}
    \]

  可以發現 :math:`\vWopblk{k}` 對於 memory cell internal states 的全微分會有 BPTT 的問題，因此作者在論文中提出額外丟棄 memory cell internal states 的部份微分，結果如下：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\vsopblk{k^\star}_i(t + 1)}{\vWopblk{k}_{p, q}} & \aptr \sum_{t^\star = 0}^t \qty[\vyopig_{k^\star}(t^\star + 1) \cdot g'\qty(\vzopblk{k^\star}_i(t^\star + 1)) \cdot \qty[\delta_{k^\star, k} \cdot \delta_{i, p} \cdot \vxt_q(t^\star) + \cancelto{\aptr 0}{\sum_{j = 1}^{\din + \dhid + \nblk \times (2 + \dblk)} \qty[\vWopblk{k^\star}_{i, j} \cdot \dv{\vxt_j(t^\star)}{\vWopblk{k}_{p, q}}]}]] \\
                                                            & \aptr \sum_{t^\star = 0}^t \qty[\vyopig_{k^\star}(t^\star + 1) \cdot g'\qty(\vzopblk{k^\star}_i(t^\star + 1)) \cdot \delta_{k^\star, k} \cdot \delta_{i, p} \cdot \vxt_q(t^\star)] \\
                                                            & \qqtext{where} \begin{dcases}
                                                                               i \in \Set{1, \dots, \dblk} \\
                                                                               k \in \Set{1, \dots, \nblk} \\
                                                                               k^\star \in \Set{1, \dots, \nblk} \\
                                                                               p \in \Set{1, \dots, \dblk} \\
                                                                               q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                               t \in \Set{0, \dots, \cT - 1}
                                                                             \end{dcases}.
      \end{align*}
    \]

  .. note::

    上式就是論文中的 A.12 式 :math:`\delta_{\opin_j l} = 0` 且 :math:`\delta_{c_j^v l} = 1` 的部份 。

  可以發現 :math:`\vWopblk{k}` 透過 memory cell internal states 得到的資訊其實都是來自於過去微分近似值的累加結果。
  實際上在執行參數更新演算法時只需要儲存過去累加而得的結果在加上當前計算結果，就可以得到最新的參數更新方向。
  使用前述推導結果我們可以得到 :math:`\vWopblk{k}` 相對於 memory cell activation blocks 的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \dv{\vyopblk{k^\star}_i(t + 1)}{\vWopblk{k}_{p, q}} & = \cancelto{\aptr 0}{\dv{\vyopog_{k^\star}(t + 1)}{\vWopblk{k}_{p, q}}} \cdot h\qty(\vsopblk{k^\star}_i(t + 1)) + \vyopog_{k^\star}(t + 1) \cdot \dv{h\qty(\vsopblk{k^\star}_i(t + 1))}{\vsopblk{k^\star}_i(t + 1)} \cdot \dv{\vsopblk{k^\star}_i(t + 1)}{\vWopblk{k}_{p, q}} \\
                                                            & \aptr \vyopog_{k^\star}(t + 1) \cdot h'\qty(\vsopblk{k^\star}_i(t + 1)) \cdot \sum_{t^\star = 0}^t \qty[\vyopig_{k^\star}(t^\star + 1) \cdot g'\qty(\vzopblk{k^\star}_i(t^\star + 1)) \cdot \delta_{k^\star, k} \cdot \delta_{i, p} \cdot \vxt_q(t^\star)] \\
                                                            & \qqtext{where} \begin{dcases}
                                                                               i \in \Set{1, \dots, \dblk} \\
                                                                               k \in \Set{1, \dots, \nblk} \\
                                                                               k^\star \in \Set{1, \dots, \nblk} \\
                                                                               p \in \Set{1, \dots, \dblk} \\
                                                                               q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                               t \in \Set{0, \dots, \cT - 1}
                                                                             \end{dcases}.
      \end{align*}
    \]

  .. note::

    上式就是論文中的 A.13 式 :math:`\delta_{\opout_j l} = \delta_{\opin_j l} = 0` 且 :math:`\delta_{c_j^v l} = 1` 的部份 。

  同前述結論，只需要儲存過去累加而得的結果再加上當前計算結果，最後乘上一些當前的計算狀態，就可以得到最新的參數更新方向。
  最後我們推得 :math:`\vWopblk{k}` 相對於誤差的微分近似值：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        & \dv{\cL\qty(\vy(t + 1) - \vyh(t + 1))}{\vWopblk{k}_{p, q}} \\
        & = \sum_{i = 1}^\dout \dv{\frac{1}{2} \qty(\vy_i(t + 1) - \vyh_i(t + 1))^2}{\vWopblk{k}_{p, q}} \\
        & = \sum_{i = 1}^\dout \qty[\dv{\frac{1}{2} \qty(\vy_i(t + 1) - \vyh_i(t + 1))^2}{\vy_i(t + 1)} \cdot \dv{\vy_i(t + 1)}{\vzopout_i(t + 1)} \cdot \sum_{j = 1}^{\din + \dhid + \nblk \times \dblk} \qty[\dv{\vzopout_i(t + 1)}{\vxopout_j(t + 1)} \cdot \cancelto{\aptr 0}{\dv{\vxopout_j(t + 1)}{\vWopblk{k}_{p, q}}}]] \\
        & \aptr \sum_{i = 1}^\dout \qty[\qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \sum_{k^\star = 1}^\nblk \sum_{j = 1}^\dblk \qty[\vWopout_{i, \din + \dhid + (k^\star - 1) \times \dblk + j} \cdot \dv{\vyopblk{k^\star}_j(t + 1)}{\vWopblk{k}_{p, q}}]] \\
        & \aptr \sum_{i = 1}^\dout \qty[\qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \sum_{k^\star = 1}^\nblk \sum_{j = 1}^\dblk \qty[\vWopout_{i, \din + \dhid + (k^\star - 1) \times \dblk + j} \cdot \vyopog_{k^\star}(t + 1) \cdot h'\qty(\vsopblk{k^\star}_j(t + 1)) \cdot \sum_{t^\star = 0}^t \qty[\vyopig_{k^\star}(t^\star + 1) \cdot g'\qty(\vzopblk{k^\star}_j(t^\star + 1)) \cdot \delta_{k^\star, k} \cdot \delta_{j, p} \cdot \vxt_q(t^\star)]]] \\
        & = \sum_{i = 1}^\dout \qty[\qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, \din + \dhid + (k - 1) \times \dblk + p} \cdot \vyopog_k(t + 1) \cdot h'\qty(\vsopblk{k}_p(t + 1)) \cdot \sum_{t^\star = 0}^t \qty[\vyopig_k(t^\star + 1) \cdot g'\qty(\vzopblk{k}_p(t^\star + 1)) \cdot \vxt_q(t^\star)]] \\
        & = \qty[\sum_{i = 1}^\dout \qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, \din + \dhid + (k - 1) \times \dblk + p}] \cdot \qty[\sum_{t^\star = 0}^t \vyopig_k(t^\star + 1) \cdot g'\qty(\vzopblk{k}_p(t^\star + 1)) \cdot \vxt_q(t^\star)] \cdot \vyopog_k(t + 1) \cdot h'\qty(\vsopblk{k}_p(t + 1)) \\
        & \qqtext{where} \begin{dcases}
                           k \in \Set{1, \dots, \nblk} \\
                           p \in \Set{1, \dots, \dblk} \\
                           q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                           t \in \Set{0, \dots, \cT - 1}
                         \end{dcases}.
      \end{align*}
    \]

.. note::

  :math:`\eqref{18}` 就是論文中 A.8 式 :math:`l = c_j^v` 的 case。

時間複雜度
----------

由於使用基於 RTRL 的最佳化演算法，計算完每個 :math:`t + 1` 時間點的誤差後就會馬上進行參數更新。
參數更新使用的演算法為 :term:`gradient descent`，:math:`\alpha` 為 learning rate：

.. math::
  :nowrap:

  \[
    \begin{align*}
      \vWopout_{p, q}    & \algoEq \vWopout_{p, q} - \alpha \cdot \dv{\cL\qty(\vy(t + 1), \vyh(t + 1))}{\vWopout_{p, q}} \qqtext{where} \begin{dcases}
                                                                                                                                          p \in \Set{1, \dots, \dout} \\
                                                                                                                                          q \in \Set{1, \dots, \din + \dhid + \nblk \times \dblk}
                                                                                                                                        \end{dcases}. \\
      \vWophid_{p, q}    & \algoEq \vWophid_{p, q} - \alpha \cdot \dv{\cL\qty(\vy(t + 1), \vyh(t + 1))}{\vWophid_{p, q}} \qqtext{where} \begin{dcases}
                                                                                                                                          p \in \Set{1, \dots, \dhid} \\
                                                                                                                                          q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)}
                                                                                                                                        \end{dcases}. \\
      \vWopog_{p, q}     & \algoEq \vWopog_{p, q} - \alpha \cdot \dv{\cL\qty(\vy(t + 1), \vyh(t + 1))}{\vWopog_{p, q}} \qqtext{where} \begin{dcases}
                                                                                                                                        p \in \Set{1, \dots, \nblk} \\
                                                                                                                                        q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)}
                                                                                                                                      \end{dcases}. \\
      \vWopig_{p, q}     & \algoEq \vWopig_{p, q} - \alpha \cdot \dv{\cL\qty(\vy(t + 1), \vyh(t + 1))}{\vWopig_{p, q}} \qqtext{where} \begin{dcases}
                                                                                                                                        p \in \Set{1, \dots, \nblk} \\
                                                                                                                                        q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)}
                                                                                                                                      \end{dcases}. \\
      \vWopblk{k}_{p, q} & \algoEq \vWopblk{k}_{p, q} - \alpha \cdot \dv{\cL\qty(\vy(t + 1), \vyh(t + 1))}{\vWopblk{k}_{p, q}} \qqtext{where} \begin{dcases}
                                                                                                                                                k \in \Set{1, \dots, \nblk} \\
                                                                                                                                                p \in \Set{1, \dots, \dblk} \\
                                                                                                                                                q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)}
                                                                                                                                              \end{dcases}.
    \end{align*}
    \tag{19}\label{19}
  \]

.. note::

  上式就是論文中的 A.15 式。

根據 :math:`\eqref{14} \eqref{15} \eqref{16} \eqref{17} \eqref{18}` 的微分近似結果，我們可以得出每個 :math:`t + 1` 時間點計算微分近似值的\ **時間複雜度**\為：

.. math::
  :nowrap:

  \[
    \order{\dim(\vWopout) + \dim(\vWophid) + \dim(\vWopog) + \dim(\vWopig) \times \dblk + \nblk \times \dim(\vWopblk{1})}
    \tag{20}\label{20}
  \]

.. dropdown:: 推導式子 :math:`\eqref{20}`

  觀察 :math:`\eqref{14} \eqref{15} \eqref{16} \eqref{17} \eqref{18}` 可以發現以下式子的計算結果可以共用：

  .. math::
    :nowrap:

    \[
      \qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \qqtext{where} \begin{dcases}
                                                                                                   i \in \Set{1, \dots, \dout} \\
                                                                                                   t \in \Set{0, \dots \cT - 1}
                                                                                                 \end{dcases}.
      \tag{20-1}\label{20-1}
    \]

  需要 :math:`\dout` 個減法與乘法可以得出 :math:`\dout` 個不同的 :math:`\eqref{20-1}`，因此所需時間複雜度為 :math:`\order{\dout}`。

  利用 :math:`\eqref{20-1}` 我們可以得出式子 :math:`\eqref{14}` 的時間複雜度為 :math:`\order{\dim(\vWopout)}`。

  .. dropdown:: 推導式子 :math:`\eqref{14}` 的時間複雜度

    有了 :math:`\eqref{20-1}` 後，共需執行 :math:`\dout \times (\din + \dhid + \nblk \times \dblk) = \dim(\vWopout)` 個乘法才能得到 :math:`\dim(\vWopout)` 個不同的

    .. math::
      :nowrap:

      \[
        \qty(\vy_p(t + 1) - \vyh_p(t + 1)) \cdot {f^\opout}'\qty(\vzopout_p(t + 1)) \cdot \vxopout_q(t + 1) \qqtext{where} \begin{dcases}
                                                                                                                             p \in \Set{1, \dots, \dout} \\
                                                                                                                             q \in \Set{1, \dots, \din + \dhid + \nblk \times \dblk} \\
                                                                                                                             t \in \Set{0, \dots \cT - 1}
                                                                                                                           \end{dcases}.
      \]

    因此計算 :math:`\eqref{14}` 所需時間複雜度為 :math:`\order{\dim(\vWopout)}`。

  接著觀察 :math:`\eqref{15} \eqref{16} \eqref{17} \eqref{18}` 可以發現以下式子的計算結果可以共用：

  .. math::
    :nowrap:

    \[
      \qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, j} \qqtext{where} \begin{dcases}
                                                                                                                         i \in \Set{1, \dots, \dout} \\
                                                                                                                         j \in \Set{\din + 1, \dots, \din + \dhid + \nblk \times \dblk} \\
                                                                                                                         t \in \Set{0, \dots \cT - 1}
                                                                                                                       \end{dcases}.
      \tag{20-2}\label{20-2}
    \]

  有了 :math:`\eqref{20-1}` 後，共需執行 :math:`\dout \times (\dhid + \nblk \times \dblk)` 個乘法才能得到 :math:`\eqref{20-2}`，因此計算 :math:`\eqref{20-2}` 所需時間複雜度為 :math:`\order{\dout \times (\dhid + \nblk \times \dblk)} = \order{\dim(\vWopout)}`。

  利用 :math:`\eqref{20-2}` 我們可以得出式子 :math:`\eqref{15}` 的時間複雜度為 :math:`\order{\dim(\vWopout) + \dim(\vWophid)}`。

  .. dropdown:: 推導式子 :math:`\eqref{15}` 的時間複雜度

    得到 :math:`\eqref{20-2}` 後，我們可以依照以下步驟計算得到 :math:`\eqref{15}`：

    1. 進行 :math:`(\dout - 1) \times \dhid` 次加法得到 :math:`\dhid` 個不同的

      .. math::
        :nowrap:

        \[
          \sum_{i = 1}^\dout \qty[\qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, \din + p}] \qqtext{where} \begin{dcases}
                                                                                                                                                             p \in \Set{1, \dots, \dhid} \\
                                                                                                                                                             t \in \Set{0, \dots \cT - 1}
                                                                                                                                                           \end{dcases}.
        \]

    2. 進行 :math:`\dhid` 次乘法得到 :math:`\dhid` 個不同的

      .. math::
        :nowrap:

        \[
          \qty(\sum_{i = 1}^\dout \qty[\qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, \din + p}]) \cdot {f^\ophid}'\qty(\vzophid_p(t + 1)) \qqtext{where} \begin{dcases}
                                                                                                                                                                                                            p \in \Set{1, \dots, \dhid} \\
                                                                                                                                                                                                            t \in \Set{0, \dots \cT - 1}
                                                                                                                                                                                                          \end{dcases}.
        \]

    3. 進行 :math:`\dhid \times (\din + \dhid + \nblk \times (2 + \dblk)) = \dim(\vWophid)` 次乘法得到 :math:`\dim(\vWophid)` 個不同的

      .. math::
        :nowrap:

        \[
          \qty(\sum_{i = 1}^\dout \qty[\qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, \din + p}]) \cdot {f^\ophid}'\qty(\vzophid_p(t + 1)) \cdot \vxt_q(t) \qqtext{where} \begin{dcases}
                                                                                                                                                                                                                            p \in \Set{1, \dots, \dhid} \\
                                                                                                                                                                                                                            q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                                                                                                                                                                            t \in \Set{0, \dots \cT - 1}
                                                                                                                                                                                                                          \end{dcases}.
        \]

    因此計算 :math:`\eqref{15}` 所需時間複雜度為

    .. math::
      :nowrap:

      \[
        \begin{align*}
          & \order{(\dout - 1) \times \dhid + \dhid + \dim(\vWophid)} \\
          & = \order{\dout \times \dhid + \dim(\vWophid)} \\
          & = \order{\dim(\vWopout) + \dim(\vWophid)}.
        \end{align*}
      \]

  利用 :math:`\eqref{20-2}` 我們可以得出式子 :math:`\eqref{16}` 的時間複雜度為 :math:`\order{\dim(\vWopout) + \dim(\vWopog)}`。

  .. dropdown:: 推導式子 :math:`\eqref{16}` 的時間複雜度

    得到 :math:`\eqref{20-2}` 後，我們可以依照以下步驟計算得到 :math:`\eqref{16}`：

    1. 進行 :math:`(\dout - 1) \times \nblk \times \dblk` 次加法得到 :math:`\nblk \times \dblk` 個不同的

      .. math::
        :nowrap:

        \[
          \sum_{i = 1}^\dout \qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, \din + \dhid + (p - 1) \times \dblk + j} \qqtext{where} \begin{dcases}
                                                                                                                                                                                      j \in \Set{1, \dots, \dblk} \\
                                                                                                                                                                                      p \in \Set{1, \dots, \nblk} \\
                                                                                                                                                                                      t \in \Set{0, \dots \cT - 1}
                                                                                                                                                                                    \end{dcases}.
        \]

    2. 進行 :math:`\nblk \times \dblk` 次乘法得到 :math:`\nblk \times \dblk` 個不同的

      .. math::
        :nowrap:

        \[
          \qty[\sum_{i = 1}^\dout \qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, \din + \dhid + (p - 1) \times \dblk + j}] \cdot h\qty(\vsopblk{p}_j(t + 1)) \qqtext{where} \begin{dcases}
                                                                                                                                                                                                                              j \in \Set{1, \dots, \dblk} \\
                                                                                                                                                                                                                              p \in \Set{1, \dots, \nblk} \\
                                                                                                                                                                                                                              t \in \Set{0, \dots \cT - 1}
                                                                                                                                                                                                                            \end{dcases}.
        \]

    3. 進行 :math:`\nblk \times (\dblk - 1)` 次加法得到 :math:`\nblk` 個不同的

      .. math::
        :nowrap:

        \[
          \sum_{j = 1}^\dblk \qty[\sum_{i = 1}^\dout \qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, \din + \dhid + (p - 1) \times \dblk + j}] \cdot h\qty(\vsopblk{p}_j(t + 1)) \qqtext{where} \begin{dcases}
                                                                                                                                                                                                                                                 p \in \Set{1, \dots, \nblk} \\
                                                                                                                                                                                                                                                 t \in \Set{0, \dots \cT - 1}
                                                                                                                                                                                                                                               \end{dcases}.
        \]

    4. 進行 :math:`\nblk` 次乘法得到 :math:`\nblk` 個不同的

      .. math::
        :nowrap:

        \[
          \qty(\sum_{j = 1}^\dblk \qty[\sum_{i = 1}^\dout \qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, \din + \dhid + (p - 1) \times \dblk + j}] \cdot h\qty(\vsopblk{p}_j(t + 1))) \cdot {f^\opog}'\qty(\vzopog_p(t + 1)) \qqtext{where} \begin{dcases}
                                                                                                                                                                                                                                                                                              p \in \Set{1, \dots, \nblk} \\
                                                                                                                                                                                                                                                                                              t \in \Set{0, \dots \cT - 1}
                                                                                                                                                                                                                                                                                            \end{dcases}.
        \]

    5. 進行 :math:`\nblk \times (\din + \dhid + \nblk \times (2 + \dblk)) = \dim(\vWopog)` 次乘法得到 :math:`\dim(\vWopog)` 個不同的

      .. math::
        :nowrap:

        \[
          \qty(\sum_{j = 1}^\dblk \qty[\sum_{i = 1}^\dout \qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, \din + \dhid + (p - 1) \times \dblk + j}] \cdot h\qty(\vsopblk{p}_j(t + 1))) \cdot {f^\opog}'\qty(\vzopog_p(t + 1)) \cdot \vxt_q(t) \qqtext{where} \begin{dcases}
                                                                                                                                                                                                                                                                                                              p \in \Set{1, \dots, \nblk} \\
                                                                                                                                                                                                                                                                                                              q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                                                                                                                                                                                                                                                              t \in \Set{0, \dots \cT - 1}
                                                                                                                                                                                                                                                                                                            \end{dcases}.
        \]

    因此計算 :math:`\eqref{16}` 所需時間複雜度為

    .. math::
      :nowrap:

      \[
        \begin{align*}
          & \order{(\dout - 1) \times \nblk \times \dblk + \nblk \times \dblk + \nblk \times (\dblk - 1) + \nblk + \dim(\vWopog)} \\
          & = \order{\dout \times \nblk \times \dblk + \nblk \times \dblk + \dim(\vWopog)} \\
          & = \order{\dim(\vWopout) + \dim(\vWopog)}.
        \end{align*}
      \]

  利用 :math:`\eqref{20-2}` 我們可以得出式子 :math:`\eqref{17}` 的時間複雜度為 :math:`\order{\dim(\vWopout) + \dim(\vWopig) \times \dblk}`。

  .. dropdown:: 推導式子 :math:`\eqref{17}` 的時間複雜度

    得到 :math:`\eqref{20-2}` 後，我們可以依照以下步驟計算得到 :math:`\eqref{17}`：

    1. 進行 :math:`(\dout - 1) \times \nblk \times \dblk` 次加法得到 :math:`\nblk \times \dblk` 個不同的

      .. math::
        :nowrap:

        \[
          \sum_{i = 1}^\dout \qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, \din + \dhid + (p - 1) \times \dblk + j} \qqtext{where} \begin{dcases}
                                                                                                                                                                                      j \in \Set{1, \dots, \dblk} \\
                                                                                                                                                                                      p \in \Set{1, \dots, \nblk} \\
                                                                                                                                                                                      t \in \Set{0, \dots \cT - 1}
                                                                                                                                                                                    \end{dcases}.
        \]

    2. 進行 :math:`\nblk \times \dblk` 次乘法得到 :math:`\nblk \times \dblk` 個不同的

      .. math::
        :nowrap:

        \[
          \qty[\sum_{i = 1}^\dout \qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, \din + \dhid + (p - 1) \times \dblk + j}] \cdot h'\qty(\vsopblk{p}_j(t + 1)) \qqtext{where} \begin{dcases}
                                                                                                                                                                                                                               j \in \Set{1, \dots, \dblk} \\
                                                                                                                                                                                                                               p \in \Set{1, \dots, \nblk} \\
                                                                                                                                                                                                                               t \in \Set{0, \dots \cT - 1}
                                                                                                                                                                                                                             \end{dcases}.
        \]

    3. 進行 :math:`\nblk \times (\din + \dhid + \nblk \times (2 + \dblk)) = \dim(\vWopig)` 次乘法得到 :math:`\dim(\vWopig)` 個不同的

      .. math::
        :nowrap:

        \[
          {f^\opig}'\qty(\vzopig_p(t + 1)) \cdot \vxt_q(t) \qqtext{where} \begin{dcases}
                                                                            p \in \Set{1, \dots, \nblk} \\
                                                                            q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                            t \in \Set{0, \dots \cT - 1}
                                                                          \end{dcases}.
        \]

    4. 進行 :math:`\dim(\vWopig) \times \dblk` 次乘法得到 :math:`\dim(\vWopig) \times \dblk` 個不同的

      .. math::
        :nowrap:

        \[
          {f^\opig}'\qty(\vzopig_p(t + 1)) \cdot \vxt_q(t) \cdot g\qty(\vzopblk{p}_j(t + 1)) \qqtext{where} \begin{dcases}
                                                                                                              j \in \Set{1, \dots, \dblk} \\
                                                                                                              p \in \Set{1, \dots, \nblk} \\
                                                                                                              q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                                                              t \in \Set{0, \dots \cT - 1}
                                                                                                            \end{dcases}.
        \]

    5. 進行 :math:`\dim(\vWopig) \times \dblk` 次加法得到 :math:`\dim(\vWopig) \times \dblk` 個不同的

      .. math::
        :nowrap:

        \[
          \dv{\vsopblk{p}_j(t + 1)}{\vWopig_{p, q}} = \dv{\vsopblk{p}_j(t)}{\vWopig_{p, q}} + {f^\opig}'\qty(\vzopig_p(t + 1)) \cdot \vxt_q(t) \cdot g\qty(\vzopblk{p}_j(t + 1)) \qqtext{where} \begin{dcases}
                                                                                                                                                                                                  j \in \Set{1, \dots, \dblk} \\
                                                                                                                                                                                                  p \in \Set{1, \dots, \nblk} \\
                                                                                                                                                                                                  q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                                                                                                                                                  t \in \Set{0, \dots \cT - 1}
                                                                                                                                                                                                \end{dcases}.
        \]

    6. 進行 :math:`\dim(\vWopig) \times \dblk` 次乘法得到 :math:`\dim(\vWopig) \times \dblk` 個不同的

      .. math::
        :nowrap:

        \[
          \qty[\sum_{i = 1}^\dout \qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, \din + \dhid + (p - 1) \times \dblk + j}] \cdot h'\qty(\vsopblk{p}_j(t + 1)) \cdot \dv{\vsopblk{p}_j(t + 1)}{\vWopig_{p, q}} \qqtext{where} \begin{dcases}
                                                                                                                                                                                                                                                                               j \in \Set{1, \dots, \dblk} \\
                                                                                                                                                                                                                                                                               p \in \Set{1, \dots, \nblk} \\
                                                                                                                                                                                                                                                                               q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                                                                                                                                                                                                                               t \in \Set{0, \dots \cT - 1}
                                                                                                                                                                                                                                                                             \end{dcases}.
        \]

    7. 進行 :math:`\dim(\vWopig) \times (\dblk - 1)` 次加法得到 :math:`\dim(\vWopig)` 個不同的

      .. math::
        :nowrap:

        \[
          \sum_{j = 1}^\dblk \qty[\sum_{i = 1}^\dout \qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, \din + \dhid + (p - 1) \times \dblk + j}] \cdot h'\qty(\vsopblk{p}_j(t + 1)) \cdot \dv{\vsopblk{p}_j(t + 1)}{\vWopig_{p, q}} \qqtext{where} \begin{dcases}
                                                                                                                                                                                                                                                                                                  p \in \Set{1, \dots, \nblk} \\
                                                                                                                                                                                                                                                                                                  q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                                                                                                                                                                                                                                                  t \in \Set{0, \dots \cT - 1}
                                                                                                                                                                                                                                                                                                \end{dcases}.
        \]

    8. 進行 :math:`\dim(\vWopig)` 次乘法得到 :math:`\dim(\vWopig)` 個不同的

      .. math::
        :nowrap:

        \[
          \qty(\sum_{j = 1}^\dblk \qty[\sum_{i = 1}^\dout \qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, \din + \dhid + (p - 1) \times \dblk + j}] \cdot h'\qty(\vsopblk{p}_j(t + 1)) \cdot \dv{\vsopblk{p}_j(t + 1)}{\vWopig_{p, q}}) \cdot \vyopog_p(t + 1) \qqtext{where} \begin{dcases}
                                                                                                                                                                                                                                                                                                                               p \in \Set{1, \dots, \nblk} \\
                                                                                                                                                                                                                                                                                                                               q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                                                                                                                                                                                                                                                                               t \in \Set{0, \dots \cT - 1}
                                                                                                                                                                                                                                                                                                                             \end{dcases}.
        \]

    因此計算 :math:`\eqref{17}` 所需時間複雜度為

    .. math::
      :nowrap:

      \[
        \begin{align*}
          & \order{(\dout - 1) \times \nblk \times \dblk + \nblk \times \dblk + \dim(\vWopig) + 3 \times \dim(\vWopig) \times \dblk + \dim(\vWopig) \times (\dblk - 1) + \dim(\vWopig)} \\
          & = \order{\dout \times \nblk \times \dblk + \dim(\vWopig) + 4 \times \dim(\vWopig) \times \dblk} \\
          & = \order{\dim(\vWopout) + \dim(\vWopig) \times \dblk}.
        \end{align*}
      \]

  利用 :math:`\eqref{20-2}` 我們可以得出式子 :math:`\eqref{18}` 的時間複雜度為 :math:`\order{\dim(\vWopout) + \nblk \times \dim(\vWopblk{1})}`。

  .. dropdown:: 推導式子 :math:`\eqref{18}` 的時間複雜度

    接下來我們推導式子 :math:`\eqref{18}` 的時間複雜度。
    得到 :math:`\eqref{20-2}` 後，我們可以依照以下步驟計算得到 :math:`\eqref{18}`：

    1. 進行 :math:`(\dout - 1) \times \nblk \times \dblk` 次加法得到 :math:`\nblk \times \dblk` 個不同的

      .. math::
        :nowrap:

        \[
          \sum_{i = 1}^\dout \qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, \din + \dhid + (k - 1) \times \dblk + p} \qqtext{where} \begin{dcases}
                                                                                                                                                                                      k \in \Set{1, \dots, \nblk} \\
                                                                                                                                                                                      p \in \Set{1, \dots, \dblk} \\
                                                                                                                                                                                      t \in \Set{0, \dots \cT - 1}
                                                                                                                                                                                    \end{dcases}.
        \]

    2. 進行 :math:`\nblk \times \dblk` 次乘法得到 :math:`\nblk \times \dblk` 個不同的

      .. math::
        :nowrap:

        \[
          \vyopig_k(t + 1) \cdot g'\qty(\vzopblk{k}_p(t + 1)) \qqtext{where} \begin{dcases}
                                                                              k \in \Set{1, \dots, \nblk} \\
                                                                              p \in \Set{1, \dots, \dblk} \\
                                                                              t \in \Set{0, \dots \cT - 1}
                                                                            \end{dcases}.
        \]

    3. 進行 :math:`\nblk \times \dblk \times (\din + \dhid + \nblk \times (2 + \dblk)) = \nblk \times \dim(\vWopblk{1})` 次乘法得到 :math:`\nblk \times \dim(\vWopblk{1})` 個不同的

      .. math::
        :nowrap:

        \[
          \vyopig_k(t + 1) \cdot g'\qty(\vzopblk{k}_p(t + 1)) \cdot \vxt_q(t) \qqtext{where} \begin{dcases}
                                                                                              k \in \Set{1, \dots, \nblk} \\
                                                                                              p \in \Set{1, \dots, \dblk} \\
                                                                                              q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                                              t \in \Set{0, \dots \cT - 1}
                                                                                            \end{dcases}.
        \]

    4. 進行 :math:`\nblk \times \dim(\vWopblk{1})` 次加法得到 :math:`\nblk \times \dim(\vWopblk{1})` 個不同的

      .. math::
        :nowrap:

        \[
          \dv{\vsopblk{k}_p(t + 1)}{\vWopblk{k}_{p, q}} = \dv{\vsopblk{k}_p(t)}{\vWopblk{k}_{p, q}} + \vyopig_k(t + 1) \cdot g'\qty(\vzopblk{k}_p(t + 1)) \cdot \vxt_q(t) \qqtext{where} \begin{dcases}
                                                                                                                                                                                          k \in \Set{1, \dots, \nblk} \\
                                                                                                                                                                                          p \in \Set{1, \dots, \dblk} \\
                                                                                                                                                                                          q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                                                                                                                                          t \in \Set{0, \dots \cT - 1}
                                                                                                                                                                                        \end{dcases}.
        \]

    5. 進行 :math:`\nblk \times \dim(\vWopblk{1})` 次乘法得到 :math:`\nblk \times \dim(\vWopblk{1})` 個不同的

      .. math::
        :nowrap:

        \[
          \qty[\sum_{i = 1}^\dout \qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, \din + \dhid + (k - 1) \times \dblk + p}] \cdot \dv{\vsopblk{k}_p(t + 1)}{\vWopblk{k}_{p, q}} \qqtext{where} \begin{dcases}
                                                                                                                                                                                                                                                k \in \Set{1, \dots, \nblk} \\
                                                                                                                                                                                                                                                p \in \Set{1, \dots, \dblk} \\
                                                                                                                                                                                                                                                q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                                                                                                                                                                                                t \in \Set{0, \dots \cT - 1}
                                                                                                                                                                                                                                              \end{dcases}.
        \]

    6. 進行 :math:`\nblk \times \dim(\vWopblk{1})` 次乘法得到 :math:`\nblk \times \dim(\vWopblk{1})` 個不同的

      .. math::
        :nowrap:

        \[
          \qty[\sum_{i = 1}^\dout \qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, \din + \dhid + (k - 1) \times \dblk + p}] \cdot \dv{\vsopblk{k}_p(t + 1)}{\vWopblk{k}_{p, q}} \cdot \vyopog_k(t + 1) \qqtext{where} \begin{dcases}
                                                                                                                                                                                                                                                                      k \in \Set{1, \dots, \nblk} \\
                                                                                                                                                                                                                                                                      p \in \Set{1, \dots, \dblk} \\
                                                                                                                                                                                                                                                                      q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                                                                                                                                                                                                                      t \in \Set{0, \dots \cT - 1}
                                                                                                                                                                                                                                                                    \end{dcases}.
        \]

    7. 進行 :math:`\nblk \times \dim(\vWopblk{1})` 次乘法得到 :math:`\nblk \times \dim(\vWopblk{1})` 個不同的

      .. math::
        :nowrap:

        \[
          \qty[\sum_{i = 1}^\dout \qty(\vy_i(t + 1) - \vyh_i(t + 1)) \cdot {f^\opout}'\qty(\vzopout_i(t + 1)) \cdot \vWopout_{i, \din + \dhid + (k - 1) \times \dblk + p}] \cdot \dv{\vsopblk{k}_p(t + 1)}{\vWopblk{k}_{p, q}} \cdot \vyopog_k(t + 1) \cdot h'\qty(\vsopblk{k}_p(t + 1)) \qqtext{where} \begin{dcases}
                                                                                                                                                                                                                                                                                                          k \in \Set{1, \dots, \nblk} \\
                                                                                                                                                                                                                                                                                                          p \in \Set{1, \dots, \dblk} \\
                                                                                                                                                                                                                                                                                                          q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                                                                                                                                                                                                                                                                                                          t \in \Set{0, \dots \cT - 1}
                                                                                                                                                                                                                                                                                                        \end{dcases}.
        \]

    因此計算 :math:`\eqref{18}` 所需時間複雜度為

    .. math::
      :nowrap:

      \[
        \begin{align*}
          & \order{(\dout - 1) \times \nblk \times \dblk + \nblk \times \dblk + 5 \times \nblk \times \dim(\vWopblk{1})} \\
          & = \order{\dout \times \nblk \times \dblk + 5 \times \nblk \times \dim(\vWopblk{1})} \\
          & = \order{\dim(\vWopout) + \nblk \times \dim(\vWopblk{1})}.
        \end{align*}
      \]

  透過前述結論我們可以得出 LSTM 最佳化演算法的時間複雜度為

  .. math::
    :nowrap:

    \[
      \begin{align*}
        & \order{\dout + \dim(\vWopout) + \dim(\vWopout) + \dim(\vWopout) + \dim(\vWophid) + \dim(\vWopout) + \dim(\vWopog) + \dim(\vWopout) + \dim(\vWopig) \times \dblk + \dim(\vWopout) + \nblk \times \dim(\vWopblk{1})} \\
        & \order{\dout + 6 \times \dim(\vWopout) + \dim(\vWophid) + \dim(\vWopog) + \dim(\vWopig) \times \dblk + \nblk \times \dim(\vWopblk{1})} \\
        & \order{\dim(\vWopout) + \dim(\vWophid) + \dim(\vWopog) + \dim(\vWopig) \times \dblk + \nblk \times \dim(\vWopblk{1})}.
      \end{align*}
    \]

.. note::

  式子 :math:`\eqref{20}` 與論文中的 A.27 式不同，但我覺得我的推論是正確的。

.. note::

  LSTM 最佳化演算法可以即時更新的特性作者稱其為 **local in time**.

空間複雜度
----------

觀察 :math:`\eqref{20}` 的推導過程，我們可以發現除了 :math:`\dv{\vsopblk{p}_j(t + 1)}{\vWopig_{p, q}}` 與 :math:`\dv{\vsopblk{p}_j(t + 1)}{\vWopblk{k}_{p, q}}` 之外，每個 :math:`t + 1` 時間點的資訊計算完畢後就可以丟棄。
因此論文得出 LSTM 最佳化演算法的\ **空間複雜度** 與時間複雜度相同：

.. math::
  :nowrap:

  \[
    \order{\dim(\vWopout) + \dim(\vWophid) + \dim(\vWopog) + \dim(\vWopig) \times \dblk + \nblk \times \dim(\vWopblk{1})}.
    \tag{21}\label{21}
  \]

.. note::

  LSTM 最佳化演算法即時更新且不須儲存過去所有資訊的特性，作者稱其為 **local in space**.

架構分析
========

Abuse Problem
-------------

在訓練 LSTM 的初期，參數的隨機初始化可能讓 memory cells 在 forward pass 中產生出無意義的值，而 LSTM 有可能濫用這些隨機值來迫使參數更新降低誤差，而不是透過學習輸入知識來降低誤差。
有可能需要透過長時間的訓練才能讓參數學會釋放 memory cells 中的隨機值並真正開始進行學習，因此學習效率會降低，甚至根本無法正常訓練 LSTM。

作者認為可行的解決方法是將 output gate units 的 **bias term** 初始化成\ **負數**，因此模型在\ **訓練初期**\就會\ **關閉 output gate units**，避免濫用 memory cells 的隨機值。

.. dropdown:: 推導初始化 output gate bias 為負數的邏輯

  .. math::
    :nowrap:

    \[
      \begin{align*}
                 & b_k^\opog \ll 0 \qqtext{where} k \in \Set{1, \dots, \nblk} \\
        \implies & \vzopog_k(t + 1) \ll 0 \qqtext{where} \begin{dcases}
                                                           k \in \Set{1, \dots, \nblk} \\
                                                           t \in \Set{1, \dots, \cT - 1}
                                                         \end{dcases} \\
        \implies & \vyopog_k(t + 1) \approx 0 \qqtext{where} \begin{dcases}
                                                               k \in \Set{1, \dots, \nblk} \\
                                                               t \in \Set{1, \dots, \cT - 1}
                                                             \end{dcases} \\
        \implies & \vyopog_k(t + 1) \cdot h\qty(\vsopblk{k}_i(t + 1)) \approx 0 \qqtext{where} \begin{dcases}
                                                                                                 i \in \Set{1, \dots, \dblk} \\
                                                                                                 k \in \Set{1, \dots, \nblk} \\
                                                                                                 t \in \Set{1, \dots, \cT - 1}
                                                                                               \end{dcases}.
      \end{align*}
    \]

作者也將「兩個不同的 memory cells 學到儲存完全相同的知識」視為類似的問題，解決方法是將 output gate units 的 bias term 初始化為\ **大小不同的負數**，迫使 memory cells 依照 bias term **由小到大開啟** output gate units（越接近 :math:`0` 越容易被開啟）。

Internal State Drift
--------------------

由於 memory cell internal states 是透過疊加的形式計算而得，作者認為 LSTM 在 forward pass 一段時間後容易讓 memory cell internal states 累加得出極正或極負的數值，作者稱此現象為 **internal state drift**。
當 :math:`h` 是 sigmoid 函數時，極正或極負的 memory cell internal states 只會讓 :math:`h'` 輸出靠近 :math:`0`\（見 :doc:`sigmoid 函數特性 </post/math/sigmoid>`）。
由於 :math:`\eqref{17} \eqref{18}` 的更新需要計算 :math:`h'`，因此 internal state drift 會造成參數 :math:`\vWopig` 與 :math:`\vWopblk{k}` 的梯度消失。

作者認為可行的解決方法是將 input gate units 的 **bias term** 初始化成\ **負數**，因此模型在\ **訓練初期**\就會\ **關閉 input gate units**，避免 memory cell internal states 在 forward pass 的前期就快速累加成極值。
後續實驗發現如果 :math:`h` 是 sigmoid 函數，則其實也不太需要將 input gate bias 設成負數。

.. dropdown:: 推導初始化 input gate bias 為負數的邏輯

  .. math::
    :nowrap:

    \[
      \begin{align*}
                 & b_k^\opig \ll 0 \qqtext{where} k \in \Set{1, \dots, \nblk} \\
        \implies & \vzopig_k(1) \ll 0 \qqtext{where} k \in \Set{1, \dots, \nblk} \\
        \implies & \vyopig_k(1) \approx 0 \qqtext{where} k \in \Set{1, \dots, \nblk} \\
        \implies & \vyopig_k(1) \cdot g\qty(\vzopblk{k}_i(1)) \approx 0 \qqtext{where} \begin{dcases}
                                                                                         i \in \Set{1, \dots, \dblk} \\
                                                                                         k \in \Set{1, \dots, \nblk}
                                                                                       \end{dcases} \\
        \implies & \vsopblk{k}_i(1) = \vsopblk{k}_i(0) + \vyopig_k(1) \cdot g\qty(\vzopblk{k}_i(1)) \approx 0 \qqtext{where} \begin{dcases}
                                                                                                                               i \in \Set{1, \dots, \dblk} \\
                                                                                                                               k \in \Set{1, \dots, \nblk}
                                                                                                                             \end{dcases} \\
        \implies & \begin{dcases}
                     \vsopblk{k}_i(t + 1) \not\ll 0 \\
                     \vsopblk{k}_i(t + 1) \not\gg 0
                   \end{dcases} \qqtext{where} \begin{dcases}
                                                 i \in \Set{1, \dots, \dblk} \\
                                                 k \in \Set{1, \dots, \nblk} \\
                                                 t \in \Set{0, \dots, \cT - 1}
                                               \end{dcases}.
      \end{align*}
    \]

雖然這種作法是種 **model bias** 而且會強迫 :math:`\vyopig` 與 :math:`{f^\opig}'` **趨近於** :math:`0`，但作者認為解決 internal state drift 比較重要。

Scaling Down Error
------------------

在訓練的初期\ **誤差**\通常比較\ **大**，導致\ **微分值**\跟著變\ **大**，容易使模型在訓練初期的參數劇烈振盪。
尤其 RNN 又會受到遞迴計算架構的限制導致梯度爆炸，因此 RNN 容易在訓練初期就訓練失敗。
然而 LSTM 唯一需要遞迴計算的微分項次只有 :math:`\eqref{17} \eqref{18}`，觀察可以發現這些項次都會乘上 :math:`\vyopog`。
由於 **output gate units** 所使用的 activation function :math:`f^\opog` 是 sigmoid，乘上 :math:`\vyopog` 可以避免\ **過大誤差**\造成的整體微分值太大。
但這些說法並沒有辦法真的保證一定會實現，算是這篇論文說服力比較薄弱的點。

.. dropdown:: 推導 output gate units 縮放 error 的邏輯

  對 :math:`\vWopig` 來說，透過遞迴造成微分值變成極值的來源為（見 :math:`\eqref{17}`）：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        & \sum_{t^\star = 0}^t {f^\opig}'\qty(\vzopig_p(t^\star + 1)) \cdot \vxt_q(t^\star) \cdot g\qty(\vzopblk{p}_j(t^\star + 1)) \\
        & \qqtext{where} \begin{dcases}
                           p \in \Set{1, \dots, \nblk} \\
                           q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                           t \in \Set{0, \dots, \cT - 1}
                         \end{dcases}
      \end{align*}
    \]

  比起 RNN 的指數增加微分（累加連乘積項次得出微分值，見 :math:`\eqref{4}`），上式只有單純的累加每個時間的部份計算狀態，因此 LSTM 的微分是成線性增長，不容易達成梯度爆炸。
  另外觀察 :math:`\eqref{17}` 可以發現上式會乘上 output gate units 進行縮減，因此 LSTM 的設計有助於穩定更新 :math:`\vWopig`。

  我們使用相同邏輯對 :math:`\vWopblk{k}` 進行分析，觀察 :math:`\eqref{18}` 可以發現透過遞迴造成微分值變成極值的來源為：

  .. math::
    :nowrap:

    \[
      \begin{align*}
        & \sum_{t^\star = 0}^t \vyopig_k(t^\star + 1) \cdot g'\qty(\vzopblk{k}_p(t^\star + 1)) \cdot \vxt_q(t^\star) \\
        & \qqtext{where} \begin{dcases}
                           k \in \Set{1, \dots, \nblk} \\
                           p \in \Set{1, \dots, \dblk} \\
                           q \in \Set{1, \dots, \din + \dhid + \nblk \times (2 + \dblk)} \\
                           t \in \Set{0, \dots, \cT - 1}
                         \end{dcases}
      \end{align*}
    \]

  同理，比起 RNN 的指數增加微分，上式只有單純的累加每個時間的部份計算狀態，因此 LSTM 的微分是成線性增長，不容易達成梯度爆炸。
  另外觀察 :math:`\eqref{18}` 可以發現上式會乘上 output gate units 進行縮減，因此 LSTM 的設計有助於穩定更新 :math:`\vWopblk{k}`。

.. note::

  上述推導就是論文中的 A.28-A.39 式。

實驗
====

實驗設計
--------

- 要測試在 long time lag 的任務上
- 資料集不可以出現 short time lag 的範例，強迫模型只能學 long time lag
- 測試任務要夠難，以至於無法靠 random weight guessing 解決
- 任務夠困難以至於模型需要比較多的參數或是高計算精度（sparse in weight space）

控制變因
--------

- 使用 online learning 進行最佳化，等同於每次只對一筆資料（一個輸入序列）進行最佳化（mini-batch size 為 1）
- 使用 sigmoid 作為 activation function，包含 :math:`f^\opout, f^\ophid, f^\opig, f^\opog`
- 所有實驗資料集都是隨機生成
- Online learning 訓練的順序是完全隨機
- 在每個時間點 :math:`t` 的計算順序為

  0. 初始化模型計算狀態
  1. 將輸入 :math:`\vx(t)` 餵給模型
  2. 計算 input gate units、output gate units、memory cells、conventional hidden units
  3. 計算輸出

- 選擇不同任務中 LSTM 所使用的 hyperparameters :math:`\nblk` 與 :math:`\dblk` 的方法如下

  - 訓練初期只使用一個 memory cell，即 :math:`\nblk = \dblk = 1`
  - 如果訓練中發現最佳化做的不好，開始增加 memory cells ，即 :math:`\dblk \algoEq \dblk + 1`
  - 部份任務會嘗試增加 memory cell blocks，一旦 memory cell blocks 增加，input/output gate units 也需要跟著增加
  - Sequential network construction 是隨著誤差停止下降後才增加 hidden units，與上述演算法不同

- :math:`h` 與 :math:`g` 函數如果沒有特別提及，就是使用以下定義：

  :math:`h: \R \to [-1, 1]` 與 :math:`g: \R \to [-2, 2]` 函數的定義為

  .. math::
    :nowrap:

    \[
      \begin{align*}
        h(x) & = \frac{2}{1 + \exp(-x)} - 1 = 2 \sigma(x) - 1. \\
        g(x) & = \frac{4}{1 + \exp(-x)} - 2 = 4 \sigma(x) - 2.
      \end{align*}
    \]

實驗 1：Embedded Reber Grammar
------------------------------

.. figure:: https://i.imgur.com/frOl0Tf.png
  :alt: Reber Grammar
  :name: paper-fig-3

  圖 3：Reber Grammar。

  一個簡單的有限狀態機，能夠生成的字母包含 BEPSTVX。
  圖片來源：:footcite:`hochreiter-etal-1997-long`。

.. figure:: https://i.imgur.com/SVfVbJN.png
  :alt: Embedded Reber Grammar。
  :name: paper-fig-4

  圖 4：Embedded Reber Grammar。

  一個簡單的有限狀態機，包含兩個完全相同的 Reber Grammar，(開頭, 結尾) 只能是 (BT, TE) 與 (BP, PE)。
  圖片來源：:footcite:`hochreiter-etal-1997-long`。

任務定義
~~~~~~~~

- Embedded Reber Grammar 是實驗 short time lag 的基準測試資料集

  - 透過 Embedded Reber Grammar 產生的字串最短為 :math:`9` 個字元（例如 BTBTXSETE）
  - 訓練與測試資料集都是透過 Embedded Reber Grammar 隨機生成，任何一個分支都有 :math:`0.5` 的機率被生成
  - 每個字串前兩個字母生成 BT 或 BP 的機率各為 :math:`0.5`
  - 如果一個字串前兩個字母生成 BT，則該字串結尾一定會生成 TE
  - 如果一個字串前兩個字母生成 BP，則該字串結尾一定會生成 PE
  - 字串中間由 Reber Grammar 生成，能夠生成的字母包含 BEPSTVX
  - 由於 Reber Grammar 有限狀態機中有 loop（見 :ref:`paper-fig-3`），因此 Reber Grammar 有可能產生\ **任意長度**\的文字

- 訓練任務設定為\ **每輸入一個字母就預測下一個字母**

  - 就是 language model 的概念
  - 除了開頭與結尾，每個字母生成機率皆為 :math:`0.5`，因此模型\ **無法**\順利預測一個字串開頭與中間的字母
  - 唯一能夠正常預測的只有結尾，因此模型必須學會記住\ **開頭**\的 T/P 才能成功預測\ **結尾** T/P

- 評估模型表現的方法為完全預測每個字串接續字母的\ **可能性**

  - 只要模型能夠成功預測下一個可能出現的字母就算正確
  - 字串中間每個位置可能出現的字母有\ **兩個**，字串結尾只有\ **一種**\可能
  - 作者設定必須要同時在訓練與測試資料集上完全答對才算一次 successful trial

- 此任務為常見的 RNN benchmark，傳統 RNN 在此資料集上仍然表現不錯
- 資料數

  - 訓練資料集：:math:`256` 筆
  - 測試資料集：:math:`256` 筆，與訓練資料集的交集為空
  - 總共產生 :math:`3` 組不同的訓練測試集
  - 每組資料集都跑 :math:`10` 次實驗，每次實驗模型都隨機初始化
  - 總共執行 :math:`30` 次實驗取平均

LSTM 架構
~~~~~~~~~

- 輸入與輸出都是 one-hot vector，維度為 :math:`7`，每個 coordinate 各自代表 BEPSTVX 中的一個字元
- 輸出取數值最大的 coordinate 的 index（即 argmax）作為預測結果

+---------------------------------------+------------------------------------------------------------+-------------------------------------------------------------+
| Hyperparameters                       | Value or Range                                             | Notes                                                       |
+=======================================+============================================================+=============================================================+
| :math:`\din`                          | :math:`7`                                                  |                                                             |
+---------------------------------------+------------------------------------------------------------+-------------------------------------------------------------+
| :math:`\dhid`                         | :math:`0`                                                  | No conventional hidden units are used.                      |
+---------------------------------------+------------------------------------------------------------+-------------------------------------------------------------+
| :math:`(\nblk, \dblk)`                | :math:`\Set{(3, 2), (4, 1)}`                               | At least :math:`3` memory cells are used.                   |
+---------------------------------------+------------------------------------------------------------+-------------------------------------------------------------+
| :math:`\dout`                         | :math:`7`                                                  |                                                             |
+---------------------------------------+------------------------------------------------------------+-------------------------------------------------------------+
| :math:`\dim(\vWophid)`                | :math:`0`                                                  | No conventional hidden units are used.                      |
+---------------------------------------+------------------------------------------------------------+-------------------------------------------------------------+
| :math:`\dim(\vWopblk{k})`             | :math:`\dblk \times (\din + \nblk \times (2 + \dblk))`     | Fully-connected layer.                                      |
+---------------------------------------+------------------------------------------------------------+-------------------------------------------------------------+
| :math:`\dim(\vWopig)`                 | :math:`\nblk \times (\din + \nblk \times (2 + \dblk) + 1)` | Fully-connected layer, used bias term on input gate units.  |
+---------------------------------------+------------------------------------------------------------+-------------------------------------------------------------+
| :math:`\dim(\vWopog)`                 | :math:`\nblk \times (\din + \nblk \times (2 + \dblk) + 1)` | Fully-connected layer, used bias term on output gate units. |
+---------------------------------------+------------------------------------------------------------+-------------------------------------------------------------+
| :math:`\dim(\vWopout)`                | :math:`\dout \times (\nblk \times \dblk)`                  | Input units are not directly connected to output units.     |
+---------------------------------------+------------------------------------------------------------+-------------------------------------------------------------+
| Weight initalization range            | :math:`[-0.2, 0.2]`                                        |                                                             |
+---------------------------------------+------------------------------------------------------------+-------------------------------------------------------------+
| Output gate bias initialization range | :math:`\Set{-1, -2, -3, -4}`                               |                                                             |
+---------------------------------------+------------------------------------------------------------+-------------------------------------------------------------+
| Learning rate                         | :math:`\Set{0.1, 0.2, 0.5}`                                |                                                             |
+---------------------------------------+------------------------------------------------------------+-------------------------------------------------------------+
| Number of paramters                   | :math:`\Set{264, 276}`                                     |                                                             |
+---------------------------------------+------------------------------------------------------------+-------------------------------------------------------------+

實驗結果
~~~~~~~~

.. figure:: https://i.imgur.com/51yPwmH.png
  :alt: Embedded Reber Grammar 實驗結果
  :name: paper-table-1

  表格 1：Embedded Reber Grammar 實驗結果。

  表格來源：:footcite:`hochreiter-etal-1997-long`。

- LSTM + 丟棄梯度 + RTRL 在不同的實驗架構中都能解決任務

  - RNN + RTRL 無法完成
  - Elman Net + ELM 無法完成

- LSTM 收斂速度比其他模型都還要快
- LSTM 使用的參數數量並沒有比其他的模型多太多
- 驗證 **output gate units** 的有效性

  - 當 LSTM 的某個 memory cell 記住第二個輸入是 T/P 之後，就會關閉對應的 output gate units，不讓該 memory cells 影響模型學習簡單的 Reber Grammar
  - 如果沒有輸出閘門，則\ **收斂速度會變慢**

實驗 2a：Noise-Free Sequences with Long Time Lags
--------------------------------------------------

任務定義
~~~~~~~~

令 :math:`p \in \Z^+`，定義字元集 :math:`V = \Set{x, y, a_1, a_2, \dots, a_{p - 1}}`。
定義 :math:`2` 種長度為 :math:`p + 1` 不同的序列 :math:`\opseq_1, \opseq_2`，分別為

.. math::
  :nowrap:

  \[
    \begin{align*}
      \opseq_1 & = x, a_1, a_2, \dots, a_{p - 1}, x \\
      \opseq_2 & = y, a_1, a_2, \dots, a_{p - 1}, y
    \end{align*}
  \]

令 :math:`\opseq_\star \in \Set{\opseq_1, \opseq_2}`，令 :math:`\opseq_\star` 第 :math:`t` 個時間點的字元為 :math:`\opseq_\star(t) \in V`。
當給予模型 :math:`\opseq_\star(t)` 時，模型要能夠根據 :math:`\opseq_\star(0), \opseq_\star(1), \dots \opseq_\star(t)` 預測 :math:`\opseq_\star(t + 1)`。

- 模型需要記住 :math:`a_1, \dots, a_{p - 1}` 的順序
- 模型也需要記住開頭的 :math:`\opseq_\star(0)` 是 :math:`x` 還是 :math:`y`，並利用 :math:`\opseq_\star(0)` 的資訊預測 :math:`\opseq_\star(p + 1)`
- 根據 :math:`p` 的大小這個任務可以是 short time lag 或 long time lag
- 訓練資料

  - 每次以各 :math:`0.5` 的機率抽出 :math:`\opseq_1, \opseq_2` 作為輸入
  - 總共訓練與更新 :math:`5000000` 次

- 測試資料

  - 每次以各 :math:`0.5` 的機率抽出 :math:`\opseq_1, \opseq_2` 作為輸入
  - 每次輸入序列後得到的 maximal absolute error 在 :math:`0.25` 以下就是成功，反之失敗
  - 連續取得 :math:`10000` 次成功稱為 successful run（這裡不知道為什麼需要多次驗證，如果訓練後參數固定，則輸入相同序列應該會得到完全相同的輸出結果，不過後續實驗有隨機性時就必須納入此考量）
  - 總共進行 :math:`18` 次不同的訓練與測試，最後將達成 successful run 的次數除 :math:`18` 得到 % successful trials

LSTM 架構
~~~~~~~~~

- 輸入與輸出都是 one-hot vector，維度為 :math:`p + 1`，每個 coordinate 各自代表 :math:`V` 中的一個字元
- 輸出取數值最大的 coordinate 的 index（即 argmax）作為預測結果

+---------------------------------------+--------------------------------------------------+----------------------------------------------------+
| Hyperparameters                       | Value or Range                                   | Notes                                              |
+=======================================+==================================================+====================================================+
| :math:`\din`                          | :math:`p + 1`                                    |                                                    |
+---------------------------------------+--------------------------------------------------+----------------------------------------------------+
| :math:`\dhid`                         | :math:`0`                                        | No conventional hidden units are used.             |
+---------------------------------------+--------------------------------------------------+----------------------------------------------------+
| :math:`\dblk`                         | :math:`1`                                        |                                                    |
+---------------------------------------+--------------------------------------------------+----------------------------------------------------+
| :math:`\nblk`                         | :math:`1`                                        | Increase :math:`\nblk` when error stop decreasing. |
+---------------------------------------+--------------------------------------------------+----------------------------------------------------+
| :math:`\dout`                         | :math:`p + 1`                                    |                                                    |
+---------------------------------------+--------------------------------------------------+----------------------------------------------------+
| :math:`g`                             | :math:`g(x) = \sigma(x)`                         | Use sigmoid function.                              |
+---------------------------------------+--------------------------------------------------+----------------------------------------------------+
| :math:`h`                             | :math:`h(x) = x`                                 | Use identity mapping.                              |
+---------------------------------------+--------------------------------------------------+----------------------------------------------------+
| :math:`\dim(\vWophid)`                | :math:`0`                                        | No conventional hidden units are used.             |
+---------------------------------------+--------------------------------------------------+----------------------------------------------------+
| :math:`\dim(\vWopblk{k})`             | :math:`\dblk \times (\din + \nblk \times \dblk)` | Fully-connected layer.                             |
+---------------------------------------+--------------------------------------------------+----------------------------------------------------+
| :math:`\dim(\vWopig)`                 | :math:`\nblk \times (\din + \nblk \times \dblk)` | Fully-connected layer.                             |
+---------------------------------------+--------------------------------------------------+----------------------------------------------------+
| :math:`\dim(\vWopog)`                 | :math:`0`                                        | No output gate units are used.                     |
+---------------------------------------+--------------------------------------------------+----------------------------------------------------+
| :math:`\dim(\vWopout)`                | :math:`\dout \times (\nblk \times \dblk)`        |                                                    |
+---------------------------------------+--------------------------------------------------+----------------------------------------------------+
| Weight initalization range            | :math:`[-0.2, 0.2]`                              |                                                    |
+---------------------------------------+--------------------------------------------------+----------------------------------------------------+
| Learning rate                         | :math:`1`                                        |                                                    |
+---------------------------------------+--------------------------------------------------+----------------------------------------------------+

實驗結果
~~~~~~~~

.. figure:: https://i.imgur.com/638FPkg.png
  :alt: 無雜訊長時間差任務實驗結果
  :name: paper-table-2

  表格 2：無雜訊長時間差任務實驗結果。
  表格來源：:footcite:`hochreiter-etal-1997-long`。


- 在 :math:`p = 4` 時使用 RNN + RTRL 仍然能夠預測部份序列，證實序列很短時 RNN 還是有能力完成任務
- 在 :math:`p \geq 10` 時使用 RNN + RTRL 時直接失敗
- 在 :math:`p = 100` 時只剩 LSTM 能夠完全完成任務
- LSTM 收斂速度最快

實驗 2b：No Local Regularities
-------------------------------

實驗設計和 LSTM 的架構與實驗 2a 完全相同，只是序列 :math:`\opseq_1, \opseq_2` 中除了頭尾之外的字元可以替換成 :math:`\Set{a_1, \dots, a_{p - 1}}` 中任意的文字，總長度維持 :math:`p + 1`。

- 此設計目的是為了驗證在實驗 2a 中的資訊無法被順利壓縮（預測機率從 :math:`1` 變成 :math:`\frac{1}{p - 1}`）的前提下 LSTM 仍然能夠順利解決 long time lag 任務
- 先創造訓練資料，測試使用與訓練完全相同的資料
- 仍然只有 LSTM 能夠完全完成任務，且誤差仍然很快就收斂

  - 當 :math:`p = 100` 時只需要 :math:`5680` 次更新就能完成任務
  - 代表 LSTM 能夠在有雜訊的情況下正常運作

實驗 2c：Very Long Time Lags - No Local Regularities
-----------------------------------------------------

任務定義
~~~~~~~~

實驗設計和 LSTM 的架構與實驗 2a 概念相同，只是 :math:`V` 增加了兩個字元 :math:`b, e`，而序列長度可以不同。

生成一個序列的概念如下：

1. 固定一個 :math:`q \in \Z^+`，代表序列基本長度，會選擇較大的數字確保所有序列都是 long time lag
2. 從 :math:`a_1, \dots, a_{p - 1}` 中隨機抽樣生成長度為 :math:`q` 的序列 :math:`\opseq`
3. 在序列的開頭補上 :math:`bx` 或 :math:`by`\（機率各為 :math:`0.5`），讓序列長度變成 :math:`q + 2`
4. 接著以 :math:`0.9` 的機率從 :math:`a_1, \dots, a_{p - 1}` 中挑一個字補在序列 :math:`\opseq` 的尾巴，或是以 :math:`0.1` 的機率補上 :math:`e`
5. 如果生成 :math:`e` 就再補上 :math:`x` 或 :math:`y`\（與開頭第二個字元相同）並結束
6. 如果不是生成 :math:`e` 則重複步驟 4

假設步驟 :math:`4` 執行了 :math:`k + 1` 次，則序列長度為 :math:`q + 2 + (k + 1) + 1 = q + k + 4`。
序列的最短長度為 :math:`q + 4`，長度的期望值為 :math:`q + 14`，每個 :math:`a_i \in V` 出現次數的期望值約為 :math:`\frac{q}{p}`

.. dropdown:: 推導期望值

  長度的期望值為

  .. math::
    :nowrap:

    \[
      \begin{align*}
        & 4 + \sum_{k = 0}^\infty \frac{1}{10} \qty(\frac{9}{10})^k (q + k) \\
        & = 4 + \frac{q}{10} \qty[\sum_{k = 0}^\infty \qty(\frac{9}{10})^k] + \frac{1}{10} \qty[\sum_{k = 0}^\infty \qty(\frac{9}{10})^k \cdot k] \\
        & = 4 + \frac{q}{10} \cdot 10 + \frac{1}{10} \cdot 100 \\
        & = q + 14.
      \end{align*}
    \]

  其中使用的公式來自於

  .. math::
    :nowrap:

    \[
      \begin{align*}
        & \qty[\sum_{k = 0}^n k x^k] - x \qty[\sum_{k = 0}^n k x^k] \\
        & = (0x^0 + 1x^1 + 2x^2 + 3x^3 + \dots + nx^n) - (0x^1 + 1x^2 + 2x^3 + 3x^4 + \dots + nx^{n + 1}) \\
        & = 0x^0 + 1x^1 + 1x^2 + 1x^3 + \dots + 1x^n - nx^{n + 1} \\
        & = \qty[\sum_{k = 0}^n x^k] - nx^{n + 1} \\
        & = \frac{1 - x^{n + 1}}{1 - x} - nx^{n + 1} \\
        & = \frac{1 - x^{n + 1} - nx^{n + 1} + nx^{n + 2}}{1 - x}.
      \end{align*}
    \]

  因此

  .. math::
    :nowrap:

    \[
      \begin{align*}
                 & \qty[\sum_{k = 0}^n k x^k] - x \qty[\sum_{k = 0}^n k x^k] = \frac{1 - x^{n + 1} - nx^{n + 1} + nx^{n + 2}}{1 - x} \\
        \implies & \sum_{k = 0}^n k x^k = \frac{1 - x^{n + 1} - nx^{n + 1} + nx^{n + 2}}{(1 - x)^2} \\
        \implies & \sum_{k = 0}^\infty k x^k = \frac{1}{(1 - x)^2} \qqtext{when} 0 \leq x \lt 1.
      \end{align*}
    \]

  利用二項式分佈的期望值公式我們可以推得 :math:`a_i \in V` 出現次數的期望值

  .. math::
    :nowrap:

    \[
      \begin{align*}
        & \sum_{k = 0}^\infty \frac{1}{10} \cdot \qty(\frac{9}{10})^k \cdot \qty[\sum_{i = 0}^{q + k} \binom{q + k}{i} \cdot \qty(\frac{1}{p - 1})^i \cdot \qty(1 - \frac{1}{p - 1})^{q + k - i}] \\
        & = \sum_{k = 0}^\infty \frac{1}{10} \cdot \qty(\frac{9}{10})^k \cdot \frac{q + k}{p - 1} \\
        & = \frac{q}{10(p - 1)} \qty[\sum_{k = 0}^\infty \qty(\frac{9}{10})^k] + \frac{1}{10(p - 1)} \qty[\sum_{k = 0}^\infty \qty(\frac{9}{10})^k \cdot k] \\
        & = \frac{q}{p - 1} + \frac{10}{p - 1} \\
        & \approx \frac{q}{p - 1} \qqtext{when} q \gg 0 \\
        & \approx \frac{q}{p} \qqtext{when} p \gg 0.
      \end{align*}
    \]

訓練誤差只考慮最後一個時間點 :math:`\opseq(2 + q + k + 2)` 的預測結果，必須要跟第 :math:`\opseq(1)` 個時間點的輸入相同（概念同實驗 2a）。
測試時會連續執行 :math:`10000` 次的實驗，預測誤差必須要永遠小於 :math:`0.2`。
每個 :math:`(p, q)` 的選擇都會進行 :math:`20` 次的測試結果取平均。

LSTM 架構
~~~~~~~~~

- 輸入是 one-hot vector，維度為 :math:`p + 4`，每個 coordinate 各自代表 :math:`V` 中的一個字元
- 輸出是 one-hot vector，維度為 :math:`2`，每個 coordinate 各自代表 :math:`x` 或 :math:`y`
- 輸出取數值最大的 coordinate 的 index（即 argmax）作為預測結果

+----------------------------+-----------------------------------------------------------+---------------------------------------------------------------+
| Hyperparameters            | Value or Range                                            | Notes                                                         |
+============================+===========================================================+===============================================================+
| :math:`\din`               | :math:`p + 4`                                             |                                                               |
+----------------------------+-----------------------------------------------------------+---------------------------------------------------------------+
| :math:`\dhid`              | :math:`0`                                                 | No conventional hidden units are used.                        |
+----------------------------+-----------------------------------------------------------+---------------------------------------------------------------+
| :math:`\dblk`              | :math:`1`                                                 |                                                               |
+----------------------------+-----------------------------------------------------------+---------------------------------------------------------------+
| :math:`\nblk`              | :math:`2`                                                 | Author believes that we actually only need :math:`\nblk = 1`. |
+----------------------------+-----------------------------------------------------------+---------------------------------------------------------------+
| :math:`\dout`              | :math:`2`                                                 | Output can only be :math:`x` or :math:`y`.                    |
+----------------------------+-----------------------------------------------------------+---------------------------------------------------------------+
| :math:`g`                  | :math:`g(x) = 4 \sigma(x) - 2`                            |                                                               |
+----------------------------+-----------------------------------------------------------+---------------------------------------------------------------+
| :math:`h`                  | :math:`h(x) = 2 \sigma(x) - 1`                            |                                                               |
+----------------------------+-----------------------------------------------------------+---------------------------------------------------------------+
| :math:`\dim(\vWophid)`     | :math:`0`                                                 | No conventional hidden units are used.                        |
+----------------------------+-----------------------------------------------------------+---------------------------------------------------------------+
| :math:`\dim(\vWopblk{k})`  | :math:`\dblk \times (\din + \nblk \times (2 + \dblk))`    | Fully-connected layer.                                        |
+----------------------------+-----------------------------------------------------------+---------------------------------------------------------------+
| :math:`\dim(\vWopig)`      | :math:`\nblk \times (\din + \nblk \times (2 + \dblk))`    | Fully-connected layer.                                        |
+----------------------------+-----------------------------------------------------------+---------------------------------------------------------------+
| :math:`\dim(\vWopog)`      | :math:`\nblk \times (\din + \nblk \times (2 + \dblk))`    | Fully-connected layer.                                        |
+----------------------------+-----------------------------------------------------------+---------------------------------------------------------------+
| :math:`\dim(\vWopout)`     | :math:`\dout \times (\nblk \times \dblk)`                 | Input units are not directly connected to output units.       |
+----------------------------+-----------------------------------------------------------+---------------------------------------------------------------+
| Weight initalization range | :math:`[-0.2, 0.2]`                                       |                                                               |
+----------------------------+-----------------------------------------------------------+---------------------------------------------------------------+
| Learning rate              | :math:`0.01`                                              |                                                               |
+----------------------------+-----------------------------------------------------------+---------------------------------------------------------------+

實驗結果
~~~~~~~~

.. figure:: https://i.imgur.com/j8e0W2U.png
  :alt: 有雜訊超長時間差任務實驗結果。
  :name: paper-table-3

  表格 3：有雜訊超長時間差任務實驗結果。
  表格來源：:footcite:`hochreiter-etal-1997-long`。

- 其他方法沒有辦法完成任務，因此不列入表格比較
- 輸入序列長度可到達 :math:`1000`
- 當 :math:`(p, q)` 成正比一起增加時，LSTM 訓練時間只會緩慢增加，作者說這是其他 RNN 模型做不到的效果
- 當單一字元的\ **出現次數期望值增加**\時（固定 :math:`q` 增加 :math:`p` 導致 :math:`\frac{q}{p}` 變大），\ **學習速度會下降**

  - 作者認為是常見字詞的出現導致參數開始振盪

實驗 3a：Two-Sequence Problem
-----------------------------

任務定義
~~~~~~~~

給予一個\ **實數**\序列 :math:`\opseq`，該序列可能隸屬於兩種類別 :math:`C_1, C_2`，隸屬機率分別是 :math:`0.5`。

- 如果 :math:`\opseq \in C_1`，則該序列的前 :math:`N` 個數字都是 :math:`1.0`，序列的最後一個數字為 :math:`1.0`。
- 如果 :math:`\opseq \in C_2`，則該序列的前 :math:`N` 個數字都是 :math:`-1.0`，序列的最後一個數字為 :math:`0.0`。

給定一個常數 :math:`T`，並從 :math:`[T, T + \frac{\cT}{10}]` 的區間中隨機挑選一個整數作為序列 :math:`\opseq` 的長度 :math:`L`。
當 :math:`L \geq N` 時，任何在 :math:`\opseq(N + 1), \dots \opseq(L - 1)` 中的數字都是由常態分佈隨機產生，常態分佈的參數為 :math:`(\mu, \sigma^2) = (0, 0.2)`。

- 此任務由 Bengio 提出
- 作者發現只要用隨機權重猜測（Random Weight Guessing）就能解決此任務，因此在實驗 3c 提出任務的改進版本
- 訓練分成兩個階段

  - ST1：事先隨機抽取的 :math:`256` 筆測試資料完全分類正確
  - ST2：達成 ST1 後再額外使用事先隨機抽取的 :math:`2560` 筆測試資料上計算平均錯誤，必須低於 :math:`0.01`

- 實驗結果是執行 :math:`10` 次實驗的平均值

LSTM 架構
~~~~~~~~~

+---------------------------------------+------------------------------------------------------------+----------------------------------------------------------------------+
| Hyperparameters                       | Value or Range                                             | Notes                                                                |
+=======================================+============================================================+======================================================================+
| :math:`\din`                          | :math:`1`                                                  |                                                                      |
+---------------------------------------+------------------------------------------------------------+----------------------------------------------------------------------+
| :math:`\dhid`                         | :math:`0`                                                  | No conventional hidden units are used.                               |
+---------------------------------------+------------------------------------------------------------+----------------------------------------------------------------------+
| :math:`\dblk`                         | :math:`1`                                                  |                                                                      |
+---------------------------------------+------------------------------------------------------------+----------------------------------------------------------------------+
| :math:`\nblk`                         | :math:`3`                                                  |                                                                      |
+---------------------------------------+------------------------------------------------------------+----------------------------------------------------------------------+
| :math:`\dout`                         | :math:`1`                                                  |                                                                      |
+---------------------------------------+------------------------------------------------------------+----------------------------------------------------------------------+
| :math:`g`                             | :math:`g(x) = 4 \sigma(x) - 2`                             |                                                                      |
+---------------------------------------+------------------------------------------------------------+----------------------------------------------------------------------+
| :math:`h`                             | :math:`h(x) = 2 \sigma(x) - 1`                             |                                                                      |
+---------------------------------------+------------------------------------------------------------+----------------------------------------------------------------------+
| :math:`\dim(\vWophid)`                | :math:`0`                                                  | No conventional hidden units are used.                               |
+---------------------------------------+------------------------------------------------------------+----------------------------------------------------------------------+
| :math:`\dim(\vWopblk{k})`             | :math:`\dblk \times (\din + \nblk \times (2 + \dblk) + 1)` | Fully-connected layer, used bias term on memory cell blocks.         |
+---------------------------------------+------------------------------------------------------------+----------------------------------------------------------------------+
| :math:`\dim(\vWopig)`                 | :math:`\nblk \times (\din + \nblk \times (2 + \dblk) + 1)` | Fully-connected layer, used bias term on input gate units.           |
+---------------------------------------+------------------------------------------------------------+----------------------------------------------------------------------+
| :math:`\dim(\vWopog)`                 | :math:`\nblk \times (\din + \nblk \times (2 + \dblk) + 1)` | Fully-connected layer, used bias term on output gate units.          |
+---------------------------------------+------------------------------------------------------------+----------------------------------------------------------------------+
| :math:`\dim(\vWopout)`                | :math:`\dout \times (\nblk \times \dblk)`                  | Input units are not directly connected to output units.              |
+---------------------------------------+------------------------------------------------------------+----------------------------------------------------------------------+
| Weight initalization range            | :math:`[-0.1, 0.1]`                                        |                                                                      |
+---------------------------------------+------------------------------------------------------------+----------------------------------------------------------------------+
| Input gate bias initialization range  | :math:`\Set{-1, -3, -5}`                                   | Different input gate biases were initialized with different values.  |
+---------------------------------------+------------------------------------------------------------+----------------------------------------------------------------------+
| Output gate bias initialization range | :math:`\Set{-2, -4, -6}`                                   | Different output gate biases were initialized with different values. |
+---------------------------------------+------------------------------------------------------------+----------------------------------------------------------------------+
| Learning rate                         | :math:`1`                                                  |                                                                      |
+---------------------------------------+------------------------------------------------------------+----------------------------------------------------------------------+

實驗結果
~~~~~~~~

.. figure:: https://i.imgur.com/e1OKDP5.png
  :alt: Two-Sequence Problem 實驗結果。
  :name: paper-table-4

  表格 4：Two-Sequence Problem 實驗結果。
  表格來源：:footcite:`hochreiter-etal-1997-long`。

- LSTM 能夠快速解決任務，但沒辦法比 random weight guessing 快
- LSTM 在輸入有雜訊（高斯分佈）時仍然能夠正常表現
- 額外實驗發現 bias term 初始化的數值其實不需要這麼準確

實驗 3b：Two-Sequence Problem + 雜訊
------------------------------------

.. figure:: https://i.imgur.com/DEkS8ST.png
  :alt: Two-Sequence Problem + 雜訊實驗結果
  :name: paper-table-5

  表格 5：Two-Sequence Problem + 雜訊實驗結果。
  表格來源：:footcite:`hochreiter-etal-1997-long`。

實驗設計與 LSTM 完全與實驗 3a 相同，但對於序列 :math:`\opseq` 前 :math:`N` 個實數加上雜訊（與實驗 2a 相同的高斯分佈）。

- 兩階段訓練稍微做點修改，測試方法不變

  - ST1：事先隨機抽取的 :math:`256` 筆測試資料少於 :math:`6` 筆資料分類錯誤
  - ST2：達成 ST1 後在 :math:`2560` 筆測試資料上平均錯誤低於 :math:`0.04`

- 結論

  - 增加雜訊導致誤差收斂時間變長
  - 相較於實驗 3a，雖然分類錯誤率上升，但 LSTM 仍然能夠保持較低的分類錯誤率

實驗 3c：強化版 Two-Sequence Problem
------------------------------------

.. figure:: https://i.imgur.com/1eXhAr4.png
  :alt: 強化版 Two-Sequence Problem 實驗結果
  :name: paper-table-6

  表格 6：強化版 Two-Sequence Problem 實驗結果。
  表格來源：:footcite:`hochreiter-etal-1997-long`。

實驗設計與 LSTM 完全與實驗 3b 相同，但進行以下修改

- :math:`C_1` 類別必須輸出 :math:`0.2`，:math:`C_2` 類別必須輸出 :math:`0.8`
- 在預測答案上加上雜訊

  - 使用高斯分佈作為雜訊來源，參數為 :math:`(\mu, \sigma^2) = (0, 0.1)`
  - 模型必須學會預測答案的\ **期望值**
  - 因此 LSTM 參數必須提供高精確度才能成功預測
  - 需要高精確度參數代表無法輕易靠 random weight guessing 解決此任務

- 預測結果與答案絕對誤差大於 :math:`0.1` 就算分類錯誤
- 任務目標是所有的預測絕對誤差平均值小於 :math:`0.015`
- 兩階段訓練改為一階段

  - 事先隨機抽取的 :math:`256` 筆測試資料完全分類正確
  - :math:`2560` 筆測試資料上絕對誤差平均值小於 :math:`0.015`

- Learning rate 改成 :math:`0.1`
- 結論

  - 任務變困難導致收斂時間變更長
  - 相較於實驗 3a，雖然分類錯誤率上升，但 LSTM 仍然能夠保持較低的分類錯誤率

..
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
  |$\dim(\vWopblk{k})$|$\dblk \times [\din + \nblk \times (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\vWopig)$|$\nblk \times [\din + \nblk \times (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\vWopog)$|$\nblk \times [\din + \nblk \times (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\vWopout)$|$\dout \times [\nblk \times \dblk + 1]$|外部輸入沒有直接連接到總輸出，有額外使用偏差項|
  |參數初始化範圍|$[-0.1, 0.1]$||
  |輸入閘門偏差項初始化範圍|$\Set{-3, -6}$|由大到小依序初始化不同 memory cells 對應輸入閘門偏差項|
  |Learning rate|$0.5$||

  #### 實驗結果

  <a name="paper-table-7"></a>

  表格 7：加法任務實驗結果。
  表格來源：:footcite:`hochreiter-etal-1997-long`。

  ![表 7](https://i.imgur.com/pGuMKyt.png)

  - LSTM 能夠達成任務目標
    - 不超過 $3$ 筆以上預測錯誤的資料
  - LSTM 能夠摹擬加法器，具有作為 distributed representation 的能力
  - 能夠儲存時間差至少有 $T / 2$ 以上的資訊，因此不會被**內部狀態偏差行為**影響

  ### 實驗 5：乘法任務

  #### 任務定義

  從 LSTM 的架構上來看實驗 4 的加法任務可以透過 $\eqref{14-1}$ 輕鬆完成，因此實驗 5 的目標是確認模型是否能夠從加法上延伸出乘法的概念，確保實驗 4 並不只是單純因模型架構而解決。

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
  表格來源：:footcite:`hochreiter-etal-1997-long`。

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
  |$\dim(\vWopblk{k})$|$\dblk \times [\din + \nblk \times (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\vWopig)$|$\nblk \times [\din + \nblk \times (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\vWopog)$|$\nblk \times [\din + \nblk \times (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\vWopout)$|$\dout \times [\nblk \times \dblk + 1]$|外部輸入沒有直接連接到總輸出，有額外使用偏差項|
  |參數初始化範圍|$[-0.1, 0.1]$||
  |輸入閘門偏差項初始化範圍|$\Set{-2, -4}$|由大到小依序初始化不同 memory cells 對應輸入閘門偏差項|
  |Learning rate|$0.5$||

  #### 實驗結果

  <a name="paper-table-9"></a>

  表格 9：Temporal Order with 4 Classes 任務實驗結果。
  表格來源：:footcite:`hochreiter-etal-1997-long`。

  ![表 9](https://i.imgur.com/ucyQoeQ.png)

  - LSTM 的平均誤差低於 $0.1$

    - 沒有超過 $3$ 筆以上的預測錯誤
  - LSTM 可能使用以下的方法進行解答
    - 擁有 $2$ 個 memory cells 時，依照順序記住出現的資訊
    - 只有 $1$ 個 memory cells 時，LSTM 可以改成記憶狀態的轉移

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
  |$\dim(\vWopblk{k})$|$\dblk \times [\din + \nblk \times (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\vWopig)$|$\nblk \times [\din + \nblk \times (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\vWopog)$|$\nblk \times [\din + \nblk \times (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\vWopout)$|$\dout \times [\nblk \times \dblk + 1]$|外部輸入沒有直接連接到總輸出，有額外使用偏差項|
  |參數初始化範圍|$[-0.1, 0.1]$||
  |輸入閘門偏差項初始化範圍|$\Set{-2, -4, -6}$|由大到小依序初始化不同 memory cells 對應輸入閘門偏差項|
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

