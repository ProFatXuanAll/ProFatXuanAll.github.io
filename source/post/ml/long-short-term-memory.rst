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

.. ====================================================================================================================
.. Define math macros.
.. ====================================================================================================================

.. math::
  :nowrap:

  \[
    % Operator names.
    \newcommand{\opnet}{\operatorname{net}}
    \newcommand{\opin}{\operatorname{in}}
    \newcommand{\opout}{\operatorname{out}}

    % Dimensions.
    \newcommand{\din}{{d_{\opin}}}
    \newcommand{\dout}{{d_{\opout}}}

    % Neural network units.
    \newcommand{\net}[2]{{\opnet_{#1}\qty(#2)}}
    \newcommand{\fnet}[2]{f_{#1}\qty(\net{#1}{#2})}

    % Total loss.
    \newcommand{\tloss}{\operatorname{TotalLoss}}
  \]

..
  <!-- Operator hid. -->
  $\providecommand{\ophid}{}$
  $\renewcommand{\ophid}{\operatorname{hid}}$
  <!-- Operator cell block. -->
  $\providecommand{\opblk}{}$
  $\renewcommand{\opblk}{\operatorname{block}}$
  <!-- Operator cell multiplicative input gate. -->
  $\providecommand{\opig}{}$
  $\renewcommand{\opig}{\operatorname{ig}}$
  <!-- Operator cell multiplicative output gate. -->
  $\providecommand{\opog}{}$
  $\renewcommand{\opog}{\operatorname{og}}$
  <!-- Operator sequence. -->
  $\providecommand{\opseq}{}$
  $\renewcommand{\opseq}{\operatorname{seq}}$

  <!-- Derivative of f with respect to net input. -->
  $\providecommand{\dfnet}{}$
  $\renewcommand{\dfnet}[2]{f_{#1}'\big(\net{#1}{#2}\big)}$

  <!-- Input dimension. -->
  $\providecommand{\din}{}$
  $\renewcommand{\din}{d_{\opin}}$
  <!-- Output dimension. -->
  $\providecommand{\dout}{}$
  $\renewcommand{\dout}{d_{\opout}}$
  <!-- Hidden dimension. -->
  $\providecommand{\dhid}{}$
  $\renewcommand{\dhid}{d_{\ophid}}$
  <!-- Cell block dimension. -->
  $\providecommand{\dblk}{}$
  $\renewcommand{\dblk}{d_{\opblk}}$
  <!-- Number of cell blocks. -->
  $\providecommand{\nblk}{}$
  $\renewcommand{\nblk}{n_{\opblk}}$

  <!-- Past and Future time -->
  $\providecommand{\tp}{}$
  $\renewcommand{\tp}{t_{\operatorname{past}}}$
  $\providecommand{\tf}{}$
  $\renewcommand{\tf}{t_{\operatorname{future}}}$
  <!-- Graident of loss(t_2) with respect to net k_0 at time t_1. -->
  $\providecommand{\dv}{}$
  $\renewcommand{\dv}[3]{\vartheta_{#1}^{#2}[#3]}$

  <!-- Cell block k. -->
  $\providecommand{\blk}{}$
  $\renewcommand{\blk}[1]{\opblk^{#1}}$

  <!-- Weight of multiplicative input gate. -->
  $\providecommand{\wig}{}$
  $\renewcommand{\wig}{w^{\opig}}$
  <!-- Weight of multiplicative output gate. -->
  $\providecommand{\wog}{}$
  $\renewcommand{\wog}{w^{\opog}}$
  <!-- Weight of hidden units. -->
  $\providecommand{\whid}{}$
  $\renewcommand{\whid}{w^{\ophid}}$
  <!-- Weight of cell block units. -->
  $\providecommand{\wblk}{}$
  $\renewcommand{\wblk}[1]{w^{\blk{#1}}}$
  <!-- Weight of output units. -->
  $\providecommand{\wout}{}$
  $\renewcommand{\wout}{w^{\opout}}$

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
  <!-- Net input of hidden unit. -->
  $\providecommand{\nethid}{}$
  $\renewcommand{\nethid}[2]{\opnet_{#1}^{\ophid}(#2)}$
  <!-- Net input of hidden unit with activatiton f. -->
  $\providecommand{\fnethid}{}$
  $\renewcommand{\fnethid}[2]{f_{#1}^{\ophid}\big(\nethid{#1}{#2}\big)}$
  <!-- Derivative of f with respect to net input of hidden units. -->
  $\providecommand{\dfnethid}{}$
  $\renewcommand{\dfnethid}[2]{f_{#1}^{\ophid}{'}\big(\nethid{#1}{#2}\big)}$
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
  $\providecommand{\netcell}{}$
  $\renewcommand{\netcell}[3]{\opnet_{#1}^{\blk{#2}}(#3)}$

  <!-- Gradient approximation by truncating gradient. -->
  $\providecommand{\aptr}{}$
  $\renewcommand{\aptr}{\approx_{\operatorname{tr}}}$


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

此論文討論的主要對象為隨著時間改變輸入的 RNN。

過往 RNN 模型的問題
-------------------

- 常用於 RNN 模型的最佳化演算法 :term:`BPTT` 與 :term:`RTRL` 都會遇到\ **梯度爆炸**\（:term:`gradient explosion`）或\ **梯度消失**\（:term:`gradient vanishing`）的問題

  - 梯度爆炸造成神經網路的\ **參數數值劇烈振盪**\（**oscillating weights**）
  - 梯度消失造成\ **訓練時間慢長**

- 關鍵輸入資訊\ **時間差較短**\（**short time lags**）的任務可以使用 time-delay neural network :footcite:`lang-etal-1990-a` 解決，但關鍵輸入資訊\ **時間差較長**\（**long time lags**）的任務並沒有好的解決方案

  - 已知的模型解決方案會隨著時間差越長導致模型所需參數越多
  - 已知的最佳化解決方案時間複雜度過高
  - 部份已知的測試任務可能過於簡單，甚至可依靠隨機參數猜測（random weight guessing）解決

計算定義
--------

給定一資料集，資料集中的每個資料點都由兩個序列組成，分別稱為輸入序列 :math:`x` 與答案序列 :math:`\hat{y}` 。
當一個 RNN 模型被用於訓練在該資料集上，我們會希望給予任意資料點中的輸入序列 :math:`x` 所得到的 RNN 輸出序列 :math:`y` 會近似於對應資料點的答案序列 :math:`\hat{y}`。

假定一個資料點中的輸入序列長度為 :math:`T`，則我們可定義以下符號：

- 定義 :math:`x(t)` 為資料點輸入序列 :math:`x` 中，時間點 :math:`t` 所對應到的資料

  - 令 :math:`t \in \Set{0, 1, \dots, T-1}`
  - 定義 :math:`x(t)` 為一向量，由 :math:`\din` 個實數組成，即 :math:`x(t) \in \R^\din`
  - 定義 :math:`x_j(t)` 為向量 :math:`x(t)` 的第 :math:`j` 個實數，:math:`j \in \Set{1, \dots, \din}`

- 定義 :math:`\hat{y}(t)` 為資料點答案序列 :math:`\hat{y}` 中，時間點 :math:`t` 所對應到的資料

  - 令 :math:`t \in \Set{1, 2, \dots, T}`，注意此處定義與 :math:`x(t)` 的 index 範圍不同
  - 定義 :math:`\hat{y}(t)` 為一向量，由 :math:`\dout` 個實數組成，即 :math:`\hat{y}(t) \in \R^\dout`
  - 定義 :math:`\hat{y}_j(t)` 為向量 :math:`\hat{y}(t)` 的第 :math:`j` 個實數，:math:`j \in \Set{1, \dots, \dout}`

- 定義 :math:`y(t)` 為 RNN 輸出序列 :math:`y` 中，時間點 :math:`t` 所對應到的資料

  - 由於目標是讓 :math:`y \approx \hat{y}`，因此定義 :math:`y(t) \in \R^\dout`
  - 定義 :math:`t \in \Set{1, 2, \dots, T}`
  - 定義 :math:`y_j(t)` 為向量 :math:`y(t)` 的第 :math:`j` 個實數，:math:`j \in \Set{1, \dots, \dout}`

- 定義常數 :math:`y(0) = \zv`

  - :math:`\zv` 是由 :math:`\dout` 個零組成的零向量
  - 注意此定義並無與 :math:`y(1), \dots, y(T)` 衝突

- 定義 :math:`w` 為 RNN 模型的參數

  - RNN 在時間點 :math:`t` 取得的輸入為資料點輸入 :math:`x(t)` 與前一次的模型輸出 :math:`y(t)`
  - RNN 在取得時間點 :math:`t` 的輸入後，乘上參數 :math:`w`，必須得到由 :math:`\dout` 個實數組成的淨輸入向量 :math:`\opnet(t+1)`
  - 因此模型參數 :math:`w` 為一矩陣，由 :math:`\dout \times (\din + \dout)` 個實數組成，即 :math:`w \in \R^{\dout \times (\din + \dout)}`

- 定義 :math:`\opnet(t)` 為 RNN 模型在時間點 :math:`t` 得到的淨輸入（net input）

  - 定義 :math:`t \in \Set{1, 2, \dots, T}`
  - 淨輸入的定義為參數矩陣 :math:`w` 乘上所有輸入
  - RNN 所有的輸入來源為資料點輸入 :math:`x(t)` 與前一次的模型輸出 :math:`y(t)`

- 定義 :math:`f` 為 RNN 模型的 :term:`activation function`

  - :math:`f` 的輸入是 :math:`\opnet(t)`
  - 定義 :math:`y(t) = f(\opnet(t))`
  - 定義 :math:`f_i` 為 :math:`f` 的第 :math:`i` 個 real valued function，:math:`i \in \Set{1, \dots, \dout}`
  - 使用下標 :math:`f_i` 是因為每個維度所使用的啟發函數可以\ **不同**
  - :math:`f` 必須要可以\ **微分**，當時與 RNN 有關的論文幾乎都是令 :math:`f_i` 為 sigmoid 函數 :math:`\sigma(z) = \frac{1}{1 + e^{-z}}`
  - 後續論文分析都是採用 sigmoid 函數，因此我們直接以 :math:`\sigma` 表達 :math:`f_i`

透過以上符號我們可以描述 RNN 模型的 :term:`forward pass`：

.. math::
  :nowrap:

  \[
    \begin{align*}
      & \algoProc{\operatorname{RNN}}(x, T) \\
      & \indent{1} y(0) \algoEq \zv \\
      & \indent{1} \algoFor{t \in \Set{0, \dots, T-1}} \\
      & \indent{2} \algoFor{i \in \Set{1, \dots, \dout}} \\
      & \indent{3} \net{i}{t+1} \algoEq \sum_{j = 1}^\din w_{i, j} \cdot x_j(t) + \sum_{j = \din + 1}^{\din + \dout} w_{i, j} \cdot y_j(t) \\
      & \indent{3} y_i(t+1) \algoEq \fnet{i}{t+1} \\
      & \indent{2} \algoEndFor \\
      & \indent{1} \algoEndFor \\
      & \indent{1} \algoReturn y(1), \dots, y(T) \\
      & \algoEndProc
    \end{align*}
  \]

之後若非必要，我們將不再展開矩陣乘法的計算內容，因此上述演算法可以改寫為

.. math::
  :nowrap:

  \[
    \begin{align*}
      & \algoProc{\operatorname{RNN}}(x, T) \\
      & \indent{1} y(0) \algoEq \zv \\
      & \indent{1} \algoFor{t \in \Set{0, \dots, T-1}} \\
      & \indent{2} \opnet(t+1) \algoEq w \cdot \begin{pmatrix}
          x(t) \\
          y(t)
        \end{pmatrix} \\
      & \indent{2} y(t+1) \algoEq f\qty(\opnet(t+1)) \\
      & \indent{1} \algoEndFor \\
      & \indent{1} \algoReturn y(1), \dots, y(T) \\
      & \algoEndProc
    \end{align*}
  \]

目標函數
---------

此論文設定 RNN 模型的目標函數為\ **最小平方差**\（**Mean Square Error**）：

.. math::
  :nowrap:

  \[
    \begin{align*}
      & \algoProc{\operatorname{MSE}}(x, \hat{y}, T) \\
      & \indent{1} y \algoEq \operatorname{RNN}(x, T) \\
      & \indent{1} \algoFor{t \in \Set{1, \dots, T}} \\
      & \indent{2} \algoFor{i \in \Set{1, \dots, \dout}} \\
      & \indent{3} \loss_i(t) \algoEq \frac{1}{2} \qty(y_i(t) - \hat{y}_{i}(t))^2 \\
      & \indent{2} \algoEndFor \\
      & \indent{2} \tloss(t) \algoEq \sum_{i = 1}^{\dout} \loss_i(t) \\
      & \indent{1} \algoEndFor \\
      & \algoEndProc
    \end{align*}
  \]

計算目標函數的梯度
------------------

根據目標函數的定義，我們知道 :math:`y_i(t+1)` 對 :math:`\tloss(t+1)` 所得梯度為

.. math::
  :nowrap:

  \begin{align*}
    \dv{\tloss(t+1)}{y_i(t+1)} &= \dv{\tloss(t+1)}{\loss_i(t+1)} \cdot \dv{\loss_i(t+1)}{y_i(t+1)} \\
    &= 1 \cdot \qty(y_i(t+1) - \hat{y}_{i}(t+1)) \\
    &= y_i(t+1) - \hat{y}_{i}(t+1). \tag{1}\label{1}
  \end{align*}

根據 :math:`\eqref{1}` 我們可以推得 :math:`\net{i}{t+1}` 對 :math:`\tloss(t+1)` 所得梯度為

.. math::
  :nowrap:

  \begin{align*}
    \dv{\tloss(t+1)}{\net{i}{t+1}} &= \dv{\tloss(t+1)}{y_i(t+1)} \cdot \dv{y_i(t+1)}{\net{i}{t+1}} \\
    &= \sigma'\qty(\net{i}{t+1}) \cdot \qty(y_i(t+1) - \hat{y}_{i}(t+1)). \tag{2}\label{2}
  \end{align*}

.. note::

  式子 :math:`\eqref{2}` 就是論文 3.1.1 節的第一條公式。

根據 :math:`\eqref{2}` 我們可以推得 :math:`y_j(t)` 對 :math:`\tloss(t+1)` 所得梯度為（注意時間差）

.. math::
  :nowrap:

  \begin{align*}
    \dv{\tloss(t+1)}{y_j(t)} & = \sum_{i = 1}^{\dout} \qty[\dv{\tloss(t+1)}{\net{i}{t+1}} \cdot \dv{\net{i}{t+1}}{y_j(t)}] \\
    & = \sum_{i = 1}^{\dout} \qty[\sigma'\qty(\net{i}{t+1}) \cdot \qty(y_i(t+1) - \hat{y}_{i}(t+1)) \cdot w_{i, j}] \tag{3}\label{3}
  \end{align*}

..
  由於第 $t$ 時間點的輸出 $y(t)$ 的計算是由 $\opnet(t)$ 而來（請見 $\eqref{2}$），所以我們也利用 $\eqref{6}$ 計算 $\net{j}{t}$ 對 $\tloss(t+1)$ 所得梯度（注意是 $t$ 不是 $t+1$）

  $$
  \begin{align*}
  \pd{\tloss(t+1)}{\net{j}{t}} & = \pd{\tloss(t+1)}{y_j(t)} \cdot \pd{y_j(t)}{\net{j}{t}} \\
  & = \sum_{i = 1}^{\dout} \bigg[\pd{\tloss(t+1)}{\net{i}{t+1}} \cdot w_{i, j} \cdot \sigma'\pa{\net{j}{t+1}}\bigg] \\
  & = \sigma'\pa{\net{j}{t+1}} \cdot \sum_{i = 1}^{\dout} \bigg[w_{i, j} \cdot \pd{\tloss(t+1)}{\net{i}{t+1}}\bigg]
  \end{align*} \tag{7}\label{7}
  $$

  式子 $\eqref{7}$ 就是論文 3.1.1 節的最後一條公式。

  模型參數 $w_{i, j}$ 對於 $\tloss(t+1)$ 所得梯度為

  $$
  \begin{align*}
  & \pd{\tloss(t+1)}{w_{i, j}} \\
  & = \sum_{k = 1}^{\dout} \pd{\tloss(t+1)}{\net{k}{t+1}} \cdot \pd{\net{k}{t+1}}{w_{i, j}} \\
  & = \sum_{k = 1}^{\dout} \pd{\tloss(t+1)}{\net{k}{t+1}} \cdot \br{\sum_{j^{\star} = 1}^{\dout + \din} \pa{\pd{w_{k, j^{\star}}}{w_{i, j}} \cdot \begin{pmatrix}
  y(t) \\
  x(t)
  \end{pmatrix}_{j^{\star}} + w_{k, j^{\star}} \cdot \pd{\begin{pmatrix}
  y(t) \\
  x(t)
  \end{pmatrix}_{j^{\star}}}{w_{i, j}}}} \\
  & = \sum_{k = 1}^{\dout} \pd{\tloss(t+1)}{\net{k}{t+1}} \cdot \br{\begin{pmatrix}
  y(t) \\
  x(t)
  \end{pmatrix}_j + \sum_{j^{\star} = 1}^{\dout} w_{k, j^{\star}} \cdot \sigma'\pa{\net{j^{\star}}{t}} \cdot \pd{\net{j^{\star}}{t}}{w_{i, j}}}
  \end{align*} \tag{8}\label{8}
  $$

  而在時間點 $t+1$ 進行參數更新的方法為

  $$
  w_{i, j} \leftarrow w_{i, j} - \alpha \pd{\tloss(t+1)}{w_{i, j}} \tag{9}\label{9}
  $$

  $\eqref{9}$ 就是最常用來最佳化神經網路的**梯度下降演算法**（Gradient Descent），$\alpha$ 代表**學習率**（Learning Rate）。

  ### 梯度爆炸 / 消失

  從 $\eqref{7}$ 式我們可以進一步推得 $t$ 時間點造成的梯度與前次時間點 ($t - 1, t - 2, \dots$) 所得的梯度**變化關係**。
  注意這裡的變化關係指的是梯度與梯度之間的**變化率**，意即用時間點 $t - 1$ 的梯度對時間點 $t$ 的梯度算微分。

  為了方便計算，我們定義新的符號

  $$
  \dv{k}{\tf}{\tp} = \pd{\Loss{\tf}}{\net{k}{\tp}} \tag{10}\label{10}
  $$

  意思是在**過去**時間點 $\tp$ 的第 $k$ 個**模型內部節點** $\net{k}{\tp}$ 對於**未來**時間點 $\tf$ 貢獻的**總誤差** $\Loss{\tf}$ 計算所得之**梯度**。

  - 注意是貢獻總誤差所得之**梯度**
  - 根據時間的限制我們有不等式 $0 \leq \tp \leq \tf$
  - 節點 $k$ 的數值範圍為 $k \in \set{1, \dots, \dout}$，見式子 $\eqref{1}$

  因此

  $$
  \begin{align*}
  & \dv{k_0}{t}{t} = \pd{\Loss{t}}{\net{k_0}{t}}; \\
  & \dv{k_1}{t}{t - 1} = \pd{\Loss{t}}{\net{k_1}{t - 1}} \\
  & = \sigma'\pa{\net{k_1}{t - 1}} \cdot \pa{\sum_{k_0 = 1}^{\dout} w_{k_0, k_1} \cdot \dv{k_0}{t}{t}}; \\
  & \dv{k_2}{t}{t - 2} = \pd{\Loss{t}}{\net{k_2}{t - 2}} \\
  & = \sum_{k_1 = 1}^{\dout} \br{\pd{\Loss{t}}{\net{k_1}{t - 1}} \cdot \pd{\net{k_1}{t - 1}}{y_{k_2}(t - 2)} \cdot \pd{y_{k_2}(t - 2)}{\net{k_2}{t - 2}}} \\
  & = \sum_{k_1 = 1}^{\dout} \br{\dv{k_1}{t}{t - 1} \cdot w_{k_1, k_2} \cdot \sigma'\pa{\net{k_2}{t - 2}}} \\
  & = \sum_{k_1 = 1}^{\dout} \br{\sigma'\pa{\net{k_1}{t - 1}} \cdot \pa{\sum_{k_0 = 1}^{\dout} w_{k_0, k_1} \cdot \dv{k_0}{t}{t}} \cdot w_{k_1, k_2} \cdot \sigma'\pa{\net{k_2}{t - 2}}} \\
  & = \sum_{k_1 = 1}^{\dout} \sum_{k_0 = 1}^{\dout} \br{w_{k_0, k_1} \cdot w_{k_1, k_2} \cdot \sigma'\pa{\net{k_1}{t - 1}} \cdot \sigma'\pa{\net{k_2}{t - 2}} \cdot \dv{k_0}{t}{t}}; \\
  & \dv{k_3}{t}{t - 3} = \sum_{k_2 = 1}^{\dout} \br{\pd{\Loss{t}}{\net{k_2}{t - 2}} \cdot \pd{\net{k_2}{t - 2}}{y_{k_3}(t - 3)} \cdot \pd{y_{k_3}(t - 3)}{\net{k_3}{t - 3}}} \\
  & = \sum_{k_2 = 1}^{\dout} \br{\dv{k_2}{t}{t - 2} \cdot w_{k_2, k_3} \cdot \sigma'\pa{\net{k_3}{t - 3}}} \\
  & = \sum_{k_2 = 1}^{\dout} \Bigg[\sum_{k_1 = 1}^{\dout} \sum_{k_0 = 1}^{\dout} \br{w_{k_0, k_1} \cdot w_{k_1, k_2} \cdot \sigma'\pa{\net{k_1}{t - 1}} \cdot \sigma'\pa{\net{k_2}{t - 2}} \cdot \dv{k_0}{t}{t}} \\
  & \quad \cdot w_{k_2, k_3} \cdot \sigma'\pa{\net{k_3}{t - 3}}\Bigg] \\
  & = \sum_{k_2 = 1}^{\dout} \sum_{k_1 = 1}^{\dout} \sum_{k_0 = 1}^{\dout} \bigg[w_{k_0, k_1} \cdot w_{k_1, k_2} \cdot w_{k_2, k_3} \cdot \\
  & \quad \sigma'\pa{\net{k_1}{t - 1}} \cdot \sigma'\pa{\net{k_2}{t - 2}} \cdot \sigma'\pa{\net{k_3}{t - 3}} \cdot \dv{k_0}{t}{t}\bigg] \\
  & = \sum_{k_2 = 1}^{\dout} \sum_{k_1 = 1}^{\dout} \sum_{k_0 = 1}^{\dout} \br{\br{\prod_{q = 1}^{3} w_{k_{q - 1}, k_q} \cdot \sigma'\pa{\net{k_q}{t - q}}} \cdot \dv{k_0}{t}{t}}
  \end{align*} \tag{11}\label{11}
  $$

  由 $\eqref{11}$ 我們可以歸納得出 $n \geq 1$ 時的公式

  $$
  \dv{k_{n}}{t}{t - n} = \sum_{k_{n - 1} = 1}^{\dout} \cdots \sum_{k_{0} = 1}^{\dout} \br{\br{\prod_{q = 1}^{n} w_{k_{q - 1}, k_{q}} \cdot \sigma'\pa{\net{k_{q}}{t - q}}} \cdot \dv{k_{0}}{t}{t}} \tag{12}\label{12}
  $$

  由 $\eqref{12}$ 我們可以看出 $\dv{k_{n}}{t}{t - n}$ 都與 $\dv{k_{0}}{t}{t}$ 相關，因此我們將 $\dv{k_{n}}{t}{t - n}$ 想成由 $\dv{k_{0}}{t}{t}$ 構成的函數。

  現在讓我們固定 $k_{0}^{\star} \in \set{1, \dots, \dout}$，我們可以計算 $\dv{k_{0}^{\star}}{t}{t}$ 對於 $\dv{k_{n}}{t}{t - n}$ 的微分，分析**梯度**在進行**反向傳遞過程**中的**變化率**

  - 當 $n = 1$ 時，根據 $\eqref{11}$ 我們可以推得論文中的 (3.1) 式

    $$
    \pd{\dv{k_{n}}{t}{t - n}}{\dv{k_{0}^{\star}}{t}{t}} = w_{k_{0}^{\star}, k_{1}} \cdot \sigma'\pa{\net{k_{1}}{t - 1}} \tag{13}\label{13}
    $$

  - 當 $n > 1$ 時，根據 $\eqref{12}$ 我們可以推得論文中的 (3.2) 式

    $$
    \pd{\dv{k_{n}}{t}{t - n}}{\dv{k_{0}^{\star}}{t}{t}} = \sum_{k_{n - 1} = 1}^{\dout} \cdots \sum_{k_{1} = 1}^{\dout} \sum_{k_{0} \in \set{k_{0}^{\star}}} \br{\prod_{q = 1}^{n} w_{k_{q - 1}, k_{q}} \cdot \sigma'\pa{\net{k_{q}}{t - q}}} \tag{14}\label{14}
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
  & = \abs{w_{k_0, k_1} \cdot \prod_{q = 2}^{n} \bigg[\sigma'\pa{\net{k_{q - 1}}{t - q + 1}} \cdot w_{k_{q - 1}, k_{q}}\bigg] \cdot \sigma'\pa{\net{k_{n}}{t - n}}} \\
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
  \pd{\Big[\sigma\big(\net{l_{m}}{t - m}\big) \cdot \Big(1 - \sigma\big(\net{l_{m}}{t - m}\big)\Big) \cdot w_{l_{m} l_{m - l}}\Big]}{w_{l_{m} l_{m - 1}}} = 0
  $$

  拆解微分式可得

  $$
  \begin{align*}
  & \pd{\Big[\sigma\big(\net{l_{m}}{t - m}\big) \cdot \Big(1 - \sigma\big(\net{l_{m}}{t - m}\big)\Big) \cdot w_{l_{m} l_{m - l}}\Big]}{w_{l_{m} l_{m - 1}}} \\
  & = \pd{\sigma\big(\net{l_{m}}{t - m}\big)}{\net{l_{m}}{t - m}} \cdot \pd{\net{l_{m}}{t - m}}{w_{l_{m} l_{m - 1}}} \cdot \Big(1 - \sigma\big(\net{l_{m}}{t - m}\big)\Big) \cdot w_{l_{m} l_{m - l}} \\
  & \quad + \sigma\big(\net{l_{m}}{t - m}\big) \cdot \pd{\Big(1 - \sigma\big(\net{l_{m}}{t - m}\big)\Big)}{\net{l_{m}}{t - m}} \cdot \pd{\net{l_{m}}{t - m}}{w_{l_{m} l_{m - 1}}} \cdot w_{l_{m} l_{m - l}} \\
  & \quad + \sigma\big(\net{l_{m}}{t - m}\big) \cdot \Big(1 - \sigma\big(\net{l_{m}}{t - m}\big)\Big) \cdot \pd{w_{l_{m} l_{m - 1}}}{w_{l_{m} l_{m - 1}}} \\
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
  \sum_{k_{0}^{\star} = 1}^{\dout} \pd{\dv{k_{n}}{t}{t - n}}{\dv{k_{0}^{\star}}{t}{t}} \tag{20}\label{20}
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
  \dv{j}{t}{t - 1} = w_{j, j} \cdot \dfnet{j}{t - 1} \cdot \dv{j}{t}{t} \tag{22}\label{22}
  $$

  為了不讓梯度 $\dv{j}{t}{t}$ 在傳遞的過程消失，作者認為需要強制達成**梯度常數（Constant Error Flow）**

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

  ### 初始狀態

  我們將 $\eqref{1}$ 中的計算重新定義，並新增幾個符號：

  |符號|意義|數值範圍|
  |-|-|-|
  |$\dhid$|**隱藏單元**的個數|$\N$|
  |$\dblk$|每個記憶細胞區域中**記憶細胞**的個數|$\Z^+$|
  |$\nblk$|**記憶細胞區域**的個數|$\Z^+$|

  - 因為論文 4.3 節有提到可以完全沒有**隱藏單元**，因此允許 $\dhid = 0$
    - 此論文的後續研究似乎都沒有使用隱藏單元
    - 例如更新 LSTM 架構的主要研究 [LSTM-2000][LSTM2000] 與 [LSTM-2002][LSTM2002] 都沒有使用隱藏單元
  - 根據論文 4.4 節，可以**同時**擁有 $\nblk$ 個不同的**記憶細胞區域**，因此允許 $\nblk \geq 1$

  接著我們定義 $t$ 時間點的模型計算狀態：

  |符號|意義|數值範圍|
  |-|-|-|
  |$y^{\ophid}(t)$|**隱藏單元（Hidden Units）**|$\R^{\dhid}$|
  |$y^{\opig}(t)$|**輸入閘門單元（Input Gate Units）**|$\R^{\nblk}$|
  |$y^{\opog}(t)$|**輸出閘門單元（Output Gate Units）**|$\R^{\nblk}$|
  |$y^{\blk{k}}(t)$|**記憶細胞區域** $k$ 的**輸出**|$\R^{\dblk}$|
  |$s^{\blk{k}}(t)$|**記憶細胞區域** $k$ 的**內部狀態**|$\R^{\dblk}$|
  |$y(t)$|**模型總輸出**|$\R^{\dout}$|

  - 以上所有向量全部都**初始化**成各自維度的**零向量**，也就是 $t = 0$ 時模型**所有節點**（除了**輸入**）都是 $0$
  - 根據論文 4.4 節，可以**同時**擁有 $\nblk$ 個不同的**記憶細胞**
    - [圖 2](#paper-fig-2) 模型共有 $2$ 個不同的記憶細胞
    - **記憶細胞區域**上標 $k$ 的數值範圍為 $k \in \set{1, \dots, \nblk}$
  - **同一個**記憶細胞區域**共享閘門單元**，因此 $y^{\opig}(t), y^{\opog}(t)$ 的維度為 $\nblk$
  - 根據論文 4.3 節，**記憶細胞**、**閘門單元**與**隱藏單元**都算是**隱藏層（Hidden Layer）**的一部份
    - **外部輸入**會與**隱藏層**和**總輸出**連接
    - **隱藏層**會與**總輸出**連接（但**閘門**不會）

  > **All units** (except for gate units) in all layers have **directed** connections (serve as input) to **all units** in the **layer above** (or to **all higher layers**; see experiments 2a and 2b)

  ### 計算定義

  當我們得到 $t$ 時間點的外部輸入 $x(t)$ 時，我們可以進行以下計算得到 $t+1$ 時間點的總輸出 $y(t+1)$

  $$
  \begin{align*}
  D & = \din + \dhid + \nblk \cdot (2 + \dblk) \tag{28}\label{28} \\
  \tilde{x}(t) & = \begin{pmatrix}
  x(t) \\
  y^{\ophid}(t) \\
  y^{\opig}(t) \\
  y^{\opog}(t) \\
  y^{\blk{1}}(t) \\
  \vdots \\
  y^{\blk{\nblk}}(t)
  \end{pmatrix} \in \R^D \tag{29}\label{29} \\
  k & \in \set{1, \dots, \nblk} \tag{30}\label{30} \\
  y^{\ophid}(t+1) & = f^{\ophid}\pa{\opnet^{\ophid}(t+1)} = f^{\ophid}\pa{\whid \cdot \tilde{x}(t)} \tag{31}\label{31} \\
  y^{\opig}(t+1) & = f^{\opig}\pa{\opnet^{\opig}(t+1)} = f^{\opig}\pa{\wig \cdot \tilde{x}(t)} \tag{32}\label{32} \\
  y^{\opog}(t+1) & = f^{\opog}\pa{\opnet^{\opog}(t+1)} = f^{\opog}\pa{\wog \cdot \tilde{x}(t)} \tag{33}\label{33} \\
  s^{\blk{k}}(t+1) & = s^{\blk{k}}(t) + y_k^{\opig}(t+1) \cdot g\pa{\opnet^{\blk{k}}(t+1)} \tag{34}\label{34} \\
  & = s^{\blk{k}}(t) + y_k^{\opig}(t+1) \cdot g\pa{\wblk{k} \cdot \tilde{x}(t)} \\
  y^{\blk{k}}(t+1) & = y_k^{\opog}(t+1) \cdot h\pa{s^{\blk{k}}(t+1)} \tag{35}\label{35} \\
  y(t+1) & = f^{\opout}(\opnet^{\opout}(t+1)) = f^{\opout}\pa{\wout \cdot \begin{pmatrix}
  x(t) \\
  y^{\ophid}(t+1) \\
  y^{\blk{1}}(t+1) \\
  \vdots \\
  y^{\blk{\nblk}}(t+1)
  \end{pmatrix}} \tag{36}\label{36}
  \end{align*}
  $$

  以上就是 LSTM（1997 版本）的計算流程。

  - $f^{\ophid}, f^{\opig}, f^{\opog}, f^{\opout}, g, h$ 都是 differentiable element-wise activation function，大部份都是 sigmoid 或是 sigmoid 的變形
  - $f^{\opig}, f^{\opog}$ 的數值範圍（range）必須限制在 $[0, 1]$，才能達成閘門的功能
  - $f^{\opout}$ 的數值範圍只跟任務有關
  - 論文並沒有給 $f^{\ophid}, g, h$ 任何數值範圍的限制

  論文 4.3 節有提到可以完全沒有**隱藏單元**，而後續的研究（例如 [LSTM-2000][LSTM2000]、[LSTM-2002][LSTM2002]）也完全沒有使用隱藏單元，因此 $\eqref{31}$ 可以完全不存在。

  - $\eqref{29}$ 中的 $y^{\ophid}(t)$ 必須去除
  - $\eqref{36}$ 中的 $y^{\ophid}(t+1)$ 必須去除
  - 隱藏單元的設計等同於**保留** $\eqref{1} \eqref{2}$ 的架構，是個不好的設計，因此論文後續在**最佳化**的過程中動了手腳

  根據 $\eqref{32} \eqref{34}$，在計算完 $t+1$ 時間點的**輸入閘門** $y^{\opig}(t+1)$ 後便可以更新 $t+1$ 時間點的**記憶細胞內部狀態** $s^{\blk{k}}(t+1)$。

  - **記憶細胞淨輸入**會與**輸入閘門**進行**相乘**，因此稱為**乘法輸入閘門**
  - 由於 $t+1$ 時間點的資訊有加上 $t$ 時間點的資訊，因此稱為**自連接線性單元**
  - 同一個記憶細胞區域會**共享**同一個輸入閘門，因此 $\eqref{34}$ 中的乘法是**純量乘上向量**，這也是 $y^{ig}(t+1) \in \R^{\nblk}$ 的理由
  - 當模型認為**輸入訊號不重要**時，模型應該要**關閉輸入閘門**，即 $y_k^{\opig}(t+1) \approx 0$
    - 丟棄**當前**輸入訊號，只以**過去資訊**進行決策
    - 在此狀態下 $t+1$ 時間點的**記憶細胞內部狀態**與 $t$ 時間點**完全相同**，達成 $\eqref{23} \eqref{25}$，藉此保障**梯度不會消失**
  - 當模型認為**輸入訊號重要**時，模型應該要**開啟輸入閘門**，即 $y_k^{\opig}(t+1) \approx 1$
  - 不論**輸入訊號** $g\pa{\opnet^{\blk{k}}(t+1)}$ 的大小，只要 $y_k^{\opig}(t+1) \approx 0$，則輸入訊號**完全無法影響**接下來的所有計算，LSTM 以此設計避免 $\eqref{26}$ 所遇到的困境

  根據 $\eqref{33} \eqref{35}$，在計算完 $t+1$ 時間點的**輸出閘門** $y^{\opog}(t+1)$ 與**記憶細胞內部狀態** $s^{\blk{k}}(t+1)$ 後便可以得到 $t+1$ 時間點的**記憶細胞輸出** $y^{\blk{k}}(t+1)$。

  - **記憶細胞啟發值**會與**輸出閘門**進行**相乘**，因此稱為**乘法輸出閘門**
  - 同一個記憶細胞區域會**共享**同一個輸出閘門，因此 $\eqref{35}$ 中的乘法是**純量乘上向量**，這也是 $y^{og}(t+1) \in \R^{\nblk}$ 的理由
  - 當模型認為**輸出訊號**會導致**當前計算錯誤**時，模型應該**關閉輸出閘門**，即 $y_k^{\opog}(t+1) \approx 0$
    - 在**輸入**閘門**開啟**的狀況下，**關閉輸出**閘門代表不讓**現在**時間點的資訊影響當前計算
    - 在**輸入**閘門**關閉**的狀況下，**關閉輸出**閘門代表不讓**過去**時間點的資訊影響當前計算
  - 當模型認為**輸出訊號包含重要資訊**時，模型應該要開啟**輸出閘門**，即 $y_k^{\opog}(t+1) \approx 1$
    - 在**輸入**閘門**開啟**的狀況下，**開啟輸出**閘門代表讓**現在**時間點的資訊影響當前計算
    - 在**輸入**閘門**關閉**的狀況下，**開啟輸出**閘門代表不讓**過去**時間點的資訊影響當前計算
  - 不論**輸出訊號** $h\pa{s^{\blk{k}}(t+1)}$ 的大小，只要 $y_k^{\opog}(t+1) \approx 0$，則輸出訊號**完全無法影響**接下來的所有計算，LSTM 以此設計避免 $\eqref{26} \eqref{27}$ 所遇到的困境
  - [PyTorch 實作的 LSTM][Pytorch-LSTM] 中 $h(t)$ 表達的意思是記憶細胞輸出 $y^{\blk{k}}(t)$

  根據 $\eqref{36}$，得到 $t+1$ 時間點的**記憶細胞輸出** $y^{\blk{k}}(t+1)$ 後就可以計算 $t+1$ 時間點的模型**總輸出** $y(t+1)$。

  - 注意在計算 $\eqref{36}$ 時並沒有使用閘門單元，與 $\eqref{29}$ 的計算不同
  - 注意 $y(t+1)$ 與 $y^{\opog}$ 不同
    - $y(t+1)$ 是**總輸出**，我的 $y(t+1)$ 是論文中的 $y^k(t+1)$
    - $y^{\opog}(t+1)$ 是**記憶細胞**的**輸出閘門**，我的 $y^{\opog}(t+1)$ 是論文中的 $y^{\opout_i}(t+1)$

  根據論文 A.7 式下方的描述，$t+1$ 時間點的**總輸出**只與 $t$ 時間點的**模型狀態**（**不含閘門與總輸出**）有關係，所以 $\eqref{31} \eqref{32} \eqref{33} \eqref{35}$ 的計算都只是在幫助 $t + 2$ 時間點的計算狀態**鋪陳**。

  我不確定這是否為作者的筆誤，畢竟附錄中所有分析的數學式都寫的蠻正確的，我認為這裡是筆誤的理由如下：

  - 同個實驗室後續的研究（例如 [LSTM-2002][LSTM2002]）寫的式子不同
  - 至少要傳播兩個時間點才能得到輸出，代表第 $1$ 個時間點的輸出完全無法利用到記憶細胞的知識
  - 後續的實驗架構設計中沒有將外部輸入連接到輸出，代表第 $1$ 個時間點的輸出完全依賴模型的初始狀態（常數），非常不合理

  因此我決定改用我認為是正確的版本撰寫後續的筆記，即 $t+1$ 時間點的**總輸出**與 $t$ 時間點的**外部輸入**和 $t+1$ 時間點的**計算狀態**有關。

  注意 $\eqref{32} \eqref{33}$ 沒有使用偏差項（bias term），但後續的分析會提到可以使用偏差項進行計算缺陷的修正。

  ### 參數結構

  |參數|意義|輸出維度|輸入維度|
  |-|-|-|-|
  |$\whid$|產生**隱藏單元**的全連接參數|$\dhid$|$\din + \dhid + \nblk \cdot (2 + \dblk)$|
  |$\wig$|產生**輸入閘門**的全連接參數|$\nblk$|$\din + \dhid + \nblk \cdot (2 + \dblk)$|
  |$\wog$|產生**輸出閘門**的全連接參數|$\nblk$|$\din + \dhid + \nblk \cdot (2 + \dblk)$|
  |$\wblk{k}$|產生第 $k$ 個**記憶細胞區域淨輸入**的全連接參數|$\dblk$|$\din + \dhid + \nblk \cdot (2 + \dblk)$|
  |$\wout$|產生**輸出**的全連接參數|$\dblk$|$\din + \dhid + \nblk \cdot \dblk$|

  ## 丟棄部份模型單元的梯度

  過去的論文中提出以**修改最佳化過程**避免 RNN 訓練遇到**梯度爆炸 / 消失**的問題（例如 Truncated BPTT）。

  論文 4.5 節提到**最佳化** LSTM 的方法為 **RTRL 的變種**，主要精神如下：

  - 最佳化的核心思想是確保能夠達成 **CEC** （見 $\eqref{25}$）
  - 使用的手段是要求所有梯度**反向傳播**的過程在經過**記憶細胞區域**與**隱藏單元**後便**停止**傳播
  - 停止傳播導致在完成 $t+1$ 時間點的 forward pass 後梯度可以**馬上計算完成**（real time 的精神便是來自於此）

  首先我們定義新的符號 $\aptr$，代表計算**梯度**的過程會有**部份梯度**故意被**丟棄**（設定為 $0$），並以丟棄結果**近似**真正的**全微分**。

  $$
  \pd{\opnet_i^a(t+1)}{y_j^b(t)} \aptr 0 \quad \text{where } a, b \in \set{\ophid, \opig, \opog, \blk{1}, \dots, \blk{\nblk}} \tag{37}\label{37}
  $$

  所有與**隱藏單元淨輸入** $\nethid{i}{t+1}$、**輸入閘門淨輸入** $\netig{i}{t+1}$、**輸出閘門淨輸入** $\netog{i}{t+1}$、**記憶細胞淨輸入** $\netcell{i}{k}{t+1}$ **直接相連**的 $t$ 時間點的**單元**，一律**丟棄梯度**

  - 注意論文在 A.1.2 節的開頭只提到**輸入閘門**、**輸出閘門**、**記憶細胞**要**丟棄梯度**
  - 但論文在 A.9 式描述可以將**隱藏單元**的梯度一起**丟棄**，害我白白推敲公式好幾天

  > Here it would be possible to use the full gradient without affecting constant error flow through internal states of memory cells.

  根據 $\eqref{37}$ 我們可以進一步推得

  $$
  \begin{align*}
  a & \in \set{\ophid, \opig, \opog} \\
  b & \in \set{\ophid, \opig, \opog, \blk{1}, \dots, \blk{\nblk}} \\
  \pd{y_i^a(t+1)}{y_j^b(t)} & = \pd{y_i^a(t+1)}{\opnet_i^a(t+1)} \cdot \cancelto{0}{\pd{\opnet_i^a(t+1)}{y_j^b(t)}} \aptr 0 \\
  k & \in \set{1, 2, \dots, \nblk} \\
  \pd{y_i^{\blk{k}}(t+1)}{y_j^b(t)} & = \pd{y_i^{\blk{k}}(t+1)}{y_k^{\opig}(t+1)} \cdot \cancelto{0}{\pd{y_k^{\opig}(t+1)}{y_j^b(t)}} \\
  & \quad + \pd{y_i^{\blk{k}}(t+1)}{\netcell{i}{k}{t+1}} \cdot \cancelto{0}{\pd{\netcell{i}{k}{t+1}}{y_j^b(t)}} \\
  & \quad + \pd{y_i^{\blk{k}}(t+1)}{y_k^{\opog}(t+1)} \cdot \cancelto{0}{\pd{y_k^{\opog}(t+1)}{y_j^b(t)}} \\
  & \aptr 0
  \end{align*} \tag{38}\label{38}
  $$

  由於 $y^{\opig}(t+1), y^{\opog}(t+1), \opnet^{\blk{k}}(t+1)$ 並不是**直接**透過 $w^{\ophid}$ 產生，因此 $w^{\ophid}$ 只能透過參與 $t$ 時間點**以前**的計算**間接**對 $t+1$ 時間點的計算造成影響（見 $\eqref{31}$），這也代表在 $\eqref{38}$ 作用的情況下 $w^{\ophid}$ **無法**從 $y^{\opig}(t+1), y^{\opog}(t+1), \opnet^{\blk{k}}(t+1)$ 收到任何的**梯度**：

  $$
  \begin{align*}
  a & \in \set{\opig, \opog, \blk{1}, \dots, \blk{\nblk}} \\
  b & \in \set{\ophid, \opig, \opog, \blk{1}, \dots, \blk{\nblk}} \\
  \pd{y_i^a(t+1)}{\whid_{p, q}} & = \sum_{j = \din + 1}^{\din + \dhid + \nblk \cdot (2 + \dblk)} \bigg[\cancelto{0}{\pd{y_i^a(t+1)}{y_j^b(t)}} \cdot \pd{y_j^b(t)}{\whid_{p, q}}\bigg] \aptr 0
  \end{align*} \tag{39}\label{39}
  $$

  ### 相對於總輸出所得剩餘梯度

  我們將論文的 A.8 式拆解成 $\eqref{41} \eqref{42} \eqref{43} \eqref{44}$。

  #### 總輸出參數

  令 $\delta_{a, b}$ 為 **Kronecker delta**，i.e.，

  $$
  \delta_{a, b} = \begin{dcases}
  1 & \text{if } a = b \\
  0 & \text{otherwise}
  \end{dcases} \tag{40}\label{40}
  $$

  由於**總輸出** $y(t+1)$ 不會像是 $\eqref{1} \eqref{2}$ 的方式**回饋**到模型的計算狀態中，因此**總輸出參數** $\wout$ 對**總輸出** $y(t+1)$ 計算所得的**梯度**為

  $$
  \begin{align*}
  i, p & \in \set{1, \dots, \dout} \\
  q & \in \set{1, \dots, \din + \dhid + \nblk \cdot \dblk} \\
  \pd{y_i(t+1)}{\wout_{p, q}} & = \pd{y_i(t+1)}{\netout{i}{t+1}} \cdot \pd{\netout{i}{t+1}}{\wout_{p, q}} \\
  & = \dfnetout{i}{t+1} \cdot \delta_{i, p} \cdot \begin{pmatrix}
  x(t) \\
  y^{\ophid}(t+1) \\
  y^{\blk{1}}(t+1) \\
  \vdots \\
  y^{\blk{\nblk}}(t+1)
  \end{pmatrix}_q
  \end{align*} \tag{41}\label{41}
  $$

  - $\eqref{41}$ 就是論文中 A.8 式的第一個 case
  - 由於 $p$ 可以是**任意**的輸出節點，因此在 $i \neq p$ 時 $\wout_{p, q}$ 對於 $y_i(t+1)$ 的梯度為 $0$

  #### 隱藏單元參數

  在 $\eqref{37} \eqref{38} \eqref{39}$ 的作用下，我們可以求得**隱藏單元參數** $\whid$ 在**丟棄**部份梯度後對於**總輸出** $y(t+1)$ 計算所得的**剩餘梯度**

  $$
  \begin{align*}
  D & = \din + \dhid + \nblk \cdot \dblk \\
  \tilde{x}(t+1) & = \begin{pmatrix}
  x(t) \\
  y^{\ophid}(t+1) \\
  y^{\blk{1}}(t+1) \\
  \vdots \\
  y^{\blk{\nblk}}(t+1)
  \end{pmatrix} \in \R^D \\
  i & \in \set{1, \dots, \dout} \\
  p & \in \set{1, \dots, \dhid} \\
  q & \in \set{1, \dots, D} \\
  \pd{y_i(t+1)}{\whid_{p, q}} & = \pd{y_i(t+1)}{\netout{i}{t+1}} \cdot \pd{\netout{i}{t+1}}{\whid_{p, q}} \\
  & = \dfnetout{i}{t+1} \cdot \sum_{j = 1}^D \br{\pd{\netout{i}{t+1}}{\tilde{x}_j(t+1)} \cdot \cancelto{\aptr}{\pd{\tilde{x}_j(t+1)}{\whid_{p, q}}}} \\
  & \aptr \dfnetout{i}{t+1} \cdot \wout_{i, p} \cdot \pd{y_p^{\ophid}(t+1)}{\whid_{p, q}}
  \end{align*} \tag{42}\label{42}
  $$

  $\eqref{42}$ 就是論文中 A.8 式的最後一個 case。

  #### 閘門單元參數

  同 $\eqref{42}$，我們可以計算**閘門單元參數** $\wig, \wog$ 對**總輸出** $y(t+1)$ 計算所得的**剩餘梯度**

  $$
  \begin{align*}
  D & = \din + \dhid + \nblk \cdot \dblk \\
  \tilde{x}(t+1) & = \begin{pmatrix}
  x(t) \\
  y^{\ophid}(t+1) \\
  y^{\blk{1}}(t+1) \\
  \vdots \\
  y^{\blk{\nblk}}(t+1)
  \end{pmatrix} \in \R^D \\
  i & \in \set{1, \dots, \dout} \\
  k & \in \set{1, \dots, \nblk} \\
  q & \in \set{1, \dots, \din + \dhid + \nblk \cdot (2 + \dblk)} \\
  \pd{y_i(t+1)}{\wog_{k, q}} & = \pd{y_i(t+1)}{\netout{i}{t+1}} \cdot \pd{\netout{i}{t+1}}{\wog_{k, q}} \\
  & = \dfnetout{i}{t+1} \cdot \sum_{j = 1}^D \br{\pd{\netout{i}{t+1}}{\tilde{x}_j(t+1)} \cdot \cancelto{\aptr}{\pd{\tilde{x}_j(t+1)}{\wog_{k, q}}}} \\
  & \aptr \dfnetout{i}{t+1} \cdot \sum_{j = 1}^{\dblk} \br{\wout_{i, \din + \dhid + (k - 1) \cdot \dblk + j} \cdot \pd{y_j^{\blk{k}}(t+1)}{\wog_{k, q}}} \\
  \pd{y_i(t+1)}{\wig_{k, q}} & \aptr \dfnetout{i}{t+1} \cdot \sum_{j = 1}^{\dblk} \br{\wout_{i, \din + \dhid + (k - 1) \cdot \dblk + j} \cdot \pd{y_j^{\blk{k}}(t+1)}{\wig_{k, q}}}
  \end{align*} \tag{43}\label{43}
  $$

  $\eqref{43}$ 就是論文中 A.8 式的第三個 case。

  #### 記憶細胞淨輸入參數

  **記憶細胞淨輸入參數** $\wblk{k}$ 對**總輸出** $y(t+1)$ 計算所得的**剩餘梯度**與 $\eqref{43}$ 幾乎**相同**

  $$
  \begin{align*}
  D & = \din + \dhid + \nblk \cdot \dblk \\
  \tilde{x}(t+1) & = \begin{pmatrix}
  x(t) \\
  y^{\ophid}(t+1) \\
  y^{\blk{1}}(t+1) \\
  \vdots \\
  y^{\blk{\nblk}}(t+1)
  \end{pmatrix} \in \R^D \\
  i & \in \set{1, \dots, \dout} \\
  k & \in \set{1, \dots, \nblk} \\
  p & \in \set{1, \dots, \dblk} \\
  q & \in \set{1, \dots, \din + \dhid + \nblk \cdot (2 + \dblk)} \\
  \pd{y_i(t+1)}{\wblk{k}_{p, q}} & = \pd{y_i(t+1)}{\netout{i}{t+1}} \cdot \pd{\netout{i}{t+1}}{\wblk{k}_{p, q}} \\
  & = \dfnetout{i}{t+1} \cdot \sum_{j = 1}^D \br{\pd{\netout{i}{t+1}}{\tilde{x}_j(t+1)} \cdot \cancelto{\aptr}{\pd{\tilde{x}_j(t+1)}{\wblk{k}_{p, q}}}} \\
  & \aptr \dfnetout{i}{t+1} \cdot \wout_{i, \din + \dhid + (k - 1) \cdot \dblk + p} \cdot \pd{y_p^{\blk{k}}(t+1)}{\wblk{k}_{p, q}}
  \end{align*} \tag{44}\label{44}
  $$

  $\eqref{44}$ 就是論文中 A.8 式的第二個 case。

  ### 相對於隱藏單元所得剩餘梯度

  我們將論文的 A.9 式拆解成 $\eqref{45} \eqref{46} \eqref{47}$。

  #### 隱藏單元參數

  根據 $\eqref{37} \eqref{38}$ 我們可以得到**隱藏單元參數** $\whid$ 對於**隱藏單元** $y^{\ophid}(t+1)$ 計算所得**剩餘梯度**

  $$
  \begin{align*}
  i, p & \in \set{1, \dots, \dhid} \\
  q & \in \set{1, \dots, \din + \dhid + \nblk \cdot (2 + \dblk)} \\
  \pd{y_i^{\ophid}(t+1)}{\whid_{p, q}} & = \pd{y_i^{\ophid}(t+1)}{\nethid{i}{t+1}} \cdot \cancelto{\aptr}{\pd{\nethid{i}{t+1}}{\whid_{p, q}}} \\
  & \aptr \dfnethid{i}{t+1} \cdot \delta_{i, p} \cdot \begin{pmatrix}
  x(t) \\
  y^{\ophid}(t) \\
  y^{\opig}(t) \\
  y^{\opog}(t) \\
  y^{\blk{1}}(t) \\
  \vdots \\
  y^{\blk{\nblk}}(t)
  \end{pmatrix}_q
  \end{align*} \tag{45}\label{45}
  $$

  #### 閘門單元參數

  由於**隱藏單元** $y^{\ophid}(t+1)$ 並不是**直接**透過**閘門參數** $\wig, \wog$ 產生，因此根據 $\eqref{37}$ 我們可以推得 $\wig, \wog$ 對於 $y^{\ophid}(t+1)$ **剩餘梯度**為 $0$

  $$
  \begin{align*}
  D & = \din + \dhid + \nblk \cdot (2 + \dblk) \\
  \tilde{x}(t) & = \begin{pmatrix}
  x(t) \\
  y^{\ophid}(t) \\
  y^{\opig}(t) \\
  y^{\opog}(t) \\
  y^{\blk{1}}(t) \\
  \vdots \\
  y^{\blk{\nblk}}(t)
  \end{pmatrix} \in \R^D \\
  i & \in \set{1, \dots, \dhid} \\
  p & \in \set{1, \dots, \nblk} \\
  q & \in \set{1, \dots, D} \\
  \pd{y_i^{\ophid}(t+1)}{\wog_{p, q}} & = \pd{y_i^{\ophid}(t+1)}{\nethid{i}{t+1}} \cdot \sum_{j = 1}^D \br{\cancelto{0}{\pd{\nethid{i}{t+1}}{\tilde{x}_j(t)}} \cdot \pd{\tilde{x}_j(t)}{\wog_{p, q}}} \aptr 0 \\
  \pd{y_i^{\ophid}(t+1)}{\wig_{p, q}} & \aptr 0
  \end{align*} \tag{46}\label{46}
  $$

  #### 記憶細胞淨輸入參數

  同 $\eqref{46}$，由於**隱藏單元** $y^{\ophid}(t+1)$ 並不是**直接**透過**記憶細胞淨輸入參數** $\wblk{k}$ 產生，因此根據 $\eqref{37}$ 我們可以推得 $\wblk{k}$ 對於 $y^{\ophid}(t+1)$ **剩餘梯度**為 $0$

  $$
  \begin{align*}
  D & = \din + \dhid + \nblk \cdot (2 + \dblk) \\
  \tilde{x}(t) & = \begin{pmatrix}
  x(t) \\
  y^{\ophid}(t) \\
  y^{\opig}(t) \\
  y^{\opog}(t) \\
  y^{\blk{1}}(t) \\
  \vdots \\
  y^{\blk{\nblk}}(t)
  \end{pmatrix} \in \R^D \\
  i & \in \set{1, \dots, \dhid} \\
  k & \in \set{1, \dots, \nblk} \\
  p & \in \set{1, \dots, \dblk} \\
  q & \in \set{1, \dots, D} \\
  \pd{y_i^{\ophid}(t+1)}{\wblk{k}_{p, q}} & = \pd{y_i^{\ophid}(t+1)}{\nethid{i}{t+1}} \cdot \sum_{j = 1}^D \br{\cancelto{0}{\pd{\nethid{i}{t+1}}{\tilde{x}_j(t)}} \cdot \pd{\tilde{x}_j(t)}{\wblk{k}_{p, q}}} \aptr 0
  \end{align*} \tag{47}\label{47}
  $$

  ### 相對於記憶細胞輸出所得剩餘梯度

  我們將論文的 A.13 式拆解成 $\eqref{48} \eqref{49} \eqref{50}$。

  #### 閘門單元參數

  根據 $\eqref{37}$ 我們可以推得**閘門單元參數** $\wig, \wog$ 對於**記憶細胞輸出** $y^{\blk{k}}(t+1)$ 計算所得**剩餘梯度**

  $$
  \begin{align*}
  i & \in \set{1, \dots, \dblk} \\
  k, p & \in \set{1, \dots, \nblk} \\
  q & \in \set{1, \dots, \din + \dhid + \nblk \cdot (2 + \dblk)} \\
  \pd{y_i^{\blk{k}}(t+1)}{\wog_{p, q}} & = \pd{y_i^{\blk{k}}(t+1)}{y_k^{\opog}(t+1)} \cdot \pd{y_k^{\opog}(t+1)}{\wog_{p, q}} + \pd{y_i^{\blk{k}}(t+1)}{s_i^{\blk{k}}(t+1)} \cdot \cancelto{0}{\pd{s_i^{\blk{k}}(t+1)}{\wog_{p, q}}} \\
  & \aptr h_i\pa{s_i^{\blk{k}}(t+1)} \cdot \delta_{k, p} \cdot \pd{y_k^{\opog}(t+1)}{\wog_{k, q}} \tag{48}\label{48} \\
  \pd{y_i^{\blk{k}}(t+1)}{\wig_{p, q}} & = \pd{y_i^{\blk{k}}(t+1)}{y_k^{\opog}(t+1)} \cdot \cancelto{0}{\pd{y_k^{\opog}(t+1)}{\wig_{p, q}}} + \pd{y_i^{\blk{k}}(t+1)}{s_i^{\blk{k}}(t+1)} \cdot \pd{s_i^{\blk{k}}(t+1)}{\wig_{p, q}} \\
  & \aptr y_k^{\opog}(t+1) \cdot h_i'\pa{s_i^{\blk{k}}(t+1)} \cdot \delta_{k, p} \cdot \pd{s_i^{\blk{k}}(t+1)}{\wig_{k, q}} \tag{49}\label{49}
  \end{align*}
  $$

  #### 記憶細胞淨輸入參數

  同 $\eqref{49}$，使用 $\eqref{37}$ 推得**記憶細胞淨輸入參數** $\wblk{k^{\star}}$ 對於**記憶細胞輸出** $y^{\blk{k}}(t+1)$ 計算所得**剩餘梯度**（注意 $k^{\star}$ 可以**不等於** $k$）

  $$
  \begin{align*}
  i, p & \in \set{1, \dots, \dblk} \\
  k, k^{\star} & \in \set{1, \dots, \nblk} \\
  q & \in \set{1, \dots, \din + \dhid + \nblk \cdot (2 + \dblk)} \\
  \pd{y_i^{\blk{k}}(t+1)}{\wblk{k^{\star}}_{p, q}} & = \pd{y_i^{\blk{k}}(t+1)}{y_k^{\opog}(t+1)} \cdot \cancelto{0}{\pd{y_k^{\opog}(t+1)}{\wblk{k^{\star}}_{p, q}}} + \pd{y_i^{\blk{k}}(t+1)}{s_i^{\blk{k}}(t+1)} \cdot \pd{s_i^{\blk{k}}(t+1)}{\wblk{k^{\star}}_{p, q}} \\
  & \aptr y_k^{\opog}(t+1) \cdot h_i'\pa{s_i^{\blk{k}}(t+1)} \cdot \delta_{k, k^{\star}} \cdot \delta_{i, p} \cdot \pd{s_i^{\blk{k}}(t+1)}{\wblk{k}_{i, q}}
  \end{align*} \tag{50}\label{50}
  $$

  **注意錯誤**：論文 A.13 式最後使用**加法** $\delta_{\opin_j l} + \delta_{c_j^v l}$，可能會導致梯度**乘上常數** $2$，因此應該修正成**乘法** $\delta_{\opin_j l} \cdot \delta_{c_j^v l}$

  ### 相對於閘門單元所得剩餘梯度

  我們將論文的 A.10, A.11 式拆解成 $\eqref{51} \eqref{52}$。

  #### 閘門單元參數

  根據 $\eqref{37} \eqref{38}$ 我們可以得到**閘門單元參數** $\wig, \wog$ 對於**閘門單元** $y^{\opig}(t+1), y^{\opog}(t+1)$ 計算所得**剩餘梯度**

  $$
  \begin{align*}
  D & = \din + \dhid + \nblk \cdot (2 + \dblk) \\
  \tilde{x}(t) & = \begin{pmatrix}
  x(t) \\
  y^{\ophid}(t) \\
  y^{\opig}(t) \\
  y^{\opog}(t) \\
  y^{\blk{1}}(t) \\
  \vdots \\
  y^{\blk{\nblk}}
  \end{pmatrix} \in \R^D \\
  k, p & \in \set{1, \dots, \nblk} \\
  q & \in \set{1, \dots, D} \\
  \pd{y_k^{\opig}(t+1)}{[\wig ; \wog]_{p, q}} & = \pd{y_k^{\opig}(t+1)}{\netig{k}{t+1}} \cdot \cancelto{\aptr}{\pd{\netig{k}{t+1}}{[\wig ; \wog]_{p, q}}} \\
  & \aptr \dfnetig{k}{t+1} \cdot \delta_{k, p} \cdot \tilde{x}_q(t) \\
  \pd{y_k^{\opog}(t+1)}{[\wig ; \wog]_{p, q}} & \aptr \delta_{k, p} \cdot \dfnetog{k}{t+1} \cdot \tilde{x}_q(t)
  \end{align*} \tag{51}\label{51}
  $$

  #### 記憶細胞淨輸入參數

  由於**閘門單元** $y^{\opig}(t+1), y^{\opog}(t+1)$ 並不是**直接**透過**記憶細胞淨輸入參數** $\wblk{k}$ 產生，因此根據 $\eqref{37}$ 我們可以推得 $\wblk{k}$ 對於 $y^{\opig}(t+1), y^{\opog}(t+1)$ **剩餘梯度**為 $0$

  $$
  \begin{align*}
  D & = \din + \dhid + \nblk \cdot (2 + \dblk) \\
  \tilde{x}(t) & = \begin{pmatrix}
  x(t) \\
  y^{\ophid}(t) \\
  y^{\opig}(t) \\
  y^{\opog}(t) \\
  y^{\blk{1}}(t) \\
  \vdots \\
  y^{\blk{\nblk}}
  \end{pmatrix} \in \R^D \\
  k & \in \set{1, \dots, \nblk} \\
  p & \in \set{1, \dots, \dblk} \\
  q & \in \set{1, \dots, D} \\
  \pd{y_k^{\opig}(t+1)}{\wblk{k}_{p, q}} & = \pd{y_k^{\opig}(t+1)}{\netig{k}{t+1}} \cdot \sum_{j = 1}^D \br{\cancelto{0}{\pd{\netig{k}{t+1}}{\tilde{x}_j(t)}} \cdot \pd{\tilde{x}_j(t)}{\wblk{k}_{p, q}}} \aptr 0 \\
  \pd{y_k^{\opog}(t+1)}{\wblk{k}_{p, q}} & \aptr 0
  \end{align*} \tag{52}\label{52}
  $$

  ### 相對於記憶細胞內部狀態所得剩餘梯度

  我們將論文的 A.12 式拆解成 $\eqref{53} \eqref{54} \eqref{55}$。

  #### 閘門單元參數

  將 $\eqref{37}$ 結合 $\eqref{51}$ 我們可以推得**閘門單元參數** $\wig, \wog$ 對於**記憶細胞內部狀態** $s^{\blk{k}}(t+1)$ 計算所得**剩餘梯度**

  $$
  \begin{align*}
  D & = \din + \dhid + \nblk \cdot (2 + \dblk) \\
  \tilde{x}(t) & = \begin{pmatrix}
  x(t) \\
  y^{\ophid}(t) \\
  y^{\opig}(t) \\
  y^{\opog}(t) \\
  y^{\blk{1}}(t) \\
  \vdots \\
  y^{\blk{\nblk}}(t)
  \end{pmatrix} \in \R^D \\
  i & \in \set{1, \dots, \dblk} \\
  k, p & \in \set{1, \dots, \nblk} \\
  q & \in \set{1, \dots, D} \\
  \pd{s_i^{\blk{k}}(t+1)}{\wog_{p, q}} & = \pd{s_i^{\blk{k}}(t+1)}{s_i^{\blk{k}}(t)} \cdot \cancelto{0}{\pd{s_i^{\blk{k}}(t)}{\wog_{p, q}}} + \pd{s_i^{\blk{k}}(t+1)}{y_k^{\opig}(t+1)} \cdot \cancelto{0}{\pd{y_k^{\opig}(t+1)}{\wog_{p, q}}} \\
  & \quad + \pd{s_i^{\blk{k}}(t+1)}{\netcell{i}{k}{t+1}} \cdot \cancelto{0}{\pd{\netcell{i}{k}{t+1}}{\wog_{p, q}}} \\
  & \aptr 0 \tag{53}\label{53} \\
  \pd{s_i^{\blk{k}}(t+1)}{\wig_{p, q}} & = \pd{s_i^{\blk{k}}(t+1)}{s_i^{\blk{k}}(t)} \cdot \pd{s_i^{\blk{k}}(t)}{\wig_{p, q}} + \pd{s_i^{\blk{k}}(t+1)}{y_k^{\opig}(t+1)} \cdot \pd{y_k^{\opig}(t+1)}{\wig_{p, q}} \\
  & \quad + \pd{s_i^{\blk{k}}(t+1)}{\netcell{i}{k}{t+1}} \cdot \cancelto{0}{\pd{\netcell{i}{k}{t+1}}{\wig_{p, q}}} \\
  & \aptr 1 \cdot \delta_{k, p} \cdot \pd{s_i^{\blk{k}}(t)}{\wig_{k, q}} + g_i\pa{\netcell{i}{k}{t+1}} \cdot \delta_{k, p} \cdot \cancelto{\aptr}{\pd{y_k^{\opig}(t+1)}{\wig_{k, q}}} \\
  & \aptr \delta_{k, p} \cdot \br{\pd{s_i^{\blk{k}}(t)}{\wig_{k, q}} + g_i\pa{\netcell{i}{k}{t+1}} \cdot \dfnetig{k}{t+1} \cdot \tilde{x}_q(t)} \tag{54}\label{54}
  \end{align*}
  $$

  #### 記憶細胞淨輸入參數

  使用 $\eqref{37}$ 推得**記憶細胞淨輸入參數** $\wblk{k^{\star}}$ 對於**記憶細胞內部狀態** $s^{\blk{k}}(t+1)$ 計算所得**剩餘梯度**（注意 $k^{\star}$ 可以**不等於** $k$）

  $$
  \begin{align*}
  D & = \din + \dhid + \nblk \cdot (2 + \dblk) \\
  \tilde{x}(t) & = \begin{pmatrix}
  x(t) \\
  y^{\ophid}(t) \\
  y^{\opig}(t) \\
  y^{\opog}(t) \\
  y^{\blk{1}}(t) \\
  \vdots \\
  y^{\blk{\nblk}}(t)
  \end{pmatrix} \in \R^D \\
  i, p & \in \set{1, \dots, \dblk} \\
  k, k^{\star} & \in \set{1, \dots, \nblk} \\
  q & \in \set{1, \dots, D} \\
  \pd{s_i^{\blk{k}}(t+1)}{\wblk{k^{\star}}_{p, q}} & = \pd{s_i^{\blk{k}}(t+1)}{s_i^{\blk{k}}(t)} \cdot \pd{s_i^{\blk{k}}(t)}{\wblk{k^{\star}}_{p, q}} + \pd{s_i^{\blk{k}}(t+1)}{y_k^{\opig}(t+1)} \cdot \cancelto{0}{\pd{y_k^{\opig}(t+1)}{\wblk{k^{\star}}_{p, q}}} \\
  & \quad + \pd{s_i^{\blk{k}}(t+1)}{\netcell{i}{k}{t+1}} \cdot \pd{\netcell{i}{k}{t+1}}{\wblk{k^{\star}}_{p, q}} \\
  & \aptr \delta_{k, k^{\star}} \cdot \delta_{i, p} \cdot 1 \cdot \pd{s_i^{\blk{k}}(t)}{\wblk{k}_{i, q}} \\
  & \quad + \delta_{k, k^{\star}} \cdot \delta_{i, p} \cdot y_k^{\opig}(t+1) \cdot g_i'\pa{\netcell{i}{k}{t+1}} \cdot \tilde{x}_q(t) \\
  & = \delta_{k, k^{\star}} \cdot \delta_{i, p} \cdot \br{\pd{s_i^{\blk{k}}(t)}{\wblk{k}_{i, q}} + y_k^{\opig}(t+1) \cdot g_i'\pa{\netcell{i}{k}{t+1}} \cdot \tilde{x}_q(t)}
  \end{align*} \tag{55}\label{55}
  $$

  **注意錯誤**：論文 A.12 式最後使用**加法** $\delta_{\opin_j l} + \delta_{c_j^v l}$，可能會導致梯度**乘上常數** $2$，因此應該修正成**乘法** $\delta_{\opin_j l} \cdot \delta_{c_j^v l}$

  ## 更新模型參數

  ### 總輸出參數

  從 $\eqref{4}$ 我們可以觀察出以下結論

  $$
  \begin{align*}
  \tilde{x}(t+1) & = \begin{pmatrix}
  x(t) \\
  y^{\ophid}(t+1) \\
  y^{\blk{1}}(t+1) \\
  \vdots \\
  y^{\blk{\nblk}}(t+1)
  \end{pmatrix} \\
  i & \in \set{1, \dots, \dout} \\
  j & \in \set{1, \dots, \din + \dhid + \nblk \cdot \dblk} \\
  \pd{\tloss(t+1)}{\wout_{i, j}} & = \pd{\tloss(t+1)}{\loss{i}{t+1}} \cdot \pd{\loss{i}{t+1}}{y_i(t+1)} \cdot \pd{y_i(t+1)}{\wout_{i, j}} \\
  & = \big(y_i(t+1) - \hat{y}_i(t+1)\big) \cdot \pd{y_i(t+1)}{\wout_{i, j}} \\
  & = \big(y_i(t+1) - \hat{y}_i(t+1)\big) \cdot \dfnetout{i}{t+1} \cdot \tilde{x}_j(t+1)
  \end{align*} \tag{56}\label{56}
  $$

  ### 隱藏單元參數

  從 $\eqref{4} \eqref{39} \eqref{42} \eqref{45}$ 我們可以觀察出以下結論

  $$
  \begin{align*}
  & p \in \set{1, \dots, \dhid} \\
  & q \in \set{1, \dots, \din + \dhid + \nblk \cdot (2 + \dblk)} \\
  & \pd{\tloss(t+1)}{\whid_{p, q}} = \sum_{i = 1}^{\dout} \br{\pd{\tloss(t+1)}{\loss{i}{t+1}} \cdot \pd{\loss{i}{t+1}}{y_i(t+1)} \cdot \pd{y_i(t+1)}{\whid_{p, q}}} \\
  & \aptr \sum_{i = 1}^{\dout} \br{\pa{y_i(t+1) - \hat{y}_i(t+1)} \cdot \dfnetout{i}{t+1} \cdot \wout_{i, p} \cdot \pd{y_p^{\ophid}(t+1)}{\whid_{p, q}}} \\
  & = \sum_{i = 1}^{\dout} \br{\pa{y_i(t+1) - \hat{y}_i(t+1)} \cdot \dfnetout{i}{t+1} \cdot \wout_{i, p}} \cdot \pd{y_p^{\ophid}(t+1)}{\whid_{p, q}} \\
  & \aptr \sum_{i = 1}^{\dout} \br{\pa{y_i(t+1) - \hat{y}_i(t+1)} \cdot \dfnetout{i}{t+1} \cdot \wout_{i, p}} \cdot \\
  & \quad \quad \dfnethid{p}{t+1} \cdot \begin{pmatrix}
  x(t) \\
  y^{\ophid}(t) \\
  y^{\blk{1}}(t) \\
  \vdots \\
  y^{\blk{\nblk}}(t)
  \end{pmatrix}_j
  \end{align*} \tag{57}\label{57}
  $$

  ### 輸出閘門單元參數

  從 $\eqref{4} \eqref{43} \eqref{48} \eqref{51} \eqref{53}$ 我們可以觀察出以下結論

  $$
  \begin{align*}
  k & \in \set{1, \dots, \nblk} \\
  q & \in \set{1, \dots, \din + \dhid + \nblk \cdot (2 + \dblk)} \\
  \pd{\tloss(t+1)}{\wog_{k, q}} & = \sum_{i = 1}^{\dout} \br{\pd{\tloss(t+1)}{\loss{i}{t+1}} \cdot \pd{\loss{i}{t+1}}{y_i(t+1)} \cdot \pd{y_i(t+1)}{\wog_{k, q}}} \\
  & \aptr \sum_{i = 1}^{\dout} \Bigg[\big(y_i(t+1) - \hat{y}_i(t+1)\big) \cdot \dfnetout{i}{t+1} \cdot \\
  & \quad \quad \sum_{j = 1}^{\dblk} \pa{\wout_{i, \din + \dhid + (k - 1) \cdot \dblk + j} \cdot \pd{y_j^{\blk{k}}(t+1)}{\wog_{k, q}}}\Bigg] \\
  & \aptr \sum_{i = 1}^{\dout} \Bigg[\big(y_i(t+1) - \hat{y}_i(t+1)\big) \cdot \dfnetout{i}{t+1} \cdot \\
  & \quad \quad \sum_{j = 1}^{\dblk} \pa{\wout_{i, \din + \dhid + (k - 1) \cdot \dblk + j} \cdot h_j\pa{s_j^{\blk{k}}(t+1)} \cdot \pd{y_k^{\opog}(t+1)}{\wog_{k, q}}}\Bigg] \\
  & = \Bigg[\sum_{i = 1}^{\dout} \big(y_i(t+1) - \hat{y}_i(t+1)\big) \cdot \dfnetout{i}{t+1} \cdot \\
  & \quad \quad \pa{\sum_{j = 1}^{\dblk} \wout_{i, \din + \dhid + (k - 1) \cdot \dblk + j} \cdot h_j\pa{s_j^{\blk{k}}(t+1)}}\Bigg] \cdot \pd{y_k^{\opog}(t+1)}{\wog_{k, q}} \\
  & \aptr \Bigg[\sum_{i = 1}^{\dout} \big(y_i(t+1) - \hat{y}_i(t+1)\big) \cdot \dfnetout{i}{t+1} \cdot \\
  & \quad \quad \pa{\sum_{j = 1}^{\dblk} \wout_{i, \din + \dhid + (k - 1) \cdot \dblk + j} \cdot h_j\pa{s_j^{\blk{k}}(t+1)}}\Bigg] \cdot \\
  & \quad \quad \dfnetog{k}{t+1} \cdot \begin{pmatrix}
  x(t) \\
  y^{\ophid}(t) \\
  y^{\opig}(t) \\
  y^{\opog}(t) \\
  y^{\blk{1}}(t) \\
  \vdots \\
  y^{\blk{\nblk}}(t)
  \end{pmatrix}_q
  \end{align*} \tag{58}\label{58}
  $$

  ### 輸入閘門單元參數

  從 $\eqref{4} \eqref{43} \eqref{49} \eqref{51} \eqref{54}$ 我們可以觀察出以下結論

  $$
  \begin{align*}
  & \tilde{x}(t) = \begin{pmatrix}
  x(t) \\
  y^{\ophid}(t) \\
  y^{\opig}(t) \\
  y^{\opog}(t) \\
  y^{\blk{1}}(t) \\
  \vdots \\
  y^{\blk{\nblk}}(t)
  \end{pmatrix} \\
  & k \in \set{1, \dots, \nblk} \\
  & q \in \set{1, \dots, \din + \dhid + \nblk \cdot (2 + \dblk)} \\
  & \pd{\tloss(t+1)}{\wig_{k, q}} = \sum_{i = 1}^{\dout} \br{\pd{\tloss(t+1)}{\loss{i}{t+1}} \cdot \pd{\loss{i}{t+1}}{y_i(t+1)} \cdot \pd{y_i(t+1)}{\wig_{k, q}}} \\
  & \aptr \sum_{i = 1}^{\dout} \Bigg[\big(y_i(t+1) - \hat{y}_i(t+1)\big) \cdot \dfnetout{i}{t+1} \cdot \\
  & \quad \quad \sum_{j = 1}^{\dblk} \pa{\wout_{i, \din + \dhid + (k - 1) \cdot \dblk + j} \cdot \pd{y_j^{\blk{k}}(t+1)}{\wig_{k, q}}}\Bigg] \\
  & \aptr \sum_{i = 1}^{\dout} \Bigg[\big(y_i(t+1) - \hat{y}_i(t+1)\big) \cdot \dfnetout{i}{t+1} \cdot \\
  & \quad \quad \sum_{j = 1}^{\dblk} \pa{\wout_{i, \din + \dhid + (k - 1) \cdot \dblk + j} \cdot y_k^{\opog}(t+1) \cdot h_j'\pa{s_j^{\blk{k}}(t+1)} \cdot \pd{s_j^{\blk{k}}(t+1)}{\wig_{k, q}}}\Bigg] \\
  & = \Bigg(\sum_{i = 1}^{\dout} \Bigg[\big(y_i(t+1) - \hat{y}_i(t+1)\big) \cdot \dfnetout{i}{t+1} \cdot \\
  & \quad \quad \sum_{j = 1}^{\dblk} \pa{\wout_{i, \din + \dhid + (k - 1) \cdot \dblk + j} \cdot h_j'\pa{s_j^{\blk{k}}(t+1)} \cdot \pd{s_j^{\blk{k}}(t+1)}{\wig_{k, q}}}\Bigg]\Bigg) \cdot y_k^{\opog}(t+1) \\
  & \aptr \Bigg(\sum_{i = 1}^{\dout} \Bigg[\big(y_i(t+1) - \hat{y}_i(t+1)\big) \cdot \dfnetout{i}{t+1} \cdot \\
  & \quad \quad \sum_{j = 1}^{\dblk} \bigg(\wout_{i, \din + \dhid + (k - 1) \cdot \dblk + j} \cdot h_j'\pa{s_j^{\blk{k}}(t+1)} \cdot \bigg[\pd{s_j^{\blk{k}}(t)}{\wig_{k, q}} + \\
  & \quad \quad g_j\pa{\netcell{j}{k}{t+1}} \cdot \dfnetig{k}{t+1} \cdot \tilde{x}_q(t)\bigg]\bigg)\Bigg]\Bigg) \cdot y_k^{\opog}(t+1)
  \end{align*} \tag{59}\label{59}
  $$

  ### 記憶細胞淨輸入參數

  從 $\eqref{4} \eqref{44} \eqref{47} \eqref{50} \eqref{52} \eqref{55}$ 我們可以觀察出以下結論

  $$
  \begin{align*}
  & \tilde{x}(t) = \begin{pmatrix}
  x(t) \\
  y^{\ophid}(t) \\
  y^{\opig}(t) \\
  y^{\opog}(t) \\
  y^{\blk{1}}(t) \\
  \vdots \\
  y^{\blk{\nblk}}(t)
  \end{pmatrix} \\
  & k \in \set{1, \dots, \nblk} \\
  & p \in \set{1, \dots, \dblk} \\
  & q \in \set{1, \dots, \din + \dhid + \nblk \cdot (2 + \dblk)} \\
  & \pd{\tloss(t+1)}{\wblk{k}_{p, q}} = \sum_{i = 1}^{\dout} \br{\pd{\tloss(t+1)}{\loss{i}{t+1}} \cdot \pd{\loss{i}{t+1}}{y_i(t+1)} \cdot \pd{y_i(t+1)}{\wblk{k}_{p, q}}} \\
  & \aptr \sum_{i = 1}^{\dout} \bigg[\big(y_i(t+1) - \hat{y}_i(t+1)\big) \cdot \dfnetout{i}{t+1} \cdot \\
  & \quad \quad \wout_{i, \din + \dhid + (k - 1) \cdot \dblk + p} \cdot \pd{y^{\blk{k}}_p(t+1)}{\wblk{k}_{p, q}}\bigg] \\
  & = \br{\sum_{i = 1}^{\dout} \big(y_i(t+1) - \hat{y}_i(t+1)\big) \cdot \dfnetout{i}{t+1} \cdot \wout_{i, \din + \dhid + (k - 1) \cdot \dblk + p}} \cdot \\
  & \quad \quad \pd{y^{\blk{k}}_p(t+1)}{\wblk{k}_{p, q}} \\
  & \aptr \br{\sum_{i = 1}^{\dout} \big(y_i(t+1) - \hat{y}_i(t+1)\big) \cdot \dfnetout{i}{t+1} \cdot \wout_{i, \din + \dhid + (k - 1) \cdot \dblk + p}} \cdot \\
  & \quad \quad y_k^{\opog}(t+1) \cdot h_p'\pa{s_p^{\blk{k}}(t+1)} \cdot \pd{s_p^{\blk{k}}(t+1)}{\wblk{k}_{p, q}}\Bigg] \\
  & \aptr \br{\sum_{i = 1}^{\dout} \big(y_i(t+1) - \hat{y}_i(t+1)\big) \cdot \dfnetout{i}{t+1} \cdot \wout_{i, \din + \dhid + (k - 1) \cdot \dblk + p}} \cdot \\
  & \quad \quad y_k^{\opog}(t+1) \cdot h_p'\pa{s_p^{\blk{k}}(t+1)} \cdot \Bigg[\pd{s_p^{\blk{k}}(t)}{\wblk{k}_{p, q}} + \\
  & \quad \quad y_k^{\opig}(t+1) \cdot g_p'\pa{\netcell{p}{k}{t+1}} \cdot \tilde{x}_q(t)\Bigg]
  \end{align*} \tag{60}\label{60}
  $$

  ## 架構分析

  ### 時間複雜度

  假設 $t+1$ 時間點的 **forward pass** 已經執行完成，則**更新** $t+1$ 時間點**所有參數**的**時間複雜度**為

  $$
  O(\dim(\whid) + \dim(\wog) + \dim(\wig) + \nblk \cdot \dim(\wblk{1}) + \dim(\wout)) \tag{61}\label{61}
  $$

  - $\eqref{61}$ 就是論文中的 A.27 式
  - 在 $t+1$ 時間點**參數更新**需要考慮 $t$ 時間點的**計算狀態**，請見 $\eqref{57} \eqref{58} \eqref{59} \eqref{60}$
  - 沒有如同 $\eqref{14}$ 的**連乘積**項，因此不會有**梯度消失**問題
  - 整個計算過程需要額外紀錄的**梯度**項次**只有** $\eqref{59} \eqref{60}$ 中的 $\pd{s_j^{\blk{k}}(t)}{\wig_{k, q}}, \pd{s_p^{\blk{k}}(t)}{\wblk{k}_{p, q}}$
    - 紀錄讓 LSTM 可以隨著 **forward pass** 的過程**即時更新**
    - **不需要**等到 $T$ 時間點的計算結束，因此不是採用 **BPTT** 的演算法
    - **即時更新**（意思是 $t+1$ 時間點的 forward pass 完成後便可計算 $t+1$ 時間點的誤差梯度）是 **RTRL** 的主要精神

  總共會執行 $t+1$ 個 **forward pass**，因此**更新所有參數**所需的**總時間複雜度**為

  $$
  O\big(T \cdot \big[\dim(\whid) + \dim(\wog) + \dim(\wig) + \nblk \cdot \dim(\wblk{1}) + \dim(\wout)\big]\big) \tag{62}\label{62}
  $$

  ### 空間複雜度

  我們也可以推得在 $t+1$ 時間點**更新所有參數**所需的**空間複雜度**

  $$
  O(\dim(\whid) + \dim(\wog) + \dim(\wig) + \nblk \cdot \dim(\wblk{1}) + \dim(\wout)) \tag{63}\label{63}
  $$

  總共會執行 $T$ 個 **forward pass**，但**更新**所需的**總空間複雜度**仍然同 $\eqref{63}$

  - 依照**時間順序**計算梯度，計算完 $t+1$ 時間點的梯度時 $t$ 的資訊便可丟棄
  - 這就是 **RTRL** 的最大優點

  ### 達成梯度常數

  根據 $\eqref{37} \eqref{38}$ 我們可以推得

  $$
  \begin{align*}
  i & \in \set{1, \dots, \dblk} \\
  k & \in \set{1, \dots, \nblk} \\
  \pd{s_i^{\blk{k}}(t+1)}{s_i^{\blk{k}}(t)} & = \pd{s_i^{\blk{k}}(t)}{s_i^{\blk{k}}(t)} + \cancelto{0}{\pd{y_k^{\opig}(t+1)}{s_i^{\blk{k}}(t)}} \cdot g_i\pa{\netcell{i}{k}{t+1}} + \\
  & \quad y_k^{\opig}(t+1) \cdot \cancelto{0}{\pd{g_i\pa{\netcell{i}{k}{t+1}}}{s_i^{\blk{k}}(t)}} \\
  & \aptr 1
  \end{align*} \tag{64}\label{64}
  $$

  由於**丟棄部份梯度**的作用，$s^{\blk{k}}$ 的**梯度**是模型中**唯一**進行**遞迴**（跨過多個時間點）的計算節點。
  透過丟棄部份梯度我們從 $\eqref{64}$ 可以看出 LSTM 達成 $\eqref{23}$ 所設想的情況。

  ### 內部狀態偏差行為

  觀察 $\eqref{54} \eqref{59}$，當 $h$ 是 sigmoid 函數時，我們可以發現

  - 如果 $s^{\blk{k}}(t+1)$ 是一個**非常大**的**正數**，則 $h_j'\pa{s_j^{\blk{k}}(t+1)}$ 會變得**非常小**
  - 如果 $s^{\blk{k}}(t+1)$ 是一個**非常小**的**負數**，則 $h_j'\pa{s_j^{\blk{k}}(t+1)}$ 也會變得**非常小**
  - 在 $s^{\blk{k}}(t+1)$ 極正或極負的情況下，**輸入閘門參數** $\wig$ 的**梯度**會**消失**
  - 此現象稱為**內部狀態偏差行為**（**Internal State Drift**）
  - 同樣的現象也會發生在**記憶細胞淨輸入參數** $\wblk{1}, \dots \wblk{\nblk}$ 身上，請見 $\eqref{60}$
  - 此分析就是論文的 A.39 式改寫而來

  ### 解決 Internal State Drift

  作者提出可以在 $\opnet^{\opig}$ 加上偏差項，並在**訓練初期**將偏差項弄成很小的**負數**，邏輯如下

  $$
  \begin{align*}
  & b^{\opig} \ll 0 \\
  \implies & \opnet^{\opig}(1) \ll 0 \\
  \implies & y^{\opig}(1) \approx 0 \\
  \implies & s^{\wblk{k}}(1) = s^{\wblk{k}}(0) + y^{\opig}(1) \odot g\big(\opnet^{\wblk{k}}(1)\big) \\
  & = y^{\opig}(1) \odot g\big(\opnet^{\wblk{k}}(1)\big) \approx 0 \\
  \implies & \begin{dcases}
  s^{\wblk{k}}(t+1) \not\ll 0 \\
  s^{\wblk{k}}(t+1) \not\gg 0
  \end{dcases} \quad \forall t = 0, \dots, T - 1
  \end{align*} \tag{65}\label{65}
  $$

  根據 $\eqref{65}$ 我們就不會得到 $s^{\blk{k}}(t)$ 極正或極負的情況，也就不會出現 Internal State Drift。

  雖然這種作法是種**模型偏差**（**Model Bias**）而且會導致 $y^{\opig}(\star)$ 與 $\dfnetig{k}{\star}$ **變小**，但作者認為這些影響比起 Internal State Drift 一點都不重要。

  ### 輸出閘門初始化

  論文 4.7 節表示，在訓練的初期模型有可能濫用**記憶細胞的初始值**作為計算的常數項（細節請見 $\eqref{41}$），導致模型在訓練的過程中學會完全**不紀錄資訊**。

  因此可以將**輸出閘門**加上偏差項，並初始化成**較小的負數**（理由類似於 $\eqref{65}$），讓記憶細胞在**計算初期**輸出值為 $0$，迫使模型只在**需要**時指派記憶細胞進行**記憶**。

  如果有多個記憶細胞，則可以給予**不同的負數**，讓模型能夠按照需要**依大小順序**取得記憶細胞（**愈大的負數**愈容易被取得）。

  ### 輸出閘門的優點

  在訓練的初期**誤差**通常比較**大**，導致**梯度**跟著變**大**，使得模型在訓練初期的參數劇烈振盪。

  由於**輸出閘門**所使用的**啟發函數** $f^{\opog}$ 是 sigmoid，數值範圍是 $(0, 1)$，我們可以發現 $\eqref{59} \eqref{60}$ 的梯度乘積包含 $y^{\opog}$，可以避免**過大誤差**造成的**梯度變大**。

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
    - 包含 $f^{\opout}, f^{\ophid}, f^{\opig}, f^{\opog}$
  - 資料隨機性
    - 資料生成為隨機
    - 訓練順序為隨機
  - 在每個時間點 $t$ 的計算順序為
    1. 將外部輸入 $x(t)$ 丟入模型
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
  - 模型會在每個時間點 $t$ 收到一個字元，並輸出下一個時間點 $t+1$ 會收到的字元
    - 輸入與輸出都是 one-hot vector，維度為 $7$，每個維度各自代表 BEPSTVX 中的一個字元，取數值最大的維度作為預測結果
    - 模型必須根據 $0, 1, \dots t - 1, t$ 時間點收到的字元預測 $t+1$ 時間點輸出的字元
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
  |$(\nblk, \dblk)$|$\set{(3, 2), (4, 1)}$|至少有 $3$ 個記憶細胞|
  |$\dout$|$7$||
  |$\dim(\whid)$|$0$|沒有隱藏單元|
  |$\dim(\wblk{k})$|$\dblk \times [\din + \nblk \cdot (2 + \dblk)]$|全連接隱藏層|
  |$\dim(\wig)$|$\nblk \times [\din + \nblk \cdot (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\wog)$|$\nblk \times [\din + \nblk \cdot (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\wout)$|$\dout \times [\nblk \cdot \dblk]$|外部輸入沒有直接連接到總輸出|
  |參數初始化範圍|$[-0.2, 0.2]$||
  |輸出閘門偏差項初始化範圍|$\set{-1, -2, -3, -4}$|由大到小依序初始化不同記憶細胞對應輸出閘門偏差項|
  |Learning rate|$\set{0.1, 0.2, 0.5}$||
  |總參數量|$\set{264, 276}$||

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

  定義 $p + 1$ 種不同的字元，標記為 $V = \set{\alpha, \beta, c_1, c_2, \dots, c_{p - 1}}$。

  定義 $2$ 種長度為 $p + 1$ 不同的序列 $\opseq_1, \opseq_2$，分別為

  $$
  \begin{align*}
  \opseq_1 & = \alpha, c_1, c_2, \dots, c_{p - 2}, c_{p - 1}, \alpha \\
  \opseq_2 & = \beta, c_1, c_2, \dots, c_{p - 2}, c_{p - 1}, \beta
  \end{align*}
  $$

  令 $\opseq_{\star} \in \set{\opseq_1, \opseq_2}$，令 $\opseq_{\star}$ 第 $t$ 個時間點的字元為 $\opseq_{\star}(t) \in V$。

  當給予模型 $\opseq_{\star}(t)$ 時，模型要能夠根據 $\opseq_{\star}(0), \opseq_{\star}(1), \dots \opseq_{\star}(t - 1), \opseq_{\star}(t)$ 預測 $\opseq_{\star}(t+1)$。

  - 模型需要記住 $c_1, \dots, c_{p - 1}$ 的順序
  - 模型也需要記住開頭的 $\opseq_{\star}(0)$ 是 $\alpha$ 還是 $\beta$，並利用 $\opseq_{\star}(0)$ 的資訊預測 $\opseq_{\star}(p + 1)$
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
  |$\dim(\whid)$|$0$|沒有隱藏單元|
  |$\dim(\wblk{k})$|$\dblk \times [\din + (1 + \nblk) \cdot \dblk]$|全連接隱藏層|
  |$\dim(\wig)$|$\nblk \times [\din + (1 + \nblk) \cdot \dblk]$|全連接隱藏層|
  |$\dim(\wog)$|$0$|沒有輸出閘門|
  |$\dim(\wout)$|$0$|總輸出就是記憶細胞的輸出|
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
  |$\dim(\whid)$|$0$|沒有隱藏單元|
  |$\dim(\wblk{k})$|$\dblk \times [\din + \nblk \cdot (2 + \dblk)]$|全連接隱藏層|
  |$\dim(\wig)$|$\nblk \times [\din + \nblk \cdot (2 + \dblk)]$|全連接隱藏層|
  |$\dim(\wog)$|$\nblk \times [\din + \nblk \cdot (2 + \dblk)]$|全連接隱藏層|
  |$\dim(\wout)$|$\dout \times [\nblk \cdot \dblk]$|外部輸入沒有直接連接到總輸出|
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

  給定一個常數 $T$，並從 $[T, T + \frac{T}{10}]$ 的區間中隨機挑選一個整數作為序列 $\opseq$ 的長度 $L$。

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
  |$\dim(\whid)$|$0$|沒有隱藏單元|
  |$\dim(\wblk{k})$|$\dblk \times [\din + \nblk \cdot (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\wig)$|$\nblk \times [\din + \nblk \cdot (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\wog)$|$\nblk \times [\din + \nblk \cdot (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\wout)$|$\dout \times [\nblk \cdot \dblk]$|外部輸入沒有直接連接到總輸出|
  |參數初始化範圍|$[-0.1, 0.1]$||
  |輸入閘門偏差項初始化範圍|$\set{-1, -3, -5}$|由大到小依序初始化不同記憶細胞對應輸入閘門偏差項|
  |輸出閘門偏差項初始化範圍|$\set{-2, -4, -6}$|由大到小依序初始化不同記憶細胞對應輸出閘門偏差項|
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
  \opseq(t) \in [-1, 1] \times \set{-1, 0, 1} \quad \forall t = 0, \dots, T
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
  \hat{y}(L + 1) = 0.5 + \frac{1}{4} \sum_{t = 0}^{L} \br{\mathbb{1}(\opseq_1(t) = 1) \cdot \opseq_2(t)}
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
  |$\dim(\whid)$|$0$|沒有隱藏單元|
  |$\dim(\wblk{k})$|$\dblk \times [\din + \nblk \cdot (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\wig)$|$\nblk \times [\din + \nblk \cdot (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\wog)$|$\nblk \times [\din + \nblk \cdot (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\wout)$|$\dout \times [\nblk \cdot \dblk + 1]$|外部輸入沒有直接連接到總輸出，有額外使用偏差項|
  |參數初始化範圍|$[-0.1, 0.1]$||
  |輸入閘門偏差項初始化範圍|$\set{-3, -6}$|由大到小依序初始化不同記憶細胞對應輸入閘門偏差項|
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

  從 LSTM 的架構上來看實驗 4 的加法任務可以透過 $\eqref{39}$ 輕鬆完成，因此實驗 5 的目標是確認模型是否能夠從加法上延伸出乘法的概念，確保實驗 4 並不只是單純因模型架構而解決。

  概念與實驗 4 的任務幾乎相同，只做以下修改：

  - 每個時間點的元素第一個數值改為 $[0, 1]$ 之間的隨機值
  - $L + 1$ 時間點的輸出目標改成

  $$
  \hat{y}(L + 1) = 0.5 + \frac{1}{4} \prod_{t = 0}^{L} \br{\mathbb{1}(\opseq_1(t) = 1) \cdot \opseq_2(t)}
  $$

  - 當連續 $2000$ 筆訓練資料中，不超過 $n_{\opseq}$ 筆資料的絕對誤差小於 $0.04$ 就停止訓練
  - $n_{\opseq} \in \set{13, 140}$
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

  給予一個序列 $\opseq$，其長度 $L$ 會落在 $[100, 110]$ 之間，序列中的所有元素都來自於集合 $V = \set{a, b, c, d, B, E, X, Y}$。

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
  |$\dim(\whid)$|$0$|沒有隱藏單元|
  |$\dim(\wblk{k})$|$\dblk \times [\din + \nblk \cdot (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\wig)$|$\nblk \times [\din + \nblk \cdot (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\wog)$|$\nblk \times [\din + \nblk \cdot (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\wout)$|$\dout \times [\nblk \cdot \dblk + 1]$|外部輸入沒有直接連接到總輸出，有額外使用偏差項|
  |參數初始化範圍|$[-0.1, 0.1]$||
  |輸入閘門偏差項初始化範圍|$\set{-2, -4}$|由大到小依序初始化不同記憶細胞對應輸入閘門偏差項|
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
  |$\dim(\whid)$|$0$|沒有隱藏單元|
  |$\dim(\wblk{k})$|$\dblk \times [\din + \nblk \cdot (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\wig)$|$\nblk \times [\din + \nblk \cdot (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\wog)$|$\nblk \times [\din + \nblk \cdot (2 + \dblk) + 1]$|全連接隱藏層，有額外使用偏差項|
  |$\dim(\wout)$|$\dout \times [\nblk \cdot \dblk + 1]$|外部輸入沒有直接連接到總輸出，有額外使用偏差項|
  |參數初始化範圍|$[-0.1, 0.1]$||
  |輸入閘門偏差項初始化範圍|$\set{-2, -4, -6}$|由大到小依序初始化不同記憶細胞對應輸入閘門偏差項|
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

