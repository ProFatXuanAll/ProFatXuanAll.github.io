=====
關於
=====

我
=======

台灣人。
曾經在國立成功大學\ `智慧型知識管理實驗室`_\ 攻讀博士班，主要研究自然語言處理（Natural Language Processing）與深度學習（Deep Learning）演算法。
後來因為一些個人因素決定不繼續唸博士班，現在是個全職軟體工程師，職稱是 AI Engineer。

基於研究需求與個人興趣，我花了不少時間研讀數學，甚至我的博士班後期都在學習數學 XD。
你可以在\ `我的 GitHub`_\ 上找到我的閱讀筆記，內容包含：

- 實變數分析（Real Analysis），參考書籍為 Terence Tao 撰寫的 Analysis I and II，筆記連結見\ `實變證明與筆記`_\。
- 線性代數（Linear Algebra），參考書籍為 Lawrence E. Spence and Stephen H. Friedberg 撰寫的 Linear Algebra，筆記連結見\ `線代證明與筆記`_\。

這個網站
===============

這個網站是用來搬移過去\ `我的 HackMD`_ 中所有的閱讀筆記。
由於 `HackMD`_ 在紀錄大量數學式時 `MathJax v2 <MathJax_>`_ 反應速度較慢（甚至會導致網頁直接無法反應），加上我撰寫筆記的數學內容含量較大，因此我無法繼續在 `HackMD`_ 上面繼續撰寫筆記，才決定自己建立個人筆記網站。

考量到 `HackMD`_ 能夠達成以下功能：

1. 即時顯示撰寫內容（WYSIWYG）
2. 進行版本紀錄
3. 即時上傳圖片
4. 支援 `MathJax`_
5. 使用 `KaTex`_ 或是 `MathJax v2 <MathJax_>`_
6. 儘量不考慮顯示方法（typesetting）
7. 儘量不考慮閱讀裝置（RWD）

我在選擇新的筆記紀錄方式時也希望能夠達成以上需求。
因此我選擇：

- 使用 markdown 格式而不是直接撰寫 :math:`\LaTeX` （滿足功能 6）
- 使用 `GitHub`_ page（滿足功能 2）
- 使用 `Jekyll`_ + `kramdown`_ 內建支援 `MathJax`_ 語法（滿足功能 4）
- 雖然 `KaTeX`_ 執行速度快，但支援的功能較少，而 `MathJax v3 <MathJax_>`_ 在執行速度上大幅提升，並提供 ``\label``, ``\eqref`` 等功能，讓撰寫變得更加方便
- 在 `VSCode`_ 中編輯 markdown，並且執行 ``bundle exec jekyll serve --livereload`` 讓更新能夠即時顯示在瀏覽器中（滿足功能 1）
- 使用 `Jekyll`_ 本身提供的主題功能完成具有 RWD 功能的頁面（滿足功能 7）

但我仍然遇到以下問題：

- 撰寫 markdown 時無法像 `HackMD`_ 即時完成撰寫與版本紀錄
- 需要上傳圖片時無法像 `HackMD`_ 能夠以拖拉方法完成圖片上傳
- `Jekyll v3 與 v4 <Jekyll_>`_ 的版本撰寫數學式時 `kramdown`_ 會不小心把 :math:`\TeX` 下標語法 ``_`` 誤解成 markdown 語法，導致部份數學式子無法正常 render

  - 據 `kramdown`_ 開發者稱這個 `問題 <kramdown-issue-47>`_ 已經解決，但我仍然會遇到部份數學式無法正常顯示，因此我無法確定是 `kramdown`_ 或是 `Jekyll`_ 的問題
  - 目前暫時是以 :math:`\LaTeX` 中的 ``\newcommand`` 指令想辦法避免此問題，主要是因為顯示錯誤通常發生在如 ``\text{some-text}_{sub}`` 的範例中使用 **多個字元** 組合而成的變數名稱結合下標導致無法正常 render，因此用 ``\newcommand{some-text}{\mycommand}`` 替換 macro（老實說我也不知道為什麼改用這個方法就沒問題了...）
  - 問題細節請見 `kramdown-issue-47`_
  - 我改回 `Jekyll 3.9.0 <Jekyll_>`_ 版本就沒問題了，剛好 GitHub Page 也只支援 `Jekyll 3.9.0 <Jekyll_>`_ 版本，但此時我又遇到 `kramdown`_ 會不小心把 :math:`\TeX` 星號法 ``*`` 誤解成 markdown 語法，導致無法使用 ``\begin{align*}`` 等環境
  - 由於不了解 `ruby`_ 、 `bundle`_ 、 `Jekyll`_ 的運作機制，真的無從 debug，只好捨棄上述方案

基於上述問題，我決定回到自己比較熟悉的 `Python`_ 環境撰寫筆記。
因此我暫時改成使用 `Sphinx`_ + `furo`_ 作為編寫的平台，並捨棄 markdown 語法改為 reStructuredText。
之後應該會改成使用 `ablog`_ ，由於目前（20230801） `furo`_ 與 `ablog`_ 無法一起使用（見 `ablog-issue-144`_ ），因此會等到 `ablog`_ 修好後再做更動。
我應該還是會繼續使用 `HackMD`_ 作為主要撰寫筆記的平台，並把撰寫完成的筆記內容上傳到這個網站。
如果未來我解決了以上問題，那我就有可能完全不使用 `HackMD`_ 作為撰寫平台。

.. _GitHub: https://github.com/
.. _HackMD: https://hackmd.io/
.. _Jekyll: https://jekyllrb.com/
.. _KaTeX: https://katex.org/
.. _MathJax: https://www.mathjax.org/
.. _Python: https://www.python.org/
.. _Sphinx: https://www.sphinx-doc.org/en/master/index.html
.. _VSCode: https://code.visualstudio.com/
.. _ablog: https://ablog.readthedocs.io/en/stable/
.. _ablog-issue-144: https://github.com/sunpy/ablog/pull/144
.. _bundle: https://bundler.io/
.. _furo: https://pradyunsg.me/furo/
.. _kramdown: https://kramdown.gettalong.org/index.html
.. _kramdown-issue-47: https://github.com/gettalong/kramdown/issues/47
.. _ruby: https://www.ruby-lang.org/en/
.. _智慧型知識管理實驗室: https://ikmlab.csie.ncku.edu.tw/
.. _我的 GitHub: https://github.com/ProFatXuanAll
.. _實變證明與筆記: https://github.com/ProFatXuanAll/terence-tao-analysis
.. _線代證明與筆記: https://github.com/ProFatXuanAll/linear-algebra
.. _我的 HackMD: https://hackmd.io/@profatxuanall
