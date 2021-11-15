---
layout: about
title: 關於
---

{% include tools/math.html %}

## 關於我

這是我的 [GitHub][my-GitHub]。
目前我正在國立成功大學[智慧型知識管理實驗室][IKMLab]攻讀博士班，主要研究自然語言處理（Natural Language Processing）與深度學習（Deep Learning）演算法。

基於研究需求與個人興趣，我花了一些時間學習了實變數分析（real analysis），你可以在我的 [GitHub][my-GitHub] 上找到我在閱讀 Analysis I-II, Terence Tao 時所撰寫的[證明與筆記][Analysis]。

## 關於這個網站

這個網站是用來搬移過去[我在 HackMD][my-HackMD] 中所有的閱讀筆記。
由於 [HackMD][HackMD] 在紀錄大量數學式時 [MathJax][MathJax] v2 反應速度較慢（甚至會導致網頁直接無法反應），加上我撰寫筆記的數學內容含量較大，因此我無法繼續在 [HackMD][HackMD] 上面繼續撰寫筆記，才決定使用 [Jekyll][Jekyll] 創建這個網站。

考量到 [HackMD][HackMD] 能夠達成以下功能：

1. 即時顯示撰寫內容（WYSIWYG）
2. 進行版本紀錄
3. 即時上傳圖片
4. 支援 [MathJax][MathJax]
5. 使用 [KaTex][KaTex] 或是 [MathJax v3][MathJax]
6. 儘量不考慮顯示方法（typesetting）
7. 儘量不考慮閱讀裝置（RWD）

我在選擇新的筆記紀錄方式時也希望能夠達成以上需求。
因此我選擇：

- 使用 markdown 格式而不是直接撰寫 $\LaTeX$ （滿足功能 6）
- 使用 [GitHub][GitHub] page（滿足功能 2）
- 使用 [Jekyll][Jekyll] + [kramdown][kramdown] 內建支援 [MathJax][MathJax] 語法（滿足功能 4）
- 雖然 [KaTeX][KaTeX] 執行速度快，但支援的功能較少，而 [MathJax v3][MathJax] 在執行速度上大幅提升，並提供 `\label`, `\eqref` 等功能，讓撰寫變得更加方便
- 在 [VSCode][VSCode] 中編輯 markdown，並且執行 `bundle exec jekyll serve --livereload` 讓更新能夠即時顯示在瀏覽器中（滿足功能 1）
- 使用 [Jekyll][Jekyll] 本身提供的主題功能完成具有 RWD 功能的頁面（滿足功能 7）

但我仍然遇到以下問題：

- 撰寫 markdown 時無法像 [HackMD][HackMD] 即時完成撰寫與版本紀錄
- 需要上傳圖片時無法像 [HackMD][HackMD] 能夠以拖拉方法完成圖片上傳
- [Jekyll 4+][Jekyll] 的版本撰寫數學式時 [kramdown][kramdown] 會不小心把 $\TeX$ 下標語法 `_` 誤解成 markdown 語法，導致部份數學式子無法正常 render
  - 據 [kramdown][kramdown] 開發者稱這個[問題][kramdown-issue-47]已經解決，但我仍然會遇到部份數學式無法正常顯示，因此我無法確定是 [kramdown][kramdown] 或是 [Jekyll][Jekyll] 的問題
  - 目前暫時是以 $\LaTeX$ 中的 `\newcommand` 指令想辦法避免此問題，主要是因為顯示錯誤通常發生在如 `\text{some-text}_{sub}` 的範例中使用**多個字元**組合而成的變數名稱結合下標導致無法正常 render，因此用 `\newcommand{some-text}{\mycommand}` 替換 macro。老實說我也不知道為什麼改用這個方法就沒問題了...
  - 問題細節請見 [kramdown-issue-47][kramdown-issue-47]
  - 我改回 [Jekyll 3.9.0][Jekyll] 版本就沒問題了，剛好 GitHub Page 也只支援 [Jekyll 3.9.0][Jekyll] 版本

因此我應該還是會繼續使用 [HackMD][HackMD] 作為主要撰寫筆記的平台，並把撰寫完成的筆記內容上傳到這個網站。
如果未來我解決了以上問題，那我就有可能完全不使用 [HackMD][HackMD] 作為撰寫平台。

[Analysis]: https://github.com/ProFatXuanAll/terence-tao-analysis
[GitHub]: https://github.com/
[HackMD]: https://hackmd.io/
[IKMLab]: https://ikmlab.csie.ncku.edu.tw/
[Jekyll]: https://jekyllrb.com/
[KaTeX]: https://katex.org/
[kramdown]: https://kramdown.gettalong.org/index.html
[kramdown-issue-47]: https://github.com/gettalong/kramdown/issues/47
[MathJax]: https://www.mathjax.org/
[my-HackMD]: https://hackmd.io/@profatxuanall
[my-GitHub]: https://github.com/ProFatXuanAll
[VSCode]: https://code.visualstudio.com/
