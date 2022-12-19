---
layout: ML-note
title: "Statistical Phrase-Based Translation"
date: 2022-03-08 14:39:00 +0800
categories: [
  Text Modeling,
]
tags: [
  machine translation,
  model architecture,
]
author: [
  Philipp Koehn,
  Franz Josef Och,
  Daniel Marcu,
]
---

|-|-|
|目標|提出以 pharse table 作為特徵進行機器翻譯|
|作者|Philipp Koehn, Franz Josef Och, Daniel Marcu|
|隸屬單位|Information Sciences Institute Department of Computer Science, University of Southern California|
|期刊/會議名稱|NAACL|
|發表時間|2003|
|論文連結|<https://dl.acm.org/doi/10.3115/1073445.1073462>|

- heuristic learning of phrase translations from word-based alignments and lexical weighting of phrase translations.  Surprisingly, learning phrases longer than three words and learning phrases from high-accuracy word-level alignment models does not have a strong impact on performance. Learning only syntactically motivated phrases degrades the performance of our systems.
- 作者認為使用語法進行翻譯的模型複雜度高但表現進步甚微
- 作者認為簡單使用最多由 3 個字組成的片語表（phrase table）進行翻譯表現效果就已經超極好
- We found extraction heuristics based on word alignments to be better than a more principled phrase-based alignment method.
- However, what constitutes the best heuristic differs from language pair to language pair and varies with the size of the training corpus.
