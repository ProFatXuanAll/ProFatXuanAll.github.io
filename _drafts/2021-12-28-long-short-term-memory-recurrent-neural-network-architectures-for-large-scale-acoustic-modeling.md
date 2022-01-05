---
layout: ML-note
title:  "Long Short-Term Memory Recurrent Neural Network Architectures for Large Scale Acoustic Modeling"
date:   2021-12-28 13:39:00 +0800
categories: [
  Deep Learning,
  Model Architecture,
]
tags: [
  RNN,
  LSTM,
]
author: [
  Hasim Sak,
  Andrew W. Senior,
  Françoise Beaufays,
]
---

|-|-|
|目標|嘗試以 LSTM 進行大規模的語音辨識|
|作者|Hasim Sak, Andrew W. Senior, Françoise Beaufays|
|隸屬單位|Google|
|期刊/會議名稱|Interspeech|
|發表時間|2014|
|論文連結|<https://research.google/pubs/pub43905/>|

## 重點

- Google 好像重複投稿，見[這篇][pub43895]
  - [PyTorch 實作的 LSTM][PyTorch-LSTM] 就是參考這篇論文的[重複投稿版本][pub43895]
- 第一篇論文嘗試以大量機器 + asynchronous stochastic gradient descent 訓練 LSTM 進行語音辨識
  - 兩層 LSTM 可以達到語音辨識的 SOTA
  - 比 RNN + feed-forward 架構表現還好
  - 比單純使用 feed-forward 架構的參數數量少好幾個數量級

[pub43895]: https://research.google/pubs/pub43895/
[PyTorch-LSTM]: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
