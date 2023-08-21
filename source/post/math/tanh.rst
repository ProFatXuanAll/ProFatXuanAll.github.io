======================
:math:`\tanh` 函數特性
======================

.. ====================================================================================================================
.. Setup SEO.
.. ====================================================================================================================

.. meta::
  :description:
    tanh 函數特性
  :keywords:
    functional analysis
    real analysis
    tanh

.. ====================================================================================================================
.. Setup front matter.
.. ====================================================================================================================

.. article-info::
  :author: ProFatXuanAll
  :date: 2023-08-21
  :class-container: sd-p-2 sd-outline-muted sd-rounded-1

.. ====================================================================================================================
.. Create visible tags from SEO keywords.
.. ====================================================================================================================

:bdg-secondary:`functional analysis`
:bdg-secondary:`real analysis`
:bdg-secondary:`tanh`

結論
====

- :math:`\tanh` 函數輸入與輸出皆為\ **實數**，**值域**\為 :math:`(-1, 1)`
- :math:`\tanh(x) = 2 \sigma(2x) - 1`，其中 :math:`\sigma` 為 sigmoid 函數
- :math:`\tanh` 函數為\ **嚴格遞增**\（**strictly monotonically increasing**）函數
- :math:`\tanh` 函數在實數域\ **連續**\且\ **可微**\（**continous/differentiable** on :math:`\R`）
- 對 :math:`\tanh` 微分後的函數\ **值域**\為 :math:`(0, 1]`

Definition
==========

Define **hyperbolic tangent** to be the function :math:`\tanh : \R \to \R` where

.. math::
  :nowrap:

  \[
     \tanh(x) = \frac{\sinh(x)}{\cosh(x)} = \frac{e^x - e^{-x}}{e^x + e^{-x}}
  \]

for all :math:`x \in \R`.

Relation with sigmoid
=====================

Let :math:`\sigma : \R \to \R` be :doc:`sigmoid </post/math/sigmoid>` function.
Then we have :math:`\tanh(x) = 2 \sigma(2x) - 1` for all :math:`x \in \R`.

.. dropdown:: Proof of relation with sigmoid.

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \tanh(x) & = \frac{e^x - e^{-x}}{e^x + e^{-x}} \\
                 & = \frac{\qty(e^x - e^{-x}) e^{-x}}{\qty(e^x + e^{-x}) e^{-x}} \\
                 & = \frac{1 - e^{-2x}}{1 + e^{-2x}} \\
                 & = \frac{2}{1 + e^{-2x}} - \frac{1 + e^{-2x}}{1 + e^{-2x}} \\
                 & = 2 \sigma(2x) - 1.
      \end{align*}
    \]

Strictly Monotonically Increasing
=================================

Let :math:`x_1, x_2` be real numbers such that :math:`x_1 < x_2`.
Then we have :math:`\tanh(x_1) < \tanh(x_2)`。

.. dropdown:: Proof of strictly monotonically increasing.

  We know that :math:`\sigma` is strictly monotonically increasing.
  Thus,

  .. math::
    :nowrap:

    \[
      \begin{align*}
                 & x_1 < x_2 \\
        \implies & 2 x_1 < 2 x_2 \\
        \implies & \sigma(2 x_1) < \sigma(2 x_2) \\
        \implies & 2 \sigma(2 x_1) < 2 \sigma(2 x_2) \\
        \implies & 2 \sigma(2 x_1) - 1 < 2 \sigma(2 x_2) - 1 \\
        \implies & \tanh(x_1) < \tanh(x_2).
      \end{align*}
    \]

Continuous and Differentiable on :math:`\R`
===========================================

For any real number :math:`x`, we have :math:`\tanh'(x) = 4 (\sigma')(2x)`.

.. dropdown:: Proof of continuous and differentiable.

  Since :math:`\sigma` is continuous and differentiable on :math:`\R`, we know that :math:`\tanh` is continuous and differentiable on :math:`\R`.
  So we have

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \tanh'(x) & = \dv{2 \sigma(2x) - 1}{x} \\
                  & = \dv{2 \sigma(2x) - 1}{2x} \cdot \dv{2x}{x} \\
                  & = 2 \cdot \dv{\sigma(2x)}{2x} \cdot 2 \\
                  & = 2 \cdot \sigma(2x) \cdot \qty(1 - \sigma(2x)) \cdot 2 \\
                  & = 4 (\sigma')(2x).
      \end{align*}
    \]

Range
=====

We have :math:`\tanh(\R) = (-1, 1)`.

.. dropdown:: Proof of range.

  We know that :math:`\sigma(\R) = (0, 1)`.
  So we have

  .. math::
    :nowrap:

    \[
      \begin{align*}
                 & \forall x \in \R, \sigma(x) \in (0, 1) \\
        \implies & \forall x \in \R, \sigma(2x) \in (0, 1) \\
        \implies & \forall x \in \R, 2 \sigma(2x) \in (0, 2) \\
        \implies & \forall x \in \R, 2 \sigma(2x) - 1 \in (-1, 1) \\
        \implies & \forall x \in \R, \tanh(x) \in (-1, 1).
      \end{align*}
    \]


Range of Derivative
===================

We have :math:`\tanh'(\R) = (0, 1]`.
The maximum value of :math:`\tanh'` happens at :math:`x = 0`.

.. dropdown:: Proof of range of derivative.

  The maximum value of :math:`\sigma'` is :math:`0.25`, which happens at :math:`x = 0`.
  Since :math:`\tanh(x) = 4 (\sigma')(2x)`, we know that the maximum value of :math:`\tanh'` also happen at :math:`x = 0`, with value equal to :math:`4 \cdot 0.25 = 1`.
  Since :math:`\sigma'(\R) = (0, 0.25]`, we have

  .. math::
    :nowrap:

    \[
      \begin{align*}
                 & \forall x \in \R, \sigma'(x) \in (0, 0.25] \\
        \implies & \forall x \in \R, \sigma'(2x) \in (0, 0.25] \\
        \implies & \forall x \in \R, 4 \sigma'(2x) \in (0, 1] \\
        \implies & \forall x \in \R, \tanh'(x) \in (0, 1].
      \end{align*}
    \]
