================
sigmoid 函數特性
================

.. ====================================================================================================================
.. Setup SEO.
.. ====================================================================================================================

.. meta::
  :description:
    sigmoid 函數特性
  :keywords:
    functional analysis
    real analysis
    sigmoid

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
:bdg-secondary:`sigmoid`

結論
====

- sigmoid 函數輸入與輸出皆為\ **實數**，**值域**\為 :math:`(0, 1)`
- sigmoid 函數為\ **嚴格遞增**\（**strictly monotonically increasing**）函數
- sigmoid 函數在實數域\ **連續**\且\ **可微**\（**continous/differentiable** on :math:`\R`）
- 對 sigmoid 微分後的函數\ **值域**\為 :math:`(0, 0.25]`，最大值 :math:`0.25` 發生在 :math:`x = 0`

Definition
==========

Define **sigmoid** to be the function :math:`\sigma : \R \to \R` where

.. math::
  :nowrap:

  \[
    \sigma(x) = \frac{1}{1 + e^{-x}}
  \]

for all :math:`x \in \R`.

Strictly Monotonically Increasing
=================================

Let :math:`x_1, x_2` be real numbers such that :math:`x_1 < x_2`.
Then we have :math:`\sigma(x_1) < \sigma(x_2)`。

.. dropdown:: Proof of strictly monotonically increasing.

  .. math::
    :nowrap:

    \[
      \begin{align*}
                 & x_1 < x_2 \\
        \implies & -x_1 > -x_2 \\
        \implies & e^{-x_1} > e^{-x_2} \\
        \implies & 1 + e^{-x_1} > 1 + e^{-x_2} \\
        \implies & \frac{1}{1 + e^{-x_1}} < \frac{1}{1 + e^{-x_2}} \\
        \implies & \sigma(x_1) < \sigma(x_2).
      \end{align*}
    \]

Continuous and Differentiable on :math:`\R`
===========================================

For any real number :math:`x`, we have :math:`\sigma'(x) = \sigma(x) \cdot \qty(1 - \sigma(x))`.

.. dropdown:: Proof of continuous and differentiable.

  Let :math:`x \in \R` and let :math:`f: \R \to \R` be the function where

  .. math::
    :nowrap:

    \[
      f(x) = 1 + e^{-x}.
    \]

  Since :math:`f` is differentiable on :math:`\R` and :math:`f(x) \neq 0`, we know that :math:`1 / f` is differentiable on :math:`\R`.
  Since :math:`\sigma = 1 / f`, we know that :math:`\sigma` is differentiable on :math:`\R`.
  Thus,

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \sigma'(x) & = \frac{e^{-x}}{(1 + e^{-x})^2} \\
                   & = \frac{1}{1 + e^{-x}} \cdot \frac{e^{-x}}{1 + e^{-x}} \\
                   & = \frac{1}{1 + e^{-x}} \cdot \qty(\frac{1 + e^{-x}}{1 + e^{-x}} - \frac{1}{1 + e^{-x}}) \\
                   & = \sigma(x) \cdot \qty(1 - \sigma(x)). \\
      \end{align*}
    \]

  Since differentiable functions are continuous, we know that :math:`\sigma` is continuous on :math:`\R`.

Range
=====

We have :math:`\sigma(\R) = (0, 1)`.

.. dropdown:: Proof of range.

  Since sigmoid is strictly monotonically increasing and continuous on :math:`\R`, we know that

  .. math::
    :nowrap:

    \[
      \sigma((a, b)) = (\sigma(a), \sigma(b))
    \]

  for any open interval :math:`(a, b)`.
  So the range of sigmoid is

  .. math::
    :nowrap:

    \[
      \sigma(\R) = \sigma((-\infty, \infty)) = \qty(\sigma(-\infty), \sigma(\infty)).
    \]

  Since we have

  .. math::
    :nowrap:

    \[
      \lim_{x \to \infty} \sigma(x)  = \frac{1}{1 + 0} = 1 \qq{and} \lim_{x \to -\infty} \sigma(x) = \frac{1}{1 + \infty} = 0,
    \]

  we know that :math:`\sigma(\R) = (0, 1)`.

Range of Derivative
===================

We have :math:`\sigma'(\R) = (0, 0.25]`.
The maximum value of :math:`\sigma'` happens at :math:`x = 0`.

.. dropdown:: Proof of range of derivative.

  The maximum value happens at the point where derivative equals to :math:`0`.
  So we can find the maximum of :math:`\sigma' = \sigma \cdot (1 - \sigma)` by solving :math:`\dv{\sigma \cdot (1 - \sigma)}{\sigma} = 0`.

  .. math::
    :nowrap:

    \[
      \begin{align*}
                 & \dv{\sigma \cdot (1 - \sigma)}{\sigma} = \dv{\sigma - \sigma^2}{\sigma} = 1 - 2\sigma = 0 \\
        \implies & \sigma = \frac{1}{2}.
      \end{align*}
    \]

  This means the maximum of :math:`\sigma'` happens when :math:`\sigma = \frac{1}{2}`.
  Thus,

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \max_{x \in \R} \sigma'(x) & = \max_{x \in \R} \sigma(x) \cdot \qty(1 - \sigma(x)) \\
                                   & = 0.5 \cdot (1 - 0.5) \\
                                   & = 0.25.
      \end{align*}
    \]

  Since

  .. math::
    :nowrap:

    \[
      \begin{align*}
                 & \sigma(x) = \frac{1}{1 + e^{-x}} = \frac{1}{2} \\
        \implies & 1 + e^{-x} = 2 \\
        \implies & e^{-x} = 1 \\
        \implies & -x = \ln(1) = 0 \\
        \implies & x = 0,
      \end{align*}
    \]

  we know the maximum of :math:`\sigma'` happens at :math:`x = 0`.
  So we have :math:`\sigma(\R) \subseteq (-\infty, 0.25]`.
  Since

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \sigma''(x) & = \sigma'(x) \cdot \qty(1 - \sigma(x)) + \sigma(x) \cdot \qty(1 - \sigma(x))' \\
                    & = \sigma(x) \cdot \qty(1 - \sigma(x))^2 - \sigma^2(x) \cdot \qty(1 - \sigma(x)) \\
                    & = \qty(1 - 2 \sigma(x)) \cdot \sigma(x) \cdot \qty(1 - \sigma(x)),
      \end{align*}
    \]

  we know that :math:`\sigma''(x) > 0` when :math:`x \in \R^-`.
  So we know that :math:`\sigma'` is monotonically increasing on :math:`\R^-`.
  Since

  .. math::
    :nowrap:

    \[
      \begin{align*}
        \lim_{x \to -\infty} \sigma'(x) & = \qty[\lim_{x \to -\infty} \sigma(x)] \cdot \qty[\lim_{x \to -\infty} \qty(1 - \sigma(x))] \\
                                        & = 0 \cdot (1 - 0) \\
                                        & = 0,
      \end{align*}
    \]

  we know that :math:`\sigma'(\R^-) \subseteq (0, 0.25)`.
  But we know that :math:`\sigma'` is continuous on :math:`\R`, so we must have :math:`\sigma'(\R^-) = (0, 0.25)`.
  Similar arguments show that :math:`\sigma'` is monotonically decreasing on :math:`\R^+` and :math:`\sigma'(\R^+) = (0, 0.25)`.
  Thus we have :math:`\sigma'(\R) = (0, 0.25]`.
