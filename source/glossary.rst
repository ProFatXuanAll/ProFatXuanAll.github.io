============
專有名詞
============

.. glossary::

  BPTT
  Back-Propagation Throught Time
    BPTT is an :term:`optimization` algorithm based on :term:`gradient descent` often used to optimize :term:`RNN` models.

  Gradient Descent
  gradient descent
  梯度下降法
    An :term:`optimization` algorithm often uses to optimize :term:`NN` models.

  Gradient Explosion
  gradient explosion
  梯度爆炸
    Gradient explosion is a phenomenon that often happens in :term:`RNN` :term:`optimization`.
    Gradient magnitude sometimes grows so large that it causes optimization to fail.

  Gradient Vanishing
  gradient vanishing
  梯度消失
    Gradient vanishing is a phenomenon that often happens in :term:`NN` :term:`optimization`.
    Gradient magnitude tends to get close to zero since back-propagation uses chain rule, which is essentially just a bunch of multiplications, and small number times small number get smaller.
    Eventually, no parameters can be updated, and it causes optimization to fail.

  LSTM
  Long Short-Term Memory
    LSTM is a :term:`RNN` variant.

  NN
  Neural Network
  神經網路
    NN is a machine learning model.

  Optimization
  optimization
  optimize
  最佳化
    Optimization refers to an algorithm that helps machine learning models achieve their objectives.

  RNN
  Recurrent Neural Network
  遞歸神經網路
    RNN is a :term:`NN` variant often used to solve problems on sequential data.

  RTRL
  Real Time Recurrent Learning
    RTRL is an :term:`optimization` algorithm based on :term:`gradient descent` often used to optimize :term:`RNN` models.

