Activation Functions
======================
====================  ==============================================  =============================================
Activation Function   When to Use                                      Example
====================  ==============================================  =============================================
ReLU (Rectified Linear Unit)  Default for hidden layers in most neural nets — fast and avoids vanishing gradients.  ``Dense(128, activation='relu')``
Leaky ReLU / ELU      When ReLU units “die” (outputs stuck at 0).    ``LeakyReLU(alpha=0.1)``
Sigmoid               For binary classification output (probability between 0 and 1).  ``Dense(1, activation='sigmoid')``
Tanh                  When you want outputs between -1 and 1 (often in RNNs).  ``activation='tanh'``
Softmax               For multi-class classification to get class probabilities.  ``Dense(10, activation='softmax')``
Linear                For regression tasks where output can take any real value.  ``Dense(1, activation='linear')``
====================  ==============================================  =============================================

Optimizers
===========

==================  ===========================================================  ================================================
Optimizer           When to Use                                                 Example
==================  ===========================================================  ================================================
SGD (Stochastic Gradient Descent)  When you need simplicity and full control; good for large datasets with steady convergence.  ``optimizer='sgd'``
SGD + Momentum      When plain SGD is too slow or oscillates; adds memory of past gradients.  ``SGD(learning_rate=0.01, momentum=0.9)``
Adam                Default choice for most deep learning models; adaptive learning rates and fast convergence.  ``optimizer='adam'``
RMSprop             Works well for RNNs or non-stationary problems (changing gradients).  ``optimizer='rmsprop'``
==================  ===========================================================  ================================================



