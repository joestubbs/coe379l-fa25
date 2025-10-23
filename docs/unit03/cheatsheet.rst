Cheatsheet
======================


Activation Functions
~~~~~~~~~~~~~~~~~~~~~
One of the most important properties of activation functions is their non-linearity, which enables the models to learn complex relationships from the data that go beyond simple, linear relationships[1].

**ReLU (Rectified Linear Unit)**:
When to use: Most commonly used activation function in deep learning today. It’s simple yet effective. Default for hidden layers in most neural nets — fast and avoids vanishing gradients.  
``Dense(128, activation='relu')``

**Leaky ReLU / ELU**
When to use: Leaky ReLU addresses the dying ReLU [2] problem by allowing a small negative slope instead of zero.    
``LeakyReLU(alpha=0.1) #Add LeakyReLU as a separate layer``

**Sigmoid**
When to use: For binary classification output (probability between 0 and 1).Sigmoid squashes values between 0 and 1, making it useful for outputs that represent probabilities.  
``Dense(1, activation='sigmoid')``

**Tanh (Hyperbolic Tangent)**
When to use: Similar to sigmoid but when you want outputs between -1 and 1. It can help with convergence in some networks.   
``Dense(1, activation='tanh')``

**Softmax**
When to use: For multi-class classification to get class probabilities.  
``Dense(10, activation='softmax')``

**Linear**
When to use: For regression tasks where output can take any real value.  
``Dense(1, activation='linear')``

Optimizers
~~~~~~~~~~~
Optimizers are algorithms or methods used to adjust the weights and biases of a neural network to minimize the loss function during training. By iteratively updating these parameters, optimizers ensure that the model learns effectively from the data, improving its predictions.[3]

**SGD (Stochastic Gradient Descent)**
When to use: SGD is a foundational optimizer that updates weights using a single data point at a time. While simple, it’s prone to oscillations in the loss function.; good for large datasets with steady convergence. 
 ``optimizer='sgd'``

**SGD + Momentum**
When to use: When plain SGD is too slow or oscillates; adds memory of past gradients. 
 ``SGD(learning_rate=0.01, momentum=0.9)``

**Adam**
When to use: Default choice for most deep learning models; adaptive learning rates and fast convergence. 
 ``optimizer='adam'``

**RMSprop**
When to use: Works well for RNNs or non-stationary problems (changing gradients).  
``optimizer='rmsprop'``


References:

1. Activation Functions [https://towardsdatascience.com/activation-functions-in-neural-networks-how-to-choose-the-right-one-cb20414c04e5/]

2. Dying ReLU problem [https://pythonguides.com/pytorch-leaky-relu/]

3. Optimizers [https://akridata.ai/blog/optimizers-in-deep-learning/]
