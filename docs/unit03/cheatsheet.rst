Cheatsheet
======================


Activation Functions
~~~~~~~~~~~~~~~~~~~~~

ReLU (Rectified Linear Unit):
When to use: Default for hidden layers in most neural nets — fast and avoids vanishing gradients.  
``Dense(128, activation='relu')``

Leaky ReLU / ELU      
When to use: When ReLU units “die” (outputs stuck at 0).    
``LeakyReLU(alpha=0.1)``

Sigmoid               
When to use: For binary classification output (probability between 0 and 1).  
``Dense(1, activation='sigmoid')``

Tanh                  
When to use: When you want outputs between -1 and 1 (often in RNNs).  
``activation='tanh'``

Softmax               
When to use: For multi-class classification to get class probabilities.  
``Dense(10, activation='softmax')``

Linear                
When to use: For regression tasks where output can take any real value.  
``Dense(1, activation='linear')``

Optimizers
~~~~~~~~~~~

SGD (Stochastic Gradient Descent)  
When to use: When you need simplicity and full control; good for large datasets with steady convergence. 
 ``optimizer='sgd'``

SGD + Momentum      
When to use: When plain SGD is too slow or oscillates; adds memory of past gradients. 
 ``SGD(learning_rate=0.01, momentum=0.9)``

Adam                
When to use: Default choice for most deep learning models; adaptive learning rates and fast convergence. 
 ``optimizer='adam'``

RMSprop            
When to use: Works well for RNNs or non-stationary problems (changing gradients).  
``optimizer='rmsprop'``



