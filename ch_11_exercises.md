1) It's not OK to initialize all the same weights.  Getting trapped at local minima is a problem so it's better to give them different intial weights.  More importantly, if the weights are all the same, the neurons will follow the same gradient.  Symmetry breaking is important.

2) It is possible and common to initialize the biases to be zero, since the asymmetry breaking is provided by the small random numbers in the weights. For ReLU non-linearities, some people like to use small constant value such as 0.01 for all biases because this ensures that all ReLU units fire in the beginning and therefore obtain and propagate some gradient. However, it is not clear if this provides a consistent improvement (in fact some results seem to indicate that this performs worse) and it is more common to simply use 0 bias initialization.  Answer from [here](http://cs231n.github.io/neural-networks-2/)

3) ELU activation advantages over ReLU:
* ELU can take negative values when z < 0.  Helps solve the vanishing gradients problem.
* Nonzero gradient where z < 0 means dying units are avoided.
* Function is smooth everywhere, which speeds up gradient descent.  Not as much bouncing around z = 0.

4) Activation functions and when to use them.

* ELU should be the default DNN activation function.  It's superior to ReLU.
* Leaky ReLU should be used if you need a lightning fast model (real time data?) or if you see dying neurons in a ReLU network.
* Tanh and logistic activation function could be used with batch normalization.  Tanh effectively normalizes output to speed up convergence.
* Randomized ReLUs can be used if the network is overfitting.
* Parametric ReLUs can be used when there is a ton of data.
* Softmax is good for catergorical output (ie MNIST and choosing 1 of 10 digits) and also NLP tasks for the final output layer.

5) If the parameter for the momentum optimizer is too close to 1, it will overshoot and oscillate, taking longer to converge than a non-momentum optimizer.

6) Can produce a sparse model by using L1 regularization or zeroing out tiny weights after training.  Can alose use FTRL instead of ADAM optimization with L1 reg.

7) Drop-out slows down training by a factor of 2.  It does not slow down inference.