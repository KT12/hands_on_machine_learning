1) A CNN can use convolutional layers and max pooling to reduce the dimensionality for input images.  For a DNN, every pixel is an input, so either the network has to contain numerous nodes of information and have high compute costs or have fewer nodes and discard information that may be necessary.

DNN have no knowledge of the structure of the image.  The pixels are raveled into a vector to use as input, but the structure of the image is not known by the DNN.  This increases the possibility of overfitting.  The DNN nodes have no way of generalizing pixels or patterns which are in one location the way a CNN does.

2) Parameters for CNN

1st layer
3 * 3 * 3 + 1 = 28
100 * 28 = 2800

2nd layer
3 * 3 * 100 + 1 = 901
901 * 200 = 180200

3rd layer
3 * 3 * 200 + 1 = 1801
1801 * 400 = 720,400

2800 + 180200 + 720400 = 903400 parameters

RAM needed to compute prediction
Assuming 32 bit floats

32 bits = 4 bytes
First layer
4 * 100 * 150 * 100 = 6000000 bytes

Second layer
4 * 50 * 75 * 200 = 3000000 bytes

Third layer
4 * 25 * 36 * 400 = 1520000 bytes

3) 5 things to do to reduce memory usage
* Reduce minibatch size
* Reduce dimensionality by increasing stride
* Remove a layer
* Use 16 bit floats instead of 32 bit
* Distribute the CNN across multiple devices

4) Max pool layer is a form of dimensionality reduction and also does not require any parameters to run.

5) Local response normalization is a competitive normalization step.  It makes neurons which most strongly activate inhibit neurons at the same location in neighboring feature maps.  This encourages different feature maps to specialize, pushing them apart and forcing them to explore different features.  This improves generalization of the model.

6) The main innovations in AlexNet are drop out regularization of 50%, data augmentation by changing training images randomly (random shifts, horizontal flipping, changing lighting), and local response normalization.

GoogLenet's innovations include the inception module which combines various convolutional layers and concatenates them.  This allows the algorithm to learn complex patterns at various scales.  While there are many layers, GoogLenet actually contains fewer parameters as the convolutional layers reduce dimensionality.

Resnet's innovations include skip connections which allow the network to train on residual learning.  This speeds up learning if weights are zero and the target function is close to the identity function.  This also allows the network to make progress even if several layers have not started learning yet.