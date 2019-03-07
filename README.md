# Programming Assignment 1

CS 640 

Name: Vijay Nagaraj Karigowdara

Team Member: Diptanshu Singh

Date: 03/03/2019

## Problem Definition

To build a neural network and implement the backpropagation algorithm to train it.

## Method and Implementation

We have to perform 3 tasks with this network -

1. Separation of linear data
2. Separation of non-linear data
3. Digit recognition
We will implement a simple 2 layer neural network with following specification.

Below are the choices for the activation functions:

Sigmoid function: Gives the weighted sum between 0 and 1 with some bias in it for it to be inactive and only fire when the threshold is reached.
Softmax function: Its capability of converting output as probabilities.

1. Layer 1 :
Input = Number of features
Output = Number of nodes in hidden layer
Activation function = Sigmoid

2. Layer 2 :
Input = Number of nodes in hidden layer
Output = Number of dimensions in the output = number of classes in y
Activation function = None
The ouput will be passed through softmax function to intepret the output as probabilities.

## Limitations

This neural network will introduce some non-linearity in the analysis but it will not be scalable. The network will also not be able to We will be hard coding the forward and backward propagation steps in the neural network.

This neural network will only be able to understand the very simple non-linear functions. As we are using only 1 hidden layer, the network can introuce only 1 layer of non-linearity in the network. To calculate more complex function, we can select a network with more number of nodes in hidden layer, but a better way to go about it to introduce more layers in the network.
Define your evaluation metrics, e.g., detection rates, accuracy, running time.

## Exploration

Hyperparameters for the network :

1. Number of nodes in the hidden layer
2. Learning Rate
3. Number of Epochs
We will also understand the effect of these hyperparameters on our network by analysing the test error and train error in our network.

## Implementation of Neural Network

Functions used for the implementation :

1. forward_prop : Input is the feature vector. Equations :
 sig_h = W_1 * X + b_1    
 o__h  = 1 / ( 1 + e^(sig_h))  
 sig_f = W_2 * o_h + b_2  
 o_f   = softmax(sig_f)  
Output of this function is the activations in the layer . We will cache these so that we can use it in backpropargation.
2. gradient_desc : Input is the activation cache from forward prop, X ( feature vectors ) and y ( output vector). We backpropagate the derivates backward. Using the cross entropy loss and softmax function in the final layer, we get

 dp/d o_f = o_f - y {where o_f is the softmax score}     ..... eq 1   
 This has been implemented in the error_calc function

 dp/d w2  = eq 1 * d o_f /d sig_f * d sig_f/d w2
 and so on.  
 This has been implemented within the gradient descent function 
3. update_weight : This will update the weight in the previous epoch with learning rate * gradient . This directly updates the weight and bias terms in the neural network.

4. predict : This is used after the neural network has been trained.

5. hyperpara : This sets the hyperparameters in the neural network which will be used in the network

6. Compute_Cost: Computes cost of the data set. This is the cost function which we hope to minimize after k epochs
fit: Computer the train error after completeing k epoch
predict: This is used after the neural network has been trained.

## Experiments and Results
### Linear Data

![alt text](https://github.com/vijaykarigowdara/ai640_pa1/blob/master/image.png)

The scatter plot shows that this is a linearly separable case
We would not need a lot of epochs to come up with suitable weights as the space is linearly separable. As a result, I have restricted the number of epochs to be low : 500.
Having higher number of epochs can lead to overfitting.

Error in test set is  0.75 %
![alt text](https://github.com/vijaykarigowdara/ai640_pa1/blob/master/linear_plots.png)

### Non-Linear Data

![alt text](https://github.com/vijaykarigowdara/ai640_pa1/blob/master/non_linear_scatter.png)
We would need more epochs since the function being calculated is complex. We will explore the result of selecting different values of num_epochs and learning rate in the next section.
![alt text](https://github.com/vijaykarigowdara/ai640_pa1/blob/master/non_linear_plots.png)
![alt text](https://github.com/vijaykarigowdara/ai640_pa1/blob/master/non_linear_2.png)

### Cross Validation

By using K Fold cross validation, we divide the data into k subset and perform the same procedure k time. In each iteration, we take one of the k subsets for validation set and other k-1 for training set. When we take the error estimation for all k iteration and get their average, we find the effectiveness of our model. As we use the most of our data for fitting, we significantly reduce bias of our network as well the the variance as we use these data in validation. Therefore interchanging the training and testing dataset improves the effectiveness of our system.

### Regularization


Regularization helps in improving the model's performance by using regularization parameter lambda. This helps in case when the model performs better in train dataset but not in test data. This is also known as Overfitting. Model tries to learn all the details and noises in the training data and creates a convoluted model to closely fit the data points such that it doesn’t generalize well for the unseen data. So regularization technique helps the model to generalize better. This in turn improves the model’s performance on the unseen data as well.  We can add a penalty in the performance function which would indicate how complex a function is. Below are 3 methods to implement them

1. L2 regularization: We use L2 Regularization to reduce the variance of the estimator by simplifying it. This will increase the bias and our expected error decreases.

2. Wrapper: This method will enumerate the model with respect to the number of nodes in the neural network. We will have list of models with respective errors and can chose the model with minimum error.

3. Feature selection: The core issue with overfitting is that it has more number of independent parameters than the number of data points. Thus, by discarding irrelevant attributes we can avoid overfitting.

### Performance effect due to L2 Regularization

BYe introducing a cost term for bringing in more features with the objective function. Which means we will penalize the hypothesis complexity. Therefore, it tries to push the coefficients for many variables to zero and hence reduce cost term.

#### Implementation of L2 regularization :  
We are adding a term lambda*weight to the gradient on every term. This is used because for L2 regularization, we add lambda/2 * ( weight ) ^ 2 to the performace function. Derivative of this function wrt weight is lambda * weight. 

![alt text](https://github.com/vijaykarigowdara/ai640_pa1/blob/master/regularized.png)


## Digit Recognition

We have taken a neural network with more than 10 nodes in the layer. If we tke less than 10, the network will have to share computations, which may lead to poor performance.

![alt text](https://github.com/vijaykarigowdara/ai640_pa1/blob/master/Digit.png)


## Learning Rate effect on Neural Network

Learning rate controls how much we are adjusting the weights of our network with respect the loss gradient. Lower the value, the slower we move along the downward slope. While this might be a good idea in terms of making sure that we don't miss any local minima, it could also mean that model will take a long time to converge. Hence, if we chose an optimal learning rate, we would spend lesser time to train our neural network.


