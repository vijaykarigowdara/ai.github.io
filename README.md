# Programming Assignment 1

CS 640 

Vijay Nagaraj Karigowdara

Diptanshu Singh

Date: 03/03/2019

## Problem Definition

To build a neural network and implement the backpropagation algorithm to train it.



## Method and Implementation

We have to perform 3 tasks with this network -

1. Separation of linear data
2. Separation of non-linear data
3. Digit recognition
We will implement a simple 2 layer neural network with following specification.

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

## Helper functions

1.sig : Gives the sigmoid of the input array ( calculated as f(z) = 1/(1+e^(-z)) )
2.softmax : Gives the softmax of the input array ( calculated as f(z) = [e^(z1)/ sum(e^(zi), e^(z1)/ sum(e^(zi),....]
3.softmax_to_y : Input softmax vector and returns the most probable class
4.split : Splits the dataframe into k parts ( both X and y) and returns a list
5.cross validation : Performs cross validation by spliting the dataset into test, train. Returns the average of test_error and train_error from the dataset
6.plot_decision_boundary : Plots the decision boundary in 2D space. Can only be used when inoput is 2 dimensional.
7.error_rate : Returns the ratio of misclassified points

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

## Experiments and Results
### Linear Data



The scatter plot shows that this is a linearly separable case

We would not need a lot of epochs to come up with suitable weights as the space is linearly separable. As a result, I have restricted the number of epochs to be low : 500.

Having higher number of epochs can lead to overfitting.

