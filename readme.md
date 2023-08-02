# Custom Layer Package

[![GitHub license](https://img.shields.io/github/license/iampareshk/decodeDeepLearning)](https://github.com/iampareshk/decodeDeepLearning/blob/main/LICENSE)

## Overview

This is a custom layer package for TensorFlow Keras that allows you to create custom layers easily. It provides a simple function to generate custom layers with custom forward pass and output shape computation. This package is useful when you want to create complex custom layers that are not available in the standard Keras library.

## Installation

To install the custom layer package, you can use `pip`. Run the following command in your terminal:

```bash
pip install git+https://github.com/iampareshk/decodeDeepLearning.git@master


##Example of using the custom layer
#Assuming you have already created your custom pooling function custom_pooling_function, you can create a custom layer using the #CustomLayer function from the decodeDeepLearning package.

'''python
#Import the necessary modules
import tensorflow as tf
from custom_pooling_layer_module import CustomLayer


#Next, define your custom forward pass function. This function will implement the logic for your custom layer. For example, let's say we #want to compute the mean of the inputs along the last dimension:
def custom_forward_pass(inputs):
    return tf.reduce_mean(inputs, axis=-1)

#Now, create a custom layer using the CustomLayer class and pass your custom forward pass function as an argument:
custom_layer = CustomLayer(call_function=custom_forward_pass)

#You can now use this custom layer in your TensorFlow Keras models just like any other Keras layer. For example, you can use it in a model #like this:
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10,)),
    custom_layer,
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#test
input_data = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
output = custom_layer(input_data)

print("Input Data:")
print(input_data)
print("\nOutput Data:")
print(output)


Input Data:
tf.Tensor(
[[1. 2. 3.]
 [4. 5. 6.]], shape=(2, 3), dtype=float32)

Output Data:
tf.Tensor(
[2. 5.], shape=(2,), dtype=float32)

'''




