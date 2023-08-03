import tensorflow as tf
from tensorflow.keras.layers import Layer

def create_custom_layer(call_function, compute_output_shape_function=None, **kwargs):
    """
    Creates a custom layer in TensorFlow Keras.

    Parameters:
        call_function (function): The forward pass function of the layer.
        compute_output_shape_function (function): Optional. The function to compute the output shape
                                                  of the layer. If not provided, the output shape
                                                  will be inferred from the call_function.

        **kwargs: aAdditional keyword arguments that will be passed to the layer's constructor.

    Returns:
        class: A custom layer class that can be added to Keras models.
    """
    class CustomLayer(Layer):
        def __init__(self, **kwargs):
            super(CustomLayer, self).__init__(**kwargs)
            self._call_function = call_function
            self._compute_output_shape_function = compute_output_shape_function

        def call(self, inputs, training=None):
            return self._call_function(inputs)

        def compute_output_shape(self, input_shape):
            if self._compute_output_shape_function:
                return self._compute_output_shape_function(input_shape)
            else:
                # If compute_output_shape_function is not provided, infer output shape from the call function
                output_shape = self._call_function(tf.zeros(input_shape))
                return output_shape.shape

        def get_config(self):
            config = super(CustomLayer, self).get_config()
            return config

        @classmethod
        def from_config(cls, config):
            return cls(**config)

    return CustomLayer(**kwargs)
