import tensorflow as tf
from decodeDeepLearning.custom_pooling_layer_module.custom_pooling_layer import CustomPoolingLayer

# Define the Custom Forward Pass Function with L2 Pooling
def l2_pooling(inputs):
    return tf.sqrt(tf.reduce_sum(tf.square(inputs), axis=-1))

# Create the Custom Layer
custom_layer = CustomPoolingLayer(call_function=l2_pooling)

# Load and preprocess the dataset (example using the Fashion MNIST dataset)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train[..., tf.newaxis].astype('float32') / 255.0
x_test = x_test[..., tf.newaxis].astype('float32') / 255.0

# Build the Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    custom_layer,  # Adding our custom L2 Pooling layer
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  # 10 classes for Fashion MNIST
])

# Compile the Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the Model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the Model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print("Test accuracy:", test_acc)
