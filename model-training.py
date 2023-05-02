# %% [markdown]
# \fontsize{16}{16}\selectfont
#
# Assignment: ML Final Project
#
# Name: Theo Crandall
#
# Description: A CNN that predicts the number drawn on a canvas.
# \fontsize{12}{12}\selectfont

# %%
# Import the necessary libraries
from scipy.io import loadmat
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# %%
# Load the data
data = loadmat("mnist-original.mat")
X = data["data"].T
y = data["label"].T

# Reshape the data
X = X.reshape(-1, 28, 28, 1)

# %%
# Split the data into training and testing sets
test_percent = 0.2
test_size = int(test_percent * len(X))
X_train, X_test = X[:-test_size], X[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]

# %% [markdown]
# \pagebreak
# Create the model
#
# - The first layer is a convolutional layer with 32 filters, a kernel size of 3x3, and a ReLU activation function.
#
# - The second layer is a max pooling layer with a pool size of 2x2. This reduces the image size.
#
# - The third layer is a flattening layer. This flattens the image into a 1D array.
#
# - The fourth layer is a dense layer with 128 neurons and a ReLU activation function.
#
# - The fifth layer is a dense layer with 10 neurons and a softmax activation function. This is the output layer.

# %%
model = Sequential(
    [
        Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(10, activation="softmax"),
    ]
)

# %%
# Compile the model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# %% [markdown]
# \pagebreak

# %%
# Train the model for 10 epochs
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=2)

# %%
# Evaluate the model
accuracy = model.evaluate(X_test, y_test, verbose=2)

# %%
# Save the model
model.save(f"models/mnist_model - {accuracy[1]:.4f}.h5")

# %%
