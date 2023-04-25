# %%
from scipy.io import loadmat
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# %%
# Load the data
data = loadmat("mnist-original.mat")
X = data["data"].T
y = data["label"].T

# %%
# Split the data into training and testing sets
test_percent = 0.2
test_size = int(test_percent * len(X))
X_train, X_test = X[:-test_size], X[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]


# %%
# Create the model
model = Sequential(
    [
        Dense(200, activation="relu", input_shape=(784,)),
        Dense(200, activation="relu"),
        Dense(200, activation="relu"),
        Dense(10, activation="softmax"),
    ]
)


# %%
# Compile the model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# %%
# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=2)

# %%
# Evaluate the model
accuracy = model.evaluate(X_test, y_test, verbose=2)

# %%
# Save the model
model.save(f"models/mnist_model - {accuracy[1]:.4f}.h5")

# %%
# Show one of the images
img = X_test[random.randint(0, len(X_test))].reshape(28, 28)
img = img.reshape(1, 784)

prediction = model.predict(img)
print(prediction.argmax())

plt.imshow(img.reshape(28, 28), cmap="gray")

Image.fromarray(img.reshape(28, 28)).save("img.png")
img2 = Image.open("img.png").convert("L")
img2 = np.asarray(img2, dtype=np.uint8).reshape(1, 784)

np.equal(img, img2).all()
# %%
