# %% [markdown]
# \pagebreak
# \fontsize{16}{16}\selectfont
#
# Running the Model
#
# \fontsize{12}{12}\selectfont
#
# Uses tkinter to create a drawing app that predicts the number drawn.

# %%
# Import the necessary libraries
import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
from tensorflow.python.keras.models import load_model

# %%
# Select the model to use
model_name = "mnist_model - 0.9728"


# %%
class App:
    def __init__(self, master: tk.Tk) -> None:
        """A drawing app that predicts the number drawn.

        Args:
            master (tk.Tk): The root window.
        """
        self.master = master
        master.title("Number Predictor")

        # Create the widgets
        self.create_canvas()
        self.create_buttons()
        self.create_prediction_box()

        # Load the model
        self.model = load_model(f"models/{model_name}.h5")

    def create_canvas(self) -> None:
        """Creates the canvas."""
        self.canvas = tk.Canvas(self.master, width=400, height=400, bg="black")
        self.canvas.pack()
        self.img = Image.new("RGB", (400, 400), "black")
        self.imgDraw = ImageDraw.Draw(self.img)

        # Set up canvas bindings
        self.canvas.bind("<B1-Motion>", self.draw)

    def create_buttons(self) -> None:
        """Creates the clear and predict buttons."""
        self.clear_button = tk.Button(
            self.master, text="Clear", command=self.clear_canvas
        )
        self.clear_button.pack(side="left")

        self.predict_button = tk.Button(
            self.master, text="Predict", command=self.predict
        )
        self.predict_button.pack(side="left")

    def create_prediction_box(self) -> None:
        """Creates a box to display the prediction in."""
        self.prediction_box = tk.Label(self.master, text="Prediction: None")
        self.prediction_box.pack(side="left")

    def update_prediction_text(self, num: int, confidence: float) -> None:
        """Updates the prediction text.

        Args:
            num (int): The number predicted.
            confidence (float): The confidence of the prediction.
        """
        self.prediction_box[
            "text"
        ] = f"Prediction: {num}, Confidence: {confidence * 100:.2f}%"

    def clear_canvas(self) -> None:
        """Clears the canvas."""
        self.canvas.delete("all")
        self.img = Image.new("RGB", (400, 400), "black")
        self.imgDraw = ImageDraw.Draw(self.img)

    def draw(self, event: tk.Event) -> None:
        """Draws a circle on the canvas.

        Args:
            event (tk.Event): The event that triggered the function.
        """
        r = 20
        x, y = event.x, event.y
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="")
        self.imgDraw.ellipse((x - r, y - r, x + r, y + r), fill="white")

    def img_from_canvas(self) -> Image:
        """Creates an image from the canvas and performs the following operations:
            - Grayscale
            - Resize to 28x28

        Returns:
            Image: The processed image.
        """
        self.img = self.img.convert("L")
        self.img = self.img.resize((28, 28))
        return self.img

    def img_to_array(self, img: Image) -> np.ndarray:
        """Converts an image to a numpy array. Using uint8 encoding.

        Args:
            img (Image): The image to convert.

        Returns:
            np.ndarray: The image as a numpy array.
        """
        return np.asarray(img, dtype=np.uint8).reshape(1, 28, 28, 1)

    def predict(self) -> None:
        """Predicts the number drawn on the canvas and displays it."""
        img = self.img_from_canvas()
        img_arr = self.img_to_array(img)
        prediction = self.model.predict(img_arr)
        self.update_prediction_text(np.argmax(prediction), np.max(prediction))


# %%
# Run the app
root = tk.Tk()
app = App(root)
root.mainloop()

# %%
