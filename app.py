"""Camp Doodle: draw a camping item with the mouse and let the CNN guess it."""

import os
import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from keras.models import load_model

from config import categories

# Build paths from this file so the app works from any working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "Camp.keras")

# A big canvas makes small details easier to draw with a mouse
SIZE = 420
# On-screen pen width; preprocess() thickens strokes of large drawings
PENCIL = 8
RADIUS = PENCIL // 2
# Must match the input size the model was trained on
MODEL_SIZE = 28
# Measured on QuickDraw samples: the longest side of a sketch is ~26 of 28 px
CONTENT_RATIO = 26 / 28
# Measured strokes are ~2 px wide on a 26 px sketch, i.e. ~8% of the longest side
STROKE_RATIO = 2 / 26
FONT = ("Times New Roman", 10)
FONT_BIG = ("Times New Roman", 12)


class CampApp:
    """Tkinter window with a drawing canvas and a CNN-based guess."""

    def __init__(self,win):
        """Build the UI, load the model and set up the hidden PIL image."""
        self.win = win
        self.win.title("Camp")
        self.model = load_model(MODEL_PATH)

        # Black background with white strokes, same as QuickDraw images
        self.canvas = tk.Canvas(self.win, width=SIZE, height=SIZE, bg="black",
                                cursor="pencil", bd=0, highlightthickness=0)
        self.canvas.pack(pady=20)

        # The model must see exactly what is on screen, so draw on both
        self.image = Image.new("L",(SIZE, SIZE), 0)
        self.tool = ImageDraw.Draw(self.image)
        # Last mouse position; None means the pen is lifted
        self.lastx = None
        self.lasty = None
        
        self.msg = tk.Label(
            self.win,
            text="Think of camping and draw the first thing that comes to mind.",
            font=FONT_BIG,
        )
        self.msg.pack(pady=10)

        # A frame keeps both buttons side by side above the result label
        self.buttons = tk.Frame(self.win)
        self.buttons.pack(pady=5)
        self.button = tk.Button(
            self.buttons, text="Guess", font=FONT, command=self.predict
        )
        self.button.pack(side=tk.LEFT, padx=30)
        self.clear_button = tk.Button(
            self.buttons, text="Clear", font=FONT, command=self.clear
        )
        self.clear_button.pack(side=tk.LEFT, padx=30)

        self.result = tk.Label(self.win, text="", font=FONT_BIG, justify=tk.LEFT)
        self.result.pack(pady=10)

        self.canvas.bind("<Button-1>", self.start_stroke)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_pen)


    def start_stroke(self, event):
        """Handle a plain click so single dots are drawn too."""
        self.draw_dot(event.x, event.y)
        self.lastx, self.lasty = event.x, event.y


    def draw_dot(self, x, y):
        """Draw a round dot on both the canvas and the PIL image."""
        self.canvas.create_oval(
            x - RADIUS, y - RADIUS, x + RADIUS, y + RADIUS,
            fill="white", outline="white",
        )
        self.tool.ellipse(
            [x - RADIUS, y - RADIUS, x + RADIUS, y + RADIUS], fill=255
        )


    def paint(self, event):
        """Connect the previous mouse position to the current one."""
        x, y = event.x, event.y
        if self.lastx is not None:
            self.canvas.create_line(
                self.lastx, self.lasty, x, y,
                fill="white", width=PENCIL, capstyle=tk.ROUND,
            )
            self.tool.line([self.lastx, self.lasty, x, y], fill=255, width=PENCIL)
            # PIL lines have flat ends, so a round dot closes the gaps at joints
            self.draw_dot(x, y)
        self.lastx, self.lasty = x, y


    def reset_pen(self, _event):
        """Lift the pen so the next stroke does not connect to the last."""
        self.lastx = None
        self.lasty = None


    def clear(self):
        """Erase the canvas, the PIL image and the previous result."""
        self.canvas.delete("all")
        self.tool.rectangle([0, 0, SIZE, SIZE], fill=0)
        # Reset the pen too, otherwise the next stroke starts at the old point
        self.lastx = None
        self.lasty = None
        self.result.config(text="")


    def preprocess(self):
        """Crop, thicken, square, resize and scale the drawing like QuickDraw data.

        Returns None when the canvas is empty.
        """
        # The model was trained on centered sketches, so remove empty space first
        box = self.image.getbbox()
        if box is None:
            return None
        crop = self.image.crop(box)
        # Big drawings would shrink to faint hairlines, so scale stroke width with size
        target = int(max(crop.size) * STROKE_RATIO)
        grow = (target - PENCIL) // 2
        if grow > 0:
            # Pad first so thickening is not clipped at the crop border
            crop = ImageOps.expand(crop, border=grow, fill=0)
            crop = crop.filter(ImageFilter.MaxFilter(2 * grow + 1))
        side = int(max(crop.size) / CONTENT_RATIO)
        square = Image.new("L", (side, side), 0)
        # Centering in a square keeps the proportions instead of stretching
        square.paste(
            crop, ((side - crop.width) // 2, (side - crop.height) // 2)
        )
        # LANCZOS gives smooth, anti-aliased strokes like the training images
        small = square.resize((MODEL_SIZE, MODEL_SIZE), Image.Resampling.LANCZOS)
        # Same 0-1 scaling as in training; the model expects (batch, h, w, channel)
        data = np.asarray(small, dtype="float32") / 255.0
        return data.reshape(1, MODEL_SIZE, MODEL_SIZE, 1)

    def predict(self):
        """Show the top 3 guesses for the current drawing."""
        data = self.preprocess()
        if data is None:
            self.result.config(text="Draw something first!")
            return
        probs = self.model.predict(data, verbose=0)[0]
        # Top 3 instead of 1 because similar shapes (e.g. map/stove) get mixed up        
        top = np.argsort(probs)[::-1][:3]
        # Label order must match the order used during training (see config.py)
        lines = [f"{categories[i]}: {probs[i] * 100:.1f}%" for i in top]
        self.result.config(text="\n".join(lines))


def main():
    """Create the window and start the Tk event loop."""
    win = tk.Tk()
    CampApp(win)
    win.mainloop()


if __name__ == "__main__":
    main()
