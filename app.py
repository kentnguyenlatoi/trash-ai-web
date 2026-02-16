from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

IMG_SIZE = (224, 224)

# Load model
model = tf.keras.models.load_model("trash_classifier.keras")

# Load labels
with open("labels.txt", "r", encoding="utf-8") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]

def predict_image(pil_img):
    pil_img = pil_img.convert("RGB").resize(IMG_SIZE)
    x = np.array(pil_img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.efficientnet.preprocess_input(x)

    preds = model.predict(x, verbose=0)[0]
    idx = np.argmax(preds)
    return CLASS_NAMES[idx], float(preds[idx])

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        file = request.files["image"]
        img = Image.open(file.stream)
        label, conf = predict_image(img)

        result = {
            "label": label,
            "confidence": round(conf * 100, 2)
        }

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)

    


