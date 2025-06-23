from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os
import gdown
app = Flask(__name__)
MODEL_PATH = "improved_dog_classifier.h5"
GOOGLE_DRIVE_ID = "1sCEwSqpKNJCGZoLxZuQetLv6Lm9QU4xu"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}", MODEL_PATH, quiet=False)

# Load the model
model = load_model(MODEL_PATH)

def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size).convert('RGB')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    filename = None

    if request.method == "POST":
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            img = Image.open(filepath)
            img_array = preprocess_image(img)  # or (150, 150) if your model was trained on that
            pred = model.predict(img_array)
            prediction = "Dog 🐶" if pred[0][0] < 0.5 else "Not a Dog ❌"
            filename = file.filename

    return render_template("index.html", prediction=prediction, filename=filename)

# @app.route("/", methods=["GET", "POST"])
# def index():
#     prediction = None
#     if request.method == "POST":
#         file = request.files["file"]
#         if file:
#             img = Image.open(file.stream)
#             img_array = preprocess_image(img)
#             pred = model.predict(img_array)[0][0]
#             label = "Dog" if pred < 0.5 else "Not a Dog"
#             confidence = (1 - pred if pred < 0.5 else pred) * 100
#             prediction = f"{label} ({confidence:.2f}%)"
#     return render_template("index.html", prediction=prediction)
if __name__ == "__main__":
    app.run(debug=True)
