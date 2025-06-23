

from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import os
import gdown
from tflite_runtime.interpreter import Interpreter
app = Flask(__name__)

MODEL_PATH = "model_quantized.tflite"
GOOGLE_DRIVE_ID = "19advHoTFArS95e9EpNqHyR31iDEtnJdA"

# Memory optimization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    print("Downloading quantized model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}", MODEL_PATH, quiet=False)

# Load TensorFlow Lite model (NOT load_model!)
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size).convert('RGB')
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_image(img_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    
    return interpreter.get_tensor(output_details[0]['index'])

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
            img_array = preprocess_image(img)
            pred = predict_image(img_array)
            prediction = "Dog 🐶" if pred[0][0] < 0.5 else "Not a Dog ❌"
            filename = file.filename
            # Clean up
            # os.remove(filepath)

    return render_template("index.html", prediction=prediction, filename=filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Fallback to 5000 for local dev
    app.run(debug=False, host="0.0.0.0", port=port)