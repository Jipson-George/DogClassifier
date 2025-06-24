import requests
import base64
from PIL import Image
import io
from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# 🔥 Replace with your Colab ngrok URL

COLAB_MODEL_URL = os.environ.get("COLAB_MODEL_URL", "https://ce18-35-221-130-249.ngrok-free.app")

def preprocess_image(image_file):
    """Convert uploaded image to base64"""
    image = Image.open(image_file)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to bytes
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    image_bytes = buffer.getvalue()
    
    # Encode to base64
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    return image_b64

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return render_template('index.html', prediction="No file uploaded")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction="No file selected")

        # Save uploaded file to show in template
        filename = secure_filename(file.filename)
        upload_path = os.path.join("static", "uploads", filename)
        os.makedirs(os.path.dirname(upload_path), exist_ok=True)
        file.save(upload_path)

        # Encode for model
        image_b64 = preprocess_image(open(upload_path, 'rb'))

        response = requests.post(
            f"{COLAB_MODEL_URL}/predict",
            json={'image': image_b64},
            timeout=30
        )
        print(response.status_code)
        if response.status_code == 200:
            result = response.json()
            class_names = ['Dog 🐶', 'Not a Dog ❌']
            predicted_label = class_names[result['predicted_class']]
            confidence = f"{result['confidence'] * 100:.2f}%"
            prediction_text = f"{predicted_label} ({confidence})"
            return render_template('index.html', prediction=prediction_text, filename=filename)

        else:
            return render_template('index.html', prediction="Model prediction failed", filename="no file")

    except requests.exceptions.Timeout:
        return render_template('index.html', prediction="Model server timeout ⏱️")
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")



@app.route('/health')
def health():
    try:
        # Check if Colab model is alive
        response = requests.get(f"{COLAB_MODEL_URL}/health", timeout=10)
        if response.status_code == 200:
            return jsonify({'status': 'healthy', 'model_server': 'online'})
        else:
            return jsonify({'status': 'degraded', 'model_server': 'offline'}), 503
    except:
        return jsonify({'status': 'degraded', 'model_server': 'offline'}), 503

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5001)))