"# DogClassifier" 
This is a binary image classification project to detect whether an image contains a **dog** or **not**. The model is built using a custom CNN with TensorFlow/Keras and deployed via a Flask application.

---

## 🧠 Model Training

Due to local machine constraints, the model was built and trained on **Google Colab**. The training data consists of 100+ images per class (dog / not-dog) with preprocessing and augmentation applied.

### ⚙️ Model Optimization
- The trained `.h5` model exceeded the Render free tier memory limit.
- To reduce size, the model was converted to **TensorFlow Lite** (`.tflite`) using **float16 quantization**.

### 🚫 Render Deployment Limitation
- Even after quantization, deploying the model within the **Render free tier** led to **CPU runtime crashes**.

### ✅ Alternative Hosting (Colab + Ngrok)
- The `.h5` model is now hosted via a **Flask API on Google Colab**, exposed using **Ngrok**.
- The frontend Flask app (deployed on Render) sends image data to this Ngrok-powered backend.
- ⚠️ **Note**: Ngrok generates a new URL every time Colab restarts. To handle this, the model URL is stored in an **environment variable** (`COLAB_MODEL_URL`).
- This setup is suitable **only for demo purposes**.

---


---

## 📦 Included in Repo

- `model_training/`: Colab notebook, `.h5` model, and `.tflite` conversion
- `app.py/`: Flask UI app with image upload, `/predict` endpoint
- `dataset/`: Sample dog and not-dog images Zip file
- `requirements.txt`: Lightweight packages to run the Flask app
`,render.yaml`: for render building

---

## 🔗 Resources

- **Google Colab Notebook**: [Colab Link](#)  
- **Model Files (Drive)**: [Google Drive Folder](#)  
- **Deployed Flask UI (Render)**: [Render Public URL](#)  
- **Ngrok Model Endpoint**: Set as `COLAB_MODEL_URL` env var

> ⚠️ Note: URLs and endpoints may not work permanently due to **Colab/Ngrok session expiry**.

RenderLink - https://dogclassifier-z3xt.onrender.com/
Model_GoogleDrive-https://drive.google.com/file/d/1sCEwSqpKNJCGZoLxZuQetLv6Lm9QU4xu/view?usp=drive_link
Model_Hosting_Google_colab- https://colab.research.google.com/drive/1ZhribtsgtxE8WnIfLLxHUJ3VBPwVD4Mk?usp=sharing