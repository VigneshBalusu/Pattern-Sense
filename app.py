from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask_cors import CORS
import numpy as np
import os
import traceback
from io import BytesIO
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the trained model
try:
    model = load_model("new_model.keras")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# Fabric class labels
class_labels = [
    "argyle", "camouflage", "checked", "dot", "floral", "geometric",
    "gradient", "graphic", "houndstooth", "leopard", "lettering",
    "muji", "paisley", "snake_skin", "snow_flake", "stripe", "tropical",
    "zebra", "zigzag"
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "❌ Model not loaded", 500

    try:
        if 'file' not in request.files:
            return "❌ No file part in request", 400

        file = request.files['file']
        if file.filename == '':
            return "❌ No file selected", 400

        # Process image in memory
        img = Image.open(BytesIO(file.read())).convert("RGB")
        img = img.resize((224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)[0]

        if len(prediction) != len(class_labels):
            return "❌ Model output mismatch with class labels", 500

        top_indices = prediction.argsort()[-2:][::-1]
        top_labels = [(class_labels[i], round(prediction[i] * 100, 2)) for i in top_indices]
        confidences = {class_labels[i]: f"{round(prediction[i] * 100, 2)}%" for i in range(len(class_labels))}

        return render_template(
            'result.html',
            label=top_labels[0][0],
            confidence=top_labels[0][1],
            second_label=top_labels[1][0],
            second_confidence=top_labels[1][1],
            confidences=confidences,
            image_path=None  # No saved image path
        )

    except Exception as e:
        traceback.print_exc()
        return f"❌ Internal Server Error: {str(e)}", 500

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
a