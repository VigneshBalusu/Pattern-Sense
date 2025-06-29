from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask_cors import CORS
import numpy as np
import os
import base64
import io
from PIL import Image

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Create upload folder if needed
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
try:
    model = load_model("new_model.keras")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# Define fabric pattern class labels
class_labels = [
    "argyle", "camouflage", "checked", "dot", "floral", "geometric",
    "gradient", "graphic", "houndstooth", "leopard", "lettering",
    "muji", "paisley", "snake_skin", "snow_flake", "stripe", "tropical",
    "zebra", "zigzag"
]

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

        # Save uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Load and preprocess image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

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
            image_path=filepath
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Internal Server Error: {str(e)}", 500

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
