from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask_cors import CORS
import numpy as np
import os
import base64
import io
from PIL import Image

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Upload folder (optional if needed)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = load_model("best_model_out.keras")
print("✅ Model loaded successfully")

# Define class labels
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
    try:
        data = request.get_json()
        if 'image_data_base64' not in data:
            return jsonify({'error': 'Missing image_data_base64'}), 400

        image_data = base64.b64decode(data['image_data_base64'])
        image_pil = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_resized = image_pil.resize((224, 224))
        img_array = np.array(image_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0]

        if len(prediction) != len(class_labels):
            return jsonify({'error': 'Mismatch in prediction and labels'}), 500

        top_indices = prediction.argsort()[-2:][::-1]
        top_labels = [(class_labels[i], round(prediction[i] * 100, 2)) for i in top_indices]
        confidences = {class_labels[i]: f"{round(prediction[i]*100, 2)}%" for i in range(len(class_labels))}

        return jsonify({
            "label": top_labels[0][0],
            "confidence": top_labels[0][1],
            "second_label": top_labels[1][0],
            "second_confidence": top_labels[1][1],
            "confidences": confidences
        })

    except Exception as e:
        print(f"❌ Internal Server Error: {e}")
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Important for Render
    app.run(debug=True, host='0.0.0.0', port=port)
