# functions/predict_fabric_pattern/main.py

import json
import os
import sys
import base64
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image # For image.load_img and image.img_to_array
from io import BytesIO
from PIL import Image # Pillow library for Image.open

# --- Global variables for model and labels (loaded once per cold start) ---
model = None
# IMPORTANT: Model path is now just the filename because 'included_files'
# places it at the root of the function's execution environment.
MODEL_PATH = 'best_model_out.keras'

# Fabric class labels - ensure these are exactly correct for your model
class_labels = [
    "argyle", "camouflage", "checked", "dot", "floral", "geometric",
    "gradient", "graphic", "houndstooth", "leopard", "lettering",
    "muji", "paisley", "snake_skin", "snow_flake", "stripe", "tropical",
    "zebra", "zigzag"
]

def load_fabric_model():
    """Loads the Keras model. This runs once per cold start of the function."""
    global model
    if model is None:
        try:
            model = load_model(MODEL_PATH)
            print(f"✅ Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            # Raising a RuntimeError will make the function invocation fail,
            # which helps in debugging via Netlify logs.
            raise RuntimeError(f"Failed to load model: {str(e)}")
    return model

# --- Netlify Function Handler ---
def handler(event, context):
    """
    Main handler for the Netlify Function.
    This function processes the incoming HTTP request.
    """
    # Netlify Functions only handle POST for API calls in this scenario
    if event['httpMethod'] == 'POST':
        try:
            # Assuming the image data (base64 encoded) is in the request body as JSON
            body_content = json.loads(event['body'])
            # The key 'image_data_base64' must match what your frontend sends.
            image_data_base64 = body_content.get('image_data_base64')

            if not image_data_base64:
                return {
                    'statusCode': 400,
                    'headers': { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
                    'body': json.dumps({'error': 'No image_data_base64 found in request body.'})
                }

            # Decode the base64 image data
            image_bytes = base64.b64decode(image_data_base64)
            img = Image.open(BytesIO(image_bytes))

            # Preprocess the image for the model - Ensure this matches your model's training
            img = img.resize((224, 224)) # Target size from your original app.py
            img_array = image.img_to_array(img) / 255.0 # Normalize pixel values if your model expects it
            img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

            # Load model and make prediction
            loaded_model = load_fabric_model()
            prediction = loaded_model.predict(img_array)[0]

            # Ensure prediction has correct size
            if len(prediction) != len(class_labels):
                return {
                    'statusCode': 500,
                    'headers': { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
                    'body': json.dumps({"error": "Model output size doesn't match number of class labels."})
                }

            # Get top 2 predictions and format for response
            top_indices = prediction.argsort()[-2:][::-1]
            top_labels = [(class_labels[i], round(prediction[i] * 100, 2)) for i in top_indices]

            # Get all class probabilities for detailed display
            confidences = {class_labels[i]: f"{round(prediction[i]*100, 2)}%" for i in range(len(class_labels))}

            # Prepare the JSON response for the frontend
            response_data = {
                'label': top_labels[0][0],
                'confidence': top_labels[0][1],
                'second_label': top_labels[1][0],
                'second_confidence': top_labels[1][1],
                'confidences': confidences,
            }

            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*' # Required for CORS if your frontend is on a different domain/port
                },
                'body': json.dumps(response_data)
            }

        except Exception as e:
            print(f"❌ Function execution error: {str(e)}") # Log error for debugging in Netlify dashboard
            return {
                'statusCode': 500,
                'headers': { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
                'body': json.dumps({'error': f"Internal server error: {str(e)}"})
            }
    else:
        # Return 405 Method Not Allowed for non-POST requests
        return {
            'statusCode': 405,
            'headers': { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
            'body': json.dumps({'error': 'Method Not Allowed. This function only accepts POST requests.'})
        }