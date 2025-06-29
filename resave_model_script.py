from keras.models import load_model

# Load the old model
model = load_model("best_model_out.keras")

# Save it in new format (compatible with Keras 3.x and TF 2.19)
model.save("new_model.keras", save_format="keras")

print("✅ Model resaved successfully as 'new_model.keras'")
