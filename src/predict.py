import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

model = load_model(r"C:\Users\brkbr\Downloads\saistockpredict\models\lstm_model.keras")


# Example input data for prediction (adjust according to your model's input shape)
input_data = np.random.rand(1, 24, 7)   # Replace with real input data
 

# Make predictions
predictions = model.predict(input_data)
print(f"Predictions: {predictions}")