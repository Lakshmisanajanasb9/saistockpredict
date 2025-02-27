import tensorflow as tf 
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from clean_data import window
import numpy as np
import matplotlib.pyplot as plt 

model = load_model(r"C:\Users\brkbr\Downloads\saistockpredict\models\lstm_model.keras")

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=2,
                                                    mode='min')

history = model.fit(
    window.train,       # Training data
    epochs=20,          # Number of epochs (iterations over dataset)
    validation_data=window.val,  # Validation data
    verbose=1   ,        # Print training progress
    callbacks=[early_stopping]
)

plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

model.save(r"C:\Users\brkbr\Downloads\saistockpredict\models\lstm_model.keras")

