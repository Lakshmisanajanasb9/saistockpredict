import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from clean_data import window


model = Sequential([
    LSTM(32,return_sequences=True,time_major=False),
    Dropout(0.2),
    Dense(units=1)
])

model.compile(optimizer="adam", loss="mean_squared_error")

history = model.fit(window.train, epochs=20, validation_data=window.val)

model.save(r"C:\Users\brkbr\Downloads\saistockpredict\models\lstm_model.keras")

MAX_EPOCHS = 20

def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history



