import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from clean_data import window,wide_window
import matplotlib.pyplot as plt 

model = Sequential([
    LSTM(30,return_sequences=False,time_major=False,input_shape=(24, 7)),  
    #LSTM(64, return_sequences=False),
    Dropout(0.2), 
    Dense(units=1)

])


MAX_EPOCHS = 20

def compile_and_fit(model, window, patience=4):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  #model.compile(loss=tf.keras.losses.MeanSquaredError(),
  #              optimizer=tf.keras.optimizers.Adam(),
  #              metrics=[tf.keras.metrics.MeanAbsoluteError()])

  model.compile(optimizer="adam", loss="mean_squared_error")

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history

history = compile_and_fit(model,wide_window)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

model.save(r"C:\Users\brkbr\Downloads\saistockpredict\models\lstm_model.keras")