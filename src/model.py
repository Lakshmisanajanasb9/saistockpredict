import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from clean_data import window,wide_window
import matplotlib.pyplot as plt 

model = Sequential([
    LSTM(30,return_sequences=False,time_major=False,input_shape=(30, 7)),  
    #LSTM(64, return_sequences=False),
    Dropout(0.2), 
    Dense(units=1)

])

multi_step_dense = tf.keras.Sequential([
    # Shape: (time, features) => (time*features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    tf.keras.layers.Reshape([1, -1]),
])

lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True,input_shape=(24,7)),
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
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

#history = compile_and_fit(model,window)
#plt.plot(history.history['loss'], label='Loss')
#plt.plot(history.history['val_loss'], label='Validation Loss')
#plt.legend()
#plt.show()


history_l = compile_and_fit(lstm_model, wide_window)
wide_window.plot(lstm_model)
plt.show()

#model.save(r"C:\Users\brkbr\Downloads\saistockpredict\models\lstm_model.keras")
lstm_model.save(r"C:\Users\brkbr\Downloads\saistockpredict\models\lstm_model.keras")