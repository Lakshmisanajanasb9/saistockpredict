import pandas as pd 
from windo_gen import WindowGenerator
import numpy as np
import tensorflow as tf


data = pd.read_csv(r"C:\Users\brkbr\Downloads\saistockpredict\data\AAPL.csv", index_col=0)
data.index = pd.to_datetime(data.index)

column_indices = {name: i for i, name in enumerate(data.columns)}
n = len(data)
train = data[0:int(n*0.7)]
val = data[int(n*0.7):int(n*0.9)]
test = data[int(n*0.9):]

mean = train.mean()
std = train.std()

train_df =  (train - mean)/std
eval_df =  (val - mean)/std
test_df =  (test - mean)/std

window = WindowGenerator(input_width=30, label_width=1, shift=1, train_df=train_df, val_df=eval_df, test_df=test_df, label_columns=["Next_close"])
print(window.total_window_size)



single_step_window = WindowGenerator(input_width = 1, label_width=1,shift=1,train_df=train_df, val_df=eval_df, test_df=test_df,label_columns=['Next_close'])

class Baseline(tf.keras.Model):
  def __init__(self, label_index=None):
    super().__init__()
    self.label_index = label_index

  def call(self, inputs):
    if self.label_index is None:
      return inputs
    result = inputs[:, :, self.label_index]
    return result[:, :, tf.newaxis]
  
baseline = Baseline(label_index=column_indices['Next_close'])

baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                 metrics=[tf.keras.metrics.MeanAbsoluteError()])

val_performance = {}
performance = {}
val_performance['Baseline'] = baseline.evaluate(single_step_window.val, return_dict=True)
performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0, return_dict=True)

wide_window = WindowGenerator(input_width = 24,label_width = 24, shift = 1, train_df=train_df, val_df=eval_df, test_df=test_df,label_columns=['Next_close'])
wide_window

CONV_WIDTH = 3
conv_window = WindowGenerator(
    input_width=CONV_WIDTH,
    label_width=1,
    shift=1,
    train_df=train_df, val_df=eval_df, test_df=test_df,
    label_columns=['Next_close'])

#train_df.to_csv("C:\Users\brkbr\Downloads\saistockpredict\data\train_dt.csv",index=True)
#eval_df.to_csv("C:\Users\brkbr\Downloads\saistockpredict\data\eval_dt.csv",index=True)
#test_df.to_csv("C:\Users\brkbr\Downloads\saistockpredict\data\test_dt.csv",index=True)

