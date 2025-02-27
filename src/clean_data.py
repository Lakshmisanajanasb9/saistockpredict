import pandas as pd 

data = pd.read_csv("C:\Users\brkbr\Downloads\saistockpredict\data\AAPL.csv", index_col=0)
data.index = pd.to_datetime(data.index)

column_indices = {name: i for i, name in enumerate(data.columns)}
n = len(data)
train = data[0:int(n*0.7)]
val = data[int(n*0.7):int(n*0.9)]
test = data[int(n*0.9):]

mean = train.mean()
std = train.std()

train_df = (train - mean)/std
eval_df = (val - mean)/std
test_df = (test - mean)/std

window = WindowGenerator(input_width=60, label_width=1, shift=1, train_df=train_df, val_df=eval_df, test_df=test_df, label_columns=["Close"])


#train_df.to_csv("C:\Users\brkbr\Downloads\saistockpredict\data\train_dt.csv",index=True)
#eval_df.to_csv("C:\Users\brkbr\Downloads\saistockpredict\data\eval_dt.csv",index=True)
#test_df.to_csv("C:\Users\brkbr\Downloads\saistockpredict\data\test_dt.csv",index=True)
