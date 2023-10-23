import pandas as pd
import os

def split_train_and_test(data_path):
    df = pd.read_csv(data_path,header=None)
    df = df.sample(frac=1, random_state=42)
    train_data = df.iloc[:int(0.8 * len(df))]
    test_data = df.iloc[int(0.8 * len(df)):]
    floder_path = os.path.dirname(data_path)
    train_data_path = os.path.join(floder_path,'train.data')
    test_data_path = os.path.join(floder_path,'test.data')
    train_data.to_csv(train_data_path,index=False,header=None)
    test_data.to_csv(test_data_path,index=False,header=None)