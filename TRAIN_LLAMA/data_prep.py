import pandas as pd
import os
import numpy as np
from datasets import load_dataset

data_dir = "../data"
dataset_path = os.path.join(data_dir, "output1.csv")
output_name = "med_dialog.csv"
output_path = os.path.join(data_dir, output_name)


def join_non_nan(row):
    non_nan_values = [str(val) for val in row if not pd.isna(val)]
    return ' '.join(non_nan_values)

df = pd.read_csv(dataset_path)
# df.index = range(0,len(df))
df = df.apply(join_non_nan, axis = 1)

df.reset_index(drop= True, inplace=True)
odd_df = df.iloc[[i for i in range(0,df.shape[0],2)]].reset_index(drop=True)
print(odd_df[0])
even_df = df.iloc[[i for i in range(1,df.shape[0],2)]].reset_index(drop=True)

print(even_df[0])

grouped = even_df+odd_df
df = grouped.rename("text")
df = df.replace('', np.nan)
df = df.dropna()

df.to_csv(output_path, index=False)

import datasets
from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np

# load the CSV files as Dataset

dataset = load_dataset('csv', data_files=output_path)['train']
dataset = dataset.train_test_split(test_size = 0.2)
for split, data in dataset.items():
    data.to_csv(os.path.join(data_dir,f"{split}_{output_name}"))

print(dataset)


