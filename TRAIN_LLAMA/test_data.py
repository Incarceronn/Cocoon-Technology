import datasets
from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np 
 
# load the CSV files as Dataset
 
dataset = load_dataset('csv', data_files='../data/med_dialog.csv')['train']
dataset = dataset.train_test_split(test_size = 0.2)

print(dataset)


dataset_name = "mlabonne/guanaco-llama2-1k"
dataset = load_dataset(dataset_name, split = 'train')
print(dataset)

