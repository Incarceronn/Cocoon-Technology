from datasets import load_dataset

dataset_name = "mlabonne/guanaco-llama2-1k"
dataset = load_dataset(dataset_name, split = "train")
print(dataset[2])
