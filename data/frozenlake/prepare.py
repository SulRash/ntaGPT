import os
import pickle
import numpy as np

# Map-to-Token to avoid the tokenization step.
# 0, 1, and 2 are reserved for <eor>, <eom>, and \n respectively
MTT = {
    "S": "3",
    "G": "4",
    "F": "5",
    "H": "6"
}

TTM = {
    0: "<eor>",
    1: "<eom>",
    2: "\n",
    3: "S",
    4: "G",
    5: "F",
    6: "H"
}

def tokenizer(string: str):
    string = string.replace("<eor>", "0")
    string = string.replace("<eom>", "1")
    string = string.replace("\n", "2")
    string = string.replace("S", MTT["S"])
    string = string.replace("G", MTT["G"])
    string = string.replace("F", MTT["F"])
    string = string.replace("H", MTT["H"])
    return list(map(int, list(string)))

def detokenizer(ids):
    ''.join([TTM[i] for i in ids])

# Use the pretraining data
input_file_path = '/Users/sultan/Documents/Github/RLALLM/data/generated_data/pt_data.txt'

# Use the finetuning data
# input_file_path = os.path.join(os.path.dirname(__file__), '/Users/sultan/Documents/Github/RLALLM/data/generated_data/ft_data.txt')

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

vocab_size = 7
print(f"vocab size: {vocab_size:,}")

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = tokenizer(train_data)
val_ids = tokenizer(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'tokenizer': tokenizer,
    'detokenizer': detokenizer,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)