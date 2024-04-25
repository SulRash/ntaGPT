import os
import pickle
import numpy as np
from argparse import ArgumentParser

from utils.frozenlake import tokenizer

parser = ArgumentParser()
parser.add_argument('--data_path', type=str)
parser.add_argument('--mode', type=str, choices=['pt', 'ft'])
args = parser.parse_args()

input_file_path = args.data_path

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
train_ids.tofile(os.path.join(os.path.dirname(__file__), f'train_{args.mode}.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), f'val_{args.mode}.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size
}
with open(os.path.join(os.path.dirname(__file__), f'meta_{args.mode}.pkl'), 'wb') as f:
    pickle.dump(meta, f)