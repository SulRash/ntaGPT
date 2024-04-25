""" Parameter information:
we see: 57427968, expected: 124337664, match: False
name                 params     ratio (%) 
emebedding/position      786432     1.3694
embedding/token            5376     0.0094
embedding                791808     1.3788
attention/ln                768     0.0013
attention/kqv           1769472     3.0812
attention/proj           589824     1.0271
attention               2360064     4.1096
mlp/ln                      768     0.0013
mlp/ffw                 2359296     4.1083
mlp/proj                2359296     4.1083
mlp                     4719360     8.2179
block                   7079424    12.3275
transformer            56635392    98.6199
ln_f                        768     0.0013
dense                         0     0.0000
total                  57427968   100.0000
"""

import time

out_dir = 'out-57m-ntagent-frozenlake'

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'ntagent'
wandb_run_name = 'medium-ntagent-fl-ft-r1-'+str(time.time())

init_from = 'resume'
mode = 'finetune'

dataset = 'frozenlake_pt'
gradient_accumulation_steps = 1
batch_size = 64

# medium ntaGPT model!
block_size = 1024
vocab_size = 7
n_layer = 8
n_head = 8
n_embd = 768

# Run 1, changed steps for batch size of 64 ~1 epoch:
max_iters = 12000 + 4300

# eval stuff
eval_interval = 1000
eval_iters = 50
log_interval = 10

# weight decay
weight_decay = 1e-1

# Run 1:
learning_rate = 8e-5

decay_lr = False

device = 'cuda'
compile = True

warmup_iters = 100 # not super necessary potentially

device = 'cuda'  # run on cpu only
compile = True # do not torch compile the model
