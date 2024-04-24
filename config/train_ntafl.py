import time

out_dir = 'out-ntagent-frozenlake'
wandb_log = True
wandb_project = 'ntagent'

# wandb_run_name='ntagent-fl-'+str(time.time())
wandb_run_name='ntagent-fl-bbr2-'+str(time.time())

# Big Boy Run 1:
# dataset = 'frozenlake_pt'
# Big Boy Run 2:
dataset = 'frozenlake_ft'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 4
block_size = 1024
gradient_accumulation_steps = 1

# Big Boy Run 1:
# 18237 is one epoch for 74,700,000 tokens of training data
# max_iters = 18237
# lr_decay_iters = 18237

# Big Boy Run 2:
max_iters = 6858 * 2
lr_decay_iters = 18237 * 2

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

device = 'cuda'
compile = True
