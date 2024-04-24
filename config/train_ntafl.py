import time

out_dir = 'out-ntagent-frozenlake'
wandb_log = True
wandb_project = 'ntagent'
wandb_run_name='ntagent-fl-'+str(time.time())

dataset = 'frozenlake'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 4
block_size = 1024
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

device = 'mps'
compile = False
