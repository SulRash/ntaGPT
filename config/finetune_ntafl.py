import time

out_dir = 'out-ntagent-frozenlake'
wandb_log = True
wandb_project = 'ntagent'
wandb_run_name='ntagent-fl-'+str(time.time())

dataset = 'frozenlake_pt'
init_from="resume"

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 4
block_size = 1024
gradient_accumulation_steps = 1

# 6858 is one epoch for 28,090,488 tokens of training data
max_iters = 6858

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False

# weight decay
weight_decay = 1e-1

device = 'cuda'
compile = True
