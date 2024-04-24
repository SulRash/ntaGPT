import time

out_dir = 'out-ntagent-frozenlake'
wandb_log = True
wandb_project = 'ntagent'
wandb_run_name='ntagent-fl-ft-bbr3'+str(time.time())

dataset = 'frozenlake_ft'
init_from="resume"

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 4
block_size = 1024
gradient_accumulation_steps = 1

# 6858 is one epoch for 28,090,488 tokens of training data
# Big Boy Run 1:
# max_iters = 6858 + 18000
# Big Boy Run 3:
# max_iters = 6858 + 19000
# Big Boy Run 4:
max_iters = 68580 + 190000

# eval stuff
eval_interval = 1000
eval_iters = 500
log_interval = 10

# finetune at constant LR
# Big Boy Run 1:
# learning_rate = 3e-5
# Big Boy Run 3 and 4:
learning_rate = 6e-5

decay_lr = False

# weight decay
weight_decay = 1e-1

device = 'cuda'
compile = True
