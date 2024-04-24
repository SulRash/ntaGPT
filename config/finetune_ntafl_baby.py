import time

out_dir = 'out-baby-ntagent-frozenlake'
eval_interval = 50 # keep frequent because we'll overfit
eval_iters = 20
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'ntagent'
wandb_run_name = 'baby-ntagent-finetune-fl-'+str(time.time())

init_from = 'resume'

dataset = 'frozenlake_ft'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 5e-5 # with baby networks can afford to go a bit higher
max_iters = 3500
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

# warmup_iters = 100 # not super necessary potentially

# on macbook also add
device = 'cuda'  # run on cpu only
compile = True # do not torch compile the model
