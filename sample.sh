#!/bin/bash

python sample.py \
    --out_dir="out-ntagent-frozenlake" \
    --start="SFHFHFFH<eor>
SFFFFFFF<eor>
FFFFFFFF<eor>
FFFFFHHF<eor>
HFFFFFFH<eor>
HFFFFFFF<eor>
FFFFFFFH<eor>
FHHFHFFF<eor>
FFFFFFFG<eor>
<eom>" \
    --num_samples=1 --max_new_tokens=1024
    