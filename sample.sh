#!/bin/bash

python sample.py \
    --out_dir="out-baby-ntagent-frozenlake" \
    --start="SHFFFF<eor>
FFHFFF<eor>
FFFFFF<eor>
FFFHFF<eor>
HFFFHH<eor>
FFFFFG<eor>
<eom>
" \
    --num_samples=5 --max_new_tokens=250
    