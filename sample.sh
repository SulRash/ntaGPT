#!/bin/bash

python sample.py \
    --out_dir="out-ntagent-frozenlake" \
    --start="SFHFHFFH<eor>
SFFFFFFFFFFF<eor>
FFFHFFFFHFFF<eor>
FFFFFHFFFFHF<eor>
HFFFFFFFFFFH<eor>
HFFFFHFFFFFF<eor>
FFFFFFFHFFFF<eor>
FHHFHFFFFFFF<eor>
FFFFFFFFHHFF<eor>
FFFFFHHFFFFF<eor>
FFFFFFFHFFFF<eor>
FFFFFFFFFFFF<eor>
FFFFHFFHFFFG<eor>
<eom>" \
    --num_samples=1 --max_new_tokens=1024
    