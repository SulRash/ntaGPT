#!/bin/bash

python NTA/frozen_lake_data.py \
    --pt_maps 1000 \
    --ft_maps 1000 \
    --map_size 12

python data/frozenlake/prepare.py \
    --data_path $PWD/generated_data/pt_data.txt \
    --mode pt --output_path $PWD/data/frozenlake/

python data/frozenlake/prepare.py \
    --data_path $PWD/generated_data/ft_data.txt \
    --mode ft --output_path $PWD/data/frozenlake/

rm -rf generated_data