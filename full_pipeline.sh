#!/bin/bash

python NTA/frozenlake_gen.py \
    config/run_A.yaml

python data/frozenlake/prepare.py \
    --data_path $PWD/generated_data/pt_data.txt \
    --mode pt --output_path $PWD/data/frozenlake/

python data/frozenlake/prepare.py \
    --data_path $PWD/generated_data/ft_data.txt \
    --mode ft --output_path $PWD/data/frozenlake/

rm -rf generated_data

python train.py config/run_A.yaml pt
python train.py config/run_A.yaml ft
