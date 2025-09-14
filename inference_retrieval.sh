#!/bin/bash

# if [ $# -lt 4 ]; then
#     echo "Ensemble Inference Based on Sample Retrieval"
#     echo "Usage $0 <path to model> <save path> <inference config> <dataset dir>"
#     exit 0
# fi


# example
python inference_classifier.py \
    --model_path cache/LimiX-16M.ckpt \
    --save_name result \
    --inference_config_path /root/autodl-tmp/LimiX/config/cls_default_retrieval.json \
    --data_dir /root/autodl-tmp/LimiX/datadir \
    --debug

# torchrun --nproc_per_node=8 inference_classifier.py \
    # --model_path $1 \
    # --save_name $2 \
    # --inference_config_path $3 \
    # --data_dir $4 \
    # --debug