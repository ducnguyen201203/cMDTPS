#!/bin/bash
DATASET_NAME="CUHK-PEDES"
#WORLD_SIZE=1  #use torch.distribution
CUDA_VISIBLE_DEVICES=0 \
python run.py --cfg /home/k64t/person-search/DAProject/UET/config_model.yml 