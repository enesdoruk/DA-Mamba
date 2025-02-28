#! /usr/bin/env bash


# CUDA_LAUNCH_BLOCKING=1 python train_ssd.py --dataset CS \
#                                             --dataset_target fogcs \
#                                             --dataset_root /AI/syndet_datasets/cityscapes_in_voc \
#                                             --dataset_target_root /AI/syndet_datasets/foggy_cityscapes_in_voc \
#                                             --batch_size 28 \
#                                             --end_epoch 100 \
#                                             --lr 1e-2 \
#                                             --wandb_name trailer \
#                                             --max_grad_norm 20.0 \



CUDA_LAUNCH_BLOCKING=1 python train_ssd.py  --batch_size 32 \
                                            --end_epoch 100 \
                                            --lr 1e-2 \
                                            --wandb_name trailer \
                                            --max_grad_norm 20.0 \