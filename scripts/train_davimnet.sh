#! /usr/bin/env bash


python train_davimnet.py    --dataset VOC \
                            --dataset_target clipart \
                            --batch_size 32 \
                            --batch_size_val 32 \
                            --end_epoch 100 \
                            --lr 3e-3 \
                            --size 224 \
                            --max_grad_norm 20.0 \
                            --num_workers 4 \
                            --warmup_steps 250 \
                            --weight_decay 5e-4 \
                            --momentum 0.9 \
                            --alpha 0.1 \
                            --adv 0.1 \
                            --wandb_name trailer_27_10 \