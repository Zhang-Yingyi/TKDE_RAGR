#!/bin/bash
#SBATCH --job-name=RAGR_job            # 作业名称-请修改为自己任务名字 
#SBATCH --cpus-per-task=8             # 每个任务使用的CPU核心数
#SBATCH --mem=100G                      # 申请100GB内存
#SBATCH --time=18:00:00               # 运行时间限制，格式为hh:mm:ss
#SBATCH --output=./log/output_beauty_%j.txt        # 标准输出文件名 (%j 表示作业ID)-请修改为自己路径
#SBATCH --error=./log/error_beauty_%j.txt          # 标准错误文件名-请修改为自己路径

###################################
####### step0 Process data #######
###################################

# code at ./data_process/data_preprocess_amazon.ipynb

###################################
####### step1 gen embedding #######
###################################
export CUDA_VISIBLE_DEVICES=0

python -u ./data_process/amazon_text_emb.py \
 --dataset Beauty \
 --root ./data \
 --plm_name t5 \
 --plm_checkpoint sentence-t5-base

python -u ./data_process/amazon_text_emb_rev.py \
 --dataset Beauty \
 --root ./data \
 --plm_name t5 \
 --plm_checkpoint sentence-t5-base


###################################
######### step2 train rqvae #######
###################################

python ./RQ-VAE/main.py \
  --device cuda:0 \
  --data_path ./data/Beauty/Beauty.emb-t5-td.npy \
  --rev_data_path  ./data/Beauty/Beauty.emb-t5-td_rev.npy \
  --alpha 0.01 \
  --beta 0.0001 \
  --ckpt_dir ./RQ-VAE/tiger_checkpoint/


###################################
###### step3 gen semantic id ######
###################################


python ./RQ-VAE/generate_indices.py\
    --dataset Beauty \
    --root_path ./RQ-VAE/tiger_checkpoint/ \
    --alpha 1e-1 \
    --beta 1e-4 \
    --epoch 2000 \
    --checkpoint ./RQ-VAE/tiger_checkpoint/xxx.pth

python ./RQ-VAE/generate_indices_rev.py\
    --dataset Beauty \
    --root_path ./RQ-VAE/tiger_checkpoint/ \
    --data_path ./data/Beauty/Beauty.emb-t5-td_rev.npy \
    --alpha 1e-1 \
    --beta 1e-4 \
    --epoch 2000 \
    --checkpoint ./RQ-VAE/tiger_checkpoint/xxx.pth

##################################
##### step4 train generative######
##################################
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1
DATASET=Beauty
OUTPUT_DIR=./ckpt_rev/$DATASET+RAGR/

torchrun --nproc_per_node=2 --master_port=2314 ./TIGER/finetune_continue.py \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --per_device_batch_size 256 \
    --learning_rate 5e-4 \
    --epochs 200 \
    --index_file .index.json \
    --rev_file .review.json \
    --temperature 1.0

###############################
#### step5 Evaluation #########
###############################
export CUDA_VISIBLE_DEVICES=0
DATASET=Beauty
DATA_PATH=./data
OUTPUT_DIR=./ckpt_rev/$DATASET+RAGR/
RESULTS_FILE=./results/$DATASET+RAGR/beauty_tiger.json
CKPT_PATH=./ckpt_rev/$DATASET+RAGR/

python ./LETTER-TIGER/test.py \
    --ckpt_path $CKPT_PATH \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file $RESULTS_FILE \
    --test_batch_size 32 \
    --num_beams 20 \
    --test_prompt_ids 0 \
    --index_file .index.json \
    --rev_file .review.json 



#################################
######## step6 train DPO ########
#################################
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1
DATASET=Beauty
DPO=$DATASET+DPO
OUTPUT_DIR=./ckpt_rev/$DPO/
SFT_CKPT=./ckpt_rev/$DATASET+RAGR/

torchrun --nproc_per_node=2 --master_port=2314 ./TIGER/finetune_dpo.py \
    --sft_ckpt $SFT_CKPT \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --per_device_batch_size 256 \
    --learning_rate 1e-6 \
    --epochs 5 \
    --index_file .index.json \
    --rev_file .review.json \
    --temperature 1.0 \
    --save_and_eval_steps 200 \
    --beta 0.7 \
    --neg_k 5

################################
##### step7 Evaluation #########
################################
export CUDA_VISIBLE_DEVICES=0
DATASET=Beauty
DATA_PATH=./data
OUTPUT_DIR=./ckpt_rev/$DATASET/
RESULTS_FILE=./results/$DATASET/beauty_tiger.json
DPO=$DATASET+DPO
CKPT_PATH=./ckpt_rev/$DPO/

python -u ./LETTER-TIGER/test.py \
    --ckpt_path $CKPT_PATH \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file $RESULTS_FILE \
    --test_batch_size 32 \
    --num_beams 20 \
    --test_prompt_ids 0 \
    --index_file .index.json \
    --rev_file .review.json 