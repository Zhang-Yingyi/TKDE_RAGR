import argparse
import os
import sys
from typing import List
from transformers import EarlyStoppingCallback

import torch
import transformers

from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from datasets import concatenate_datasets
from datasets import Dataset as HFDataset

from modeling_letter import LETTER,TIGER,TIGER_continue, TIGER_continue_embd
import wandb
from trl import DPOTrainer, DPOConfig
from utils import *

wandb.login(key="")


def load_rev_post_datasets(args):

    tasks = args.tasks.split(",")

    train_prompt_sample_num = [int(_) for _ in "1".split(",")]
    assert len(tasks) == len(train_prompt_sample_num), "prompt sample number does not match task number"
    train_data_sample_num = [int(_) for _ in "-1".split(",")]
    assert len(tasks) == len(train_data_sample_num), "data sample number does not match task number"

    train_datasets = []
    for task, prompt_sample_num,data_sample_num in zip(tasks,train_prompt_sample_num,train_data_sample_num):
        if task.lower() == "seqrec":
            dataset = SeqRevRecDatasetPost(args, mode="train", prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)
        else:
            raise NotImplementedError
        train_datasets.append(dataset)
    
    train_data = HFDataset.from_list(train_datasets[0].inter_data)

    valid_obj = SeqRevRecDatasetPost(args, "valid", 2)
    valid_data = HFDataset.from_list(valid_obj.inter_data)

    return train_data, valid_data


def load_rev_post_test_dataset(args):

    if args.test_task.lower() == "seqrec":
        # test_data = SeqRecDataset(args, mode="test_ranking", sample_num=args.sample_num)
        test_obj = SeqRevRecDatasetPost(args, mode="test", sample_num=args.sample_num)
        test_data = HFDataset.from_list(test_obj.inter_data)
    else:
        raise NotImplementedError

    return test_data



def train(args):
    print(torch.cuda.is_available())

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    # ddp = True
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    if local_rank == 0:
        print(vars(args))

    if ddp:
        device_map = {"": local_rank}
    device = torch.device("cuda", local_rank)

    # args.base_model = "./LETTER-TIGER/ckpt/TIGER"
    config = T5Config.from_pretrained(args.sft_ckpt)
    tokenizer = T5Tokenizer.from_pretrained(
        args.sft_ckpt,
        model_max_length=512,
    )

    model = TIGER_continue.from_pretrained(args.sft_ckpt, config=config)
    ref_model = TIGER_continue.from_pretrained(args.sft_ckpt, config=config)
    model.set_hyper(args.temperature)
    model.to(device)

    train_data, valid_data = load_rev_post_datasets(args)

    config.vocab_size = len(tokenizer)


    if local_rank == 0:
        print("data num:", len(train_data))
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)
        print(train_data[100])
        print(train_data[101])
        print(valid_data[100])
        print(valid_data[101])
    if local_rank == 0:
        wandb.init(
            project="TIGER-DPO-"+args.dataset,      # 项目名称
            name=f"beta-{args.beta}", # 本次实验的名称
            config=vars(args)         # 将所有参数记录到 WandB
        )
    if local_rank == 0:
        print(model)

    dpo_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        beta=args.beta,                 
        logging_steps=args.logging_step,
        save_strategy=args.save_and_eval_strategy,
        save_steps=args.save_and_eval_steps,
        eval_strategy=args.save_and_eval_strategy, 
        eval_steps=args.save_and_eval_steps,   
        save_total_limit=2,             # 只保留最好的2个ckpt，节省硬盘空间   
        load_best_model_at_end=True,
        remove_unused_columns=False,
        report_to=["wandb"],
        fp16=args.fp16,
        bf16=args.bf16,
        seed=args.seed,
        metric_for_best_model="eval_rewards/accuracies",
        # max_prompt_length=512,
        # max_completion_length=128,  
        # max_length=640,            
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model, 
        args=dpo_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        processing_class=tokenizer,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=10)],
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    config.save_pretrained(args.output_dir)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLMRec')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)

    args = parser.parse_args()
    
    train(args)
