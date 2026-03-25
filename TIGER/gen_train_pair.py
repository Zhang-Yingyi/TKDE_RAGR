import argparse
import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import sys
from typing import List

import torch
import transformers
# from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, T5Tokenizer, T5Config, T5ForConditionalGeneration

from utils import *
from collator import TestCollator
from evaluate import get_topk_results, get_metrics_results
from generation_trie import Trie


def test(args):

    set_seed(args.seed)
    print(vars(args))
    data_set = args.dataset
    save_jsonl = args.data_path+"/{}/{}_{}_results.jsonl".format(data_set,data_set,args.pair_mode)
    writer = open(save_jsonl, "w") if save_jsonl else None

    # device_map = {"": args.gpu_id}
    # device = torch.device("cuda",args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # config = T5Config.from_pretrained("t5-small")
    # tokenizer = T5Tokenizer.from_pretrained(
    #     "t5-small",
    #     model_max_length=512,
    # )
    config = T5Config.from_pretrained(args.ckpt_path)
    tokenizer = T5Tokenizer.from_pretrained(
        args.ckpt_path,
        model_max_length=512,
    )
    # train_data, valid_data = load_datasets(args)
    train_data, valid_data = load_rev_datasets(args)
    add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    config.vocab_size = len(tokenizer)

    print("add {} new token.".format(add_num))
    print("data num:", len(train_data))

    # tokenizer = T5Tokenizer.from_pretrained(args.ckpt_path)
    model = T5ForConditionalGeneration.from_pretrained(
        args.ckpt_path,
        low_cpu_mem_usage=True,
        # device_map=device,
    )
    model.to(device)
    prompt_ids = [0]

    # test_data = load_test_dataset(args)
    if args.pair_mode=='train':
        test_data = load_rev_test_dataset_pair(args)
    else:
        test_data = valid_data
    print(test_data[100])

    collator = TestCollator(args, tokenizer)
    all_items = test_data.get_all_items()


    candidate_trie = Trie(
        [
            [0] + tokenizer.encode(candidate)
            for candidate in all_items
        ]
    )
    prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)

    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collator,
                             shuffle=True, num_workers=4, pin_memory=True)


    print("data num:", len(test_data))

    model.eval()
    
    metrics = args.metrics.split(",")
    all_prompt_results = []
    with torch.no_grad():
        for prompt_id in prompt_ids:

            test_loader.dataset.set_prompt(prompt_id)
            metrics_results = {}
            total = 0

            for step, batch in enumerate(tqdm(test_loader)):
                inputs = batch[0].to(device)
                targets = batch[1]
                prompt_texts =  batch[2]
                total += len(targets)
                if step == 0:
                    print(inputs)
                    print(targets)
                    print(prompt_texts)

                output = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=10,
                    # max_length=10,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens,
                    num_beams=args.num_beams,
                    num_return_sequences=args.num_beams,
                    output_scores=True,
                    return_dict_in_generate=True,
                    early_stopping=True,
                )
                output_ids = output["sequences"]
                scores = output["sequences_scores"]

                output = tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )

                num_beams = args.num_beams
                B = len(targets)
                # 最稳：直接用 inputs 解码当 prompt（如果你能从 collator 拿 raw prompt 更好）
                # prompt_texts = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
                

                for i in range(B):
                    prompt = prompt_texts[i]
                    chosen = targets[i]

                    # 取出该样本的 num_beams 个生成结果
                    beams = output[i * num_beams : (i + 1) * num_beams]

                    # 去重（可选，但一般建议做，避免 beam 重复导致 rejected 很水）
                    seen = set()
                    beams_uniq = []
                    for b in beams:
                        if b not in seen:
                            beams_uniq.append(b)
                            seen.add(b)

                    if writer:
                        writer.write(json.dumps({
                            "prompt": prompt,
                            "chosen": chosen,
                            "rejected": beams_uniq
                        }, ensure_ascii=False) + "\n")
                if step == 0:
                    print("B:", B, "num_beams:", num_beams, "decoded:", len(prompt_texts))
                    print("sample prompt:", prompt_texts[0])
                    print("chosen:", targets[0])
                    print("rejected beams:", prompt_texts[:num_beams])

                topk_res = get_topk_results(output,scores,targets,args.num_beams,
                                            all_items=all_items if args.filter_items else None)

                batch_metrics_res = get_metrics_results(topk_res, metrics)
                # print(batch_metrics_res)

                for m, res in batch_metrics_res.items():
                    if m not in metrics_results:
                        metrics_results[m] = res
                    else:
                        metrics_results[m] += res

                # if (step+1)%10 == 0:
                temp={}
                for m in metrics_results:
                    temp[m] = metrics_results[m] / total
                # print(temp)

            for m in metrics_results:
                metrics_results[m] = metrics_results[m] / total
            all_prompt_results.append(metrics_results)
            print("======================================================")
            print("Prompt {} results: ".format(prompt_id), metrics_results)
            print("======================================================")
            print("")
    if writer:
        writer.close()

    mean_results = {}
    min_results = {}
    max_results = {}

    for m in metrics:
        all_res = [_[m] for _ in all_prompt_results]
        mean_results[m] = sum(all_res)/len(all_res)
        min_results[m] = min(all_res)
        max_results[m] = max(all_res)

    print("======================================================")
    print("Mean results: ", mean_results)
    print("Min results: ", min_results)
    print("Max results: ", max_results)
    print("======================================================")


    save_data={}
    save_data["test_prompt_ids"] = args.test_prompt_ids
    save_data["mean_results"] = mean_results
    save_data["min_results"] = min_results
    save_data["max_results"] = max_results
    save_data["all_prompt_results"] = all_prompt_results

    with open(args.results_file, "w") as f:
        json.dump(save_data, f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMRec_test")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    args = parser.parse_args()

    test(args)
