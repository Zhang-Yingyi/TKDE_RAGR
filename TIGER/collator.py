import torch
import copy
import argparse
from dataclasses import dataclass

import transformers
import math
from torch.utils.data import Sampler
import torch.distributed as dist
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
# from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, T5Tokenizer, T5Config, T5ForConditionalGeneration


class Collator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.only_train_response = args.only_train_response
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        # print(self.tokenizer.model_max_length)

    def __call__(self, batch):

        input_texts = [d["input_ids"] for d in batch]
        label_texts = [d["labels"] for d in batch]

        inputs = self.tokenizer(input_texts,
                                return_tensors="pt",
                                padding="longest",
                                max_length=self.tokenizer.model_max_length,
                                truncation=True,
                                return_attention_mask=True)

        labels = self.tokenizer(label_texts,
                                return_tensors="pt",
                                padding="longest",
                                max_length=self.tokenizer.model_max_length,
                                truncation=True,
                                return_attention_mask=True)
        inputs['labels'] = labels['input_ids']
        inputs['texts'] = input_texts

        inputs['labels'][inputs['labels'] == self.tokenizer.pad_token_id] = -100

        return inputs


class Collator_qwen(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.only_train_response = args.only_train_response
        self.tokenizer = tokenizer
        
        # Qwen 模型通常没有默认的 pad_token_id，一般将其设置为 eos_token_id
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = (
                self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
            )

    def __call__(self, batch):
        # 注意：这里你的数据字典 key 叫 "input_ids" 和 "labels"，
        # 但从你原来的代码逻辑看，它们实际上存的是纯文本 (string)。
        input_texts = [d["input_ids"] for d in batch]
        label_texts = [d["labels"] for d in batch]

        batch_input_ids = []
        batch_labels = []

        for prompt, answer in zip(input_texts, label_texts):
            # 1. 分别将 prompt 和 answer 转换为 token IDs
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            # Decoder-only 必须在 answer 末尾加上 EOS token，告诉模型生成结束
            answer_tokens = self.tokenizer.encode(answer, add_special_tokens=False) + [self.tokenizer.eos_token_id]

            # 2. 拼接成完整的输入序列
            input_ids = prompt_tokens + answer_tokens

            # 3. 处理超出最大长度 (Truncation)
            max_len = self.tokenizer.model_max_length
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
                # 重新计算截断后的 prompt 和 answer 长度
                prompt_len = min(len(prompt_tokens), max_len)
                prompt_tokens = input_ids[:prompt_len]
                answer_tokens = input_ids[prompt_len:]

            # 4. 构造 Labels 
            if self.only_train_response:
                # 如果只训练 response：将 prompt 部分全部替换为 -100，让 PyTorch 忽略这部分的 Loss
                labels = [-100] * len(prompt_tokens) + answer_tokens
            else:
                # 如果要训练整段文本：labels 就是 input_ids 的完整拷贝
                labels = input_ids.copy()

            batch_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            batch_labels.append(torch.tensor(labels, dtype=torch.long))

        # 5. Padding (对齐到一个 Batch 内的最长序列)
        # 将 input_ids 补齐到相同长度，填充 pad_token_id
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        
        # 将 labels 补齐到相同长度，填充处使用 -100 避免计算 Loss
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            batch_labels, batch_first=True, padding_value=-100
        )

        # 6. 生成 Attention Mask (非 pad 的位置为 1，pad 的位置为 0)
        attention_mask = input_ids_padded.ne(self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded,
            "texts": input_texts  # 保留你原来的 texts 键以防下游有用到
        }


class TestCollator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

    def __call__(self, batch):

        input_texts = [d["input_ids"] for d in batch]
        targets = [d["labels"] for d in batch]

        inputs = self.tokenizer(
            text=input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )
        

        return (inputs, targets, input_texts)

class TestCollatorQwen(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        
        # 1. 确保有 pad_token_id (Qwen 默认可能没有)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = (
                self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
            )
            
        # 2. 【核心关键】推理阶段必须将 padding 方向改为左侧！
        # 这样才能保证 prompt 的最后一个 token 是真实的字，模型才能顺着往下生成。
        self.tokenizer.padding_side = "left"

    def __call__(self, batch):
        # 提取真实的字符串
        input_texts = [d["input_ids"] for d in batch]  # 这是 Prompt
        targets = [d["labels"] for d in batch]         # 这是真实的 Answer（仅用于评测比对，不喂给模型）

        # 仅对输入进行 tokenize
        inputs = self.tokenizer(
            text=input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )

        # 返回模型需要的 inputs，以及用于计算评估指标的 targets 和 input_texts
        return (inputs, targets, input_texts)