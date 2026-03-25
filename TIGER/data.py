import copy
import random
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict
import torch.distributed as dist
import logging
import re
import pdb
import json
import numpy as np
from transformers import T5Tokenizer
from collections import defaultdict


class BaseDataset(Dataset):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.dataset = args.dataset
        self.data_path = os.path.join(args.data_path, self.dataset)

        self.max_his_len = args.max_his_len
        self.his_sep = args.his_sep
        self.index_file = args.index_file
        self.rev_file = args.rev_file
        self.add_prefix = args.add_prefix

        self.new_tokens = None
        self.allowed_tokens = None
        self.all_items = None


    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)

    def get_new_tokens(self):

        if self.new_tokens is not None:
            return self.new_tokens

        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))

        return self.new_tokens

    def get_all_items(self):

        if self.all_items is not None:
            return self.all_items

        self.all_items = set()
        for index in self.indices.values():
            self.all_items.add("".join(index))

        return self.all_items

    def get_all_items_v2(self):
        if self.all_items is not None:
            return self.all_items

        self.all_items = []
        for index in self.indices.values():
            self.all_items.append("".join(index))

        return self.all_items       
    def get_prefix_allowed_tokens_fn(self, tokenizer):


        if self.allowed_tokens is None:
            self.allowed_tokens = {}
            for index in self.indices.values():
                for i, token in enumerate(index):
                    token_id = tokenizer(token)["input_ids"][0]
                    if i not in self.allowed_tokens.keys():
                        self.allowed_tokens[i] = set()
                    self.allowed_tokens[i].add(token_id)
            self.allowed_tokens[len(self.allowed_tokens.keys())] = set([tokenizer.eos_token_id])
        sep = [0]


        def prefix_allowed_tokens_fn(batch_id, sentence):
            sentence = sentence.tolist()
            reversed_sent = sentence[::-1]
            for i in range(len(reversed_sent)):
                if reversed_sent[i:i + len(sep)] == sep[::-1]:
                    # print(list(self.allowed_tokens[i]))
                    return list(self.allowed_tokens[i])

        return prefix_allowed_tokens_fn

    def _process_data(self):

        raise NotImplementedError



class SeqRecDataset(BaseDataset):
        
    def __init__(self, args, mode="train",
                 prompt_sample_num=1, prompt_id=0, sample_num=-1):
        super().__init__(args)

        self.mode = mode
        self.prompt_id = prompt_id
        self.sample_num = sample_num


        # load data
        self._load_data()
        self._remap_items()
        
        # load data
        if self.mode == 'train':
            self.inter_data = self._process_train_data()
        elif self.mode == 'valid':
            self.inter_data = self._process_valid_data()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data()
        elif self.mode == 'test_ranking':
            self.inter_data = self._process_test_data_ids()
        else:
            raise NotImplementedError



    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)

    def _remap_items(self):

        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = new_items


    def _process_train_data(self):

        inter_data = []
        for uid  in self.remapped_inters:
            items = self.remapped_inters[uid][:-2]
            for i in range(1, len(items)):
                one_data = dict()
                # one_data["user"] = uid
                one_data["item"] = items[i]
                history = items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                if self.add_prefix:
                    history = [str(k+1) + ". " + item_idx for k, item_idx in enumerate(history)]
                one_data["inters"] = "".join(history)
                inter_data.append(one_data)

        return inter_data
    
    def _process_valid_data(self):

        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-2]
            history = items[:-2]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = "".join(history)
            inter_data.append(one_data)

        return inter_data

    def _process_test_data(self):

        inter_data = []
        for uid in self.remapped_inters:
            # if uid not in cold_user:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-1]
            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = "".join(history)
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            # print(sample_idx[:10])##################
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data
    
    def _process_test_data_ids(self):

        inter_data = []
        for uid in self.inters:
            # if uid not in cold_user:
            items = self.inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-1]
            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = history
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            # print(sample_idx[:10])##################
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data       
    

    def set_prompt(self, prompt_id):

        self.prompt_id = prompt_id

    def __len__(self):

        return len(self.inter_data)

    def __getitem__(self, index):


        d = self.inter_data[index]

        return dict(input_ids=d["inters"], labels=d["item"])


class SeqRevRecDataset(BaseDataset):
        
    def __init__(self, args, mode="train",
                 prompt_sample_num=1, prompt_id=0, sample_num=-1):
        super().__init__(args)

        self.mode = mode
        self.prompt_id = prompt_id
        self.sample_num = sample_num


        # load data
        self._load_data()
        # self._remap_items()
        
        # load data
        if self.mode == 'train':
            self._remap_items()
            self.inter_data = self._process_train_data()
            # self.inter_data = self._process_train_data_v1()
        elif self.mode == 'valid':
            self._remap_items_test()
            self.inter_data = self._process_valid_data()
        elif self.mode == 'test':
            self._remap_items_test()
            self.inter_data = self._process_test_data()
        elif self.mode == 'test_ranking':
            self.inter_data = self._process_test_data_ids()
        else:
            raise NotImplementedError



    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.rev_file), 'r') as f:
            self.rev_indices = json.load(f)

    def _remap_items(self):
        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items[0]]
            new_revs = ["".join(self.rev_indices[str(i)]) for i in items[1]]
            new_revs_v2 = []
            for revs in new_revs:
                new_revs_v2.append(revs)
            res = []
            for i in range(len(new_items)):
                res.append(new_items[i])
                res.append(new_revs_v2[i])
            self.remapped_inters[uid] = res
    
    
    def _remap_items_test(self):

        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items[0]]
            new_revs = ["".join(self.rev_indices[str(i)]) for i in items[1]]
            new_revs_v2 = []
            for revs in new_revs:
                new_revs_v2.append(revs)
            res = []
            for i in range(len(new_items)):
                # res.append(new_items[i] + new_revs_v2[i])
                res.append(new_items[i])
                res.append(new_revs_v2[i])
            self.remapped_inters[uid] = res


    def _process_train_data(self):

        inter_data = []
        for uid  in self.remapped_inters:
            items = self.remapped_inters[uid][:-4]
            for i in range(1, len(items)):
                one_data = dict()
                # one_data["user"] = uid
                one_data["item"] = items[i]
                history = items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                if self.add_prefix:
                    history = [str(k+1) + ". " + item_idx for k, item_idx in enumerate(history)]
                one_data["inters"] = "".join(history)
                inter_data.append(one_data)

        return inter_data

    def _process_train_data_v1(self):
        inter_data = []
        for uid  in self.remapped_inters:
            items = self.remapped_inters[uid][:-4]
            for i in range(2, len(items),2):
                one_data = dict()
                # one_data["user"] = uid
                one_data["item"] = items[i]
                history = items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                if self.add_prefix:
                    history = [str(k+1) + ". " + item_idx for k, item_idx in enumerate(history)]
                one_data["inters"] = "".join(history)
                inter_data.append(one_data)
        return inter_data
    
    def _process_valid_data(self):

        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-4]
            history = items[:-4]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = "".join(history)

            inter_data.append(one_data)

        return inter_data

    def _process_test_data(self):

        inter_data = []
        for uid in self.remapped_inters:
            # if uid not in cold_user:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-2]
            history = items[:-2]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = "".join(history)

            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            # print(sample_idx[:10])##################
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data
    
    def _process_test_data_ids(self):

        inter_data = []
        for uid in self.inters:
            # if uid not in cold_user:
            items = self.inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-1]
            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = history
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            # print(sample_idx[:10])##################
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data     

    def get_new_tokens(self):

        if self.new_tokens is not None:
            return self.new_tokens

        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        for index in self.rev_indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))

        return self.new_tokens  
    
    def get_all_items(self):

        if self.all_items is not None:
            return self.all_items

        self.all_items = set()
        for index in self.indices.values():
            self.all_items.add("".join(index))

        for index in self.rev_indices.values():
            self.all_items.add("".join(index)) 
        return self.all_items

    def set_prompt(self, prompt_id):

        self.prompt_id = prompt_id

    def __len__(self):

        return len(self.inter_data)

    def __getitem__(self, index):


        d = self.inter_data[index]


        return dict(input_ids=d["inters"], labels=d["item"])



class SeqRevRecDataset_tiger(BaseDataset):
        
    def __init__(self, args, mode="train",
                 prompt_sample_num=1, prompt_id=0, sample_num=-1):
        super().__init__(args)

        self.mode = mode
        self.prompt_id = prompt_id
        self.sample_num = sample_num


        # load data
        self._load_data()
        # self._remap_items()
        
        # load data
        if self.mode == 'train':
            self._remap_items()
            self.inter_data = self._process_train_data()
        elif self.mode == 'valid':
            self._remap_items_test()
            self.inter_data = self._process_valid_data()
        elif self.mode == 'test':
            self._remap_items_test()
            self.inter_data = self._process_test_data()
        elif self.mode == 'test_ranking':
            self.inter_data = self._process_test_data_ids()
        else:
            raise NotImplementedError



    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.rev_file), 'r') as f:
            self.rev_indices = json.load(f)

    def _remap_items(self):
        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items[0]]
            new_revs = ["".join(self.rev_indices[str(i)]) for i in items[1]]
            new_revs_v2 = []
            for revs in new_revs:
                new_revs_v2.append(revs)
            res = []
            for i in range(len(new_items)):
                res.append(new_items[i])
                # res.append(new_revs_v2[i])
            self.remapped_inters[uid] = res
    
    
    def _remap_items_test(self):

        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items[0]]
            new_revs = ["".join(self.rev_indices[str(i)]) for i in items[1]]
            new_revs_v2 = []
            for revs in new_revs:
                new_revs_v2.append(revs)
            res = []
            for i in range(len(new_items)):
                res.append(new_items[i])
                # res.append(new_revs_v2[i])
            self.remapped_inters[uid] = res

    def _keep_item(self, item):
        pattern = r'(<a_\d+><b_\d+><c_\d+><d_\d+>)'
        match = re.search(pattern, item)
        if match:
            clean_label = match.group(1)
        else:
            clean_label = item
        return clean_label
    
    def _process_train_data(self):

        inter_data = []
        for uid  in self.remapped_inters:
            items = self.remapped_inters[uid][:-2]
            for i in range(1, len(items)):
                one_data = dict()
                # one_data["user"] = uid
                one_data["item"] = self._keep_item(items[i])
                history = items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                if self.add_prefix:
                    history = [str(k+1) + ". " + item_idx for k, item_idx in enumerate(history)]
                one_data["inters"] = "".join(history)
                inter_data.append(one_data)

        return inter_data
    
    def _process_valid_data(self):

        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = self._keep_item(items[-2])
            history = items[:-2]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = "".join(history)

            inter_data.append(one_data)

        return inter_data

    def _process_test_data(self):

        inter_data = []
        for uid in self.remapped_inters:
            # if uid not in cold_user:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = self._keep_item(items[-1])
            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = "".join(history)

            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            # print(sample_idx[:10])##################
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data
    
    def _process_test_data_ids(self):

        inter_data = []
        for uid in self.inters:
            # if uid not in cold_user:
            items = self.inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-1]
            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = history
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            # print(sample_idx[:10])##################
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data     

    def get_new_tokens(self):

        if self.new_tokens is not None:
            return self.new_tokens

        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        for index in self.rev_indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))

        return self.new_tokens  
    
    def get_all_items(self):

        if self.all_items is not None:
            return self.all_items

        self.all_items = set()
        for index in self.indices.values():
            self.all_items.add("".join(index))

        for index in self.rev_indices.values():
            self.all_items.add("".join(index)) 
        return self.all_items

    def set_prompt(self, prompt_id):

        self.prompt_id = prompt_id

    def __len__(self):

        return len(self.inter_data)

    def __getitem__(self, index):


        d = self.inter_data[index]


        return dict(input_ids=d["inters"], labels=d["item"])




class SeqRevRecDatasetPost(BaseDataset):
        
    def __init__(self, args, mode="train",
                 prompt_sample_num=1, prompt_id=0, sample_num=-1):
        super().__init__(args)

        self.mode = mode
        self.prompt_id = prompt_id
        self.sample_num = sample_num
        self.neg_k = args.neg_k


        # load data
        self._load_data()
        # self._remap_items()
        self.all_item_ids = ["".join(self.indices[str(i)]) for i in self.indices]
        self.all_rev_ids = ["".join(self.rev_indices[str(i)]) for i in self.rev_indices]
        
        # load data
        if self.mode == 'train':
            self._remap_items()
            self.inter_data = self._process_train_data()
        elif self.mode == 'valid':
            self._remap_items_test()
            self.inter_data = self._process_valid_data()
        elif self.mode == 'test':
            self._remap_items_test()
            self.inter_data = self._process_test_data()
        elif self.mode == 'test_ranking':
            self.inter_data = self._process_test_data_ids()
        else:
            raise NotImplementedError

        self._build_prefix_maps()

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + ".review.json"), 'r') as f:
            self.rev_indices = json.load(f)

    def _remap_items(self):

        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items[0]]
            new_revs = ["".join(self.rev_indices[str(i)]) for i in items[1]]
            new_revs_v2 = []
            for revs in new_revs:
                new_revs_v2.append(revs)
            res = []
            for i in range(len(new_items)):
                res.append(new_items[i])
                res.append(new_revs_v2[i])
            self.remapped_inters[uid] = res
    
    def _remap_items_test(self):

        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items[0]]
            new_revs = ["".join(self.rev_indices[str(i)]) for i in items[1]]
            new_revs_v2 = []
            for revs in new_revs:
                new_revs_v2.append(revs)
            res = []
            for i in range(len(new_items)):
                # res.append(new_items[i]+new_revs_v2[i])
                res.append(new_items[i])
                res.append(new_revs_v2[i])
            self.remapped_inters[uid] = res

    def _keep_item(self, item):
        pattern = r'(<a_\d+><b_\d+><c_\d+><d_\d+>)'
        match = re.search(pattern, item)
        if match:
            clean_label = match.group(1)
        else:
            clean_label = item
        return clean_label

    def _keep_item_tokens(self, item):
        # 把 <a_..><b_..><c_..><d_..> 拆成 4 个分组
        pattern = r'(<a_\d+>)(<b_\d+>)(<c_\d+>)(<d_\d+>)'
        match = re.search(pattern, item)
        if match:
            a_tok, b_tok, c_tok, d_tok = match.groups()
            return [a_tok, b_tok, c_tok, d_tok]
        else:
            return []  # 或者返回 None 看你下游怎么处理

    def _split_item_review(self, s: str):
        # 从一个字符串里找出所有形如 <a_><b_><c_><d_> 的四元组片段
        segs = re.findall(r'(<a_\d+><b_\d+><c_\d+><d_\d+>)', s)
        if len(segs) >= 2:
            return segs[0], segs[1]   # itemSID, reviewSID
        elif len(segs) == 1:
            return segs[0], ""        # 只有 itemSID
        else:
            return "", ""             # 异常样本

    def _sample_item_negative(self, pos_item_sid: str, history_sids: set):
        # 负例：全库 itemSID 中采样，但排除正例与历史出现过的 item
        # 注意：self.all_item_ids 本身就是 itemSID 串（你已构造）
        while True:
            neg = random.choice(self.all_item_ids)
            if (neg != pos_item_sid) and (neg not in history_sids):
                return neg
            
    def _split_item_review(self, s: str):
        segs = re.findall(r'(<a_\d+><b_\d+><c_\d+><d_\d+>)', s)
        if len(segs) >= 2:
            return segs[0], segs[1]
        elif len(segs) == 1:
            return segs[0], ""
        return "", ""

    def _sid_to_vec4(self, sid: str):
        # 把 <a_i><b_j><c_k><d_l> -> [i,j,k,l]
        if not sid:
            return np.array([-1, -1, -1, -1], dtype=np.float32)

        def get(tag):
            m = re.search(rf"<{tag}_(\d+)>", sid)
            return int(m.group(1)) if m else -1

        return np.array([get("a"), get("b"), get("c"), get("d")], dtype=np.float32)

    
    def _sid_to_tokens4(self, sid: str):
        m = re.fullmatch(r'(<a_\d+>)(<b_\d+>)(<c_\d+>)(<d_\d+>)', sid)
        return m.groups() if m else None
    
    def _build_prefix_maps(self):
        self.map3 = defaultdict(list)  # (a,b,c) -> [itemSID]
        self.map2 = defaultdict(list)  # (a,b) -> ...
        self.map1 = defaultdict(list)  # (a,) -> ...
        for it in self.all_item_ids:
            toks = self._sid_to_tokens4(it)
            if toks is None: 
                continue
            a,b,c,d = toks
            self.map3[(a,b,c)].append(it)
            self.map2[(a,b)].append(it)
            self.map1[(a,)].append(it)

    def _sample_hard_item_negative(self, pos_item, history_item_sids, p3=0.5, p2=0.3, p1=0.15):
        toks = self._sid_to_tokens4(pos_item)
        if toks is None:
            return self._sample_item_negative(pos_item, history_item_sids)

        a,b,c,d = toks
        r = random.random()

        cand = None
        if r < p3:
            pool = self.map3.get((a,b,c), [])
            cand = random.choice(pool) if pool else None
        elif r < p3 + p2:
            pool = self.map2.get((a,b), [])
            cand = random.choice(pool) if pool else None
        elif r < p3 + p2 + p1:
            pool = self.map1.get((a,), [])
            cand = random.choice(pool) if pool else None

        # 过滤掉 pos / history，不行就 fallback random
        if (cand is None) or (cand == pos_item) or (cand in history_item_sids):
            return self._sample_item_negative(pos_item, history_item_sids)
        return cand
    
    def _build_item_review_centroid(self):
        """
        为每个 itemSID 构建一个“典型 review 向量”（centroid）。
        用你现成的 inter.json: items[0], items[1] 对齐的 (item_id, review_id)
        """
        bucket = {}  # itemSID -> list(vec4)
        for uid, pair in self.inters.items():
            item_ids = pair[0]
            rev_ids = pair[1]
            for it, rv in zip(item_ids, rev_ids):
                item_sid = "".join(self.indices[str(it)])
                rev_sid = "".join(self.rev_indices[str(rv)])
                bucket.setdefault(item_sid, []).append(self._sid_to_vec4(rev_sid))
        self._item_sid_list = []
        mat = []
        for item_sid, vecs in bucket.items():
            self._item_sid_list.append(item_sid)
            mat.append(np.mean(np.stack(vecs, axis=0), axis=0))
        self._item_review_mat = np.stack(mat, axis=0).astype(np.float32)  # (N,4)

    def _build_item_item_centroid(self):
        # 每个 item 的 itemSID -> vec4
        self._item_sid_list = list(self.all_item_ids)  # 确保全库 itemSID 顺序固定
        self._item_item_mat = np.stack([self._sid_to_vec4(sid) for sid in self._item_sid_list], axis=0).astype(np.float32)

    def _sample_review_guided_hard_item(self, review_sid: str, pos_item_sid: str, history_item_sids: set, topk=50):
        """
        用 review 向量在 item_review_mat 中找近邻 item，采 hard negative。
        """
        if not hasattr(self, "_item_review_mat"):
            self._build_item_review_centroid()

        q = self._sid_to_vec4(review_sid)  # (4,)
        diff = self._item_review_mat - q[None, :]
        dist = np.sum(diff * diff, axis=1)   # L2

        nn_idx = np.argsort(dist)[:topk]
        for idx in nn_idx:
            cand = self._item_sid_list[int(idx)]
            if cand != pos_item_sid and cand not in history_item_sids:
                return cand

        # fallback：找不到就退回 random（仍排除历史/正例）
        while True:
            neg = random.choice(self.all_item_ids)
            if neg != pos_item_sid and neg not in history_item_sids:
                return neg

    def _share_prefix(self, item_sid1: str, item_sid2: str, m: int = 2) -> bool:
        # m=1: share <a_*>; m=2: share <a_*><b_*>
        t1 = self._keep_item_tokens(item_sid1)
        t2 = self._keep_item_tokens(item_sid2)
        if len(t1) < m or len(t2) < m:
            return False
        return t1[:m] == t2[:m]

    def _retrieve_candidates_by_review(self, review_sid: str, topk: int = 200):
        # 你已有 _sample_review_guided_hard_item 的距离计算逻辑
        # 这里改成返回排序后的候选列表，而不是只返回一个
        if not hasattr(self, "_item_review_mat"):
            self._build_item_review_centroid()
        q = self._sid_to_vec4(review_sid)
        diff = self._item_review_mat - q[None, :]
        dist = np.sum(diff * diff, axis=1)
        idx = np.argsort(dist)[:topk]
        return [self._item_sid_list[int(i)] for i in idx]
    
    def _retrieve_candidates_by_context(self,
                                   last_item_sid: str,
                                   last_review_sid: str,
                                   topk: int = 200,
                                   alpha: float = 0.2,
                                   pool_mul: int = 10):
        """
        用 joint query 检索候选 item：
        dist = alpha * dist(review_centroid, q_rev) + (1-alpha) * dist(item_vec, q_item)

        pool_mul: 先取 topk*pool_mul 的粗候选，再按 joint dist 重排，避免 topk 太小造成过滤后为空
        """
        if not hasattr(self, "_item_review_mat"):
            self._build_item_review_centroid()
        if not hasattr(self, "_item_item_mat"):
            self._build_item_item_centroid()

        q_rev = self._sid_to_vec4(last_review_sid).astype(np.float32)
        q_item = self._sid_to_vec4(last_item_sid).astype(np.float32)

        # --- L2 距离（向量化）---
        # review distance
        diff_r = self._item_review_mat - q_rev[None, :]
        dist_r = np.sum(diff_r * diff_r, axis=1)

        # 先用 review 粗召回更大的候选池（更贴你“用反馈引导决策”的叙事）
        pool_k = min(len(dist_r), max(topk * pool_mul, topk))
        idx_pool = np.argpartition(dist_r, pool_k - 1)[:pool_k]  # 比 argsort 快很多

        # item distance（只算池内）
        diff_i = self._item_item_mat[idx_pool] - q_item[None, :]
        dist_i = np.sum(diff_i * diff_i, axis=1)

        # joint distance
        dist_joint = alpha * dist_r[idx_pool] + (1.0 - alpha) * dist_i

        # 取 joint topk
        order = np.argsort(dist_joint)[:topk]
        idx = idx_pool[order]

        return [self._item_sid_list[int(i)] for i in idx]
    
    def _process_train_data_new(self):
        inter_data = []

        # 每个正例配几个负例（建议 2~4）
        n_neg = getattr(self.args, "n_neg", 5)

        for uid in self.remapped_inters:
            seq = self.remapped_inters[uid][:-2]   # 你坚持这个我尊重
            if len(seq) < 2:
                continue
            # 关键：别只用最后一步，尽量用多步增加样本（你要“阔数据”）
            # 如果你只想靠近 tail，也可以用 range(max(1, len(seq)-K), len(seq))
            for t in range(1, len(seq)):
                ori = seq[t]
                pos_item, pos_rev = self._split_item_review(ori)
                if not pos_item:
                    continue

                history = seq[:t]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                if self.add_prefix:
                    history = [str(k + 1) + ". " + x for k, x in enumerate(history)]

                # ★ action token：把“此处要决策 item”写死
                prompt = "".join(history)

                # history itemSID 集合，用于排除伪负例
                history_item_sids = set()
                for h in history:
                    h_item, _ = self._split_item_review(h)
                    if h_item:
                        history_item_sids.add(h_item)

                # multi-negative：1 个 hard + (n_neg-1) 个 random（或 hard-2/hard-1）
                negs = []
                negs.append(self._sample_hard_item_negative(pos_item, history_item_sids))
                for _ in range(n_neg - 1):
                    negs.append(self._sample_item_negative(pos_item, history_item_sids))

                for neg in negs:
                    inter_data.append({"prompt": prompt, "chosen": pos_item, "rejected": neg})
        return inter_data
    
    def _process_train_data_v4(self):
        inter_data = []

        topk = int(getattr(self.args, "hard_topk", 10))        # 建议 >=200
        K = int(getattr(self.args, "num_neg", 1))               # 总负例数
        hard_n = int(getattr(self.args, "hard_n", 1))           # hard 负例数
        prefix_m0 = int(getattr(self.args, "neg_prefix_m", 2))  # 建议 2 或 3（别用4）

        # 只用最后一步你自己坚持，我就不改；想更“阔数据”就把这行换成 tail_k
        for uid in self.remapped_inters:
            seq = self.remapped_inters[uid][:-2]
            # if len(seq) < 2:
            #     continue

            # for t in range(len(seq)-1, len(seq)):
            for t in range(1, len(seq)):
                pos_item, pos_rev = self._split_item_review(seq[t])
                if (not pos_item) or (not pos_rev):
                    continue

                history = seq[:t]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                if self.add_prefix:
                    history = [str(k + 1) + ". " + x for k, x in enumerate(history)]
                prompt = "".join(history)

                # history itemSID set
                history_item_sids = set()
                for h in history:
                    h_item, _ = self._split_item_review(h)
                    if h_item:
                        history_item_sids.add(h_item)

                # --- 1) 检索候选（建议 topk 大一点，否则过滤后常为空） ---
                # cands = self._retrieve_candidates_by_review(pos_rev, topk=topk)
                cands = self._retrieve_candidates_by_context(pos_item, pos_rev, topk=topk)

                # 去重（保持顺序）
                seen = set()
                cands = [x for x in cands if not (x in seen or seen.add(x))]

                # --- 2) 过滤 + prefix 逐级放宽，避免 cands 为空 ---
                # 注意：prefix_m 绝对不要是 4
                prefix_m = min(prefix_m0, 3)
                filtered = []
                for m in range(prefix_m, 0, -1):
                    filtered = [
                        x for x in cands
                        if (x != pos_item) and (x not in history_item_sids) and self._share_prefix(x, pos_item, m=m)
                    ]
                    if len(filtered) >= max(2, hard_n):  # 至少够抽一些 hard
                        break

                cands = filtered

                # --- 3) 分层采样：hard + medium + random ---
                negs = []

                if len(cands) > 0:
                    n = len(cands)

                    # 分段比例：前 20% 当 hard，其后 60% 当 medium（你可以调）
                    hard_cut = max(1, min(int(0.2 * n), 50))
                    med_cut = max(hard_cut, min(int(0.8 * n), 200))

                    hard_pool = cands[:hard_cut]
                    med_pool = cands[hard_cut:med_cut]
                    # hard：直接取前 hard_n 个
                    negs.extend(hard_pool[:hard_n])

                    # medium：继续按顺序补
                    remain_for_med = max(0, K - len(negs))
                    if remain_for_med > 0:
                        negs.extend(med_pool[:remain_for_med])
                    # # hard
                    # if hard_pool:
                    #     k_h = min(hard_n, len(hard_pool))
                    #     if k_h > 0:
                    #         negs.extend(random.sample(hard_pool, k=k_h))

                    # # medium（留 1 个位置给 random，增强稳定性）
                    # remain_for_med = max(0, K - len(negs) - 1)
                    # if remain_for_med > 0 and med_pool:
                    #     k_m = min(remain_for_med, len(med_pool))
                    #     if k_m > 0:
                    #         negs.extend(random.sample(med_pool, k=k_m))

                # random 补齐（强制排除 pos 和 history）
                while len(negs) < K:
                    negs.append(self._sample_item_negative(pos_item, history_item_sids))

                # 去重一次（避免 hard/med 与 random 撞）
                seen = set([pos_item])
                uniq_negs = []
                for x in negs:
                    if x not in seen:
                        uniq_negs.append(x)
                        seen.add(x)
                    if len(uniq_negs) == K:
                        break
                while len(uniq_negs) < K:
                    uniq_negs.append(self._sample_item_negative(pos_item, history_item_sids))
                uniq_negs.append(pos_rev)
                for neg_item in uniq_negs:
                    inter_data.append({"prompt": prompt, "chosen": pos_item, "rejected": neg_item})

        return inter_data
        
    def _process_train_data_v3(self):
        inter_data = []

        # type-negative 比例：建议 0.3 左右，别太大
        p_type_neg = getattr(self.args, "p_type_neg", 0.3)

        for uid in self.remapped_inters:
            # 你这里用 [:-2] 是为了不泄露 valid/test，我尊重你的设定
            seq = self.remapped_inters[uid][:-2]
            if len(seq) < 2:
                continue

            # 从第 1 个开始做 next-step（history 至少 1）
            for t in range(len(seq)-1, len(seq)):
                ori = seq[t]
                pos_item, pos_rev = self._split_item_review(ori)
                if not pos_item:
                    continue

                history = seq[:t]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                if self.add_prefix:
                    history = [str(k + 1) + ". " + x for k, x in enumerate(history)]
                prompt = "".join(history)

                # history 里出现过的 itemSID，用于排除“伪负例”
                history_item_sids = set()
                for h in history:
                    h_item, _ = self._split_item_review(h)
                    if h_item:
                        history_item_sids.add(h_item)

                # 负例采样：二选一（混合）
                # 1) Type-negative：用同一步 review 当 rejected（语法约束：先决策 item）
                # 2) Item-negative：采一个不在 history 且 != pos 的 item（更对齐 ranking）
                if (pos_rev != "") and (random.random() < p_type_neg):
                    neg = pos_rev
                else:
                    neg = self._sample_item_negative(pos_item, history_item_sids)

                inter_data.append({
                    "prompt": prompt,
                    "chosen": pos_item,
                    "rejected": neg
                })

        return inter_data

    def _process_train_data_v2(self):

        inter_data = []
        for uid  in self.remapped_inters:
            items = self.remapped_inters[uid][:-2]
            for i in range(1, len(items)):
                one_data = dict()
                # one_data["user"] = uid
                chosen = self._keep_item(items[i])
                # rejected = items[-3].replace(chosen,'')
                rejected = random.choice(self.all_item_ids)
                while rejected == chosen:
                    rejected = random.choice(self.all_item_ids)
                history = items[i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                if self.add_prefix:
                    history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]

                prompt = "".join(history)
                inter_data.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected
                })
                inter_data.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": items[-3].replace(chosen, "", 1) 
                })

        return inter_data
    
    def should_add_pair(self, chosen: str, rejected: str, history_list: list, skip_prefix_ge: int = 2) -> bool:
        """
        返回 True 表示保留该 (prompt, chosen, rejected) pair，否则跳过。

        skip_prefix_ge=3 表示：如果 chosen 和 rejected 共享前缀长度 >=3（共享 a,b,c）就跳过。
        """
        SID4_RE = re.compile(r'(<a_\d+>)(<b_\d+>)(<c_\d+>)(<d_\d+>)')
        if chosen == rejected:
            return False

        mc = SID4_RE.search(chosen)
        mr = SID4_RE.search(rejected)
        if (mc is None) or (mr is None):   # 解析不出 SID4，跳过脏样本
            return False

        # rejected 出现在 history 里，跳过（伪负例风险）
        if rejected in history_list:
            return False

        # 共享前缀长度
        c = mc.groups()
        r = mr.groups()
        shared = 0
        for i in range(4):
            if c[i] == r[i]:
                shared += 1
            else:
                break

        # 太不相似就跳过
        if shared >= skip_prefix_ge:
            return False

        return True
    
    def _process_train_data(self):

        inter_data = []
        for uid  in self.remapped_inters:
            seq = self.remapped_inters[uid][:-4]
            for t in range(2, len(seq), 2):
                items = self.remapped_inters[uid]
                chosen = items[t]
                rev_token = items[t+1]
                history = items[:t]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                if self.add_prefix:
                    history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]

                prompt = "".join(history)

                if chosen!= rev_token:
                    inter_data.append({
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rev_token
                    })

        return inter_data

                # for i in range(self.neg_k):
                #     rej_item = self.pair_data[prompt][i]
                #     if rej_item != chosen:
                #         inter_data.append({
                #                 "prompt": prompt,
                #                 "chosen": chosen,
                #                 "rejected": rej_item
                #             })
    
    def _process_valid_data_v3(self):
        inter_data = []
        topk = int(getattr(self.args, "hard_topk", 10))        # 建议 >=200
        K = int(getattr(self.args, "num_neg", 1))               # 总负例数
        hard_n = int(getattr(self.args, "hard_n", 1))           # hard 负例数
        prefix_m0 = int(getattr(self.args, "neg_prefix_m", 2))  # 建议 2 或 3（别用4）

        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            # if len(items) < 3:
            #     continue

            # 你坚持用 -2 当正例 item（我不改）
            ori = items[-2]
            pos_item, pos_rev = self._split_item_review(ori)
            if not pos_item:
                continue
            history = items[:-2]  # 到 -3 为止
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + x for k, x in enumerate(history)]
            prompt = "".join(history)

            # history itemSID set
            history_item_sids = set()
            for h in history:
                h_item, _ = self._split_item_review(h)
                if h_item:
                    history_item_sids.add(h_item)
                    

            neg_item = None
            negs = []
            # 1) 检索候选
            # cands = self._retrieve_candidates_by_review(review_for_retrieval, topk=topk)
            cands = self._retrieve_candidates_by_context(pos_item, pos_rev, topk=topk)
            seen = set()
            cands = [x for x in cands if not (x in seen or seen.add(x))]
            if len(cands) > 0:
                n = len(cands)
                hard_cut = max(1, min(int(0.2 * n), 50))
                med_cut  = max(hard_cut, min(int(0.8 * n), 200))

                hard_pool = cands[:hard_cut]
                med_pool  = cands[hard_cut:med_cut]
                negs.extend(hard_pool[:hard_n])
                remain_for_med = max(0, K - len(negs))
                if remain_for_med > 0:
                    negs.extend(med_pool[:remain_for_med])

            while len(negs) < K:
                negs.append(self._sample_item_negative(pos_item, history_item_sids))
                neg_item = self._sample_item_negative(pos_item, history_item_sids)

            for neg_item in negs:
                inter_data.append({
                    "prompt": prompt,
                    "chosen": pos_item,
                    "rejected": neg_item
                })

        return inter_data
    
    def _process_valid_data_v2(self):
        inter_data = []
        topk = getattr(self.args, "hard_topk", 50)

        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            if len(items) < 3:
                continue

            # 你坚持用 -2，我不改
            ori = items[-2]
            pos_item, pos_rev = self._split_item_review(ori)
            if (not pos_item) or (not pos_rev):
                continue

            history = items[:-2]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + x for k, x in enumerate(history)]
            prompt = "".join(history)

            history_item_sids = set()
            for h in history:
                h_item, _ = self._split_item_review(h)
                if h_item:
                    history_item_sids.add(h_item)

            # ✅ 与 train 对齐：用 review-guided hard item negative
            neg_item = self._sample_review_guided_hard_item(
                review_sid=pos_rev,
                pos_item_sid=pos_item,
                history_item_sids=history_item_sids,
                topk=topk
            )

            inter_data.append({
                "prompt": prompt,
                "chosen": pos_item,
                "rejected": neg_item
            })

        return inter_data
    
    def _process_valid_data_v1(self):
        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            if len(items) < 3:
                continue

            # 你这里 chosen 用 -2 我不质疑
            ori = items[-2]
            pos_item, pos_rev = self._split_item_review(ori)
            if not pos_item:
                continue

            history = items[:-2]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + x for k, x in enumerate(history)]
            prompt = "".join(history)

            history_item_sids = set()
            for h in history:
                h_item, _ = self._split_item_review(h)
                if h_item:
                    history_item_sids.add(h_item)

            neg = self._sample_item_negative(pos_item, history_item_sids)

            inter_data.append({
                "prompt": prompt,
                "chosen": pos_item,
                "rejected": neg
            })

        return inter_data
    
    def _process_valid_data(self):

        inter_data = []
        for uid  in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            chosen = self._keep_item(items[-2])
            # rejected = items[-3].replace(chosen,'')
            rejected = random.choice(self.all_item_ids)
            while rejected == chosen:
                rejected = random.choice(self.all_item_ids)
            history = items[:-2]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]

            prompt = "".join(history)
            # inter_data.append({
            #     "prompt": prompt,
            #     "chosen": chosen,
            #     "rejected": rejected
            # })
            inter_data.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": items[-2].replace(chosen, "", 1) 
            })

        return inter_data
    def _process_test_data(self):

        inter_data = []
        for uid in self.remapped_inters:
            # if uid not in cold_user:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = self._keep_item(items[-1])
            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = "".join(history)

            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            # print(sample_idx[:10])##################
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data
    
    def _process_test_data_ids(self):

        inter_data = []
        for uid in self.inters:
            # if uid not in cold_user:
            items = self.inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-1]
            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = history
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            # print(sample_idx[:10])##################
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data     

    def get_new_tokens(self):

        if self.new_tokens is not None:
            return self.new_tokens

        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        for index in self.rev_indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))

        return self.new_tokens  
    
    def get_all_items(self):

        if self.all_items is not None:
            return self.all_items

        self.all_items = set()
        for index in self.indices.values():
            self.all_items.add("".join(index))

        for index in self.rev_indices.values():
            self.all_items.add("".join(index)) 
        return self.all_items

    def set_prompt(self, prompt_id):

        self.prompt_id = prompt_id

    def __len__(self):

        return len(self.inter_data)

    def __getitem__(self, index):
        d = self.inter_data[index]

        if self.mode in ["train", "valid"]:
            return {
                "prompt": d["prompt"],
                "chosen": d["chosen"],
                "rejected": d["rejected"],
            }
        else:
            return {
                "prompt": d["inters"], 
                "chosen": d["item"],
                "rejected": d["reject"] # DPO 验证时通常也需要这三项，或者你单独写 eval 逻辑
            }
    
