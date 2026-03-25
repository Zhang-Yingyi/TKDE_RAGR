import argparse
import collections
import gzip
import html
import json
import os
import random
import re
import torch
from tqdm import tqdm
import numpy as np
from utils import set_device, load_json, load_plm, clean_text
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

def load_data(args):

    item2feature_path = os.path.join(args.root, f'{args.dataset}.inter.json')
    item2feature = load_json(item2feature_path)

    return item2feature

def generate_text(item2feature, features):
    rev_ls = []
    for item in item2feature:
        data = item2feature[item]
        rev_data = data[1]
        rev_ls+=rev_data
    return rev_ls

def preprocess_text(args):
    print('Process text data: ')
    print(' Dataset: ', args.dataset)

    item2feature = load_data(args)
    # load item text and clean
    item_text_list = generate_text(item2feature, ['title', 'description'])
    # item_text_list = generate_text(item2feature, ['title'])
    # return: list of (item_ID, cleaned_item_text)
    return item_text_list

def generate_item_embedding(args, item_text_list, model, word_drop_ratio=-1):
    print(f'Generate Text Embedding: ')
    print(' Dataset: ', args.dataset)
    # item_text_list # [rev1,rev2,...]
    embeddings = []
    start, batch_size = 0, 1
    while start < len(item_text_list):
        if (start+1)%100==0:
            print("==>",start+1)
        field_texts = item_text_list[start: start + batch_size]
        rev_embedding = model.encode(field_texts, convert_to_tensor=True, device=args.device)
        embeddings.append(rev_embedding)
        start += batch_size

    embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
    print('Embeddings shape: ', embeddings.shape)

    file = os.path.join(args.root, args.dataset + '.emb-' + args.plm_name + "-td_rev" + ".npy")
    np.save(file, embeddings)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Instruments', help='Instruments / Arts / Games')
    parser.add_argument('--root', type=str, default="")
    parser.add_argument('--plm_name', type=str, default='llama')
    parser.add_argument('--plm_checkpoint', type=str,
                        default='')
    parser.add_argument('--max_sent_len', type=int, default=2048)
    parser.add_argument('--word_drop_ratio', type=float, default=-1, help='word drop ratio, do not drop by default')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.root = os.path.join(args.root, args.dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    args.device = device

    item_text_list = preprocess_text(args)
    # print(item_text_list)
    plm_model = SentenceTransformer(args.plm_checkpoint)
    plm_model = plm_model.to(device)

    generate_item_embedding(args, item_text_list,plm_model, word_drop_ratio=args.word_drop_ratio)


