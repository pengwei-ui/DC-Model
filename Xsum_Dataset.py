import argparse
import torch
from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration, AutoTokenizer
import json
from utils import read_config
from typing import Optional, Dict
import torch, os, json, jsonpath
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import shutil
import pprint


def clean_text(text):
    text = text.replace("\n", "")
    text = text.replace("\\", "")
    return text


class MyDataset(Dataset):
    def __init__(self, dir_path, dataset="cnndm", cand_size=8, max_summary_len=512, is_test=False):
        super(MyDataset, self).__init__()
        self.dir_path = dir_path
        self.dataset = dataset
        self.isdir = os.path.isdir(dir_path)
        self.is_test = is_test
        if not self.is_test:
            self.cand_size = cand_size
            self.max_summary_len = max_summary_len

        if os.path.exists(f"{dir_path}/.ipynb_checkpoints"):
            shutil.rmtree(f"{dir_path}/.ipynb_checkpoints")
            print("remove ok")
        else:
            print(f"is not exist {dir_path}/.ipynb_checkpoints")

        if self.isdir:
            self.files = os.listdir(dir_path)
            self.files_num = len(self.files)
        else:
            print("is not a dir path!!!")
        pass

    def __len__(self):
        return self.files_num

    def __getitem__(self, item):
        if not self.is_test:
            assert self.cand_size <= 16
            if self.isdir:
                with open(f'{self.dir_path}/{self.files[item]}', 'r') as f:
                    data = json.load(f)
            src_text = "".join(item for item in data['article_untok'])
            src_text = clean_text(src_text)
            tgt_text = "".join(item for item in data['abstract_untok'])
            tgt_text = clean_text(tgt_text)
            orgin_cands = jsonpath.jsonpath(data, "$..candidates_untok[*]")[0:self.cand_size]
            candidates_ = sorted(orgin_cands, key=lambda x: x[1], reverse=True)
            candidates, candidate = [], ""
            for item in candidates_:
                for i in item[0]:
                    candidate += "".join(i)
                candidates.append(candidate)
            return {"src_text": src_text,
                    "tgt_text": tgt_text, "candidates": candidates}
        else:
            if self.isdir:
                with open(f'{self.dir_path}/{self.files[item]}', 'r') as f:
                    data = json.load(f)
            src_text = "".join(item for item in data['article_untok'])
            src_text = clean_text(src_text)
            tgt_text = "".join(item for item in data['abstract_untok'])
            tgt_text = clean_text(tgt_text)
            return {"src_text": src_text, "tgt_text": tgt_text}


@dataclass()
class DataCollator:
    def __init__(self, tokenizer, configs, dataset, is_test=False):
        self.tokenizer = tokenizer
        self.configs = configs
        self.dataset = dataset
        self.is_test = is_test

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        if not self.is_test:
            candidates, batch_size = [], len(batch)
            for index in range(batch_size):
                candidates += jsonpath.jsonpath(batch[index], f'$.candidates.[*]')
            candidates = [x["tgt_text"] for x in batch] + candidates
            src_text = self.tokenizer.batch_encode_plus([x["src_text"] for x in batch],
                                                        max_length=self.configs.train.max_document_len,
                                                        return_tensors="pt",
                                                        padding=True, truncation=True)

            candidates = self.tokenizer.batch_encode_plus(candidates,
                                                          max_length=self.configs.train.max_summary_len,
                                                          return_tensors="pt",
                                                          padding=True, truncation=True)
            candidates = candidates.input_ids
            if self.dataset == "xsum":
                _candidate_ids = candidates.new_zeros(candidates.size(0), candidates.size(1) + 1)
                _candidate_ids[:, 1:] = candidates.clone()
                _candidate_ids[:, 0] = self.tokenizer.pad_token_id
                candidates = _candidate_ids

            tgt_text = candidates[0:batch_size]

            candidates = candidates[batch_size:, ].view(batch_size, -1, candidates.size()[-1])

            return {"src_text": src_text.input_ids,
                    "tgt_text": tgt_text,
                    "candidates": candidates}
        else:
            src_text = self.tokenizer.batch_encode_plus([x["src_text"] for x in batch],
                                                        max_length=self.configs.test.max_document_len,
                                                        return_tensors="pt",
                                                        padding=True, truncation=True)
            tgt_text = self.tokenizer.batch_encode_plus([x["tgt_text"] for x in batch],
                                                        max_length=self.configs.test.max_summary_len,
                                                        return_tensors="pt",
                                                        padding=True, truncation=True)
            return {"src_text": src_text, "tgt_text": tgt_text}
