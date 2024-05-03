import torch
from transformers import BartTokenizer, BartForConditionalGeneration
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


aug_list = {"orgin_candidate": "orgin_candidate", "randomdelete_candidate": "randomdelete_candidate",
            "randomswap_candidate": "randomswap_candidate"}


class MyDataset(Dataset):
    def __init__(self, dir_path, dataset="cnndm", cand_size=8, max_summary_len=512,
                 choice_aug=["randomswap_candidate", "randomdelete_candidate", "orgin_candidate"],
                 is_test=False):
        super(MyDataset, self).__init__()
        self.dir_path = dir_path
        self.dataset = dataset
        self.isdir = os.path.isdir(dir_path)
        self.is_test = is_test
        if not self.is_test:
            self.cand_size = cand_size
            self.max_summary_len = max_summary_len
            self.choice_aug = choice_aug

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
            src_text = clean_text(data[0]['document'])
            aug1_text = clean_text(data[0]['document_aug'][0])
            aug2_text = clean_text(data[0]['document_aug'][1])
            tgt_text = clean_text(data[0]['summary'])
            orgin_cands = jsonpath.jsonpath(data, "$..candidates[*]")[0:self.cand_size]
            for index in range(len(orgin_cands)):
                orgin_cands[index]["orgin_candidate"] = "".join(orgin_cands[index]["orgin_candidate"])
            orgin_cands = sorted(orgin_cands, key=lambda x: (x["rouge-1"] + x["rouge-2"] + x["rouge-l"]) / 3,
                                 reverse=True)
            random_delete_cands = jsonpath.jsonpath(data, "$..candidates[*]")[16:16 + self.cand_size]
            for index in range(len(random_delete_cands)):
                random_delete_cands[index]["randomswap_candidate"] = "".join(
                    random_delete_cands[index]["randomswap_candidate"])
            random_delete_cands = sorted(random_delete_cands,
                                         key=lambda x: (x["rouge-1"] + x["rouge-2"] + x["rouge-l"]) / 3, reverse=True)

            random_swap_cands = jsonpath.jsonpath(data, "$..candidates[*]")[32:32 + self.cand_size]
            for index in range(len(random_swap_cands)):
                random_swap_cands[index]["randomdelete_candidate"] = "".join(
                    random_swap_cands[index]["randomdelete_candidate"])
            random_swap_cands = sorted(random_swap_cands,
                                       key=lambda x: (x["rouge-1"] + x["rouge-2"] + x["rouge-l"]) / 3, reverse=True)
            if ["orgin_candidate", "randomdelete_candidate"] == self.choice_aug:
                candidates = orgin_cands + random_delete_cands
            elif ["randomdelete_candidate", "randomswap_candidate"] == self.choice_aug:
                candidates = random_delete_cands + random_swap_cands
            else:
                candidates = orgin_cands + random_swap_cands

            return {"src_text": src_text, "randomdelete_aug": aug1_text,
                    "randomswap_aug": aug2_text,
                    "tgt_text": tgt_text, "candidates": candidates}
        else:
            if self.isdir:
                with open(f'{self.dir_path}/{self.files[item]}', 'r') as f:
                    data = json.load(f)
            src_text = clean_text(data['document'])
            tgt_text = clean_text(data['summary'])
            return {"src_text": src_text, "tgt_text": tgt_text}


@dataclass()
class DataCollator:
    def __init__(self, tokenizer, configs, dataset, choice_aug=None, is_test=False):
        self.tokenizer = tokenizer
        self.configs = configs
        self.dataset = dataset
        self.is_test = is_test
        self.choice_aug = choice_aug

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        if not self.is_test:
            candidates, batch_size = [], len(batch)
            for index in range(batch_size):
                candidates += jsonpath.jsonpath(batch[index], f'$.candidates.[*].[{aug_list[self.choice_aug[0]]}]')
                candidates += jsonpath.jsonpath(batch[index], f'$.candidates.[*].[{aug_list[self.choice_aug[1]]}]')
            candidates = [x["tgt_text"] for x in batch] + candidates
            src_text = self.tokenizer.batch_encode_plus([x["src_text"] for x in batch],
                                                        max_length=self.configs.train.max_document_len,
                                                        return_tensors="pt",
                                                        pad_to_max_length=False, padding=True, truncation=True)
            randomdelete_aug = self.tokenizer.batch_encode_plus([x["randomdelete_aug"] for x in batch],
                                                                max_length=self.configs.train.max_document_len,
                                                                return_tensors="pt",
                                                                pad_to_max_length=False, padding=True, truncation=True)
            randomswap_aug = self.tokenizer.batch_encode_plus([x["randomswap_aug"] for x in batch],
                                                              max_length=self.configs.train.max_document_len,
                                                              return_tensors="pt",
                                                              pad_to_max_length=False, padding=True, truncation=True)
            candidates = self.tokenizer.batch_encode_plus(candidates,
                                                          max_length=self.configs.train.max_summary_len,
                                                          return_tensors="pt",
                                                          pad_to_max_length=False, padding=True, truncation=True)
            candidates = candidates.input_ids
            if self.dataset == "xsum":
                _candidate_ids = candidates.new_zeros(candidates.size(0), candidates.size(1) + 1)
                _candidate_ids[:, 1:] = candidates.clone()
                _candidate_ids[:, 0] = self.tokenizer.pad_token_id
                candidates = _candidate_ids
            tgt_text = candidates[0:batch_size]
            candidates = candidates[batch_size:, ].view(batch_size, -1, candidates.size()[-1])
            return {"src_text": src_text.input_ids,
                    "randomdelete_aug": randomdelete_aug.input_ids,
                    "randomswap_aug": randomswap_aug.input_ids, "tgt_text": tgt_text,
                    "candidates": candidates}
        else:
            src_text = self.tokenizer.batch_encode_plus([x["src_text"] for x in batch],
                                                        max_length=self.configs.test.max_document_len,
                                                        return_tensors="pt",
                                                        pad_to_max_length=False, padding=True, truncation=True)
            tgt_text = self.tokenizer.batch_encode_plus([x["tgt_text"] for x in batch],
                                                        max_length=self.configs.test.max_summary_len,
                                                        return_tensors="pt",
                                                        pad_to_max_length=False, padding=True, truncation=True)
            return {"src_text": src_text, "tgt_text": tgt_text}
