import argparse
import logging
import os
import sys
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
from Xsum_Dataset import MyDataset, DataCollator
from utils import read_config, Optimzier_Scheduler, label_smoothing_loss, padding_truncation, computer_rouge
from model import PW, RankingLoss
from bart_model import BartForConditionalGeneration
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader
from transformers import BartTokenizer, PegasusTokenizer, AutoTokenizer
import copy,json

device = "cuda"


def test(model, tokenizer, configs, dataset, epoch_index):
    model.eval()
    if torch.cuda.device_count() > 1:
        model = model.module
    else:
        model = model
    model.generation_mode()
    rouge1, rouge2, rougeL = 0, 0, 0
    test_dataset = MyDataset(dir_path=configs.test.file_path,
                             dataset=dataset, is_test=True,
                             max_summary_len=configs.test.max_document_len,
                             )
    collate_fn = DataCollator(tokenizer, configs, dataset, is_test=True)
    if torch.cuda.device_count() > 1:
        test_sample = DistributedSampler(test_dataset)
        test_dataLoader = DataLoader(test_dataset, batch_size=configs.test.batch_size, shuffle=False, num_workers=6,
                                     collate_fn=collate_fn, sampler=test_sample)
    else:
        test_dataLoader = DataLoader(test_dataset, batch_size=configs.test.batch_size, shuffle=False, num_workers=6,
                                     collate_fn=collate_fn)

    count, steps, index = 0, 0,0
    generates_, gold_summarys_ = [], []
    for batch in tqdm(test_dataLoader):
        src_text = batch['src_text'].to(device)
        tgt_text = batch['tgt_text'].to(device)
        summaries = model.generate(input_ids=src_text["input_ids"].to(device),
                                   attention_mask=src_text["attention_mask"].to(device),
                                   num_return_sequences=1,
                                   max_length=configs.test.gen_max_len + 2, num_beam_groups=1,
                                   diversity_penalty=0.0, num_beams=8,
                                   no_repeat_ngram_size=3,
                                   length_penalty=configs.test.length_penalty,
                                   early_stopping=True)
        gold_summarys = tokenizer.batch_decode(tgt_text["input_ids"], skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)
        summaries_ = tokenizer.batch_decode(summaries, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)
        generates_.append(summaries_)
        gold_summarys_.append(gold_summarys)
        scores = computer_rouge(gold_summarys, summaries_, tokenizer)
        for index in range(configs.test.batch_size):
            count += 1
            steps += 1
            rouge1 += scores[index]['rouge1']
            rouge2 += scores[index]['rouge2']
            rougeL += scores[index]['rougel']
            print(f"rouge1:{rouge1 / count}")
            print(f"rouge2:{rouge2 / count}")
            print(f"rougeL:{rougeL / count}")
            print("*"*30)
            write.add_scalar(f"rouge1_{epoch_index}", rouge1 / count, count)
            write.add_scalar(f"rouge2_{epoch_index}", rouge2 / count, count)
            write.add_scalar(f"rougeL_{epoch_index}", rougeL / count, count)

    with open(f"generates_xsum.json", "w")  as f:
            f.write(json.dumps(generates_))
    
    with open(f"gold_summarys_xsum.json", "w")  as f:
            f.write(json.dumps(gold_summarys_))

    rouge1 = rouge1 / steps
    rouge2 = rouge2 / steps
    rougeL = rougeL / steps
    write.add_scalar("rouge1_mean", rouge1, epoch_index)
    write.add_scalar("rouge2_mean", rouge2, epoch_index)
    write.add_scalar("rougeL_mean", rougeL, epoch_index)
    model.scoring_mode()
    model.train()
    print({
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL,
    })
    return {
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL,
    }



if __name__ == '__main__':
    config_path = "./configs.json"
    configs = read_config(config_path)
    configs = configs.xsum
    write = SummaryWriter(configs.test.log_path)
    tokenizer = AutoTokenizer.from_pretrained(configs.train.model_path)
    model = PW(dataset="xsum", configs=configs,
               tokenizer=tokenizer, freeze=True,
               gpu_count=torch.cuda.device_count())
    #replace you model path
    model_paths = [f"/mnt/pw/Abstract_Summarization_demo2/model/xsum_4346/train_generate.pth"]
    for index, model_path in enumerate(model_paths):
        checkpoint = torch.load(model_path, map_location='cuda:0')
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        result = test(model, tokenizer, configs, "xsum", index + 1)
        print(result)
