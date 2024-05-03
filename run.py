import argparse
import logging
import os
import sys
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
# from data_utils import MyDataset, Seq2SeqDataCollator
from tqdm import tqdm
# from Dataset import MyDataset, DataCollator
from Xsum_Dataset import MyDataset, DataCollator
from utils import read_config, Optimzier_Scheduler, label_smoothing_loss, padding_truncation, computer_rouge, pad, \
    compute_kl_loss
from model import PW, RankingLoss
from bart_model import BartForConditionalGeneration
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader
from transformers import BartTokenizer, PegasusTokenizer, AutoTokenizer
import copy, json
from torch.cuda.amp import autocast as autocast, GradScaler

config_path = "./configs-Copy1.json"

# add local_rank and other params
parser = argparse.ArgumentParser()
parser.add_argument("--freeze", default=True, type=bool)
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--dataset", default="xsum", type=str)
parser.add_argument("--cand_size", default=10, type=int)
args = parser.parse_args()
local_rank = args.local_rank

# read config file
configs = read_config(config_path)


# Initialize the distributed training environment,Linux use nccl, windows use gloo
def init_distributed():
    dist.init_process_group(backend='nccl', init_method="env://", world_size=configs.train.gpus,
                            rank=args.local_rank)


if args.dataset == "cnndm":
    configs = configs.cnndm
    tokenizer = BartTokenizer.from_pretrained(configs.train.model_path)


    def eval_fn(rouge1, rouge2, rougel):
        return 1 - (rouge1 * rouge2 + rougel) / 3
else:
    configs = configs.xsum
    tokenizer = AutoTokenizer.from_pretrained(configs.train.model_path)


    def eval_fn(rouge1, rouge2, rougel):
        return 1 - 2 * rouge1 * rouge2 / (rouge1 + rouge2) if (rouge1 + rouge2) != 0 else 1 - (
                rouge1 * rouge2 + rougel) / 3
# set random seed and Add the following lines to use multiple GPUs
if torch.cuda.is_available():
    logging.warning("Cuda is available!")
    np.random.seed(configs.train.seed)
    torch.manual_seed(configs.train.seed)
    device = torch.device("cuda")
    if torch.cuda.device_count() > 1:
        init_distributed()
        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        logging.warning(f"Find {torch.cuda.device_count()} GPUs!")
    else:
        logging.warning("Too few GPU!")
else:
    device = torch.device("cpu")
    logging.warning("Cuda is not available! Exit!")

model = PW(dataset=args.dataset, configs=configs,
           tokenizer=tokenizer, freeze=args.freeze,
           gpu_count=torch.cuda.device_count())

# model need to gpu before DDP
model = model.to(device)
if torch.cuda.device_count() > 1:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

write = SummaryWriter(configs.train.log_path)


def train(configs=configs):
    model.train()

    collate_fn = DataCollator(tokenizer, configs, args.dataset)
    print(configs.train.file_path)
    train_dataset = MyDataset(dir_path=configs.train.file_path,
                              dataset=args.dataset, cand_size=args.cand_size,
                              max_summary_len=configs.train.max_document_len,
                              )

    if torch.cuda.device_count() > 1:
        train_sample = DistributedSampler(train_dataset)
        train_dataLoader = DataLoader(train_dataset, batch_size=configs.train.batch_size, shuffle=False, num_workers=8,
                                      collate_fn=collate_fn, sampler=train_sample)
        model.module.scoring_mode()
    else:
        train_dataLoader = DataLoader(train_dataset, batch_size=configs.train.batch_size, shuffle=True, num_workers=8,
                                      collate_fn=collate_fn)
        model.scoring_mode()

    # modify
    optimizer = optim.Adam(model.parameters())
    print(f"optimizer:{optimizer}")

    print(f"len(train_dataLoader):{len(train_dataLoader)}")

    # init loss function
    if configs.train.label_smooth > 0:
        mle_fn = label_smoothing_loss(ignore_index=tokenizer.pad_token_id, epsilon=configs.train.label_smooth)
    else:
        mle_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    # start train
    print("start train")
    print(f"configs.train.epoch:{configs.train.epoch}")

    min_loss, steps_total, all_step_cnt, epoch = 100, 0, 0, 0
    for epoch_index in range(configs.train.epoch):
        step_count = 0
        avg_mle_loss, avg_ranking_loss, avg_loss = 0, 0, 0
        if torch.cuda.device_count() > 1:
            train_sample.set_epoch(epoch_index)
        for index, batch in tqdm(enumerate(train_dataLoader), total=len(train_dataLoader)):
            steps_total += 1
            step_count += 1
            tgt_text = batch['tgt_text']
            src_text = batch['src_text']
            candidates = batch['candidates']
            steps = epoch_index * len(train_dataLoader) + index + 1
            candidates1 = torch.cat([tgt_text.unsqueeze(1), candidates[:, 0:args.cand_size]], dim=1)
            candidates = torch.cat([candidates1, candidates1], dim=0)
            src_text = torch.cat([src_text, src_text], dim=0)
            if configs.train.fp16:
                scaler = GradScaler()
                with autocast(dtype=torch.bfloat16):
                    output1 = model(src_text.to(device), tgt_text.to(device),
                                    candidates.to(device))
                    similarity1, gold_similarity1 = output1['score'], output1['summary_score']
                    scale = configs.train.scale
                    similarity1 = similarity1 * scale
                    gold_similarity1 = gold_similarity1 * scale
                    ranking_loss = RankingLoss(similarity1, gold_similarity1, configs.train.margin,
                                               configs.train.gold_margin,
                                               configs.train.gold_weight)
                    # probs: [bz, seq_len, word_num]
                    probs = output1["probs"][:, :-1]  # truncate last token
                    gold = batch['tgt_text'][:, 1:]
                    probs1 = probs[0].unsqueeze(0)
                    probs2 = probs[1].unsqueeze(0)
                    mle_loss1 = mle_fn(probs1, gold)
                    mle_loss2 = mle_fn(probs2, gold)
                    mle_loss = (mle_loss1 + mle_loss2) / 2
                    loss = (
                                   configs.train.mle_weight * mle_loss + configs.train.rank_weight * ranking_loss) / configs.train.accumulate_step

                scaler.scale(loss).backward()
                if step_count == configs.train.accumulate_step:
                    step_count = 0
                    all_step_cnt += 1
                    lr = configs.train.lr * min(all_step_cnt ** (-0.5),
                                                all_step_cnt * (configs.train.warmup_steps ** (-1.5)))
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    scaler.step(optimizer)
                    optimizer.zero_grad()
                    scaler.update()

            else:
                output1 = model(src_text.to(device), tgt_text.to(device),
                                candidates.to(device))
                similarity1, gold_similarity1 = output1['score'], output1['summary_score']
                scale = configs.train.scale
                similarity1 = similarity1 * scale
                gold_similarity1 = gold_similarity1 * scale
                ranking_loss = RankingLoss(similarity1, gold_similarity1, configs.train.margin,
                                           configs.train.gold_margin,
                                           configs.train.gold_weight)
                # probs: [bz, seq_len, word_num]
                probs = output1["probs"][:, :-1]  # truncate last token
                # print(f"probs.size():{probs.size()}")
                gold = batch['tgt_text'][:, 1:]
                probs1 = probs[0].unsqueeze(0)
                probs2 = probs[1].unsqueeze(0)
                mle_loss1 = mle_fn(probs1, gold)
                mle_loss2 = mle_fn(probs2, gold)
                mle_loss = (mle_loss1 + mle_loss2) / 2
                # loss = (configs.train.mle_weight * mle_loss + configs.train.rank_weight * ranking_loss + configs.train.kl_weight * kl_loss) / configs.train.accumulate_step
                loss = (
                               configs.train.mle_weight * mle_loss + configs.train.rank_weight * ranking_loss) / configs.train.accumulate_step
                loss.backward()
                if step_count == configs.train.accumulate_step:
                    step_count = 0
                    all_step_cnt += 1
                    lr = configs.train.lr * min(all_step_cnt ** (-0.5),
                                                all_step_cnt * (configs.train.warmup_steps ** (-1.5)))
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    optimizer.step()
                    optimizer.zero_grad()

            if steps % configs.train.logging_steps == 0:
                if torch.cuda.device_count() > 1 and dist.get_rank() == 0:
                    write.add_scalar("lr", optimizer.state_dict()['param_groups'][0]['lr'], steps)
                    write.add_scalar("mle_loss", mle_loss.item(), steps)
                    write.add_scalar("ranking_loss", ranking_loss.item(), steps)
                    write.add_scalar("loss", loss, steps)
                else:
                    write.add_scalar("lr", optimizer.state_dict()['param_groups'][0]['lr'], steps)
                    write.add_scalar("mle_loss", mle_loss.item(), steps)
                    write.add_scalar("ranking_loss", ranking_loss.item(), steps)
                    write.add_scalar("loss", loss, steps)
                del ranking_loss, mle_loss, loss

            if steps_total < 400000:
                save_step = 10000
            elif steps_total < 500000:
                save_step = 5000
            else:
                save_step = 2000

            if steps_total % save_step == 0 and steps_total >= 100000:
                result = evalute(model, tokenizer, configs, args.dataset, steps_total, (steps_total / save_step),
                                 steps_total)
                rouge1 = result["rouge1"]
                rouge2 = result["rouge2"]
                rougel = result["rougel"]
                loss = eval_fn(result["rouge1"], result["rouge2"], result["rougel"])
                if loss <= min_loss:
                    min_loss = loss
                    print(f"The best rouge is :rouge1:{rouge1}, rouge2:{rouge2}, rougel:{rougel}")
                    if torch.cuda.device_count() > 1 and dist.get_rank() == 0:
                        torch.save({
                            "epoch": epoch_index,
                            "learning_rate": optimizer.state_dict()['param_groups'][0]['lr'],
                            "optimizer": optimizer.state_dict(),
                            "model_state_dict": model.module.state_dict(),
                            "all_step_cnt": all_step_cnt
                        }, os.path.join(configs.train.save_model_path, "train_generate.pth"))
                    else:
                        torch.save({
                            "epoch": epoch_index,
                            "learning_rate": optimizer.state_dict()['param_groups'][0]['lr'],
                            "optimizer": optimizer.state_dict(),
                            "model_state_dict": model.state_dict(),
                            "all_step_cnt": all_step_cnt
                        }, os.path.join(configs.train.save_model_path, "train_generate.pth"))


pass


def test(model, tokenizer, configs, dataset, steps_total, epoch_index, step):
    model.eval()
    if torch.cuda.device_count() > 1:
        model = model.module
    else:
        model = model
    model.generation_mode()
    rouge1, rouge2, rougel = 0, 0, 0
    test_dataset = MyDataset(dir_path=configs.test.file_path,
                             dataset=dataset, is_test=True,
                             max_summary_len=configs.test.max_document_len,
                             )
    collate_fn = DataCollator(tokenizer, configs, dataset, is_test=True)
    if torch.cuda.device_count() > 1:
        test_sample = DistributedSampler(test_dataset)
        test_dataLoader = DataLoader(test_dataset, batch_size=configs.test.batch_size, shuffle=False, num_workers=4,
                                     collate_fn=collate_fn, sampler=test_sample)
    else:
        test_dataLoader = DataLoader(test_dataset, batch_size=configs.test.batch_size, shuffle=False, num_workers=4,
                                     collate_fn=collate_fn)

    count, steps, index = 0, 0, 0
    generates_, gold_summarys_, orgins_ = [], [], []
    for batch in tqdm(test_dataLoader):
        src_text = batch['src_text'].to(device)
        tgt_text = batch['tgt_text'].to(device)

        summaries = model.generate(input_ids=src_text["input_ids"].to(device),
                                   attention_mask=src_text["attention_mask"].to(device),
                                   num_return_sequences=1,
                                   max_length=configs.test.gen_max_len + 2, num_beam_groups=1,
                                   diversity_penalty=0.0, num_beams=configs.test.num_beams,
                                   no_repeat_ngram_size=3,
                                   length_penalty=configs.test.length_penalty,
                                   early_stopping=True)

        gold_summarys = tokenizer.batch_decode(tgt_text["input_ids"], skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)
        summaries_ = tokenizer.batch_decode(summaries, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
        src_text_ = tokenizer.batch_decode(src_text["input_ids"], skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)
        generates_.append(summaries_)
        gold_summarys_.append(gold_summarys)
        orgins_.append(src_text_)
        scores = computer_rouge(gold_summarys, summaries_, tokenizer)
        for index in range(configs.test.batch_size):
            count += 1
            steps += 1
            rouge1 += scores[index]['rouge1']
            rouge2 += scores[index]['rouge2']
            rougel += scores[index]['rougel']
            write.add_scalar(f"rouge1_{epoch_index}", rouge1 / count, count)
            write.add_scalar(f"rouge2_{epoch_index}", rouge2 / count, count)
            write.add_scalar(f"rougel_{epoch_index}", rougel / count, count)

    with open(f"result/test/origins_{step}.json", "w") as f:
        f.write(json.dumps(orgins_))

    with open(f"result/test/generates_{step}.json", "w") as f:
        f.write(json.dumps(generates_))

    with open(f"result/test/gold_summarys_{step}.json", "w") as f:
        f.write(json.dumps(gold_summarys_))

    rouge1 = rouge1 / steps
    rouge2 = rouge2 / steps
    rougel = rougel / steps
    model.scoring_mode()
    model.train()
    return {
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougel": rougel,
    }


def evalute(model, tokenizer, configs, dataset, steps_total, epoch_index, step):
    model.eval()
    if torch.cuda.device_count() > 1:
        model = model.module
    else:
        model = model
    model.generation_mode()
    rouge1, rouge2, rougel = 0, 0, 0
    val_dataset = MyDataset(dir_path=configs.eval.file_path,
                            dataset=dataset, is_test=True,
                            max_summary_len=configs.eval.max_document_len,
                            )
    collate_fn = DataCollator(tokenizer, configs, dataset, is_test=True)
    if torch.cuda.device_count() > 1:
        val_sample = DistributedSampler(val_dataset)
        val_dataLoader = DataLoader(val_dataset, batch_size=configs.eval.batch_size, shuffle=False, num_workers=8,
                                    collate_fn=collate_fn, sampler=val_sample)
    else:
        val_dataLoader = DataLoader(val_dataset, batch_size=configs.eval.batch_size, shuffle=False, num_workers=8,
                                    collate_fn=collate_fn)

    count, steps, index = 0, 0, 0
    generates_, gold_summarys_, orgins_ = [], [], []
    for batch in tqdm(val_dataLoader):
        with torch.no_grad():
            src_text = batch['src_text'].to(device)
            tgt_text = batch['tgt_text'].to(device)

            summaries = model.generate(input_ids=src_text["input_ids"].to(device),
                                       attention_mask=src_text["attention_mask"].to(device),
                                       num_return_sequences=1,
                                       max_length=configs.eval.gen_max_len + 2,
                                       min_length=configs.eval.gen_min_len + 1,
                                       num_beam_groups=1,
                                       num_beams=configs.eval.num_beams,
                                       no_repeat_ngram_size=3,
                                       length_penalty=configs.test.length_penalty,
                                       early_stopping=True)

            gold_summarys = tokenizer.batch_decode(tgt_text["input_ids"], skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False)
            summaries_ = tokenizer.batch_decode(summaries, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)
            src_text_ = tokenizer.batch_decode(src_text["input_ids"], skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)
            generates_.append(summaries_)
            gold_summarys_.append(gold_summarys)
            orgins_.append(src_text_)
            scores = computer_rouge(gold_summarys, summaries_, tokenizer)
            for index in range(configs.eval.batch_size):
                count += 1
                steps += 1
                rouge1 += scores[index]['rouge1']
                rouge2 += scores[index]['rouge2']
                rougel += scores[index]['rougel']
                write.add_scalar(f"rouge1_{steps_total}", rouge1 / count, count)
                write.add_scalar(f"rouge2_{steps_total}", rouge2 / count, count)
                write.add_scalar(f"rougel_{steps_total}", rougel / count, count)

    with open(f"result/eval/origins_{step}.json", "w") as f:
        f.write(json.dumps(orgins_))

    with open(f"result/eval/generates_{step}.json", "w") as f:
        f.write(json.dumps(generates_))

    with open(f"result/eval/gold_summarys_{step}.json", "w") as f:
        f.write(json.dumps(gold_summarys_))

    rouge1 = rouge1 / steps
    rouge2 = rouge2 / steps
    rougel = rougel / steps
    model.scoring_mode()
    model.train()
    return {
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougel": rougel,
    }


if __name__ == '__main__':
    train()
