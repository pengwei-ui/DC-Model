from typing import Callable, Dict, Iterable, List, Tuple
import numpy as np
import torch
from nltk import word_tokenize
from rouge import Rouge
from torch import nn
import json
from types import SimpleNamespace
from torch.utils.tensorboard import SummaryWriter
from transformers import EvalPrediction, PreTrainedTokenizer, AutoTokenizer, logging
import torch.nn.functional as F
from torch.optim import Adam, SGD
from transformers.optimization import (
    Adafactor,
    AdamW,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

rouge = Rouge()
logger = logging.get_logger(__name__)
arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule,
    "constant_w_warmup": get_constant_schedule_with_warmup,
}


class Optimzier_Scheduler:
    def __init__(self, configs, model, optimizer=None):
        self.configs = configs
        self.model = model
        self.optimizer = optimizer
        pass

    def create_optimizer(self):
        """
        Setup the self.optimizer and the learning rate scheduler.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if
                               not any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": self.configs.train.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if
                               any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": 0.0,
                },
            ]
            if self.configs.train.adafactor:
                self.optimizer = Adafactor(
                    optimizer_grouped_parameters,
                    lr=self.configs.train.lr,
                    scale_parameter=False,
                    relative_step=False,
                )
            else:
                if self.configs.train.optimizer == "Adam":
                    self.optimizer = Adam(
                    optimizer_grouped_parameters, lr=self.configs.train.lr, eps=self.configs.train.eps, betas=(0.9, 0.999)
                )
                elif self.configs.train.optimizer == "AdamW":
                    self.optimizer = AdamW(
                        optimizer_grouped_parameters, lr=self.configs.train.lr, eps=self.configs.train.eps
                    )
                else:
                    self.optimizer = SGD(
                        optimizer_grouped_parameters, lr=self.configs.train.lr, momentum=0.9
                    )
        return self.optimizer

    def create_scheduler(self, num_training_steps, optimizer):
        if self.configs.train.lr_scheduler is not None:
            schedule_func = arg_to_scheduler[self.configs.train.lr_scheduler]
            print(f"schedule_func:{schedule_func}")
            if self.configs.train.lr_scheduler == "constant":
                self.scheduler = schedule_func(optimizer)
            elif self.configs.train.lr_scheduler == "constant_w_warmup":
                self.scheduler = schedule_func(optimizer, num_warmup_steps=self.configs.train.warmup_steps)
            else:
                self.scheduler = schedule_func(
                    optimizer, num_warmup_steps=self.configs.train.warmup_steps, num_training_steps=num_training_steps
                )
        return self.scheduler


def lmap(f: Callable, x: Iterable) -> List:
    """delete strip: such as \n"""
    return list(map(f, x))


def non_pad_len(tokens: np.ndarray, tokenizer: PreTrainedTokenizer) -> int:
    return np.count_nonzero(tokens != tokenizer.pad_token_id)

def pad(x1, x2, tokenizer):
    diff = abs(x1.size()[1] - x2.size()[1])
    if x1.size()[1] < x2.size()[1]:
        x1 = F.pad(x1, (0, diff), value=tokenizer.pad_token_id)
    elif x1.shape[1] > x2.shape[1]:
        x2 = F.pad(x2, (0, diff), value=tokenizer.pad_token_id)
    return x1, x2

def compute_sentence_length(sentence: str):
    words = word_tokenize(sentence)
    return len(words)


def computer_rouge(gold_summary: str, generate: str, tokenizer: PreTrainedTokenizer):
    scores = []
    try:
        generate = lmap(str.strip, generate)
        gold_summary = lmap(str.strip, gold_summary)
        result = rouge.get_scores(generate, gold_summary, avg=True)
        score = {
            "rouge1": result['rouge-1']['f'],
            "rouge2": result['rouge-2']['f'],
            "rougel": result['rouge-l']['f'],
        }
    except:
        score = {
            "rouge1": 0,
            "rouge2": 0,
            "rougel": 0,
        }
    scores.append(score)
    return scores


def _freeze_layer(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def read_config(config_path):
    with open(config_path, encoding="utf-8") as f:
        configs = json.loads(f.read(), object_hook=lambda x: SimpleNamespace(**x))
        return configs


def beam_search(model, tokenizer, batch, beams, src_texts, gpus_count=1):
    lens = len(batch)
    if gpus_count > 1:
        generate = model.module.generate(**batch, num_return_sequences=beams, num_beam_groups=beams,
                                         diversity_penalty=0.3,
                                         num_beams=beams, length_penalty=0.3)
    else:
        generate = model.generate(**batch, num_return_sequences=beams, num_beam_groups=beams,
                                  diversity_penalty=0.3,
                                  num_beams=beams, length_penalty=0.3)
    decodes = tokenizer.batch_decode(generate, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    candidates = []
    for index, item in enumerate(decodes):
        rouge = computer_rouge(src_texts[index // beams], decodes[index])
        candidates.append([decodes[index], rouge])
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    return candidates


def padding_truncation(input, target, tokenizer):
    if input.size()[1] < target.size()[1]:
        padding = torch.full((input.size()[0], target.size()[1] - input.size()[1]), tokenizer.pad_token_id,
                             dtype=torch.int64)
        padding = padding.to(input.device)
        input = torch.cat([input, padding], dim=1)
    else:
        input = input[:, 0:target.size()[1]]
    return input


# lable smoothing loss
class label_smoothing_loss(nn.Module):
    def __init__(self, ignore_index, epsilon=0.1):
        super(label_smoothing_loss, self).__init__()
        self.ignore_idx = ignore_index
        self.epsilon = epsilon
        self.input_ = None

    def forward(self, input, target):
        # input = input.transpose(1, 2)  # transpose after [batch_size, seq_len, word_num], target :[batch_size, seq_Len]
        # you can set input scale [batch_size, seq_len, word_num] directly
        input = torch.log_softmax(input, dim=2)
        # print(f"this is input:{input}")
        k = input.size(2)
        target_prob = torch.ones_like(input).type_as(input) * self.epsilon * 1 / k

        mask = torch.arange(k).unsqueeze(0).unsqueeze(0).expand(target.size(0), target.size(1), -1).type_as(target)
        mask = torch.eq(mask, target.unsqueeze(-1).expand(-1, -1, k))
        mask = mask.to(input.device)
        trget_prob = target_prob.to(input.device)
        # self.epsilon = self.epsilon.to(target.device)
        target_prob.masked_fill_(mask, 1 - self.epsilon + (self.epsilon * 1 / k))
        
        loss = - torch.mul(target_prob, input)
        loss = loss.sum(2)
        # mask ignore_idx
        mask = (target != self.ignore_idx).type_as(input)
        loss = (torch.mul(loss, mask).sum() / mask.sum()).mean()
        return loss


def compute_kl_loss(p, q, target, pad_mask=None):
    p_loss = F.kl_div(F.softmax(p, dim=-1).log(), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.softmax(q, dim=-1).log(), F.softmax(p, dim=-1), reduction='none')
    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()
    loss = (p_loss + q_loss) / 2
    return torch.sigmoid(torch.sqrt(loss)) - 0.5
           

def save_log(log_path, title, y, x):
    writer = SummaryWriter(log_path)
    writer.add_scalar(title, y, x)
    pass


