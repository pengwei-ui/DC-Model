from typing import Optional, Callable, List, Iterable
from bart_model import BartScorer, BartForConditionalGeneration
from pegasus_model import PegasusScorer
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from utils import _freeze_layer, computer_rouge, lmap, read_config
import torch.nn.functional as F


# config_path = "./config.json"
# configs = read_config(config_path)


def RankingLoss(score, summary_score=None, margin=0, gold_margin=0, gold_weight=1, no_gold=False, no_cand=False):
    ones = torch.ones_like(score)
    loss_func = torch.nn.MarginRankingLoss(0.0)
    TotalLoss = loss_func(score, score, ones)
    # candidate loss
    n = score.size(1)
    if not no_cand:
        for i in range(1, n):
            pos_score = score[:, :-i]
            neg_score = score[:, i:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            ones = torch.ones_like(pos_score)
            loss_func = torch.nn.MarginRankingLoss(margin * i)
            loss = loss_func(pos_score, neg_score, ones)
            TotalLoss += loss
    if no_gold:
        return TotalLoss
    # gold summary loss
    pos_score = summary_score.unsqueeze(-1).expand_as(score)
    neg_score = score
    pos_score = pos_score.contiguous().view(-1)
    neg_score = neg_score.contiguous().view(-1)
    ones = torch.ones_like(pos_score)
    loss_func = torch.nn.MarginRankingLoss(gold_margin)
    TotalLoss += gold_weight * loss_func(pos_score, neg_score, ones)
    return TotalLoss


class PW(nn.Module):
    def __init__(self, dataset, tokenizer, freeze, gpu_count, configs):
        super(PW, self).__init__()
        self.configs = configs
        model_path = self.configs.train.model_path
        if dataset == "cnndm":
            self.model = BartScorer.from_pretrained(model_path)
        else:
            self.model = PegasusScorer.from_pretrained(model_path)
        self.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer
        self.freeze_init(freeze)
        self.gpus_count = gpu_count

    def freeze_init(self, freeze):
        if freeze:
            freeze_encoder_layer = self.configs.train.freeze_encoder_layer
            _freeze_layer(self.model.get_encoder().layers[:freeze_encoder_layer])
            freeze_decoder_layer = self.configs.train.freeze_decoder_layer
            _freeze_layer(self.model.get_decoder().layers[:freeze_decoder_layer])

    def forward(self, src_text, tgt_text, candidates, score_mode="log", require_gold=True):
        input_ids = src_text
        tgt_text = tgt_text
        tgt_len = tgt_text != self.pad_token_id
        candidates = candidates
        batch_size = input_ids.size(0)
        input_mask = input_ids != self.pad_token_id
        cand_mask = candidates != self.pad_token_id
        cand_mask[:, :, 0] = 1

        output = self.model(
            input_ids=input_ids,
            attention_mask=input_mask,
            decoder_input_ids=candidates,
            decoder_attention_mask=cand_mask,
            output_hidden_states=True
        )

        output = output[0]  # [bz x cand_num, seq_len, word_dim]

        output = output.view(batch_size, -1, output.size(1), output.size(2))  # [bz, cand_num, seq_len, word_dim]
        probs = output[:, 0]

        output = output[:, :, :-1]
        candidates = candidates[:, :, 1:]
        cand_mask = candidates != self.pad_token_id
        candidates = candidates.unsqueeze(-1)
        # print(f"candidates:{candidates}")        
        if score_mode == "log":
            _output = F.log_softmax(output, dim=3)
        else:
            _output = F.softmax(output, dim=3)
        scores = torch.gather(_output, 3, candidates).squeeze(-1)  # [bz, cand_num, seq_len]
        cand_mask = cand_mask.float()
        scores = torch.mul(scores, cand_mask).sum(-1) / (
                    (cand_mask.sum(-1) + 0) ** self.configs.train.length_penalty)  # [bz, cand_num]
        length = torch.repeat_interleave(tgt_len.sum(-1) - 1, 2 * cand_mask.size(1), dim=0).unsqueeze(0).view(
            cand_mask.size(0), -1)
        length = torch.abs(cand_mask.sum(-1) - length)
        length = torch.sigmoid(torch.sqrt(length))
        length = torch.full(length.size(), 1.5).to(length.device) - length
        scores = torch.mul(scores, length)
        if require_gold:
            output = {'score': scores[:, 1:], "summary_score": scores[:, 0], "probs": probs}
        else:
            output = {'score': scores, "probs": probs}
        return output

    def scoring_mode(self):
        self.model.model.scoring_mode()

    def generation_mode(self):
        self.model.model.generation_mode()

    def generate(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            max_length: Optional[int] = None,
            min_length: Optional[int] = None,
            do_sample: Optional[bool] = None,
            early_stopping: Optional[bool] = None,
            num_beams: Optional[int] = None,
            temperature: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            repetition_penalty: Optional[float] = None,
            bad_words_ids: Optional[Iterable[int]] = None,
            bos_token_id: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            length_penalty: Optional[float] = None,
            no_repeat_ngram_size: Optional[int] = None,
            encoder_no_repeat_ngram_size: Optional[int] = None,
            num_return_sequences: Optional[int] = None,
            max_time: Optional[float] = None,
            decoder_start_token_id: Optional[int] = None,
            use_cache: Optional[bool] = None,
            num_beam_groups: Optional[int] = None,
            diversity_penalty: Optional[float] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            forced_bos_token_id: Optional[int] = None,
            forced_eos_token_id: Optional[int] = None,
            remove_invalid_values: Optional[bool] = None,
            synced_gpus: Optional[bool] = None,
            **model_kwargs,
    ):
        return self.model.generate(input_ids=input_ids,
                                   max_length=max_length,
                                   min_length=min_length,
                                   do_sample=do_sample,
                                   early_stopping=early_stopping,
                                   num_beams=num_beams,
                                   temperature=temperature,
                                   top_k=top_k,
                                   top_p=top_p,
                                   repetition_penalty=repetition_penalty,
                                   bad_words_ids=bad_words_ids,
                                   bos_token_id=bos_token_id,
                                   pad_token_id=pad_token_id,
                                   eos_token_id=eos_token_id,
                                   length_penalty=length_penalty,
                                   no_repeat_ngram_size=no_repeat_ngram_size,
                                   encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
                                   num_return_sequences=num_return_sequences,
                                   max_time=max_time,
                                   decoder_start_token_id=decoder_start_token_id,
                                   use_cache=use_cache,
                                   num_beam_groups=num_beam_groups,
                                   diversity_penalty=diversity_penalty,
                                   prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states,
                                   output_scores=output_scores,
                                   return_dict_in_generate=return_dict_in_generate,
                                   forced_bos_token_id=forced_bos_token_id,
                                   forced_eos_token_id=forced_eos_token_id,
                                   remove_invalid_values=remove_invalid_values,
                                   synced_gpus=synced_gpus,
                                   **model_kwargs)
