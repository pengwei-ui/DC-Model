import argparse
import random
from augmentation import DocumentAugmentation
import os
import json
from rouge import Rouge
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PegasusForConditionalGeneration, PegasusTokenizer
from transformers import BartTokenizer, BartForConditionalGeneration
import nlpaug.augmenter.word as naw

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATA_DIR = "cnndm"
N = 3
device = "cuda" if torch.cuda.is_available() else "cpu"

if DATA_DIR == "cnndm":
    mname = r"../model/facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(mname)
    model = model.to(device)
    model.eval()
    tokenizer = BartTokenizer.from_pretrained(mname)
else:
    mname = r"../model/google/pegasus-xsum"
    model = PegasusForConditionalGeneration.from_pretrained(mname)
    model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(mname)
rouge = Rouge()

augmentation1, augmentation2, augmentation3 = "randomswap", "randomdelete", "backtranslation"
if augmentation1 is not None and augmentation2 is not None and augmentation3 is not None:
    AUGMENTATION = sorted([augmentation1, augmentation2, augmentation3])
else:
    print(f"No Valid Augmentation Methods")

augs = ["randomdelete", "randomswap", "backtranslation"]

if __name__ == '__main__':
    max_length = 140
    min_length = 55
    for element in ['train', 'validation', 'test']:
        # replace you data path:need csv files
        data = pd.read_csv(f"../data/{DATA_DIR}_csv/{element}.csv",
                           encoding='utf8')
        data = data.dropna()
        data.reset_index(drop=True, inplace=True)
        documents, summarys = data['document'], data['summary']
        documents, summarys = documents.values.tolist(), summarys.values.tolist()
        count = 1
        for index in range(len(documents)):
            datalist, sent, sents, candidates = [], [], [], []
            global aug_summary

            src_texts = documents[index:index + 1]
            tgt_texts = summarys[index:index + 1]

            for i in range(len(AUGMENTATION)):
                method = AUGMENTATION[i]
                # set the seed
                if i == 0:
                    random.seed(97)
                elif i == 1:
                    random.seed(41)
                augmentation = DocumentAugmentation(n=N, input=src_texts[0])
                if method.lower() == 'randomswap':
                    augmentation.RandomSwap()
                elif method.lower() == 'randomdelete':
                    augmentation.RandomDeletion()
                elif method.lower() == "backtranslation":
                    augmentation.BackTranslation()

                sent.append(augmentation.augmented_sentences)
                aug_summary = augmentation.BackTranslation_sentence(tgt_texts[0])

            for index in range(len(sent)):
                sents.append("".join(item for item in sent[index]))

            for index in range(len(sents)):
                batch = tokenizer.prepare_seq2seq_batch(src_texts=sents[index], tgt_texts=tgt_texts, truncation=True,
                                                        padding="longest", return_tensors="pt", max_length=1024).to(
                    device)
                generates = model.generate(**batch, num_return_sequences=8, num_beam_groups=8, diversity_penalty=0.3,
                                           num_beams=8, length_penalty=0.6, max_length=max_length + 2,
                                           min_length=min_length + 1,  # +1 from original because we start at step=1
                                           no_repeat_ngram_size=3,
                                           early_stopping=True)
                res = tokenizer.batch_decode(generates, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                for re in res:
                    if re is None:
                        re = ""
                    try:
                        rouge_score = rouge.get_scores(re, tgt_texts[0])
                        candidates.append(
                            {f"{augs[index]}_candidate": re, 'rouge-1': rouge_score[0]['rouge-1']['f'],
                             'rouge-2': rouge_score[0]['rouge-2']['f'],
                             'rouge-l': rouge_score[0]['rouge-l']['f']})
                    except:
                        candidates.append({f"{augs[index]}_candidate": re, 'rouge-1': 0.0,
                                           'rouge-2': 0.0, 'rouge-l': 0.0})

            datalist.append(
                {"document": src_texts, "document_aug": sents, "summary": tgt_texts,
                 "aug_summary": aug_summary, "candidates": candidates})
            with open(f"../data/{DATA_DIR}/{element}/{count}.json", 'w') as file:
                json.dump(datalist, file)
                count += 1
