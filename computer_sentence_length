import os
import nltk 
from tqdm import tqdm
import jsonpath
import json
from multiprocessing import Pool

def clean_text(text):
    text = text.replace("\n", "")
    text = text.replace("\\", "")
    return text

def process_file(file_name):
    with open(f'{dir_path}/{file_name}', 'r') as f:
        data = json.load(f)
        tgt_text = "".join(item for item in data[0]['summary'])
        tgt_text = clean_text(tgt_text)
        ref_length = len(nltk.word_tokenize(tgt_text))
        orgin_cands = jsonpath.jsonpath(data[0], "$..candidates[*]")[0:16]
        candidates_ = sorted(orgin_cands, key=lambda x: (x["rouge-1"] + x["rouge-2"] + x["rouge-l"]) / 3, reverse=True)
        candidates = ["".join(candidate["orgin_candidate"]) for candidate in candidates_]
        try:
            cands_length = [len(nltk.word_tokenize(item)) for item in candidates]
            result = sorted(cands_length, reverse=True)
            min_cand_length = result[-1]
            max_cand_length = result[0]
            result_ = [ref_length/item for item in cands_length]
            result_ = sorted(result_, reverse=True)
            min_rc = result_[-1]
            max_rc = result_[0]
            return ref_length, min_cand_length, max_cand_length, min_rc, max_rc, 1
        except:
            return 0, 0, 0, 0, 0, 0

if __name__ == '__main__':
    dir_path = r"/root/autodl-tmp/project/Abstract_Summarization_demo2/data/new_cnndm_cased/train"
    file_names = os.listdir(dir_path)
    len(file_names)

    ref_length_mean, min_cand_length, max_cand_length, min_rc, max_rc, count = 0, 0, 0, 0, 0, 0 

    with Pool(processes=4) as pool:
        results = list(tqdm(pool.imap(process_file, file_names), total=len(file_names)))

    for result in results:
        ref_length_mean += result[0]
        min_cand_length += result[1]
        max_cand_length += result[2]
        min_rc += result[3]
        max_rc += result[4]
        count += result[5]

    print(ref_length_mean/count)
    print(min_cand_length/count)
    print(max_cand_length/count)
    print(min_rc/count)
    print(max_rc/count)
