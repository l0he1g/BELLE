from datasets import load_dataset,  DatasetInfo
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer


def get_prompt(sample):
    question = sample["title"] if len(sample["title"]) > len(
        sample["desc"]) else sample["desc"]
    return "<Human>: " + question + "\n<Assistant>: "


def get_prompt_and_chosen(sample):
    question = sample["title"] if len(sample["title"]) > len(
        sample["desc"]) else sample["desc"]
    return "<Human>: " + question + "\n<Assistant>: " + sample['answer']


def tokenize_sample(sample, max_seq_len=1024):
    prompt_text = get_prompt(sample)
    tokenized_prompt_text = tokenizer(
        prompt_text, truncation=True, max_length=max_seq_len, padding=False, return_tensors=None)
    user_prompt_len = len(tokenized_prompt_text["input_ids"])

    chosen_sentence = get_prompt_and_chosen(sample)  # the accept response

    chosen_token = tokenizer(chosen_sentence,
                             max_length=max_seq_len,
                             padding=False,
                             truncation=True)

    # Make sure tokenizer.padding_side is left
    if chosen_token["input_ids"][-1] != tokenizer.eos_token_id:
        chosen_token["input_ids"].append(tokenizer.eos_token_id)
        chosen_token["attention_mask"].append(1)

    pad_token_num = sum(
        np.equal(chosen_token["input_ids"], tokenizer.pad_token_id))

    chosen_token["labels"] = [-100] * (pad_token_num+user_prompt_len) + \
        chosen_token["input_ids"][pad_token_num+user_prompt_len:]

    chosen_token["input_ids"] = torch.LongTensor(
        chosen_token["input_ids"]).squeeze(0)
    chosen_token["attention_mask"] = torch.LongTensor(
        chosen_token["attention_mask"]).squeeze(0)
    chosen_token["labels"] = torch.LongTensor(
        chosen_token["labels"]).squeeze(0)
    return chosen_token


if __name__ == "__main__":
    data_pts = {"train": "/root/data/baikeQA/baike_qa_train.json",
                "test": "/root/data/baikeQA/baike_qa_valid.json"
                }

    output_pt = "/root/dataset/baikeQA"
    tokenizer_pt = "/root/model/bloomz-1b1"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_pt)

    data = load_dataset("json", data_files=data_pts)
    print("n(train)=%d, n(test)=%d" % (len(data["train"]), len(data["test"])))

    print(data["test"][0])
    tokenized_data = data.map(
        tokenize_sample, remove_columns=data["test"].column_names)
    print(tokenized_data["test"][0])

    tokenized_data.save_to_disk(output_pt)
    print("write:" + output_pt)
