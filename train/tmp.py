from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq    
)

from torch.utils.data import DataLoader
from datasets import load_from_disk
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler



if __name__ == "__main__":
    model_pt = "/root/mp/BELLE/data/Belle_1w"
    data = load_from_disk(model_pt)

    tokenizer_pt = "/root/model/bloomz-1b1"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_pt)

    dataset = data["test"]  
    print(dataset[:2])
    
    print([len(ids) for ids in dataset["input_ids"]][:4])
    print([len(ids) for ids in dataset["labels"]][:4])
    data_collator = DataCollatorForSeq2Seq(tokenizer,
                                           pad_to_multiple_of=8,                                                                                      
                                           padding="longest")

    dataloader = DataLoader(dataset,
                            collate_fn=data_collator,                                  
                            batch_size=2)
    for batch in dataloader:
        print(batch)
        #exit(1)
        

