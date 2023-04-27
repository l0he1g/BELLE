### out-of-memory
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_scheduler
)
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import TrainingArguments, Trainer
from accelerate import Accelerator
from torch.optim import AdamW
from tqdm import tqdm
accelerator = Accelerator(gradient_accumulation_steps=4)

def train(data, model, tokenizer):
    training_args = TrainingArguments(output_dir="output",
                                      learning_rate=5e-5,
                                      lr_scheduler_type="cosine",
                                      per_device_train_batch_size=4,
                                      per_device_eval_batch_size=2,
                                      gradient_accumulation_steps=4,
                                      gradient_checkpointing=True,
                                      num_train_epochs=8,                                      
                                      save_steps=5000,
                                      logging_steps=500,
                                      fp16=True,
                                      evaluation_strategy="steps",
                                      eval_steps=500,
                                      report_to="tensorboard",
                                      logging_dir="/root/tf-logs",
                                      optim="adafactor", 
                                      )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
        train_dataset=data["train"],
        eval_dataset=data["test"],
    )

    trainer.train()


def train_no_trainer(data, model, tokenizer):
    data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)
    train_dataloader = DataLoader(data["train"], shuffle=True, batch_size=1, collate_fn=data_collator)

    device = accelerator.device
    model.to(device)
 
    num_epochs = 8
    num_training_steps = num_epochs * len(train_dataloader)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    lr_scheduler = get_scheduler(
        name="cosine", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
         model, optimizer, train_dataloader, lr_scheduler
    )

    gradient_accumulation_steps = 4
    progress_bar = tqdm(range(num_training_steps))
    model.train()    
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss                
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            

if __name__ == "__main__":
    llm_pt = "/root/model/bloomz-1b1"
    print("load tokenizer:" + llm_pt)
    tokenizer = AutoTokenizer.from_pretrained(llm_pt)
    tokenizer.padding_side = "left"

    print("load model:" + llm_pt)
    model = AutoModelForCausalLM.from_pretrained(llm_pt)

    data_pt = "/root/mp/BELLE/data/Belle_1M"
    print("load data:" + data_pt)
    data = load_from_disk(data_pt)
    print("n(train)=%d, n(test)=%d" % (len(data["train"]), len(data["test"])))
    
    #train(data, model, tokenizer)
    train_no_trainer(data, model, tokenizer)

    output_dir = "/root/model/bloomz-1b1-belle/"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)