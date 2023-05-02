from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq
)

from datasets import load_from_disk
from transformers import TrainingArguments, Trainer


def train(data, model, tokenizer):
    training_args = TrainingArguments(output_dir="/root/model/pangu-2.6b-igpt",
                                      learning_rate=5e-5,
                                      lr_scheduler_type="linear",
                                      per_device_train_batch_size=2,
                                      per_device_eval_batch_size=2,
                                      gradient_accumulation_steps=4,
                                      num_train_epochs=8,
                                      warmup_steps=0,
                                      save_steps=5000,
                                      logging_steps=1000,                                      
                                      optim="adamw_torch",
                                      save_total_limit=6,
                                      evaluation_strategy="steps",
                                      eval_steps=1000,
                                      report_to="tensorboard",
                                      logging_dir="/root/tf-logs"                                      
                                      )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, max_length=1024, padding="longest"),
        train_dataset=data["train"],
        eval_dataset=data["test"],
    )

    trainer.train(resume_from_checkpoint=False)


if __name__ == "__main__":
    llm_pt = "/root/model/pangu-2.6B/"
    print("load tokenizer:" + llm_pt)
    tokenizer = AutoTokenizer.from_pretrained(llm_pt, trust_remote_code=True)    
    tokenizer.padding_side = "left"

    print("load model:" + llm_pt)
    model = AutoModelForCausalLM.from_pretrained(llm_pt, trust_remote_code=True)

    data_pt = "/root/mp/BELLE/data/Belle_1M"
    print("load data:" + data_pt)
    data = load_from_disk(data_pt)
    print("n(train)=%d, n(test)=%d" % (len(data["train"]), len(data["test"])))
    
    train(data, model, tokenizer)

    output_dir = "/root/model/pangu-2.6b-igpt/"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)