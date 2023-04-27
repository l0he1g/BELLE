from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq
)

from datasets import load_from_disk
from transformers import TrainingArguments, Trainer


def train(data, model, tokenizer):
    training_args = TrainingArguments(output_dir="output",
                                      learning_rate=5e-5,
                                      lr_scheduler_type="cosine",
                                      per_device_train_batch_size=2,
                                      per_device_eval_batch_size=2,
                                      gradient_accumulation_steps=4,
                                      num_train_epochs=8,
                                      warmup_steps=500,
                                      save_steps=5000,
                                      logging_steps=100,
                                      fp16=True,
                                      evaluation_strategy="steps",
                                      eval_steps=500,
                                      report_to="tensorboard",
                                      logging_dir="/root/tf-logs"
                                      )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
        train_dataset=data["train"],
        eval_dataset=data["test"],
    )

    trainer.train()


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
    
    train(data, model, tokenizer)

    output_dir = "/root/model/bloomz-1b1-belle/"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)