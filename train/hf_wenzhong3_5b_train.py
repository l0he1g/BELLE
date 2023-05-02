from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq
)

from datasets import load_from_disk
from transformers import TrainingArguments, Trainer


def train(data, model, tokenizer):
    training_args = TrainingArguments(output_dir="/root/model/wenzhong-3.5b-igpt",
                                      learning_rate=5e-5,
                                      lr_scheduler_type="linear",
                                      per_device_train_batch_size=2,
                                      per_device_eval_batch_size=2,
                                      gradient_accumulation_steps=4,
                                      num_train_epochs=8,
                                      warmup_steps=0,
                                      save_steps=5000,
                                      logging_steps=1000,
                                      fp16=True,
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
    llm_pt = "/root/model/Wenzhong2.0-GPT2-3.5B-chinese/"
    print("load tokenizer:" + llm_pt)
    tokenizer = AutoTokenizer.from_pretrained(llm_pt)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = "left"

    print("load model:" + llm_pt)
    model = AutoModelForCausalLM.from_pretrained(llm_pt).half().cuda()

    data_pt = "/root/mp/BELLE/data/Belle_1M"
    print("load data:" + data_pt)
    data = load_from_disk(data_pt)
    print("n(train)=%d, n(test)=%d" % (len(data["train"]), len(data["test"])))
    
    train(data, model, tokenizer)

    output_dir = "/root/model/wenzhong-3.5b-igpt/"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)