from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    default_data_collator,
    DataCollatorForSeq2Seq
)

from datasets import load_from_disk
from transformers import TrainingArguments, Trainer


def train(data, model, tokenizer):
    training_args = TrainingArguments(
        output_dir="/root/model/illm",
        #learning_rate=5e-5,
        #per_device_eval_batch_size=1,
        gradient_checkpointing=True,
        num_train_epochs=8,
        save_steps=5000,
        save_total_limit=6,
        evaluation_strategy="steps",
        eval_steps=1000,
        logging_steps=500,
        warmup_steps=100,
        #lr_scheduler_type="linear",
        #optim="adamw_torch",
        # deepspeed="/root/mp/BELLE/train/ds_illm.json",
        report_to="tensorboard",
        logging_dir="/root/tf-logs"
    )
    print(training_args)
    trainer = Trainer(
        model=model,
        args=training_args,
        # DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, padding="longest"),
        data_collator=default_data_collator,
        train_dataset=data["train"],
        eval_dataset=data["test"],
    )

    trainer.train()  # resume_from_checkpoint=True)


if __name__ == "__main__":
    llm_pt = "/root/model/ocn-llama"
    print("load tokenizer:" + llm_pt)
    tokenizer = AutoTokenizer.from_pretrained(llm_pt)
    tokenizer.padding_side = "left"
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    print("load model:" + llm_pt)
    model = LlamaForCausalLM.from_pretrained(llm_pt)
    # model.resize_token_embeddings(len(tokenizer))

    data_pt = "/root/mp/BELLE/data/Belle_1M"
    print("load data:" + data_pt)
    data = load_from_disk(data_pt)
    print("n(train)=%d, n(test)=%d" % (len(data["train"]), len(data["test"])))

    train(data, model, tokenizer)

    output_dir = "/root/model/illm-7b"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
