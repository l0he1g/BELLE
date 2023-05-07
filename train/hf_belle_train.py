from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq
)
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import TrainingArguments, Trainer

class ShufflingTrainer(Trainer):
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        self.train_dataset.cache_file_names = None
        # train_dataset = self.train_dataset.shuffle(seed=self.args.seed)
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.args.train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

def train(data, model, tokenizer):
    training_args = TrainingArguments(
        #output_dir="/root/model/bloomz-7b1-mt-igpt",
                                      output_dir="output",
                                      learning_rate=5e-5,
                                      lr_scheduler_type="linear",
                                      per_device_train_batch_size=2,
                                      per_device_eval_batch_size=2,
                                      gradient_accumulation_steps=4,
                                      num_train_epochs=8,
                                      warmup_steps=200,
                                      save_steps=2000,
                                      logging_steps=200,
                                      fp16=True,
                                      save_total_limit=3,
                                      evaluation_strategy="steps",
                                      eval_steps=500,
                                      optim="adamw_torch",
                                      report_to="tensorboard",
                                      logging_dir="/root/tf-logs"                                      
                                      )
    print(training_args)
    trainer = ShufflingTrainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, max_length=512, padding="longest"),
        train_dataset=data["train"],
        eval_dataset=data["test"],
    )

    trainer.train()#resume_from_checkpoint=True)


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

    output_dir = "/root/model/bloomz-1b1-igpt"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)