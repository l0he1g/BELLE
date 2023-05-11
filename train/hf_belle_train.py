from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq
)
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import TrainingArguments, Trainer, HfArgumentParser

from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import sys

@dataclass
class ModelArguments:
    model_path: Optional[str] = field(
        default="",
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    data_path: str = field(
        default=None, metadata={"help": "Dataset path include training and test data in huggingface Dataset files."}
    )


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


def train(data, model, tokenizer, training_args):
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, max_length=512, padding="longest"),
        train_dataset=data["train"],
        eval_dataset=data["test"],
    )

    trainer.train(resume_from_checkpoint=True)


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    print(training_args)
    print("load tokenizer:" + model_args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path)
    tokenizer.padding_side = "left"

    print("load model:" + model_args.model_path)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_path)
    
    print("load data:" + data_args.data_path)
    data = load_from_disk(data_args.data_path)
    print("n(train)=%d, n(test)=%d" % (len(data["train"]), len(data["test"])))

    train(data, model, tokenizer, training_args)
    
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
