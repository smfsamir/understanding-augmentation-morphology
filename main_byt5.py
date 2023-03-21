import os
from transformers import AutoTokenizer
from transformers import T5ForConditionalGeneration
from datasets import Dataset
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
import torch

from packages.utils.constants import ST_2023, SCRATCH_PATH

model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")
num_special_tokens = 3
tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

# TODO: load sigmorphon 2023 dataset and save it to disk. 
def load_dataset(lang_code: str) -> Dataset:
    # the dataset is tab separated, with the following unnamed columns:
        # 0: input word form
        # 1: feature tag
        # 2: gold output word form
    # load those 3 columns into a HuggingFace dataset.
    # return the dataset.
    train_file = f"{ST_2023}/{lang_code}.trn"
    with open(train_file, "r") as f:
        lines = f.readlines()
    lines = [line.strip().split("\t") for line in lines]
    lines = [line[:3] for line in lines]
    dataset = Dataset.from_dict({"input": [line[0] for line in lines],
                                    "feature": [line[1] for line in lines],
                                    "output": [line[2] for line in lines]})
    return dataset

def preprocess_dataset(dataset: Dataset) -> Dataset:
    # encode the input using the tokenizer for byt5
    # join the input and feature columns
    inputs = [f"{dataset['input'][i]}.{dataset['feature'][i]}" for i in range(len(dataset["input"]))]

    model_inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(dataset["output"], padding=True, truncation=True, return_tensors="pt")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def train_model(dataset: Dataset, lang_code: str):
    # instantiate training arguments
    # instantiate trainer
    # train model
    # save model
    logging_dir = f"{SCRATCH_PATH}/augmentation_subset_select/byt5_checkpoints_{lang_code}"
    output_dir = f"{SCRATCH_PATH}/augmentation_subset_select/byt5_models_{lang_code}"
    # create the logging dir and output dir if they don't exist
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    training_arguments = TrainingArguments(
        output_dir=f"models/{lang_code}",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=logging_dir,
        logging_steps=10
    )
    train_dataset = dataset.map(preprocess_dataset, batched=True) 
    val_dataset = dataset.map(preprocess_dataset, batched=True) # TODO: currently using the same dataset for validation. fix later.
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset, 
        eval_dataset=val_dataset,
        data_collator=collator, 
        tokenizer=tokenizer
    )
    return trainer

def main():
    fin_dataset = load_dataset("fin")
    trainer = train_model(fin_dataset, "fin")
    trainer.train()

main()