import numpy as np
import click
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5ForConditionalGeneration
from datasets import Dataset
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
import torch

from packages.utils.constants import ST_2023, SCRATCH_PATH

model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")
tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

# TODO: load sigmorphon 2023 dataset and save it to disk. 
def load_dataset(lang_code: str, extension: str) -> Dataset:
    # the dataset is tab separated, with the following unnamed columns:
        # 0: input word form
        # 1: feature tag
        # 2: gold output word form
    # load those 3 columns into a HuggingFace dataset.
    # return the dataset.
    train_file = f"{ST_2023}/{lang_code}.{extension}"
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

def run_trainer(train_dataset: Dataset, 
                val_dataset: Dataset,
                lang_code: str):
    # instantiate training arguments
    # instantiate trainer
    # train model
    # save model
    logging_dir = f"{SCRATCH_PATH}/augmentation_subset_select/byt5_logs_{lang_code}"
    output_dir = f"{SCRATCH_PATH}/augmentation_subset_select/byt5_checkpoints_{lang_code}"
    # create the logging dir and output dir if they don't exist
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=20,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=8,
        warmup_steps=1000,
        weight_decay=0.0001,
        learning_rate=1e-4,
        logging_dir=logging_dir,
        logging_steps=500, 
        save_steps=500, 
        save_total_limit=2, 
        evaluation_strategy="steps",
        eval_steps=500
    )
    train_dataset = train_dataset.map(preprocess_dataset, batched=True) 
    val_dataset = val_dataset.map(preprocess_dataset, batched=True) # TODO: currently using the same dataset for validation. fix later.
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id)
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset, 
        eval_dataset=val_dataset,
        data_collator=collator, 
        tokenizer=tokenizer
    )
    trainer.train()

@click.command()
@click.argument("lang_code", type=str)
def test_model(lang_code):
    print(f"Generating predictions for {lang_code}.")
    # load model. Get the model from the output dir. Use the latest checkpoint.
    # use AutoModelForSeq2SeqLM and load from the output dir.
    output_dir = f"{SCRATCH_PATH}/byt5_checkpoints_{lang_code}"
    most_recent_checkpoint = max(os.listdir(output_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(f"{output_dir}/{most_recent_checkpoint}")
    with open(f"{ST_2023}/{lang_code}.dev", "r") as f:
        lines = f.readlines()
    lines = [line.strip().split("\t") for line in lines]
    lines = [line[:3] for line in lines]
    dataset = Dataset.from_dict({"input": [line[0] for line in lines],
                                    "feature": [line[1] for line in lines],
                                    "output": [line[2] for line in lines]})
    inputs = [f"{dataset['input'][i]}.{dataset['feature'][i]}" for i in range(len(dataset["input"]))]
    model_inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(dataset["output"], padding=True, truncation=True, return_tensors="pt")
    model_inputs["labels"] = labels["input_ids"]
    model_inputs = model_inputs.to("cuda")
    model = model.to("cuda")
    model.eval()
    with torch.no_grad():
        outputs = model(**model_inputs)
    # Decode the outputs to be human readable.
    # print the outputs.
    outputs = tokenizer.batch_decode(outputs.logits.argmax(dim=-1), skip_special_tokens=True)
    # print a random sample of the outputs.
    for _ in range(10):
        rand_num = np.random.randint(0, len(outputs))
        print(f"Input: {inputs[rand_num]}; Output: {outputs[rand_num]}; Gold: {dataset['output'][rand_num]}")
    

@click.command()
@click.argument("lang_code", type=str)
def train_model(lang_code: str):
    print(f"Training model for {lang_code}.")
    train_fin_dataset = load_dataset(lang_code, "trn")
    val_fin_dataset = load_dataset(lang_code, "dev")
    run_trainer(train_fin_dataset, val_fin_dataset, lang_code)

@click.group()
def main():
    pass

main.add_command(train_model)
main.add_command(test_model)

if __name__ == "__main__":
    main()
