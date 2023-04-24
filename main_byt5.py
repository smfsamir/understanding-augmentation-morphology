import evaluate
import pdb
from functools import partial
import numpy as np
import click
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5ForConditionalGeneration
from datasets import Dataset, concatenate_datasets, load_from_disk, disable_caching
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch

from packages.utils.constants import ST_2023, SCRATCH_PATH

model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")
tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

def load_dataset(lang_code: str, extension: str, is_covered: bool=False) -> Dataset:
    train_file = f"{ST_2023}/{lang_code}.{extension}"
    with open(train_file, "r") as f:
        lines = f.readlines()
    lines = [line.strip().split("\t") for line in lines]
    # if is_covered, then there will be no output column.
    if is_covered:
        dataset = Dataset.from_dict({"input": [f"<{lang_code}>{line[0]}" for line in lines],
                                    "feature": [line[1] for line in lines]})
    else:
        dataset = Dataset.from_dict({"input": [f"<{lang_code}>{line[0]}" for line in lines],
                                        "feature": [line[1] for line in lines],
                                        "output": [line[2] for line in lines]})
    return dataset

# to make it batched, we have to return a dictionary with the new elements to be added.
def preprocess_dataset(batch: Dataset, is_labelled: bool=True) -> Dataset:
    # encode the input using the tokenizer for byt5
    # join the input and feature columns

    # TODO: can this be done better?
    inputs = [f"{batch['input'][i]}+{batch['feature'][i]}" for i in range(len(batch["input"]))]
    new_batch = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
    if is_labelled:
        with tokenizer.as_target_tokenizer():
            label_input_ids = tokenizer(batch["output"], padding=True, truncation=True, return_tensors="pt")
        new_batch["labels"] = label_input_ids["input_ids"]
        return new_batch
    else:
        return new_batch

def run_trainer(train_dataset: Dataset, 
                val_dataset: Dataset,
                lang_code: str):
    # instantiate training arguments
    # instantiate trainer
    # train model
    # save model
    logging_dir = f"{SCRATCH_PATH}/byt5_logs_{lang_code}"
    output_dir = f"{SCRATCH_PATH}/byt5_checkpoints_{lang_code}"
    # create the logging dir and output dir if they don't exist
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    training_arguments = Seq2SeqTrainingArguments(
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
        eval_steps=500,
        # use accuracy as the metric for best model
        eval_accumulation_steps=20
    )
    train_dataset = train_dataset.map(preprocess_dataset, batched=True) 
    val_dataset = val_dataset.map(preprocess_dataset, batched=True) # TODO: currently using the same dataset for validation. fix later.
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    # trainer = Trainer(
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset, 
        eval_dataset=val_dataset,
        data_collator=collator, 
        tokenizer=tokenizer
    )
    trainer.train()

metric = evaluate.load("accuracy")
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    print(f"Preds: {preds}")
    print(f"Length of predictions: {len(preds)}")
    print(f"Label shape: {labels.shape}")

    labels[labels == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
    acc = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"accuracy": acc}

@click.command()
def test_model():
    disable_caching()
    test_dataset = load_from_disk(f"{SCRATCH_PATH}/byt5_all_val_dataset")

    # filter the test dataset to only get the ones where the "input" column starts with "<eng>".
    test_dataset = test_dataset.filter(lambda example: example["input"].startswith("<eng>"))

    # load model. Get the model from the output dir. Use the latest checkpoint.
    # use AutoModelForSeq2SeqLM and load from the output dir.
    output_dir = f"{SCRATCH_PATH}/byt5_checkpoints_all"
    most_recent_checkpoint = max(os.listdir(output_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(f"{output_dir}/{most_recent_checkpoint}")
    test_dataset = test_dataset.map(preprocess_dataset, batched=True, load_from_cache_file=False) # TODO: currently using the same dataset for validation. fix later.
    batch_size = 16

    # run the model on the test dataset. Use a batch size of 16.
    # get the predictions and the labels.
    # compute the accuracy.
    # print the accuracy.

    num_correct = 0
    num_total = 0
    pdb.set_trace()
    for i in range(0, len(test_dataset), batch_size):
        batch = test_dataset[i:i+batch_size]

        model.eval()
        with torch.no_grad():
            # only need to pass the input_ids and attention_mask
            input_batch = {k: v for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
            input_batch['input_ids'] = torch.tensor(input_batch['input_ids'])
            input_batch['attention_mask'] = torch.tensor(input_batch['attention_mask'])
            outputs = model(decoder_input_ids=input_batch['input_ids'], **input_batch)
        
        # Generate the predictions for the batch
        generated_texts = tokenizer.batch_decode(outputs.logits.argmax(dim=-1), skip_special_tokens=True)
        num_correct += sum([1 if generated_texts[i] == batch['output'][i] else 0 for i in range(len(generated_texts))])
        num_total += len(generated_texts)
    print(f"Overall accuracy: {num_correct/num_total}")

@click.command()
@click.option("--construct_arrow_dataset", is_flag=True, default=False) # should be on the first time we run the training script.
def generate_covered_predictions(construct_arrow_dataset):
    if construct_arrow_dataset:
        lang_codes = []
        for fname in os.listdir(ST_2023):
            lang_codes.append(fname.split(".")[0])
        lang_codes = set(lang_codes)
        covered_datasets = []
        for lang_code in lang_codes:
            # the extension is .covered.tst
            covered_dataset = load_dataset(lang_code, "covered.tst", is_covered=True)
            covered_datasets.append(covered_dataset)
        covered_dataset = concatenate_datasets(covered_datasets)
        covered_dataset.save_to_disk(f"{SCRATCH_PATH}/byt5_all_covered_dataset")
    else:
        covered_dataset = load_from_disk(f"{SCRATCH_PATH}/byt5_all_covered_dataset")
    output_dir = f"{SCRATCH_PATH}/byt5_checkpoints_all"
    most_recent_checkpoint = max(os.listdir(output_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(f"{output_dir}/{most_recent_checkpoint}")
    preprocess_covered_dataset = partial(preprocess_dataset, is_labelled=False)
    covered_dataset = covered_dataset.map(preprocess_covered_dataset, batched=True) # TODO: currently using the same dataset for validation. fix later.
    batch_size = 16

    # generate the predictions for the covered dataset.
    # save the predictions to a file.
    for i in range(0, len(covered_dataset), batch_size):
        batch = covered_dataset[i:i+batch_size]
        input_ids = [example["input_values"] for example in batch]
        # Generate the predictions for the batch
        generated_ids = model.generate(
            input_ids,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
        generated_texts = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for j in range(len(generated_texts)):
            print(generated_texts[j])

    # test_model(covered_dataset)

@click.command()
# add a flag, indicating whether or not to load the datasets from scratch in order to save them to disk.
@click.option("--construct_arrow_dataset", is_flag=True, default=False) # should be on the first time we run the training script.
def train_model(construct_arrow_dataset: bool):
    if construct_arrow_dataset:
        lang_codes = []
        for fname in os.listdir(ST_2023):
            lang_codes.append(fname.split(".")[0])
        lang_codes = set(lang_codes)
        train_datasets = []
        val_datasets = []
        for lang_code in lang_codes:
            train_dataset = load_dataset(lang_code, "trn")
            train_datasets.append(train_dataset)
            val_dataset = load_dataset(lang_code, "dev")
            val_datasets.append(val_dataset)
        train_dataset = concatenate_datasets(train_datasets)
        val_dataset = concatenate_datasets(val_datasets)
        train_dataset.save_to_disk(f"{SCRATCH_PATH}/byt5_all_train_dataset") # TODO: change the path
        val_dataset.save_to_disk(f"{SCRATCH_PATH}/byt5_all_val_dataset") # TODO: change the path
    else:
        train_dataset = load_from_disk(f"{SCRATCH_PATH}/byt5_all_train_dataset")
        val_dataset = load_from_disk(f"{SCRATCH_PATH}/byt5_all_val_dataset")
    run_trainer(train_dataset, val_dataset, "all")

@click.group()
def main():
    pass

main.add_command(train_model)
main.add_command(test_model)

if __name__ == "__main__":
    main()