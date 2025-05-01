import os
import ast
import pandas as pd
import numpy as np

from datasets import Dataset, DatasetDict
from transformers import (
    T5TokenizerFast, 
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer
)

import wandb
from google.cloud import secretmanager

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_secret(project_id: str, secret_id: str, version_id: str = "latest") -> str:
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

# Usage
project = "923326131319"
secret  = "WANDB_API_KEY_DAVIDE"
wandb_api_key = get_secret(project, secret)
wandb.login(host="https://wandb.mlops.ingka.com", key=wandb_api_key)

sweep_config = {
    "method": "bayes",
    "metric": {"name": "eval/f1_micro", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"min": 5e-6, "max": 1e-4},
        "per_device_train_batch_size": {"values": [2, 4, 8]},
        "num_train_epochs": {"values": [5, 7, 9]},
        "weight_decay": {"min": 0.0, "max": 0.1}
    }
}

def train():
    # 1. init W&B run
    run = wandb.init(job_type="training")
    config = run.config

    # 2. load data artifact
    artifact = run.use_artifact("ground_truth_20k:latest")
    data_dir = artifact.download()
    df_train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    df_val   = pd.read_csv(os.path.join(data_dir, "test.csv"))
    print(f"Loaded train: {df_train.shape}, val: {df_val.shape}")
    print("Sample train rows:", df_train.head(2), sep="\n")

    # 3. parse topic lists
    def parse_list(txt):
        try:
            return ast.literal_eval(txt)
        except:
            return []
    for df in (df_train, df_val):
        df["all_topics"]      = df["all_topics"].apply(parse_list)
        df["selected_topics"] = df["selected_topics"].apply(parse_list)
    print("After parsing topics:")
    print(df_train[['all_topics','selected_topics']].head(2), sep="\n")

    # 4. build prompt & target strings
    def make_input(row):
        cand = "; ".join(row["all_topics"])
        return f"Review: {row['review']}\nCandidate topics: {cand}\nSelected topics:"
    
    def make_target(row):
        return "; ".join(row["selected_topics"])


    for df in (df_train, df_val):
        df["source_text"] = df.apply(make_input, axis=1)
        df["target_text"] = df.apply(make_target, axis=1)

    print("Sample source-target:")
    print(df_train[['source_text','target_text']].head(1), sep="\n")

    # 5. create HF DatasetDict
    train_ds = Dataset.from_pandas(df_train[["source_text","target_text"]]).shuffle(seed=42)
    val_ds   = Dataset.from_pandas(df_val[["source_text","target_text"]]).shuffle(seed=42)
    datasets = DatasetDict({"train": train_ds, "validation": val_ds})

    # 6. load tokenizer & model
    model_name = "google/flan-t5-small"
    tokenizer  = T5TokenizerFast.from_pretrained(model_name)
    model      = T5ForConditionalGeneration.from_pretrained(model_name)

    # 7. tokenization
    def preprocess(batch):
        inputs = tokenizer(
            batch["source_text"],
            truncation=True,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["target_text"],
                truncation=True,
            )
        batch["input_ids"]   = inputs["input_ids"]
        batch["attention_mask"] = inputs["attention_mask"]
        batch["labels"]      = labels["input_ids"]
        return batch

    tokenized = datasets.map(
        preprocess,
        batched=True,
        remove_columns=["source_text","target_text"]
    )
    tokenized.set_format(type="torch")
    print("Tokenized example input_ids, attention_mask, labels shapes:")
    example = tokenized['train'][0]
    print({
    "input_ids":   example["input_ids"].shape,
    "attention_mask": example["attention_mask"].shape,
    "labels":      example["labels"].shape})


    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 8. metrics: decode & compute micro-F1
    from sklearn.metrics import f1_score, precision_score, recall_score

    def compute_metrics(eval_pred):
        preds_ids, labels_ids = eval_pred
        # replace -100 in labels as pad token id
        labels_ids = np.where(labels_ids != -100, labels_ids, tokenizer.pad_token_id)
        preds = tokenizer.batch_decode(preds_ids, skip_special_tokens=True)
        refs  = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        y_true, y_pred = [], []
        for ref, pred in zip(refs, preds):
            ref_set  = set([t.strip() for t in ref.split(";") if t.strip()])
            pred_set = set([t.strip() for t in pred.split(";") if t.strip()])
            # union of topics for consistent vector
            all_topics = sorted(list(ref_set | pred_set))
            vec_true = [1 if t in ref_set else 0 for t in all_topics]
            vec_pred = [1 if t in pred_set else 0 for t in all_topics]
            y_true.extend(vec_true)
            y_pred.extend(vec_pred)
        return {
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0)
    }

    # 9. training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./flan-t5-small-finetuned",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        weight_decay=config.weight_decay,
        logging_dir="./logs",
        logging_steps=50,
        report_to="wandb",
        predict_with_generate=True,
    )

    # 10. Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )

    # 11. train, save, and log artifact
    trainer.train()
    trainer.save_model("./flan-t5-small-finetuned")
    tokenizer.save_pretrained("./flan-t5-small-finetuned")

    model_art = wandb.Artifact("flan-t5-small-finetuned", type="model")
    model_art.add_dir("./flan-t5-small-finetuned")
    run.log_artifact(model_art)

    run.finish()


if __name__ == "__main__":
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project="topic-assignment",
        entity="digital-ethics-responsible-ai"
    )
    wandb.agent(sweep_id, function=train, count=15)
