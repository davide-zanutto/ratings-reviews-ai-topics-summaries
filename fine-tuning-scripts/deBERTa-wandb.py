import os
import ast
import pandas as pd
import torch

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
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

# ===============================
# 0. Define the sweep configuration
# ===============================
sweep_config = {
    "method": "bayes",  # other options: "grid", "random"
    "metric": {
        "name": "eval/f1_micro",
        "goal": "maximize"
    },
    "parameters": {
        "learning_rate": {
            "min": 1e-6,
            "max": 7e-5
        },
        "batch_size": {
            "values": [8, 16, 32]
        },
        "num_train_epochs": {
            "values": [5, 10, 15, 20]
        },
        "max_length": {
            "values": [128, 256]
        },
        "weight_decay": {
            "min": 0.0,
            "max": 0.3
        }
    }
}



def train():
    # 1. Start a new W&B run, pulling in the swept-in config
    run = wandb.init(job_type="training")
    config = run.config

    artifact = wandb.use_artifact("ground_truth_12k:latest")
    data_dir = artifact.download()               
    df_train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    df_test  = pd.read_csv(os.path.join(data_dir, "test.csv"))

    # 3. Preprocess
    def parse_list(text):
        try:
            return ast.literal_eval(text)
        except:
            return []

    for df in (df_train, df_test):
        df["all_topics"]      = df["all_topics"].apply(parse_list)
        df["selected_topics"] = df["selected_topics"].apply(parse_list)

    all_df        = pd.concat([df_train, df_test], ignore_index=True)
    global_topics = sorted({t for topics in all_df["all_topics"] for t in topics})
    label2id      = {topic: i for i, topic in enumerate(global_topics)}
    id2label      = {i: topic for topic, i in label2id.items()}
    num_labels    = len(global_topics)

    def combine_text(row):
        topics_str = ', '.join(row["all_topics"])
        return f"Review: {row['review']} | Topics: {topics_str}"

    def create_label_vector(selected_topics):
        vec = [0] * num_labels
        for t in selected_topics:
            vec[label2id[t]] = 1
        return [float(v) for v in vec]

    for df in (df_train, df_test):
        df["text"]   = df.apply(combine_text, axis=1)
        df["labels"] = df["selected_topics"].apply(create_label_vector)

    # 4. Build HF datasets & tokenize
    train_ds = Dataset.from_pandas(df_train[["text","labels"]], split="train")
    test_ds  = Dataset.from_pandas(df_test[["text","labels"]],  split="test")
    datasets = DatasetDict({"train": train_ds, "test": test_ds})

    model_name = "microsoft/deberta-v3-base"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=config.max_length
        )

    tokenized = datasets.map(tokenize_fn, batched=True)
    tokenized = tokenized.remove_columns(["text"])
    tokenized.set_format(
        type="torch",
        columns=["input_ids","attention_mask","labels"]
    )
    data_collator = DataCollatorWithPadding(tokenizer)

    # 5. Model & Trainer
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        problem_type="multi_label_classification",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
        from sklearn.metrics import f1_score, precision_score, recall_score
        return {
            "f1_micro":       f1_score(labels, preds, average="micro", zero_division=0),
            "precision_micro": precision_score(labels, preds, average="micro", zero_division=0),
            "recall_micro":    recall_score(labels, preds, average="micro", zero_division=0),
        }

    training_args = TrainingArguments(
        output_dir="./deberta-finetuned",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,           # <= keep only the last checkpoint
        load_best_model_at_end=True,
        overwrite_output_dir=True,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size= config.batch_size,
        num_train_epochs=config.num_train_epochs,
        weight_decay=config.weight_decay,
        logging_dir="./logs",
        logging_steps=20,
        report_to="wandb",    # enable W&B logging
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset= tokenized["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 6. Train and log
    trainer.train()
    trainer.save_model("./deberta-finetuned")
    tokenizer.save_pretrained("./deberta-finetuned")

    # 7. Log the final model as an artifact
    model_art = wandb.Artifact(
        name="deberta-v3-base-finetuned",
        type="model"
    )
    model_art.add_dir("./deberta-finetuned")
    run.log_artifact(model_art)

    run.finish()


if __name__ == "__main__":
    # Create the sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project="topic-assignment",
        entity="digital-ethics-responsible-ai"
    )

    wandb.agent(sweep_id, function=train, count=10)