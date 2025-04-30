import os
import ast
import pandas as pd
from datasets import Dataset
import wandb
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

wandb_api_key = os.getenv("WANDB_API_KEY")
wandb.login(host="https://wandb.mlops.ingka.com", key=wandb_api_key)

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    pairs = []
    for _, row in df.iterrows():
        review = row["review"]
        all_topics = ast.literal_eval(row["all_topics"])
        selected = set(ast.literal_eval(row["selected_topics"]))
        for topic in all_topics:
            label = 1 if topic in selected else 0
            pairs.append({"review": review, "topic": topic, "label": label})
    return pd.DataFrame(pairs)

sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'eval/f1_micro', 'goal': 'maximize'},
    'parameters': {
        'learning_rate': {'min': 1e-6, 'max': 5e-5},
        'per_device_train_batch_size': {'values': [8, 16, 32]},
        'weight_decay': {'min': 0.0, 'max': 0.3},
        'num_train_epochs': {'values': [2, 4, 6]},
    }
}


def train():
    # This function will be called by wandb.agent
    run = wandb.init(job_type="training")
    config = run.config

    artifact = wandb.use_artifact("ground_truth_20k:latest")
    data_dir = artifact.download()               
    df_train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    df_test  = pd.read_csv(os.path.join(data_dir, "test.csv"))

    pairs_train_df = process_dataframe(df_train)
    pairs_test_df  = process_dataframe(df_test)

    train_ds = Dataset.from_pandas(pairs_train_df)
    test_ds  = Dataset.from_pandas(pairs_test_df)

    # 3. Tokenizer and preprocess function
    model_name = "microsoft/deberta-v3-base"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)

    def preprocess(examples):
        # Build natural-language hypotheses
        texts_topic = [f"This review is about '{t}'." for t in examples["topic"]]
        model_inputs = tokenizer(
            examples["review"],
            texts_topic,
            truncation=True,
            padding="max_length",
            max_length=256,
        )
        model_inputs["labels"] = examples["label"]
        return model_inputs

    # Map preprocessing across the datasets
    train_ds = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    test_ds  = test_ds.map(preprocess,  batched=True, remove_columns=test_ds.column_names)

    # Use dynamic padding for efficiency
    data_collator = DataCollatorWithPadding(tokenizer)

    training_args = TrainingArguments(
        output_dir        = "outputs",
        eval_strategy = "epoch",
        save_strategy       = "epoch",
        save_total_limit=1,           # <= keep only the last checkpoint
        overwrite_output_dir=True,
        logging_strategy    = "steps",
        logging_steps       = 100,
        logging_dir="./logs",
        learning_rate       = config.learning_rate,
        per_device_train_batch_size = config.per_device_train_batch_size,
        per_device_eval_batch_size  = config.per_device_train_batch_size * 2,
        weight_decay        = config.weight_decay,
        num_train_epochs    = config.num_train_epochs,
        load_best_model_at_end = True,
        greater_is_better      = True,
        report_to             = "wandb",
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )

    def compute_metrics(pred):
        labels = pred.label_ids
        preds  = pred.predictions.argmax(-1)
        from sklearn.metrics import f1_score, precision_score, recall_score
        return {
            "f1_micro":       f1_score(labels, preds, average="micro", zero_division=0),
            "precision_micro": precision_score(labels, preds, average="micro", zero_division=0),
            "recall_micro":    recall_score(labels, preds, average="micro", zero_division=0),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("./deberta-pairwise-finetuned")
    tokenizer.save_pretrained("./deberta-pairwise-finetuned")

    # 7. Log the final model as an artifact
    model_art = wandb.Artifact(
        name="deberta-v3-pairwise-finetuned",
        type="model"
    )
    model_art.add_dir("./deberta-pairwise-finetuned")
    run.log_artifact(model_art)

    run.finish()


# 5. Launch the sweep (runs until you stop or the sweep completes)
if __name__ == "__main__":
    # Create the sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project="topic-assignment",
        entity="digital-ethics-responsible-ai"
    )

    wandb.agent(sweep_id, function=train, count=5)
