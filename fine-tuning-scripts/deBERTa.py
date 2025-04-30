import pandas as pd
import ast
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import torch
import numpy as np

# ===============================
# Step 1. Load and preprocess the data
# ===============================
TRAIN_CSV = "../csv/train.csv"
TEST_CSV = "../csv/test.csv"

print("Loading training data from:", TRAIN_CSV)
print("Loading test data from:", TEST_CSV)

df_train = pd.read_csv(TRAIN_CSV)
df_test = pd.read_csv(TEST_CSV)

# Parse the string representations into Python lists
def parse_list(text):
    try:
        return ast.literal_eval(text)
    except Exception as e:
        print("Error parsing:", text, e)
        return []

for df in (df_train, df_test):
    df["all_topics"] = df["all_topics"].apply(parse_list)
    df["selected_topics"] = df["selected_topics"].apply(parse_list)

# Combine train and test to build a global topic vocabulary
all_df = pd.concat([df_train, df_test], ignore_index=True)
global_topics = sorted({topic for topics in all_df["all_topics"] for topic in topics})
label2id = {topic: i for i, topic in enumerate(global_topics)}
id2label = {i: topic for topic, i in label2id.items()}
num_labels = len(global_topics)

print("\nGlobal Topics Label Mapping:")
print(label2id)

# Function to combine review text with its topics for model input
def combine_text(row):
    topics_str = ', '.join(row["all_topics"])
    return f"Review: {row['review']} | Topics: {topics_str}"

# Function to create multi-hot label vectors
def create_label_vector(selected_topics):
    vec = [0] * num_labels
    for t in selected_topics:
        if t in label2id:
            vec[label2id[t]] = 1
    return [float(v) for v in vec]

# Apply transformations
for df in (df_train, df_test):
    df["text"] = df.apply(combine_text, axis=1)
    df["labels"] = df["selected_topics"].apply(create_label_vector)

# ===============================
# Step 2. Create Hugging Face datasets
# ===============================
train_dataset = Dataset.from_pandas(df_train[["text", "labels"]], split="train")
test_dataset = Dataset.from_pandas(df_test[["text", "labels"]], split="test")

datasets = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

# ===============================
# Step 3. Tokenize the data
# ===============================
model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

max_length = 256

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)

tokenized_datasets = datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets.set_format(type="torch",
                              columns=["input_ids", "attention_mask", "labels"])

# Data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ===============================
# Step 4. Define model and training
# ===============================
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    problem_type="multi_label_classification",
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# Metrics computation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
    from sklearn.metrics import f1_score, precision_score, recall_score
    return {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "precision_micro": precision_score(labels, preds, average="micro", zero_division=0),
        "recall_micro": recall_score(labels, preds, average="micro", zero_division=0)
    }

training_args = TrainingArguments(
    output_dir="./deberta-finetuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=15,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=20,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ===============================
# Step 5. Train and save model
# ===============================
print("Starting training...")
trainer.train()

print("Training complete. Saving model and tokenizer...")
trainer.save_model("./deberta-finetuned")
tokenizer.save_pretrained("./deberta-finetuned")