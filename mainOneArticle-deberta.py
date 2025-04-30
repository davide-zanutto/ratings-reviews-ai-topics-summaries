from google.cloud import bigquery
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from generateTopics import get_topics
from generateSummaries import get_reviews_summary
import pandas as pd
import logging
import json
import time

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import wandb  # ← new

start_time = time.time()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------------------------------------------------------------
# BigQuery setup
# -----------------------------------------------------------------------------
client     = bigquery.Client()
project_id = 'ingka-tugc-infra-prod'
dataset_id = 'eu_ai_content'
table_id   = 'reviews'
table_ref  = f'{project_id}.{dataset_id}.{table_id}'

article_id = '20275814'
query = f"""
    SELECT concat(title, '. ', text) as review_text
    FROM {table_ref}
    WHERE franchise='set-11'
      AND content_lang_code = 'en'
      AND art_id = '{article_id}'
"""
query_job = client.query(query)
reviews   = [row['review_text'] for row in query_job]
print(f"Processing {len(reviews)} reviews")

# -----------------------------------------------------------------------------
# OpenAI client
# -----------------------------------------------------------------------------
load_dotenv()
api_key     = os.getenv("API_KEY")

wandb_api_key = os.getenv("WANDB_API_KEY")
wandb.login(host="https://wandb.mlops.ingka.com", key=wandb_api_key)

llm_client  = AzureOpenAI(
    api_key=api_key,
    api_version="2023-07-01-preview",
    azure_endpoint="https://derai-vision.openai.azure.com/",
)
model_name = "gpt-4o"

# -----------------------------------------------------------------------------
# Topic generation & summarization
# -----------------------------------------------------------------------------
print("Generating topics...")
topics = get_topics(reviews, llm_client, model_name)

print("Topics generated:")
for t in topics:
    print(" –", t)

print("Generating summaries...")
summary = get_reviews_summary(reviews, llm_client, model_name)

ARTIFACT_NAME = 'digital-ethics-responsible-ai/topic-assignment/deberta-v3-base-finetuned:latest'
LOCAL_MODEL_DIR = 'models/deberta-v3-base-finetuned:v11'

if not os.path.isdir(LOCAL_MODEL_DIR):
    # first run: download from W&B
    run      = wandb.init(job_type="inference")
    artifact = run.use_artifact(ARTIFACT_NAME, type='model')
    artifact_dir = artifact.download()      # this folder will contain your checkpoint-3160, model.safetensors, etc.
    run.finish()  
else:
    # subsequent runs: just point at the local copy
    artifact_dir = LOCAL_MODEL_DIR

# -----------------------------------------------------------------------------
# Download fine-tuned DeBERTa from W&B
# -----------------------------------------------------------------------------
# Make sure WANDB_API_KEY is set in your env, or call wandb.login()


# now point at the downloaded artifact
model_dir = 'artifacts/deberta-v3-base-finetuned:v11'  # replace with the actual path to your downloaded artifact

# -----------------------------------------------------------------------------
# Load tokenizer & model for classification
# -----------------------------------------------------------------------------
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.to(device)
model.eval()

# -----------------------------------------------------------------------------
# Assign topics to each review
# -----------------------------------------------------------------------------
from assignTopics import get_reviews_labels_deBERTa

print("Assigning topics to reviews...")
results = []
for review in reviews:
    try:
        assigned = get_reviews_labels_deBERTa(tokenizer, model, device, review, topics)
        results.append([review, assigned])
    except Exception as e:
        msg = str(e)
        if "content_filter" in msg or "ResponsibleAIPolicyViolation" in msg:
            logging.error(f"Content filter triggered for review: {review} – skipping.")
        else:
            logging.error(f"Error processing review: {review} – {e}")
        results.append([review, []])

# -----------------------------------------------------------------------------
# Write JSON and launch Streamlit
# -----------------------------------------------------------------------------
print("Visualizing results...")
output_data = {
    "article_id": article_id,
    "summary":    summary,
    "reviews":    [{"review": r, "topics": t} for r, t in results]
}

output_file = f"json/deBERTa_{article_id}.json"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w") as f:
    json.dump(output_data, f, indent=4)

print(f"Filters generated in {time.time() - start_time:.2f} seconds")
import subprocess
subprocess.run(["streamlit", "run", "streamlit/app.py", "--", output_file])
