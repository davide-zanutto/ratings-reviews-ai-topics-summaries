from google.cloud import bigquery
import os
from openai import AzureOpenAI
from utils.generateTopics import get_topics
from utils.generateSummaries import get_reviews_summary
from utils.getSecret import get_secret
import logging
import json
import time
import sys
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from google.cloud import secretmanager
import wandb 


start_time = time.time()

os.environ["TOKENIZER_PARALLELISM"] = "false"

# -----------------------------------------------------------------------------
# BigQuery setup
# -----------------------------------------------------------------------------
client     = bigquery.Client()
project_id = 'ingka-tugc-infra-prod'
dataset_id = 'eu_ai_content'
table_id   = 'reviews'
table_ref  = f'{project_id}.{dataset_id}.{table_id}'

article_id = '40103751'
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
# OpenAI client (for topic *generation* & summarization)
# -----------------------------------------------------------------------------
project = "923326131319"
secret  = "derai-azure"
api_key = get_secret(project, secret)

llm_client  = AzureOpenAI(
    api_key=api_key,
    api_version="2023-07-01-preview",
    azure_endpoint="https://derai-vision.openai.azure.com/",
)
model_name = "gpt-4o"

# -----------------------------------------------------------------------------
# Topic generation & summarization (unchanged)
# -----------------------------------------------------------------------------
print("Generating topics...")
topics = get_topics(reviews, llm_client, model_name)

print("Topics generated:")
for t in topics:
    print(" –", t)

print("Generating summaries...")
summary = get_reviews_summary(reviews, llm_client, model_name)
print("Summaries OK!")

# -----------------------------------------------------------------------------
# Flan-T5 setup for topic *assignment*
# -----------------------------------------------------------------------------
FLAN_ARTIFACT_NAME = 'digital-ethics-responsible-ai/topic-assignment/flan-t5-base-finetuned:latest'
LOCAL_FLAN_DIR     = 'artifacts/flan-t5-base-finetuned:v9'

project = "923326131319"
secret  = "WANDB_API_KEY_DAVIDE"
wandb_api_key = get_secret(project, secret)

if not os.path.isdir(LOCAL_FLAN_DIR):
    wandb.login(host="https://wandb.mlops.ingka.com", key=wandb_api_key)
    # first run: download from W&B
    run          = wandb.init(job_type="inference")
    artifact     = run.use_artifact(FLAN_ARTIFACT_NAME, type='model')
    artifact_dir = artifact.download()      # contains your checkpoint files
    run.finish()
else:
    artifact_dir = LOCAL_FLAN_DIR

# now point at the downloaded artifact
model_dir = artifact_dir  # adjust if your artifact lives in a subdirectory

# -----------------------------------------------------------------------------
# Load tokenizer & Flan-T5 for classification
# -----------------------------------------------------------------------------
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model     = T5ForConditionalGeneration.from_pretrained(model_dir)
model.to(device)
model.eval()

# -----------------------------------------------------------------------------
# Assign topics to each review via Flan-T5
# -----------------------------------------------------------------------------
from assignTopics import get_reviews_labels_flanT5

print("Assigning topics to reviews...")
results = []
for review in reviews:
    try:
        assigned = get_reviews_labels_flanT5(
            tokenizer, model, device, review, topics
        )
        results.append([review, assigned])
    except Exception as e:
        msg = str(e)
        if "content_filter" in msg or "ResponsibleAIPolicyViolation" in msg:
            logging.error(f"Content filter triggered for review: {review} – skipping.")
        else:
            logging.error(f"Error processing review: {review} – {e}")
        results.append([review, []])
print("Assignment OK!")

# -----------------------------------------------------------------------------
# Write JSON and launch Streamlit
# -----------------------------------------------------------------------------
print("Visualizing results...")
output_data = {
    "article_id": article_id,
    "summary":    summary,
    "reviews":    [{"review": r, "topics": t} for r, t in results]
}

output_file = f"json/flanT5_{article_id}.json"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w") as f:
    json.dump(output_data, f, indent=4)

# Fix the 'topics' field
with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)

# Determine where to apply the fix
if isinstance(data, list):
    target = data
elif isinstance(data, dict) and isinstance(data.get("reviews"), list):
    target = data["reviews"]
else:
    sys.exit("Error: JSON must be a list or a dict with a 'reviews' list.")

# Split semicolon-separated topics in place
for item in target:
    if not isinstance(item, dict):
        continue
    topics = item.get("topics")
    if not isinstance(topics, list):
        continue
    new_topics = []
    for t in topics:
        for part in t.split(";"):
            part = part.strip()
            if part:
                new_topics.append(part)
    item["topics"] = new_topics

# Overwrite the original file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Filters generated in {time.time() - start_time:.2f} seconds")
import subprocess
subprocess.run(["streamlit", "run", "streamlit/app.py", "--", output_file])
