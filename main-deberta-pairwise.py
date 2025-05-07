from google.cloud import bigquery
import os
from openai import AzureOpenAI
from utils.generateTopics import get_topics
from utils.generateSummaries import get_reviews_summary
from utils.getSecret import get_secret
import logging
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import wandb 
from collections import defaultdict
import csv
import logging

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

project_id = "ingka-feed-student-dev"
dataset_id = "RR"

# 1) CTE “filtered_articles” finds all art_id’s with ≥25 reviews
# 2) CTE “random_articles” grabs 25 of those at random
# 3) Main SELECT joins back to fetch every review for each chosen art_id
# 1) Pick N random articles with between 25 and 1000 reviews
select_articles_query = f"""
SELECT
  art_id AS article_id
FROM `{table_ref}`
WHERE
  franchise = 'set-11'
  AND content_lang_code = 'en'
GROUP BY
  art_id
HAVING
  COUNT(*) BETWEEN 25 AND 1000
ORDER BY
  RAND()
LIMIT 25
"""

# 2) Take those article_ids and pull their image URLs (exportable to CSV)
select_reviews_and_images_query = f"""
WITH random_articles AS (
  {select_articles_query}
),
one_image AS (
  SELECT
    local_id    AS article_id,
    ANY_VALUE(IMAGE_URL) AS image_url
  FROM
    `{project_id}.{dataset_id}.product_images`
  GROUP BY local_id
)
SELECT
  r.art_id                      AS article_id,
  CONCAT(r.title, '. ', r.text) AS review_text,
  IFNULL(oi.image_url, '')      AS image_url
FROM
  `{table_ref}` AS r
  JOIN random_articles ra
    ON r.art_id = ra.article_id
  LEFT JOIN one_image AS oi
    ON r.art_id = oi.article_id
WHERE
  r.franchise = 'set-11'
  AND r.content_lang_code = 'en'
"""


# 1) Run the first query to get your N random article IDs
article_ids = [row.article_id
               for row in client.query(select_articles_query)]

# 2) Run the second query to get every review + image_url
query_job = client.query(select_reviews_and_images_query)

# 3) Build your in-memory structures
reviews_by_article = defaultdict(list)
image_map = {}

for row in query_job:
    reviews_by_article[row.article_id].append(row.review_text)
    # overwrite is fine since all rows for same article carry the same image_url
    image_map[row.article_id] = row.image_url

print(f"Pulled {sum(len(v) for v in reviews_by_article.values())} reviews across {len(reviews_by_article)} articles")

# 4) Write out images.csv
with open('csv/images.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['article_id', 'image_url'])
    for art_id in reviews_by_article:
        writer.writerow([art_id, image_map.get(art_id, '')])

# -----------------------------------------------------------------------------
# WandB setup
# -----------------------------------------------------------------------------

project = "923326131319"
secret  = "WANDB_API_KEY_DAVIDE"
wandb_api_key = get_secret(project, secret)
wandb.login(host="https://wandb.mlops.ingka.com", key=wandb_api_key)


# -----------------------------------------------------------------------------
# OpenAI client
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


ARTIFACT_NAME = 'digital-ethics-responsible-ai/topic-assignment/deberta-v3-pairwise-finetuned:latest'
LOCAL_MODEL_DIR = 'artifacts/deberta-v3-pairwise-finetuned:v1'

if not os.path.isdir(LOCAL_MODEL_DIR):
    # first run: download from W&B
    run      = wandb.init(project="topic-assignment", job_type="inference")
    artifact = run.use_artifact(ARTIFACT_NAME, type='model')
    artifact_dir = artifact.download()      # this folder will contain your checkpoint-3160, model.safetensors, etc.
    run.finish()  
else:
    # subsequent runs: just point at the local copy
    artifact_dir = LOCAL_MODEL_DIR

# -----------------------------------------------------------------------------
# Load tokenizer & model for classification
# -----------------------------------------------------------------------------
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(artifact_dir)
model     = AutoModelForSequenceClassification.from_pretrained(artifact_dir)
model.to(device)
model.eval()


from utils.assignTopics import get_review_labels_deBERTa_pairwise

# reviews_by_article: dict mapping art_id -> list of review texts
# llm_client, model_name, tokenizer, model, device are assumed defined

topic_rows    = []
summary_rows  = []

print("Processing articles for topic assignment and summaries...")
for article_id, reviews in reviews_by_article.items():
    print(f"● Article {article_id}: generating topics & summary")
    # 1) get the universe of topics for this article
    topics  = get_topics(reviews, llm_client, model_name)
    # 2) get one summary per article
    summary = get_reviews_summary(reviews, llm_client, model_name)
    summary_rows.append([article_id, summary])

    print(f"  → Assigning topics to {len(reviews)} reviews")
    for review in reviews:
        try:
            assigned = get_review_labels_deBERTa_pairwise(tokenizer, model, device, review, topics)
        except Exception as e:
            msg = str(e)
            if "content_filter" in msg or "ResponsibleAIPolicyViolation" in msg:
                logging.error(f"Content filter triggered for review: {review} – skipping.")
            else:
                logging.error(f"Error processing review for article {article_id}: {e}")
            assigned = []
        # accumulate a row per review
        topic_rows.append([
            article_id,
            review,
            topics,
            assigned
        ])

# ensure output dir
output_dir = "csv"
os.makedirs(output_dir, exist_ok=True)

# 3) write the topics CSV
topics_file = os.path.join(output_dir, "prototype_topics.csv")
with open(topics_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["article_id", "review", "all_topics", "selected_topics"])
    for art_id, review, all_topics, sel_topics in topic_rows:
        # Option A: use repr() to get Python‐style single‐quoted lists
        writer.writerow([
            art_id,
            review,
            repr(all_topics),
            repr(sel_topics)
        ])

# 4) write the summaries CSV
summaries_file = os.path.join(output_dir, "prototype_summaries.csv")
with open(summaries_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["article_id", "summary"])
    for art_id, summary in summary_rows:
        writer.writerow([art_id, summary])

print(f"✅ Wrote topics → {topics_file}")
print(f"✅ Wrote summaries → {summaries_file}")
print(f"Done in {time.time() - start_time:.2f}s")


run = wandb.init(
    project="topic-assignment",   # replace with your project name
    job_type="upload_csv_artifact"
)

artifact = wandb.Artifact(
    name="streamlit", 
    type="dataset", 
)

# Add local CSV files to the artifact
artifact.add_file("csv/prototype_topics.csv")
artifact.add_file("csv/prototype_summaries.csv")
artifact.add_file("csv/images.csv")

# Log (upload) the artifact to W&B
run.log_artifact(artifact)

# Optionally, wait until the artifact is uploaded before ending the run
artifact.wait()

# Finish the W&B run
run.finish()

print("Artifact uploaded!")