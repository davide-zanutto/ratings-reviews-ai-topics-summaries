from google.cloud import bigquery
import os
from openai import AzureOpenAI
from utils.generateTopics import get_topics
from utils.generateSummaries import get_reviews_summary
from utils.assignTopics import get_reviews_labels_LLM_3shots
from utils.getSecret import get_secret
import pandas as pd
import logging
import json
import time

start_time = time.time()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

client = bigquery.Client()

project_id = 'ingka-tugc-infra-prod'
dataset_id = 'eu_ai_content'
table_id = 'reviews'

table_ref = f'{project_id}.{dataset_id}.{table_id}'

# First 5 articles with under 1k reviews

# articles_1000reviews = ['00577935', '30393063', '40577943', '50361792', '20393073']

article_id = '40598766'

query = f"""
    SELECT concat(title, '. ', text) as review_text
    FROM {table_ref}
    WHERE franchise='set-11' AND content_lang_code = 'en' AND art_id = '{article_id}'
"""

query_job = client.query(query)

reviews = [row['review_text'] for row in query_job]

print(f"Processing {len(reviews)} reviews")

project = "923326131319"
secret  = "derai-azure"
api_key = get_secret(project, secret)

llm_client = AzureOpenAI(
    api_key=api_key,
    api_version="2023-07-01-preview",
    azure_endpoint="https://derai-vision.openai.azure.com/",
)

model = "gpt-4o" 

print("Generating topics...")
topics = get_topics(reviews, llm_client, model)

topics = ['Color', 'Quality', 'Price', 'Fabric', 'Delivery', 'Fit', 'Washability', 'Decor']
print("Generating summaries...")   
summary = get_reviews_summary(reviews, llm_client, model)

model = "gpt-4o-mini" 

print("Assigning topics to reviews...")


results = []
for review in reviews:
    try:
        result = get_reviews_labels_LLM_3shots(review, topics, llm_client, model)
        result = [topic for topic in eval(result) if topic in topics]
        results.append([review, result])
    except Exception as e:
        # Check for content filter issues
        if "content_filter" in str(e) or "ResponsibleAIPolicyViolation" in str(e):
            logging.error(f"Content filter triggered for review: {review} - Skipping.")
            results.append([review, []])  # Add review with empty topics
        else:
            # Log other exceptions
            logging.error(f"Error processing review: {review} - {e}")
            results.append([review, []])  # Add review with empty topics

# Create a DataFrame from the results
df = pd.DataFrame(results, columns=["review", "topics"])

# Save the DataFrame to a CSV file
output_csv = f'csv/ground_truth_{article_id}.csv'
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df.to_csv(output_csv, index=False)

print(f"CSV file saved at {output_csv}")

"""print("Visualizing results...")
# Save results and summary to a JSON structure
output_data = {
    "article_id": article_id,
    "summary": summary,
    "reviews": [{"review": row[0], "topics": row[1]} for row in results]
}

# Save JSON to file
output_file = f'json/LLM3shots_{article_id}.json'
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=4)

print(f"Filters generated in {time.time() - start_time:.2f} seconds")

import subprocess
subprocess.run(["streamlit", "run", "app.py", "--", output_file])"""