import pandas as pd
from google.cloud import bigquery

input_csv = "csv/GroundTruthProdArea10kV3.csv"
project_id = "ingka-feed-student-dev"
dataset_id = "RR"
output_csv = "csv/images.csv"

df_ids = pd.read_csv(input_csv, usecols=["article_id"], dtype={"article_id": str}).drop_duplicates(subset=["article_id"])

# ensure they're strings, unique, and non-null
ids = df_ids["article_id"].dropna().astype(str).unique().tolist()
print(len(ids))
# 2) Set up BigQuery client
client = bigquery.Client(project=project_id)

# 3) Query only those IDs from product_images
query = f"""
SELECT
    local_id AS article_id,
    IMAGE_URL AS image_url
FROM
    `{project_id}.{dataset_id}.product_images`
WHERE
    local_id IN UNNEST(@ids)
"""
job_config = bigquery.QueryJobConfig(
    query_parameters=[
        bigquery.ArrayQueryParameter("ids", "STRING", ids)
    ]
)

# run the query and load into a DataFrame
df_images = client.query(query, job_config=job_config).to_dataframe()

print(f"Found {len(df_images)} images for {len(ids)} IDs")
print(df_images.head(2))

# 4) Merge back to preserve original ordering (and drop any IDs not found if desired)
df_merged = pd.merge(
    df_ids,
    df_images,
    on="article_id",
    how="inner"            # only keep IDs with images; use "left" if you want all IDs even when missing
)


df_merged = df_merged.drop_duplicates(subset=["article_id"])

# 5) Write out the result
df_merged.to_csv(output_csv, index=False)
print(f"Wrote {len(df_merged)} rows to {output_csv}")