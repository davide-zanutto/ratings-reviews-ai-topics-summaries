import streamlit as st
import pandas as pd
import wandb
from utils.getSecret import get_secret
import re

# Page config for full-width layout
st.set_page_config(layout="wide")

@st.cache_resource(show_spinner=False)  # caches the return value once per session
def load_wandb_artifact():
    project = "923326131319"
    secret  = "WANDB_API_KEY_DAVIDE"
    wandb_api_key = get_secret(project, secret)
    wandb.login(host="https://wandb.mlops.ingka.com", key=wandb_api_key)

    run = wandb.init(project="topic-assignment", job_type="UI")
    artifact = run.use_artifact(
        'digital-ethics-responsible-ai/topic-assignment/streamlit:latest',
        type='dataset'
    )
    artifact_dir = artifact.download()
    run.finish()
    return artifact_dir

# call this once; subsequent reruns will reuse the cached directory
artifact_dir = load_wandb_artifact()

print(f"Artifact directory: {artifact_dir}")

# Constants
IMAGES_CSV     = f'{artifact_dir}/images.csv'               # Contains article_id, image_url
REVIEWS_CSV    = f'{artifact_dir}/prototype_topics.csv'       # Contains article_id, review, all_topics, selected_topics
SUMMARIES_CSV  = f'{artifact_dir}/prototype_summaries.csv'    # Contains article_id, summary
PLACEHOLDER    = 'https://placehold.co/300x300?text=No+Image+Available'


# Load and cache CSV data
@st.cache_data
def load_data():
    articles = pd.read_csv(IMAGES_CSV,    dtype={'article_id': str, 'image_url': str})
    reviews  = pd.read_csv(REVIEWS_CSV,   dtype={'article_id': str, 'review': str, 'all_topics': str, 'selected_topics': str})
    summaries= pd.read_csv(SUMMARIES_CSV, dtype={'article_id': str, 'summary': str})


    # helper to clean one comma-separated string
    def clean_list(x):
        if not x or pd.isna(x):
            return []
        # split on commas, then remove [, ], ' and " from each chunk
        return [
            re.sub(r"[\[\]'\"\s]*", "", t) 
            for t in x.split(",") 
            if t.strip()
        ]

    reviews['all_topics'] = reviews['all_topics'].apply(clean_list)
    reviews['selected_topics'] = reviews['selected_topics'].apply(clean_list)

    return articles, reviews, summaries

# Load data once
a_df, r_df, s_df = load_data()

# Combine article_ids from reviews only
all_ids = set(r_df['article_id'])
combined = pd.DataFrame({'article_id': list(all_ids)})
combined = combined.merge(
    a_df[['article_id', 'image_url']],
    on='article_id',
    how='left'
)


# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'select'
if 'article' not in st.session_state:
    st.session_state.article = None

# Page: Article selection
if st.session_state.page == 'select':
    st.title('Select a Product')
    cols = st.columns(3)
    for i, row in combined.iterrows():
        col = cols[i % 3]
        url = row['image_url'] if pd.notna(row['image_url']) and row['image_url'] else PLACEHOLDER
        col.image(url, use_container_width=True)
        if col.button(f"Select {row['article_id']}", key=row['article_id']):
            st.session_state.article = row['article_id']
            st.session_state.page = 'reviews'
            st.rerun()

# Page: Reviews & multi-topic filter
elif st.session_state.page == 'reviews':
    aid = st.session_state.article
    st.title(f'Reviews for Article {aid}')

    # Back button
    if st.button('Back to Products'):
        st.session_state.page = 'select'
        st.session_state.article = None
        st.rerun()

    summary_row = s_df[s_df['article_id'] == str(aid)]
    if not summary_row.empty:
        st.subheader("Summary")
        st.write(summary_row['summary'].iloc[0])

        st.markdown("<hr style='border: double 1px;'>", unsafe_allow_html=True)

    # Get all reviews for selected article
    subset = r_df[r_df['article_id'] == aid]
    st.subheader("Reviews")
    if subset.empty:
        st.write('No reviews found for this product.')
        st.stop()

    # Build list of all topics
    topic_options = sorted({topic for topics in subset['all_topics'] for topic in topics})

    # Allow multiple selections
    selected_topics = st.multiselect('Filter by Topics', options=topic_options)

    # Filter reviews: must include all selected topics
    if not selected_topics:
        filtered = subset
    else:
        filtered = subset[
            subset['selected_topics']
                  .apply(lambda ts: all(topic in ts for topic in selected_topics))
        ]

    if filtered.empty:
        st.write('No reviews match the selected topic combination.')
    else:
        for idx, rev in filtered.iterrows():
            text = rev['review'] or ''
            if '.' in text:
                head, body = text.split('.', 1)
                st.write(f"**Title:** {head.strip()}")
                st.write(f"**Text:** {body.strip()}")
            else:
                st.write(f"**Text:** {text.strip()}")
            st.write('**Selected Topics:**', ', '.join(rev['selected_topics']))
            st.markdown('---')
