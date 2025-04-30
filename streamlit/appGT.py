import streamlit as st
import pandas as pd
import random

# Page config for full-width layout
st.set_page_config(layout="wide")

# Constants
ARTICLES_CSV = 'csv/images.csv'  # Contains article_id, image_url
REVIEWS_CSV = 'csv/GroundTruthProdArea10kV3.csv'  # Contains article_id, review, all_topics, selected_topics
CAT_PLACEHOLDERS = [
    'https://placecats.com/neo/300/300',
    'https://placekitten.com/300/300',
    'https://placecats.com/millie_neo/300/300',
    'https://placecats.com/neo_2/300/300',
    'https://placecats.com/louie/300/300',
    'https://placecats.com/millie/300/300'
]

# Load and cache CSV data
@st.cache_data
def load_data():
    articles = pd.read_csv(ARTICLES_CSV, dtype={'article_id': str, 'image_url': str})
    reviews = pd.read_csv(
        REVIEWS_CSV,
        dtype={'article_id': str, 'review': str, 'all_topics': str, 'selected_topics': str}
    )
    # Parse comma-separated topics into lists
    reviews['all_topics'] = (
        reviews['all_topics'].fillna('')
               .apply(lambda x: [t.strip() for t in x.split(',')] if x else [])
    )
    reviews['selected_topics'] = (
        reviews['selected_topics'].fillna('')
               .apply(lambda x: [t.strip() for t in x.split(',')] if x else [])
    )
    return articles, reviews

# Load data once
a_df, r_df = load_data()

# Combine article_ids from both sources
all_ids = set(a_df['article_id']) | set(r_df['article_id'])
combined = pd.DataFrame({'article_id': list(all_ids)})
combined = combined.merge(a_df[['article_id', 'image_url']], on='article_id', how='left')

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
        # Random cat placeholder if missing URL
        if pd.isna(row['image_url']) or not row['image_url']:
            url = random.choice(CAT_PLACEHOLDERS)
        else:
            url = row['image_url']
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

    # Get all reviews for selected article
    subset = r_df[r_df['article_id'] == aid]
    if subset.empty:
        st.write('No reviews found for this product.')
        st.stop()

    # Build list of all topics
    topic_options = sorted({topic for topics in subset['all_topics'] for topic in topics})
    # Allow multiple selections
    selected_topics = st.multiselect('Filter by Topics (select multiple)', options=topic_options)

    # Filter reviews: must include all selected topics
    if not selected_topics:
        filtered = subset
    else:
        filtered = subset[subset['selected_topics'].apply(lambda ts: all(topic in ts for topic in selected_topics))]

    if filtered.empty:
        st.write('No reviews match the selected topic combination.')
    else:
        for idx, rev in filtered.iterrows():
            text = rev['review'] or ''
            if '.' in text:
                head, body = text.split('.', 1)
                st.markdown('---')
                st.write(f"**Title:** {head.strip()}")
                st.write(f"**Text:** {body.strip()}")
            else:
                st.markdown('---')
                st.write(f"**Text:** {text.strip()}")
            # Present selected topics cleanly
            st.write('**Selected Topics:**', ', '.join(rev['selected_topics']))
