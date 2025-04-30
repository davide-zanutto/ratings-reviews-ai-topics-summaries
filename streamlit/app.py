import streamlit as st
import pandas as pd
import ast

# Configure page and title
st.set_page_config(page_title="Review Visualizer", layout="wide")

# Placeholder image URL for missing images
PLACEHOLDER_IMAGE_URL = "https://via.placeholder.com/300x200?text=No+Image+Available"

# --- Data Loading ---
@st.cache_data
def load_review_data():
    df = pd.read_csv('csv/GroundTruthProdArea10kV3.csv')
    df['all_topics'] = df['all_topics'].apply(ast.literal_eval)
    df['selected_topics'] = df['selected_topics'].apply(ast.literal_eval)
    return df

@st.cache_data
def load_image_data():
    images_csv = 'csv/images.csv'
    try:
        images_df = pd.read_csv(images_csv, dtype={'article_id': str, 'image_url': str})
    except FileNotFoundError:
        images_df = pd.DataFrame(columns=['article_id', 'image_url'])
    return images_df

# Load data
df = load_review_data()
images_df = load_image_data()

# Retrieve selected_article from URL query params
query_params = st.query_params
selected_article = query_params.get('article_id', [None])[0]

if selected_article is None:
    # --- Product Catalogue View ---
    st.title("Review Visualizer")
    st.header("Product Catalogue")
    article_ids = df['article_id'].astype(str).unique()
    cols = st.columns(4)
    for idx, art_id in enumerate(article_ids):
        col = cols[idx % 4]
        # Lookup image URL or placeholder
        match = images_df[images_df['article_id'] == art_id]
        image_url = match.iloc[0]['image_url'] if not match.empty and pd.notna(match.iloc[0]['image_url']) else PLACEHOLDER_IMAGE_URL
        col.image(image_url, use_column_width=True)
        if col.button("Select", key=f"select_{art_id}"):
            st.set_query_params(article_id=art_id)
            st.rerun()
else:
    # --- Review Viewer ---
    st.title("Review Visualizer")
    st.header(f"Reviews for Article {selected_article}")
    if st.sidebar.button("← Back to Catalogue"):
        st.set_query_params()
        st.rerun()

    # Filter reviews for selected article
    article_df = df[df['article_id'].astype(str) == selected_article]

    # Sidebar topic filters
    all_topics = sorted({topic for topics in article_df['all_topics'] for topic in topics})
    st.sidebar.header("Filter Reviews by Topics")
    selected_filters = [t for t in all_topics if st.sidebar.checkbox(t)]

    # Apply filters
    if selected_filters:
        filtered_df = article_df[article_df['selected_topics'].apply(lambda topics: all(f in topics for f in selected_filters))]
    else:
        filtered_df = article_df

    # Display reviews
    st.subheader("Reviews")
    filtered_reviews = filtered_df.to_dict('records')
    if filtered_reviews:
        for idx, review in enumerate(filtered_reviews, start=1):
            review_text = review.get('review', '')
            if '.' in review_text:
                title, text = review_text.split('.', 1)
                formatted_review = f"**Title:** {title.strip()}  \n**Text:** {text.strip()}"
            else:
                formatted_review = f"**Text:** {review_text.strip()}"
            st.markdown("---")
            st.write(formatted_review)
            topics = review.get('selected_topics', [])
            st.write("**Selected Topics:**", ", ".join(topics))
    else:
        st.info("No reviews match the selected topics.")
