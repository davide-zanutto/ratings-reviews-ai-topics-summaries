import streamlit as st
import pandas as pd
import ast

# --- Get the filename from command-line arguments ---
"""if len(sys.argv) > 1:
    file_name = 'LLM_3shots_20351884.csv'  # Get the filename from the command-line
else:
    st.error("Please provide a CSV file as a command-line argument.")
    st.stop()"""

file_name = 'BERTopicTop8_20351884.csv'

# --- Load CSV Data ---
@st.cache_data
def load_data():
    df = pd.read_csv(f"csv/{file_name}")
    # Convert the string representation of lists into actual Python lists
    df["topics"] = df["topics"].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
    return df

data = load_data()

# --- Extract Unique Topics ---
all_topics = sorted({topic for topics in data["topics"] for topic in topics})

# --- Streamlit App Layout ---
st.title("Topic-Based Filtering of Reviews")
# Extract the first part of the filename before the first '_'
method_used = file_name.split('_')[0]
st.markdown(f"*Created with: {method_used}*", unsafe_allow_html=True)

# Multiselect for selecting multiple topics
selected_topics = st.multiselect("Select one or more topics:", options=all_topics)

# Filter reviews based on the selected topics:
# If one or more topics are selected, a review is displayed only if its labels include all the selected topics.
if selected_topics:
    filtered_data = data[data["topics"].apply(lambda topics: all(topic in topics for topic in selected_topics))]
else:
    filtered_data = data

# Display only the review text
st.markdown("### Reviews")
for idx, row in filtered_data.iterrows():
    st.write(row['review'])