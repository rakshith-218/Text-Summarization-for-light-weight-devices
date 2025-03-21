import streamlit as st
import pandas as pd
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS

nltk.download('punkt')

# Function to rank articles using LexRank
def rank_articles(df):
    documents = df['article'].tolist()
    lxr = LexRank(documents, stopwords=STOPWORDS['en'])
    ranks = lxr.rank_sentences(documents, threshold=None, fast_power_method=False)
    ranked_indices = np.argsort(ranks)[-5:][::-1]  # Select top 5 ranked sentences
    return df.iloc[ranked_indices]

# Streamlit app
def main():
    # Initialize session state variables if they don't exist
    if 'summary_done' not in st.session_state:
        st.session_state.summary_done = False
    if 'show_summarized_data' not in st.session_state:
        st.session_state.show_summarized_data = False
    if 'show_ranked_articles' not in st.session_state:
        st.session_state.show_ranked_articles = False

    st.title("Text Summarization")

    # Sidebar options
    st.sidebar.header("Upload or Enter Data")
    
    # File upload in sidebar
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    df = None
    csv_uploaded = uploaded_file is not None

    if csv_uploaded:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.write(df)
    else:
        st.sidebar.write("Or enter the data manually:")
        num_rows = st.sidebar.number_input("Number of rows", min_value=1, value=1)
        articles = []
        highlights = []
        for i in range(num_rows):
            st.sidebar.write(f"**Input for Article {i+1}**")
            article = st.sidebar.text_area(f"Article {i+1}", key=f"article_{i}")
            highlight = st.sidebar.text_area(f"Highlight {i+1} (Optional)", key=f"highlight_{i}")
            articles.append(article)
            highlights.append(highlight)
        df = pd.DataFrame({"article": articles, "highlights": highlights})

    # Minimum and maximum sentence count inputs for centrality-based summaries
    st.sidebar.header("Summary Length Settings")
    min_sentences = st.sidebar.number_input("Minimum sentences", min_value=1, value=2)
    max_sentences = st.sidebar.number_input("Maximum sentences", min_value=1, value=5)

    # If the summarization has been done previously, load the results from session state
    if st.session_state.summary_done:
        df = st.session_state.df  # Load previously summarized data

    # Summarize button in sidebar
    if st.sidebar.button("Summarize"):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        df['ext-highlights'] = ""
        df['summary_cosim'] = ""

        for row in range(df.shape[0]):
            article = df['article'][row]
            highlight = df['highlights'][row]

            # Extractive Summarization (based on similarity to highlight or general sentences)
            art_list = nltk.sent_tokenize(article)
            art_embed = model.encode(art_list, convert_to_tensor=True)

            if highlight:
                high_embed = model.encode(highlight, convert_to_tensor=True)
                sim_scores = util.pytorch_cos_sim(high_embed, art_embed)[0]
            else:
                # Without highlights, compute similarity among article sentences
                sim_scores = util.pytorch_cos_sim(art_embed, art_embed).mean(dim=0)

            # Select top-k sentences for extractive highlights
            top_k = 5  # Number of sentences to select
            top_indices = np.argsort(sim_scores.cpu().numpy())[-top_k:].tolist()
            selected_sentences = [art_list[i] for i in sorted(top_indices)]

            # Remove duplicates for coherence
            seen = set()
            coherent_ext_summary = []
            for sentence in selected_sentences:
                if sentence not in seen:
                    coherent_ext_summary.append(sentence)
                    seen.add(sentence)

            df.loc[row, 'ext-highlights'] = " ".join(coherent_ext_summary)

            # Centrality-based Summarization
            sentences = nltk.sent_tokenize(article)
            embeddings = model.encode(sentences, convert_to_tensor=True)

            # SVD for dimensionality reduction (optional)
            svd = TruncatedSVD(n_components=14)
            B = svd.fit_transform(embeddings)

            # Calculate cosine similarity and centrality scores
            cos_scores = cosine_similarity(embeddings)
            centrality_scores = cos_scores.sum(axis=1)
            most_central_sentence_indices = np.argsort(-centrality_scores)

            # Adjust summary length based on user-defined min and max sentence counts
            coherent_summary = []
            seen_sentences = set()
            for idx in most_central_sentence_indices:
                if len(coherent_summary) >= max_sentences:
                    break
                sentence = sentences[idx].strip()
                if sentence not in seen_sentences:
                    coherent_summary.append(sentence)
                    seen_sentences.add(sentence)

            # If summary length is below the minimum, pad with additional sentences
            while len(coherent_summary) < min_sentences:
                next_idx = len(coherent_summary)
                if next_idx < len(sentences):
                    coherent_summary.append(sentences[next_idx].strip())

            df.at[row, 'summary_cosim'] = " ".join(coherent_summary)

        # Store the summarized data in session state
        st.session_state.df = df
        st.session_state.summary_done = True  # Mark summarization as done

        # Display the summary based on article count and file upload
        if not csv_uploaded and df.shape[0] < 10:
            st.write("Summarized Data (Paragraph Format):")
            for i, row in df.iterrows():
                st.write(f"Article {i+1} Summary:")
                st.write("Extracted Summary:")
                st.write(row['ext-highlights'])
                st.write("Centrality-based Summary:")
                st.write(row['summary_cosim'])
        else:
            # Toggle between summarized data and ranked articles if more than 10 articles or a CSV is uploaded
            st.session_state.show_summarized_data = False
            st.session_state.show_ranked_articles = False

    # Sidebar buttons to toggle between summarized data and ranked articles if CSV is uploaded or articles >= 10
    if csv_uploaded or (df.shape[0] >= 10):
        st.sidebar.header("View Options")
        if st.sidebar.button("Show Summarized Data"):
            st.session_state.show_summarized_data = True
            st.session_state.show_ranked_articles = False

        if st.sidebar.button("Show Top 5 Ranked Articles"):
            st.session_state.show_ranked_articles = True
            st.session_state.show_summarized_data = False

        # Display the selected data frame based on toggle state
        if st.session_state.show_summarized_data:
            st.write("Summarized Data:")
    
         # Ensure 'ext-highlights' and 'summary_cosim' exist before displaying
            required_columns = ['article', 'ext-highlights', 'summary_cosim']
            available_columns = [col for col in required_columns if col in df.columns]
    
            st.dataframe(df[available_columns])  # Display only existing columns


        if st.session_state.show_ranked_articles:
            # Rank articles using LexRank
            ranked_df = rank_articles(df)
            st.write("Top 5 Ranked Articles:")
            st.dataframe(ranked_df[['article', 'ext-highlights', 'summary_cosim']])

if __name__ == "__main__":
    main()
