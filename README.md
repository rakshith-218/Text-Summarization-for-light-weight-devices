# Text Summarization Using SBERT and SVD

## Overview
This repository contains a **text summarization system** that utilizes **SBERT embeddings, Singular Value Decomposition (SVD), and Cosine Similarity** to generate extractive and centrality-based summaries. The system is designed to process multiple articles or paragraphs, summarize them efficiently, and rank them based on importance.

## Features
- **Dual Summarization Methods**: 
  - **Extractive Summarization**: Identifies key sentences based on similarity to provided highlights.
  - **Centrality-Based Summarization**: Selects important sentences using sentence centrality and dimensionality reduction (SVD).
- **Batch Processing**: Supports multiple articles or paragraphs via CSV upload.
- **Semantic Understanding**: Uses **SBERT** embeddings to capture contextual meaning.
- **Ranking Capability**: Implements LexRank to rank articles based on importance.
- **User-Friendly Interface**: Built using **Streamlit** for easy interaction.
- **Multilingual Support** (requires fine-tuning).

## Installation

### Prerequisites
Ensure you have Python 3.8+ installed. Install dependencies using:
```bash
pip install -r requirements.txt
```

### Required Libraries
- `streamlit`
- `pandas`
- `numpy`
- `nltk`
- `sentence-transformers`
- `scikit-learn`

## Usage

### Running the Application
```bash
streamlit run app.py
```

### Uploading Articles
- Users can **upload a CSV file** or manually input articles and highlights.
- The system processes the text and **generates summaries** based on selected methods.
- For large datasets, results are displayed in a DataFrame format for better readability.

## Project Structure
```
├── app.py                 # Main Streamlit application
├── requirements.txt       # List of dependencies
├── data/                  # Sample dataset (if any)
├── models/                # Pretrained SBERT models (if applicable)
└── README.md              # Project documentation
```

## Applications
- **News Summarization**: Quickly extract key insights from long news articles.
- **Content Curation**: Summarize and rank articles for research or media analysis.
- **Business Reports**: Generate concise reports from lengthy business documents.
- **Educational Use**: Help students and researchers condense academic content.

## Future Enhancements
- **Fine-tuning for Multilingual Support**
- **Implementation on Mobile Devices**
- **Integration with Cloud Services for Scalability**
- **Customizable Summary Length and Format Options**
- **Improved Ranking Mechanism for Summarized Content**

## Contributors
- **Rakshith** *(Project Lead)*

## License
This project is licensed under the MIT License. See `LICENSE` for details.

---
**Feel free to contribute, suggest improvements, or report issues!** 🚀
