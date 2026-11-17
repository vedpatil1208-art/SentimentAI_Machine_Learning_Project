# SentimentAI – Social Media Sentiment Analysis Using Machine Learning

## Project Overview

SentimentAI is a machine-learning-based web application that analyzes social media text and classifies sentiment into three categories: Positive, Negative, and Neutral.

The project uses Twitter and Reddit datasets and applies Natural Language Processing (NLP) and machine learning techniques to identify opinions and emotions expressed in social media posts.

## Project Objectives

- Analyze sentiment in social media posts.
- Classify text into Positive, Negative, and Neutral categories.
- Compare sentiment patterns across Twitter and Reddit.
- Develop an interactive application for sentiment visualization.
- Apply machine learning and NLP techniques to real-world text data.

## Features

- **Single-Page Interface:** Simple interface for submitting social media URLs or text.
- **Sentiment Analysis:** Predicts the sentiment of submitted text.
- **Reddit URL Analysis:** Attempts to retrieve public Reddit posts and comments.
- **Sentiment Visualization:** Displays sentiment distribution using pie charts.
- **Simple Review:** Provides a short explanation of the overall predicted sentiment.
- **Analyzed Text:** Allows users to view the retrieved text and its individual predictions.

## Machine Learning Workflow

1. **Data Collection:** Collected Twitter and Reddit sentiment datasets.
2. **Data Cleaning:** Handled missing values, duplicates, and unnecessary text.
3. **Text Preprocessing:** Converted text to lowercase, removed URLs and unwanted characters, and removed stopwords while preserving negation words.
4. **Feature Extraction:** Applied TF-IDF (Term Frequency–Inverse Document Frequency).
5. **Model Training:** Trained and evaluated three machine learning algorithms.
6. **Model Selection:** Selected Logistic Regression based on its test performance.
7. **Deployment:** Developed a web application using Streamlit.

## Machine Learning Algorithms

Three classification algorithms were evaluated:

| Algorithm | Accuracy |
|---|---:|
| Naive Bayes | 72.39% |
| Logistic Regression | 89.85% |
| Decision Tree | 64.02% |

**Selected Model:** Logistic Regression

The reported accuracy values are from the original notebook evaluation. Results may vary if the model is retrained.

## Dataset Information

The project uses two datasets:

- Twitter Sentiment Dataset
- Reddit Sentiment Dataset

The sentiment labels are:

| Label | Sentiment |
|---|---|
| 1 | Positive |
| 0 | Neutral |
| -1 | Negative |

The original combined dataset contained approximately 199,768 records.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| NLTK | Text preprocessing |
| TF-IDF | Text feature extraction |
| Plotly | Data visualization |
| Streamlit | Web application |
| Joblib | Saving and loading trained models |
| GitHub | Version control and project hosting |

## Project Structure

```text
SentimentAI/
│
├── app.py
├── requirements.txt
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── Twitter_Data.csv
├── Reddit_Data.csv
├── README.md
└── SentimentAI_Machine_Learning_Project.ipynb
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/vedpatil1208-art/Social_Media_Sentiment_Analysis.git
```

### 2. Navigate to the project directory

```bash
cd Social_Media_Sentiment_Analysis
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

## Application Workflow

1. Open the SentimentAI web application.
2. Paste a public Reddit post URL or enter social media text.
3. Click the Analyze Sentiment button.
4. The application processes the submitted text using the trained Logistic Regression model.
5. The predicted sentiment is displayed.
6. A pie chart visualizes the sentiment distribution or estimated probabilities.
7. A simple review summarizes the result.

## Limitations

- Predictions depend on the quality and characteristics of the training dataset.
- The training datasets may not represent all social media users or topics.
- Automatic URL extraction is currently supported for public Reddit posts.
- Other platforms may require API access or manually pasted text.
- Sarcasm, slang, and context-dependent language can affect prediction accuracy.

## Future Improvements

- Improve classification accuracy using advanced NLP techniques.
- Add support for additional social media platforms.
- Implement real-time sentiment monitoring.
- Improve social media URL extraction.
- Add more detailed model evaluation metrics.
- Expand the training dataset with diverse topics and languages.

## Author

**Ved Patil**

Computer Science Engineering Student

GitHub: https://github.com/vedpatil1208-art

---

## Project Information

**Project Title:** Social Media Sentiment Analysis Using Machine Learning

**Application Name:** SentimentAI

**Domain:** Machine Learning and Natural Language Processing

**Project Type:** Academic Project

Developed as an academic project to explore machine learning, text classification, and sentiment analysis of social media content.
