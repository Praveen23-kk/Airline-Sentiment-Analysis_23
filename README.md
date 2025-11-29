# Airline Sentiment Analysis (NLP | WordCloud | ML)

A complete end-to-end sentiment analysis project on airline customer reviews.  
The notebook performs data cleaning, preprocessing, visualization, and sentiment classification using NLP techniques and machine learning models.  
WordClouds are generated to highlight the most frequent terms in positive and negative feedback.

---

## 📌 Project Overview

Airline companies receive thousands of reviews daily.  
This project analyzes those reviews and classifies them into:

- Positive  
- Neutral  
- Negative  

The goal is to understand customer emotions, identify dissatisfaction patterns, and visualize text insights using WordCloud.

---

## 🔥 Key Features

- Text preprocessing (tokenization, stopword removal, stemming/lemmatization)
- Sentiment classification using ML models
- WordClouds for positive and negative feedback
- Sentiment distribution & data visualization
- Clean, structured Jupyter notebook for reproducible results

---

## 📂 Repository Structure

Airline-Sentiment-Analysis_23/
│── Airline_Review_Sentimental_Analysis_(NLP)_&_Visulalization_in_Wordcloud.ipynb
│── README.md (you are here)
└── data/


---

## 📊 Dataset Details

| Attribute | Info |
|----------|------|
| Source   | Kaggle / Custom Dataset |
| Format   | CSV / JSON |
| Fields   | review_text, airline, sentiment_label, etc |
| Size     | ~10K+ rows (update with exact count) |

> Add dataset source link here if allowed.

---

## 🧠 ML/NLP Workflow

1. Load and inspect dataset  
2. Preprocess text (cleaning + normalization)  
3. Encode sentiment labels  
4. Perform EDA (sentiment counts, word frequency, barplots)  
5. Create WordClouds per sentiment class  
6. Convert text into TF-IDF / Bag-of-Words vectors  
7. Train ML models (Naive Bayes / Logistic Regression / SVM etc.)  
8. Evaluate using accuracy & confusion matrix  

---

## 🏗 Technologies Used

| Tech | Purpose |
|------|----------|
| Python | Core programming |
| Pandas, NumPy | Data handling |
| Scikit-Learn | Machine learning models |
| NLTK / SpaCy | NLP preprocessing |
| Matplotlib, Seaborn | Visualizations |
| WordCloud | Text visualization |
| Google Colab / Jupyter | Notebook execution |

---

## Install dependencies:

pip install pandas numpy scikit-learn nltk seaborn matplotlib wordcloud

## Run notebook:
jupyter notebook
open the .ipynb file and run cell-by-cell

## Visulization Plots:
<img src="images/download.png" width="500" />
<img src="" width="500" />
<img src="images/download (1).png" width="500" />
<img src="images/download (8).png" width="500" />


## 🚀 How to Run Locally

Clone the repository:

```bash
git clone https://github.com/<your-username>/Airline-Sentiment-Analysis_23.git
cd Airline-Sentiment-Analysis_23

