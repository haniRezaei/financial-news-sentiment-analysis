# financial-news-sentiment-analysis
This project uses Natural Language Processing (NLP) and Machine Learning (ML) to classify financial news headlines as positive, negative, or neutral. By automating sentiment analysis, it helps investors, analysts, and financial institutions monitor market sentiment at scale.


Dataset: News headlines labeled with sentiment (positive, neutral, negative), sourced from this Kaggle dataset.

Text Processing & Feature Engineering:

Extracted sentiment scores using VADER and TextBlob

Transformed headlines into numerical vectors using Bag-of-Words and TF-IDF

Addressed class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)

Modeling:

Trained classical ML models: Logistic Regression, Naive Bayes, Support Vector Machine (SVM), KNN, XGBoost

Built deep learning models: LSTM and GRU using GloVe word embeddings

Evaluation:

SVM, XGBoost, and Logistic Regression performed best (~60–70% accuracy)

LSTM and GRU faced overfitting; FinBERT is recommended for future improvements

Neutral sentiment is most accurately classified; negative sentiment remains the most challenging


Highest Accuracy with Classical Models
Among the models tested, Support Vector Machine (SVM) and XGBoost delivered the most reliable results, correctly classifying sentiment with an accuracy of around 65–70%. These models are efficient and can be deployed in real-time environments.

Deep Learning Models Had Limitations
We also tested LSTM and GRU neural networks. While promising in theory, they overfitted the data—performing well during training but poorly on unseen examples. This suggests that without more tailored data or advanced models, deep learning alone may not yield practical results for this specific task.

Neutral Sentiment Was Easiest to Detect
Headlines that were emotionally neutral were accurately classified most of the time. However, distinguishing between positive and negative sentiment proved more challenging—especially for short or ambiguous headlines.
