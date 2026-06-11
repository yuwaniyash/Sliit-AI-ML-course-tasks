# Sliit-AI-ML-course-tasks

# [Task 1 – Spam Detection](./Spam-Detection/Task1.ipynb)

Builds a text classification model to detect whether an SMS message is spam or legitimate (ham).

The dataset is preprocessed using NLP techniques — lowercasing, tokenization, stopword removal, 
and stemming via PorterStemmer. Text is then converted into numerical features using TF-IDF 
vectorization (3000 features), and a Multinomial Naïve Bayes classifier is trained on an 80/20 
train-test split.

**Tools:** Python, pandas, nltk, scikit-learn  

# [Task 2 – Mobile Price Classification](./Mobile-Price-Classification/Task2.ipynb)

Predicts the price range of a mobile phone (0 = low, 3 = high) based on hardware specifications 
such as RAM, battery power, camera resolution, and more.

A Sequential Artificial Neural Network is built using Keras with 2 hidden ReLU layers, trained 
over 100 epochs with the Adam optimizer. Features are standardized using StandardScaler before 
training, and model weights are saved to an .h5 file after evaluation.

**Tools:** Python, pandas, scikit-learn, TensorFlow/Keras 

# [Task 3 – Sentiment Analysis](./Sentiment-Analysis/Task3.ipynb)

Classifies the sentiment of tweets about US Airlines as positive, negative, or neutral using NLP.

Text is cleaned by removing URLs, stopwords, and applying stemming. TF-IDF vectorization is used 
for feature extraction (3000 features). Two classifiers are trained and compared — Multinomial 
Naïve Bayes and Random Forest — on an 80/20 train-test split.

**Tools:** Python, pandas, nltk, scikit-learn
