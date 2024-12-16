# %%
if False:
    """
    Project 3: Movie Review Sentiment Analysis

    Author: Chaojie Wang (netID: 656449601) UIUC MCS Online Fall 2024
    """

# %%
import os
import pickle
import requests

import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import nltk
from nltk import sent_tokenize

import warnings
warnings.filterwarnings("ignore")

seed = 9601
np.random.seed(seed)

# %%
if False:
    """
    Section 1: Build a Binary Classification Model

    1.1 Technical Details
    Model Configuration:
    - Uses Elastic Net regularization (penalty='elasticnet')
        - SAGA optimizer for handling both L1 and L2 penalties
        - l1_ratio=0.5: Equal weight between L1 and L2 regularization
        - max_iter=1000: Maximum iterations for convergence
        - Fixed random seed (9601) for reproducibility

    Hyperparameter Tuning
    - Selected the best C value: 2.782559 from a grid search
    - 5-fold cross-validation with ROC-AUC scoring metric

    1.2 Performance
    The code was run on a Ubuntu desktop
    - CPU: AMD Ryzen 9 7950X3D 16-Core Processor - 5.7GHz; 
    - Memory: 64GB
    
    The execution times and AUC of model predictions on the test set are as follows:
    Split 1: 58.70 seconds, AUC = 0.98698
    Split 2: 64.29 seconds, AUC = 0.98654
    Split 3: 61.14 seconds, AUC = 0.98631
    Split 4: 58.84 seconds, AUC = 0.98681
    Split 5: 63.51 seconds, AUC = 0.98620
    """

# %%
def download_model(url, save_path='best_model.pkl'):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Model downloaded successfully to {save_path}")
    else:
        raise Exception(f"Failed to download model. Status code: {response.status_code}")


# %%
def get_bert_embeddings(texts, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    embeddings = []
    
    for text in tqdm(texts, desc="Getting BERT embeddings"):
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)
        
        # Get BERT embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        embeddings.append(embedding[0])
    
    return np.array(embeddings)


# %%
if False:
    """
    Section 2: Interpretability Analysis

    2.1 Embedding model 
    - Use BERT Model (bert-base-uncased) to create new embeddings (embeddings of sentences)

    2.2 Embedding Alignment
    - Takes BERT embeddings (created from the sample reviews) and OpenAI embeddings (from the csv file) as input
    - Both embeddings are standardized using StandardScaler
    - A linear regression model is trained to map between the two embedding spaces
    - Sampled 20000 reviews from the training set for alignment

    2.3 Interpretability Analysis
    For 5 randomly sampled positive and negative reviews, 
    - Splits reviews into sentences using NLTK's sent_tokenize
    - Gets BERT embeddings for each sentence
    - Transforms BERT embeddings through alignment model:
        BERT → StandardScaler → Linear Transform → Inverse Scale → OpenAI space
    - Uses aligned embeddings to get sentence-level sentiment probabilities

    2.4 Visualization
    Highlights key sentiment-indicating sentences:
    - For positive reviews: Underlines sentences with prediction scores > 0.9
    - For negative reviews: Underlines sentences with prediction scores < 0.1
    """


# %%
def train_embedding_alignment(bert_embeddings, openai_embeddings):
    scaler_bert = StandardScaler()
    scaler_openai = StandardScaler()
    
    bert_scaled = scaler_bert.fit_transform(bert_embeddings)
    openai_scaled = scaler_openai.fit_transform(openai_embeddings)
    
    alignment_model = LinearRegression()
    alignment_model.fit(bert_scaled, openai_scaled)
    
    return alignment_model, scaler_bert, scaler_openai


def analyze_reviews(model, data, alignment_model, scaler_bert, scaler_openai, bert_model, bert_tokenizer, n_samples=5):
    pos_reviews = data[data['sentiment'] == 1].sample(n_samples, random_state=seed)
    neg_reviews = data[data['sentiment'] == 0].sample(n_samples, random_state=seed)
    selected_reviews = pd.concat([pos_reviews, neg_reviews])
    
    results = []
    for _, row in selected_reviews.iterrows():
        review_text = row['review']
        sentences = sent_tokenize(review_text)
        
        sentence_embeddings = get_bert_embeddings(sentences, bert_model, bert_tokenizer)
        scaled_bert = scaler_bert.transform(sentence_embeddings)
        aligned_embeddings = alignment_model.predict(scaled_bert)
        aligned_embeddings = scaler_openai.inverse_transform(aligned_embeddings)
        sentence_predictions = model.predict_proba(aligned_embeddings)[:, 1]
        
        results.append({
            'id': row['id'],
            'review': review_text,
            'true_sentiment': row['sentiment'],
            'sentences': sentences,
            'sentence_scores': sentence_predictions
        })
    
    return results


def visualize_results(results):
    for review in results:
        print(f"Review ID: {review['id']} (True Sentiment: {'Positive' if review['true_sentiment'] == 1 else 'Negative'})")
        print("Full Review (sentences underlined are key parts that contribute to the sentiment):")
        is_positive = review['true_sentiment'] == 1
        
        for i, (sentence, score) in enumerate(zip(review['sentences'], review['sentence_scores'])):
            if (is_positive and score > 0.9) or (not is_positive and score < 0.1):
                underline = '\u0332'.join(sentence) + '\u0332'  # Unicode combining underline character
                print(underline)
            else:
                print(sentence)
        print("\n\n" + "="*80 + "\n")


# %%
def interpret():
    # Load data
    train = pd.read_csv('train.csv')
    train_X = train.iloc[:, 3:]    
    
    nltk.download('punkt_tab')
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_model = AutoModel.from_pretrained('bert-base-uncased')

    sample_size = 20000
    sample_reviews = train['review'].sample(sample_size, random_state=seed)
    bert_embeddings = get_bert_embeddings(sample_reviews, bert_model, bert_tokenizer)        
    openai_embeddings = train_X.iloc[sample_reviews.index].values
    
    # Train alignment model
    alignment_model, scaler_bert, scaler_openai = train_embedding_alignment(
        bert_embeddings, 
        openai_embeddings
    )
    
    print("\nAnalyzing review interpretability...")
    model_url = "https://github.com/cwang717/psl_f24/raw/main/project_03/best_model.pkl"
    if not os.path.exists('best_model.pkl'):
        download_model(model_url)
    best_model = pickle.load(open('best_model.pkl', 'rb'))

    results = analyze_reviews(
        best_model, 
        train,
        alignment_model,
        scaler_bert,
        scaler_openai,
        bert_model,
        bert_tokenizer
    )
    visualize_results(results)


# %%
os.chdir(os.path.join(os.path.dirname(__file__), "F24_Proj3_data", "split_1"))
interpret()

# %%
