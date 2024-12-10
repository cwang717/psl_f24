import pandas as pd
import os
from sklearn.metrics import roc_auc_score

base_dir = "F24_Proj3_data"

for i in range(1, 6):
    split_dir = f"split_{i}"
    test_path = os.path.join(base_dir, split_dir, "test_y.csv")
    pred_path = os.path.join(base_dir, split_dir, "mysubmission.csv")

    # Load submission data
    predict_df = pd.read_csv(pred_path)
    
    # Load true labels data
    true_labels_df = pd.read_csv(test_path)
    # print(true_labels_df.head())

    # Extract true sentiment and probabilities
    true_sentiment = true_labels_df['sentiment']
    predicted_probabilities = predict_df['prob']

    # print(true_sentiment.head())

    # Calculate AUC score
    auc_score = roc_auc_score(true_sentiment, predicted_probabilities)
    
    print(f"{split_dir} - AUC Score: {auc_score:.5f}")