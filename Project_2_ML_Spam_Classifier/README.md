# Project 2: SMS Spam Classifier (ML)

## Objective
Load SMS Spam Collection, use an 80:20 stratified train/test split, TF-IDF text features, and Multinomial Naive Bayes classifier. Evaluate accuracy, precision and recall for the **spam** class, view a confusion matrix, and test three original messages.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
The app automatically reads an online SMS Spam Collection mirror. Alternatively upload the Kaggle `spam.csv` or place it beside `app.py`.

Dataset: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

## Methodology
1. Read labelled messages (ham/spam).
2. Stratified train/test split using random_state=42.
3. TF-IDF vectorization, fitted only on training data via sklearn Pipeline (prevents leakage).
4. Train MultinomialNB, test on held-out 20%.
5. Show confusion matrix, per-class classification report and 3 prediction examples.
