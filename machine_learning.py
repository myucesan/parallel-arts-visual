from dask_ml.model_selection import train_test_split
from langdetect import detect
from nltk import WordNetLemmatizer
from nltk import tokenize
from pymongo import MongoClient
import pandas as pd
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import tree

import dask_ml_benchmark
from stopwatch import Stopwatch

def getLabeledData(df):
    #################### Fixing Kaggle Dataset here #############################
    print("Ontaining and preparing data")
    kaggle = df
    kaggleNegative = kaggle["Negative_Review"].to_frame().rename(columns={"Negative_Review": "Review"})
    kaggleNegative["Label"] = 0
    kagglePositive = kaggle["Positive_Review"].to_frame().rename(columns={"Positive_Review": "Review"})
    kagglePositive["Label"] = 1

    endResult = pd.concat([kaggleNegative, kagglePositive], ignore_index=True)

    return endResult


if __name__ == '__main__':
    stopwatch = Stopwatch()
    client = MongoClient("localhost:27017")
    db = client.deds
    df = pd.DataFrame(list(db.reviews.find({}).limit(1000)))
    df = getLabeledData(df)
    df = pd.DataFrame(df, columns=["Review", "Label"])

    # # Remove irrelevant reviews
    end_df = df
    end_df = end_df[(end_df["Review"] != "No Negative")]
    end_df = end_df[end_df["Review"] != "No Positive"]
    end_df = end_df[end_df["Review"] != "null"]
    end_df = end_df[end_df["Review"] != "Nothing"]
    end_df = end_df[end_df["Review"].apply(lambda x: len(x) > 20)]
    end_df = end_df[end_df["Review"].apply(lambda x: detect(x) == "en")]
    end_df = end_df
    # stopwatch.start()
    #
    print("====== Lemmatization begin ==== ")
    WNlemmatizer = WordNetLemmatizer()

    tokenized = df.apply(lambda row: tokenize.word_tokenize(row['Review']), axis=1)
    lem_tokens = []
    text = ""
    for tokenize in tokenized:
        listie = []
        for token in tokenize:
            listie.append(WNlemmatizer.lemmatize(token))
        lem_tokens.append(listie)

    end_df = []
    for review in lem_tokens:
        end_df.append(" ".join(review))
    end_df = pd.DataFrame(end_df, columns=["Review"])
    end_df["Label"] = df["Label"]

    print("====== Lemmatization end ==== ")
    #
    # # Dask benchmark starts here
    #
    vectorizer = CountVectorizer(max_features=100, ngram_range=(1,1), stop_words=ENGLISH_STOP_WORDS)
    vectorizer.fit(df["Review"])
    sparse_matrix = vectorizer.transform(df["Review"])
    sparse_matrix_df = pd.DataFrame(sparse_matrix.toarray(), columns=vectorizer.get_feature_names())

    end_df = pd.merge(end_df, sparse_matrix_df, right_index=True, left_index=True)

    end_df = end_df.drop('Review', axis=1)
    #
    y = end_df["Label"].values
    X = end_df.drop("Label", axis=1).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=325)

    #
    # # Logictic Regression
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict((X_test))
    y_pred_prob = log_reg.predict_proba(X_test)[:, 1]
    # print(y_pred_prob)

    y_pred_prob = log_reg.predict_proba(X_test)
    # print(y_pred_prob[, 1:])




    # print(classification_report(y_test, y_pred))
    # print(confusion_matrix(y_test, y_pred))
    #
    # Decision Trees
    dec_tree = tree.DecisionTreeClassifier()
    dec_tree.fit(X_train, y_train)
    dec_tree_y_pred = dec_tree.predict((X_test))
    dec_tree_prob = dec_tree.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, dec_tree_y_pred))
    print(confusion_matrix(y_test, dec_tree_y_pred))
    #
    #
    # Naive Bayes Multinomial
    naive_bayes = MultinomialNB()
    naive_bayes.fit(X_train, y_train)
    naive_bayes_y_pred = naive_bayes.predict(X_test)
    naive_bayes_prob = naive_bayes.predict_proba(X_test)[:, 1]
    # print(classification_report(y_test, naive_bayes_y_pred))
    # print(confusion_matrix(y_test, naive_bayes_y_pred))