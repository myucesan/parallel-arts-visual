from dask_ml.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression as lr
from nltk import WordNetLemmatizer
from pymongo import MongoClient
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from dask_ml.feature_extraction.text import CountVectorizer
import pandas as pd
import dask.dataframe as dd
import dask.bag as daskbag
from langdetect import detect
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from dask.distributed import Client
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import tree
import time


def getLabeledData(df):
    #################### Fixing Kaggle Dataset here #############################
    print("Obtaining and preparing data")
    kaggle = df
    kaggleNegative = kaggle["Negative_Review"].to_frame().rename(columns={"Negative_Review": "Review"})
    kaggleNegative["Label"] = 0
    kagglePositive = kaggle["Positive_Review"].to_frame().rename(columns={"Positive_Review": "Review"})
    kagglePositive["Label"] = 1

    endResult = pd.concat([kaggleNegative, kagglePositive], ignore_index=True)

    return endResult


if __name__ == '__main__':

    client = MongoClient("localhost:27017")
    db = client.deds
    df = pd.DataFrame(list(db.reviews.find({}).limit(5000)))
    df = getLabeledData(df)
    df = pd.DataFrame(df, columns=["Review", "Label"])

    print("====== Clean irrelevant reviews begin ====")
    start = time.perf_counter()
    df = df[df["Review"] != "No Negative"]
    df = df[df["Review"] != "No Positive"]
    df = df[df["Review"] != "null"]
    df = df[df["Review"] != "Nothing"]
    df = df[df["Review"].apply(lambda x: len(x) > 20)]
    df = df[df["Review"].apply(lambda x: detect(x) == "en")]
    stop = time.perf_counter()
    print("Cleaning took " + str(stop-start) + "seconds")
    print("====== Clean irrelevant reviews end ====")

    from nltk import tokenize

    print("====== Lemmatization begin ==== ")
    start = time.perf_counter()
    WNlemmatizer = WordNetLemmatizer()

    tokenized = df.apply(lambda row: tokenize.word_tokenize(row['Review']), axis=1)
    lem_tokens = []
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
    stop = time.perf_counter()
    print("Lemmatization took " + str(stop-start) + " seconds")
    print("====== Lemmatization end ==== ")

    print("====== CountVectorization Dask with delayed dataframe begin ====")
    start = time.perf_counter()
    end_df = dd.from_pandas(df, 1)
    dfReview = daskbag.from_sequence(end_df["Review"], npartitions=1)
    vectorizer = CountVectorizer(max_features=100, ngram_range=(1, 1), stop_words=ENGLISH_STOP_WORDS)
    vectorizer.fit(dfReview)
    sparse_matrix = vectorizer.transform(dfReview)
    sparse_matrix_df = pd.DataFrame(sparse_matrix.compute().toarray(), columns=vectorizer.get_feature_names())
    stop = time.perf_counter()

    print("CountVectorization took " + str(stop-start) + "seconds")
    print("====== CountVectorization Dask with delayed dataframe end ====")

    end_df = end_df.drop('Review', axis=1)

    end_df = dd.merge(end_df, sparse_matrix_df, right_index=True, left_index=True)

    y = end_df["Label"].compute().values
    print(y)
    X = end_df.drop("Label", axis=1).compute().values
    print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=325)

    # Logictic Regression
    start = time.perf_counter()
    log_reg = LogisticRegression()  # https://stackoverflow.com/questions/56474033/python-dask-ml-linear-regression-multiple-constant-columns-detected-error
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    y_pred_prob = log_reg.predict_proba(X_test)
    stop = time.perf_counter()
    print("Logistic Regression took " + str(stop-start) + " seconds")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    client = Client(processes=False)

    with joblib.parallel_backend('dask'):
        start = time.perf_counter()
        # Decision Trees
        dec_tree = tree.DecisionTreeClassifier()
        dec_tree.fit(X_train, y_train)
        dec_tree_y_pred = dec_tree.predict((X_test))
        dec_tree_prob = dec_tree.predict_proba(X_test)[:, 1]
        stop = time.perf_counter()
        print("Decision Trees took " + str(stop-start) + " seconds")
        print(classification_report(y_test, dec_tree_y_pred))
        print(confusion_matrix(y_test, dec_tree_y_pred))

        start = time.perf_counter()
        # Naive Bayes Multinomial
        naive_bayes = MultinomialNB()
        naive_bayes.fit(X_train, y_train)
        naive_bayes_y_pred = naive_bayes.predict(X_test)
        naive_bayes_prob = naive_bayes.predict_proba(X_test)[:, 1]
        stop = time.perf_counter()
        print("Naive Bayes Multinomial took " + str(stop-start) + " seconds")
        print(classification_report(y_test, naive_bayes_y_pred))
        print(confusion_matrix(y_test, naive_bayes_y_pred))
