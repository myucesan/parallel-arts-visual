from dask_ml.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression as lr
from dask_ml.model_selection import train_test_split
from nltk import WordNetLemmatizer
from pymongo import MongoClient
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from dask_ml.feature_extraction.text import CountVectorizer
import pandas as pd
import dask.dataframe as dd
import dask.bag as daskbag


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

    client = MongoClient("localhost:27017")
    db = client.deds
    df = pd.DataFrame(list(db.reviews.find({}).limit(1000)))
    df = getLabeledData(df)
    df = pd.DataFrame(df, columns=["Review", "Label"])

    from nltk import tokenize

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

    print("====== CountVectorization Dask with delayed dataframe begin ====")
    dfReview = daskbag.from_sequence(df["Review"], npartitions=2)
    vectorizer = CountVectorizer(max_features=100, ngram_range=(1, 1), stop_words=ENGLISH_STOP_WORDS)
    vectorizer.fit(dfReview)
    sparse_matrix = vectorizer.transform(dfReview)
    sparse_matrix_df = pd.DataFrame(sparse_matrix.compute().toarray(), columns=vectorizer.get_feature_names())
    print("====== CountVectorization Dask with delayed dataframe end ====")


    # Remove irrelevant reviews
    end_df = dd.merge(end_df, sparse_matrix_df, right_index=True, left_index=True)
    end2lol = end_df
    end2lol = end2lol[(end2lol["Review"] != "No Negative")]
    end2lol = end2lol[end2lol["Review"] != "No Positive"]
    end2lol = end2lol[end2lol["Review"] != "null"]
    end2lol = end2lol[end2lol["Review"] != "Nothing"]
    end_df = end2lol
    end_df = end_df.drop('Review', axis=1)
    end_df = dd.from_pandas(end_df, 2)


    y = end_df["Label"].values
    X = end_df.drop("Label", axis=1).values
    X_train, X_test, y_train, y_test = train_test_split(X.compute_chunk_sizes(), y.compute_chunk_sizes(), test_size=0.25, random_state=325)
    #
    # Logictic Regression
    log_reg = LogisticRegression()
    log_ref2 = lr()
    log_reg.fit(X_train, y_train)
    log_ref2.fit(X_train, y_train)

    y_pred = log_reg.predict(X_test)
    Y_pred2 = log_ref2.predict(X_test)
    print(log_reg.predict_proba(X_test)[:, 1])

    y_pred_prob = log_reg.predict_proba(X_test)[:, 1]
    print(9123456789)
    # print(classification_report(y_test, y_pred))

    # print(confusion_matrix(y_test, y_pred))
