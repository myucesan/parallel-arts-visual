from pyspark import SparkContext
from pyspark.ml.classification import NaiveBayes, LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import CountVectorizer, NGram, StopWordsRemover
from pyspark.shell import spark
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from sklearn.linear_model import LogisticRegression as lr
from nltk import WordNetLemmatizer
from pymongo import MongoClient
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
import pandas as pd
from langdetect import detect
from sklearn.metrics import classification_report, confusion_matrix
import time

from sklearn.model_selection import train_test_split


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
    # sc = SparkContext("local", "First App").getOrCreate().stop()
    spark = SparkSession \
        .builder \
        .appName("A spark session") \
        .getOrCreate()
    start = time.perf_counter()

    client = MongoClient("localhost:27017")
    db = client.deds
    df = pd.DataFrame(list(db.reviews.find({}).limit(10)))
    df = getLabeledData(df)
    df = pd.DataFrame(df, columns=["Review", "Label"])

    print("====== Clean irrelevant reviews begin ====")
    df = df[df["Review"] != "No Negative"]
    df = df[df["Review"] != "No Positive"]
    df = df[df["Review"] != "null"]
    df = df[df["Review"] != "Nothing"]
    df = df[df["Review"].apply(lambda x: len(x) > 20)]
    df = df[df["Review"].apply(lambda x: detect(x) == "en")]
    stop = time.perf_counter()
    print("Cleaning took " + str(stop-start) + "seconds")
    print("====== Clean irrelevant reviews end ====")

    from nltk import tokenize as tk

    print("====== Lemmatization begin ==== ")
    start = time.perf_counter()

    WNlemmatizer = WordNetLemmatizer()

    tokenized = df.apply(lambda row: tk.word_tokenize(row['Review']), axis=1)
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
    end_df = end_df.apply(lambda row: tk.word_tokenize(row['Review']), axis=1)
    end_df = pd.DataFrame(end_df, columns=["Review"])
    end_df["Label"] = df["Label"]
    end_df = end_df.dropna()

    stop = time.perf_counter()
    print("Lemmatization took " + str(stop-start) + " seconds")
    print("====== Lemmatization end ==== ")

    # Input data: Each row is a bag of words with a ID.
    # fit a CountVectorizerModel from the corpus.

    print("====== StopwordsRemover begin ====")
    remover = StopWordsRemover(inputCol="Review", outputCol="Filtered")
    dfCV = spark.createDataFrame(end_df, ["Review", "Label"])
    dfCV = remover.transform(dfCV)
    print("====== StopwordsRemover end ====")

    print("====== Unigram begin ====")
    ngram = NGram(n=1, inputCol="Filtered", outputCol="NGram")
    dfCV = ngram.transform(dfCV)
    print("====== Unigram end ====")

    print("====== Fix for CV begin ====")
    dfCV = dfCV.drop("Filtered")
    dfCV = dfCV.drop("Review")
    dfCV = dfCV.withColumnRenamed("NGram", "Review")
    dfCV.show(truncate=False)
    print("====== Fix for CV end ====")

    print("====== CountVectorization begin ====")
    start = time.perf_counter()
    cv = CountVectorizer(inputCol="Review", outputCol="Features", vocabSize=100, minDF=2.0)
    model = cv.fit(dfCV)
    result = model.transform(dfCV)
    end_df = result.drop('Review')
    sparse_matrix = end_df
    sparse_matrix.show(truncate=False)
    stop = time.perf_counter()

    print("CountVectorization took " + str(stop-start) + "seconds")
    print("====== CountVectorization end ====")


    #
    # X_train, X_test, y_train, y_test = train_test_split(X.select(), y.select(), test_size=0.25, random_state=325)
    #

    # Validation prepare
    # y = end_df.select("Label")
    # X = end_df.select("Features")
    train, test = sparse_matrix.randomSplit([0.75, 0.25], seed=325)
    f1 = MulticlassClassificationEvaluator(
        labelCol="Label", predictionCol="prediction", metricName="f1")
    accuracy = MulticlassClassificationEvaluator(
        labelCol="Label", predictionCol="prediction", metricName="accuracy")
    weightedPrecision = MulticlassClassificationEvaluator(
        labelCol="Label", predictionCol="prediction", metricName="weightedPrecision")
    weightedRecall = MulticlassClassificationEvaluator(
        labelCol="Label", predictionCol="prediction", metricName="weightedRecall")
    weightedTruePositiveRate = MulticlassClassificationEvaluator(
        labelCol="Label", predictionCol="prediction", metricName="weightedTruePositiveRate")
    weightedFalsePositiveRate = MulticlassClassificationEvaluator(
        labelCol="Label", predictionCol="prediction", metricName="weightedFalsePositiveRate")
    weightedFMeasure = MulticlassClassificationEvaluator(
        labelCol="Label", predictionCol="prediction", metricName="weightedFMeasure")


    # Logictic Regression
    start = time.perf_counter()
    log_reg = LogisticRegression(featuresCol="Features", labelCol="Label")  # https://stackoverflow.com/questions/56474033/python-dask-ml-linear-regression-multiple-constant-columns-detected-error
    log_reg_model = log_reg.fit(train)
    log_reg_predictions = log_reg_model.transform(test)
    f1Score = f1.evaluate(log_reg_predictions)
    accuracyScore = accuracy.evaluate(log_reg_predictions)
    precision = weightedPrecision.evaluate(log_reg_predictions)
    recall = weightedRecall.evaluate(log_reg_predictions)
    tpRate = weightedTruePositiveRate.evaluate(log_reg_predictions)
    fpRate = weightedFalsePositiveRate.evaluate(log_reg_predictions)
    fMeasure = weightedFMeasure.evaluate(log_reg_predictions)
    print("F1 ="  + str(f1Score))
    print("Accuracy ="  + str(accuracyScore))
    print("precision ="  + str(precision))
    print("recall ="  + str(recall))
    print("tpRate ="  + str(tpRate))
    print("fpRate ="  + str(fpRate))
    print("fMeasure ="  + str(fMeasure))
    stop = time.perf_counter()
    print("Logistic Regression took " + str(stop-start) + " seconds")


    # Decision Trees
    start = time.perf_counter()
    dec_tree = DecisionTreeClassifier(labelCol="Label", featuresCol="Features")
    dec_tree_model = dec_tree.fit(train)
    dec_tree_predictions = dec_tree_model.transform(test)
    # dec_tree_predictions.select("prediction", "Label", "Features").show(100)
    # Select (prediction, true label) and compute test error


    stop = time.perf_counter()
    print("Decision Trees took " + str(stop-start) + " seconds")
    f1Score = f1.evaluate(dec_tree_predictions)
    accuracyScore = accuracy.evaluate(dec_tree_predictions)
    precision = weightedPrecision.evaluate(dec_tree_predictions)
    recall = weightedRecall.evaluate(dec_tree_predictions)
    tpRate = weightedTruePositiveRate.evaluate(dec_tree_predictions)
    fpRate = weightedFalsePositiveRate.evaluate(dec_tree_predictions)
    fMeasure = weightedFMeasure.evaluate(dec_tree_predictions)
    print("F1 ="  + str(f1Score))
    print("Accuracy ="  + str(accuracyScore))
    print("precision ="  + str(precision))
    print("recall ="  + str(recall))
    print("tpRate ="  + str(tpRate))
    print("fpRate ="  + str(fpRate))
    print("fMeasure ="  + str(fMeasure))



    # Naive Bayes Multinomial
    start = time.perf_counter()

    naive_bayes = NaiveBayes(featuresCol="Features", labelCol="Label", modelType="multinomial")
    naive_bayes_model = naive_bayes.fit(train)
    naive_bayes_predictions = naive_bayes_model.transform(test)
    stop = time.perf_counter()
    print("Naive Bayes Multinomial took " + str(stop-start) + " seconds")
    f1Score = f1.evaluate(naive_bayes_predictions)
    accuracyScore = accuracy.evaluate(naive_bayes_predictions)
    precision = weightedPrecision.evaluate(naive_bayes_predictions)
    recall = weightedRecall.evaluate(naive_bayes_predictions)
    tpRate = weightedTruePositiveRate.evaluate(naive_bayes_predictions)
    fpRate = weightedFalsePositiveRate.evaluate(naive_bayes_predictions)
    fMeasure = weightedFMeasure.evaluate(naive_bayes_predictions)
    print("F1 ="  + str(f1Score))
    print("Accuracy ="  + str(accuracyScore))
    print("precision ="  + str(precision))
    print("recall ="  + str(recall))
    print("tpRate ="  + str(tpRate))
    print("fpRate ="  + str(fpRate))
    print("fMeasure ="  + str(fMeasure))