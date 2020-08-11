# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 10:23:50 2020

@author: Francisco Parrilla
"""

"""
This code has been taken from https://www.kaggle.com/hsrobo/titlebased-semantic-subject-indexing
The initial code was made by Florian Mai, using Logistic Regression, but it was adapted to use Naive Bayes
and to use the whole dataset using multi-processing to make the cross-validation step faster

"""
import numpy as np
import sys
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
import multiprocessing
import multiprocessing.pool


clf = Pipeline([("vectorizer",
                 TfidfVectorizer(max_features = 25000)), 
                ("classifier", OneVsRestClassifier(ComplementNB()))])

def fit_results(train_df,y_train,test_df,y_test):
    
    clf.fit(train_df, y_train)
    y_pred = clf.predict(test_df)

    return (f1_score(y_test, y_pred, average="samples"))

def load_csv_disk (dataset_path,all_titles = False):
    
    df = pd.read_csv(dataset_path)

    return df
    

def filter_dataset(df, fold_i, all_titles = False):


    if all_titles != "True":
        print("Using only a portion of the dataset...")
        df = df[df["fold"].isin(range(0,10))]
    else:
        print("Using all dataset...")
        
    labels = df["labels"].values
    labels = [[l for l in label_string.split()] for label_string in labels]
    multilabel_binarizer = MultiLabelBinarizer(sparse_output = True)
    multilabel_binarizer.fit(labels)

    def to_indicator_matrix(some_df):
        some_df_labels = some_df["labels"].values
        some_df_labels = [[l for l in label_string.split()] for label_string in some_df_labels]
        return multilabel_binarizer.transform(some_df_labels)

    test_df = df[df["fold"] == fold_i]
    X_test = test_df["title"].values
    y_test = to_indicator_matrix(test_df)

    train_df = df[df["fold"] != fold_i]
    X_train = train_df["title"].values
    y_train = to_indicator_matrix(train_df)
    
    return X_train, y_train, X_test, y_test

def evaluate(dataset):
    
    df = load_csv_disk(dataset)
    scores = []
    args_fit = []
    
    for i in range(0, 10):
        train_df, y_train, test_df, y_test = filter_dataset(df, i, all_titles = ALL_TITLES)
        args_fit.append(tuple((train_df,y_train,test_df,y_test)))
       
    with multiprocessing.Pool(processes= 10) as pool:
        result_f1 = pool.starmap(fit_results,args_fit)
    
    scores.append(result_f1)
    
    return np.mean(scores)


ALL_TITLES = "True"
print("EconBiz average F-1 score:", evaluate("econbiz.csv"))