from sklearn.externals import joblib
import pandas as pd
import numpy as np
from text_analyzer import TextAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def read_data():
    df1 = pd.read_csv("../data/Gourmet.tsv", sep = "\t", header = None)
    df2 = pd.read_csv("../data/Keitai.tsv", sep = "\t", header = None)
    df3 = pd.read_csv("../data/Kyoto.tsv", sep = "\t", header = None)
    df4 = pd.read_csv("../data/Sports.tsv", sep = "\t", header = None)
    df1["label"] = "Gourmet"
    df2["label"] = "Keitai"
    df3["label"] = "Kyoto"
    df4["label"] = "Sports"
    df = pd.concat([df1, df2, df3, df4]).iloc[:, [1, 6]]
    df.columns = ["text", "label"]
    df = df.reset_index()
    return df

def create_corpus(df):
    ta = TextAnalyzer()
    ma = df["text"].apply(lambda x: ta.morphological_analize(x))
    corpus = ma.apply(lambda x: " ".join(x[0]))
    return corpus

def vectrize(corpus):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    X = X.toarray()
    joblib.dump(vectorizer, './model/vectorizer.pkl')
    return X

def train(train_X, train_y):
    clf = MultinomialNB()
    clf.fit(train_X, train_y)
    joblib.dump(clf, './model/clf.pkl')

if __name__ == "__main__":
    df = read_data()
    corpus = create_corpus(df)
    train_X = vectrize(corpus)
    train_y = df["label"]
    train(train_X, train_y)