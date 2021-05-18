import pickle
import spacy
import pandas as pd
from tqdm import tqdm
import os

def tokenize(string):
    nlp = spacy.load("en_core_web_trf")
    doc = nlp(string)
    new_string = ' '.join([token.text for token in doc])
    return new_string

vectorizer = pickle.load(open(os.path.dirname(os.path.realpath(__file__))+'/data/vectorizer', 'rb'))
transformer = pickle.load(open(os.path.dirname(os.path.realpath(__file__))+'/data/transformer', 'rb'))
tree = pickle.load(open(os.path.dirname(os.path.realpath(__file__))+'/data/tree', 'rb'))

def predict(e):
    exception_info = [tokenize(e)]
    test_tfidf = transformer.transform(vectorizer.transform(exception_info))
    predict = tree.predict(test_tfidf)
    return predict[0]

def preict_dataset(excel):
    predicts = []
    data = pd.read_excel(excel)
    for et, ec in tqdm(zip(data['exception type'], data['exception content'])):
        exception_info = [tokenize(et + ' ' + ec)]
        test_tfidf = transformer.transform(vectorizer.transform(exception_info))
        predict = tree.predict(test_tfidf)
        predicts.append(predict[0])
    data['detect'] = predicts
    data.to_excel(excel, index=False)

if __name__ == '__main__':
    preict_dataset('../SFData/ICSE2020ToRepair.xlsx')
    preict_dataset('../SFData/StackOverflow.xlsx')
